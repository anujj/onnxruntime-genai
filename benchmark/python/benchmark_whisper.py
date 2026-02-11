# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This is an end-to-end benchmarking script for Whisper models.
#
# Prerequisites:
# 0) Install onnxruntime-genai and onnxruntime
#
# 1) Use builder.py to build the desired Whisper ONNX model
#
# 2) Run this script with the desired arguments. Run benchmark_whisper.py -h for help.
#
# Example:
#   python benchmark_whisper.py -m /path/to/whisper -a audio.mp3 -e cuda -r 10 -w 5

import argparse
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import List

import onnxruntime_genai as og
import psutil
from metrics import BenchmarkRecord
from tqdm import tqdm

# Global variables for memory monitoring
peak_cpu_memory = 0.0
peak_gpu_memory = 0.0
peak_memory_lock = threading.Lock()
stop_monitoring = False

# Check if nvidia-smi is available
try:
    subprocess.run(["nvidia-smi"], check=True, capture_output=True)
    IS_NVIDIA_SYSTEM = True
except Exception:
    IS_NVIDIA_SYSTEM = False


@dataclass
class TimingStats:
    """Statistics for a collection of timing measurements."""

    values: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        return stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        return median(self.values) if self.values else 0.0

    def add(self, value: float):
        self.values.append(value)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "count": len(self.values),
        }


@dataclass
class WhisperBenchmarkResult:
    """Container for all Whisper benchmark metrics."""

    batch_size: int
    num_beams: int
    benchmark_mode: str
    audio_duration_sec: float
    model_load_time_ms: float
    audio_load_time: TimingStats = field(default_factory=TimingStats)
    audio_processing_time: TimingStats = field(default_factory=TimingStats)
    prefill_time: TimingStats = field(default_factory=TimingStats)  # set_inputs() time (encoder + decoder-init)
    per_token_latency: TimingStats = field(default_factory=TimingStats)  # Decoder-with-past tokens only
    generation_phase_time: TimingStats = field(default_factory=TimingStats)  # Decoder-with-past phase only
    end_to_end_inference_time: TimingStats = field(default_factory=TimingStats)  # Prefill + generation phase
    decode_time: TimingStats = field(default_factory=TimingStats)
    wall_clock_time: TimingStats = field(default_factory=TimingStats)
    tokens_generated: List[int] = field(default_factory=list)  # Total tokens including first
    generation_tokens: List[int] = field(default_factory=list)  # Tokens excluding first
    peak_memory_gb: float = 0.0

    def compute_derived_metrics(self) -> dict:
        """Compute throughput and RTF metrics."""
        avg_tokens = mean(self.tokens_generated) if self.tokens_generated else 0
        avg_gen_tokens = mean(self.generation_tokens) if self.generation_tokens else 0
        wall_clock_s = self.wall_clock_time.mean / 1000.0  # Convert ms to s
        gen_phase_s = self.generation_phase_time.mean / 1000.0  # Convert ms to s

        # End-to-end throughput (all tokens / wall clock time)
        e2e_throughput_tps = avg_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

        # Generation-only throughput (decoder-with-past tokens / generation phase time)
        gen_throughput_tps = avg_gen_tokens / gen_phase_s if gen_phase_s > 0 else 0.0

        rtf = wall_clock_s / self.audio_duration_sec if self.audio_duration_sec > 0 else 0.0

        return {
            "e2e_throughput_tokens_per_sec": e2e_throughput_tps,
            "generation_throughput_tokens_per_sec": gen_throughput_tps,
            "real_time_factor": rtf,
            "avg_tokens_generated": avg_tokens,
            "avg_generation_tokens": avg_gen_tokens,
        }

    def to_dict(self) -> dict:
        derived = self.compute_derived_metrics()
        return {
            "batch_size": self.batch_size,
            "num_beams": self.num_beams,
            "benchmark_mode": self.benchmark_mode,
            "audio_duration_sec": self.audio_duration_sec,
            "model_load_time_ms": self.model_load_time_ms,
            "audio_load_time_ms": self.audio_load_time.to_dict(),
            "audio_processing_time_ms": self.audio_processing_time.to_dict(),
            "prefill_time_ms": self.prefill_time.to_dict(),
            "per_token_latency_ms": self.per_token_latency.to_dict(),
            "generation_phase_time_ms": self.generation_phase_time.to_dict(),
            "end_to_end_inference_time_ms": self.end_to_end_inference_time.to_dict(),
            "decode_time_ms": self.decode_time.to_dict(),
            "wall_clock_time_ms": self.wall_clock_time.to_dict(),
            "tokens_generated": self.tokens_generated,
            "generation_tokens": self.generation_tokens,
            "peak_memory_gb": self.peak_memory_gb,
            **derived,
        }


def monitor_gpu_memory():
    """Monitor GPU memory usage via nvidia-smi."""
    global peak_gpu_memory

    while not stop_monitoring:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )

        memory_usage = result.stdout.splitlines()

        if len(memory_usage) >= 1:
            gpu_memory = [float(line) for line in memory_usage]
            current_peak = round(max(gpu_memory) / 1024, 2)
            with peak_memory_lock:
                peak_gpu_memory = max(current_peak, peak_gpu_memory)
        time.sleep(0.1)


def monitor_cpu_memory():
    """Monitor CPU memory usage via psutil."""
    global peak_cpu_memory

    while not stop_monitoring:
        current_used_memory = round(psutil.virtual_memory().used / 1024**3, 2)
        with peak_memory_lock:
            peak_cpu_memory = max(peak_cpu_memory, current_used_memory)
        time.sleep(0.1)


def get_audio_duration(audio_paths: List[str]) -> float:
    """Get total audio duration in seconds. Returns estimate if actual duration unavailable."""
    # For now, return a placeholder - actual duration would require audio processing library
    # Users can provide this via command line if needed
    return 0.0


def get_target_pip_package_version(target_pip_package_name_list: List[str]) -> tuple:
    """Get package name and version from installed packages."""
    import importlib.metadata

    installed_packages_list = sorted(
        [
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"] in target_pip_package_name_list
        ]
    )

    pkg_name = ""
    pkg_version = ""
    if installed_packages_list:
        pkg_name = installed_packages_list[0].split("==")[0]
        pkg_version = installed_packages_list[0].split("==")[1]
    return pkg_name, pkg_version


def run_benchmark(
    args,
    model: og.Model,
    processor,
    audio_paths: List[str],
    batch_size: int,
    num_beams: int,
) -> WhisperBenchmarkResult:
    """Run the Whisper benchmark for a given configuration."""
    num_repetitions = args.repetitions
    max_length = args.max_length
    model_only_mode = args.benchmark_mode == "model_only"

    # Initialize result container
    result = WhisperBenchmarkResult(
        batch_size=batch_size,
        num_beams=num_beams,
        benchmark_mode=args.benchmark_mode,
        audio_duration_sec=args.audio_duration if args.audio_duration > 0 else get_audio_duration(audio_paths),
        model_load_time_ms=0.0,  # Set externally
    )

    # Prepare decoder prompt for Whisper
    decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    prompts = ["".join(decoder_prompt_tokens)] * batch_size

    # Select audio files for batch (cycle if fewer files than batch size)
    batch_audio_paths = []
    for i in range(batch_size):
        batch_audio_paths.append(audio_paths[i % len(audio_paths)])

    prepared_inputs = None
    if model_only_mode:
        # Reuse one prepared input across all runs to remove audio I/O and preprocessing variance.
        prepared_audios = og.Audios.open(*batch_audio_paths)
        prepared_inputs = processor(prompts, audios=prepared_audios)

    # Warmup runs
    if args.verbose:
        print(f"Running {args.warmup} warmup iterations...")
    for _ in tqdm(range(args.warmup), desc="Warmup", disable=not args.verbose):
        if model_only_mode:
            inputs = prepared_inputs
        else:
            audios = og.Audios.open(*batch_audio_paths)
            inputs = processor(prompts, audios=audios)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_length=max_length,
            batch_size=batch_size,
        )

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            generator.generate_next_token()

        # Decode to warm up that path too
        for i in range(batch_size * num_beams):
            tokens = generator.get_sequence(i)
            processor.decode(tokens)

        del generator

    # Measured runs
    if args.verbose:
        print(f"Running {num_repetitions} measured iterations...")
    for _ in tqdm(range(num_repetitions), desc="Benchmark", disable=not args.verbose):
        wall_clock_start = time.perf_counter()
        token_count = 0
        per_token_times = []

        # 1. Input preparation phase
        if model_only_mode:
            inputs = prepared_inputs
        else:
            audio_load_start = time.perf_counter()
            audios = og.Audios.open(*batch_audio_paths)
            audio_load_end = time.perf_counter()
            result.audio_load_time.add((audio_load_end - audio_load_start) * 1000)

            process_start = time.perf_counter()
            inputs = processor(prompts, audios=audios)
            process_end = time.perf_counter()
            result.audio_processing_time.add((process_end - process_start) * 1000)

        # 3. Generator setup
        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_length=max_length,
            batch_size=batch_size,
        )

        generator = og.Generator(model, params)

        # 3. Prefill phase (set_inputs runs encoder + decoder-init for Whisper)
        prefill_start = time.perf_counter()
        generator.set_inputs(inputs)
        prefill_end = time.perf_counter()
        prefill_time_ms = (prefill_end - prefill_start) * 1000
        result.prefill_time.add(prefill_time_ms)

        # Consume prefill logits once. This call is token selection from prefill logits and
        # does not measure decoder-with-past runtime.
        if not generator.is_done():
            generator.generate_next_token()
            token_count += 1

        # 4. Generation phase (decoder-with-past only, excluding token selected from prefill logits)
        gen_phase_start = time.perf_counter()
        gen_token_count = 0
        while not generator.is_done():
            token_start = time.perf_counter()
            generator.generate_next_token()
            token_end = time.perf_counter()
            per_token_times.append((token_end - token_start) * 1000)
            token_count += 1
            gen_token_count += 1
        gen_phase_end = time.perf_counter()
        generation_phase_time_ms = (gen_phase_end - gen_phase_start) * 1000
        result.generation_phase_time.add(generation_phase_time_ms)
        result.generation_tokens.append(gen_token_count)
        result.end_to_end_inference_time.add(prefill_time_ms + generation_phase_time_ms)

        # Record per-token latencies (generation phase only, excluding first token)
        for t in per_token_times:
            result.per_token_latency.add(t)

        # 5. Decode tokens to text
        decode_start = time.perf_counter()
        for i in range(batch_size * num_beams):
            tokens = generator.get_sequence(i)
            processor.decode(tokens)
        decode_end = time.perf_counter()
        result.decode_time.add((decode_end - decode_start) * 1000)

        wall_clock_end = time.perf_counter()
        result.wall_clock_time.add((wall_clock_end - wall_clock_start) * 1000)
        result.tokens_generated.append(token_count)

        if args.print_transcription:
            for i in range(batch_size * num_beams):
                tokens = generator.get_sequence(i)
                transcription = processor.decode(tokens)
                print(f"  [batch {i // num_beams}, beam {i % num_beams}]: {transcription}")

        del generator

    return result


def run_benchmark_with_memory(
    args,
    model: og.Model,
    processor,
    audio_paths: List[str],
    batch_size: int,
    num_beams: int,
) -> WhisperBenchmarkResult:
    """Run benchmark with memory monitoring."""
    global stop_monitoring, peak_gpu_memory, peak_cpu_memory

    stop_monitoring = False
    peak_gpu_memory = 0.0
    peak_cpu_memory = 0.0

    if IS_NVIDIA_SYSTEM:
        monitor_thread = threading.Thread(target=monitor_gpu_memory)
    else:
        monitor_thread = threading.Thread(target=monitor_cpu_memory)

    monitor_thread.start()

    result = run_benchmark(args, model, processor, audio_paths, batch_size, num_beams)

    stop_monitoring = True
    monitor_thread.join()

    result.peak_memory_gb = peak_gpu_memory if IS_NVIDIA_SYSTEM else peak_cpu_memory
    return result


def save_results(args, results: List[WhisperBenchmarkResult], filename: str):
    """Save benchmark results to JSON (and optionally CSV)."""
    genai_package_name, genai_package_version = get_target_pip_package_version(
        ["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime-genai-directml"]
    )

    records = []
    for result in results:
        record = BenchmarkRecord(
            args.model_name,
            args.precision,
            "onnxruntime-genai",
            args.execution_provider,
            genai_package_name,
            genai_package_version,
            batch_size=result.batch_size,
            warmup_runs=args.warmup,
            measured_runs=args.repetitions,
        )

        # Add Whisper-specific metrics
        record.config.customized["num_beams"] = result.num_beams
        record.config.customized["max_length"] = args.max_length
        record.config.customized["audio_duration_sec"] = result.audio_duration_sec
        record.config.customized["benchmark_mode"] = result.benchmark_mode

        record.metrics.customized["model_load_time_ms"] = result.model_load_time_ms
        record.metrics.customized["audio_load_time_ms_mean"] = result.audio_load_time.mean
        record.metrics.customized["audio_processing_time_ms_mean"] = result.audio_processing_time.mean

        # Prefill phase metrics (encoder + decoder-init)
        record.metrics.customized["prefill_time_ms_mean"] = result.prefill_time.mean
        record.metrics.customized["prefill_time_ms_std"] = result.prefill_time.std

        # Generation phase metrics (decoder-with-past)
        record.metrics.customized["per_token_latency_ms_mean"] = result.per_token_latency.mean
        record.metrics.customized["per_token_latency_ms_std"] = result.per_token_latency.std
        record.metrics.customized["generation_phase_time_ms_mean"] = result.generation_phase_time.mean
        record.metrics.customized["generation_phase_time_ms_std"] = result.generation_phase_time.std

        # End-to-end model inference metric (prefill + generation phase)
        record.metrics.customized["end_to_end_inference_time_ms_mean"] = result.end_to_end_inference_time.mean
        record.metrics.customized["end_to_end_inference_time_ms_std"] = result.end_to_end_inference_time.std

        record.metrics.customized["decode_time_ms_mean"] = result.decode_time.mean
        record.metrics.customized["wall_clock_time_ms_mean"] = result.wall_clock_time.mean

        derived = result.compute_derived_metrics()
        record.metrics.customized["e2e_throughput_tokens_per_sec"] = derived["e2e_throughput_tokens_per_sec"]
        record.metrics.customized["generation_throughput_tokens_per_sec"] = derived["generation_throughput_tokens_per_sec"]
        record.metrics.customized["real_time_factor"] = derived["real_time_factor"]
        record.metrics.customized["avg_tokens_generated"] = derived["avg_tokens_generated"]
        record.metrics.customized["avg_generation_tokens"] = derived["avg_generation_tokens"]

        if args.monitor_memory:
            record.metrics.customized["peak_memory_gb"] = result.peak_memory_gb

        records.append(record)

    # Save JSON
    BenchmarkRecord.save_as_json(filename, records)
    print(f"Results saved to {filename}")

    # Optionally save CSV
    if args.save_csv:
        csv_filename = filename.replace(".json", ".csv")
        BenchmarkRecord.save_as_csv(csv_filename, records)
        print(f"CSV results saved to {csv_filename}")


def print_summary(result: WhisperBenchmarkResult, verbose: bool = False):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 70)
    print(f"Batch Size: {result.batch_size}, Num Beams: {result.num_beams}")
    print("=" * 70)

    print(f"Model Load Time:              {result.model_load_time_ms:.2f} ms")
    print(f"Benchmark Mode:               {result.benchmark_mode}")
    if result.benchmark_mode == "e2e":
        print(f"Audio Load Time (mean):       {result.audio_load_time.mean:.2f} ms")
        print(f"Audio Processing (mean):      {result.audio_processing_time.mean:.2f} ms")

    print("\n--- Prefill Phase (Encoder + Decoder-Init) ---")
    print(f"Prefill Time (mean):          {result.prefill_time.mean:.2f} ms (+/- {result.prefill_time.std:.2f})")

    print("\n--- Generation Phase (Decoder-with-past) ---")
    print(f"Per-Token Latency (mean):     {result.per_token_latency.mean:.2f} ms (+/- {result.per_token_latency.std:.2f})")
    print(f"Generation Phase Time (mean): {result.generation_phase_time.mean:.2f} ms (+/- {result.generation_phase_time.std:.2f})")

    print("\n--- End-to-End Model Inference ---")
    print(f"End-to-End Inference (mean):  {result.end_to_end_inference_time.mean:.2f} ms (+/- {result.end_to_end_inference_time.std:.2f})")

    print(f"\nDecode Time (mean):           {result.decode_time.mean:.2f} ms")
    print(f"Wall Clock Time (mean):       {result.wall_clock_time.mean:.2f} ms")

    derived = result.compute_derived_metrics()
    print("\n--- Throughput ---")
    print(f"E2E Throughput:               {derived['e2e_throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"Generation Throughput:        {derived['generation_throughput_tokens_per_sec']:.2f} tokens/sec (decoder-with-past)")
    if result.audio_duration_sec > 0:
        print(f"Real-Time Factor:             {derived['real_time_factor']:.3f}x")
    print(f"\nTotal Tokens Generated:       {derived['avg_tokens_generated']:.1f}")
    print(f"Generation Tokens (decoder):  {derived['avg_generation_tokens']:.1f}")

    if result.peak_memory_gb > 0:
        memory_type = "GPU" if IS_NVIDIA_SYSTEM else "CPU"
        print(f"Peak {memory_type} Memory:          {result.peak_memory_gb:.2f} GB")

    if verbose:
        print("\nDetailed Statistics:")
        print(f"  Prefill Time - min: {result.prefill_time.min:.2f}, max: {result.prefill_time.max:.2f}, median: {result.prefill_time.median:.2f}")
        print(f"  Per-Token Latency - min: {result.per_token_latency.min:.2f}, max: {result.per_token_latency.max:.2f}, median: {result.per_token_latency.median:.2f}")
        print(f"  E2E Inference Time - min: {result.end_to_end_inference_time.min:.2f}, max: {result.end_to_end_inference_time.max:.2f}, median: {result.end_to_end_inference_time.median:.2f}")


def main(args):
    """Main entry point for the benchmark."""
    # Validate audio paths
    audio_paths = [p.strip() for p in args.audio_paths.split(",")]
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Parse batch sizes and beam counts
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    num_beams_list = [int(n) for n in args.num_beams_list.split(",")]

    # Load model (measure load time)
    print("Loading model...")
    model_load_start = time.perf_counter()
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            if args.verbose:
                print(f"Setting execution provider to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()
    model_load_end = time.perf_counter()
    model_load_time_ms = (model_load_end - model_load_start) * 1000
    print(f"Model loaded in {model_load_time_ms:.2f} ms")

    # Run benchmarks for all configurations
    all_results = []
    for batch_size in batch_sizes:
        for num_beams in num_beams_list:
            print(f"\nBenchmarking: batch_size={batch_size}, num_beams={num_beams}")

            if args.monitor_memory:
                result = run_benchmark_with_memory(
                    args, model, processor, audio_paths, batch_size, num_beams
                )
            else:
                result = run_benchmark(
                    args, model, processor, audio_paths, batch_size, num_beams
                )

            result.model_load_time_ms = model_load_time_ms
            all_results.append(result)
            print_summary(result, args.verbose)

    # Save results
    save_results(args, all_results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper model benchmarking for ONNX Runtime GenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmark_whisper.py -m /path/to/whisper -a audio.mp3

  # Full benchmark suite
  python benchmark_whisper.py -m /path/to/whisper -a "audio1.mp3,audio2.wav" \\
      -e cuda -b 1,2,4 -n 1,4 -w 5 -r 10 --monitor_memory -o results.json --save_csv -v
        """,
    )

    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the Whisper ONNX model directory (must contain genai_config.json)",
    )
    parser.add_argument(
        "-a",
        "--audio_paths",
        type=str,
        required=True,
        help="Comma-separated paths to audio files (e.g., 'audio1.mp3,audio2.wav')",
    )
    parser.add_argument(
        "-e",
        "--execution_provider",
        type=str,
        default="follow_config",
        choices=["cpu", "cuda", "dml", "follow_config"],
        help="Execution provider for ONNX Runtime. 'follow_config' uses genai_config.json setting (default: follow_config)",
    )
    parser.add_argument(
        "-b",
        "--batch_sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes to benchmark (default: 1)",
    )
    parser.add_argument(
        "-n",
        "--num_beams_list",
        type=str,
        default="1",
        help="Comma-separated beam counts to benchmark (default: 1)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=448,
        help="Maximum generation length (default: 448)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=10,
        help="Number of measured iterations (default: 10)",
    )
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="e2e",
        choices=["e2e", "model_only"],
        help="Benchmark mode: 'e2e' includes audio load + preprocessing; 'model_only' reuses prepared inputs (default: e2e)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="whisper_benchmark.json",
        help="Output JSON file name (default: whisper_benchmark.json)",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save results as CSV",
    )
    parser.add_argument(
        "--monitor_memory",
        action="store_true",
        help="Monitor and report peak memory usage",
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=0.0,
        help="Audio duration in seconds (for RTF calculation). If not provided, RTF won't be computed.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="whisper",
        help="Model name for metrics output (default: whisper)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        help="Model precision for metrics info (default: fp16)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--print_transcription",
        action="store_true",
        help="Print transcription output for each iteration",
    )

    args = parser.parse_args()
    main(args)
