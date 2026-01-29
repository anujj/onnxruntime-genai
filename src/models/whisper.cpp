// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace Generators {

// ============================================================================
// DEBUG LOGGING (Controlled by WHISPER_DEBUG_LOG environment variable)
// ============================================================================
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // Suppress getenv deprecation warning
#endif

static bool IsWhisperDebugEnabled() {
  static bool initialized = false;
  static bool enabled = false;
  if (!initialized) {
    const char* env = std::getenv("WHISPER_DEBUG_LOG");
    enabled = (env != nullptr && (std::string(env) == "1" || std::string(env) == "true"));
    if (enabled) {
      std::cout << "[WHISPER_DEBUG] Debug logging ENABLED (WHISPER_DEBUG_LOG=" << env << ")" << std::endl;
    }
    initialized = true;
  }
  return enabled;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define WHISPER_DEBUG_LOG(tag, msg) \
  do { \
    if (IsWhisperDebugEnabled()) { \
      std::cout << "[WHISPER_DEBUG][" << tag << "] " << msg << std::endl; \
    } \
  } while(0)

static void LogTensorShape(const char* name, const OrtValue* tensor) {
  if (!IsWhisperDebugEnabled() || !tensor) return;
  auto shape_info = tensor->GetTensorTypeAndShapeInfo();
  auto shape = shape_info->GetShape();
  std::cout << "  " << name << ": shape [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i < shape.size() - 1) std::cout << ", ";
  }
  std::cout << "]";
}
// ============================================================================

WhisperModel::WhisperModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  encoder_session_options_ = OrtSessionOptions::Create();
  CreateSessionOptionsFromConfig(config_->model.encoder.session_options.has_value() ? config_->model.encoder.session_options.value() : config_->model.decoder.session_options, *encoder_session_options_, true, false);

  session_encoder_ = CreateSession(ort_env, config_->model.encoder.filename, encoder_session_options_.get());
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());

  session_info_.Add(*session_decoder_);
  session_info_.Add(*session_encoder_);
}

std::unique_ptr<State> WhisperModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<WhisperState>(*this, params, sequence_lengths);
}

AudioEncoderState::AudioEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {}

void AudioEncoderState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  // Add audio features
  audio_features_ = std::make_unique<AudioFeatures>(*this, model_.config_->model.encoder.inputs.audio_features, extra_inputs);
  audio_features_->Add();

  // Verify that the frame size is expected
  const int num_frames = static_cast<int>(audio_features_->GetShape()[2]);
  if (num_frames != GetNumFrames()) {
    throw new std::runtime_error("Whisper uses num_frames = 3000. The provided inputs have num_frames = " + std::to_string(num_frames));
  }

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), GetNumFrames() / 2, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), hidden_states_shape, audio_features_->GetType());
  outputs_.push_back(hidden_states_.get());
  output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
}

DeviceSpan<float> AudioEncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (model_.config_->model.encoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.encoder.run_options.value());
  }
  State::Run(*model_.session_encoder_);
  return {};
}

WhisperDecoderState::WhisperDecoderState(const WhisperModel& model, const GeneratorParams& params, const int num_frames)
    : State{params, model},
      model_{model},
      kv_cache_(CreateKeyValueCache(*this)),
      num_frames_{num_frames} {
  input_ids_.Add();
  logits_.Add();
  kv_cache_->Add();

  // Add past sequence length
  if (HasPastSequenceLengthInput()) {
    auto past_sequence_length_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length);
    auto past_sequence_length_shape = std::array<int64_t, 1>{1};
    past_sequence_length_ = OrtValue::CreateTensor(GetDeviceInterface(DeviceType::CPU)->GetAllocator(), past_sequence_length_shape, past_sequence_length_type);
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = 0;

    input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    inputs_.push_back(past_sequence_length_.get());
  }

  // Add cache indirection
  if (HasCacheIndirectionInput()) {
    auto cache_indirection_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->search.batch_size, params_->search.num_beams, params_->search.max_length};
    cache_indirection_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cache_indirection_shape, cache_indirection_type);
    cache_indirection_index_ = inputs_.size();

    input_names_.push_back(model_.config_->model.decoder.inputs.cache_indirection.c_str());
    inputs_.push_back(cache_indirection_.get());

    ByteWrapTensor(*model_.p_device_inputs_, *cache_indirection_).Zero();
  }

  // Check if model has attention_mask input (for GQA in-place KV cache)
  has_attention_mask_input_ = HasAttentionMaskInput();
  // Note: attention_mask will be initialized lazily on first run when current_length is known

  output_cross_qk_name_ = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, 0);
}

template <typename T>
void WhisperDecoderState::CreateAndInitializeAttentionMask(int64_t valid_length) {
  WHISPER_DEBUG_LOG("CREATE_MASK", "  >> CreateAndInitializeAttentionMask ENTRY"); std::cout.flush();
  WHISPER_DEBUG_LOG("CREATE_MASK", "     valid_length=" << valid_length); std::cout.flush();

  try {
    int64_t batch_size = attention_mask_shape_[0];
    int64_t buffer_length = attention_mask_shape_[1];  // max_length for static, current length for dynamic
    WHISPER_DEBUG_LOG("CREATE_MASK", "     batch_size=" << batch_size << " buffer_length=" << buffer_length); std::cout.flush();

    // Check if we're on GPU - if so, we need to create on CPU first, then copy
    bool is_gpu = (model_.p_device_inputs_->GetType() != DeviceType::CPU);
    WHISPER_DEBUG_LOG("CREATE_MASK", "     is_gpu=" << is_gpu << " device_type=" << static_cast<int>(model_.p_device_inputs_->GetType())); std::cout.flush();

    if (is_gpu) {
      // GPU path: Create on CPU, initialize, then copy to GPU
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Creating CPU tensor for initialization..."); std::cout.flush();
      auto& cpu_allocator = GetDeviceInterface(DeviceType::CPU)->GetAllocator();
      auto cpu_tensor = OrtValue::CreateTensor(cpu_allocator, attention_mask_shape_, mask_type_);
      WHISPER_DEBUG_LOG("CREATE_MASK", "     CPU tensor created"); std::cout.flush();

      auto* mask_data = cpu_tensor->GetTensorMutableData<T>();
      WHISPER_DEBUG_LOG("CREATE_MASK", "     CPU mask_data ptr=" << (void*)mask_data); std::cout.flush();

      // Initialize mask values on CPU
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Initializing mask values on CPU (use_static_buffer_=" << use_static_buffer_ << ")..."); std::cout.flush();
      if (use_static_buffer_) {
        // For static buffer: valid tokens followed by padding
        for (int64_t batch = 0; batch < batch_size; ++batch) {
          for (int64_t i = 0; i < buffer_length; ++i) {
            mask_data[batch * buffer_length + i] = (i < valid_length) ? static_cast<T>(1) : static_cast<T>(0);
          }
        }
      } else {
        // For dynamic mode: all tokens are valid (no pre-allocated padding)
        int64_t total_elements = batch_size * buffer_length;
        for (int64_t i = 0; i < total_elements; ++i) {
          mask_data[i] = static_cast<T>(1);
        }
      }
      WHISPER_DEBUG_LOG("CREATE_MASK", "     CPU initialization complete"); std::cout.flush();

      // Create GPU tensor
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Creating GPU tensor..."); std::cout.flush();
      auto& gpu_allocator = model_.p_device_inputs_->GetAllocator();
      attention_mask_ = OrtValue::CreateTensor(gpu_allocator, attention_mask_shape_, mask_type_);
      WHISPER_DEBUG_LOG("CREATE_MASK", "     GPU tensor created"); std::cout.flush();

      // Copy CPU to GPU
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Copying CPU to GPU..."); std::cout.flush();
      auto cpu_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_tensor);
      auto gpu_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
      gpu_span.CopyFrom(cpu_span);
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Copy complete"); std::cout.flush();
    } else {
      // CPU path: Create and initialize directly
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Creating CPU tensor directly..."); std::cout.flush();
      auto& allocator = model_.p_device_inputs_->GetAllocator();
      attention_mask_ = OrtValue::CreateTensor(allocator, attention_mask_shape_, mask_type_);
      WHISPER_DEBUG_LOG("CREATE_MASK", "     CPU tensor created"); std::cout.flush();

      auto* mask_data = attention_mask_->GetTensorMutableData<T>();
      WHISPER_DEBUG_LOG("CREATE_MASK", "     mask_data ptr=" << (void*)mask_data); std::cout.flush();

      // Initialize mask values
      WHISPER_DEBUG_LOG("CREATE_MASK", "     Initializing mask values (use_static_buffer_=" << use_static_buffer_ << ")..."); std::cout.flush();
      if (use_static_buffer_) {
        for (int64_t batch = 0; batch < batch_size; ++batch) {
          for (int64_t i = 0; i < buffer_length; ++i) {
            mask_data[batch * buffer_length + i] = (i < valid_length) ? static_cast<T>(1) : static_cast<T>(0);
          }
        }
      } else {
        int64_t total_elements = batch_size * buffer_length;
        for (int64_t i = 0; i < total_elements; ++i) {
          mask_data[i] = static_cast<T>(1);
        }
      }
    }

    WHISPER_DEBUG_LOG("CREATE_MASK", "  >> CreateAndInitializeAttentionMask COMPLETE"); std::cout.flush();
  } catch (const std::exception& e) {
    WHISPER_DEBUG_LOG("CREATE_MASK", "  >> ERROR in CreateAndInitializeAttentionMask: " << e.what()); std::cout.flush();
    throw;
  } catch (...) {
    WHISPER_DEBUG_LOG("CREATE_MASK", "  >> ERROR: Unknown exception in CreateAndInitializeAttentionMask"); std::cout.flush();
    throw;
  }
}

template <typename T>
void WhisperDecoderState::UpdateAttentionMaskStaticImpl(
    T* mask_data,
    int64_t batch_size,
    int64_t current_length,
    int64_t max_length) {
  // Static buffer mode: Update in-place by setting position current_length-1 to 1
  // Example: if current_length=5, set mask[4] = 1 (0-indexed)
  // Buffer is pre-allocated: [1,1,1,1,0,0,...,0] -> [1,1,1,1,1,0,...,0]
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    if (current_length - 1 < max_length) {
      mask_data[batch * max_length + (current_length - 1)] = 1;
    }
  }
}

template <typename T>
void WhisperDecoderState::UpdateAttentionMaskDynamicImpl(
    T* next_mask_data,
    const T* current_mask_data,
    int64_t batch_size,
    int64_t old_seq_length,
    int64_t new_seq_length) {
  // Dynamic mode: Copy old mask + append 1s for new tokens
  for (int64_t i = 0; i < batch_size; ++i) {
    // Copy existing mask values
    for (int64_t j = 0; j < old_seq_length; ++j) {
      next_mask_data[i * new_seq_length + j] =
        current_mask_data[i * old_seq_length + j];
    }
    // Append 1s for new tokens
    for (int64_t j = old_seq_length; j < new_seq_length; ++j) {
      next_mask_data[i * new_seq_length + j] = 1;
    }
  }
}

void WhisperDecoderState::UpdateAttentionMask(int current_length) {
  int64_t batch_size = attention_mask_shape_[0];
  bool is_gpu = (model_.p_device_inputs_->GetType() != DeviceType::CPU);

  if (use_static_buffer_) {
    // Static buffer mode (in-place KV cache): Update in-place
    // Buffer is pre-allocated to max_length, just set new token positions to 1
    // Example: [1,1,1,1,0,0,...] -> [1,1,1,1,1,0,...]
    int64_t max_length = attention_mask_shape_[1];

    if (is_gpu) {
      // GPU path: Need to update GPU memory via copy
      // Create a small CPU buffer with the update value, copy to correct position
      // For efficiency, we copy the entire mask to CPU, update, copy back
      // (For a single value update, this is suboptimal but correct)

      // Copy GPU to CPU
      auto& cpu_allocator = GetDeviceInterface(DeviceType::CPU)->GetAllocator();
      auto cpu_tensor = OrtValue::CreateTensor(cpu_allocator, attention_mask_shape_, mask_type_);

      auto cpu_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_tensor);
      auto gpu_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
      cpu_span.CopyFrom(gpu_span);

      // Update on CPU
      if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        auto* mask_data = cpu_tensor->GetTensorMutableData<int32_t>();
        UpdateAttentionMaskStaticImpl<int32_t>(mask_data, batch_size, current_length, max_length);
      } else {
        auto* mask_data = cpu_tensor->GetTensorMutableData<int64_t>();
        UpdateAttentionMaskStaticImpl<int64_t>(mask_data, batch_size, current_length, max_length);
      }

      // Copy back to GPU
      gpu_span.CopyFrom(cpu_span);
    } else {
      // CPU path: Update directly
      if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        auto* mask_data = attention_mask_->GetTensorMutableData<int32_t>();
        UpdateAttentionMaskStaticImpl<int32_t>(mask_data, batch_size, current_length, max_length);
      } else {
        auto* mask_data = attention_mask_->GetTensorMutableData<int64_t>();
        UpdateAttentionMaskStaticImpl<int64_t>(mask_data, batch_size, current_length, max_length);
      }
    }
    // No need to update input pointer - using same buffer
  } else {
    // Dynamic mode: Create new tensor with expanded length
    int64_t old_seq_length = attention_mask_shape_[1];
    int64_t new_seq_length = current_length;

    // Update shape
    attention_mask_shape_[1] = new_seq_length;

    if (is_gpu) {
      // GPU path: Create CPU tensors, do the update, copy to GPU
      auto& cpu_allocator = GetDeviceInterface(DeviceType::CPU)->GetAllocator();

      // Copy current GPU mask to CPU
      auto old_shape = std::array<int64_t, 2>{batch_size, old_seq_length};
      auto cpu_current = OrtValue::CreateTensor(cpu_allocator, old_shape, mask_type_);
      auto cpu_current_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_current);
      auto gpu_current_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
      cpu_current_span.CopyFrom(gpu_current_span);

      // Create new CPU tensor with expanded size
      auto cpu_next = OrtValue::CreateTensor(cpu_allocator, attention_mask_shape_, mask_type_);

      // Update on CPU
      if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        UpdateAttentionMaskDynamicImpl<int32_t>(
          cpu_next->GetTensorMutableData<int32_t>(),
          cpu_current->GetTensorMutableData<int32_t>(),
          batch_size, old_seq_length, new_seq_length);
      } else {
        UpdateAttentionMaskDynamicImpl<int64_t>(
          cpu_next->GetTensorMutableData<int64_t>(),
          cpu_current->GetTensorMutableData<int64_t>(),
          batch_size, old_seq_length, new_seq_length);
      }

      // Create new GPU tensor and copy from CPU
      auto& gpu_allocator = model_.p_device_inputs_->GetAllocator();
      attention_mask_next_ = OrtValue::CreateTensor(gpu_allocator, attention_mask_shape_, mask_type_);

      auto cpu_next_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_next);
      auto gpu_next_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_next_);
      gpu_next_span.CopyFrom(cpu_next_span);
    } else {
      // CPU path: Create and update directly
      auto& allocator = model_.p_device_inputs_->GetAllocator();

      if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        attention_mask_next_ = OrtValue::CreateTensor(
          allocator, attention_mask_shape_, mask_type_);

        UpdateAttentionMaskDynamicImpl<int32_t>(
          attention_mask_next_->GetTensorMutableData<int32_t>(),
          attention_mask_->GetTensorMutableData<int32_t>(),
          batch_size, old_seq_length, new_seq_length);
      } else {
        attention_mask_next_ = OrtValue::CreateTensor(
          allocator, attention_mask_shape_, mask_type_);

        UpdateAttentionMaskDynamicImpl<int64_t>(
          attention_mask_next_->GetTensorMutableData<int64_t>(),
          attention_mask_->GetTensorMutableData<int64_t>(),
          batch_size, old_seq_length, new_seq_length);
      }
    }

    // Swap: next becomes current
    std::swap(attention_mask_, attention_mask_next_);

    // Update input pointer
    for (size_t i = 0; i < input_names_.size(); ++i) {
      if (input_names_[i] == model_.config_->model.decoder.inputs.attention_mask) {
        inputs_[i] = attention_mask_.get();
        break;
      }
    }
  }
}

DeviceSpan<float> WhisperDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  static int iteration = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  WHISPER_DEBUG_LOG("DECODER_ENTRY", "=== ITERATION " << ++iteration << " ===");
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Phase: " << (first_run_ ? "CONTEXT" : "GENERATION"));
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Current length: " << current_length);
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Batch size: " << params_->BatchBeamSize());
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Input sequence length: " << (next_tokens.size() / params_->BatchBeamSize()));
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Total inputs: " << inputs_.size());
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "Total outputs: " << outputs_.size());
  WHISPER_DEBUG_LOG("DECODER_ENTRY", "first_run_: " << first_run_ << " has_attention_mask_input_: " << has_attention_mask_input_);
  std::cout.flush();  // Flush before potential crash points

  // Initialize attention_mask on first run (lazy initialization)
  if (first_run_ && has_attention_mask_input_) {
    try {
      WHISPER_DEBUG_LOG("DECODER_INIT", "Initializing attention mask..."); std::cout.flush();
      // Get data type from model session
      WHISPER_DEBUG_LOG("DECODER_INIT", "  Getting mask data type..."); std::cout.flush();
      mask_type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);
      WHISPER_DEBUG_LOG("DECODER_INIT", "  mask_type_: " << mask_type_); std::cout.flush();

      // Validate type (must be INT32 or INT64)
      if (mask_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
          mask_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("attention_mask must be int32 or int64 type");
      }

      // Check if using in-place KV cache (static buffer)
      WHISPER_DEBUG_LOG("DECODER_INIT", "  Checking static buffer mode..."); std::cout.flush();
      use_static_buffer_ = params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);
      WHISPER_DEBUG_LOG("DECODER_INIT", "  use_static_buffer_: " << use_static_buffer_); std::cout.flush();

      // Set shape based on mode:
      // - Static buffer (in-place KV cache): pre-allocate to max_length
      // - Dynamic mode: allocate to current_length
      if (use_static_buffer_) {
        attention_mask_shape_ = {params_->BatchBeamSize(), params_->search.max_length};
      } else {
        attention_mask_shape_ = {params_->BatchBeamSize(), current_length};
      }
      WHISPER_DEBUG_LOG("DECODER_INIT", "  attention_mask_shape_: [" << attention_mask_shape_[0] << ", " << attention_mask_shape_[1] << "]"); std::cout.flush();

      // Create initial attention mask tensor
      WHISPER_DEBUG_LOG("DECODER_INIT", "  Creating attention mask tensor (mask_type=" << mask_type_ << ")..."); std::cout.flush();
      WHISPER_DEBUG_LOG("DECODER_INIT", "  About to call CreateAndInitializeAttentionMask..."); std::cout.flush();
      if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        WHISPER_DEBUG_LOG("DECODER_INIT", "  Calling CreateAndInitializeAttentionMask<int32_t>..."); std::cout.flush();
        CreateAndInitializeAttentionMask<int32_t>(current_length);
      } else {
        WHISPER_DEBUG_LOG("DECODER_INIT", "  Calling CreateAndInitializeAttentionMask<int64_t>..."); std::cout.flush();
        CreateAndInitializeAttentionMask<int64_t>(current_length);
      }
      WHISPER_DEBUG_LOG("DECODER_INIT", "  Attention mask tensor created successfully"); std::cout.flush();

      // Register as input
      WHISPER_DEBUG_LOG("DECODER_INIT", "  Registering attention mask as input..."); std::cout.flush();
      input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
      inputs_.push_back(attention_mask_.get());
      WHISPER_DEBUG_LOG("DECODER_INIT", "Attention mask initialized"); std::cout.flush();
    } catch (const std::exception& e) {
      WHISPER_DEBUG_LOG("DECODER_INIT", "ERROR in attention mask init: " << e.what());
      throw;
    } catch (...) {
      WHISPER_DEBUG_LOG("DECODER_INIT", "ERROR: Unknown exception in attention mask init");
      throw;
    }
  }

  WHISPER_DEBUG_LOG("DECODER_INIT", "Checking output_cross_qk...");

  // Add output QK on first run
  if (first_run_ && model_.session_info_.HasOutput(output_cross_qk_name_)) {
    WHISPER_DEBUG_LOG("DECODER_INIT", "Initializing output_cross_qk...");
    output_cross_qk_type_ = model_.session_info_.GetOutputDataType(output_cross_qk_name_);
    output_cross_qk_shape_ = std::array<int64_t, 4>{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, current_length, num_frames_ / 2};
    output_cross_qk_index_ = outputs_.size();

    int num_hidden_layers = model_.config_->model.decoder.num_hidden_layers;
    output_cross_qk_names_.reserve(num_hidden_layers);
    output_cross_qk_.reserve(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++) {
      output_cross_qk_.emplace_back(OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), output_cross_qk_shape_, output_cross_qk_type_));
      output_cross_qk_names_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, i));

      output_names_.emplace_back(output_cross_qk_names_.back().c_str());
      outputs_.emplace_back(output_cross_qk_.back().get());
    }
    WHISPER_DEBUG_LOG("DECODER_INIT", "output_cross_qk initialized");
  }

  WHISPER_DEBUG_LOG("DECODER_INIT", "All initialization complete, about to log inputs...");

  // Log all decoder inputs before ONNX session run
  if (IsWhisperDebugEnabled()) {
    WHISPER_DEBUG_LOG("DECODER_INPUTS", "Count: " << input_names_.size());
    for (size_t i = 0; i < input_names_.size() && i < inputs_.size(); ++i) {
      std::cout << "[WHISPER_DEBUG][INPUT_" << i << "] " << input_names_[i];
      LogTensorShape(input_names_[i], inputs_[i]);

      // Log attention_mask details
      if (std::string(input_names_[i]) == "attention_mask" ||
          std::string(input_names_[i]).find("attention_mask") != std::string::npos) {
        std::cout << " (GPU tensor - values not accessible from CPU)";
      }
      // Log cross-attention KV cache
      else if (std::string(input_names_[i]).find("past_key_cross") != std::string::npos ||
               std::string(input_names_[i]).find("past_value_cross") != std::string::npos) {
        std::cout << " [ENCODER KV]";
      }
      std::cout << std::endl;
    }
  }

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  try {
    WHISPER_DEBUG_LOG("DECODER_RUN", "Calling ONNX Runtime session...");
    std::cout.flush();  // Flush output before session run
    State::Run(*model_.session_decoder_);
    WHISPER_DEBUG_LOG("DECODER_RUN", "ONNX Runtime session completed");
  } catch (const std::exception& e) {
    WHISPER_DEBUG_LOG("DECODER_RUN", "ERROR in session run: " << e.what());
    throw;
  } catch (...) {
    WHISPER_DEBUG_LOG("DECODER_RUN", "ERROR: Unknown exception in session run");
    throw;
  }

  // Log decoder outputs
  if (IsWhisperDebugEnabled()) {
    WHISPER_DEBUG_LOG("DECODER_OUTPUTS", "Count: " << output_names_.size());
    for (size_t i = 0; i < output_names_.size() && i < outputs_.size(); ++i) {
      std::cout << "[WHISPER_DEBUG][OUTPUT_" << i << "] " << output_names_[i];
      LogTensorShape(output_names_[i], outputs_[i]);
      std::cout << std::endl;
    }

    // Check logits for NaN/Inf
    auto logits = logits_.Get();
    WHISPER_DEBUG_LOG("DECODER_OUTPUTS", "Logits shape: [" << params_->BatchBeamSize() << ", "
                      << logits.size() / params_->BatchBeamSize() / model_.config_->model.vocab_size << ", "
                      << model_.config_->model.vocab_size << "]");
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  WHISPER_DEBUG_LOG("DECODER_EXIT", "Status: SUCCESS");
  WHISPER_DEBUG_LOG("DECODER_EXIT", "Execution time: " << duration << " ms");
  WHISPER_DEBUG_LOG("DECODER_EXIT", "=== END ITERATION " << iteration << " ===\n");

  return logits_.Get();
}

void WhisperDecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length, bool first_update) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;

  WHISPER_DEBUG_LOG("UPDATE_INPUTS", "Current length: " << current_length << ", New length: " << new_length);
  WHISPER_DEBUG_LOG("UPDATE_INPUTS", "First update: " << (first_update ? "YES" : "NO") << ", First run: " << (first_run_ ? "YES" : "NO"));

  input_ids_.Update(next_tokens);
  kv_cache_->Update(beam_indices, current_length);
  logits_.Update(next_tokens, first_run_ ? current_length : new_length);

  // Return early if this method is just initializing the above OrtValue objects and not updating them
  if (first_run_) {
    WHISPER_DEBUG_LOG("UPDATE_INPUTS", "First run - initialization only, returning early");
    return;
  }

  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
    WHISPER_DEBUG_LOG("UPDATE_INPUTS", "Updated past_sequence_length: " << *data);
  }

  // Update attention_mask if model supports it (GQA in-place KV cache)
  if (has_attention_mask_input_) {
    WHISPER_DEBUG_LOG("ATTENTION_MASK", "Before update - shape: [" << attention_mask_shape_[0] << ", " << attention_mask_shape_[1] << "]");
    WHISPER_DEBUG_LOG("ATTENTION_MASK", "Updating for current_length=" << current_length);

    UpdateAttentionMask(current_length);

    WHISPER_DEBUG_LOG("ATTENTION_MASK", "After update - shape: [" << attention_mask_shape_[0] << ", " << attention_mask_shape_[1] << "]");
  }

  if (cache_indirection_ && params_->search.num_beams > 1 && !first_update) {
    // Only update after having run one pass through the decoder with past KV caches
    auto beam_indices_span = beam_indices.Span();
    if (beam_indices_span.empty()) {
      auto beam_indices_cpu = beam_indices.CpuSpan();
      std::iota(beam_indices_cpu.begin(), beam_indices_cpu.end(), 0);
      beam_indices.CopyCpuToDevice();
    }
    std::unique_ptr<OrtValue> new_cache_indirection;
    auto cache_indirection_type = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->search.batch_size, params_->search.num_beams, params_->search.max_length};
    new_cache_indirection = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cache_indirection_shape, cache_indirection_type);

    model_.p_device_inputs_->UpdateCacheIndirection(new_cache_indirection->GetTensorMutableData<int32_t>(),
                                                    cache_indirection_->GetTensorData<int32_t>(),
                                                    beam_indices_span.data(),
                                                    params_->search.batch_size,
                                                    params_->search.num_beams,
                                                    static_cast<int>(new_length),  // sequence length in input ids & logits during each forward pass
                                                    params_->search.max_length,    // max sequence length
                                                    current_length);               // total sequence length after N iterations (prompt's sequence length + number of generated tokens)

    cache_indirection_ = std::move(new_cache_indirection);
    inputs_[cache_indirection_index_] = cache_indirection_.get();
  }

  if (output_cross_qk_.size() && output_cross_qk_shape_[2] != 1) {
    // Resize output QKs from (batch_size, num_heads, sequence_length, total_sequence_length) for audio processing
    // to (batch_size, num_heads, 1, total_sequence_length) for token generation
    output_cross_qk_shape_[2] = 1;
    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_[i] = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), output_cross_qk_shape_, output_cross_qk_type_);
      outputs_[output_cross_qk_index_ + i] = output_cross_qk_[i].get();
    }
  }
}

WhisperState::WhisperState(const WhisperModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths_unk)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<AudioEncoderState>(model, params);
  decoder_state_ = std::make_unique<WhisperDecoderState>(model, params, encoder_state_->GetNumFrames());

  if (encoder_state_->HasCrossKVCacheOutputs()) {
    cross_cache_ = std::make_unique<CrossCache>(*this, encoder_state_->GetNumFrames() / 2);
    encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_->AddCrossCache(cross_cache_);
    transpose_k_cache_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_cache_->GetShape(), cross_cache_->GetType());
  }
}

void WhisperState::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  encoder_state_->SetExtraInputs(extra_inputs);

  if (!encoder_state_->HasCrossKVCacheOutputs()) {
    decoder_state_->inputs_.push_back(encoder_state_->hidden_states_.get());
    decoder_state_->input_names_.push_back(model_.config_->model.decoder.inputs.encoder_hidden_states.c_str());
  }

  // Check if alignment heads input exists
  void* alignment_heads_input = nullptr;
  for (const auto& [name, value] : extra_inputs) {
    if (name == Generators::Config::Defaults::AlignmentHeadsName) {
      alignment_heads_input = value.get();
    }
  }
  // Add alignment heads
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_input != nullptr) {
    auto alignment_heads = std::move(reinterpret_cast<Tensor*>(alignment_heads_input)->ort_tensor_);
    if (model_.p_device_inputs_->GetType() == DeviceType::CPU) {
      alignment_heads_ = std::move(alignment_heads);
    } else {
      auto alignment_heads_type_and_shape_info = alignment_heads->GetTensorTypeAndShapeInfo();
      auto alignment_heads_type = alignment_heads_type_and_shape_info->GetElementType();  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
      auto alignment_heads_shape = alignment_heads_type_and_shape_info->GetShape();
      alignment_heads_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), alignment_heads_shape, alignment_heads_type);

      // Since alignment_heads is a user input, we need to copy from CPU to GPU
      auto alignment_heads_src = ByteWrapTensor(*model_.p_device_inputs_, *alignment_heads);
      auto alignment_heads_dest = ByteWrapTensor(*model_.p_device_inputs_, *alignment_heads_);
      alignment_heads_dest.CopyFrom(alignment_heads_src);

      auto cross_qk_search_buffer_shape = std::array<int64_t, 4>{params_->BatchBeamSize(), alignment_heads_shape[0], params_->search.max_length, encoder_state_->GetNumFrames() / 2};
      cross_qk_search_buffer_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_qk_search_buffer_shape, decoder_state_->output_cross_qk_type_);

      // Allocate GPU buffer for storing output_cross_qk_{i} pointers
      cross_qk_ptrs_ = model_.p_device_inputs_->Allocate<void*>(model_.config_->model.decoder.num_hidden_layers);
    }
  }
}

void WhisperState::TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches) {
  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel (done on CUDA only)
  auto kv_cache_info = kv_caches[0]->GetTensorTypeAndShapeInfo();
  auto kv_cache_type = kv_cache_info->GetElementType();
  if (model_.p_device_inputs_->GetType() != DeviceType::CUDA || !(decoder_state_->UsesDecoderMaskedMHA())) {
    return;
  }

  auto kv_cache_dims = kv_cache_info->GetShape();
  auto kv_cache_element_size = Ort::SizeOf(kv_cache_type);
  auto temp_span = ByteWrapTensor(*model_.p_device_inputs_, *transpose_k_cache_buffer_);

  /* Use pre-allocated temporary buffer since we need to reformat the `K` caches for
   * `DecoderMaskedMultiHeadAttention` and we need some extra memory to do so.
   *
   * Since the self attention K caches are of size (batch_size, num_heads, past_sequence_length, head_size),
   * the cross attention K caches are of size (batch_size, num_heads, num_frames / 2, head_size), and
   * past_sequence_length <= max_sequence_length < num_frames / 2, we have pre-allocated a temporary buffer that is the
   * size of a cross attention K cache. This lets us use the same temporary buffer for both
   * the self attention and cross attention K caches.
   */

  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel
  for (int i = 0; i < kv_caches.size(); i += 2) {
    auto dest_span = ByteWrapTensor(*model_.p_device_inputs_, *kv_caches[i]);

    // Treat the 'K' caches as if they are of shape [B, N, max_length, head_size / x, x]
    // and transpose each 'K' cache into [B, N, head_size / x, max_length, x], where x = 16 / sizeof(T)
    int chunk_size = static_cast<int>(16 / kv_cache_element_size);
    if (chunk_size != 4 && chunk_size != 8) {
      throw std::runtime_error("ReorderPastStatesKernelLauncher only supports float32 or float16 precision");
    }

    // Copy the original 'K' caches to a temporary buffer in order to
    // use the destination buffer to store the transposed 'K' caches
    temp_span.subspan(0, dest_span.size()).CopyFrom(dest_span);

    // Transpose each 'K' cache
    model_.p_device_inputs_->ReorderPastStates(kv_caches[i]->GetTensorMutableRawData(),
                                               transpose_k_cache_buffer_->GetTensorMutableRawData(),
                                               static_cast<int32_t>(kv_cache_dims[0]),
                                               static_cast<int32_t>(kv_cache_dims[1]),
                                               static_cast<int32_t>(kv_cache_dims[2]),
                                               static_cast<int32_t>(kv_cache_dims[3]),
                                               chunk_size);
  }
}

template <typename T>
void WhisperState::UpdateCrossQKSearchBuffer(int current_length) {
  auto output_cross_qk_size = decoder_state_->output_cross_qk_.size();
  if (output_cross_qk_size && alignment_heads_ && model_.p_device_inputs_->GetType() == DeviceType::CUDA) {
    // Collect a GPU array of T* pointers from the list of OrtValues to pass to the kernel
    auto cross_qk_ptrs_cpu = cross_qk_ptrs_.CpuSpan();
    for (int i = 0; i < output_cross_qk_size; i++) {
      cross_qk_ptrs_cpu[i] = decoder_state_->output_cross_qk_[i]->GetTensorMutableData<T>();
    }
    cross_qk_ptrs_.CopyCpuToDevice();

    model_.p_device_inputs_->CopyCrossQK(cross_qk_search_buffer_->GetTensorMutableData<T>(),
                                         cross_qk_ptrs_.Span().data(),
                                         current_length - (first_run_ ? prompt_length_ : 1),
                                         params_->BatchBeamSize(),
                                         model_.config_->model.decoder.num_hidden_layers,
                                         static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[1]),
                                         static_cast<int32_t>(alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0]),
                                         alignment_heads_->GetTensorData<int32_t>(),
                                         static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                         params_->search.max_length,
                                         first_run_ ? prompt_length_ : 1);
  }
}

template <typename T>
void WhisperState::FinalizeCrossQK(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_ && decoder_state_->cache_indirection_ && model_.p_device_inputs_->GetType() == DeviceType::CUDA) {
    // Instantiate final output for cross QKs
    auto num_alignment_heads = alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0];
    auto cross_qk_shape = std::array<int64_t, 5>{params_->search.batch_size, params_->search.num_return_sequences, num_alignment_heads, current_length - 1, encoder_state_->GetNumFrames() / 2};
    cross_qk_final_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), cross_qk_shape, decoder_state_->output_cross_qk_type_);

    model_.p_device_inputs_->FinalizeCrossQK(current_length - 1,
                                             prompt_length_,
                                             static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[0]),
                                             params_->search.num_beams,
                                             params_->search.max_length,
                                             static_cast<int32_t>(num_alignment_heads),
                                             static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                             cross_qk_search_buffer_->GetTensorData<T>(),
                                             cross_qk_final_->GetTensorMutableData<T>(),
                                             params_->search.num_return_sequences,
                                             decoder_state_->cache_indirection_->GetTensorData<int32_t>());
  }
}

DeviceSpan<float> WhisperState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Initialize inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder-init
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    if (first_run_ && decoder_state_->model_.session_info_.HasOutput(decoder_state_->output_cross_qk_name_)) {
      prompt_length_ = static_cast<int>(decoder_state_->output_cross_qk_shape_[2]);
    }

    if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
      UpdateCrossQKSearchBuffer<Ort::Float16_t>(current_length);
    } else {
      UpdateCrossQKSearchBuffer<float>(current_length);
    }

    return logits;
  }

  if (first_run_ && decoder_state_->UsesDecoderMaskedMHA()) {
    // Transpose the K caches only when the else branch is run for the first time.
    // Otherwise the GetOutput(present_key_{self/cross}_{i}) method returns transposed K caches.
    TransposeKCaches(cross_cache_->GetValues());

    auto default_kv_cache_ptr = dynamic_cast<DefaultKeyValueCache*>(decoder_state_->kv_cache_.get());
    if (!default_kv_cache_ptr) {
      throw std::runtime_error("Unable to convert KeyValueCache to DefaultKeyValueCache");
    }
    TransposeKCaches(default_kv_cache_ptr->GetPresents());
  }

  // Update inputs and outputs for decoder
  decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

  // Run decoder-with-past
  auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

  if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
    UpdateCrossQKSearchBuffer<Ort::Float16_t>(current_length);
  } else {
    UpdateCrossQKSearchBuffer<float>(current_length);
  }

  first_run_ = false;
  return logits;
}

void WhisperState::Finalize(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_) {
    if (decoder_state_->output_cross_qk_type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
      FinalizeCrossQK<Ort::Float16_t>(current_length);
    } else {
      FinalizeCrossQK<float>(current_length);
    }
  }
}

OrtValue* WhisperState::GetInput(const char* name) {
  // Check if input name is in encoder state's inputs
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0) {
      return encoder_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0) {
      return decoder_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* WhisperState::GetOutput(const char* name) {
  // Check if output name is in encoder state's outputs
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0) {
      return encoder_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      // Note: K caches will be transposed when returned
      return decoder_state_->outputs_[i];
    }
  }

  // cross_qk_final_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk", name) == 0) {
    return cross_qk_final_.get();
  }

  // cross_qk_search_buffer_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk_search", name) == 0) {
    return cross_qk_search_buffer_.get();
  }

  return State::GetOutput(name);
};

// Explicit template instantiations
template void WhisperDecoderState::CreateAndInitializeAttentionMask<int32_t>(int64_t valid_length);
template void WhisperDecoderState::CreateAndInitializeAttentionMask<int64_t>(int64_t valid_length);
template void WhisperDecoderState::UpdateAttentionMaskStaticImpl<int32_t>(int32_t* mask_data, int64_t batch_size, int64_t current_length, int64_t max_length);
template void WhisperDecoderState::UpdateAttentionMaskStaticImpl<int64_t>(int64_t* mask_data, int64_t batch_size, int64_t current_length, int64_t max_length);
template void WhisperDecoderState::UpdateAttentionMaskDynamicImpl<int32_t>(int32_t* next_mask_data, const int32_t* current_mask_data, int64_t batch_size, int64_t old_seq_length, int64_t new_seq_length);
template void WhisperDecoderState::UpdateAttentionMaskDynamicImpl<int64_t>(int64_t* next_mask_data, const int64_t* current_mask_data, int64_t batch_size, int64_t old_seq_length, int64_t new_seq_length);

}  // namespace Generators
