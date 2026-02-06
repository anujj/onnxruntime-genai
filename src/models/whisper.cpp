// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"
namespace Generators {
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

  // attention_mask is only supported for NvTensorRtRtx EP
  const bool is_trt_rtx = model_.p_device_inputs_->GetType() == DeviceType::NvTensorRtRtx;
  has_attention_mask_input_ = is_trt_rtx && HasAttentionMaskInput();
  // Note: attention_mask will be initialized lazily on first run when current_length is known

  // position_ids input (optional)
  has_position_ids_input_ = model_.session_info_.HasInput(model_.config_->model.decoder.inputs.position_ids);

  output_cross_qk_name_ = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, 0);
}

template <typename T>
void WhisperDecoderState::CreateAndInitializeAttentionMask(int64_t valid_length) {
  int64_t batch_size = attention_mask_shape_[0];
  int64_t buffer_length = attention_mask_shape_[1];  // max_length for static, current length for dynamic

  // Check if we're on GPU - if so, we need to create on CPU first, then copy
  bool is_gpu = (model_.p_device_inputs_->GetType() != DeviceType::CPU);

  if (is_gpu) {
    // GPU path: Create on CPU, initialize, then copy to GPU
    auto& cpu_allocator = GetDeviceInterface(DeviceType::CPU)->GetAllocator();
    auto cpu_tensor = OrtValue::CreateTensor(cpu_allocator, attention_mask_shape_, mask_type_);

    auto* mask_data = cpu_tensor->GetTensorMutableData<T>();

    // Initialize mask values on CPU
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

    // Create GPU tensor
    auto& gpu_allocator = model_.p_device_inputs_->GetAllocator();
    attention_mask_ = OrtValue::CreateTensor(gpu_allocator, attention_mask_shape_, mask_type_);

    // Copy CPU to GPU
    auto cpu_span = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *cpu_tensor);
    auto gpu_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
    gpu_span.CopyFrom(cpu_span);
  } else {
    // CPU path: Create and initialize directly
    auto& allocator = model_.p_device_inputs_->GetAllocator();
    attention_mask_ = OrtValue::CreateTensor(allocator, attention_mask_shape_, mask_type_);

    auto* mask_data = attention_mask_->GetTensorMutableData<T>();

    // Initialize mask values
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

void WhisperDecoderState::UpdateAttentionMask(int current_length, int new_kv_length) {
  int64_t batch_size = attention_mask_shape_[0];

  if (use_static_buffer_) {
    // Static buffer mode: update in-place
    int64_t max_length = attention_mask_shape_[1];
    if (!model_.p_device_inputs_->UpdateAttentionMask(nullptr,
                                                      attention_mask_->GetTensorMutableRawData(),
                                                      static_cast<int>(batch_size),
                                                      new_kv_length,
                                                      current_length,
                                                      static_cast<int>(max_length),
                                                      true,
                                                      mask_type_)) {
      auto attention_mask_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
      GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(nullptr,
                                                               attention_mask_span.CopyDeviceToCpu().data(),
                                                               static_cast<int>(batch_size),
                                                               new_kv_length,
                                                               current_length,
                                                               static_cast<int>(max_length),
                                                               true,
                                                               mask_type_);
      attention_mask_span.CopyCpuToDevice();
    }
    return;
  }

  // Dynamic mode: create new tensor with expanded length
  attention_mask_shape_[1] = current_length;
  attention_mask_next_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), attention_mask_shape_, mask_type_);

  if (!model_.p_device_inputs_->UpdateAttentionMask(attention_mask_next_->GetTensorMutableRawData(),
                                                    attention_mask_->GetTensorMutableRawData(),
                                                    static_cast<int>(batch_size),
                                                    new_kv_length,
                                                    current_length,
                                                    params_->search.max_length,
                                                    false,
                                                    mask_type_)) {
    auto attention_mask_next_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_next_);
    auto attention_mask_span = ByteWrapTensor(*model_.p_device_inputs_, *attention_mask_);
    GetDeviceInterface(DeviceType::CPU)->UpdateAttentionMask(attention_mask_next_span.CopyDeviceToCpu().data(),
                                                             attention_mask_span.CopyDeviceToCpu().data(),
                                                             static_cast<int>(batch_size),
                                                             new_kv_length,
                                                             current_length,
                                                             params_->search.max_length,
                                                             false,
                                                             mask_type_);
    attention_mask_next_span.CopyCpuToDevice();
    attention_mask_span.CopyCpuToDevice();
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

DeviceSpan<float> WhisperDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  // Initialize attention_mask on first run (lazy initialization)
  if (first_run_ && has_attention_mask_input_) {
    // Get data type from model session
    mask_type_ = model_.session_info_.GetInputDataType(model_.config_->model.decoder.inputs.attention_mask);

    // Validate type (must be INT32 or INT64)
    if (mask_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
        mask_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      throw std::runtime_error("attention_mask must be int32 or int64 type");
    }

    // Check if using in-place KV cache (static buffer)
    use_static_buffer_ = params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);

    // Set shape based on mode:
    // - Static buffer (in-place KV cache): pre-allocate to max_length
    // - Dynamic mode: allocate to current_length
    if (use_static_buffer_) {
      attention_mask_shape_ = {params_->BatchBeamSize(), params_->search.max_length};
    } else {
      attention_mask_shape_ = {params_->BatchBeamSize(), current_length};
    }

    // Create initial attention mask tensor
    if (mask_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      CreateAndInitializeAttentionMask<int32_t>(current_length);
    } else {
      CreateAndInitializeAttentionMask<int64_t>(current_length);
    }

    // Register as input
    input_names_.push_back(model_.config_->model.decoder.inputs.attention_mask.c_str());
    inputs_.push_back(attention_mask_.get());
  }

  // Add output QK on first run
  if (first_run_ && model_.session_info_.HasOutput(output_cross_qk_name_)) {
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
  }

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }
  State::Run(*model_.session_decoder_);

  return logits_.Get();
}

void WhisperDecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length, bool first_update) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  size_t new_length = next_tokens.size() / batch_size;

  input_ids_.Update(next_tokens);
  kv_cache_->Update(beam_indices, current_length);
  logits_.Update(next_tokens, first_run_ ? current_length : new_length);

  // Update position_ids if model expects it
  if (has_position_ids_input_) {
    const auto& pos_name = model_.config_->model.decoder.inputs.position_ids;
    if (position_ids_index_ == ~0U) {
      position_ids_type_ = model_.session_info_.GetInputDataType(pos_name);
      if (position_ids_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
          position_ids_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error("position_ids must be int32 or int64 type");
      }
      position_ids_shape_ = {params_->BatchBeamSize(), static_cast<int64_t>(new_length)};
      position_ids_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), position_ids_shape_, position_ids_type_);
      position_ids_index_ = inputs_.size();
      input_names_.push_back(pos_name.c_str());
      inputs_.push_back(position_ids_.get());
    } else if (position_ids_shape_[1] != static_cast<int64_t>(new_length)) {
      position_ids_shape_[1] = static_cast<int64_t>(new_length);
      position_ids_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), position_ids_shape_, position_ids_type_);
      inputs_[position_ids_index_] = position_ids_.get();
    }

    if (!model_.p_device_inputs_->UpdatePositionIds(position_ids_->GetTensorMutableRawData(),
                                                    static_cast<int>(position_ids_shape_[0]),
                                                    current_length,
                                                    static_cast<int>(new_length),
                                                    position_ids_type_)) {
      auto position_ids_span = ByteWrapTensor(*model_.p_device_inputs_, *position_ids_);
      GetDeviceInterface(DeviceType::CPU)->UpdatePositionIds(position_ids_span.CopyDeviceToCpu().data(),
                                                            static_cast<int>(position_ids_shape_[0]),
                                                            current_length,
                                                            static_cast<int>(new_length),
                                                            position_ids_type_);
      position_ids_span.CopyCpuToDevice();
    }
  }

  // Return early if this method is just initializing the above OrtValue objects and not updating them
  if (first_run_) {
    return;
  }

  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }

  // Update attention_mask if model supports it (GQA in-place KV cache)
  if (has_attention_mask_input_) {
    UpdateAttentionMask(current_length, static_cast<int>(new_length));
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
