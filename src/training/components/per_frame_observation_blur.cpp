/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "per_frame_observation_blur.hpp"

#include "core/path_utils.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "lfs/kernels/per_frame_observation_blur.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace lfs::training {

    namespace {
        constexpr uint32_t SERIALIZATION_MAGIC = 0x4C46424C; // "LFBL"
        constexpr uint32_t SERIALIZATION_VERSION = 1;
        constexpr uint32_t JSON_EXPORT_VERSION = 1;
        constexpr uint32_t MAX_IMAGE_NAME_BYTES = 1024 * 1024;
        constexpr size_t MAX_FRAME_COUNT =
            static_cast<size_t>(std::numeric_limits<int>::max()) /
            PerFrameObservationBlur::PARAMS_PER_FRAME;

        struct SerializedState {
            PerFrameObservationBlur::Config config;
            int total_iterations = 0;
            int64_t motion_step = 0;
            int64_t defocus_step = 0;
            std::vector<PerFrameObservationBlur::FrameIdentity> frames;
            lfs::core::Tensor parameters;
            lfs::core::Tensor exp_avg;
            lfs::core::Tensor exp_avg_sq;
        };

        template <typename T>
        void write_pod(std::ostream& os, const T& value) {
            os.write(reinterpret_cast<const char*>(&value), sizeof(T));
            if (!os) {
                throw std::runtime_error("Failed to write PerFrameObservationBlur state");
            }
        }

        template <typename T>
        T read_pod(std::istream& is, const char* const field) {
            T value{};
            is.read(reinterpret_cast<char*>(&value), sizeof(T));
            if (!is) {
                throw std::runtime_error(std::format(
                    "Truncated PerFrameObservationBlur state while reading {}", field));
            }
            return value;
        }

        void validate_config(
            const PerFrameObservationBlur::Config& config,
            const int total_iterations) {
            const auto finite_nonnegative = [](const double value) {
                return std::isfinite(value) && value >= 0.0;
            };
            if (total_iterations <= 0) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur total_iterations must be positive");
            }
            if (!config.motion_enabled && !config.defocus_enabled) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur requires motion or defocus enabled");
            }
            if (!finite_nonnegative(config.motion_rot_lr) ||
                !finite_nonnegative(config.motion_trans_lr) ||
                !finite_nonnegative(config.defocus_scale_lr) ||
                !finite_nonnegative(config.defocus_focus_lr) ||
                !finite_nonnegative(config.motion_rot_reg_weight) ||
                !finite_nonnegative(config.motion_trans_reg_weight) ||
                !finite_nonnegative(config.defocus_reg_weight) ||
                !finite_nonnegative(config.max_rot_variance) ||
                !finite_nonnegative(config.max_trans_variance) ||
                !finite_nonnegative(config.max_defocus_radius_sq) ||
                !finite_nonnegative(config.focus_inverse_depth_abs_max) ||
                !std::isfinite(config.initial_focus_inverse_depth)) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur rates, weights, caps, and focus initialization must be finite");
            }
            if (config.motion_enabled &&
                (!(config.max_rot_variance > 0.0) ||
                 !(config.max_trans_variance > 0.0))) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur motion caps must be positive");
            }
            if (config.defocus_enabled &&
                (!(config.max_defocus_radius_sq > 0.0) ||
                 !(config.focus_inverse_depth_abs_max > 0.0) ||
                 config.initial_focus_inverse_depth < 0.0 ||
                 config.initial_focus_inverse_depth >
                     config.focus_inverse_depth_abs_max)) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur defocus caps must be positive and focus initialization non-negative");
            }
            if (config.motion_start_iter < 0 || config.defocus_start_iter < 0) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur start iterations must be non-negative");
            }
            if (!(config.beta1 >= 0.0 && config.beta1 < 1.0) ||
                !(config.beta2 >= 0.0 && config.beta2 < 1.0) ||
                !std::isfinite(config.beta1) || !std::isfinite(config.beta2) ||
                !(config.eps > 0.0) || !std::isfinite(config.eps)) {
                throw std::invalid_argument(
                    "PerFrameObservationBlur Adam parameters are invalid");
            }
        }

        void validate_state_tensor(
            const lfs::core::Tensor& tensor,
            const size_t num_frames,
            const char* const name) {
            if (!tensor.is_valid() ||
                tensor.dtype() != lfs::core::DataType::Float32 ||
                tensor.ndim() != 2 || tensor.shape()[0] != num_frames ||
                tensor.shape()[1] != PerFrameObservationBlur::PARAMS_PER_FRAME) {
                throw std::runtime_error(std::format(
                    "Invalid PerFrameObservationBlur {} tensor; expected [{},{}] Float32",
                    name, num_frames, PerFrameObservationBlur::PARAMS_PER_FRAME));
            }
        }

        void validate_state_values(
            const lfs::core::Tensor& tensor,
            const PerFrameObservationBlur::Config& config,
            const char* const name) {
            const std::vector<float> values = tensor.cpu().to_vector();
            for (size_t index = 0; index < values.size(); ++index) {
                const float value = values[index];
                if (!std::isfinite(value)) {
                    throw std::runtime_error(std::format(
                        "Invalid PerFrameObservationBlur {} tensor: value {} is not finite",
                        name, index));
                }
                if (std::string_view(name) == "exp_avg_sq" && value < 0.0f) {
                    throw std::runtime_error(std::format(
                        "Invalid PerFrameObservationBlur {} tensor: value {} is negative",
                        name, index));
                }
            }
            if (std::string_view(name) != "parameters") {
                return;
            }
            constexpr size_t stride = PerFrameObservationBlur::PARAMS_PER_FRAME;
            for (size_t offset = 0; offset + stride <= values.size(); offset += stride) {
                for (size_t parameter = 0; parameter < 7; ++parameter) {
                    if (values[offset + parameter] < 0.0f) {
                        throw std::runtime_error(std::format(
                            "Invalid PerFrameObservationBlur parameters tensor: parameter {} is negative",
                            offset + parameter));
                    }
                }
                if (values[offset + 7] < 0.0f ||
                    values[offset + 7] > static_cast<float>(config.focus_inverse_depth_abs_max)) {
                    throw std::runtime_error(std::format(
                        "Invalid PerFrameObservationBlur parameters tensor: focus inverse depth at row {} is outside [0, cap]",
                        offset / stride));
                }
                if (config.motion_enabled &&
                    (values[offset] > static_cast<float>(config.max_rot_variance) ||
                     values[offset + 1] > static_cast<float>(config.max_rot_variance) ||
                     values[offset + 2] > static_cast<float>(config.max_rot_variance) ||
                     values[offset + 3] > static_cast<float>(config.max_trans_variance) ||
                     values[offset + 4] > static_cast<float>(config.max_trans_variance) ||
                     values[offset + 5] > static_cast<float>(config.max_trans_variance))) {
                    throw std::runtime_error(
                        "Invalid PerFrameObservationBlur parameters tensor: motion variance exceeds configured cap");
                }
            }
        }

        size_t serialized_tensor_size(const lfs::core::Tensor& tensor) {
            return sizeof(lfs::core::TensorFileHeader) +
                   tensor.ndim() * sizeof(uint64_t) + tensor.bytes();
        }

        SerializedState read_serialized_state(std::istream& is) {
            const uint32_t magic = read_pod<uint32_t>(is, "magic");
            const uint32_t version = read_pod<uint32_t>(is, "version");
            if (magic != SERIALIZATION_MAGIC) {
                throw std::runtime_error(
                    "Invalid PerFrameObservationBlur state: wrong magic");
            }
            if (version != SERIALIZATION_VERSION) {
                throw std::runtime_error(
                    "Unsupported PerFrameObservationBlur state version");
            }

            const uint32_t frame_count = read_pod<uint32_t>(is, "frame count");
            if (frame_count == 0 || frame_count > MAX_FRAME_COUNT) {
                throw std::runtime_error(
                    "Invalid PerFrameObservationBlur frame count");
            }

            SerializedState state;
            state.config.motion_enabled =
                read_pod<uint8_t>(is, "motion enabled") != 0;
            state.config.defocus_enabled =
                read_pod<uint8_t>(is, "defocus enabled") != 0;
            state.config.motion_rot_lr = read_pod<double>(is, "motion rotation lr");
            state.config.motion_trans_lr = read_pod<double>(is, "motion translation lr");
            state.config.defocus_scale_lr = read_pod<double>(is, "defocus scale lr");
            state.config.defocus_focus_lr = read_pod<double>(is, "defocus focus lr");
            state.config.motion_rot_reg_weight =
                read_pod<double>(is, "motion rotation regularization");
            state.config.motion_trans_reg_weight =
                read_pod<double>(is, "motion translation regularization");
            state.config.defocus_reg_weight =
                read_pod<double>(is, "defocus regularization");
            state.config.motion_start_iter = read_pod<int32_t>(is, "motion start iteration");
            state.config.defocus_start_iter = read_pod<int32_t>(is, "defocus start iteration");
            state.config.max_rot_variance = read_pod<double>(is, "maximum rotation variance");
            state.config.max_trans_variance = read_pod<double>(is, "maximum translation variance");
            state.config.max_defocus_radius_sq =
                read_pod<double>(is, "maximum defocus radius squared");
            state.config.focus_inverse_depth_abs_max =
                read_pod<double>(is, "focus inverse depth cap");
            state.config.initial_focus_inverse_depth =
                read_pod<double>(is, "initial focus inverse depth");
            state.config.beta1 = read_pod<double>(is, "beta1");
            state.config.beta2 = read_pod<double>(is, "beta2");
            state.config.eps = read_pod<double>(is, "epsilon");
            state.motion_step = read_pod<int64_t>(is, "motion step");
            state.defocus_step = read_pod<int64_t>(is, "defocus step");
            state.total_iterations = read_pod<int32_t>(is, "total iterations");
            validate_config(state.config, state.total_iterations);
            if (state.motion_step < 0 || state.defocus_step < 0) {
                throw std::runtime_error(
                    "Invalid PerFrameObservationBlur optimizer step");
            }

            state.frames.reserve(frame_count);
            for (uint32_t index = 0; index < frame_count; ++index) {
                PerFrameObservationBlur::FrameIdentity frame;
                frame.uid = read_pod<int32_t>(is, "frame uid");
                frame.camera_id = read_pod<int32_t>(is, "frame camera id");
                const uint32_t name_size = read_pod<uint32_t>(is, "image name size");
                if (name_size > MAX_IMAGE_NAME_BYTES) {
                    throw std::runtime_error(
                        "PerFrameObservationBlur image name exceeds size limit");
                }
                frame.image_name.resize(name_size);
                if (name_size > 0) {
                    is.read(frame.image_name.data(),
                            static_cast<std::streamsize>(name_size));
                    if (!is) {
                        throw std::runtime_error(
                            "Truncated PerFrameObservationBlur image name");
                    }
                }
                state.frames.push_back(std::move(frame));
            }

            is >> state.parameters >> state.exp_avg >> state.exp_avg_sq;
            validate_state_tensor(state.parameters, state.frames.size(), "parameters");
            validate_state_tensor(state.exp_avg, state.frames.size(), "exp_avg");
            validate_state_tensor(state.exp_avg_sq, state.frames.size(), "exp_avg_sq");
            validate_state_values(state.parameters, state.config, "parameters");
            validate_state_values(state.exp_avg, state.config, "exp_avg");
            validate_state_values(state.exp_avg_sq, state.config, "exp_avg_sq");
            return state;
        }

        std::expected<void, std::string> replace_export_file(
            const std::filesystem::path& path,
            const std::filesystem::path& temp_path) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            if (ec) {
                return std::unexpected(std::format(
                    "Failed to remove existing per-frame blur export '{}': {}",
                    lfs::core::path_to_utf8(path), ec.message()));
            }
            std::filesystem::rename(temp_path, path, ec);
            if (ec) {
                return std::unexpected(std::format(
                    "Failed to replace per-frame blur export '{}': {}",
                    lfs::core::path_to_utf8(path), ec.message()));
            }
            return {};
        }
    } // namespace

    PerFrameObservationBlur::PerFrameObservationBlur(
        const int total_iterations,
        Config config)
        : config_(std::move(config)), total_iterations_(total_iterations) {
        validate_config(config_, total_iterations_);
    }

    std::string PerFrameObservationBlur::make_frame_key(
        const std::string_view image_name,
        const int camera_id) {
        return std::format("{}\n{}", camera_id, image_name);
    }

    void PerFrameObservationBlur::register_frame(
        const int uid,
        std::string image_name,
        const int camera_id) {
        if (finalized_) {
            throw std::logic_error(
                "Cannot register PerFrameObservationBlur frames after finalize()");
        }
        if (uid_to_slot_.contains(uid)) {
            throw std::invalid_argument(std::format(
                "Duplicate PerFrameObservationBlur frame UID {}", uid));
        }
        if (image_name.size() > MAX_IMAGE_NAME_BYTES) {
            throw std::invalid_argument(
                "PerFrameObservationBlur image name exceeds size limit");
        }
        const std::string key = make_frame_key(image_name, camera_id);
        if (frame_key_to_slot_.contains(key)) {
            throw std::invalid_argument(std::format(
                "Duplicate PerFrameObservationBlur frame key ('{}', camera {})",
                image_name, camera_id));
        }
        if (frames_.size() >= MAX_FRAME_COUNT) {
            throw std::overflow_error("Too many PerFrameObservationBlur frames");
        }

        const int slot = static_cast<int>(frames_.size());
        frames_.push_back(FrameIdentity{
            .uid = uid,
            .image_name = std::move(image_name),
            .camera_id = camera_id});
        uid_to_slot_.emplace(uid, slot);
        frame_key_to_slot_.emplace(key, slot);
    }

    void PerFrameObservationBlur::finalize() {
        if (finalized_) {
            throw std::logic_error(
                "PerFrameObservationBlur is already finalized");
        }
        if (frames_.empty()) {
            throw std::logic_error(
                "Cannot finalize PerFrameObservationBlur without real frames");
        }

        std::vector<float> initial(
            frames_.size() * PARAMS_PER_FRAME, 0.0f);
        for (size_t frame = 0; frame < frames_.size(); ++frame) {
            initial[frame * PARAMS_PER_FRAME + 7] =
                static_cast<float>(config_.initial_focus_inverse_depth);
        }
        parameters_ = lfs::core::Tensor::from_vector(
            initial,
            lfs::core::TensorShape({frames_.size(), PARAMS_PER_FRAME}),
            lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros(
            parameters_.shape(), lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);
        exp_avg_sq_ = lfs::core::Tensor::zeros(
            parameters_.shape(), lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);
        gradients_ = lfs::core::Tensor::zeros(
            parameters_.shape(), lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);
        regularization_loss_ = lfs::core::Tensor::zeros(
            {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        finalized_ = true;
    }

    bool PerFrameObservationBlur::is_known_frame(const int uid) const noexcept {
        return uid_to_slot_.contains(uid);
    }

    bool PerFrameObservationBlur::has_frame_key(
        const std::string_view image_name,
        const int camera_id) const {
        return frame_key_to_slot_.contains(make_frame_key(image_name, camera_id));
    }

    bool PerFrameObservationBlur::motion_active(const int iteration) const noexcept {
        return config_.motion_enabled && iteration >= config_.motion_start_iter;
    }

    bool PerFrameObservationBlur::defocus_active(const int iteration) const noexcept {
        return config_.defocus_enabled && iteration >= config_.defocus_start_iter;
    }

    bool PerFrameObservationBlur::any_active(const int iteration) const noexcept {
        return motion_active(iteration) || defocus_active(iteration);
    }

    int PerFrameObservationBlur::slot_for_uid(const int uid) const {
        const auto it = uid_to_slot_.find(uid);
        if (it == uid_to_slot_.end()) {
            throw std::out_of_range(std::format(
                "Unknown PerFrameObservationBlur frame UID {}", uid));
        }
        return it->second;
    }

    void PerFrameObservationBlur::require_finalized() const {
        if (!finalized_) {
            throw std::logic_error(
                "PerFrameObservationBlur must be finalized before use");
        }
    }

    const float* PerFrameObservationBlur::parameters_for_uid(const int uid) const {
        require_finalized();
        return parameters_.ptr<float>() +
               static_cast<size_t>(slot_for_uid(uid)) * PARAMS_PER_FRAME;
    }

    float* PerFrameObservationBlur::gradients_for_uid(const int uid) {
        require_finalized();
        return gradients_.ptr<float>() +
               static_cast<size_t>(slot_for_uid(uid)) * PARAMS_PER_FRAME;
    }

    lfs::core::Tensor PerFrameObservationBlur::regularization_loss_gpu(
        const int iteration) {
        require_finalized();
        cudaMemsetAsync(
            regularization_loss_.ptr<float>(), 0,
            regularization_loss_.bytes(), nullptr);
        kernels::launch_per_frame_observation_blur_regularization_forward(
            parameters_.ptr<float>(), regularization_loss_.ptr<float>(),
            num_frames(), motion_active(iteration), defocus_active(iteration),
            static_cast<float>(config_.motion_rot_reg_weight),
            static_cast<float>(config_.motion_trans_reg_weight),
            static_cast<float>(config_.defocus_reg_weight), nullptr);
        return regularization_loss_;
    }

    void PerFrameObservationBlur::regularization_backward(const int iteration) {
        require_finalized();
        kernels::launch_per_frame_observation_blur_regularization_backward(
            gradients_.ptr<float>(), num_frames(),
            motion_active(iteration), defocus_active(iteration),
            static_cast<float>(config_.motion_rot_reg_weight),
            static_cast<float>(config_.motion_trans_reg_weight),
            static_cast<float>(config_.defocus_reg_weight), nullptr);
    }

    void PerFrameObservationBlur::optimizer_step(const int iteration) {
        require_finalized();
        const bool update_motion = motion_active(iteration);
        const bool update_defocus = defocus_active(iteration);
        if (!update_motion && !update_defocus) {
            return;
        }

        const auto bias_terms = [&](const int64_t step) {
            const double correction1 = 1.0 - std::pow(config_.beta1, step + 1);
            const double correction2 = 1.0 - std::pow(config_.beta2, step + 1);
            return std::pair{
                static_cast<float>(1.0 / correction1),
                static_cast<float>(1.0 / std::sqrt(correction2))};
        };
        const auto motion_bias = bias_terms(motion_step_);
        const auto defocus_bias = bias_terms(defocus_step_);

        kernels::launch_per_frame_observation_blur_adam_update(
            parameters_.ptr<float>(), exp_avg_.ptr<float>(), exp_avg_sq_.ptr<float>(),
            gradients_.ptr<float>(), num_frames(), update_motion, update_defocus,
            static_cast<float>(config_.motion_rot_lr),
            static_cast<float>(config_.motion_trans_lr),
            static_cast<float>(config_.defocus_scale_lr),
            static_cast<float>(config_.defocus_focus_lr),
            static_cast<float>(config_.beta1),
            static_cast<float>(config_.beta2),
            motion_bias.first, motion_bias.second,
            defocus_bias.first, defocus_bias.second,
            static_cast<float>(config_.eps),
            static_cast<float>(config_.max_rot_variance),
            static_cast<float>(config_.max_trans_variance),
            static_cast<float>(config_.focus_inverse_depth_abs_max), nullptr);

        if (update_motion) {
            ++motion_step_;
        }
        if (update_defocus) {
            ++defocus_step_;
        }
    }

    void PerFrameObservationBlur::zero_grad() {
        require_finalized();
        cudaMemsetAsync(gradients_.ptr<float>(), 0, gradients_.bytes(), nullptr);
    }

    void PerFrameObservationBlur::rebuild_indices() {
        uid_to_slot_.clear();
        frame_key_to_slot_.clear();
        uid_to_slot_.reserve(frames_.size());
        frame_key_to_slot_.reserve(frames_.size());
        for (size_t index = 0; index < frames_.size(); ++index) {
            const auto& frame = frames_[index];
            const int slot = static_cast<int>(index);
            if (!uid_to_slot_.emplace(frame.uid, slot).second) {
                throw std::runtime_error(std::format(
                    "Duplicate UID {} in PerFrameObservationBlur state", frame.uid));
            }
            if (!frame_key_to_slot_.emplace(
                    make_frame_key(frame.image_name, frame.camera_id), slot).second) {
                throw std::runtime_error(std::format(
                    "Duplicate stable frame key ('{}', camera {}) in PerFrameObservationBlur state",
                    frame.image_name, frame.camera_id));
            }
        }
    }

    std::expected<void, std::string> PerFrameObservationBlur::export_parameters_json(
        const std::filesystem::path& path,
        const int iteration) const {
        try {
            require_finalized();
            if (path.empty()) {
                return std::unexpected(
                    "Cannot export per-frame blur parameters: output path is empty");
            }
            validate_state_tensor(parameters_, frames_.size(), "parameters");
            validate_state_values(parameters_, config_, "parameters");
            const std::vector<float> values = parameters_.cpu().to_vector();
            if (values.size() != frames_.size() * PARAMS_PER_FRAME) {
                return std::unexpected(
                    "Cannot export per-frame blur parameters: invalid parameter count");
            }

            nlohmann::json root;
            root["format_version"] = JSON_EXPORT_VERSION;
            root["iteration"] = std::max(iteration, 0);
            root["frame_count"] = frames_.size();
            root["motion_enabled"] = config_.motion_enabled;
            root["defocus_enabled"] = config_.defocus_enabled;
            root["motion_rotation_variance_unit"] = "radian^2";
            root["motion_translation_variance_unit"] = "scene_unit^2";
            root["motion_max_rotation_std_deg"] =
                std::sqrt(config_.max_rot_variance) * 180.0 / std::numbers::pi;
            root["motion_max_translation_std_scene_unit"] =
                std::sqrt(config_.max_trans_variance);
            root["defocus_max_radius_px"] =
                std::sqrt(config_.max_defocus_radius_sq);
            root["focus_inverse_depth_range"] = nlohmann::json::array({
                0.0, config_.focus_inverse_depth_abs_max});
            root["motion_optimizer_step"] = motion_step_;
            root["defocus_optimizer_step"] = defocus_step_;
            root["defocus_model"] = "s_px2 = min(beta * (rho - 1/depth)^2, max_radius_px^2)";
            root["frames"] = nlohmann::json::array();

            for (size_t slot = 0; slot < frames_.size(); ++slot) {
                const size_t offset = slot * PARAMS_PER_FRAME;
                nlohmann::json entry;
                entry["uid"] = frames_[slot].uid;
                entry["image_name"] = frames_[slot].image_name;
                entry["camera_id"] = frames_[slot].camera_id;
                entry["motion_rotation_variance"] = nlohmann::json::array({
                    values[offset], values[offset + 1], values[offset + 2]});
                entry["motion_rotation_std_deg"] = nlohmann::json::array({
                    std::sqrt(std::max(values[offset], 0.0f)) * 180.0 / std::numbers::pi,
                    std::sqrt(std::max(values[offset + 1], 0.0f)) * 180.0 / std::numbers::pi,
                    std::sqrt(std::max(values[offset + 2], 0.0f)) * 180.0 / std::numbers::pi});
                entry["motion_translation_variance"] = nlohmann::json::array({
                    values[offset + 3], values[offset + 4], values[offset + 5]});
                entry["motion_translation_std"] = nlohmann::json::array({
                    std::sqrt(std::max(values[offset + 3], 0.0f)),
                    std::sqrt(std::max(values[offset + 4], 0.0f)),
                    std::sqrt(std::max(values[offset + 5], 0.0f))});
                entry["defocus_beta"] = values[offset + 6];
                entry["focus_inverse_depth_rho"] = values[offset + 7];
                root["frames"].push_back(std::move(entry));
            }

            const auto parent = path.parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create per-frame blur export directory '{}': {}",
                        lfs::core::path_to_utf8(parent), ec.message()));
                }
            }
            auto temp_path = path;
            temp_path += ".tmp";
            std::ofstream out;
            if (!lfs::core::open_file_for_write(
                    temp_path, std::ios::binary | std::ios::trunc, out)) {
                return std::unexpected(std::format(
                    "Failed to open per-frame blur export '{}' for writing",
                    lfs::core::path_to_utf8(temp_path)));
            }
            out << root.dump(2) << '\n';
            out.flush();
            if (!out) {
                out.close();
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return std::unexpected(std::format(
                    "Failed to write per-frame blur export '{}'",
                    lfs::core::path_to_utf8(temp_path)));
            }
            out.close();
            if (!out) {
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return std::unexpected(std::format(
                    "Failed to finalize per-frame blur export '{}'",
                    lfs::core::path_to_utf8(temp_path)));
            }
            if (auto result = replace_export_file(path, temp_path); !result) {
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return result;
            }
            return {};
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Failed to export per-frame blur parameters: {}", error.what()));
        }
    }

    size_t PerFrameObservationBlur::serialized_size_bytes() const {
        require_finalized();
        size_t result = 0;
        result += sizeof(uint32_t) * 3; // magic, version, frame count
        result += sizeof(uint8_t) * 2;  // enabled flags
        result += sizeof(double) * 15;  // config floating-point fields
        result += sizeof(int32_t) * 3;  // starts + total iterations
        result += sizeof(int64_t) * 2;  // optimizer steps
        for (const auto& frame : frames_) {
            result += sizeof(int32_t) * 2;
            result += sizeof(uint32_t);
            result += frame.image_name.size();
        }
        result += serialized_tensor_size(parameters_);
        result += serialized_tensor_size(exp_avg_);
        result += serialized_tensor_size(exp_avg_sq_);
        return result;
    }

    void PerFrameObservationBlur::serialize(std::ostream& os) const {
        require_finalized();
        validate_state_tensor(parameters_, frames_.size(), "parameters");
        validate_state_tensor(exp_avg_, frames_.size(), "exp_avg");
        validate_state_tensor(exp_avg_sq_, frames_.size(), "exp_avg_sq");
        validate_state_values(parameters_, config_, "parameters");
        validate_state_values(exp_avg_, config_, "exp_avg");
        validate_state_values(exp_avg_sq_, config_, "exp_avg_sq");
        write_pod(os, SERIALIZATION_MAGIC);
        write_pod(os, SERIALIZATION_VERSION);
        write_pod(os, static_cast<uint32_t>(frames_.size()));
        write_pod(os, static_cast<uint8_t>(config_.motion_enabled));
        write_pod(os, static_cast<uint8_t>(config_.defocus_enabled));
        write_pod(os, config_.motion_rot_lr);
        write_pod(os, config_.motion_trans_lr);
        write_pod(os, config_.defocus_scale_lr);
        write_pod(os, config_.defocus_focus_lr);
        write_pod(os, config_.motion_rot_reg_weight);
        write_pod(os, config_.motion_trans_reg_weight);
        write_pod(os, config_.defocus_reg_weight);
        write_pod(os, static_cast<int32_t>(config_.motion_start_iter));
        write_pod(os, static_cast<int32_t>(config_.defocus_start_iter));
        write_pod(os, config_.max_rot_variance);
        write_pod(os, config_.max_trans_variance);
        write_pod(os, config_.max_defocus_radius_sq);
        write_pod(os, config_.focus_inverse_depth_abs_max);
        write_pod(os, config_.initial_focus_inverse_depth);
        write_pod(os, config_.beta1);
        write_pod(os, config_.beta2);
        write_pod(os, config_.eps);
        write_pod(os, motion_step_);
        write_pod(os, defocus_step_);
        write_pod(os, static_cast<int32_t>(total_iterations_));
        for (const auto& frame : frames_) {
            write_pod(os, static_cast<int32_t>(frame.uid));
            write_pod(os, static_cast<int32_t>(frame.camera_id));
            write_pod(os, static_cast<uint32_t>(frame.image_name.size()));
            os.write(frame.image_name.data(),
                     static_cast<std::streamsize>(frame.image_name.size()));
            if (!os) {
                throw std::runtime_error(
                    "Failed to write PerFrameObservationBlur image name");
            }
        }
        os << parameters_ << exp_avg_ << exp_avg_sq_;
    }

    void PerFrameObservationBlur::deserialize(std::istream& is) {
        SerializedState state = read_serialized_state(is);
        if (frames_.empty()) {
            frames_ = state.frames;
            rebuild_indices();
        } else {
            if (frames_.size() != state.frames.size()) {
                throw std::runtime_error(std::format(
                    "PerFrameObservationBlur frame count mismatch: current {}, saved {}",
                    frames_.size(), state.frames.size()));
            }
            rebuild_indices();
        }

        std::unordered_map<std::string, size_t> saved_slot_by_key;
        saved_slot_by_key.reserve(state.frames.size());
        for (size_t index = 0; index < state.frames.size(); ++index) {
            const auto& frame = state.frames[index];
            if (!saved_slot_by_key.emplace(
                    make_frame_key(frame.image_name, frame.camera_id), index).second) {
                throw std::runtime_error(
                    "Duplicate saved PerFrameObservationBlur frame key");
            }
        }

        const auto remap_tensor = [&](const lfs::core::Tensor& saved) {
            const std::vector<float> saved_values = saved.to_vector();
            std::vector<float> remapped(saved_values.size(), 0.0f);
            for (size_t current_slot = 0; current_slot < frames_.size(); ++current_slot) {
                const auto& current = frames_[current_slot];
                const auto it = saved_slot_by_key.find(
                    make_frame_key(current.image_name, current.camera_id));
                if (it == saved_slot_by_key.end()) {
                    throw std::runtime_error(std::format(
                        "Saved PerFrameObservationBlur state is missing frame ('{}', camera {})",
                        current.image_name, current.camera_id));
                }
                std::copy_n(
                    saved_values.begin() + static_cast<std::ptrdiff_t>(
                                               it->second * PARAMS_PER_FRAME),
                    PARAMS_PER_FRAME,
                    remapped.begin() + static_cast<std::ptrdiff_t>(
                                           current_slot * PARAMS_PER_FRAME));
            }
            return lfs::core::Tensor::from_vector(
                remapped,
                lfs::core::TensorShape({frames_.size(), PARAMS_PER_FRAME}),
                lfs::core::Device::CUDA);
        };

        parameters_ = remap_tensor(state.parameters);
        exp_avg_ = remap_tensor(state.exp_avg);
        exp_avg_sq_ = remap_tensor(state.exp_avg_sq);
        gradients_ = lfs::core::Tensor::zeros(
            parameters_.shape(), lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);
        regularization_loss_ = lfs::core::Tensor::zeros(
            {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        config_ = state.config;
        total_iterations_ = state.total_iterations;
        motion_step_ = state.motion_step;
        defocus_step_ = state.defocus_step;
        finalized_ = true;
    }

    std::expected<void, std::string> PerFrameObservationBlur::skip_serialized(
        std::istream& is) {
        try {
            static_cast<void>(read_serialized_state(is));
            return {};
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Failed to skip PerFrameObservationBlur state: {}", error.what()));
        }
    }

} // namespace lfs::training
