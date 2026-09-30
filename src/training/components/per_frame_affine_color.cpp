/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "per_frame_affine_color.hpp"

#include "core/path_utils.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "lfs/kernels/per_frame_affine_color.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace lfs::training {

    namespace {
        constexpr uint32_t SERIALIZATION_MAGIC = 0x4C464143; // "LFAC"
        constexpr uint32_t SERIALIZATION_VERSION = 1;
        constexpr uint32_t JSON_EXPORT_VERSION = 1;
        constexpr size_t PARAMS_PER_FRAME = 12;
        constexpr uint32_t MAX_IMAGE_NAME_BYTES = 1024 * 1024;
        constexpr size_t MAX_FRAME_COUNT =
            static_cast<size_t>(std::numeric_limits<int>::max()) / PARAMS_PER_FRAME;

        struct SerializedState {
            PerFrameAffineColor::Config config;
            int total_iterations = 0;
            int64_t step = 0;
            double current_lr = 0.0;
            double initial_lr = 0.0;
            std::vector<PerFrameAffineColor::FrameIdentity> frames;
            lfs::core::Tensor parameters;
            lfs::core::Tensor exp_avg;
            lfs::core::Tensor exp_avg_sq;
        };

        template <typename T>
        void write_pod(std::ostream& os, const T& value) {
            os.write(reinterpret_cast<const char*>(&value), sizeof(T));
            if (!os) {
                throw std::runtime_error("Failed to write PerFrameAffineColor state");
            }
        }

        template <typename T>
        T read_pod(std::istream& is, const char* const field) {
            T value{};
            is.read(reinterpret_cast<char*>(&value), sizeof(T));
            if (!is) {
                throw std::runtime_error(std::format(
                    "Truncated PerFrameAffineColor state while reading {}", field));
            }
            return value;
        }

        void validate_config(
            const PerFrameAffineColor::Config& config,
            const int total_iterations) {
            if (total_iterations <= 0) {
                throw std::invalid_argument("PerFrameAffineColor total_iterations must be positive");
            }
            if (config.lr < 0.0 || !std::isfinite(config.lr)) {
                throw std::invalid_argument("PerFrameAffineColor lr must be non-negative and finite");
            }
            if (!(config.beta1 >= 0.0 && config.beta1 < 1.0) || !std::isfinite(config.beta1) ||
                !(config.beta2 >= 0.0 && config.beta2 < 1.0) || !std::isfinite(config.beta2)) {
                throw std::invalid_argument("PerFrameAffineColor beta values must be finite and in [0, 1)");
            }
            if (!(config.eps > 0.0) || !std::isfinite(config.eps)) {
                throw std::invalid_argument("PerFrameAffineColor eps must be positive and finite");
            }
            if (config.warmup_steps < 0) {
                throw std::invalid_argument("PerFrameAffineColor warmup_steps must be non-negative");
            }
            if (!(config.warmup_start_factor > 0.0) || !std::isfinite(config.warmup_start_factor) ||
                !(config.final_lr_factor > 0.0) || !std::isfinite(config.final_lr_factor)) {
                throw std::invalid_argument("PerFrameAffineColor LR factors must be positive and finite");
            }
        }

        void validate_regularization_weights(
            const float identity_weight,
            const float gauge_weight,
            const float bias_weight) {
            if (identity_weight < 0.0f || !std::isfinite(identity_weight) ||
                gauge_weight < 0.0f || !std::isfinite(gauge_weight) ||
                bias_weight < 0.0f || !std::isfinite(bias_weight)) {
                throw std::invalid_argument(
                    "PerFrameAffineColor regularization weights must be non-negative and finite");
            }
        }

        void validate_image_tensor(const lfs::core::Tensor& tensor, const char* const name) {
            if (!tensor.is_valid() || tensor.is_empty()) {
                throw std::invalid_argument(std::format("PerFrameAffineColor {} is empty", name));
            }
            if (tensor.device() != lfs::core::Device::CUDA ||
                tensor.dtype() != lfs::core::DataType::Float32) {
                throw std::invalid_argument(std::format(
                    "PerFrameAffineColor {} must be a CUDA Float32 tensor", name));
            }
            if (tensor.ndim() != 3 || tensor.shape()[0] != 3 ||
                tensor.shape()[1] == 0 || tensor.shape()[2] == 0) {
                throw std::invalid_argument(std::format(
                    "PerFrameAffineColor {} must have CHW shape [3,H,W]", name));
            }
            if (!tensor.is_contiguous()) {
                throw std::invalid_argument(std::format(
                    "PerFrameAffineColor {} must be contiguous", name));
            }
            if (tensor.shape()[1] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                tensor.shape()[2] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                tensor.shape()[1] * tensor.shape()[2] >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument(std::format(
                    "PerFrameAffineColor {} dimensions exceed CUDA kernel limits", name));
            }
        }

        void validate_state_tensor(
            const lfs::core::Tensor& tensor,
            const size_t num_frames,
            const char* const name) {
            if (!tensor.is_valid() || tensor.dtype() != lfs::core::DataType::Float32 ||
                tensor.ndim() != 2 || tensor.shape()[0] != num_frames ||
                tensor.shape()[1] != PARAMS_PER_FRAME) {
                throw std::runtime_error(std::format(
                    "Invalid PerFrameAffineColor {} tensor; expected [{},12] Float32",
                    name, num_frames));
            }
        }

        SerializedState read_serialized_state(std::istream& is) {
            const uint32_t magic = read_pod<uint32_t>(is, "magic");
            const uint32_t version = read_pod<uint32_t>(is, "version");
            if (magic != SERIALIZATION_MAGIC) {
                throw std::runtime_error("Invalid PerFrameAffineColor state: wrong magic");
            }
            if (version != SERIALIZATION_VERSION) {
                throw std::runtime_error("Unsupported PerFrameAffineColor state version");
            }

            const uint32_t frame_count = read_pod<uint32_t>(is, "frame count");
            if (frame_count == 0 || frame_count > MAX_FRAME_COUNT) {
                throw std::runtime_error("Invalid PerFrameAffineColor frame count");
            }

            SerializedState state;
            state.config.lr = read_pod<double>(is, "lr");
            state.config.beta1 = read_pod<double>(is, "beta1");
            state.config.beta2 = read_pod<double>(is, "beta2");
            state.config.eps = read_pod<double>(is, "eps");
            state.config.warmup_steps = read_pod<int32_t>(is, "warmup steps");
            state.config.warmup_start_factor = read_pod<double>(is, "warmup start factor");
            state.config.final_lr_factor = read_pod<double>(is, "final lr factor");
            state.step = read_pod<int64_t>(is, "step");
            state.current_lr = read_pod<double>(is, "current lr");
            state.initial_lr = read_pod<double>(is, "initial lr");
            state.total_iterations = read_pod<int32_t>(is, "total iterations");
            validate_config(state.config, state.total_iterations);
            if (state.step < 0 || state.current_lr < 0.0 || !std::isfinite(state.current_lr) ||
                state.initial_lr < 0.0 || !std::isfinite(state.initial_lr)) {
                throw std::runtime_error("Invalid PerFrameAffineColor optimizer state");
            }

            state.frames.reserve(frame_count);
            for (uint32_t index = 0; index < frame_count; ++index) {
                PerFrameAffineColor::FrameIdentity frame;
                frame.uid = read_pod<int32_t>(is, "frame uid");
                frame.camera_id = read_pod<int32_t>(is, "frame camera id");
                const uint32_t name_size = read_pod<uint32_t>(is, "image name size");
                if (name_size > MAX_IMAGE_NAME_BYTES) {
                    throw std::runtime_error("PerFrameAffineColor image name exceeds size limit");
                }
                frame.image_name.resize(name_size);
                if (name_size > 0) {
                    is.read(frame.image_name.data(), static_cast<std::streamsize>(name_size));
                    if (!is) {
                        throw std::runtime_error(
                            "Truncated PerFrameAffineColor state while reading image name");
                    }
                }
                state.frames.push_back(std::move(frame));
            }

            is >> state.parameters >> state.exp_avg >> state.exp_avg_sq;
            validate_state_tensor(state.parameters, state.frames.size(), "parameters");
            validate_state_tensor(state.exp_avg, state.frames.size(), "exp_avg");
            validate_state_tensor(state.exp_avg_sq, state.frames.size(), "exp_avg_sq");
            return state;
        }

        size_t serialized_tensor_size(const lfs::core::Tensor& tensor) {
            return sizeof(lfs::core::TensorFileHeader) +
                   tensor.ndim() * sizeof(uint64_t) + tensor.bytes();
        }

        std::expected<void, std::string> replace_export_file(
            const std::filesystem::path& path,
            const std::filesystem::path& temp_path) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            if (ec) {
                return std::unexpected(std::format(
                    "Failed to remove existing per-frame affine color export '{}': {}",
                    lfs::core::path_to_utf8(path), ec.message()));
            }

            std::filesystem::rename(temp_path, path, ec);
            if (ec) {
                return std::unexpected(std::format(
                    "Failed to replace per-frame affine color export '{}': {}",
                    lfs::core::path_to_utf8(path), ec.message()));
            }
            return {};
        }
    } // namespace

    PerFrameAffineColor::PerFrameAffineColor(const int total_iterations, Config config)
        : config_(config),
          total_iterations_(total_iterations),
          current_lr_(config.lr),
          initial_lr_(config.lr) {
        validate_config(config_, total_iterations_);
    }

    std::string PerFrameAffineColor::make_frame_key(
        const std::string_view image_name,
        const int camera_id) {
        return std::format("{}\n{}", camera_id, image_name);
    }

    void PerFrameAffineColor::register_frame(
        const int uid,
        std::string image_name,
        const int camera_id) {
        if (finalized_) {
            throw std::logic_error("Cannot register PerFrameAffineColor frames after finalize()");
        }
        if (uid_to_slot_.contains(uid)) {
            throw std::invalid_argument(std::format(
                "Duplicate PerFrameAffineColor frame UID {}", uid));
        }
        if (image_name.size() > MAX_IMAGE_NAME_BYTES) {
            throw std::invalid_argument("PerFrameAffineColor image name exceeds size limit");
        }

        const std::string key = make_frame_key(image_name, camera_id);
        if (frame_key_to_slot_.contains(key)) {
            throw std::invalid_argument(std::format(
                "Duplicate PerFrameAffineColor frame key ('{}', camera {})",
                image_name, camera_id));
        }
        if (frames_.size() >= MAX_FRAME_COUNT) {
            throw std::overflow_error("Too many PerFrameAffineColor frames");
        }

        const int slot = static_cast<int>(frames_.size());
        frames_.push_back(FrameIdentity{
            .uid = uid,
            .image_name = std::move(image_name),
            .camera_id = camera_id});
        uid_to_slot_.emplace(uid, slot);
        frame_key_to_slot_.emplace(key, slot);
    }

    void PerFrameAffineColor::finalize() {
        if (finalized_) {
            throw std::logic_error("PerFrameAffineColor is already finalized");
        }
        if (frames_.empty()) {
            throw std::logic_error("Cannot finalize PerFrameAffineColor without registered frames");
        }

        const lfs::core::TensorShape parameter_shape = {
            frames_.size(), PARAMS_PER_FRAME};
        parameters_ = lfs::core::Tensor::zeros(
            parameter_shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        exp_avg_ = lfs::core::Tensor::zeros(
            parameter_shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        exp_avg_sq_ = lfs::core::Tensor::zeros(
            parameter_shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        accumulated_grads_ = lfs::core::Tensor::zeros(
            parameter_shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        parameter_means_ = lfs::core::Tensor::zeros(
            {PARAMS_PER_FRAME}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        regularization_loss_ = lfs::core::Tensor::zeros(
            {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        finalized_ = true;
    }

    bool PerFrameAffineColor::is_known_frame(const int uid) const noexcept {
        return uid_to_slot_.contains(uid);
    }

    bool PerFrameAffineColor::has_frame_key(
        const std::string_view image_name,
        const int camera_id) const {
        return frame_key_to_slot_.contains(make_frame_key(image_name, camera_id));
    }

    int PerFrameAffineColor::slot_for_frame_key(
        const std::string_view image_name,
        const int camera_id) const {
        const auto it = frame_key_to_slot_.find(make_frame_key(image_name, camera_id));
        if (it == frame_key_to_slot_.end()) {
            throw std::out_of_range(std::format(
                "Unknown PerFrameAffineColor frame key ('{}', camera {})",
                image_name, camera_id));
        }
        return it->second;
    }

    int PerFrameAffineColor::slot_for_uid(const int uid) const {
        const auto it = uid_to_slot_.find(uid);
        if (it == uid_to_slot_.end()) {
            throw std::out_of_range(std::format(
                "Unknown PerFrameAffineColor frame UID {}", uid));
        }
        return it->second;
    }

    void PerFrameAffineColor::require_finalized() const {
        if (!finalized_) {
            throw std::logic_error("PerFrameAffineColor must be finalized before use");
        }
    }

    lfs::core::Tensor PerFrameAffineColor::apply(
        const lfs::core::Tensor& rgb,
        const int uid) const {
        return apply_slot(rgb, slot_for_uid(uid));
    }

    lfs::core::Tensor PerFrameAffineColor::apply_by_key(
        const lfs::core::Tensor& rgb,
        const std::string_view image_name,
        const int camera_id) const {
        return apply_slot(rgb, slot_for_frame_key(image_name, camera_id));
    }

    lfs::core::Tensor PerFrameAffineColor::apply_slot(
        const lfs::core::Tensor& rgb,
        const int slot) const {
        require_finalized();
        validate_image_tensor(rgb, "rgb");

        auto output = lfs::core::Tensor::empty(
            rgb.shape(), lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        const float* const frame_parameters =
            parameters_.ptr<float>() + static_cast<size_t>(slot) * PARAMS_PER_FRAME;
        kernels::launch_per_frame_affine_color_forward(
            frame_parameters,
            rgb.ptr<float>(),
            output.ptr<float>(),
            static_cast<int>(rgb.shape()[1]),
            static_cast<int>(rgb.shape()[2]),
            nullptr);
        return output;
    }

    lfs::core::Tensor PerFrameAffineColor::backward(
        const lfs::core::Tensor& rgb,
        const lfs::core::Tensor& grad_output,
        const int uid,
        const bool accumulate_parameters) {
        return backward_slot(rgb, grad_output, slot_for_uid(uid), accumulate_parameters);
    }

    lfs::core::Tensor PerFrameAffineColor::backward_by_key(
        const lfs::core::Tensor& rgb,
        const lfs::core::Tensor& grad_output,
        const std::string_view image_name,
        const int camera_id,
        const bool accumulate_parameters) {
        return backward_slot(
            rgb, grad_output, slot_for_frame_key(image_name, camera_id),
            accumulate_parameters);
    }

    lfs::core::Tensor PerFrameAffineColor::backward_slot(
        const lfs::core::Tensor& rgb,
        const lfs::core::Tensor& grad_output,
        const int slot,
        const bool accumulate_parameters) {
        require_finalized();
        validate_image_tensor(rgb, "rgb");
        validate_image_tensor(grad_output, "grad_output");
        if (rgb.shape() != grad_output.shape()) {
            throw std::invalid_argument(
                "PerFrameAffineColor rgb and grad_output shapes must match");
        }

        auto grad_rgb = lfs::core::Tensor::empty(
            rgb.shape(), lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        const size_t parameter_offset = static_cast<size_t>(slot) * PARAMS_PER_FRAME;
        kernels::launch_per_frame_affine_color_backward(
            parameters_.ptr<float>() + parameter_offset,
            rgb.ptr<float>(),
            grad_output.ptr<float>(),
            accumulated_grads_.ptr<float>() + parameter_offset,
            grad_rgb.ptr<float>(),
            static_cast<int>(rgb.shape()[1]),
            static_cast<int>(rgb.shape()[2]),
            accumulate_parameters,
            nullptr);
        return grad_rgb;
    }

    lfs::core::Tensor PerFrameAffineColor::regularization_loss_gpu(
        const float identity_weight,
        const float gauge_weight,
        const float bias_weight) {
        require_finalized();
        validate_regularization_weights(identity_weight, gauge_weight, bias_weight);

        cudaMemsetAsync(
            regularization_loss_.ptr<float>(), 0,
            regularization_loss_.bytes(), nullptr);
        kernels::launch_per_frame_affine_color_regularization_forward(
            parameters_.ptr<float>(),
            parameter_means_.ptr<float>(),
            regularization_loss_.ptr<float>(),
            num_frames(), identity_weight, gauge_weight, bias_weight,
            nullptr);
        return regularization_loss_;
    }

    void PerFrameAffineColor::regularization_backward(
        const float identity_weight,
        const float gauge_weight,
        const float bias_weight) {
        require_finalized();
        validate_regularization_weights(identity_weight, gauge_weight, bias_weight);

        kernels::launch_per_frame_affine_color_parameter_means(
            parameters_.ptr<float>(), parameter_means_.ptr<float>(),
            num_frames(), nullptr);
        kernels::launch_per_frame_affine_color_regularization_backward(
            parameters_.ptr<float>(), parameter_means_.ptr<float>(),
            accumulated_grads_.ptr<float>(), num_frames(),
            identity_weight, gauge_weight, bias_weight, nullptr);
    }

    void PerFrameAffineColor::optimizer_step() {
        require_finalized();
        const double bias_correction1 = 1.0 - std::pow(config_.beta1, step_ + 1);
        const double bias_correction2 = 1.0 - std::pow(config_.beta2, step_ + 1);
        const float bias_corr1_rcp = static_cast<float>(1.0 / bias_correction1);
        const float bias_corr2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bias_correction2));

        kernels::launch_per_frame_affine_color_adam_update(
            parameters_.ptr<float>(), exp_avg_.ptr<float>(), exp_avg_sq_.ptr<float>(),
            accumulated_grads_.ptr<float>(), static_cast<int>(parameters_.numel()),
            static_cast<float>(current_lr_),
            static_cast<float>(config_.beta1),
            static_cast<float>(config_.beta2),
            bias_corr1_rcp, bias_corr2_sqrt_rcp,
            static_cast<float>(config_.eps), nullptr);
    }

    void PerFrameAffineColor::zero_grad() {
        require_finalized();
        cudaMemsetAsync(
            accumulated_grads_.ptr<float>(), 0,
            accumulated_grads_.bytes(), nullptr);
    }

    void PerFrameAffineColor::scheduler_step() {
        require_finalized();
        ++step_;

        const int warmup_steps = std::min(config_.warmup_steps, total_iterations_);
        if (warmup_steps > 0 && step_ <= warmup_steps) {
            const double progress = static_cast<double>(step_) /
                                    static_cast<double>(warmup_steps);
            const double factor = config_.warmup_start_factor +
                                  (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * factor;
            return;
        }

        const int decay_steps = total_iterations_ - warmup_steps;
        if (decay_steps <= 0) {
            current_lr_ = initial_lr_;
            return;
        }

        const int64_t elapsed = std::clamp<int64_t>(
            step_ - warmup_steps, 0, decay_steps);
        const double gamma = std::pow(
            config_.final_lr_factor, 1.0 / static_cast<double>(decay_steps));
        current_lr_ = initial_lr_ * std::pow(gamma, static_cast<double>(elapsed));
    }

    void PerFrameAffineColor::rebuild_indices() {
        uid_to_slot_.clear();
        frame_key_to_slot_.clear();
        uid_to_slot_.reserve(frames_.size());
        frame_key_to_slot_.reserve(frames_.size());

        for (size_t index = 0; index < frames_.size(); ++index) {
            const auto& frame = frames_[index];
            const int slot = static_cast<int>(index);
            if (!uid_to_slot_.emplace(frame.uid, slot).second) {
                throw std::runtime_error(std::format(
                    "Duplicate UID {} in PerFrameAffineColor state", frame.uid));
            }
            if (!frame_key_to_slot_.emplace(
                    make_frame_key(frame.image_name, frame.camera_id), slot).second) {
                throw std::runtime_error(std::format(
                    "Duplicate stable frame key ('{}', camera {}) in PerFrameAffineColor state",
                    frame.image_name, frame.camera_id));
            }
        }
    }

    std::expected<void, std::string> PerFrameAffineColor::export_parameters_json(
        const std::filesystem::path& path,
        const int iteration) const {
        try {
            require_finalized();
            if (path.empty()) {
                return std::unexpected(
                    "Cannot export per-frame affine color parameters: output path is empty");
            }

            const std::vector<float> values = parameters_.cpu().to_vector();
            if (values.size() != frames_.size() * PARAMS_PER_FRAME) {
                return std::unexpected(std::format(
                    "Cannot export per-frame affine color parameters: expected {} values, got {}",
                    frames_.size() * PARAMS_PER_FRAME, values.size()));
            }
            for (size_t index = 0; index < values.size(); ++index) {
                if (!std::isfinite(values[index])) {
                    return std::unexpected(std::format(
                        "Cannot export per-frame affine color parameters: value {} is not finite",
                        index));
                }
            }

            nlohmann::json root;
            root["format_version"] = JSON_EXPORT_VERSION;
            root["iteration"] = std::max(iteration, 0);
            root["frame_count"] = frames_.size();
            root["transform"] = "rgb_out = matrix * rgb_in + bias";
            root["matrix_parameterization"] = "matrix = identity + delta_matrix";
            root["frames"] = nlohmann::json::array();

            for (size_t slot = 0; slot < frames_.size(); ++slot) {
                const size_t offset = slot * PARAMS_PER_FRAME;
                nlohmann::json delta_matrix = nlohmann::json::array();
                nlohmann::json matrix = nlohmann::json::array();
                for (size_t row = 0; row < 3; ++row) {
                    nlohmann::json delta_row = nlohmann::json::array();
                    nlohmann::json matrix_row = nlohmann::json::array();
                    for (size_t column = 0; column < 3; ++column) {
                        const float delta = values[offset + row * 3 + column];
                        delta_row.push_back(delta);
                        matrix_row.push_back(delta + (row == column ? 1.0f : 0.0f));
                    }
                    delta_matrix.push_back(std::move(delta_row));
                    matrix.push_back(std::move(matrix_row));
                }

                const auto& frame = frames_[slot];
                nlohmann::json entry;
                entry["uid"] = frame.uid;
                entry["image_name"] = frame.image_name;
                entry["camera_id"] = frame.camera_id;
                entry["matrix"] = std::move(matrix);
                entry["delta_matrix"] = std::move(delta_matrix);
                entry["bias"] = nlohmann::json::array({
                    values[offset + 9], values[offset + 10], values[offset + 11]});
                root["frames"].push_back(std::move(entry));
            }

            const auto parent = path.parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create per-frame affine color export directory '{}': {}",
                        lfs::core::path_to_utf8(parent), ec.message()));
                }
            }

            auto temp_path = path;
            temp_path += ".tmp";
            std::ofstream out;
            if (!lfs::core::open_file_for_write(
                    temp_path, std::ios::binary | std::ios::trunc, out)) {
                return std::unexpected(std::format(
                    "Failed to open per-frame affine color export '{}' for writing",
                    lfs::core::path_to_utf8(temp_path)));
            }

            out << root.dump(2) << '\n';
            out.flush();
            if (!out) {
                out.close();
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return std::unexpected(std::format(
                    "Failed to write per-frame affine color export '{}'",
                    lfs::core::path_to_utf8(temp_path)));
            }
            out.close();
            if (!out) {
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return std::unexpected(std::format(
                    "Failed to finalize per-frame affine color export '{}'",
                    lfs::core::path_to_utf8(temp_path)));
            }

            if (auto result = replace_export_file(path, temp_path); !result) {
                std::error_code cleanup_ec;
                std::filesystem::remove(temp_path, cleanup_ec);
                return result;
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "Failed to export per-frame affine color parameters: {}", e.what()));
        }
    }

    size_t PerFrameAffineColor::serialized_size_bytes() const {
        require_finalized();

        size_t result = 0;
        result += sizeof(uint32_t) * 3; // magic, version, frame count
        result += sizeof(double) * 6;   // Config's double fields
        result += sizeof(int32_t);      // warmup_steps
        result += sizeof(int64_t);      // step
        result += sizeof(double) * 2;   // current_lr, initial_lr
        result += sizeof(int32_t);      // total_iterations
        for (const auto& frame : frames_) {
            result += sizeof(int32_t) * 2; // uid, camera_id
            result += sizeof(uint32_t);    // image_name length
            result += frame.image_name.size();
        }
        result += serialized_tensor_size(parameters_);
        result += serialized_tensor_size(exp_avg_);
        result += serialized_tensor_size(exp_avg_sq_);
        return result;
    }

    void PerFrameAffineColor::serialize(std::ostream& os) const {
        require_finalized();
        if (frames_.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("Too many PerFrameAffineColor frames to serialize");
        }

        write_pod(os, SERIALIZATION_MAGIC);
        write_pod(os, SERIALIZATION_VERSION);
        write_pod(os, static_cast<uint32_t>(frames_.size()));
        write_pod(os, config_.lr);
        write_pod(os, config_.beta1);
        write_pod(os, config_.beta2);
        write_pod(os, config_.eps);
        write_pod(os, static_cast<int32_t>(config_.warmup_steps));
        write_pod(os, config_.warmup_start_factor);
        write_pod(os, config_.final_lr_factor);
        write_pod(os, step_);
        write_pod(os, current_lr_);
        write_pod(os, initial_lr_);
        write_pod(os, static_cast<int32_t>(total_iterations_));

        for (const auto& frame : frames_) {
            if (frame.image_name.size() > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("PerFrameAffineColor image name is too long");
            }
            write_pod(os, static_cast<int32_t>(frame.uid));
            write_pod(os, static_cast<int32_t>(frame.camera_id));
            write_pod(os, static_cast<uint32_t>(frame.image_name.size()));
            os.write(frame.image_name.data(),
                     static_cast<std::streamsize>(frame.image_name.size()));
            if (!os) {
                throw std::runtime_error("Failed to write PerFrameAffineColor image name");
            }
        }

        os << parameters_ << exp_avg_ << exp_avg_sq_;
    }

    void PerFrameAffineColor::deserialize(std::istream& is) {
        SerializedState state = read_serialized_state(is);

        const bool preserve_current_mapping = !frames_.empty();
        if (!preserve_current_mapping) {
            frames_ = state.frames;
            rebuild_indices();
        } else {
            if (frames_.size() != state.frames.size()) {
                throw std::runtime_error(std::format(
                    "PerFrameAffineColor frame count mismatch: current {}, saved {}",
                    frames_.size(), state.frames.size()));
            }
            rebuild_indices();
        }

        std::unordered_map<std::string, size_t> saved_slot_by_key;
        saved_slot_by_key.reserve(state.frames.size());
        for (size_t index = 0; index < state.frames.size(); ++index) {
            const auto& frame = state.frames[index];
            const std::string key = make_frame_key(frame.image_name, frame.camera_id);
            if (!saved_slot_by_key.emplace(key, index).second) {
                throw std::runtime_error(std::format(
                    "Duplicate saved PerFrameAffineColor frame key ('{}', camera {})",
                    frame.image_name, frame.camera_id));
            }
        }

        const auto remap_tensor = [&](const lfs::core::Tensor& saved) {
            const std::vector<float> saved_values = saved.to_vector();
            std::vector<float> remapped(saved_values.size(), 0.0f);
            for (size_t current_slot = 0; current_slot < frames_.size(); ++current_slot) {
                const auto& current_frame = frames_[current_slot];
                const auto saved_it = saved_slot_by_key.find(
                    make_frame_key(current_frame.image_name, current_frame.camera_id));
                if (saved_it == saved_slot_by_key.end()) {
                    throw std::runtime_error(std::format(
                        "Saved PerFrameAffineColor state is missing frame ('{}', camera {})",
                        current_frame.image_name, current_frame.camera_id));
                }
                const size_t saved_offset = saved_it->second * PARAMS_PER_FRAME;
                const size_t current_offset = current_slot * PARAMS_PER_FRAME;
                std::copy_n(
                    saved_values.begin() + static_cast<std::ptrdiff_t>(saved_offset),
                    PARAMS_PER_FRAME,
                    remapped.begin() + static_cast<std::ptrdiff_t>(current_offset));
            }
            return lfs::core::Tensor::from_vector(
                remapped,
                lfs::core::TensorShape({frames_.size(), PARAMS_PER_FRAME}),
                lfs::core::Device::CUDA);
        };

        parameters_ = remap_tensor(state.parameters);
        exp_avg_ = remap_tensor(state.exp_avg);
        exp_avg_sq_ = remap_tensor(state.exp_avg_sq);
        accumulated_grads_ = lfs::core::Tensor::zeros(
            parameters_.shape(), lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        parameter_means_ = lfs::core::Tensor::zeros(
            {PARAMS_PER_FRAME}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        regularization_loss_ = lfs::core::Tensor::zeros(
            {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        config_ = state.config;
        total_iterations_ = state.total_iterations;
        step_ = state.step;
        current_lr_ = state.current_lr;
        initial_lr_ = state.initial_lr;
        finalized_ = true;
    }

    std::expected<void, std::string> PerFrameAffineColor::skip_serialized(
        std::istream& is) {
        try {
            static_cast<void>(read_serialized_state(is));
            return {};
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Failed to skip PerFrameAffineColor state: {}", error.what()));
        }
    }

} // namespace lfs::training
