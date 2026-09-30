/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    struct PerFrameAffineColorConfig {
        double lr = 2e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-15;
        int warmup_steps = 500;
        double warmup_start_factor = 0.01;
        double final_lr_factor = 0.01;
    };

    /// Per-frame global RGB affine appearance model.
    ///
    /// Each registered frame owns 12 zero-initialized parameters laid out as
    /// [delta_W(3x3 row-major), q(3)]. The applied transform is
    /// rgb_out = (I + delta_W) * rgb_in + q.
    class PerFrameAffineColor {
    public:
        using Config = PerFrameAffineColorConfig;

        struct FrameIdentity {
            int uid = -1;
            std::string image_name;
            int camera_id = -1;
        };

        explicit PerFrameAffineColor(int total_iterations, Config config = {});

        /// Register real frames before finalize(). Both uid and
        /// (image_name, camera_id) must be unique.
        void register_frame(int uid, std::string image_name, int camera_id);

        /// Allocate CUDA parameter and optimizer tensors. At least one frame is required.
        void finalize();

        [[nodiscard]] bool is_finalized() const noexcept { return finalized_; }
        [[nodiscard]] bool is_known_frame(int uid) const noexcept;
        [[nodiscard]] bool has_frame_key(std::string_view image_name, int camera_id) const;
        [[nodiscard]] int slot_for_frame_key(std::string_view image_name, int camera_id) const;
        [[nodiscard]] int num_frames() const noexcept { return static_cast<int>(frames_.size()); }
        [[nodiscard]] const std::vector<FrameIdentity>& frame_identities() const noexcept { return frames_; }

        lfs::core::Tensor apply(const lfs::core::Tensor& rgb, int uid) const;
        lfs::core::Tensor apply_by_key(
            const lfs::core::Tensor& rgb,
            std::string_view image_name,
            int camera_id) const;

        /// Return dL/d(rgb). Parameter gradients accumulate across calls/tiles when
        /// accumulate_parameters is true. Passing false is useful for pseudo views
        /// that inherit a source frame's appearance without updating that source.
        lfs::core::Tensor backward(
            const lfs::core::Tensor& rgb,
            const lfs::core::Tensor& grad_output,
            int uid,
            bool accumulate_parameters = true);
        lfs::core::Tensor backward_by_key(
            const lfs::core::Tensor& rgb,
            const lfs::core::Tensor& grad_output,
            std::string_view image_name,
            int camera_id,
            bool accumulate_parameters = false);

        /// Weighted identity and mean-gauge regularization.
        /// Bias entries use the additional bias_weight multiplier.
        lfs::core::Tensor regularization_loss_gpu(
            float identity_weight,
            float gauge_weight,
            float bias_weight = 1.0f);
        void regularization_backward(
            float identity_weight,
            float gauge_weight,
            float bias_weight = 1.0f);

        void optimizer_step();
        void zero_grad();
        void scheduler_step();

        [[nodiscard]] double get_lr() const noexcept { return current_lr_; }
        [[nodiscard]] int64_t get_step() const noexcept { return step_; }
        [[nodiscard]] const Config& get_config() const noexcept { return config_; }
        [[nodiscard]] const lfs::core::Tensor& parameters() const noexcept { return parameters_; }
        [[nodiscard]] lfs::core::Tensor& parameters() noexcept { return parameters_; }
        [[nodiscard]] const lfs::core::Tensor& gradients() const noexcept { return accumulated_grads_; }

        /// Export the learned real-frame transforms as a human-readable JSON file.
        /// Each entry contains the actual matrix (I + delta_W), the learned
        /// delta matrix, and bias q. The destination is replaced via a temp file.
        [[nodiscard]] std::expected<void, std::string> export_parameters_json(
            const std::filesystem::path& path,
            int iteration) const;

        [[nodiscard]] size_t serialized_size_bytes() const;
        void serialize(std::ostream& os) const;

        /// Restore state. If frames are already registered, saved rows are remapped
        /// to the current slots by the stable (image_name, camera_id) key, preserving
        /// current runtime UIDs.
        void deserialize(std::istream& is);

        /// Consume one serialized component without allocating CUDA state.
        [[nodiscard]] static std::expected<void, std::string> skip_serialized(std::istream& is);

    private:
        [[nodiscard]] static std::string make_frame_key(std::string_view image_name, int camera_id);
        [[nodiscard]] int slot_for_uid(int uid) const;
        [[nodiscard]] lfs::core::Tensor apply_slot(const lfs::core::Tensor& rgb, int slot) const;
        [[nodiscard]] lfs::core::Tensor backward_slot(
            const lfs::core::Tensor& rgb,
            const lfs::core::Tensor& grad_output,
            int slot,
            bool accumulate_parameters);
        void rebuild_indices();
        void require_finalized() const;

        Config config_;
        int total_iterations_ = 0;
        int64_t step_ = 0;
        double current_lr_ = 0.0;
        double initial_lr_ = 0.0;

        std::vector<FrameIdentity> frames_;
        std::unordered_map<int, int> uid_to_slot_;
        std::unordered_map<std::string, int> frame_key_to_slot_;
        bool finalized_ = false;

        // [num_frames, 12], storing [delta_W, q].
        lfs::core::Tensor parameters_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor accumulated_grads_;
        lfs::core::Tensor parameter_means_;
        lfs::core::Tensor regularization_loss_;
    };

} // namespace lfs::training
