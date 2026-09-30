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

    /// Per-real-frame nuisance parameters used by the Robust Gaussian
    /// Splatting motion/defocus observation model.
    ///
    /// Row layout:
    ///   [0:3] rotation variances (rad^2)
    ///   [3:6] translation variances (scene_unit^2)
    ///   [6]   beta = A^2 for defocus
    ///   [7]   focus-plane inverse depth rho
    class PerFrameObservationBlur {
    public:
        static constexpr int PARAMS_PER_FRAME = 8;

        struct Config {
            bool motion_enabled = false;
            bool defocus_enabled = false;

            double motion_rot_lr = 1e-4;
            double motion_trans_lr = 1e-4;
            double defocus_scale_lr = 1e-3;
            double defocus_focus_lr = 1e-4;

            double motion_rot_reg_weight = 1e-3;
            double motion_trans_reg_weight = 1e-3;
            double defocus_reg_weight = 1e-3;

            int motion_start_iter = 1000;
            int defocus_start_iter = 3000;

            double max_rot_variance = 0.0;
            double max_trans_variance = 0.0;
            double max_defocus_radius_sq = 0.0;
            double focus_inverse_depth_abs_max = 1e3;
            double initial_focus_inverse_depth = 0.0;

            double beta1 = 0.9;
            double beta2 = 0.999;
            double eps = 1e-15;
        };

        struct FrameIdentity {
            int uid = -1;
            std::string image_name;
            int camera_id = -1;
        };

        explicit PerFrameObservationBlur(int total_iterations, Config config);

        void register_frame(int uid, std::string image_name, int camera_id);
        void finalize();

        [[nodiscard]] bool is_finalized() const noexcept { return finalized_; }
        [[nodiscard]] bool is_known_frame(int uid) const noexcept;
        [[nodiscard]] bool has_frame_key(std::string_view image_name, int camera_id) const;
        [[nodiscard]] int num_frames() const noexcept { return static_cast<int>(frames_.size()); }
        [[nodiscard]] const std::vector<FrameIdentity>& frame_identities() const noexcept { return frames_; }

        [[nodiscard]] bool motion_active(int iteration) const noexcept;
        [[nodiscard]] bool defocus_active(int iteration) const noexcept;
        [[nodiscard]] bool any_active(int iteration) const noexcept;

        [[nodiscard]] const float* parameters_for_uid(int uid) const;
        [[nodiscard]] float* gradients_for_uid(int uid);

        [[nodiscard]] lfs::core::Tensor regularization_loss_gpu(int iteration);
        void regularization_backward(int iteration);
        void optimizer_step(int iteration);
        void zero_grad();

        [[nodiscard]] int64_t motion_step() const noexcept { return motion_step_; }
        [[nodiscard]] int64_t defocus_step() const noexcept { return defocus_step_; }
        [[nodiscard]] const Config& config() const noexcept { return config_; }
        [[nodiscard]] const lfs::core::Tensor& parameters() const noexcept { return parameters_; }
        [[nodiscard]] lfs::core::Tensor& parameters() noexcept { return parameters_; }
        [[nodiscard]] const lfs::core::Tensor& gradients() const noexcept { return gradients_; }

        [[nodiscard]] std::expected<void, std::string> export_parameters_json(
            const std::filesystem::path& path,
            int iteration) const;

        [[nodiscard]] size_t serialized_size_bytes() const;
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        [[nodiscard]] static std::expected<void, std::string> skip_serialized(std::istream& is);

    private:
        [[nodiscard]] static std::string make_frame_key(std::string_view image_name, int camera_id);
        [[nodiscard]] int slot_for_uid(int uid) const;
        void rebuild_indices();
        void require_finalized() const;

        Config config_;
        int total_iterations_ = 0;
        int64_t motion_step_ = 0;
        int64_t defocus_step_ = 0;

        std::vector<FrameIdentity> frames_;
        std::unordered_map<int, int> uid_to_slot_;
        std::unordered_map<std::string, int> frame_key_to_slot_;
        bool finalized_ = false;

        lfs::core::Tensor parameters_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor gradients_;
        lfs::core::Tensor regularization_loss_;
    };

} // namespace lfs::training

