/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    class PoseRefiner {
    public:
        struct Diagnostics {
            int camera_uid = -1;
            int base_uid = -1;
            int base_index = -1;
            int adam_step = 0;
            bool camera_is_pseudo = false;
            bool finite = true;
            std::string camera_name;

            float lr_trans = 0.0f;
            float lr_rot = 0.0f;

            size_t real_count = 0;
            size_t pseudo_count = 0;
            size_t nonzero_grad_rows = 0;
            size_t clamp_trans_rows = 0;
            size_t clamp_rot_rows = 0;

            float reg_loss = 0.0f;
            float grad_trans_l2 = 0.0f;
            float grad_trans_mean = 0.0f;
            float grad_trans_max = 0.0f;
            float grad_rot_l2 = 0.0f;
            float grad_rot_mean = 0.0f;
            float grad_rot_max = 0.0f;
            float selected_grad_trans = 0.0f;
            float selected_grad_rot = 0.0f;

            float delta_trans_mean = 0.0f;
            float delta_trans_max = 0.0f;
            float delta_rot_mean_deg = 0.0f;
            float delta_rot_max_deg = 0.0f;
            float selected_delta_trans = 0.0f;
            float selected_delta_rot_deg = 0.0f;

            float step_trans_mean = 0.0f;
            float step_trans_max = 0.0f;
            float step_rot_mean_deg = 0.0f;
            float step_rot_max_deg = 0.0f;
            float selected_step_trans = 0.0f;
            float selected_step_rot_deg = 0.0f;

            std::vector<float> delta_before;
        };

        explicit PoseRefiner(const lfs::core::param::OptimizationParameters& params);

        std::expected<void, std::string> initialize(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);

        [[nodiscard]] bool enabled() const noexcept { return enabled_; }
        [[nodiscard]] bool is_active(int iteration) const noexcept;
        [[nodiscard]] size_t real_camera_count() const noexcept { return real_states_.size(); }
        [[nodiscard]] size_t pseudo_camera_count() const noexcept { return pseudo_states_.size(); }

        void zero_grad();
        lfs::core::Tensor& grad_w2c_scratch(cudaStream_t stream = nullptr);
        void accumulate_w2c_gradient(const lfs::core::Camera& camera, const lfs::core::Tensor& grad_w2c);
        lfs::core::Tensor regularization_loss();
        void step(int iteration);
        [[nodiscard]] Diagnostics capture_diagnostics_pre_step(
            const lfs::core::Camera& camera,
            float regularization_loss) const;
        void capture_diagnostics_post_step(Diagnostics& diagnostics) const;
        [[nodiscard]] std::string diagnostics_summary(const Diagnostics& diagnostics) const;
        [[nodiscard]] std::string inactive_reason(int iteration, bool in_controller_phase) const;

        void apply_to_camera(lfs::core::Camera& camera);
        void apply_to_cameras(const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);
        [[nodiscard]] std::expected<void, std::string> export_refined_poses(
            const std::filesystem::path& output_path,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            int iteration) const;
        [[nodiscard]] std::string metrics_summary() const;
        [[nodiscard]] size_t serialized_size_bytes() const noexcept;
        void serialize(std::ostream& os) const;
        [[nodiscard]] std::expected<void, std::string> deserialize(std::istream& is);
        static std::expected<void, std::string> skip_serialized(std::istream& is);

    private:
        struct RealState {
            int uid = -1;
            std::array<float, 16> base_c2w{};
        };

        struct PseudoState {
            int uid = -1;
            int base_uid = -1;
            int base_index = -1;
            int relative_index = -1;
            std::array<float, 16> relative_c2w{};
        };

        [[nodiscard]] std::array<float, 6> copy_delta_to_host(int camera_index) const;
        [[nodiscard]] std::array<float, 16> current_real_c2w(int camera_index) const;
        void apply_c2w_to_camera(lfs::core::Camera& camera, const std::array<float, 16>& c2w) const;
        [[nodiscard]] std::expected<void, std::string> export_colmap_images_txt(
            const std::filesystem::path& images_txt_path,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) const;
        void use_stream(cudaStream_t stream);
        void zero_grad_if_pending();

        bool enabled_ = false;
        lfs::core::param::OptimizationParameters params_;
        std::vector<RealState> real_states_;
        std::vector<PseudoState> pseudo_states_;
        std::unordered_map<int, int> real_index_by_uid_;
        std::unordered_map<int, int> pseudo_index_by_uid_;

        lfs::core::Tensor base_c2w_device_;
        lfs::core::Tensor pseudo_relative_c2w_device_;
        lfs::core::Tensor delta_;
        lfs::core::Tensor grad_delta_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor grad_w2c_scratch_;
        lfs::core::Tensor regularization_loss_;
        int adam_step_ = 0;
        cudaStream_t pose_stream_ = nullptr;
        bool zero_grad_pending_ = false;
        float last_lr_trans_ = 0.0f;
        float last_lr_rot_ = 0.0f;
    };

} // namespace lfs::training
