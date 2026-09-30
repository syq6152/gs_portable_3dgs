/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime_api.h>

namespace lfs::training::kernels {

    void launch_per_frame_observation_blur_regularization_forward(
        const float* parameters,
        float* loss,
        int num_frames,
        bool motion_active,
        bool defocus_active,
        float rotation_weight,
        float translation_weight,
        float defocus_weight,
        cudaStream_t stream);

    void launch_per_frame_observation_blur_regularization_backward(
        float* gradients,
        int num_frames,
        bool motion_active,
        bool defocus_active,
        float rotation_weight,
        float translation_weight,
        float defocus_weight,
        cudaStream_t stream);

    void launch_per_frame_observation_blur_adam_update(
        float* parameters,
        float* exp_avg,
        float* exp_avg_sq,
        const float* gradients,
        int num_frames,
        bool motion_active,
        bool defocus_active,
        float rotation_lr,
        float translation_lr,
        float defocus_scale_lr,
        float defocus_focus_lr,
        float beta1,
        float beta2,
        float motion_bias_corr1_rcp,
        float motion_bias_corr2_sqrt_rcp,
        float defocus_bias_corr1_rcp,
        float defocus_bias_corr2_sqrt_rcp,
        float eps,
        float max_rotation_variance,
        float max_translation_variance,
        float focus_inverse_depth_abs_max,
        cudaStream_t stream);

} // namespace lfs::training::kernels
