/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_per_frame_affine_color_forward(
        const float* parameters,
        const float* rgb,
        float* output,
        int height,
        int width,
        cudaStream_t stream = nullptr);

    void launch_per_frame_affine_color_backward(
        const float* parameters,
        const float* rgb,
        const float* grad_output,
        float* grad_parameters,
        float* grad_rgb,
        int height,
        int width,
        bool accumulate_parameters,
        cudaStream_t stream = nullptr);

    void launch_per_frame_affine_color_regularization_forward(
        const float* parameters,
        float* parameter_means,
        float* loss,
        int num_frames,
        float identity_weight,
        float gauge_weight,
        float bias_weight,
        cudaStream_t stream = nullptr);

    void launch_per_frame_affine_color_parameter_means(
        const float* parameters,
        float* parameter_means,
        int num_frames,
        cudaStream_t stream = nullptr);

    void launch_per_frame_affine_color_regularization_backward(
        const float* parameters,
        const float* parameter_means,
        float* grad_parameters,
        int num_frames,
        float identity_weight,
        float gauge_weight,
        float bias_weight,
        cudaStream_t stream = nullptr);

    void launch_per_frame_affine_color_adam_update(
        float* parameters,
        float* exp_avg,
        float* exp_avg_sq,
        const float* gradients,
        int num_elements,
        float lr,
        float beta1,
        float beta2,
        float bias_corr1_rcp,
        float bias_corr2_sqrt_rcp,
        float eps,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
