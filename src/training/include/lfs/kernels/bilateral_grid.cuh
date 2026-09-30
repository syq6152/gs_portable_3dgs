/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // HWC layout kernels

    void launch_bilateral_grid_slice_forward(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        cudaStream_t stream = nullptr);

    // CHW layout kernels (zero-copy for rasterizer output)

    void launch_bilateral_grid_slice_forward_chw(
        const float* grid, const float* rgb, float* output,
        int L, int H, int W, int h, int w,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_slice_backward_chw(
        const float* grid, const float* rgb, const float* grad_output,
        float* grad_grid, float* grad_rgb,
        int L, int H, int W, int h, int w,
        cudaStream_t stream = nullptr);

    // TV loss kernels

    void launch_bilateral_grid_tv_forward(
        const float* grids, float* tv_loss, float* temp_buffer,
        int N, int L, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_tv_backward(
        const float* grids, float grad_output, float* grad_grids,
        int N, int L, int H, int W,
        cudaStream_t stream = nullptr);

    // Utility kernels

    void launch_bilateral_grid_init_identity(
        float* grids, int N, int L, int H, int W,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_accumulate_grad(
        float* dst, const float* src, int num_elements,
        cudaStream_t stream = nullptr);

    void launch_bilateral_grid_adam_update(
        float* grid, float* exp_avg, float* exp_avg_sq, const float* grad_grid,
        int num_elements, float lr, float beta1, float beta2,
        float bias_corr1_rcp, float bias_corr2_sqrt_rcp, float eps,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
