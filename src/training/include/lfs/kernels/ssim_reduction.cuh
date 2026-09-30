/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused mean reduction for SSIM map with optional valid padding
     *
     * Computes mean(ssim_map) with optional cropping (5 pixels from each side)
     * in a single optimized pass without creating intermediate sliced tensors.
     *
     * @param ssim_map Input SSIM map [N, C, H, W]
     * @param temp_buffer Temporary buffer for partial sums (min(1024, total_pixels/256) elements)
     * @param result_buffer Device buffer for result (1 element)
     * @param N Batch size
     * @param C Number of channels
     * @param H Height
     * @param W Width
     * @param apply_valid_padding If true, crop 5 pixels from each side
     * @param stream CUDA stream
     * @return Mean SSIM value
     */
    float launch_fused_ssim_mean(
        const float* ssim_map,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
