/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * Compute normal consistency loss (GGGS Eq 25) and gradient w.r.t. render_normal.
     *
     * For each valid pixel:
     *   error = 1 - dot(render_normal, depth_normal)
     *   grad_render_normal = -depth_normal / N_valid
     *
     * Valid pixel: depth > 0, |render_normal| > eps, |depth_normal| > eps.
     *
     * Two-pass: first kernel computes per-block partial sums + counts and writes
     * per-pixel unscaled gradient. Second kernel reduces to final scalar loss and
     * scales gradients by 1/N_valid.
     */
    void launch_normal_consistency_loss(
        const float* render_normal,  // [3, H, W] CHW, GPU
        const float* depth_normal,   // [3, H, W] CHW, GPU
        const float* depth,          // [H, W] GPU
        float* grad_render_normal,   // [3, H, W] CHW output, GPU
        float* grad_depth_normal,    // [3, H, W] CHW output, GPU (dL/d(dn) = -rn / N_valid)
        float* loss_out,             // [1] scalar output, GPU
        float* temp_sum_buffer,      // workspace for block-level partial sums
        int* temp_count_buffer,      // workspace for block-level valid pixel counts
        int H,
        int W,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
