/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * GPU forward of depth_to_normal_map (GGGS formula 24).
     *
     * Converts a depth map [H,W] to a camera-space normal map [3,H,W] (CHW)
     * using finite differences on unprojected 3D points.
     * Boundary pixels (row 0, H-1, col 0, W-1) are zero.
     *
     * @param depth   [H, W] depth map on GPU
     * @param normal  [3, H, W] output normal map on GPU (must be pre-zeroed for boundary pixels)
     */
    void launch_depth_to_normal_forward(
        const float* depth,    // [H, W] GPU
        float* normal,         // [3, H, W] CHW output, GPU
        int H,
        int W,
        float fx,
        float fy,
        float cx,
        float cy,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
