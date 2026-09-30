/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * Backward of depth_to_normal_map(): propagates grad_depth_normal [3,H,W] to grad_depth [H,W].
     *
     * Each output normal at (y,x) depends on depths at 4 neighbors:
     * (y,x-1), (y,x+1), (y-1,x), (y+1,x). The backward scatters gradient to
     * those 4 depth values via atomicAdd.
     */
    void launch_depth_to_normal_backward(
        const float* grad_depth_normal,  // [3, H, W] CHW, GPU
        const float* depth,              // [H, W] GPU
        float* grad_depth,               // [H, W] output, GPU (accumulated with atomicAdd)
        int H,
        int W,
        float fx,
        float fy,
        float cx,
        float cy,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
