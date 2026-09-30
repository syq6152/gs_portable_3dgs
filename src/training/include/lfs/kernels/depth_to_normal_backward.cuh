/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Redirect: the canonical declaration is in the .hpp sibling.
// This .cuh is kept for inclusion from CUDA translation units but
// the .hpp is preferred from C++ code.
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_depth_to_normal_backward(
        const float* grad_depth_normal,
        const float* depth,
        float* grad_depth,
        int H,
        int W,
        float fx,
        float fy,
        float cx,
        float cy,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
