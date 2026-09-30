/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs {

    // Adjust saturation of Gaussians under the brush
    // sh0: pointer to SH0 data [N, 3] (or [N, 1, 3] viewed as [N*1, 3])
    // screen_positions: [N, 2] screen positions from last render
    // brush_x, brush_y: brush center in screen coords
    // brush_radius: radius in pixels
    // saturation_delta: -1 to 1, negative = desaturate, positive = increase saturation
    // num_gaussians: number of Gaussians
    // stream: CUDA stream (0 for default)
    void launchAdjustSaturation(
        float* sh0,
        const float* screen_positions,
        float brush_x,
        float brush_y,
        float brush_radius,
        float saturation_delta,
        int num_gaussians,
        cudaStream_t stream = 0);

} // namespace lfs
