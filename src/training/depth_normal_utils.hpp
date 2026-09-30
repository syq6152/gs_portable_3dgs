/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::training {

    // Convert a depth tensor ([1,H,W], [H,W,1], or [H,W]) into a camera-space normal map [3,H,W].
    // CPU implementation — use depth_to_normal_map_gpu for performance-critical paths.
    lfs::core::Tensor depth_to_normal_map(
        const lfs::core::Tensor& depth_map,
        float fx,
        float fy,
        float cx,
        float cy);

    // GPU implementation of depth_to_normal_map. Input may be on CPU or GPU;
    // output is always [3,H,W] on CUDA.
    lfs::core::Tensor depth_to_normal_map_gpu(
        const lfs::core::Tensor& depth_map,
        float fx,
        float fy,
        float cx,
        float cy);

    // Colorize depth to RGB [3,H,W] in [0,1] using a jet-like colormap.
    lfs::core::Tensor colorize_depth_map(const lfs::core::Tensor& depth_map);

    // Convert normal map [-1,1] to RGB [0,1] for visualization.
    lfs::core::Tensor colorize_normal_map(const lfs::core::Tensor& normal_map);

} // namespace lfs::training
