/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    enum class MeshInitRgbCounter : int {
        Projected = 0,
        Valid = 1,
        ProjectionFail = 2,
        OcclusionFail = 3,
        SourceMaskFail = 4,
        Count = 5,
    };

    // Assign the first valid RGB observation for each Gaussian. Rows whose
    // weight_sum is already positive are skipped, allowing callers to process
    // many cameras without repeatedly reprojecting resolved points.
    void launch_mesh_init_rgb_accumulate(
        const float* means_xyz,
        int point_count,
        const float* source_depth,
        const float* source_rgb,
        const float* source_mask,
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        const float* source_w2c_4x4,
        float scene_radius,
        float rel_depth_epsilon,
        float source_mask_threshold,
        float* rgb_sum,
        float* weight_sum,
        int32_t* counters,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
