/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lfs::training::kernels {

    // Evaluate one fixed-grid set of pose-derived RGB patches.  RGB tensors
    // are contiguous CHW float32 tensors; depth tensors are contiguous HW (or
    // [1,H,W]) float32 tensors.  The target depth is rasterized at pixel centers
    // (x + 0.5, y + 0.5).  Projected source coordinates are in the same raster
    // coordinate system and are shifted by -0.5 internally for array sampling.
    // Each approximately 32x32 region contributes one complete 11x11 patch.
    // One CUDA block writes one patch score and its sample counts. Target mesh
    // depth establishes the pose-derived correspondence; source mesh depth is
    // retained only for API/report compatibility and is not a rejection test.
    void launch_scan_pose_zncc_regions(
        const float* target_rgb_chw,
        const float* target_depth_hw,
        const float* target_c2w_4x4,
        int target_width,
        int target_height,
        float target_fx,
        float target_fy,
        float target_cx,
        float target_cy,
        const float* source_rgb_chw,
        const float* source_depth_hw,
        const float* source_w2c_4x4,
        int source_width,
        int source_height,
        float source_fx,
        float source_fy,
        float source_cx,
        float source_cy,
        int region_count_x,
        int region_count_y,
        int samples_per_axis,
        int min_valid_samples,
        float relative_depth_epsilon,
        float variance_epsilon,
        float* region_zncc_out,
        int32_t* region_photometric_samples_out,
        cudaStream_t stream = nullptr,
        int32_t* region_target_samples_out = nullptr,
        int32_t* region_projected_samples_out = nullptr,
        int32_t* region_depth_consistent_samples_out = nullptr);

} // namespace lfs::training::kernels
