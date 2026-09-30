/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_api.h"
#include <cstdint>
#include <functional>
#include <tuple>

namespace fast_lfs::rasterization {

    struct ForwardRuntimeInfo {
        const char* stage = "not_started";
        uint64_t n_visible_primitives = 0;
        uint64_t n_instances = 0;
        uint64_t n_buckets = 0;
    };

    std::tuple<int, int, int, int, int> forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        std::function<char*(size_t)> per_instance_buffers_func,
        std::function<char*(size_t)> per_bucket_buffers_func,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        float* image,
        float* alpha,
        float* depth,
        float* normal_map,
        const int n_primitives,
        const int active_sh_bases,
        const int total_bases_sh_rest,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near,
        const float far,
        bool require_depth,
        bool mip_filter,
        bool require_normal_backward = false,
        float* normal_accum_length_map = nullptr,
        const float* mesh_depth_cull = nullptr,
        int mesh_depth_width = 0,
        int mesh_depth_height = 0,
        int mesh_depth_x_offset = 0,
        int mesh_depth_y_offset = 0,
        ForwardRuntimeInfo* runtime_info = nullptr,
        const ObservationBlurSettings& observation_blur = {});

}
