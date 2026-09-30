/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <functional>
#include <limits>
#include <stdexcept>

// Sorting is done separately for depth and tile as proposed in
// https://github.com/m-schuetz/Splatshop.
std::tuple<int, int, int, int, int> fast_lfs::rasterization::forward(
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
    const float near_, // near and far are macros in Windows
    const float far_,
    bool require_depth,
    bool mip_filter,
    bool require_normal_backward,
    float* normal_accum_length_map,
    const float* mesh_depth_cull,
    int mesh_depth_width,
    int mesh_depth_height,
    int mesh_depth_x_offset,
    int mesh_depth_y_offset,
    ForwardRuntimeInfo* runtime_info,
    const ObservationBlurSettings& observation_blur) {

    ForwardRuntimeInfo local_runtime_info{};
    if (!runtime_info) {
        runtime_info = &local_runtime_info;
    }
    const auto set_stage = [runtime_info](const char* stage) {
        runtime_info->stage = stage;
    };

    set_stage("forward.entry");
    check_no_pending_cuda_error("forward.entry.pending_cuda_error");
    if (n_primitives <= 0 || width <= 0 || height <= 0) {
        throw std::invalid_argument("FastGS forward requires positive primitive and image dimensions");
    }

    set_stage("forward.grid_sizing");
    const dim3 grid(div_round_up(width, config::tile_width),
                    div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    if (n_tiles_u64 == 0 || n_tiles_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("FastGS tile count is outside the supported CUB int range");
    }
    // Instance tile keys are ushort, so every tile index must fit exactly.
    if (n_tiles_u64 > static_cast<uint64_t>(std::numeric_limits<unsigned short>::max()) + 1ULL) {
        throw std::length_error("FastGS tile count exceeds the 16-bit instance-key range");
    }
    const int n_tiles = static_cast<int>(n_tiles_u64);

    set_stage("buffer_size.per_tile");
    const size_t requested_per_tile_bytes = required<PerTileBuffers>(n_tiles);
    set_stage("arena_allocate.per_tile");
    char* const per_tile_buffers_blob = per_tile_buffers_func(requested_per_tile_bytes);
    if (!per_tile_buffers_blob) {
        throw std::runtime_error("OUT_OF_MEMORY: FastGS per-tile arena allocation returned null");
    }
    char* per_tile_cursor = per_tile_buffers_blob;
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_cursor, n_tiles);

    set_stage("memset.tile_instance_ranges");
    const size_t tile_range_bytes = checked_size_multiply(
        static_cast<size_t>(n_tiles), sizeof(uint2), "memset.tile_instance_ranges");
    check_cuda(cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, tile_range_bytes, nullptr),
               "memset.tile_instance_ranges");

    set_stage("buffer_size.per_primitive");
    const size_t requested_per_primitive_bytes = required<PerPrimitiveBuffers>(n_primitives);
    set_stage("arena_allocate.per_primitive");
    char* const per_primitive_buffers_blob = per_primitive_buffers_func(requested_per_primitive_bytes);
    if (!per_primitive_buffers_blob) {
        throw std::runtime_error("OUT_OF_MEMORY: FastGS per-primitive arena allocation returned null");
    }
    char* per_primitive_cursor = per_primitive_buffers_blob;
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(
        per_primitive_cursor, n_primitives);

    set_stage("memset.forward_counts");
    check_cuda(cudaMemsetAsync(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint), nullptr),
               "memset.n_visible_primitives");
    check_cuda(cudaMemsetAsync(per_primitive_buffers.n_instances, 0,
                               sizeof(unsigned long long), nullptr),
               "memset.n_instances");

    set_stage("kernel.preprocess");
    kernels::forward::preprocess_cu<<<
        div_round_up(n_primitives, config::block_size_preprocess),
        config::block_size_preprocess>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys.Current(),
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.depth,
        per_primitive_buffers.ray_plane,
        per_primitive_buffers.normal,
        per_primitive_buffers.n_visible_primitives,
        per_primitive_buffers.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        near_,
        far_,
        mip_filter,
        require_depth,
        mesh_depth_cull,
        mesh_depth_width,
        mesh_depth_height,
        mesh_depth_x_offset,
        mesh_depth_y_offset,
        observation_blur.parameters_ptr,
        observation_blur.motion_enabled,
        observation_blur.defocus_enabled,
        observation_blur.max_defocus_radius_sq,
        observation_blur.max_observation_radius_sq);
    CHECK_CUDA(config::debug, "kernel.preprocess")

    set_stage("copy.forward_counts");
    uint n_visible_primitives_u32 = 0;
    check_cuda(cudaMemcpy(&n_visible_primitives_u32,
                          per_primitive_buffers.n_visible_primitives,
                          sizeof(n_visible_primitives_u32), cudaMemcpyDeviceToHost),
               "copy.n_visible_primitives");
    unsigned long long n_instances_u64 = 0;
    check_cuda(cudaMemcpy(&n_instances_u64,
                          per_primitive_buffers.n_instances,
                          sizeof(n_instances_u64), cudaMemcpyDeviceToHost),
               "copy.n_instances");

    runtime_info->n_visible_primitives = n_visible_primitives_u32;
    runtime_info->n_instances = n_instances_u64;
    if (n_visible_primitives_u32 > static_cast<uint>(n_primitives)) {
        throw std::runtime_error("FastGS visible primitive counter exceeded the input count");
    }
    if (n_instances_u64 > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        throw std::length_error(
            "FastGS instance count exceeds INT_MAX before CUB sort (count=" +
            std::to_string(n_instances_u64) + ")");
    }
    const int n_visible_primitives = static_cast<int>(n_visible_primitives_u32);
    const int n_instances = static_cast<int>(n_instances_u64);

    const int alloc_instances = std::max(n_instances, 1);
    const int end_bit = extract_end_bit(static_cast<uint>(n_tiles - 1));
    set_stage("buffer_size.per_instance");
    const size_t requested_per_instance_bytes = required<PerInstanceBuffers>(alloc_instances, end_bit);
    set_stage("arena_allocate.per_instance");
    char* const per_instance_buffers_blob = per_instance_buffers_func(requested_per_instance_bytes);
    if (!per_instance_buffers_blob) {
        throw std::runtime_error("OUT_OF_MEMORY: FastGS per-instance arena allocation returned null");
    }
    char* per_instance_cursor = per_instance_buffers_blob;
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(
        per_instance_cursor, alloc_instances, end_bit);

    if (n_visible_primitives > 0) {
        set_stage("cub_run.depth_sort");
        size_t workspace_bytes = per_primitive_buffers.cub_workspace_size;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       per_primitive_buffers.cub_workspace,
                       workspace_bytes,
                       per_primitive_buffers.depth_keys,
                       per_primitive_buffers.primitive_indices,
                       n_visible_primitives),
                   "cub_run.depth_sort");

        set_stage("kernel.apply_depth_ordering");
        kernels::forward::apply_depth_ordering_cu<<<
            div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering),
            config::block_size_apply_depth_ordering>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_visible_primitives);
        CHECK_CUDA(config::debug, "kernel.apply_depth_ordering")

        set_stage("cub_run.primitive_offset_scan");
        workspace_bytes = per_primitive_buffers.cub_workspace_size;
        check_cuda(cub::DeviceScan::ExclusiveSum(
                       per_primitive_buffers.cub_workspace,
                       workspace_bytes,
                       per_primitive_buffers.offset,
                       per_primitive_buffers.offset,
                       n_visible_primitives),
                   "cub_run.primitive_offset_scan");

        set_stage("kernel.create_instances");
        kernels::forward::create_instances_cu<<<
            div_round_up(n_visible_primitives, config::block_size_create_instances),
            config::block_size_create_instances>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.offset,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            grid.x,
            n_visible_primitives);
        CHECK_CUDA(config::debug, "kernel.create_instances")

        if (n_instances > 0) {
            set_stage("cub_run.tile_sort");
            workspace_bytes = per_instance_buffers.cub_workspace_size;
            check_cuda(cub::DeviceRadixSort::SortPairs(
                           per_instance_buffers.cub_workspace,
                           workspace_bytes,
                           per_instance_buffers.keys,
                           per_instance_buffers.primitive_indices,
                           n_instances, 0, end_bit),
                       "cub_run.tile_sort");
        }
    }

    if (n_instances > 0) {
        set_stage("kernel.extract_instance_ranges");
        kernels::forward::extract_instance_ranges_cu<<<
            div_round_up(n_instances, config::block_size_extract_instance_ranges),
            config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances);
        CHECK_CUDA(config::debug, "kernel.extract_instance_ranges")
    }

    set_stage("kernel.extract_bucket_counts");
    kernels::forward::extract_bucket_counts<<<
        div_round_up(n_tiles, config::block_size_extract_bucket_counts),
        config::block_size_extract_bucket_counts>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.n_buckets,
        n_tiles);
    CHECK_CUDA(config::debug, "kernel.extract_bucket_counts")

    set_stage("cub_run.bucket_offset_scan");
    size_t tile_workspace_bytes = per_tile_buffers.cub_workspace_size;
    check_cuda(cub::DeviceScan::InclusiveSum(
                   per_tile_buffers.cub_workspace,
                   tile_workspace_bytes,
                   per_tile_buffers.n_buckets,
                   per_tile_buffers.bucket_offsets,
                   n_tiles),
               "cub_run.bucket_offset_scan");

    set_stage("copy.n_buckets");
    uint n_buckets_u32 = 0;
    check_cuda(cudaMemcpy(&n_buckets_u32,
                          per_tile_buffers.bucket_offsets + n_tiles - 1,
                          sizeof(n_buckets_u32), cudaMemcpyDeviceToHost),
               "copy.n_buckets");
    runtime_info->n_buckets = n_buckets_u32;
    if (n_buckets_u32 > static_cast<uint>(std::numeric_limits<int>::max())) {
        throw std::length_error(
            "FastGS bucket count exceeds INT_MAX (count=" +
            std::to_string(n_buckets_u32) + ")");
    }
    const int n_buckets = static_cast<int>(n_buckets_u32);

    const int alloc_buckets = std::max(n_buckets, 1);
    set_stage("buffer_size.per_bucket");
    const size_t requested_per_bucket_bytes = required<PerBucketBuffers>(
        alloc_buckets, require_normal_backward);
    set_stage("arena_allocate.per_bucket");
    char* const per_bucket_buffers_blob = per_bucket_buffers_func(requested_per_bucket_bytes);
    if (!per_bucket_buffers_blob) {
        throw std::runtime_error("OUT_OF_MEMORY: FastGS per-bucket arena allocation returned null");
    }
    char* per_bucket_cursor = per_bucket_buffers_blob;
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(
        per_bucket_cursor, alloc_buckets, require_normal_backward);

    if (require_depth && depth != nullptr) {
        set_stage("kernel.blend_depth");
        kernels::forward::blend_cu<true><<<grid, block>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.bucket_offsets,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            per_primitive_buffers.ray_plane,
            per_primitive_buffers.normal,
            image,
            alpha,
            depth,
            normal_map,
            require_normal_backward ? normal_accum_length_map : nullptr,
            per_tile_buffers.max_n_contributions,
            per_tile_buffers.n_contributions,
            per_bucket_buffers.tile_index,
            per_bucket_buffers.checkpoint_uint8,
            per_bucket_buffers.checkpoint_normal_uint8,
            width,
            height,
            grid.x,
            fx,
            fy,
            cx,
            cy);
        CHECK_CUDA(config::debug, "kernel.blend_depth")

        set_stage("kernel.depth_refine");
        kernels::forward::depth_refine_cu<<<grid, block>>>(
            per_tile_buffers.instance_ranges,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.ray_plane,
            per_tile_buffers.n_contributions,
            depth,
            width,
            height,
            grid.x,
            fx,
            fy,
            cx,
            cy);
        CHECK_CUDA(config::debug, "kernel.depth_refine")
    } else {
        set_stage("kernel.blend_color");
        kernels::forward::blend_cu<false><<<grid, block>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.bucket_offsets,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            nullptr,
            nullptr,
            image,
            alpha,
            nullptr,
            nullptr,
            nullptr,
            per_tile_buffers.max_n_contributions,
            per_tile_buffers.n_contributions,
            per_bucket_buffers.tile_index,
            per_bucket_buffers.checkpoint_uint8,
            nullptr,
            width,
            height,
            grid.x,
            fx,
            fy,
            cx,
            cy);
        CHECK_CUDA(config::debug, "kernel.blend_color")
    }

    set_stage("forward.complete");
    return {n_visible_primitives,
            n_instances,
            n_buckets,
            per_primitive_buffers.primitive_indices.selector,
            per_instance_buffers.primitive_indices.selector};
}
