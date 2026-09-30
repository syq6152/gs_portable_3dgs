/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernels_backward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <functional>
#include <utility>

namespace {

    class ScopedCudaBuffer {
    public:
        ScopedCudaBuffer() = default;
        ScopedCudaBuffer(const ScopedCudaBuffer&) = delete;
        ScopedCudaBuffer& operator=(const ScopedCudaBuffer&) = delete;

        ~ScopedCudaBuffer() {
            if (ptr_) {
                // Preserve an in-flight exception. Successful paths call
                // free_checked() so deallocation failures are still reported.
                cudaFree(ptr_);
            }
        }

        void allocate(size_t bytes, const char* stage) {
            if (ptr_) {
                throw std::logic_error("FastGS ScopedCudaBuffer allocated twice");
            }
            fast_lfs::check_cuda(cudaMalloc(&ptr_, bytes), stage);
        }

        template <typename T>
        [[nodiscard]] T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        void free_checked(const char* stage) {
            void* const ptr = std::exchange(ptr_, nullptr);
            if (ptr) {
                fast_lfs::check_cuda(cudaFree(ptr), stage);
            }
        }

    private:
        void* ptr_ = nullptr;
    };

} // namespace

void fast_lfs::rasterization::backward(
    const float* densification_error_map,
    const float* grad_image,
    const float* grad_alpha,
    const float* image,
    const float* alpha,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float4* grad_w2c,
    float* densification_info,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter,
    const ObservationBlurSettings& observation_blur) {
    if (n_visible_primitives == 0 || n_instances == 0 || n_buckets == 0)
        return;

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = checked_cub_count(
        checked_size_multiply(static_cast<size_t>(grid.x), static_cast<size_t>(grid.y),
                              "backward.tile_count"),
        "backward.tile_count");

    // These blobs are from the arena and are guaranteed to be valid
    const int end_bit = extract_end_bit(static_cast<uint>(n_tiles - 1));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);

    // Restore selectors from forward pass
    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    // Backward blend (template dispatch eliminates densification branch from inner loop)
    auto launch_blend_backward = [&]<bool HAS_DENSIFICATION>() {
        kernels::backward::blend_backward_cu<HAS_DENSIFICATION><<<n_buckets, 32>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.bucket_offsets,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            raw_opacities,
            grad_image,
            grad_alpha,
            image,
            alpha,
            per_tile_buffers.max_n_contributions,
            per_tile_buffers.n_contributions,
            per_bucket_buffers.tile_index,
            per_bucket_buffers.checkpoint_uint8,
            grad_mean2d_helper,
            grad_conic_helper,
            per_primitive_buffers.grad_compensated_opacity,
            grad_opacities_raw,
            grad_sh_coefficients_0, // used to store intermediate gradients
            densification_info,
            densification_error_map,
            n_buckets,
            n_primitives,
            width,
            height,
            grid.x,
            mip_filter,
            observation_blur.enabled());
    };
    if (densification_info != nullptr && densification_error_map != nullptr) {
        launch_blend_backward.template operator()<true>();
    } else {
        launch_blend_backward.template operator()<false>();
    }
    CHECK_CUDA(config::debug, "blend_backward")

    // Backward preprocess
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        raw_opacities,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        grad_sh_coefficients_0,
        grad_sh_coefficients_rest,
        grad_w2c,
        densification_error_map == nullptr ? densification_info : nullptr,
        n_primitives,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        mip_filter,
        observation_blur.parameters_ptr,
        observation_blur.gradients_ptr,
        per_primitive_buffers.grad_compensated_opacity,
        observation_blur.motion_enabled,
        observation_blur.defocus_enabled,
        observation_blur.max_defocus_radius_sq,
        observation_blur.max_observation_radius_sq);
    CHECK_CUDA(config::debug, "preprocess_backward")
}

void fast_lfs::rasterization::backward_normal(
    const float* grad_render_normal,
    const float* render_normal,
    const float* normal_accum_length_map,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float4* w2c,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter) {
    if (n_visible_primitives == 0 || n_instances == 0 || n_buckets == 0)
        return;

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = checked_cub_count(
        checked_size_multiply(static_cast<size_t>(grid.x), static_cast<size_t>(grid.y),
                              "backward_normal.tile_count"),
        "backward_normal.tile_count");
    const int end_bit = extract_end_bit(static_cast<uint>(n_tiles - 1));

    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets, true);

    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    const size_t grad_mean2d_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), sizeof(float2), "backward_normal.grad_mean2d");
    const size_t grad_conic_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), 3 * sizeof(float), "backward_normal.grad_conic");
    check_cuda(cudaMemset(grad_mean2d_helper, 0, grad_mean2d_bytes),
               "memset.backward_normal.grad_mean2d");
    check_cuda(cudaMemset(grad_conic_helper, 0, grad_conic_bytes),
               "memset.backward_normal.grad_conic");

    // Allocate intermediate per-Gaussian normal gradient buffer
    const size_t grad_normal_per_gaussian_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), sizeof(float3), "backward_normal.per_gaussian");
    ScopedCudaBuffer grad_normal_per_gaussian_buffer;
    grad_normal_per_gaussian_buffer.allocate(
        grad_normal_per_gaussian_bytes, "malloc.backward_normal.per_gaussian");
    float3* const grad_normal_per_gaussian = grad_normal_per_gaussian_buffer.as<float3>();
    check_cuda(cudaMemset(grad_normal_per_gaussian, 0, grad_normal_per_gaussian_bytes),
               "memset.backward_normal.per_gaussian");

    // Step 1: Compute grad_normal_accum from grad_render_normal.
    // render_normal = normalize(normal_accum), so:
    //   grad_normal_accum = (grad_render_normal - render_normal * dot(grad_render_normal, render_normal)) / nlen
    // We precompute this into a temporary buffer.
    const size_t n_pixels_size = checked_size_multiply(
        static_cast<size_t>(width), static_cast<size_t>(height), "backward_normal.pixel_count");
    const int n_pixels = checked_cub_count(n_pixels_size, "backward_normal.pixel_count");
    const size_t grad_normal_accum_bytes = checked_size_multiply(
        n_pixels_size, 3 * sizeof(float), "backward_normal.accum");
    ScopedCudaBuffer grad_normal_accum_buffer;
    grad_normal_accum_buffer.allocate(
        grad_normal_accum_bytes, "malloc.backward_normal.accum");
    float* const grad_normal_accum = grad_normal_accum_buffer.as<float>();

    // Launch a simple kernel to compute grad_normal_accum
    // Using an inline lambda isn't possible in CUDA, use a kernel in kernels_backward.cuh
    // For now, launch the normalization backward as a simple element-wise kernel
    {
        const int threads = 256;
        const int blocks = div_round_up(n_pixels, threads);
        kernels::backward::normalize_normal_backward_cu<<<blocks, threads>>>(
            grad_render_normal,
            render_normal,
            normal_accum_length_map,
            grad_normal_accum,
            n_pixels);
        CHECK_CUDA(config::debug, "normalize_normal_backward")
    }

    // Step 2: Backward blend for normals
    kernels::backward::blend_backward_normal_cu<<<n_buckets, 32>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.bucket_offsets,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.normal,
        raw_opacities,
        grad_normal_accum,
        per_tile_buffers.max_n_contributions,
        per_tile_buffers.n_contributions,
        per_bucket_buffers.tile_index,
        per_bucket_buffers.checkpoint_uint8,
        per_bucket_buffers.checkpoint_normal_uint8,
        render_normal,
        normal_accum_length_map,
        grad_normal_per_gaussian,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_opacities_raw,
        n_buckets,
        n_primitives,
        width,
        height,
        grid.x,
        mip_filter);
    CHECK_CUDA(config::debug, "blend_backward_normal")

    // Consume the additional mean2d/conic contributions introduced by the
    // normal alpha term so they reach mean/scale/rotation gradients.
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        raw_opacities,
        nullptr,
        w2c,
        nullptr,
        per_primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        n_primitives,
        0,
        0,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        mip_filter,
        nullptr,
        nullptr,
        nullptr,
        false,
        false,
        0.0f,
        0.0f);
    CHECK_CUDA(config::debug, "preprocess_backward_normal_splatting")

    // Step 3: Backward preprocess for normals (nJ transform backward)
    kernels::backward::preprocess_backward_normal_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        w2c,
        grad_normal_per_gaussian,
        per_primitive_buffers.n_touched_tiles,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        n_primitives,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy);
    CHECK_CUDA(config::debug, "preprocess_backward_normal")

    grad_normal_per_gaussian_buffer.free_checked("free.backward_normal.per_gaussian");
    grad_normal_accum_buffer.free_checked("free.backward_normal.accum");
}

void fast_lfs::rasterization::backward_depth(
    const float* grad_depth,
    const float* depth_map,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* raw_opacities,
    const float4* w2c,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_bucket_buffers_blob,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float3* grad_means,
    float3* grad_scales_raw,
    float4* grad_rotations_raw,
    float* grad_opacities_raw,
    const int n_primitives,
    const int n_visible_primitives,
    const int n_instances,
    const int n_buckets,
    const int primitive_primitive_indices_selector,
    const int instance_primitive_indices_selector,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    bool mip_filter) {
    if (n_visible_primitives == 0 || n_instances == 0 || n_buckets == 0)
        return;

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = checked_cub_count(
        checked_size_multiply(static_cast<size_t>(grid.x), static_cast<size_t>(grid.y),
                              "backward_depth.tile_count"),
        "backward_depth.tile_count");
    const int end_bit = extract_end_bit(static_cast<uint>(n_tiles - 1));
    const size_t n_pixels_size = checked_size_multiply(
        static_cast<size_t>(width), static_cast<size_t>(height), "backward_depth.pixel_count");
    const int n_pixels = checked_cub_count(n_pixels_size, "backward_depth.pixel_count");

    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    PerBucketBuffers per_bucket_buffers = PerBucketBuffers::from_blob(per_bucket_buffers_blob, n_buckets);

    per_primitive_buffers.primitive_indices.selector = primitive_primitive_indices_selector;
    per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    const size_t grad_mean2d_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), sizeof(float2), "backward_depth.grad_mean2d");
    const size_t grad_conic_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), 3 * sizeof(float), "backward_depth.grad_conic");
    check_cuda(cudaMemset(grad_mean2d_helper, 0, grad_mean2d_bytes),
               "memset.backward_depth.grad_mean2d");
    check_cuda(cudaMemset(grad_conic_helper, 0, grad_conic_bytes),
               "memset.backward_depth.grad_conic");

    // ===== GGGS IFT Pass 1: Compute per-pixel dT/dt_m =====
    const size_t pixel_float_bytes = checked_size_multiply(
        n_pixels_size, sizeof(float), "backward_depth.pixel_float_bytes");
    ScopedCudaBuffer dT_dtm_buffer;
    dT_dtm_buffer.allocate(pixel_float_bytes, "malloc.backward_depth.dT_dtm");
    float* const dT_dtm_map = dT_dtm_buffer.as<float>();
    check_cuda(cudaMemset(dT_dtm_map, 0, pixel_float_bytes),
               "memset.backward_depth.dT_dtm");

    kernels::backward::compute_dT_dtm_depth_cu<<<grid, config::block_size_blend>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.ray_plane,
        depth_map,
        per_tile_buffers.n_contributions,
        dT_dtm_map,
        width,
        height,
        grid.x,
        fx, fy, cx, cy);
    CHECK_CUDA(config::debug, "compute_dT_dtm_depth")

    // ===== Elementwise: dL_dmt_dT_dtm = grad_depth_ray / max(-dT_dtm, eps) =====
    ScopedCudaBuffer dL_dmt_dT_dtm_buffer;
    dL_dmt_dT_dtm_buffer.allocate(pixel_float_bytes, "malloc.backward_depth.dL_dmt_dT_dtm");
    float* const dL_dmt_dT_dtm_map = dL_dmt_dT_dtm_buffer.as<float>();

    {
        const int threads = 256;
        const int blocks_elem = div_round_up(n_pixels, threads);
        kernels::backward::compute_dL_dmt_dT_dtm_cu<<<blocks_elem, threads>>>(
            dT_dtm_map,
            grad_depth,
            dL_dmt_dT_dtm_map,
            width, height,
            fx, fy, cx, cy);
        CHECK_CUDA(config::debug, "compute_dL_dmt_dT_dtm")
    }

    dT_dtm_buffer.free_checked("free.backward_depth.dT_dtm");

    // ===== GGGS IFT Pass 2: Per-Gaussian gradient computation =====
    const size_t grad_ray_plane_bytes = checked_size_multiply(
        static_cast<size_t>(n_primitives), sizeof(float4), "backward_depth.grad_ray_plane");
    ScopedCudaBuffer grad_ray_plane_buffer;
    grad_ray_plane_buffer.allocate(grad_ray_plane_bytes, "malloc.backward_depth.grad_ray_plane");
    float4* const grad_ray_plane = grad_ray_plane_buffer.as<float4>();
    check_cuda(cudaMemset(grad_ray_plane, 0, grad_ray_plane_bytes),
               "memset.backward_depth.grad_ray_plane");

    kernels::backward::blend_backward_depth_ift_cu<<<n_buckets, 32>>>(
        per_tile_buffers.instance_ranges,
        per_tile_buffers.bucket_offsets,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.ray_plane,
        raw_opacities,
        depth_map,
        dL_dmt_dT_dtm_map,
        per_tile_buffers.max_n_contributions,
        per_tile_buffers.n_contributions,
        per_bucket_buffers.tile_index,
        grad_ray_plane,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_opacities_raw,
        n_buckets,
        n_primitives,
        width,
        height,
        grid.x,
        fx, fy, cx, cy,
        mip_filter);
    CHECK_CUDA(config::debug, "blend_backward_depth_ift")

    // Consume the additional mean2d/conic contributions introduced by the
    // depth alpha term so they reach mean/scale/rotation gradients.
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        raw_opacities,
        nullptr,
        w2c,
        nullptr,
        per_primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        n_primitives,
        0,
        0,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        mip_filter,
        nullptr,
        nullptr,
        nullptr,
        false,
        false,
        0.0f,
        0.0f);
    CHECK_CUDA(config::debug, "preprocess_backward_depth_splatting")

    dL_dmt_dT_dtm_buffer.free_checked("free.backward_depth.dL_dmt_dT_dtm");

    // ===== Preprocess: grad_ray_plane → grad_means/scales/rotations =====
    kernels::backward::preprocess_backward_depth_ift_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales_raw,
        rotations_raw,
        w2c,
        grad_ray_plane,
        per_primitive_buffers.n_touched_tiles,
        grad_means,
        grad_scales_raw,
        grad_rotations_raw,
        n_primitives,
        static_cast<float>(width),
        static_cast<float>(height),
        fx, fy, cx, cy);
    CHECK_CUDA(config::debug, "preprocess_backward_depth_ift")

    grad_ray_plane_buffer.free_checked("free.backward_depth.grad_ray_plane");
}
