/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_api.h"
#include <functional>

namespace fast_lfs::rasterization {

    void backward(
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
        const ObservationBlurSettings& observation_blur = {});

    // GGGS normal backward: propagates grad_render_normal through normal blending
    // and nJ transform to accumulate into grad_means, grad_scales, grad_rotations.
    // Call AFTER backward() since it accumulates (+=) into the same gradient buffers.
    void backward_normal(
        const float* grad_render_normal, // [3, H, W] - from normal consistency + multi-view loss
        const float* render_normal,      // [3, H, W] - from forward
        const float* normal_accum_length_map, // [H, W] - |normal_accum| per pixel from forward blend
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* raw_opacities,
        const float4* w2c,
        char* per_primitive_buffers_blob,
        char* per_tile_buffers_blob,
        char* per_instance_buffers_blob,
        char* per_bucket_buffers_blob,
        float2* grad_mean2d_helper, // from forward context, accumulate
        float* grad_conic_helper,   // from forward context, accumulate
        float3* grad_means,       // accumulate
        float3* grad_scales_raw,  // accumulate
        float4* grad_rotations_raw, // accumulate
        float* grad_opacities_raw, // accumulate
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
        bool mip_filter);

    // GGGS depth backward: GGGS IFT (Equation 18) median depth backward.
    // Two-pass kernel: (1) compute per-pixel dT/dt_m, (2) propagate to per-Gaussian gradients.
    // Produces grad_ray_plane → preprocess to grad_means/scales/rotations.
    // The depth_to_normal_backward (grad_depth_normal → grad_depth) is called at a
    // higher level (fast_rasterizer.cpp) before this function.
    // Call AFTER backward_normal() since it accumulates (+=) into the same gradient buffers.
    void backward_depth(
        const float* grad_depth,        // [H, W] - gradient w.r.t. depth map (camera-z)
        const float* depth_map,         // [H, W] - forward depth map (camera-z, for IFT)
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
        bool mip_filter);

}
