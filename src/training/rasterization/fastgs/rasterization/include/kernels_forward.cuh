/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_lfs::rasterization::kernels::forward {

    __device__ inline mat3x3 full_from_triu(const mat3x3_triu& a) {
        return {a.m11, a.m12, a.m13,
                a.m12, a.m22, a.m23,
                a.m13, a.m23, a.m33};
    }

    __device__ inline mat3x3_triu triu_from_full(const mat3x3& a) {
        return {a.m11, a.m12, a.m13, a.m22, a.m23, a.m33};
    }

    __device__ inline mat3x3 mat3_mul(const mat3x3& a, const mat3x3& b) {
        return {
            a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31,
            a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32,
            a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33,
            a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31,
            a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32,
            a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33,
            a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31,
            a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32,
            a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33};
    }

    __device__ inline mat3x3 mat3_transpose(const mat3x3& a) {
        return {a.m11, a.m21, a.m31,
                a.m12, a.m22, a.m32,
                a.m13, a.m23, a.m33};
    }

    __device__ inline float mat3_det(const mat3x3& a) {
        return a.m11 * (a.m22 * a.m33 - a.m23 * a.m32) -
               a.m12 * (a.m21 * a.m33 - a.m23 * a.m31) +
               a.m13 * (a.m21 * a.m32 - a.m22 * a.m31);
    }

    __device__ inline mat3x3 mat3_inverse(const mat3x3& a, const float det) {
        const float rcp = 1.0f / det;
        return {
            (a.m22 * a.m33 - a.m23 * a.m32) * rcp,
            (a.m13 * a.m32 - a.m12 * a.m33) * rcp,
            (a.m12 * a.m23 - a.m13 * a.m22) * rcp,
            (a.m23 * a.m31 - a.m21 * a.m33) * rcp,
            (a.m11 * a.m33 - a.m13 * a.m31) * rcp,
            (a.m13 * a.m21 - a.m11 * a.m23) * rcp,
            (a.m21 * a.m32 - a.m22 * a.m31) * rcp,
            (a.m12 * a.m31 - a.m11 * a.m32) * rcp,
            (a.m11 * a.m22 - a.m12 * a.m21) * rcp};
    }

    __device__ inline mat3x3 motion_covariance_camera(
        const float3& p,
        const float3& rotation_variance,
        const float3& translation_variance) {
        // [p]_x diag(v) [p]_x^T, expanded to avoid a device matrix helper.
        const float px = p.x, py = p.y, pz = p.z;
        const float vx = rotation_variance.x;
        const float vy = rotation_variance.y;
        const float vz = rotation_variance.z;
        const float tx = translation_variance.x;
        const float ty = translation_variance.y;
        const float tz = translation_variance.z;
        return {
            vy * pz * pz + vz * py * py + tx, -vz * px * py, -vy * px * pz,
            -vz * px * py, vx * pz * pz + vz * px * px + ty, -vx * py * pz,
            -vy * px * pz, -vx * py * pz, vx * py * py + vy * px * px + tz};
    }

    struct RayPlaneAndNormal {
        float4 ray_plane;
        float3 normal;
    };

    // GGGS-style cov_cam_inv via factored M_inv = S_inv * R^T * W^T path.
    // Also computes per-Gaussian camera-space normal via nJ transform.
    __device__ inline RayPlaneAndNormal compute_ray_plane_and_normal(
        const mat3x3& rotation,
        const float3& variance,
        const float4& w2c_r1,
        const float4& w2c_r2,
        const float4& w2c_r3,
        const float tx,
        const float ty,
        const float depth,
        const float fx,
        const float fy,
        const float fallback_depth) {
        RayPlaneAndNormal result;
        result.ray_plane = make_float4(0.0f, 0.0f, fallback_depth, 0.0f);
        result.normal = make_float3(0.0f, 0.0f, -1.0f);

        // Inverse scales: 1/sqrt(variance) = 1/exp(clamped_raw_scale)
        const float inv_s0 = rsqrtf(fmaxf(variance.x, 1e-20f));
        const float inv_s1 = rsqrtf(fmaxf(variance.y, 1e-20f));
        const float inv_s2 = rsqrtf(fmaxf(variance.z, 1e-20f));

        // Columns of rotation matrix (local-to-world, row-major)
        const float3 r_col0 = make_float3(rotation.m11, rotation.m21, rotation.m31);
        const float3 r_col1 = make_float3(rotation.m12, rotation.m22, rotation.m32);
        const float3 r_col2 = make_float3(rotation.m13, rotation.m23, rotation.m33);

        // World-to-camera rotation rows
        const float3 w_r0 = make_float3(w2c_r1.x, w2c_r1.y, w2c_r1.z);
        const float3 w_r1 = make_float3(w2c_r2.x, w2c_r2.y, w2c_r2.z);
        const float3 w_r2 = make_float3(w2c_r3.x, w2c_r3.y, w2c_r3.z);

        // M_inv = S_inv * R^T * W^T, row i = (1/si) * [dot(r_col_i, w_rj) for j=0..2]
        const float3 m0 = inv_s0 * make_float3(dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2));
        const float3 m1 = inv_s1 * make_float3(dot(r_col1, w_r0), dot(r_col1, w_r1), dot(r_col1, w_r2));
        const float3 m2 = inv_s2 * make_float3(dot(r_col2, w_r0), dot(r_col2, w_r1), dot(r_col2, w_r2));

        // cov_cam_inv = M_inv^T * M_inv (symmetric, always PSD)
        const float cc00 = m0.x * m0.x + m1.x * m1.x + m2.x * m2.x;
        const float cc01 = m0.x * m0.y + m1.x * m1.y + m2.x * m2.y;
        const float cc02 = m0.x * m0.z + m1.x * m1.z + m2.x * m2.z;
        const float cc11 = m0.y * m0.y + m1.y * m1.y + m2.y * m2.y;
        const float cc12 = m0.y * m0.z + m1.y * m1.z + m2.y * m2.z;
        const float cc22 = m0.z * m0.z + m1.z * m1.z + m2.z * m2.z;

        const float u = tx;
        const float v = ty;
        const float uvh_m0 = cc00 * u + cc01 * v + cc02;
        const float uvh_m1 = cc01 * u + cc11 * v + cc12;
        const float uvh_m2 = cc02 * u + cc12 * v + cc22;
        const float vb = uvh_m0 * u + uvh_m1 * v + uvh_m2;
        if (!(vb > 1e-10f))
            return result;

        const float ray_len2 = u * u + v * v + 1.0f;
        const float tc = norm3df(u * depth, v * depth, depth);
        if (!(ray_len2 > 1e-10f) || !(tc > 1e-10f))
            return result;

        const float inv_vb = 1.0f / vb;
        const float vec0 = uvh_m0 * inv_vb;
        const float vec1 = uvh_m1 * inv_vb;
        const float vec2 = uvh_m2 * inv_vb;

        // nJ_inv * vec (GGGS formula, full 3rd column terms)
        const float u2 = u * u;
        const float v2 = v * v;
        const float uv = u * v;
        const float plane_x = (v2 + 1.0f) * vec0 - uv * vec1 - u * vec2;
        const float plane_y = -uv * vec0 + (u2 + 1.0f) * vec1 - v * vec2;

        const float factor_normal = tc / ray_len2;
        const float rsigma = sqrtf(vb / ray_len2);
        if (!isfinite(rsigma))
            return result;

        result.ray_plane = make_float4(
            plane_x * factor_normal / fx,
            plane_y * factor_normal / fy,
            fallback_depth,
            rsigma);

        // Per-Gaussian camera-space normal via nJ transform of ray-space normal
        const float ray_nx = -plane_x * factor_normal;
        const float ray_ny = -plane_y * factor_normal;
        constexpr float ray_nz = -1.0f;

        // nJ (row-major) maps ray-space to camera-space:
        //   [1/depth,     0,         tx*depth/tc]
        //   [0,           1/depth,   ty*depth/tc]
        //   [-tx/depth,   -ty/depth, depth/tc   ]
        const float inv_depth = 1.0f / depth;
        const float depth_over_tc = depth / tc;
        const float cam_nx = ray_nx * inv_depth + ray_nz * tx * depth_over_tc;
        const float cam_ny = ray_ny * inv_depth + ray_nz * ty * depth_over_tc;
        const float cam_nz = -ray_nx * tx * inv_depth - ray_ny * ty * inv_depth + ray_nz * depth_over_tc;

        const float nlen = norm3df(cam_nx, cam_ny, cam_nz);
        if (nlen > 1e-8f) {
            result.normal = make_float3(cam_nx / nlen, cam_ny / nlen, cam_nz / nlen);
        }

        return result;
    }

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float* __restrict__ raw_opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        float* __restrict__ primitive_depth,
        float4* __restrict__ primitive_ray_plane,
        float3* __restrict__ primitive_normal,
        uint* __restrict__ n_visible_primitives,
        unsigned long long* __restrict__ n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_, // near and far are macros in windowns
        const float far_,
        const bool mip_filter,
        const bool require_depth,
        const float* __restrict__ mesh_depth_cull,
        const int mesh_depth_width,
        const int mesh_depth_height,
        const int mesh_depth_x_offset,
        const int mesh_depth_y_offset,
        const float* __restrict__ observation_blur_params,
        const bool observation_motion_enabled,
        const bool observation_defocus_enabled,
        const float observation_max_defocus_radius_sq,
        const float observation_max_radius_sq) {
        auto primitive_idx = cg::this_grid().thread_rank();
        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (active)
            primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // z culling
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_ || depth > far_)
            active = false;

        // early exit if whole warp is inactive
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // compute projected Gaussian mean once. These camera-space normalized
        // coordinates also feed the EWA Jacobian below.
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;
        const float2 mean2d = make_float2(x * fx + cx, y * fy + cy);

        // Optional mesh-depth occlusion: if the Gaussian mean is behind the
        // rendered mesh surface at its projected center pixel, remove it from
        // the current view before it enters depth/tile sorting.
        if (active && mesh_depth_cull != nullptr) {
            const int local_px = __float2int_rd(mean2d.x);
            const int local_py = __float2int_rd(mean2d.y);
            const int mesh_px = local_px + mesh_depth_x_offset;
            const int mesh_py = local_py + mesh_depth_y_offset;
            if (mesh_px >= 0 && mesh_px < mesh_depth_width &&
                mesh_py >= 0 && mesh_py < mesh_depth_height) {
                const float mesh_depth = mesh_depth_cull[mesh_py * mesh_depth_width + mesh_px];
                const float depth_eps = fmaxf(1e-4f, fabsf(mesh_depth) * 1e-4f);
                if (mesh_depth > 0.0f && isfinite(mesh_depth) && depth > mesh_depth + depth_eps) {
                    active = false;
                }
            }
        }
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // load opacity
        const float raw_opacity = raw_opacities[primitive_idx];
        const float opacity = 1.0f / (1.0f + expf(-raw_opacity));
        if (opacity < config::min_alpha_threshold)
            active = false;

        // compute 3d covariance from raw scale and rotation
        const float3 raw_scale = active ? raw_scales[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        if (q_norm_sq < 1e-8f)
            active = false;
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-8f);
        const float qxx = 2.0f * qxx_raw / q_norm_sq_safe, qyy = 2.0f * qyy_raw / q_norm_sq_safe, qzz = 2.0f * qzz_raw / q_norm_sq_safe;
        const float qxy = 2.0f * qx * qy / q_norm_sq_safe, qxz = 2.0f * qx * qz / q_norm_sq_safe, qyz = 2.0f * qy * qz / q_norm_sq_safe;
        const float qrx = 2.0f * qr * qx / q_norm_sq_safe, qry = 2.0f * qr * qy / q_norm_sq_safe, qrz = 2.0f * qr * qz / q_norm_sq_safe;
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
        const mat3x3 rotation_scaled = {
            rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
            rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
            rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
        const mat3x3_triu cov3d{
            rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
            rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
            rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
            rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
            rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
            rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
        };

        // Optional Robust-GS observation motion model. The deterministic
        // Gaussian remains canonical; only this frame's projected covariance
        // is expanded. Motion is evaluated in camera space and projected
        // directly through the camera-space EWA Jacobian, leaving the
        // deterministic world covariance and the disabled path untouched.
        float motion_opacity_compensation = 1.0f;
        float defocus_opacity_compensation = 1.0f;
        float defocus_radius_sq = 0.0f;
        mat3x3 cov_camera = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f};
        mat3x3 motion_camera_raw = cov_camera;
        if (observation_blur_params != nullptr &&
            (observation_motion_enabled || observation_defocus_enabled)) {
            const float3 p_cam = make_float3(
                w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w,
                w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w,
                depth);
            const mat3x3 rotation_wc = {
                w2c_r1.x, w2c_r1.y, w2c_r1.z,
                w2c_r2.x, w2c_r2.y, w2c_r2.z,
                w2c_r3.x, w2c_r3.y, w2c_r3.z};
            const mat3x3 cov_world = full_from_triu(cov3d);
            const mat3x3 rotation_cw = mat3_transpose(rotation_wc);
            cov_camera = mat3_mul(
                mat3_mul(rotation_wc, cov_world), rotation_cw);
            if (observation_motion_enabled) {
                motion_camera_raw = motion_covariance_camera(
                    p_cam,
                    make_float3(observation_blur_params[0], observation_blur_params[1], observation_blur_params[2]),
                    make_float3(observation_blur_params[3], observation_blur_params[4], observation_blur_params[5]));
            }
        }

        // compute ray-distance for GGGS depth rendering/sorting
        const float ray_distance = norm3df(x * depth, y * depth, depth);

        // ewa splatting
        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float j11 = fx / depth;
        const float j13 = -j11 * tx;
        const float j22 = fy / depth;
        const float j23 = -j22 * ty;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z);
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z);
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);
        float3 cov2d = make_float3(dot(jwc_r1, jw_r1), dot(jwc_r1, jw_r2), dot(jwc_r2, jw_r2));

        // Project the raw camera-space motion covariance. The combined
        // observation contribution (motion + defocus) is capped below before
        // it is added to the canonical covariance, preventing unbounded tile
        // footprints while preserving its anisotropic direction.
        const float3 jc_r1 = make_float3(j11, 0.0f, j13);
        const float3 jc_r2 = make_float3(0.0f, j22, j23);
        float3 motion_cov2d_raw = make_float3(0.0f, 0.0f, 0.0f);
        if (observation_blur_params != nullptr && observation_motion_enabled) {
            const float3 jcm_r1 = make_float3(
                jc_r1.x * motion_camera_raw.m11 + jc_r1.y * motion_camera_raw.m21 + jc_r1.z * motion_camera_raw.m31,
                jc_r1.x * motion_camera_raw.m12 + jc_r1.y * motion_camera_raw.m22 + jc_r1.z * motion_camera_raw.m32,
                jc_r1.x * motion_camera_raw.m13 + jc_r1.y * motion_camera_raw.m23 + jc_r1.z * motion_camera_raw.m33);
            const float3 jcm_r2 = make_float3(
                jc_r2.x * motion_camera_raw.m11 + jc_r2.y * motion_camera_raw.m21 + jc_r2.z * motion_camera_raw.m31,
                jc_r2.x * motion_camera_raw.m12 + jc_r2.y * motion_camera_raw.m22 + jc_r2.z * motion_camera_raw.m32,
                jc_r2.x * motion_camera_raw.m13 + jc_r2.y * motion_camera_raw.m23 + jc_r2.z * motion_camera_raw.m33);
            motion_cov2d_raw = make_float3(
                dot(jcm_r1, jc_r1),
                dot(jcm_r1, jc_r2),
                dot(jcm_r2, jc_r2));
        }

        float defocus_radius_sq_raw = 0.0f;
        if (observation_blur_params != nullptr && observation_defocus_enabled) {
            const float beta = fmaxf(observation_blur_params[6], 0.0f);
            const float rho = observation_blur_params[7];
            const float q = rho - 1.0f / fmaxf(depth, 1e-4f);
            defocus_radius_sq_raw = beta * q * q;
            if (observation_max_defocus_radius_sq > 0.0f) {
                defocus_radius_sq_raw =
                    fminf(defocus_radius_sq_raw, observation_max_defocus_radius_sq);
            }
        }

        const float3 observation_cov2d_raw = make_float3(
            motion_cov2d_raw.x + defocus_radius_sq_raw,
            motion_cov2d_raw.y,
            motion_cov2d_raw.z + defocus_radius_sq_raw);
        const auto blur_cap = observation_blur_cap(
            observation_cov2d_raw, observation_max_radius_sq);
        const float observation_scale = blur_cap.scale;
        const mat3x3 motion_camera = {
            observation_scale * motion_camera_raw.m11,
            observation_scale * motion_camera_raw.m12,
            observation_scale * motion_camera_raw.m13,
            observation_scale * motion_camera_raw.m21,
            observation_scale * motion_camera_raw.m22,
            observation_scale * motion_camera_raw.m23,
            observation_scale * motion_camera_raw.m31,
            observation_scale * motion_camera_raw.m32,
            observation_scale * motion_camera_raw.m33};
        const float3 motion_cov2d = observation_scale * motion_cov2d_raw;
        defocus_radius_sq = observation_scale * defocus_radius_sq_raw;

        if (observation_blur_params != nullptr && observation_motion_enabled) {
            const mat3x3 cov_camera_blurred = {
                cov_camera.m11 + motion_camera.m11,
                cov_camera.m12 + motion_camera.m12,
                cov_camera.m13 + motion_camera.m13,
                cov_camera.m21 + motion_camera.m21,
                cov_camera.m22 + motion_camera.m22,
                cov_camera.m23 + motion_camera.m23,
                cov_camera.m31 + motion_camera.m31,
                cov_camera.m32 + motion_camera.m32,
                cov_camera.m33 + motion_camera.m33};
            const float det_clear = fmaxf(mat3_det(cov_camera), 1e-12f);
            const float det_blurred = fmaxf(mat3_det(cov_camera_blurred), det_clear);
            motion_opacity_compensation = sqrtf(det_clear / det_blurred);
            cov2d += motion_cov2d;
        }

        const float3 cov2d_motion = cov2d;
        if (observation_blur_params != nullptr && observation_defocus_enabled) {
            cov2d.x += defocus_radius_sq;
            cov2d.z += defocus_radius_sq;
            const float det_before = fmaxf(
                cov2d_motion.x * cov2d_motion.z - cov2d_motion.y * cov2d_motion.y,
                config::min_cov2d_determinant);
            const float det_after = fmaxf(
                cov2d.x * cov2d.z - cov2d.y * cov2d.y,
                det_before);
            defocus_opacity_compensation = sqrtf(det_before / det_after);
        }

        // Mip filter: use smaller dilation and compensate opacity
        const float det_raw = mip_filter ? fmaxf(cov2d.x * cov2d.z - cov2d.y * cov2d.y, 0.0f) : 0.0f;
        const float kernel_size = mip_filter ? config::dilation_mip_filter : config::dilation;
        cov2d.x += kernel_size;
        cov2d.z += kernel_size;
        const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (det < config::min_cov2d_determinant)
            active = false;
        const float det_rcp = 1.0f / det;
        const float blur_compensation = motion_opacity_compensation * defocus_opacity_compensation;
        const float output_opacity = mip_filter
                                         ? opacity * blur_compensation * sqrtf(det_raw * det_rcp)
                                         : opacity * blur_compensation;
        if (output_opacity < config::min_alpha_threshold)
            active = false;

        const float3 conic = make_float3(cov2d.z * det_rcp, -cov2d.y * det_rcp, cov2d.x * det_rcp);
        // GGGS depth rendering uses ray distance ordering during depth pass.
        const float depth_for_sort = require_depth ? ray_distance : depth;
        float4 ray_plane = make_float4(0.0f, 0.0f, ray_distance, 0.0f);
        float3 normal = make_float3(0.0f, 0.0f, -1.0f);
        if (require_depth) {
            auto rpn = compute_ray_plane_and_normal(
                rotation,
                variance,
                w2c_r1,
                w2c_r2,
                w2c_r3,
                tx,
                ty,
                depth,
                fx,
                fy,
                ray_distance);
            ray_plane = rpn.ray_plane;
            normal = rpn.normal;
        }

        // Compute bounds
        const float power_threshold = logf(output_opacity * config::min_alpha_threshold_rcp);
        const float power_threshold_factor = sqrtf(2.0f * power_threshold);
        float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
        float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))),   // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))),   // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height)))))  // y_max
        );
        const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles_max == 0)
            active = false;

        // early exit if whole warp is inactive
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // compute exact number of tiles the primitive overlaps
        const uint n_touched_tiles = compute_exact_n_touched_tiles(
            mean2d, conic, screen_bounds,
            power_threshold, n_touched_tiles_max, active);

        // cooperative threads no longer needed
        if (n_touched_tiles == 0 || !active)
            return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w));
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, output_opacity);
        primitive_color[primitive_idx] = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);
        primitive_depth[primitive_idx] = depth_for_sort;
        primitive_ray_plane[primitive_idx] = ray_plane;
        primitive_normal[primitive_idx] = normal;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth_for_sort);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, static_cast<unsigned long long>(n_touched_tiles));
    }

    __global__ void apply_depth_ordering_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_n_touched_tiles,
        uint* primitive_offset,
        const uint n_visible_primitives) {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= n_visible_primitives)
            return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        ushort* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives) {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32u>(block);
        uint idx = cg::this_grid().thread_rank();

        bool active = true;
        if (idx >= n_visible_primitives) {
            active = false;
            idx = n_visible_primitives - 1;
        }

        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        const uint primitive_idx = primitive_indices_sorted[idx];

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint tile_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_screen_bounds[block.thread_rank()] = screen_bounds;
        collected_mean2d_shifted[block.thread_rank()] = primitive_mean2d[primitive_idx] - 0.5f;
        const float4 conic_opacity_loaded = primitive_conic_opacity[primitive_idx];
        const float power_threshold_precomputed = logf(conic_opacity_loaded.w * config::min_alpha_threshold_rcp);
        collected_conic_power_threshold[block.thread_rank()] = make_float4(make_float3(conic_opacity_loaded), power_threshold_precomputed);

        uint current_write_offset = primitive_offsets[idx];

        if (active) {
            const float2 mean2d_shifted = collected_mean2d_shifted[block.thread_rank()];
            const float4 conic_pt = collected_conic_power_threshold[block.thread_rank()];
            const float3 conic = make_float3(conic_pt);
            const float power_threshold = conic_pt.w;

            for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
                const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
                const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
                if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                    instance_keys[current_write_offset] = tile_key;
                    instance_primitive_indices[current_write_offset] = primitive_idx;
                    current_write_offset++;
                }
            }
        }

        const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;
        const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;
        const uint lane_mask_allprev_excl = (1u << lane_idx) - 1u;
        const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
        if (remaining_threads == 0)
            return;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (int n = 0; n < n_remaining_threads && n < 32; n++) {
            int current_lane = __fns(remaining_threads, 0, n + 1);
            uint primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, current_lane);
            uint current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, current_lane);

            const ushort4 screen_bounds_coop = collected_screen_bounds[warp.meta_group_rank() * 32 + current_lane];
            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint tile_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);

            const float2 mean2d_shifted_coop = collected_mean2d_shifted[warp.meta_group_rank() * 32 + current_lane];
            const float4 conic_pt_coop = collected_conic_power_threshold[warp.meta_group_rank() * 32 + current_lane];
            const float3 conic_coop = make_float3(conic_pt_coop);
            const float power_threshold_coop = conic_pt_coop.w;

            const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
            const int n_iterations = div_round_up(remaining_tile_count, 32u);
            for (int i = 0; i < n_iterations; i++) {
                const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;
                const int active_current = instance_idx < tile_count_coop;
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint write = active_current && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = __ballot_sync(0xffffffffu, write);
                const uint n_writes = __popc(write_ballot);
                const uint write_offset_current = __popc(write_ballot & lane_mask_allprev_excl);
                const uint write_offset = current_write_offset_coop + write_offset_current;
                if (write) {
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                    instance_keys[write_offset] = tile_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                current_write_offset_coop += n_writes;
            }

            __syncwarp();
        }
    }

    __global__ void extract_instance_ranges_cu(
        const ushort* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances) {
        auto instance_idx = cg::this_grid().thread_rank();
        if (instance_idx >= n_instances)
            return;
        const ushort instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0)
            tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const ushort previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1)
            tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void extract_bucket_counts(
        uint2* tile_instance_ranges,
        uint* tile_n_buckets,
        const uint n_tiles) {
        auto tile_idx = cg::this_grid().thread_rank();
        if (tile_idx >= n_tiles)
            return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, static_cast<uint>(config::checkpoint_interval));
        tile_n_buckets[tile_idx] = n_buckets;
    }

    template <bool RENDER_DEPTH>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float4* __restrict__ primitive_ray_plane,
        const float3* __restrict__ primitive_normal,
        float* __restrict__ image,
        float* __restrict__ alpha_map,
        float* __restrict__ depth_map,
        float* __restrict__ normal_map,
        float* __restrict__ normal_accum_length_map, // [H*W] or nullptr: saves |normal_accum| for backward
        uint* __restrict__ tile_max_n_contributions,
        uint* __restrict__ tile_n_contributions,
        uint* __restrict__ bucket_tile_index,
        uint* __restrict__ bucket_checkpoint_uint8,
        uint* __restrict__ bucket_checkpoint_normal_uint8, // or nullptr: packed normal checkpoint for backward
        const uint width,
        const uint height,
        const uint grid_width,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;

        uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int n_buckets = div_round_up(n_points_total, config::checkpoint_interval); // re-computing is faster than reading from tile_n_buckets
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets)
                bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }

        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        __shared__ float4 collected_ray_plane[config::block_size_blend];
        __shared__ float3 collected_normal[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        float depth_init = 0.0f;
        float3 normal_accum = make_float3(0.0f, 0.0f, 0.0f);
        uint n_possible_contributions = 0;
        uint n_contributions = 0;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend)
                break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fminf(fmaxf(primitive_color[primitive_idx], 0.0f), config::max_checkpoint_color);
                collected_color[thread_rank] = color;
                if constexpr (RENDER_DEPTH) {
                    collected_ray_plane[thread_rank] = primitive_ray_plane[primitive_idx];
                    collected_normal[thread_rank] = primitive_normal[primitive_idx];
                }
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                if (j % config::checkpoint_interval == 0) {
                    constexpr float COLOR_SCALE = 255.0f / config::max_checkpoint_color;
                    const uint r = static_cast<uint>(color_pixel.x * COLOR_SCALE + 0.5f);
                    const uint g = static_cast<uint>(color_pixel.y * COLOR_SCALE + 0.5f);
                    const uint b = static_cast<uint>(color_pixel.z * COLOR_SCALE + 0.5f);
                    const uint t = min(static_cast<uint>(fmaxf(transmittance, 0.0f) * 255.0f + 0.5f), 255u);
                    bucket_checkpoint_uint8[bucket_offset * config::block_size_blend + thread_rank] = r | (g << 8) | (b << 16) | (t << 24);
                    // Write normal checkpoint if buffer is available (for GGGS normal backward)
                    if constexpr (RENDER_DEPTH) {
                        if (bucket_checkpoint_normal_uint8 != nullptr) {
                            // Map normal_accum from [-1, 1] to [0, 255] (saturate for safety)
                            constexpr float NORMAL_SCALE = 127.5f;
                            constexpr float NORMAL_BIAS = 127.5f;
                            const uint nx = min(static_cast<uint>(fmaxf(normal_accum.x * NORMAL_SCALE + NORMAL_BIAS, 0.0f) + 0.5f), 255u);
                            const uint ny = min(static_cast<uint>(fmaxf(normal_accum.y * NORMAL_SCALE + NORMAL_BIAS, 0.0f) + 0.5f), 255u);
                            const uint nz = min(static_cast<uint>(fmaxf(normal_accum.z * NORMAL_SCALE + NORMAL_BIAS, 0.0f) + 0.5f), 255u);
                            bucket_checkpoint_normal_uint8[bucket_offset * config::block_size_blend + thread_rank] = nx | (ny << 8) | (nz << 16);
                        }
                    }
                    bucket_offset++;
                }
                n_possible_contributions++;
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float2 delta = collected_mean2d[j] - pixel;
                const float opacity = conic_opacity.w;
                const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                if (sigma_over_2 < 0.0f)
                    continue;
                const float gaussian = expf(-sigma_over_2);
                const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold)
                    continue;

                if constexpr (RENDER_DEPTH) {
                    const float4 ray_plane = collected_ray_plane[j];
                    const float t_peak = ray_plane.x * delta.x + ray_plane.y * delta.y + ray_plane.z;
                    // GGGS MEDIAN_DEPTH_INIT: initialize around the T=0.5 crossing.
                    if (transmittance > 0.5f) {
                        depth_init = t_peak;
                    }
                    normal_accum += (transmittance * alpha) * collected_normal[j];
                }

                color_pixel += transmittance * alpha * collected_color[j];
                transmittance *= (1.0f - alpha);
                n_contributions = n_possible_contributions;
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }

        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            // store results
            image[pixel_idx] = color_pixel.x;
            image[pixel_idx + n_pixels] = color_pixel.y;
            image[pixel_idx + n_pixels * 2] = color_pixel.z;
            alpha_map[pixel_idx] = 1.0f - transmittance;
            if constexpr (RENDER_DEPTH) {
                // Write depth_init for the separate refinement kernel.
                // Pixels with no contributions or T never crossing 0.5 get depth_init=0.
                depth_map[pixel_idx] = (n_contributions > 0 && transmittance <= config::depth_min_transmittance) ? depth_init : 0.0f;
                if (normal_map != nullptr) {
                    const float nlen = norm3df(normal_accum.x, normal_accum.y, normal_accum.z);
                    const float inv_nlen = (nlen > 1e-8f) ? (1.0f / nlen) : 0.0f;
                    normal_map[pixel_idx] = normal_accum.x * inv_nlen;
                    normal_map[pixel_idx + n_pixels] = normal_accum.y * inv_nlen;
                    normal_map[pixel_idx + n_pixels * 2] = normal_accum.z * inv_nlen;
                    // Save |normal_accum| for normalization backward (GGGS losses)
                    if (normal_accum_length_map != nullptr) {
                        normal_accum_length_map[pixel_idx] = nlen;
                    }
                }
            }
            tile_n_contributions[pixel_idx] = n_contributions;
        }

        // max reduce the number of contributions
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        n_contributions = BlockReduce(temp_storage).Reduce(n_contributions, thrust::maximum<uint>());
        if (thread_rank == 0)
            tile_max_n_contributions[tile_idx] = n_contributions;
    }

    // Separate depth refinement kernel: iteratively brackets the ray-distance
    // where transmittance crosses 0.5 (GGGS median-depth).  Reads depth_init
    // written by blend_cu<true> and overwrites depth_map with refined values.
    //
    // Key optimizations over the previous inline approach:
    //   1. Shared-memory cooperative loading of xy / conic_opacity / ray_plane
    //      (eliminates redundant global-memory traffic across the 5×8 samples).
    //   2. Per-pixel early-out via tile_n_contributions[] – once a pixel has
    //      seen all of its contributing Gaussians it stops iterating.
    __global__ void __launch_bounds__(config::block_size_blend) depth_refine_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float4* __restrict__ primitive_ray_plane,
        const uint* __restrict__ tile_n_contributions, // per-pixel count from blend_cu
        float* __restrict__ depth_map,                 // IN: depth_init, OUT: refined depth
        const uint width,
        const uint height,
        const uint grid_width,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width + thread_index.x,
            group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;

        // Per-pixel state
        const int pixel_idx = inside ? static_cast<int>(width * pixel_coords.y + pixel_coords.x) : 0;
        const float depth_init = inside ? depth_map[pixel_idx] : 0.0f;
        const uint my_n_contributions = inside ? tile_n_contributions[pixel_idx] : 0;
        // Skip refinement for pixels with no valid depth_init
        const bool needs_refine = inside && depth_init > 0.0f && my_n_contributions > 0;

        float depth_min = fmaxf(depth_init - config::depth_sample_range, 0.0f);
        float depth_max = fmaxf(depth_init + config::depth_sample_range, 0.0f);
        float T_p[config::depth_split + 1];
        bool in_range = needs_refine;

        // Shared memory for cooperative loading
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float4 collected_ray_plane[config::block_size_blend];

        for (int iteration = 0; iteration < config::depth_split_iterations; ++iteration) {
            const int sample_start = (iteration == 0) ? 0 : 1;
            const int sample_end = (iteration == 0) ? config::depth_split : (config::depth_split - 1);

            for (int s = sample_start; s <= sample_end; ++s)
                T_p[s] = 1.0f;

            const float interval = (depth_max - depth_min) / static_cast<float>(config::depth_split);
            uint contributor = 0;
            bool done = !in_range;

            for (int n_points_remaining = n_points_total,
                     current_fetch_idx = tile_range.x + thread_rank;
                 n_points_remaining > 0;
                 n_points_remaining -= config::block_size_blend,
                     current_fetch_idx += config::block_size_blend) {
                // Early exit if all threads in this tile are done
                if (__syncthreads_count(done) == config::block_size_blend)
                    break;

                // Cooperative load into shared memory
                if (current_fetch_idx < tile_range.y) {
                    const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                    collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                    collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                    collected_ray_plane[thread_rank] = primitive_ray_plane[primitive_idx];
                }
                block.sync();

                const int current_batch_size = min(config::block_size_blend, n_points_remaining);
                for (int j = 0; !done && j < current_batch_size; ++j) {
                    // Count ALL Gaussians iterated (matching GGGS contributor counter)
                    contributor++;
                    done = contributor >= my_n_contributions;

                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float2 delta = collected_mean2d[j] - pixel;
                    const float opacity = conic_opacity.w;
                    const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                    if (sigma_over_2 < 0.0f)
                        continue;
                    const float gaussian = expf(-sigma_over_2);
                    const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                    if (alpha < config::min_alpha_threshold)
                        continue;

                    const float4 ray_plane = collected_ray_plane[j];
                    const float t_peak = ray_plane.x * delta.x + ray_plane.y * delta.y + ray_plane.z;
                    const float rsigma = ray_plane.w;
                    const bool ball = rsigma > 0.0f;

                    for (int s = sample_start; s <= sample_end; ++s) {
                        const float ts = depth_min + interval * static_cast<float>(s);
                        const float delta_sample = (ts - t_peak) * rsigma;
                        const float g = ball ? expf(-0.5f * delta_sample * delta_sample) : 0.0f;
                        const float one_minus_gaussian = fmaxf(1.0f - alpha * g, 1e-6f);
                        const float rvacancy = rsqrtf(one_minus_gaussian);
                        T_p[s] *= (ts > t_peak ? (1.0f - alpha) : one_minus_gaussian) * rvacancy;
                    }
                }
            }

            if (iteration == 0) {
                in_range = in_range && (T_p[0] >= 0.5f) && (T_p[config::depth_split] <= 0.5f);
            }

            int start_id = 0;
            for (int p = 1; p < config::depth_split; ++p) {
                start_id = (T_p[p] >= 0.5f) ? p : start_id;
            }

            depth_max = depth_min + static_cast<float>(start_id + 1) * interval;
            depth_min = depth_min + static_cast<float>(start_id) * interval;
            T_p[0] = T_p[start_id];
            T_p[config::depth_split] = T_p[start_id + 1];
        }

        if (inside) {
            float depth_pixel = 0.0f;
            if (in_range) {
                const float denom = T_p[0] - T_p[config::depth_split];
                const float w_max = (fabsf(denom) > 1e-8f)
                                        ? __saturatef((T_p[0] - 0.5f) / denom)
                                        : 0.5f;
                const float mdepth = w_max * depth_max + (1.0f - w_max) * depth_min;
                const float pix_nf_x = (pixel.x - cx) / fx;
                const float pix_nf_y = (pixel.y - cy) / fy;
                const float rln = rsqrtf(pix_nf_x * pix_nf_x + pix_nf_y * pix_nf_y + 1.0f);
                depth_pixel = mdepth * rln;
            }
            depth_map[pixel_idx] = depth_pixel;
        }
    }

} // namespace fast_lfs::rasterization::kernels::forward
