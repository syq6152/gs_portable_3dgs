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

namespace fast_lfs::rasterization::kernels::backward {

    // Gradient clamping to prevent NaN from exploding gradients
    constexpr float GRAD_CLAMP_MAX = 1e4f;

    __device__ inline float clamp_grad(const float g) {
        return fminf(fmaxf(g, -GRAD_CLAMP_MAX), GRAD_CLAMP_MAX);
    }

    __device__ inline float3 clamp_grad3(const float3 g) {
        return make_float3(clamp_grad(g.x), clamp_grad(g.y), clamp_grad(g.z));
    }

    __device__ inline float4 clamp_grad4(const float4 g) {
        return make_float4(clamp_grad(g.x), clamp_grad(g.y), clamp_grad(g.z), clamp_grad(g.w));
    }

    __device__ inline mat3x3 blur_full_from_triu(const mat3x3_triu& a) {
        return {a.m11, a.m12, a.m13,
                a.m12, a.m22, a.m23,
                a.m13, a.m23, a.m33};
    }

    __device__ inline mat3x3 blur_mat3_mul(const mat3x3& a, const mat3x3& b) {
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

    __device__ inline mat3x3 blur_mat3_transpose(const mat3x3& a) {
        return {a.m11, a.m21, a.m31,
                a.m12, a.m22, a.m32,
                a.m13, a.m23, a.m33};
    }

    __device__ inline float blur_mat3_det(const mat3x3& a) {
        return a.m11 * (a.m22 * a.m33 - a.m23 * a.m32) -
               a.m12 * (a.m21 * a.m33 - a.m23 * a.m31) +
               a.m13 * (a.m21 * a.m32 - a.m22 * a.m31);
    }

    __device__ inline mat3x3 blur_mat3_inverse(const mat3x3& a, const float det) {
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

    __device__ inline mat3x3 blur_motion_covariance_camera(
        const float3& p,
        const float3& rotation_variance,
        const float3& translation_variance) {
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

    __device__ inline mat3x3 blur_add_mat3(const mat3x3& a, const mat3x3& b) {
        return {a.m11 + b.m11, a.m12 + b.m12, a.m13 + b.m13,
                a.m21 + b.m21, a.m22 + b.m22, a.m23 + b.m23,
                a.m31 + b.m31, a.m32 + b.m32, a.m33 + b.m33};
    }

    __device__ inline void accumulate_w2c_from_camera_space_point(
        float4* __restrict__ grad_w2c,
        const float3& mean3d,
        const float3& dL_dmean_cam) {
        atomicAdd(&grad_w2c[0].w, dL_dmean_cam.x);
        atomicAdd(&grad_w2c[1].w, dL_dmean_cam.y);
        atomicAdd(&grad_w2c[2].w, dL_dmean_cam.z);
        atomicAdd(&grad_w2c[0].x, dL_dmean_cam.x * mean3d.x);
        atomicAdd(&grad_w2c[0].y, dL_dmean_cam.x * mean3d.y);
        atomicAdd(&grad_w2c[0].z, dL_dmean_cam.x * mean3d.z);
        atomicAdd(&grad_w2c[1].x, dL_dmean_cam.y * mean3d.x);
        atomicAdd(&grad_w2c[1].y, dL_dmean_cam.y * mean3d.y);
        atomicAdd(&grad_w2c[1].z, dL_dmean_cam.y * mean3d.z);
        atomicAdd(&grad_w2c[2].x, dL_dmean_cam.z * mean3d.x);
        atomicAdd(&grad_w2c[2].y, dL_dmean_cam.z * mean3d.y);
        atomicAdd(&grad_w2c[2].z, dL_dmean_cam.z * mean3d.z);
    }

    __device__ inline void accumulate_w2c_rotation_rows(
        float4* __restrict__ grad_w2c,
        const float3& dL_dr0,
        const float3& dL_dr1,
        const float3& dL_dr2) {
        atomicAdd(&grad_w2c[0].x, dL_dr0.x);
        atomicAdd(&grad_w2c[0].y, dL_dr0.y);
        atomicAdd(&grad_w2c[0].z, dL_dr0.z);
        atomicAdd(&grad_w2c[1].x, dL_dr1.x);
        atomicAdd(&grad_w2c[1].y, dL_dr1.y);
        atomicAdd(&grad_w2c[1].z, dL_dr1.z);
        atomicAdd(&grad_w2c[2].x, dL_dr2.x);
        atomicAdd(&grad_w2c[2].y, dL_dr2.y);
        atomicAdd(&grad_w2c[2].z, dL_dr2.z);
    }

    __device__ inline void accumulate_w2c_from_camera_center(
        float4* __restrict__ grad_w2c,
        const float4& w2c_r1,
        const float4& w2c_r2,
        const float4& w2c_r3,
        const float3& dL_dcamera_center) {
        const float3 r0 = make_float3(w2c_r1.x, w2c_r1.y, w2c_r1.z);
        const float3 r1 = make_float3(w2c_r2.x, w2c_r2.y, w2c_r2.z);
        const float3 r2 = make_float3(w2c_r3.x, w2c_r3.y, w2c_r3.z);

        atomicAdd(&grad_w2c[0].w, -dot(r0, dL_dcamera_center));
        atomicAdd(&grad_w2c[1].w, -dot(r1, dL_dcamera_center));
        atomicAdd(&grad_w2c[2].w, -dot(r2, dL_dcamera_center));

        atomicAdd(&grad_w2c[0].x, -w2c_r1.w * dL_dcamera_center.x);
        atomicAdd(&grad_w2c[0].y, -w2c_r1.w * dL_dcamera_center.y);
        atomicAdd(&grad_w2c[0].z, -w2c_r1.w * dL_dcamera_center.z);
        atomicAdd(&grad_w2c[1].x, -w2c_r2.w * dL_dcamera_center.x);
        atomicAdd(&grad_w2c[1].y, -w2c_r2.w * dL_dcamera_center.y);
        atomicAdd(&grad_w2c[1].z, -w2c_r2.w * dL_dcamera_center.z);
        atomicAdd(&grad_w2c[2].x, -w2c_r3.w * dL_dcamera_center.x);
        atomicAdd(&grad_w2c[2].y, -w2c_r3.w * dL_dcamera_center.y);
        atomicAdd(&grad_w2c[2].z, -w2c_r3.w * dL_dcamera_center.z);
    }

    /**
     * @brief Per-pixel normalization backward: grad_render_normal → grad_normal_accum.
     *
     * render_normal = normal_accum / |normal_accum|, so:
     * grad_normal_accum_i = (grad_render_normal_i - render_normal_i * dot(grad, render_normal)) / nlen
     *
     * Output layout: [3, H, W] matching input layout.
     */
    __global__ void normalize_normal_backward_cu(
        const float* __restrict__ grad_render_normal, // [3, H, W]
        const float* __restrict__ render_normal,       // [3, H, W]
        const float* __restrict__ normal_accum_length_map, // [H, W]
        float* __restrict__ grad_normal_accum,         // [3, H, W] output
        const int n_pixels) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_pixels) return;

        const float nlen = normal_accum_length_map[idx];
        if (nlen < 1e-8f) {
            grad_normal_accum[idx] = 0.0f;
            grad_normal_accum[idx + n_pixels] = 0.0f;
            grad_normal_accum[idx + 2 * n_pixels] = 0.0f;
            return;
        }

        const float gx = grad_render_normal[idx];
        const float gy = grad_render_normal[idx + n_pixels];
        const float gz = grad_render_normal[idx + 2 * n_pixels];
        const float nx = render_normal[idx];
        const float ny = render_normal[idx + n_pixels];
        const float nz = render_normal[idx + 2 * n_pixels];

        const float proj = gx * nx + gy * ny + gz * nz;
        const float inv_nlen = 1.0f / nlen;

        grad_normal_accum[idx] = (gx - nx * proj) * inv_nlen;
        grad_normal_accum[idx + n_pixels] = (gy - ny * proj) * inv_nlen;
        grad_normal_accum[idx + 2 * n_pixels] = (gz - nz * proj) * inv_nlen;
    }

    __global__ void preprocess_backward_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float* __restrict__ raw_opacities,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        float3* __restrict__ grad_means,
        float3* __restrict__ grad_raw_scales,
        float4* __restrict__ grad_raw_rotations,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
        float4* __restrict__ grad_w2c,
        float* __restrict__ densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const bool mip_filter,
        const float* __restrict__ observation_blur_params,
        float* __restrict__ observation_blur_gradients,
        const float* __restrict__ grad_compensated_opacity,
        const bool observation_motion_enabled,
        const bool observation_defocus_enabled,
        const float observation_max_defocus_radius_sq,
        const float observation_max_radius_sq) {
        auto primitive_idx = cg::this_grid().thread_rank();
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        float3 dL_dmean3d_from_color = make_float3(0.0f, 0.0f, 0.0f);
        if (sh_coefficients_rest != nullptr &&
            cam_position != nullptr &&
            grad_sh_coefficients_0 != nullptr &&
            grad_sh_coefficients_rest != nullptr) {
            dL_dmean3d_from_color = convert_sh_to_color_backward(
                sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
                mean3d, cam_position[0],
                primitive_idx, active_sh_bases, total_bases_sh_rest);
        }

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float depth_safe = fmaxf(depth, 1e-4f);
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth_safe;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth_safe;

        if (grad_w2c != nullptr) {
            accumulate_w2c_from_camera_center(
                grad_w2c,
                w2c_r1,
                w2c_r2,
                w2c_r3,
                make_float3(
                    -dL_dmean3d_from_color.x,
                    -dL_dmean3d_from_color.y,
                    -dL_dmean3d_from_color.z));
        }

        // compute 3d covariance from raw scale and rotation
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-7f);
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

        const bool observation_blur_enabled =
            observation_blur_params != nullptr &&
            (observation_motion_enabled || observation_defocus_enabled);
        const float3 p_camera = make_float3(x * depth_safe, y * depth_safe, depth_safe);
        const mat3x3 rotation_wc = {
            w2c_r1.x, w2c_r1.y, w2c_r1.z,
            w2c_r2.x, w2c_r2.y, w2c_r2.z,
            w2c_r3.x, w2c_r3.y, w2c_r3.z};
        const mat3x3 rotation_cw = blur_mat3_transpose(rotation_wc);
        const mat3x3 cov_world_full = blur_full_from_triu(cov3d);
        const mat3x3 cov_camera = blur_mat3_mul(
            blur_mat3_mul(rotation_wc, cov_world_full), rotation_cw);
        mat3x3 motion_camera_raw = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f};
        if (observation_blur_enabled && observation_motion_enabled) {
            motion_camera_raw = blur_motion_covariance_camera(
                p_camera,
                make_float3(observation_blur_params[0], observation_blur_params[1], observation_blur_params[2]),
                make_float3(observation_blur_params[3], observation_blur_params[4], observation_blur_params[5]));
        }

        // ewa splatting gradient helpers
        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float j11 = fx / depth_safe;
        const float j13 = -j11 * tx;
        const float j22 = fy / depth_safe;
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

        // Recompute the exact capped observation covariance used in forward.
        // The no-blur branch deliberately retains the original arithmetic.
        const float3 cov2d_clear = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2));
        const float3 jc_r1 = make_float3(j11, 0.0f, j13);
        const float3 jc_r2 = make_float3(0.0f, j22, j23);
        float3 motion_cov2d_raw = make_float3(0.0f, 0.0f, 0.0f);
        if (observation_blur_enabled && observation_motion_enabled) {
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
        float defocus_q = 0.0f;
        float defocus_beta = 0.0f;
        bool defocus_radius_saturated = false;
        if (observation_blur_enabled && observation_defocus_enabled) {
            defocus_beta = fmaxf(observation_blur_params[6], 0.0f);
            defocus_q = observation_blur_params[7] - 1.0f / depth_safe;
            const float defocus_radius_sq_unclamped =
                defocus_beta * defocus_q * defocus_q;
            defocus_radius_sq_raw = defocus_radius_sq_unclamped;
            if (observation_max_defocus_radius_sq > 0.0f &&
                defocus_radius_sq_unclamped > observation_max_defocus_radius_sq) {
                defocus_radius_sq_raw = observation_max_defocus_radius_sq;
                defocus_radius_saturated = true;
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
        const float defocus_radius_sq =
            observation_scale * defocus_radius_sq_raw;

        const mat3x3 cov_camera_motion = blur_add_mat3(cov_camera, motion_camera);
        const float det_cov_camera_raw = blur_mat3_det(cov_camera);
        const bool det_cov_camera_active = det_cov_camera_raw > 1e-12f;
        const float det_cov_camera = fmaxf(det_cov_camera_raw, 1e-12f);
        const float det_cov_camera_motion_raw = blur_mat3_det(cov_camera_motion);
        const bool det_cov_camera_motion_active =
            det_cov_camera_motion_raw > det_cov_camera;
        const float det_cov_camera_motion = fmaxf(
            det_cov_camera_motion_raw, det_cov_camera);
        const float motion_compensation =
            (observation_blur_enabled && observation_motion_enabled)
                ? sqrtf(det_cov_camera / det_cov_camera_motion)
                : 1.0f;

        const float3 cov2d_motion = cov2d_clear + motion_cov2d;
        const float det_cov2d_motion_raw =
            cov2d_motion.x * cov2d_motion.z -
            cov2d_motion.y * cov2d_motion.y;
        const bool det_cov2d_motion_active =
            det_cov2d_motion_raw > config::min_cov2d_determinant;
        const float det_cov2d_motion = fmaxf(
            det_cov2d_motion_raw, config::min_cov2d_determinant);
        const float3 cov2d_defocus = make_float3(
            cov2d_motion.x + defocus_radius_sq,
            cov2d_motion.y,
            cov2d_motion.z + defocus_radius_sq);
        const float det_cov2d_defocus_raw =
            cov2d_defocus.x * cov2d_defocus.z -
            cov2d_defocus.y * cov2d_defocus.y;
        const bool det_cov2d_defocus_active =
            det_cov2d_defocus_raw > det_cov2d_motion;
        const float det_cov2d_defocus = fmaxf(
            det_cov2d_defocus_raw, det_cov2d_motion);
        const float defocus_compensation =
            (observation_blur_enabled && observation_defocus_enabled)
                ? sqrtf(det_cov2d_motion / det_cov2d_defocus)
                : 1.0f;

        // 2d covariance gradient (use same dilation as forward pass)
        const float kernel_size = mip_filter ? config::dilation_mip_filter : config::dilation;
        const float a = cov2d_defocus.x + kernel_size;
        const float b = cov2d_defocus.y;
        const float c = cov2d_defocus.z + kernel_size;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float determinant_safe = fmaxf(determinant, config::min_cov2d_determinant);
        const float determinant_rcp = 1.0f / determinant_safe;
        const float determinant_rcp_sq = determinant_rcp * determinant_rcp;
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]);
        const float3 dL_dcov2d_defocus = determinant_rcp_sq * make_float3(
            2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
            bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
            2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

        const float original_opacity = __frcp_rn(
            1.0f + __expf(-raw_opacities[primitive_idx]));
        const float dL_dtotal_compensation = observation_blur_enabled
                                                ? grad_compensated_opacity[primitive_idx] * original_opacity
                                                : 0.0f;

        // c_db = sqrt(det(S_mb) / det(S_mb + sI)).  Its covariance
        // derivative is merged with the conic path before projecting back to
        // the canonical 3D covariance and the motion covariance.
        float3 dL_dcov2d_motion = dL_dcov2d_defocus;
        float dL_ddefocus_radius_sq =
            dL_dcov2d_defocus.x + dL_dcov2d_defocus.z;
        if (observation_blur_enabled && observation_defocus_enabled &&
            det_cov2d_defocus_active) {
            float3 inv_motion = make_float3(0.0f, 0.0f, 0.0f);
            if (det_cov2d_motion_active) {
                const float inv_det_motion = 1.0f / det_cov2d_motion_raw;
                inv_motion = make_float3(
                    cov2d_motion.z * inv_det_motion,
                    -cov2d_motion.y * inv_det_motion,
                    cov2d_motion.x * inv_det_motion);
            }
            const float inv_det_defocus = 1.0f / det_cov2d_defocus_raw;
            const float3 inv_defocus = make_float3(
                cov2d_defocus.z * inv_det_defocus,
                -cov2d_defocus.y * inv_det_defocus,
                cov2d_defocus.x * inv_det_defocus);
            const float compensation_scale =
                dL_dtotal_compensation * motion_compensation * defocus_compensation;
            dL_dcov2d_motion += compensation_scale * make_float3(
                0.5f * (inv_motion.x - inv_defocus.x),
                0.5f * (inv_motion.y - inv_defocus.y),
                0.5f * (inv_motion.z - inv_defocus.z));
            dL_ddefocus_radius_sq += compensation_scale *
                (-0.5f * (inv_defocus.x + inv_defocus.z));
        }

        // 3d covariance gradient
        mat3x3_triu dL_dcov3d = {
            (jw_r1.x * jw_r1.x) * dL_dcov2d_motion.x + 2.0f * (jw_r1.x * jw_r2.x) * dL_dcov2d_motion.y + (jw_r2.x * jw_r2.x) * dL_dcov2d_motion.z,
            (jw_r1.x * jw_r1.y) * dL_dcov2d_motion.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d_motion.y + (jw_r2.x * jw_r2.y) * dL_dcov2d_motion.z,
            (jw_r1.x * jw_r1.z) * dL_dcov2d_motion.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d_motion.y + (jw_r2.x * jw_r2.z) * dL_dcov2d_motion.z,
            (jw_r1.y * jw_r1.y) * dL_dcov2d_motion.x + 2.0f * (jw_r1.y * jw_r2.y) * dL_dcov2d_motion.y + (jw_r2.y * jw_r2.y) * dL_dcov2d_motion.z,
            (jw_r1.y * jw_r1.z) * dL_dcov2d_motion.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d_motion.y + (jw_r2.y * jw_r2.z) * dL_dcov2d_motion.z,
            (jw_r1.z * jw_r1.z) * dL_dcov2d_motion.x + 2.0f * (jw_r1.z * jw_r2.z) * dL_dcov2d_motion.y + (jw_r2.z * jw_r2.z) * dL_dcov2d_motion.z,
        };

        float3 dL_dp_motion = make_float3(0.0f, 0.0f, 0.0f);
        float3 dL_djc_motion_r1 = make_float3(0.0f, 0.0f, 0.0f);
        float3 dL_djc_motion_r2 = make_float3(0.0f, 0.0f, 0.0f);
        const float ga = dL_dcov2d_motion.x;
        const float gb = dL_dcov2d_motion.y;
        const float gc = dL_dcov2d_motion.z;
        mat3x3 dL_dmotion_camera_effective = {
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f};
        float dL_ddefocus_radius_sq_raw =
            observation_scale * dL_ddefocus_radius_sq;
        if (observation_blur_enabled && observation_motion_enabled) {
            dL_dmotion_camera_effective = {
                ga * jc_r1.x * jc_r1.x + 2.0f * gb * jc_r1.x * jc_r2.x + gc * jc_r2.x * jc_r2.x,
                ga * jc_r1.x * jc_r1.y + gb * (jc_r1.x * jc_r2.y + jc_r2.x * jc_r1.y) + gc * jc_r2.x * jc_r2.y,
                ga * jc_r1.x * jc_r1.z + gb * (jc_r1.x * jc_r2.z + jc_r2.x * jc_r1.z) + gc * jc_r2.x * jc_r2.z,
                ga * jc_r1.y * jc_r1.x + gb * (jc_r1.y * jc_r2.x + jc_r2.y * jc_r1.x) + gc * jc_r2.y * jc_r2.x,
                ga * jc_r1.y * jc_r1.y + 2.0f * gb * jc_r1.y * jc_r2.y + gc * jc_r2.y * jc_r2.y,
                ga * jc_r1.y * jc_r1.z + gb * (jc_r1.y * jc_r2.z + jc_r2.y * jc_r1.z) + gc * jc_r2.y * jc_r2.z,
                ga * jc_r1.z * jc_r1.x + gb * (jc_r1.z * jc_r2.x + jc_r2.z * jc_r1.x) + gc * jc_r2.z * jc_r2.x,
                ga * jc_r1.z * jc_r1.y + gb * (jc_r1.z * jc_r2.y + jc_r2.z * jc_r1.y) + gc * jc_r2.z * jc_r2.y,
                ga * jc_r1.z * jc_r1.z + 2.0f * gb * jc_r1.z * jc_r2.z + gc * jc_r2.z * jc_r2.z};

            // Motion determinant compensation.  dL/dC is added to the
            // canonical covariance chain; dL/dB is merged with the projected
            // covariance path before differentiating B(p, vR, vt).
            const float compensation_scale =
                dL_dtotal_compensation * motion_compensation * defocus_compensation;
            if (det_cov_camera_motion_active) {
                mat3x3 inv_cov_camera = {
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f};
                if (det_cov_camera_active) {
                    inv_cov_camera =
                        blur_mat3_inverse(cov_camera, det_cov_camera_raw);
                }
                const mat3x3 inv_cov_camera_motion =
                    blur_mat3_inverse(
                        cov_camera_motion, det_cov_camera_motion_raw);
                const mat3x3 dL_dcov_camera_compensation = {
                    0.5f * compensation_scale * (inv_cov_camera.m11 - inv_cov_camera_motion.m11),
                    0.5f * compensation_scale * (inv_cov_camera.m12 - inv_cov_camera_motion.m12),
                    0.5f * compensation_scale * (inv_cov_camera.m13 - inv_cov_camera_motion.m13),
                    0.5f * compensation_scale * (inv_cov_camera.m21 - inv_cov_camera_motion.m21),
                    0.5f * compensation_scale * (inv_cov_camera.m22 - inv_cov_camera_motion.m22),
                    0.5f * compensation_scale * (inv_cov_camera.m23 - inv_cov_camera_motion.m23),
                    0.5f * compensation_scale * (inv_cov_camera.m31 - inv_cov_camera_motion.m31),
                    0.5f * compensation_scale * (inv_cov_camera.m32 - inv_cov_camera_motion.m32),
                    0.5f * compensation_scale * (inv_cov_camera.m33 - inv_cov_camera_motion.m33)};
                const mat3x3 dL_dcov_world_compensation = blur_mat3_mul(
                    blur_mat3_mul(rotation_cw, dL_dcov_camera_compensation),
                    rotation_wc);
                if (grad_w2c != nullptr) {
                    const mat3x3 dL_drotation_wc_compensation = blur_mat3_mul(
                        blur_mat3_mul(dL_dcov_camera_compensation, rotation_wc),
                        cov_world_full);
                    accumulate_w2c_rotation_rows(
                        grad_w2c,
                        2.0f * make_float3(
                                   dL_drotation_wc_compensation.m11,
                                   dL_drotation_wc_compensation.m12,
                                   dL_drotation_wc_compensation.m13),
                        2.0f * make_float3(
                                   dL_drotation_wc_compensation.m21,
                                   dL_drotation_wc_compensation.m22,
                                   dL_drotation_wc_compensation.m23),
                        2.0f * make_float3(
                                   dL_drotation_wc_compensation.m31,
                                   dL_drotation_wc_compensation.m32,
                                   dL_drotation_wc_compensation.m33));
                }
                dL_dcov3d.m11 += dL_dcov_world_compensation.m11;
                dL_dcov3d.m12 += dL_dcov_world_compensation.m12;
                dL_dcov3d.m13 += dL_dcov_world_compensation.m13;
                dL_dcov3d.m22 += dL_dcov_world_compensation.m22;
                dL_dcov3d.m23 += dL_dcov_world_compensation.m23;
                dL_dcov3d.m33 += dL_dcov_world_compensation.m33;

                dL_dmotion_camera_effective.m11 -= 0.5f * compensation_scale * inv_cov_camera_motion.m11;
                dL_dmotion_camera_effective.m12 -= 0.5f * compensation_scale * inv_cov_camera_motion.m12;
                dL_dmotion_camera_effective.m13 -= 0.5f * compensation_scale * inv_cov_camera_motion.m13;
                dL_dmotion_camera_effective.m21 -= 0.5f * compensation_scale * inv_cov_camera_motion.m21;
                dL_dmotion_camera_effective.m22 -= 0.5f * compensation_scale * inv_cov_camera_motion.m22;
                dL_dmotion_camera_effective.m23 -= 0.5f * compensation_scale * inv_cov_camera_motion.m23;
                dL_dmotion_camera_effective.m31 -= 0.5f * compensation_scale * inv_cov_camera_motion.m31;
                dL_dmotion_camera_effective.m32 -= 0.5f * compensation_scale * inv_cov_camera_motion.m32;
                dL_dmotion_camera_effective.m33 -= 0.5f * compensation_scale * inv_cov_camera_motion.m33;
            }

            // Exact VJP of the shared observation cap.  The direct branch
            // differentiates B_hat=gamma*B and s_hat=gamma*s; when saturated,
            // gamma also depends on lambda_max(J*B*J^T+sI).
            const float dL_dobservation_scale =
                dL_dmotion_camera_effective.m11 * motion_camera_raw.m11 +
                dL_dmotion_camera_effective.m12 * motion_camera_raw.m12 +
                dL_dmotion_camera_effective.m13 * motion_camera_raw.m13 +
                dL_dmotion_camera_effective.m21 * motion_camera_raw.m21 +
                dL_dmotion_camera_effective.m22 * motion_camera_raw.m22 +
                dL_dmotion_camera_effective.m23 * motion_camera_raw.m23 +
                dL_dmotion_camera_effective.m31 * motion_camera_raw.m31 +
                dL_dmotion_camera_effective.m32 * motion_camera_raw.m32 +
                dL_dmotion_camera_effective.m33 * motion_camera_raw.m33 +
                dL_ddefocus_radius_sq * defocus_radius_sq_raw;
            float3 dL_dobservation_cov2d_raw = make_float3(0.0f, 0.0f, 0.0f);
            if (blur_cap.saturated) {
                const float cap_scale =
                    -dL_dobservation_scale * observation_scale /
                    blur_cap.largest_eigenvalue;
                dL_dobservation_cov2d_raw =
                    cap_scale * blur_cap.eigen_projector;
                dL_ddefocus_radius_sq_raw +=
                    dL_dobservation_cov2d_raw.x +
                    dL_dobservation_cov2d_raw.z;
            }

            const float cap_ga = dL_dobservation_cov2d_raw.x;
            const float cap_gb = dL_dobservation_cov2d_raw.y;
            const float cap_gc = dL_dobservation_cov2d_raw.z;
            const mat3x3 dL_dmotion_camera_cap = {
                cap_ga * jc_r1.x * jc_r1.x + 2.0f * cap_gb * jc_r1.x * jc_r2.x + cap_gc * jc_r2.x * jc_r2.x,
                cap_ga * jc_r1.x * jc_r1.y + cap_gb * (jc_r1.x * jc_r2.y + jc_r2.x * jc_r1.y) + cap_gc * jc_r2.x * jc_r2.y,
                cap_ga * jc_r1.x * jc_r1.z + cap_gb * (jc_r1.x * jc_r2.z + jc_r2.x * jc_r1.z) + cap_gc * jc_r2.x * jc_r2.z,
                cap_ga * jc_r1.y * jc_r1.x + cap_gb * (jc_r1.y * jc_r2.x + jc_r2.y * jc_r1.x) + cap_gc * jc_r2.y * jc_r2.x,
                cap_ga * jc_r1.y * jc_r1.y + 2.0f * cap_gb * jc_r1.y * jc_r2.y + cap_gc * jc_r2.y * jc_r2.y,
                cap_ga * jc_r1.y * jc_r1.z + cap_gb * (jc_r1.y * jc_r2.z + jc_r2.y * jc_r1.z) + cap_gc * jc_r2.y * jc_r2.z,
                cap_ga * jc_r1.z * jc_r1.x + cap_gb * (jc_r1.z * jc_r2.x + jc_r2.z * jc_r1.x) + cap_gc * jc_r2.z * jc_r2.x,
                cap_ga * jc_r1.z * jc_r1.y + cap_gb * (jc_r1.z * jc_r2.y + jc_r2.z * jc_r1.y) + cap_gc * jc_r2.z * jc_r2.y,
                cap_ga * jc_r1.z * jc_r1.z + 2.0f * cap_gb * jc_r1.z * jc_r2.z + cap_gc * jc_r2.z * jc_r2.z};
            const mat3x3 dL_dmotion_camera = {
                observation_scale * dL_dmotion_camera_effective.m11 + dL_dmotion_camera_cap.m11,
                observation_scale * dL_dmotion_camera_effective.m12 + dL_dmotion_camera_cap.m12,
                observation_scale * dL_dmotion_camera_effective.m13 + dL_dmotion_camera_cap.m13,
                observation_scale * dL_dmotion_camera_effective.m21 + dL_dmotion_camera_cap.m21,
                observation_scale * dL_dmotion_camera_effective.m22 + dL_dmotion_camera_cap.m22,
                observation_scale * dL_dmotion_camera_effective.m23 + dL_dmotion_camera_cap.m23,
                observation_scale * dL_dmotion_camera_effective.m31 + dL_dmotion_camera_cap.m31,
                observation_scale * dL_dmotion_camera_effective.m32 + dL_dmotion_camera_cap.m32,
                observation_scale * dL_dmotion_camera_effective.m33 + dL_dmotion_camera_cap.m33};

            const float vx = observation_blur_params[0];
            const float vy = observation_blur_params[1];
            const float vz = observation_blur_params[2];
            const float px = p_camera.x;
            const float py = p_camera.y;
            const float pz = p_camera.z;
            const float h11 = dL_dmotion_camera.m11;
            const float h12 = 0.5f * (dL_dmotion_camera.m12 + dL_dmotion_camera.m21);
            const float h13 = 0.5f * (dL_dmotion_camera.m13 + dL_dmotion_camera.m31);
            const float h22 = dL_dmotion_camera.m22;
            const float h23 = 0.5f * (dL_dmotion_camera.m23 + dL_dmotion_camera.m32);
            const float h33 = dL_dmotion_camera.m33;
            if (observation_blur_gradients != nullptr) {
                atomicAdd(&observation_blur_gradients[0], clamp_grad(
                    pz * pz * h22 + py * py * h33 - 2.0f * py * pz * h23));
                atomicAdd(&observation_blur_gradients[1], clamp_grad(
                    pz * pz * h11 + px * px * h33 - 2.0f * px * pz * h13));
                atomicAdd(&observation_blur_gradients[2], clamp_grad(
                    py * py * h11 + px * px * h22 - 2.0f * px * py * h12));
                atomicAdd(&observation_blur_gradients[3], clamp_grad(h11));
                atomicAdd(&observation_blur_gradients[4], clamp_grad(h22));
                atomicAdd(&observation_blur_gradients[5], clamp_grad(h33));
            }

            dL_dp_motion = make_float3(
                2.0f * vz * px * h22 + 2.0f * vy * px * h33 -
                    2.0f * vz * py * h12 - 2.0f * vy * pz * h13,
                2.0f * vz * py * h11 + 2.0f * vx * py * h33 -
                    2.0f * vz * px * h12 - 2.0f * vx * pz * h23,
                2.0f * vy * pz * h11 + 2.0f * vx * pz * h22 -
                    2.0f * vy * px * h13 - 2.0f * vx * py * h23);

            const float3 bm_r1 = make_float3(
                motion_camera.m11 * jc_r1.x + motion_camera.m12 * jc_r1.y + motion_camera.m13 * jc_r1.z,
                motion_camera.m21 * jc_r1.x + motion_camera.m22 * jc_r1.y + motion_camera.m23 * jc_r1.z,
                motion_camera.m31 * jc_r1.x + motion_camera.m32 * jc_r1.y + motion_camera.m33 * jc_r1.z);
            const float3 bm_r2 = make_float3(
                motion_camera.m11 * jc_r2.x + motion_camera.m12 * jc_r2.y + motion_camera.m13 * jc_r2.z,
                motion_camera.m21 * jc_r2.x + motion_camera.m22 * jc_r2.y + motion_camera.m23 * jc_r2.z,
                motion_camera.m31 * jc_r2.x + motion_camera.m32 * jc_r2.y + motion_camera.m33 * jc_r2.z);
            dL_djc_motion_r1 = 2.0f * ga * bm_r1 +
                               2.0f * gb * bm_r2;
            dL_djc_motion_r2 = 2.0f * gb * bm_r1 +
                               2.0f * gc * bm_r2;

            if (blur_cap.saturated) {
                const float3 bm_raw_r1 = make_float3(
                    motion_camera_raw.m11 * jc_r1.x + motion_camera_raw.m12 * jc_r1.y + motion_camera_raw.m13 * jc_r1.z,
                    motion_camera_raw.m21 * jc_r1.x + motion_camera_raw.m22 * jc_r1.y + motion_camera_raw.m23 * jc_r1.z,
                    motion_camera_raw.m31 * jc_r1.x + motion_camera_raw.m32 * jc_r1.y + motion_camera_raw.m33 * jc_r1.z);
                const float3 bm_raw_r2 = make_float3(
                    motion_camera_raw.m11 * jc_r2.x + motion_camera_raw.m12 * jc_r2.y + motion_camera_raw.m13 * jc_r2.z,
                    motion_camera_raw.m21 * jc_r2.x + motion_camera_raw.m22 * jc_r2.y + motion_camera_raw.m23 * jc_r2.z,
                    motion_camera_raw.m31 * jc_r2.x + motion_camera_raw.m32 * jc_r2.y + motion_camera_raw.m33 * jc_r2.z);
                dL_djc_motion_r1 +=
                    2.0f * cap_ga * bm_raw_r1 +
                    2.0f * cap_gb * bm_raw_r2;
                dL_djc_motion_r2 +=
                    2.0f * cap_gb * bm_raw_r1 +
                    2.0f * cap_gc * bm_raw_r2;
            }
        }

        // Pure defocus still needs the total-cap VJP even when the motion
        // branch is disabled (and therefore has no B/J contribution).
        if (observation_blur_enabled && !observation_motion_enabled &&
            observation_defocus_enabled && blur_cap.saturated) {
            const float dL_dobservation_scale =
                dL_ddefocus_radius_sq * defocus_radius_sq_raw;
            const float cap_scale =
                -dL_dobservation_scale * observation_scale /
                blur_cap.largest_eigenvalue;
            dL_ddefocus_radius_sq_raw += cap_scale *
                (blur_cap.eigen_projector.x + blur_cap.eigen_projector.z);
        }

        float dL_ddepth_defocus = 0.0f;
        if (observation_blur_enabled && observation_defocus_enabled &&
            !defocus_radius_saturated) {
            const float dL_dbeta = dL_ddefocus_radius_sq_raw *
                                   defocus_q * defocus_q;
            const float dL_drho = dL_ddefocus_radius_sq_raw *
                                  2.0f * defocus_beta * defocus_q;
            if (observation_blur_gradients != nullptr) {
                // beta is projected to beta >= 0 by the optimizer.  At zero
                // use the right derivative so an all-zero initialization can
                // begin learning defocus.
                if (observation_blur_params[6] >= 0.0f) {
                    atomicAdd(&observation_blur_gradients[6], clamp_grad(dL_dbeta));
                }
                atomicAdd(&observation_blur_gradients[7], clamp_grad(dL_drho));
            }
            dL_ddepth_defocus = dL_drho / (depth_safe * depth_safe);
        }

        // gradient of J * W
        const float3 dL_djw_r1 = 2.0f * make_float3(
                                            jwc_r1.x * dL_dcov2d_motion.x + jwc_r2.x * dL_dcov2d_motion.y,
                                            jwc_r1.y * dL_dcov2d_motion.x + jwc_r2.y * dL_dcov2d_motion.y,
                                            jwc_r1.z * dL_dcov2d_motion.x + jwc_r2.z * dL_dcov2d_motion.y);
        const float3 dL_djw_r2 = 2.0f * make_float3(
                                            jwc_r1.x * dL_dcov2d_motion.y + jwc_r2.x * dL_dcov2d_motion.z,
                                            jwc_r1.y * dL_dcov2d_motion.y + jwc_r2.y * dL_dcov2d_motion.z,
                                            jwc_r1.z * dL_dcov2d_motion.y + jwc_r2.z * dL_dcov2d_motion.z);

        if (grad_w2c != nullptr) {
            accumulate_w2c_rotation_rows(
                grad_w2c,
                j11 * dL_djw_r1,
                j22 * dL_djw_r2,
                j13 * dL_djw_r1 + j23 * dL_djw_r2);
        }

        // gradient of non-zero entries in J
        const float dL_dj11 =
            w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z +
            dL_djc_motion_r1.x;
        const float dL_dj22 =
            w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z +
            dL_djc_motion_r2.y;
        const float dL_dj13 =
            w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z +
            dL_djc_motion_r1.z;
        const float dL_dj23 =
            w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z +
            dL_djc_motion_r2.z;

        // mean3d camera space gradient from J and mean2d (accounts for tx/ty clipping)
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];
        float3 dL_dmean3d_cam = make_float3(
            j11 * dL_dmean2d.x,
            j22 * dL_dmean2d.y,
            -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y);
        dL_dmean3d_cam += dL_dp_motion;
        dL_dmean3d_cam.z += dL_ddepth_defocus;
        const bool valid_x = x >= clip_left && x <= clip_right;
        const bool valid_y = y >= clip_top && y <= clip_bottom;
        if (valid_x)
            dL_dmean3d_cam.x -= j11 * dL_dj13 / depth_safe;
        if (valid_y)
            dL_dmean3d_cam.y -= j22 * dL_dj23 / depth_safe;
        const float factor_x = 1.0f + static_cast<float>(valid_x);
        const float factor_y = 1.0f + static_cast<float>(valid_y);
        dL_dmean3d_cam.z += (j11 * (factor_x * tx * dL_dj13 - dL_dj11) + j22 * (factor_y * ty * dL_dj23 - dL_dj22)) / depth_safe;

        if (grad_w2c != nullptr) {
            accumulate_w2c_from_camera_space_point(grad_w2c, mean3d, dL_dmean3d_cam);
        }

        // 3d mean gradient from splatting
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z);

        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        grad_means[primitive_idx] += clamp_grad3(dL_dmean3d);

        // raw scale gradient (zero gradient for clamped scales)
        const float dL_dvariance_x = rotation.m11 * rotation.m11 * dL_dcov3d.m11 + rotation.m21 * rotation.m21 * dL_dcov3d.m22 + rotation.m31 * rotation.m31 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m11 * rotation.m21 * dL_dcov3d.m12 + rotation.m11 * rotation.m31 * dL_dcov3d.m13 + rotation.m21 * rotation.m31 * dL_dcov3d.m23);
        const float dL_dvariance_y = rotation.m12 * rotation.m12 * dL_dcov3d.m11 + rotation.m22 * rotation.m22 * dL_dcov3d.m22 + rotation.m32 * rotation.m32 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m12 * rotation.m22 * dL_dcov3d.m12 + rotation.m12 * rotation.m32 * dL_dcov3d.m13 + rotation.m22 * rotation.m32 * dL_dcov3d.m23);
        const float dL_dvariance_z = rotation.m13 * rotation.m13 * dL_dcov3d.m11 + rotation.m23 * rotation.m23 * dL_dcov3d.m22 + rotation.m33 * rotation.m33 * dL_dcov3d.m33 +
                                     2.0f * (rotation.m13 * rotation.m23 * dL_dcov3d.m12 + rotation.m13 * rotation.m33 * dL_dcov3d.m13 + rotation.m23 * rotation.m33 * dL_dcov3d.m23);
        const float3 dL_draw_scale = make_float3(
            (raw_scale.x < config::max_raw_scale) ? 2.0f * variance.x * dL_dvariance_x : 0.0f,
            (raw_scale.y < config::max_raw_scale) ? 2.0f * variance.y * dL_dvariance_y : 0.0f,
            (raw_scale.z < config::max_raw_scale) ? 2.0f * variance.z * dL_dvariance_z : 0.0f);
        grad_raw_scales[primitive_idx] += clamp_grad3(dL_draw_scale);

        // raw rotation gradient
        const mat3x3 dL_drotation = {
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m11 + rotation_scaled.m21 * dL_dcov3d.m12 + rotation_scaled.m31 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m11 + rotation_scaled.m22 * dL_dcov3d.m12 + rotation_scaled.m32 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m11 + rotation_scaled.m23 * dL_dcov3d.m12 + rotation_scaled.m33 * dL_dcov3d.m13),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m12 + rotation_scaled.m21 * dL_dcov3d.m22 + rotation_scaled.m31 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m12 + rotation_scaled.m22 * dL_dcov3d.m22 + rotation_scaled.m32 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m12 + rotation_scaled.m23 * dL_dcov3d.m22 + rotation_scaled.m33 * dL_dcov3d.m23),
            2.0f * (rotation_scaled.m11 * dL_dcov3d.m13 + rotation_scaled.m21 * dL_dcov3d.m23 + rotation_scaled.m31 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m12 * dL_dcov3d.m13 + rotation_scaled.m22 * dL_dcov3d.m23 + rotation_scaled.m32 * dL_dcov3d.m33),
            2.0f * (rotation_scaled.m13 * dL_dcov3d.m13 + rotation_scaled.m23 * dL_dcov3d.m23 + rotation_scaled.m33 * dL_dcov3d.m33)};
        const float dL_dqxx = -dL_drotation.m22 - dL_drotation.m33;
        const float dL_dqyy = -dL_drotation.m11 - dL_drotation.m33;
        const float dL_dqzz = -dL_drotation.m11 - dL_drotation.m22;
        const float dL_dqxy = dL_drotation.m12 + dL_drotation.m21;
        const float dL_dqxz = dL_drotation.m13 + dL_drotation.m31;
        const float dL_dqyz = dL_drotation.m23 + dL_drotation.m32;
        const float dL_dqrx = dL_drotation.m32 - dL_drotation.m23;
        const float dL_dqry = dL_drotation.m13 - dL_drotation.m31;
        const float dL_dqrz = dL_drotation.m21 - dL_drotation.m12;
        const float dL_dq_norm_helper = qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz + qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz + qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
        const float4 dL_draw_rotation = 2.0f * make_float4(qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper, 2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper, 2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper, 2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) / q_norm_sq_safe;
        grad_raw_rotations[primitive_idx] += clamp_grad4(dL_draw_rotation);

        // TODO: only needed for adaptive density control from the original 3dgs
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d * make_float2(0.5f * w, 0.5f * h));
        }
    }

    /**
     * @brief Backward for per-Gaussian camera-space normal through nJ transform.
     *
     * Given dL/d(cam_normal_i) from blend_backward_normal_cu, computes:
     * - dL/d(mean3d): gradient through camera-space position (tx, ty, depth)
     * - dL/d(rotation): gradient through M_inv = S_inv * R^T * W^T → cov_cam_inv → normal
     * - dL/d(raw_scale): gradient through S_inv → M_inv → cov_cam_inv → normal
     *
     * The full chain: cam_normal = normalize(nJ @ ray_normal), where
     *   ray_normal = [-plane_x*fn, -plane_y*fn, -1], fn = tc/ray_len2
     *   plane = nJ_inv @ vec, vec = cov_cam_inv @ h / vb
     *   cov_cam_inv = M_inv^T * M_inv, M_inv = S_inv * R^T * W^T
     */
    __global__ void preprocess_backward_normal_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float4* __restrict__ w2c,
        const float3* __restrict__ grad_normal_per_gaussian, // from blend_backward_normal
        const uint* __restrict__ primitive_n_touched_tiles,
        float3* __restrict__ grad_means,       // accumulate
        float3* __restrict__ grad_raw_scales,  // accumulate
        float4* __restrict__ grad_raw_rotations, // accumulate
        const uint n_primitives,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        auto primitive_idx = cg::this_grid().thread_rank();
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        // Check if gradient is non-zero
        const float3 dL_dcam_normal = grad_normal_per_gaussian[primitive_idx];
        const float grad_norm = fabsf(dL_dcam_normal.x) + fabsf(dL_dcam_normal.y) + fabsf(dL_dcam_normal.z);
        if (grad_norm < 1e-12f)
            return;

        // ===== Recompute forward values =====
        const float3 mean3d = means[primitive_idx];
        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < 1e-4f) return;

        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float x_grad_mul = (x >= clip_left && x <= clip_right) ? 1.0f : 0.0f;
        const float y_grad_mul = (y >= clip_top && y <= clip_bottom) ? 1.0f : 0.0f;

        // Rotation and scale
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        const float inv_s0 = rsqrtf(fmaxf(variance.x, 1e-20f));
        const float inv_s1 = rsqrtf(fmaxf(variance.y, 1e-20f));
        const float inv_s2 = rsqrtf(fmaxf(variance.z, 1e-20f));

        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float q_norm_sq = qr*qr + qx*qx + qy*qy + qz*qz;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-7f);
        const float qxx = 2.0f * qx*qx / q_norm_sq_safe, qyy = 2.0f * qy*qy / q_norm_sq_safe, qzz = 2.0f * qz*qz / q_norm_sq_safe;
        const float qxy = 2.0f * qx*qy / q_norm_sq_safe, qxz = 2.0f * qx*qz / q_norm_sq_safe, qyz = 2.0f * qy*qz / q_norm_sq_safe;
        const float qrx = 2.0f * qr*qx / q_norm_sq_safe, qry = 2.0f * qr*qy / q_norm_sq_safe, qrz = 2.0f * qr*qz / q_norm_sq_safe;
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};

        // M_inv rows (recomputed from forward)
        const float3 r_col0 = make_float3(rotation.m11, rotation.m21, rotation.m31);
        const float3 r_col1 = make_float3(rotation.m12, rotation.m22, rotation.m32);
        const float3 r_col2 = make_float3(rotation.m13, rotation.m23, rotation.m33);
        const float3 w_r0 = make_float3(w2c_r1.x, w2c_r1.y, w2c_r1.z);
        const float3 w_r1 = make_float3(w2c_r2.x, w2c_r2.y, w2c_r2.z);
        const float3 w_r2 = make_float3(w2c_r3.x, w2c_r3.y, w2c_r3.z);
        const float3 m0 = inv_s0 * make_float3(dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2));
        const float3 m1 = inv_s1 * make_float3(dot(r_col1, w_r0), dot(r_col1, w_r1), dot(r_col1, w_r2));
        const float3 m2 = inv_s2 * make_float3(dot(r_col2, w_r0), dot(r_col2, w_r1), dot(r_col2, w_r2));

        // cov_cam_inv
        const float cc00 = m0.x*m0.x + m1.x*m1.x + m2.x*m2.x;
        const float cc01 = m0.x*m0.y + m1.x*m1.y + m2.x*m2.y;
        const float cc02 = m0.x*m0.z + m1.x*m1.z + m2.x*m2.z;
        const float cc11 = m0.y*m0.y + m1.y*m1.y + m2.y*m2.y;
        const float cc12 = m0.y*m0.z + m1.y*m1.z + m2.y*m2.z;
        const float cc22 = m0.z*m0.z + m1.z*m1.z + m2.z*m2.z;

        const float u = tx, v = ty;
        const float uvh_m0 = cc00*u + cc01*v + cc02;
        const float uvh_m1 = cc01*u + cc11*v + cc12;
        const float uvh_m2 = cc02*u + cc12*v + cc22;
        const float vb = uvh_m0*u + uvh_m1*v + uvh_m2;
        if (!(vb > 1e-10f)) return;

        const float ray_len2 = u*u + v*v + 1.0f;
        const float tc = norm3df(u*depth, v*depth, depth);
        if (!(tc > 1e-10f)) return;

        const float inv_vb = 1.0f / vb;
        const float vec0 = uvh_m0 * inv_vb;
        const float vec1 = uvh_m1 * inv_vb;
        const float vec2 = uvh_m2 * inv_vb;

        const float u2 = u*u, v2 = v*v, uv = u*v;
        const float plane_x = (v2 + 1.0f)*vec0 - uv*vec1 - u*vec2;
        const float plane_y = -uv*vec0 + (u2 + 1.0f)*vec1 - v*vec2;

        const float factor_normal = tc / ray_len2;
        const float ray_len_inv = rsqrtf(ray_len2);
        const float inv_depth = 1.0f / depth;
        const float depth_over_tc = depth / tc;

        // Ray-space normal
        const float ray_nx = -plane_x * factor_normal;
        const float ray_ny = -plane_y * factor_normal;
        constexpr float ray_nz = -1.0f;

        // Camera-space normal (un-normalized)
        const float cam_nx = ray_nx * inv_depth + ray_nz * tx * depth_over_tc;
        const float cam_ny = ray_ny * inv_depth + ray_nz * ty * depth_over_tc;
        const float cam_nz = -ray_nx * tx * inv_depth - ray_ny * ty * inv_depth + ray_nz * depth_over_tc;
        const float nlen = norm3df(cam_nx, cam_ny, cam_nz);
        if (nlen < 1e-8f) return;

        // ===== Backward: dL/d(cam_normal_normalized) → dL/d(cam_normal_unnormalized) =====
        // dL/d(v) = (dL/d(v/|v|) - (v/|v|) * dot(dL/d(v/|v|), v/|v|)) / |v|
        const float3 cam_n_hat = make_float3(cam_nx / nlen, cam_ny / nlen, cam_nz / nlen);
        const float proj = dot(dL_dcam_normal, cam_n_hat);
        const float inv_nlen = 1.0f / nlen;
        const float dL_dcnx = (dL_dcam_normal.x - cam_n_hat.x * proj) * inv_nlen;
        const float dL_dcny = (dL_dcam_normal.y - cam_n_hat.y * proj) * inv_nlen;
        const float dL_dcnz = (dL_dcam_normal.z - cam_n_hat.z * proj) * inv_nlen;

        // ===== Backward: cam_normal → ray_normal / nJ (GGGS full chain) =====
        const float dL_dray_normal_x = dL_dcnx * inv_depth - dL_dcnz * u * inv_depth;
        const float dL_dray_normal_y = dL_dcny * inv_depth - dL_dcnz * v * inv_depth;
        const float dL_dnJ11 = dL_dcnx * ray_nx;
        const float dL_dnJ13 = dL_dcnx * ray_nz;
        const float dL_dnJ22 = dL_dcny * ray_ny;
        const float dL_dnJ23 = dL_dcny * ray_nz;
        const float dL_dnJ31 = dL_dcnz * ray_nx;
        const float dL_dnJ32 = dL_dcnz * ray_ny;
        const float dL_dnJ33 = dL_dcnz * ray_nz;

        const float dL_dfactor_normal = plane_x * (-dL_dray_normal_x) + plane_y * (-dL_dray_normal_y);
        const float dL_dplane_x = (-dL_dray_normal_x) * factor_normal;
        const float dL_dplane_y = (-dL_dray_normal_y) * factor_normal;

        // ===== Backward: plane → vec (through nJ_inv) =====
        const float dL_dvec0 = dL_dplane_x * (v2+1.0f) + dL_dplane_y * (-uv);
        const float dL_dvec1 = dL_dplane_x * (-uv)     + dL_dplane_y * (u2+1.0f);
        const float dL_dvec2 = dL_dplane_x * (-u)       + dL_dplane_y * (-v);

        // ===== Backward: vec → uvh_m, inv_vb =====
        // vec_i = uvh_mi * inv_vb
        const float dL_duvh_m0 = dL_dvec0 * inv_vb;
        const float dL_duvh_m1 = dL_dvec1 * inv_vb;
        const float dL_duvh_m2 = dL_dvec2 * inv_vb;
        const float dL_dinv_vb = dL_dvec0 * uvh_m0 + dL_dvec1 * uvh_m1 + dL_dvec2 * uvh_m2;
        const float dL_dvb = -dL_dinv_vb * inv_vb * inv_vb;

        // ===== Backward: uvh_m, vb → cov_cam_inv (cc00..cc22) =====
        // uvh_m0 = cc00*u + cc01*v + cc02
        // uvh_m1 = cc01*u + cc11*v + cc12
        // uvh_m2 = cc02*u + cc12*v + cc22
        // vb = uvh_m0*u + uvh_m1*v + uvh_m2
        const float dL_duvh_m0_tot = dL_duvh_m0 + dL_dvb * u;
        const float dL_duvh_m1_tot = dL_duvh_m1 + dL_dvb * v;
        const float dL_duvh_m2_tot = dL_duvh_m2 + dL_dvb * 1.0f;

        float dL_dcc00 = dL_duvh_m0_tot * u;
        float dL_dcc01 = dL_duvh_m0_tot * v + dL_duvh_m1_tot * u;
        float dL_dcc02 = dL_duvh_m0_tot * 1.0f + dL_duvh_m2_tot * u;
        float dL_dcc11 = dL_duvh_m1_tot * v;
        float dL_dcc12 = dL_duvh_m1_tot * 1.0f + dL_duvh_m2_tot * v;
        float dL_dcc22 = dL_duvh_m2_tot * 1.0f;

        // ===== Backward: cov_cam_inv → M_inv rows =====
        // cc00 = m0.x^2 + m1.x^2 + m2.x^2
        // cc01 = m0.x*m0.y + m1.x*m1.y + m2.x*m2.y, etc.
        float3 dL_dm0 = make_float3(
            2.0f*m0.x*dL_dcc00 + m0.y*dL_dcc01 + m0.z*dL_dcc02,
            m0.x*dL_dcc01 + 2.0f*m0.y*dL_dcc11 + m0.z*dL_dcc12,
            m0.x*dL_dcc02 + m0.y*dL_dcc12 + 2.0f*m0.z*dL_dcc22);
        float3 dL_dm1 = make_float3(
            2.0f*m1.x*dL_dcc00 + m1.y*dL_dcc01 + m1.z*dL_dcc02,
            m1.x*dL_dcc01 + 2.0f*m1.y*dL_dcc11 + m1.z*dL_dcc12,
            m1.x*dL_dcc02 + m1.y*dL_dcc12 + 2.0f*m1.z*dL_dcc22);
        float3 dL_dm2 = make_float3(
            2.0f*m2.x*dL_dcc00 + m2.y*dL_dcc01 + m2.z*dL_dcc02,
            m2.x*dL_dcc01 + 2.0f*m2.y*dL_dcc11 + m2.z*dL_dcc12,
            m2.x*dL_dcc02 + m2.y*dL_dcc12 + 2.0f*m2.z*dL_dcc22);

        // ===== Backward: M_inv rows → inv_scale, rotation, w2c =====
        // m0 = inv_s0 * [dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2)]
        // dL/d(inv_s0) = dot(dL_dm0, [dot(r_col0, w_r0), ...])
        // dL/d(r_col0) from m0 = inv_s0 * (dL_dm0.x * w_r0 + dL_dm0.y * w_r1 + dL_dm0.z * w_r2)
        const float dL_dinv_s0 = dot(dL_dm0, make_float3(dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2)));
        const float dL_dinv_s1 = dot(dL_dm1, make_float3(dot(r_col1, w_r0), dot(r_col1, w_r1), dot(r_col1, w_r2)));
        const float dL_dinv_s2 = dot(dL_dm2, make_float3(dot(r_col2, w_r0), dot(r_col2, w_r1), dot(r_col2, w_r2)));

        const float3 dL_dr_col0 = inv_s0 * (dL_dm0.x * w_r0 + dL_dm0.y * w_r1 + dL_dm0.z * w_r2);
        const float3 dL_dr_col1 = inv_s1 * (dL_dm1.x * w_r0 + dL_dm1.y * w_r1 + dL_dm1.z * w_r2);
        const float3 dL_dr_col2 = inv_s2 * (dL_dm2.x * w_r0 + dL_dm2.y * w_r1 + dL_dm2.z * w_r2);

        // ===== Accumulate gradients =====

        // Gradient w.r.t. raw_scale: inv_s = 1/sqrt(variance) = 1/exp(clamped_raw_scale)
        // d(inv_s)/d(raw_scale) = -inv_s (when not clamped)
        const float3 dL_draw_scale_from_normal = make_float3(
            (raw_scale.x < config::max_raw_scale) ? -dL_dinv_s0 * inv_s0 : 0.0f,
            (raw_scale.y < config::max_raw_scale) ? -dL_dinv_s1 * inv_s1 : 0.0f,
            (raw_scale.z < config::max_raw_scale) ? -dL_dinv_s2 * inv_s2 : 0.0f);
        grad_raw_scales[primitive_idx] += clamp_grad3(dL_draw_scale_from_normal);

        // Gradient w.r.t. rotation: dL/d(R) where r_col0 = R[*, 0], r_col1 = R[*, 1], r_col2 = R[*, 2]
        // dL/d(R) = [dL_dr_col0 | dL_dr_col1 | dL_dr_col2] (columns)
        const mat3x3 dL_drotation_from_normal = {
            dL_dr_col0.x, dL_dr_col1.x, dL_dr_col2.x,
            dL_dr_col0.y, dL_dr_col1.y, dL_dr_col2.y,
            dL_dr_col0.z, dL_dr_col1.z, dL_dr_col2.z};

        // Convert dL/d(rotation_matrix) → dL/d(raw_quaternion) using the same formula as preprocess_backward_cu
        const float dL_dqxx_n = -dL_drotation_from_normal.m22 - dL_drotation_from_normal.m33;
        const float dL_dqyy_n = -dL_drotation_from_normal.m11 - dL_drotation_from_normal.m33;
        const float dL_dqzz_n = -dL_drotation_from_normal.m11 - dL_drotation_from_normal.m22;
        const float dL_dqxy_n = dL_drotation_from_normal.m12 + dL_drotation_from_normal.m21;
        const float dL_dqxz_n = dL_drotation_from_normal.m13 + dL_drotation_from_normal.m31;
        const float dL_dqyz_n = dL_drotation_from_normal.m23 + dL_drotation_from_normal.m32;
        const float dL_dqrx_n = dL_drotation_from_normal.m32 - dL_drotation_from_normal.m23;
        const float dL_dqry_n = dL_drotation_from_normal.m13 - dL_drotation_from_normal.m31;
        const float dL_dqrz_n = dL_drotation_from_normal.m21 - dL_drotation_from_normal.m12;
        const float dL_dq_norm_n = qxx*dL_dqxx_n + qyy*dL_dqyy_n + qzz*dL_dqzz_n + qxy*dL_dqxy_n + qxz*dL_dqxz_n + qyz*dL_dqyz_n + qrx*dL_dqrx_n + qry*dL_dqry_n + qrz*dL_dqrz_n;
        const float4 dL_draw_rotation_from_normal = 2.0f * make_float4(
            qx*dL_dqrx_n + qy*dL_dqry_n + qz*dL_dqrz_n - qr*dL_dq_norm_n,
            2.0f*qx*dL_dqxx_n + qy*dL_dqxy_n + qz*dL_dqxz_n + qr*dL_dqrx_n - qx*dL_dq_norm_n,
            2.0f*qy*dL_dqyy_n + qx*dL_dqxy_n + qz*dL_dqyz_n + qr*dL_dqry_n - qy*dL_dq_norm_n,
            2.0f*qz*dL_dqzz_n + qx*dL_dqxz_n + qy*dL_dqyz_n + qr*dL_dqrz_n - qz*dL_dq_norm_n) / q_norm_sq_safe;
        grad_raw_rotations[primitive_idx] += clamp_grad4(dL_draw_rotation_from_normal);

        // Gradient w.r.t. mean3d (GGGS computeCov2DCUDA full u/v/z chain)
        const float aux = dL_dplane_x * plane_x + dL_dplane_y * plane_y;
        const float nJ_inv_t_dp0 = (v2 + 1.0f) * dL_dplane_x - uv * dL_dplane_y;
        const float nJ_inv_t_dp1 = -uv * dL_dplane_x + (u2 + 1.0f) * dL_dplane_y;
        const float nJ_inv_t_dp2 = -u * dL_dplane_x - v * dL_dplane_y;
        const float dL_duvh_plane0 = 2.0f * (-aux) * vec0 + (cc00 * nJ_inv_t_dp0 + cc01 * nJ_inv_t_dp1 + cc02 * nJ_inv_t_dp2) * inv_vb;
        const float dL_duvh_plane1 = 2.0f * (-aux) * vec1 + (cc01 * nJ_inv_t_dp0 + cc11 * nJ_inv_t_dp1 + cc12 * nJ_inv_t_dp2) * inv_vb;

        const float aux_nJ = (-dL_dnJ13 * u - dL_dnJ23 * v - dL_dnJ33) / ray_len2 * ray_len_inv;
        const float dL_du_nJ = -dL_dnJ31 / depth + dL_dnJ13 * ray_len_inv + aux_nJ * u;
        const float dL_dv_nJ = -dL_dnJ32 / depth + dL_dnJ23 * ray_len_inv + aux_nJ * v;
        const float dL_dz_nJ = (dL_dnJ11 + dL_dnJ22 - dL_dnJ31 * u - dL_dnJ32 * v) / (-depth * depth);

        const float dL_du_plane = dL_duvh_plane0 + (dL_dplane_x * vec1 + dL_dplane_y * vec0) * (-v) + 2.0f * dL_dplane_y * vec1 * u - dL_dplane_x * vec2;
        const float dL_dv_plane = dL_duvh_plane1 + (dL_dplane_x * vec1 + dL_dplane_y * vec0) * (-u) + 2.0f * dL_dplane_x * vec0 * v - dL_dplane_y * vec2;

        const float aux_factor = dL_dfactor_normal * (-depth / ray_len2 * ray_len_inv);
        const float dL_du_factor = aux_factor * u;
        const float dL_dv_factor = aux_factor * v;
        const float dL_dz_factor = dL_dfactor_normal * ray_len_inv;

        const float dL_du = dL_du_nJ + dL_du_plane + dL_du_factor;
        const float dL_dv = dL_dv_nJ + dL_dv_plane + dL_dv_factor;
        const float dL_dz = dL_dz_nJ + dL_dz_factor;

        const float t_clipped_x = u * depth;
        const float t_clipped_y = v * depth;
        const float depth_sq = depth * depth;
        const float dL_dtx = x_grad_mul * dL_du / depth;
        const float dL_dty = y_grad_mul * dL_dv / depth;
        const float dL_dtz = -(x_grad_mul * dL_du * t_clipped_x + y_grad_mul * dL_dv * t_clipped_y) / depth_sq + dL_dz;

        const float3 dL_dmean3d_from_normal = make_float3(
            w2c_r1.x * dL_dtx + w2c_r2.x * dL_dty + w2c_r3.x * dL_dtz,
            w2c_r1.y * dL_dtx + w2c_r2.y * dL_dty + w2c_r3.y * dL_dtz,
            w2c_r1.z * dL_dtx + w2c_r2.z * dL_dty + w2c_r3.z * dL_dtz);
        grad_means[primitive_idx] += clamp_grad3(dL_dmean3d_from_normal);
    }

    // based on https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    template <bool HAS_DENSIFICATION>
    __global__ void blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ grad_image,
        const float* __restrict__ grad_alpha_map,
        const float* __restrict__ image,
        const float* __restrict__ alpha_map,
        const uint* __restrict__ tile_max_n_contributions,
        const uint* __restrict__ tile_n_contributions,
        const uint* __restrict__ bucket_tile_index,
        const uint* __restrict__ bucket_checkpoint_uint8,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_compensated_opacity,
        float* __restrict__ grad_raw_opacity,
        float3* __restrict__ grad_color,
        float* __restrict__ densification_info,
        const float* __restrict__ densification_error_map,
        const uint n_buckets,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool mip_filter,
        const bool observation_blur_enabled) {
        auto block = cg::this_thread_block();
        const uint bucket_idx = block.group_index().x;
        if (bucket_idx >= n_buckets)
            return;
        auto warp = cg::tiled_partition<32>(block);
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
            return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float compensated_opacity = 0.0f;
        float original_opacity = 0.0f;
        float3 color = {0.0f, 0.0f, 0.0f};
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            compensated_opacity = conic_opacity.w;
            original_opacity = (mip_filter || observation_blur_enabled)
                                   ? __frcp_rn(1.0f + __expf(-raw_opacities[primitive_idx]))
                                   : compensated_opacity;
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fminf(fmaxf(color_unclamped, 0.0f), config::max_checkpoint_color);
            if (color_unclamped.x >= 0.0f && color_unclamped.x <= config::max_checkpoint_color)
                color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f && color_unclamped.y <= config::max_checkpoint_color)
                color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f && color_unclamped.z <= config::max_checkpoint_color)
                color_grad_factor.z = 1.0f;
        }

        // helpers
        const uint n_pixels = width * height;

        // gradient accumulation
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_draw_opacity_partial_accum = 0.0f;
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};
        float densification_weight_accum = 0.0f;
        float densification_error_weighted_accum = 0.0f;

        // tile metadata
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        uint last_contributor;
        float3 color_pixel_after;
        float transmittance;
        float3 grad_color_pixel;
        float grad_alpha_common;

        bucket_checkpoint_uint8 += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];

#pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                const uint packed = bucket_checkpoint_uint8[local_idx];
                constexpr float COLOR_INV_SCALE = config::max_checkpoint_color / 255.0f;
                constexpr float TRANS_INV_SCALE = 1.0f / 255.0f;
                const float3 checkpoint_color = make_float3(
                    static_cast<float>(packed & 0xFF) * COLOR_INV_SCALE,
                    static_cast<float>((packed >> 8) & 0xFF) * COLOR_INV_SCALE,
                    static_cast<float>((packed >> 16) & 0xFF) * COLOR_INV_SCALE);
                const float checkpoint_transmittance = static_cast<float>((packed >> 24) & 0xFF) * TRANS_INV_SCALE;
                const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                const bool pixel_in_bounds = pixel_coords.x < width && pixel_coords.y < height;
                // final values from forward pass before background blend and the respective gradients
                float3 color_pixel = {0.0f, 0.0f, 0.0f};
                float3 grad_color_pixel_local = {0.0f, 0.0f, 0.0f};
                float alpha_pixel = 0.0f;
                float grad_alpha_pixel = 0.0f;
                uint last_contrib_val = 0;
                if (pixel_in_bounds) {
                    color_pixel = make_float3(
                        image[pixel_idx],
                        image[n_pixels + pixel_idx],
                        image[2 * n_pixels + pixel_idx]);
                    grad_color_pixel_local = make_float3(
                        grad_image[pixel_idx],
                        grad_image[n_pixels + pixel_idx],
                        grad_image[2 * n_pixels + pixel_idx]);
                    alpha_pixel = alpha_map[pixel_idx];
                    grad_alpha_pixel = grad_alpha_map[pixel_idx];
                    last_contrib_val = tile_n_contributions[pixel_idx];
                }
                // color_pixel_after = final_color - checkpoint_color
                collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                    color_pixel - checkpoint_color,
                    checkpoint_transmittance);
                collected_grad_info_pixel[lane_idx] = make_float4(
                    grad_color_pixel_local,
                    grad_alpha_pixel * (1.0f - alpha_pixel));
                collected_last_contributor[lane_idx] = last_contrib_val;
                __syncwarp();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
            }

            // which pixel index should this thread deal with?
            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            // leader thread loads values from shared memory into registers
            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip)
                continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            if (sigma_over_2 < 0.0f)
                continue;
            const float gaussian = expf(-sigma_over_2);
            const float unclamped_alpha = compensated_opacity * gaussian;
            const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
            if (alpha < config::min_alpha_threshold)
                continue;
            const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;
            const float one_minus_alpha = 1.0f - alpha;

            const float blending_weight = transmittance * alpha;

            if constexpr (HAS_DENSIFICATION) {
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                const float pixel_error = densification_error_map[pixel_idx];
                densification_weight_accum += blending_weight;
                densification_error_weighted_accum += blending_weight * pixel_error;
            }

            // color gradient
            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            color_pixel_after -= blending_weight * color;

            // alpha gradient
            const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
            const float one_minus_alpha_rcp = 1.0f / one_minus_alpha_safe;
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
            // opacity gradient w.r.t. compensated_opacity (zero when alpha is clamped - no gradient flows through clamp)
            // dL/d(compensated_opacity) = dL/d(alpha) * d(alpha)/d(compensated_opacity) = dL_dalpha * gaussian
            const float dL_dcompensated_opacity = alpha_saturated ? 0.0f : gaussian * dL_dalpha;
            dL_draw_opacity_partial_accum += dL_dcompensated_opacity;

            // conic and mean2d gradient (zero when alpha is clamped)
            const float gaussian_grad_helper = alpha_saturated ? 0.0f : -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                                                                 conic.x * delta.x + conic.y * delta.y,
                                                                 conic.y * delta.x + conic.z * delta.y);
            dL_dmean2d_accum += dL_dmean2d;

            transmittance *= one_minus_alpha;
        }

        // Add clamped gradients using atomics
        if (valid_primitive) {
            const float2 clamped_mean2d = make_float2(clamp_grad(dL_dmean2d_accum.x), clamp_grad(dL_dmean2d_accum.y));
            atomicAdd(&grad_mean2d[primitive_idx].x, clamped_mean2d.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, clamped_mean2d.y);
            const float3 clamped_conic = clamp_grad3(dL_dconic_accum);
            atomicAdd(&grad_conic[primitive_idx], clamped_conic.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], clamped_conic.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], clamped_conic.z);
            // Chain rule: dL/d(raw_opacity) = dL/d(comp_opacity) * d(comp)/d(orig) * d(orig)/d(raw)
            // d(comp)/d(orig) = conv_factor = comp_opacity / orig_opacity (when mip_filter)
            // d(orig)/d(raw) = sigmoid_derivative = orig_opacity * (1 - orig_opacity)
            const float conv_factor = (mip_filter || observation_blur_enabled)
                                          ? compensated_opacity / fmaxf(original_opacity, 1e-6f)
                                          : 1.0f;
            const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
            const float dL_draw_opacity = clamp_grad(dL_draw_opacity_partial_accum * conv_factor * sigmoid_derivative);
            // Keep the unclamped VJP for the blur compensation branch.  The
            // final per-parameter atomics apply the same safety clamp as the
            // rest of the rasterizer; clamping here would bias determinant
            // compensation when several pixels contribute to one primitive.
            if (observation_blur_enabled) {
                atomicAdd(&grad_compensated_opacity[primitive_idx],
                          dL_draw_opacity_partial_accum);
            }
            atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
            const float3 clamped_color = clamp_grad3(dL_dcolor_accum);
            atomicAdd(&grad_color[primitive_idx].x, clamped_color.x);
            atomicAdd(&grad_color[primitive_idx].y, clamped_color.y);
            atomicAdd(&grad_color[primitive_idx].z, clamped_color.z);

            if constexpr (HAS_DENSIFICATION) {
                atomicAdd(&densification_info[primitive_idx], densification_weight_accum);
                atomicAdd(&densification_info[n_primitives + primitive_idx], densification_error_weighted_accum);
            }
        }
    }

    /**
     * @brief Backward blend for normal gradients (GGGS losses).
     *
     * Same bucket-based iteration as blend_backward_cu, but processes
     * grad_render_normal → per-Gaussian grad_normal, and additional
     * contributions to grad_mean2d, grad_conic, grad_raw_opacity from the
     * normal blending alpha term.
     *
     * Uses normal checkpoints (uint8 packed) written during forward.
     * Also uses normal_accum_length_map for normalization backward.
     */
    __global__ void blend_backward_normal_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_normal_data,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ grad_normal_accum_map,  // [3, H, W] gradient w.r.t. un-normalized normal_accum
        const uint* __restrict__ tile_max_n_contributions,
        const uint* __restrict__ tile_n_contributions,
        const uint* __restrict__ bucket_tile_index,
        const uint* __restrict__ bucket_checkpoint_uint8,            // for transmittance
        const uint* __restrict__ bucket_checkpoint_normal_uint8,     // for normal checkpoints
        const float* __restrict__ normal_map,             // [3, H, W] normalized normal from forward
        const float* __restrict__ normal_accum_length_map, // [H, W] |normal_accum| per pixel
        float3* __restrict__ grad_normal_per_gaussian,    // [N] per-Gaussian normal gradient output
        float2* __restrict__ grad_mean2d,                 // accumulate additional alpha gradient
        float* __restrict__ grad_conic,                   // accumulate additional conic gradient
        float* __restrict__ grad_raw_opacity,             // accumulate additional opacity gradient
        const uint n_buckets,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool mip_filter) {
        auto block = cg::this_thread_block();
        const uint bucket_idx = block.group_index().x;
        if (bucket_idx >= n_buckets)
            return;
        auto warp = cg::tiled_partition<32>(block);
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
            return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // Load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float compensated_opacity = 0.0f;
        float original_opacity = 0.0f;
        float3 prim_normal = {0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            compensated_opacity = conic_opacity.w;
            original_opacity = mip_filter ? __frcp_rn(1.0f + __expf(-raw_opacities[primitive_idx])) : compensated_opacity;
            prim_normal = primitive_normal_data[primitive_idx];
        }

        const uint n_pixels = width * height;
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        // Gradient accumulators
        float3 dL_dnormal_accum = {0.0f, 0.0f, 0.0f};
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_draw_opacity_partial_accum = 0.0f;

        uint last_contributor;
        float3 normal_pixel_after;
        float transmittance;
        float3 grad_normal_pixel;

        const uint* checkpoint_normal = bucket_checkpoint_normal_uint8 + bucket_idx * config::block_size_blend;
        const uint* checkpoint_color = bucket_checkpoint_uint8 + bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_normal_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_normal_pixel_alpha[32]; // xyz=grad_normal, w=grad_alpha_from_normal

#pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                // Get transmittance from color checkpoint
                const uint packed_color = checkpoint_color[local_idx];
                const float checkpoint_transmittance = static_cast<float>((packed_color >> 24) & 0xFF) / 255.0f;

                // Get normal checkpoint
                const uint packed_normal = checkpoint_normal[local_idx];
                constexpr float NORMAL_INV_SCALE = 1.0f / 127.5f;
                constexpr float NORMAL_BIAS = 127.5f;
                const float3 checkpoint_normal_val = make_float3(
                    (static_cast<float>(packed_normal & 0xFF) - NORMAL_BIAS) * NORMAL_INV_SCALE,
                    (static_cast<float>((packed_normal >> 8) & 0xFF) - NORMAL_BIAS) * NORMAL_INV_SCALE,
                    (static_cast<float>((packed_normal >> 16) & 0xFF) - NORMAL_BIAS) * NORMAL_INV_SCALE);

                const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                const bool pixel_in_bounds = pixel_coords.x < width && pixel_coords.y < height;

                float3 final_normal_accum = {0.0f, 0.0f, 0.0f};
                float3 grad_normal_accum_local = {0.0f, 0.0f, 0.0f};
                uint last_contrib_val = 0;
                if (pixel_in_bounds) {
                    // Reconstruct un-normalized normal_accum from normal_map * nlen
                    const float nlen = normal_accum_length_map[pixel_idx];
                    final_normal_accum = nlen * make_float3(
                        normal_map[pixel_idx],
                        normal_map[n_pixels + pixel_idx],
                        normal_map[2 * n_pixels + pixel_idx]);

                    // grad_normal_accum_map is the pre-computed gradient w.r.t. un-normalized normal_accum
                    grad_normal_accum_local = make_float3(
                        grad_normal_accum_map[pixel_idx],
                        grad_normal_accum_map[n_pixels + pixel_idx],
                        grad_normal_accum_map[2 * n_pixels + pixel_idx]);
                    last_contrib_val = tile_n_contributions[pixel_idx];
                }

                // normal_pixel_after = final_normal_accum - checkpoint_normal_accum
                collected_normal_pixel_after_transmittance[lane_idx] = make_float4(
                    final_normal_accum - checkpoint_normal_val,
                    checkpoint_transmittance);
                collected_grad_normal_pixel_alpha[lane_idx] = make_float4(
                    grad_normal_accum_local,
                    0.0f); // alpha contribution computed inline
                collected_last_contributor[lane_idx] = last_contrib_val;
                __syncwarp();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                normal_pixel_after.x = warp.shfl_up(normal_pixel_after.x, 1);
                normal_pixel_after.y = warp.shfl_up(normal_pixel_after.y, 1);
                normal_pixel_after.z = warp.shfl_up(normal_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_normal_pixel.x = warp.shfl_up(grad_normal_pixel.x, 1);
                grad_normal_pixel.y = warp.shfl_up(grad_normal_pixel.y, 1);
                grad_normal_pixel.z = warp.shfl_up(grad_normal_pixel.z, 1);
            }

            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 npa_t = collected_normal_pixel_after_transmittance[current_shmem_index];
                normal_pixel_after = make_float3(npa_t);
                transmittance = npa_t.w;
                grad_normal_pixel = make_float3(collected_grad_normal_pixel_alpha[current_shmem_index]);
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip)
                continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            if (sigma_over_2 < 0.0f)
                continue;
            const float gaussian = expf(-sigma_over_2);
            const float unclamped_alpha = compensated_opacity * gaussian;
            const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
            if (alpha < config::min_alpha_threshold)
                continue;
            const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;
            const float one_minus_alpha = 1.0f - alpha;
            const float blending_weight = transmittance * alpha;

            // Normal gradient: dL/d(prim_normal_i) = blending_weight * grad_normal_pixel
            const float3 dL_dnormal = blending_weight * grad_normal_pixel;
            dL_dnormal_accum += dL_dnormal;

            normal_pixel_after -= blending_weight * prim_normal;

            // Alpha gradient from normal blending:
            // normal_accum = sum_i T_i * alpha_i * n_i
            // dL/d(alpha_i) from normal = dot(T * n - normal_after / (1-alpha), grad_normal)
            const float one_minus_alpha_safe = fmaxf(one_minus_alpha, 1e-4f);
            const float one_minus_alpha_rcp = 1.0f / one_minus_alpha_safe;
            const float dL_dalpha = dot(transmittance * prim_normal - normal_pixel_after * one_minus_alpha_rcp, grad_normal_pixel);
            const float dL_dcompensated_opacity = alpha_saturated ? 0.0f : gaussian * dL_dalpha;
            dL_draw_opacity_partial_accum += dL_dcompensated_opacity;

            // Conic and mean2d gradient from alpha (same structure as color backward)
            const float gaussian_grad_helper = alpha_saturated ? 0.0f : -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                                                                 conic.x * delta.x + conic.y * delta.y,
                                                                 conic.y * delta.x + conic.z * delta.y);
            dL_dmean2d_accum += dL_dmean2d;

            transmittance *= one_minus_alpha;
        }

        // Atomic accumulate
        if (valid_primitive) {
            const float3 clamped_normal = clamp_grad3(dL_dnormal_accum);
            atomicAdd(&grad_normal_per_gaussian[primitive_idx].x, clamped_normal.x);
            atomicAdd(&grad_normal_per_gaussian[primitive_idx].y, clamped_normal.y);
            atomicAdd(&grad_normal_per_gaussian[primitive_idx].z, clamped_normal.z);

            const float2 clamped_mean2d = make_float2(clamp_grad(dL_dmean2d_accum.x), clamp_grad(dL_dmean2d_accum.y));
            atomicAdd(&grad_mean2d[primitive_idx].x, clamped_mean2d.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, clamped_mean2d.y);
            const float3 clamped_conic = clamp_grad3(dL_dconic_accum);
            atomicAdd(&grad_conic[primitive_idx], clamped_conic.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], clamped_conic.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], clamped_conic.z);
            const float conv_factor = mip_filter ? compensated_opacity / fmaxf(original_opacity, 1e-6f) : 1.0f;
            const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
            const float dL_draw_opacity = clamp_grad(dL_draw_opacity_partial_accum * conv_factor * sigmoid_derivative);
            atomicAdd(&grad_raw_opacity[primitive_idx], dL_draw_opacity);
        }
    }

    /**
     * @brief GGGS IFT Pass 1: Compute per-pixel dT/dt_m (Equation 18 denominator).
     *
     * Tile-based forward-order kernel (thread = pixel, iterates over Gaussians).
     * Accumulates: dT_dtm[p] = sum_i -0.25 * Gt_i / (1 - Gt_i) * |t_delta_i| * rsigma_i
     * where Gt_i = alpha_i * G_exp_i, t_delta_i = (mDepth[p] - t_peak_i) * rsigma_i.
     */
    __global__ void compute_dT_dtm_depth_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float4* __restrict__ primitive_ray_plane,
        const float* __restrict__ depth_map,         // [H, W] camera-z depth from forward
        const uint* __restrict__ tile_n_contributions,
        float* __restrict__ dT_dtm_map,              // [H, W] output
        const uint width,
        const uint height,
        const uint grid_width,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        auto block = cg::this_thread_block();
        const uint tile_idx = block.group_index().y * grid_width + block.group_index().x;
        const uint thread_rank = block.thread_rank();

        const uint2 start_pixel = {block.group_index().x * config::tile_width,
                                   block.group_index().y * config::tile_height};
        const uint2 pixel_coords = {start_pixel.x + thread_rank % config::tile_width,
                                    start_pixel.y + thread_rank / config::tile_width};
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;

        const uint pixel_idx = pixel_coords.y * width + pixel_coords.x;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

        // Convert camera-z depth to ray-distance for GGGS IFT
        const float depth_camz = inside ? depth_map[pixel_idx] : 0.0f;
        const float pnf_x = (pixel.x - cx) / fx;
        const float pnf_y = (pixel.y - cy) / fy;
        const float rln = rsqrtf(pnf_x * pnf_x + pnf_y * pnf_y + 1.0f);
        const float mDepth = (rln > 0.0f) ? (depth_camz / rln) : 0.0f;

        const uint n_contributions_pixel = inside ? tile_n_contributions[pixel_idx] : 0;

        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;

        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float4 collected_ray_plane_sh[config::block_size_blend];

        float dT_dtm = 0.0f;
        uint n_possible = 0;
        bool done = !inside || n_contributions_pixel == 0 || depth_camz <= 0.0f;

        for (int n_remaining = n_points_total, fetch_idx = tile_range.x + thread_rank;
             n_remaining > 0;
             n_remaining -= config::block_size_blend, fetch_idx += config::block_size_blend) {

            if (__syncthreads_count(done) == config::block_size_blend)
                break;

            if (fetch_idx < tile_range.y) {
                const uint prim_idx = instance_primitive_indices[fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[prim_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[prim_idx];
                collected_ray_plane_sh[thread_rank] = primitive_ray_plane[prim_idx];
            }
            block.sync();

            const int batch_size = min(config::block_size_blend, n_remaining);
            for (int j = 0; !done && j < batch_size; ++j) {
                n_possible++;

                const float4 co = collected_conic_opacity[j];
                const float2 delta = collected_mean2d[j] - pixel;
                const float sigma_over_2 = 0.5f * (co.x * delta.x * delta.x + co.z * delta.y * delta.y) + co.y * delta.x * delta.y;
                if (sigma_over_2 < 0.0f) {
                    if (n_possible >= n_contributions_pixel) done = true;
                    continue;
                }
                const float gaussian = expf(-sigma_over_2);
                const float alpha = fminf(co.w * gaussian, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold) {
                    if (n_possible >= n_contributions_pixel) done = true;
                    continue;
                }

                const float4 rp = collected_ray_plane_sh[j];
                const float rsigma = rp.w;
                if (rsigma > 0.0f) {
                    const float t_peak = rp.x * delta.x + rp.y * delta.y + rp.z;
                    const float t_delta = (mDepth - t_peak) * rsigma;
                    const float G_exp = expf(-0.5f * t_delta * t_delta);
                    const float Gt = alpha * G_exp;
                    const float one_minus_Gt = fmaxf(1.0f - Gt, 1e-7f);
                    dT_dtm += -0.25f * Gt / one_minus_Gt * fabsf(t_delta) * rsigma;
                }

                if (n_possible >= n_contributions_pixel)
                    done = true;
            }
        }

        if (inside) {
            dT_dtm_map[pixel_idx] = dT_dtm;
        }
    }

    /**
     * @brief GGGS IFT: Per-pixel computation of dL/d(mDepth) / (-dT/dt_m).
     *
     * Converts grad_depth from camera-z to ray-distance, then divides by -dT/dt_m.
     */
    __global__ void compute_dL_dmt_dT_dtm_cu(
        const float* __restrict__ dT_dtm_map,
        const float* __restrict__ grad_depth_camz,
        float* __restrict__ dL_dmt_dT_dtm_map,
        const uint width,
        const uint height,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;

        const uint px = idx % width;
        const uint py = idx / width;
        const float pixel_x = static_cast<float>(px) + 0.5f;
        const float pixel_y = static_cast<float>(py) + 0.5f;
        const float pnf_x = (pixel_x - cx) / fx;
        const float pnf_y = (pixel_y - cy) / fy;
        const float rln = rsqrtf(pnf_x * pnf_x + pnf_y * pnf_y + 1.0f);

        // dL/d(mDepth) = dL/d(depth_camz) * rln  (chain: depth_camz = mDepth * rln)
        const float grad_depth_ray = grad_depth_camz[idx] * rln;
        const float dT_dtm = dT_dtm_map[idx];
        dL_dmt_dT_dtm_map[idx] = grad_depth_ray / fmaxf(-dT_dtm, 1e-7f);
    }

    /**
     * @brief GGGS IFT Pass 2: Per-Gaussian gradient computation (Equation 18).
     *
     * Bucket-based backward iteration (thread = primitive, iterates over pixels).
     * Uses precomputed dL_dmt_dT_dtm per pixel from pass 1 to compute per-Gaussian:
     *   - grad_ray_plane[N,4]: gradient w.r.t. ray_plane (plane_x*fn/fx, plane_y*fn/fy, tc, rsigma)
     *   - grad_mean2d/conic/opacity: additional gradients through 2D Gaussian alpha
     */
    __global__ void blend_backward_depth_ift_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float4* __restrict__ primitive_ray_plane,
        const float* __restrict__ raw_opacities,
        const float* __restrict__ depth_map,
        const float* __restrict__ dL_dmt_dT_dtm_map,
        const uint* __restrict__ tile_max_n_contributions,
        const uint* __restrict__ tile_n_contributions,
        const uint* __restrict__ bucket_tile_index,
        float4* __restrict__ grad_ray_plane,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_raw_opacity,
        const uint n_buckets,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const bool mip_filter) {
        auto block = cg::this_thread_block();
        const uint bucket_idx = block.group_index().x;
        if (bucket_idx >= n_buckets)
            return;
        auto warp = cg::tiled_partition<32>(block);
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_contributions[tile_idx])
            return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // Load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic_v = {0.0f, 0.0f, 0.0f};
        float compensated_opacity = 0.0f;
        float original_opacity = 0.0f;
        float4 rp = {0.0f, 0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 co = primitive_conic_opacity[primitive_idx];
            conic_v = make_float3(co);
            compensated_opacity = co.w;
            original_opacity = mip_filter ? __frcp_rn(1.0f + __expf(-raw_opacities[primitive_idx])) : compensated_opacity;
            rp = primitive_ray_plane[primitive_idx];
        }

        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        // Gradient accumulators
        float4 dL_drp_accum = {0.0f, 0.0f, 0.0f, 0.0f};
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_draw_opacity_partial_accum = 0.0f;

        uint last_contributor;
        float dL_dmt_val;
        float mDepth_pixel;

        __shared__ uint collected_last_contributor[32];
        __shared__ float2 collected_dLdmt_mDepth[32];

#pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width,
                                            start_pixel_coords.y + local_idx / config::tile_width};
                const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                const bool pixel_in_bounds = pixel_coords.x < width && pixel_coords.y < height;

                float dL_dmt_local = 0.0f;
                float mD = 0.0f;
                uint last_contrib = 0;
                if (pixel_in_bounds) {
                    dL_dmt_local = dL_dmt_dT_dtm_map[pixel_idx];
                    // Convert camera-z depth to ray-distance
                    const float depth_camz = depth_map[pixel_idx];
                    const float pixel_x = static_cast<float>(pixel_coords.x) + 0.5f;
                    const float pixel_y = static_cast<float>(pixel_coords.y) + 0.5f;
                    const float pnf_x = (pixel_x - cx) / fx;
                    const float pnf_y = (pixel_y - cy) / fy;
                    const float rln = rsqrtf(pnf_x * pnf_x + pnf_y * pnf_y + 1.0f);
                    mD = (rln > 0.0f) ? (depth_camz / rln) : 0.0f;
                    last_contrib = tile_n_contributions[pixel_idx];
                }

                collected_dLdmt_mDepth[lane_idx] = make_float2(dL_dmt_local, mD);
                collected_last_contributor[lane_idx] = last_contrib;
                __syncwarp();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                dL_dmt_val = warp.shfl_up(dL_dmt_val, 1);
                mDepth_pixel = warp.shfl_up(mDepth_pixel, 1);
            }

            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width,
                                        start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int shmem_idx = i % 32;
                last_contributor = collected_last_contributor[shmem_idx];
                const float2 dm = collected_dLdmt_mDepth[shmem_idx];
                dL_dmt_val = dm.x;
                mDepth_pixel = dm.y;
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip)
                continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float sigma_over_2 = 0.5f * (conic_v.x * delta.x * delta.x + conic_v.z * delta.y * delta.y) + conic_v.y * delta.x * delta.y;
            if (sigma_over_2 < 0.0f)
                continue;
            const float gaussian = expf(-sigma_over_2);
            const float unclamped_alpha = compensated_opacity * gaussian;
            const float alpha = fminf(unclamped_alpha, config::max_fragment_alpha);
            if (alpha < config::min_alpha_threshold)
                continue;
            const bool alpha_saturated = unclamped_alpha >= config::max_fragment_alpha;

            const float rsigma = rp.w;
            if (!(rsigma > 0.0f))
                continue;

            // GGGS IFT gradient derivation (Equation 18)
            const float t_peak = rp.x * delta.x + rp.y * delta.y + rp.z;
            const float t_delta = (mDepth_pixel - t_peak) * rsigma;
            const float G_exp = expf(-0.5f * t_delta * t_delta);
            const float Gt = alpha * G_exp;

            float dL_dGt = dL_dmt_val * 0.25f / fmaxf(1.0f - Gt, 1e-7f);
            dL_dGt = (mDepth_pixel > t_peak) ? dL_dGt : -dL_dGt;

            const float dL_dopa = dL_dGt * G_exp - dL_dmt_val * (t_delta > 0.0f ? 0.5f / fmaxf(1.0f - alpha, 1e-7f) : 0.0f);
            const float dL_ddelta_t = -dL_dGt * Gt * t_delta;

            // Ray plane gradients
            const float dL_drsigma = dL_ddelta_t * (mDepth_pixel - t_peak);
            const float dL_dt_peak = -dL_ddelta_t * rsigma;

            dL_drp_accum.x += dL_dt_peak * delta.x;
            dL_drp_accum.y += dL_dt_peak * delta.y;
            dL_drp_accum.z += dL_dt_peak;
            dL_drp_accum.w += dL_drsigma;

            // Alpha gradient through 2D Gaussian (same pattern as color backward)
            const float gaussian_grad_helper = alpha_saturated ? 0.0f : -alpha * dL_dopa;
            dL_dconic_accum += 0.5f * gaussian_grad_helper * make_float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
            dL_dmean2d_accum += gaussian_grad_helper * make_float2(
                                    conic_v.x * delta.x + conic_v.y * delta.y,
                                    conic_v.y * delta.x + conic_v.z * delta.y);
            // Mean2d gradient through t_peak
            dL_dmean2d_accum += dL_dt_peak * make_float2(rp.x, rp.y);

            // Opacity gradient
            const float dL_dcompensated_opacity = alpha_saturated ? 0.0f : gaussian * dL_dopa;
            dL_draw_opacity_partial_accum += dL_dcompensated_opacity;
        }

        // Atomic accumulate
        if (valid_primitive) {
            atomicAdd(&grad_ray_plane[primitive_idx].x, clamp_grad(dL_drp_accum.x));
            atomicAdd(&grad_ray_plane[primitive_idx].y, clamp_grad(dL_drp_accum.y));
            atomicAdd(&grad_ray_plane[primitive_idx].z, clamp_grad(dL_drp_accum.z));
            atomicAdd(&grad_ray_plane[primitive_idx].w, clamp_grad(dL_drp_accum.w));

            const float2 clamped_mean2d = make_float2(clamp_grad(dL_dmean2d_accum.x), clamp_grad(dL_dmean2d_accum.y));
            atomicAdd(&grad_mean2d[primitive_idx].x, clamped_mean2d.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, clamped_mean2d.y);
            const float3 clamped_conic = clamp_grad3(dL_dconic_accum);
            atomicAdd(&grad_conic[primitive_idx], clamped_conic.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], clamped_conic.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], clamped_conic.z);
            const float conv_factor = mip_filter ? compensated_opacity / fmaxf(original_opacity, 1e-6f) : 1.0f;
            const float sigmoid_derivative = original_opacity * (1.0f - original_opacity);
            atomicAdd(&grad_raw_opacity[primitive_idx], clamp_grad(dL_draw_opacity_partial_accum * conv_factor * sigmoid_derivative));
        }
    }

    /**
     * @brief Per-Gaussian preprocess backward for GGGS IFT depth: converts
     * grad_ray_plane[N,4] to grad_means/grad_raw_scales/grad_raw_rotations.
     *
    * Backward chain: ray_plane = (plane_x*fn/fx, plane_y*fn/fy, ray_depth, rsigma)
    *   fn = tc / ray_len2, tc = |clipped_cam_pos|, ray_depth = |cam_pos|,
    *   rsigma = sqrt(vb / ray_len2)
     *   plane = nJ_inv @ (cov_cam_inv @ uvh / vb)
     *   cov_cam_inv = M_inv^T * M_inv, M_inv = S_inv * R^T * W^T
     */
    __global__ void preprocess_backward_depth_ift_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float4* __restrict__ w2c,
        const float4* __restrict__ grad_ray_plane,
        const uint* __restrict__ primitive_n_touched_tiles,
        float3* __restrict__ grad_means,
        float3* __restrict__ grad_raw_scales,
        float4* __restrict__ grad_raw_rotations,
        const uint n_primitives,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {
        auto primitive_idx = cg::this_grid().thread_rank();
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0)
            return;

        const float4 dL_drp = grad_ray_plane[primitive_idx];
        const float grad_norm = fabsf(dL_drp.x) + fabsf(dL_drp.y) + fabsf(dL_drp.z) + fabsf(dL_drp.w);
        if (grad_norm < 1e-12f)
            return;

        // ===== Recompute forward values =====
        const float3 mean3d = means[primitive_idx];
        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < 1e-4f) return;

        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;
        const float ray_distance = norm3df(x * depth, y * depth, depth);

        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float x_grad_mul = (x >= clip_left && x <= clip_right) ? 1.0f : 0.0f;
        const float y_grad_mul = (y >= clip_top && y <= clip_bottom) ? 1.0f : 0.0f;

        // Rotation and scale
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        const float inv_s0 = rsqrtf(fmaxf(variance.x, 1e-20f));
        const float inv_s1 = rsqrtf(fmaxf(variance.y, 1e-20f));
        const float inv_s2 = rsqrtf(fmaxf(variance.z, 1e-20f));

        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float q_norm_sq = qr*qr + qx*qx + qy*qy + qz*qz;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-7f);
        const float qxx = 2.0f*qx*qx/q_norm_sq_safe, qyy = 2.0f*qy*qy/q_norm_sq_safe, qzz = 2.0f*qz*qz/q_norm_sq_safe;
        const float qxy = 2.0f*qx*qy/q_norm_sq_safe, qxz = 2.0f*qx*qz/q_norm_sq_safe, qyz = 2.0f*qy*qz/q_norm_sq_safe;
        const float qrx = 2.0f*qr*qx/q_norm_sq_safe, qry = 2.0f*qr*qy/q_norm_sq_safe, qrz = 2.0f*qr*qz/q_norm_sq_safe;
        const mat3x3 rotation = {
            1.0f - (qyy+qzz), qxy-qrz, qry+qxz,
            qrz+qxy, 1.0f - (qxx+qzz), qyz-qrx,
            qxz-qry, qrx+qyz, 1.0f - (qxx+qyy)};

        // M_inv rows: m_k = inv_s_k * R_col_k^T * W^T
        const float3 r_col0 = make_float3(rotation.m11, rotation.m21, rotation.m31);
        const float3 r_col1 = make_float3(rotation.m12, rotation.m22, rotation.m32);
        const float3 r_col2 = make_float3(rotation.m13, rotation.m23, rotation.m33);
        const float3 w_r0 = make_float3(w2c_r1.x, w2c_r1.y, w2c_r1.z);
        const float3 w_r1 = make_float3(w2c_r2.x, w2c_r2.y, w2c_r2.z);
        const float3 w_r2 = make_float3(w2c_r3.x, w2c_r3.y, w2c_r3.z);
        const float3 m0 = inv_s0 * make_float3(dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2));
        const float3 m1 = inv_s1 * make_float3(dot(r_col1, w_r0), dot(r_col1, w_r1), dot(r_col1, w_r2));
        const float3 m2 = inv_s2 * make_float3(dot(r_col2, w_r0), dot(r_col2, w_r1), dot(r_col2, w_r2));

        // cov_cam_inv = M_inv^T * M_inv
        const float cc00 = m0.x*m0.x + m1.x*m1.x + m2.x*m2.x;
        const float cc01 = m0.x*m0.y + m1.x*m1.y + m2.x*m2.y;
        const float cc02 = m0.x*m0.z + m1.x*m1.z + m2.x*m2.z;
        const float cc11 = m0.y*m0.y + m1.y*m1.y + m2.y*m2.y;
        const float cc12 = m0.y*m0.z + m1.y*m1.z + m2.y*m2.z;
        const float cc22 = m0.z*m0.z + m1.z*m1.z + m2.z*m2.z;

        const float u = tx, v = ty;
        const float uvh_m0 = cc00*u + cc01*v + cc02;
        const float uvh_m1 = cc01*u + cc11*v + cc12;
        const float uvh_m2 = cc02*u + cc12*v + cc22;
        const float vb = uvh_m0*u + uvh_m1*v + uvh_m2;
        if (!(vb > 1e-10f)) return;

        const float ray_len2 = u*u + v*v + 1.0f;
        const float tc = norm3df(u*depth, v*depth, depth);
        if (!(tc > 1e-10f)) return;

        const float inv_vb = 1.0f / vb;
        const float vec0 = uvh_m0 * inv_vb;
        const float vec1 = uvh_m1 * inv_vb;
        const float vec2 = uvh_m2 * inv_vb;

        const float u2 = u*u, v2 = v*v, uv = u*v;
        const float plane_x = (v2 + 1.0f)*vec0 - uv*vec1 - u*vec2;
        const float plane_y = -uv*vec0 + (u2 + 1.0f)*vec1 - v*vec2;

        const float fn = tc / ray_len2;
        const float rsigma = sqrtf(vb / ray_len2);
        const float ray_len_inv = rsqrtf(ray_len2);

        // ===== Backward: dL/d(ray_plane) → intermediates =====
        // rp.x = plane_x * fn / fx, rp.y = plane_y * fn / fy,
        // rp.z = ray_depth, rp.w = rsigma
        const float dL_dplane_x = dL_drp.x * fn / fx;
        const float dL_dplane_y = dL_drp.y * fn / fy;

        // rp.z is the unclipped ray depth term stored in forward.
        float dL_dtc = dL_drp.z;

        // rsigma = sqrt(vb/rl2) → dL_dvb, dL_drl2
        const float rsigma_safe = fmaxf(rsigma, 1e-10f);
        float dL_dvb = dL_drp.w / (2.0f * rsigma_safe * ray_len2);

        // ===== Backward: dL_dplane → dL_dvec → dL_duvh_m, dL_dinv_vb =====
        const float dL_dvec0 = dL_dplane_x * (v2+1.0f) + dL_dplane_y * (-uv);
        const float dL_dvec1 = dL_dplane_x * (-uv)      + dL_dplane_y * (u2+1.0f);
        const float dL_dvec2 = dL_dplane_x * (-u)        + dL_dplane_y * (-v);

        const float dL_duvh_m0 = dL_dvec0 * inv_vb;
        const float dL_duvh_m1 = dL_dvec1 * inv_vb;
        const float dL_duvh_m2 = dL_dvec2 * inv_vb;
        const float dL_dinv_vb = dL_dvec0 * uvh_m0 + dL_dvec1 * uvh_m1 + dL_dvec2 * uvh_m2;
        dL_dvb += -dL_dinv_vb * inv_vb * inv_vb;

        // ===== Backward: dL_duvh_m, dL_dvb → dL_dcc =====
        const float dL_duvh_m0_tot = dL_duvh_m0 + dL_dvb * u;
        const float dL_duvh_m1_tot = dL_duvh_m1 + dL_dvb * v;
        const float dL_duvh_m2_tot = dL_duvh_m2 + dL_dvb * 1.0f;

        const float dL_dcc00 = dL_duvh_m0_tot * u;
        const float dL_dcc01 = dL_duvh_m0_tot * v + dL_duvh_m1_tot * u;
        const float dL_dcc02 = dL_duvh_m0_tot + dL_duvh_m2_tot * u;
        const float dL_dcc11 = dL_duvh_m1_tot * v;
        const float dL_dcc12 = dL_duvh_m1_tot + dL_duvh_m2_tot * v;
        const float dL_dcc22 = dL_duvh_m2_tot;

        // ===== Backward: dL_dcc → dL_dm (M_inv rows) =====
        const float3 dL_dm0 = make_float3(
            2.0f*m0.x*dL_dcc00 + m0.y*dL_dcc01 + m0.z*dL_dcc02,
            m0.x*dL_dcc01 + 2.0f*m0.y*dL_dcc11 + m0.z*dL_dcc12,
            m0.x*dL_dcc02 + m0.y*dL_dcc12 + 2.0f*m0.z*dL_dcc22);
        const float3 dL_dm1 = make_float3(
            2.0f*m1.x*dL_dcc00 + m1.y*dL_dcc01 + m1.z*dL_dcc02,
            m1.x*dL_dcc01 + 2.0f*m1.y*dL_dcc11 + m1.z*dL_dcc12,
            m1.x*dL_dcc02 + m1.y*dL_dcc12 + 2.0f*m1.z*dL_dcc22);
        const float3 dL_dm2 = make_float3(
            2.0f*m2.x*dL_dcc00 + m2.y*dL_dcc01 + m2.z*dL_dcc02,
            m2.x*dL_dcc01 + 2.0f*m2.y*dL_dcc11 + m2.z*dL_dcc12,
            m2.x*dL_dcc02 + m2.y*dL_dcc12 + 2.0f*m2.z*dL_dcc22);

        // ===== Backward: dL_dm → dL_dinv_s, dL_dr_col =====
        const float dL_dinv_s0 = dot(dL_dm0, make_float3(dot(r_col0, w_r0), dot(r_col0, w_r1), dot(r_col0, w_r2)));
        const float dL_dinv_s1 = dot(dL_dm1, make_float3(dot(r_col1, w_r0), dot(r_col1, w_r1), dot(r_col1, w_r2)));
        const float dL_dinv_s2 = dot(dL_dm2, make_float3(dot(r_col2, w_r0), dot(r_col2, w_r1), dot(r_col2, w_r2)));

        const float3 dL_dr_col0 = inv_s0 * (dL_dm0.x * w_r0 + dL_dm0.y * w_r1 + dL_dm0.z * w_r2);
        const float3 dL_dr_col1 = inv_s1 * (dL_dm1.x * w_r0 + dL_dm1.y * w_r1 + dL_dm1.z * w_r2);
        const float3 dL_dr_col2 = inv_s2 * (dL_dm2.x * w_r0 + dL_dm2.y * w_r1 + dL_dm2.z * w_r2);

        // ===== Scale gradient: inv_s = 1/sqrt(variance) = 1/exp(raw_scale) =====
        const float3 dL_draw_scale_from_depth = make_float3(
            (raw_scale.x < config::max_raw_scale) ? -dL_dinv_s0 * inv_s0 : 0.0f,
            (raw_scale.y < config::max_raw_scale) ? -dL_dinv_s1 * inv_s1 : 0.0f,
            (raw_scale.z < config::max_raw_scale) ? -dL_dinv_s2 * inv_s2 : 0.0f);
        grad_raw_scales[primitive_idx] += clamp_grad3(dL_draw_scale_from_depth);

        // ===== Rotation gradient =====
        const mat3x3 dL_drotation_from_depth = {
            dL_dr_col0.x, dL_dr_col1.x, dL_dr_col2.x,
            dL_dr_col0.y, dL_dr_col1.y, dL_dr_col2.y,
            dL_dr_col0.z, dL_dr_col1.z, dL_dr_col2.z};

        const float dL_dqxx_n = -dL_drotation_from_depth.m22 - dL_drotation_from_depth.m33;
        const float dL_dqyy_n = -dL_drotation_from_depth.m11 - dL_drotation_from_depth.m33;
        const float dL_dqzz_n = -dL_drotation_from_depth.m11 - dL_drotation_from_depth.m22;
        const float dL_dqxy_n = dL_drotation_from_depth.m12 + dL_drotation_from_depth.m21;
        const float dL_dqxz_n = dL_drotation_from_depth.m13 + dL_drotation_from_depth.m31;
        const float dL_dqyz_n = dL_drotation_from_depth.m23 + dL_drotation_from_depth.m32;
        const float dL_dqrx_n = dL_drotation_from_depth.m32 - dL_drotation_from_depth.m23;
        const float dL_dqry_n = dL_drotation_from_depth.m13 - dL_drotation_from_depth.m31;
        const float dL_dqrz_n = dL_drotation_from_depth.m21 - dL_drotation_from_depth.m12;
        const float dL_dq_norm_n = qxx*dL_dqxx_n + qyy*dL_dqyy_n + qzz*dL_dqzz_n + qxy*dL_dqxy_n + qxz*dL_dqxz_n + qyz*dL_dqyz_n + qrx*dL_dqrx_n + qry*dL_dqry_n + qrz*dL_dqrz_n;
        const float4 dL_draw_rotation_from_depth = 2.0f * make_float4(
            qx*dL_dqrx_n + qy*dL_dqry_n + qz*dL_dqrz_n - qr*dL_dq_norm_n,
            2.0f*qx*dL_dqxx_n + qy*dL_dqxy_n + qz*dL_dqxz_n + qr*dL_dqrx_n - qx*dL_dq_norm_n,
            2.0f*qy*dL_dqyy_n + qx*dL_dqxy_n + qz*dL_dqyz_n + qr*dL_dqry_n - qy*dL_dq_norm_n,
            2.0f*qz*dL_dqzz_n + qx*dL_dqxz_n + qy*dL_dqyz_n + qr*dL_dqrz_n - qz*dL_dq_norm_n) / q_norm_sq_safe;
        grad_raw_rotations[primitive_idx] += clamp_grad4(dL_draw_rotation_from_depth);

        // ===== Mean gradient (GGGS computeCov2DCUDA full u/v/z chain) =====
        const float dL_dplane_x_mean = dL_drp.x * fn / fx;
        const float dL_dplane_y_mean = dL_drp.y * fn / fy;
        const float dL_dfactor_normal = plane_x * dL_drp.x / fx + plane_y * dL_drp.y / fy;
        const float aux = dL_dplane_x_mean * plane_x + dL_dplane_y_mean * plane_y;
        const float nJ_inv_t_dp0 = (v2 + 1.0f) * dL_dplane_x_mean - uv * dL_dplane_y_mean;
        const float nJ_inv_t_dp1 = -uv * dL_dplane_x_mean + (u2 + 1.0f) * dL_dplane_y_mean;
        const float nJ_inv_t_dp2 = -u * dL_dplane_x_mean - v * dL_dplane_y_mean;
        const float dL_duvh_plane0 = 2.0f * (-aux) * vec0 + (cc00 * nJ_inv_t_dp0 + cc01 * nJ_inv_t_dp1 + cc02 * nJ_inv_t_dp2) * inv_vb;
        const float dL_duvh_plane1 = 2.0f * (-aux) * vec1 + (cc01 * nJ_inv_t_dp0 + cc11 * nJ_inv_t_dp1 + cc12 * nJ_inv_t_dp2) * inv_vb;
        const float dL_du_plane = dL_duvh_plane0 + (dL_dplane_x_mean * vec1 + dL_dplane_y_mean * vec0) * (-v) + 2.0f * dL_dplane_y_mean * vec1 * u - dL_dplane_x_mean * vec2;
        const float dL_dv_plane = dL_duvh_plane1 + (dL_dplane_x_mean * vec1 + dL_dplane_y_mean * vec0) * (-u) + 2.0f * dL_dplane_x_mean * vec0 * v - dL_dplane_y_mean * vec2;

        const float dL_dray_len2_sigma_x2 = -dL_drp.w * rsigma / ray_len2;
        const float dL_du_sigma = dL_dray_len2_sigma_x2 * u;
        const float dL_dv_sigma = dL_dray_len2_sigma_x2 * v;
        const float aux_factor = dL_dfactor_normal * (-depth / ray_len2 * ray_len_inv);
        const float dL_du_factor = aux_factor * u;
        const float dL_dv_factor = aux_factor * v;
        const float dL_dz_factor = dL_dfactor_normal * ray_len_inv;

        const float dL_du = dL_du_plane + dL_du_factor + dL_du_sigma;
        const float dL_dv = dL_dv_plane + dL_dv_factor + dL_dv_sigma;
        const float dL_dz = dL_dz_factor;

        const float t_clipped_x = u * depth;
        const float t_clipped_y = v * depth;
        const float depth_sq = depth * depth;
        const float dL_dtx = x_grad_mul * dL_du / depth;
        const float dL_dty = y_grad_mul * dL_dv / depth;
        const float dL_dtz = -(x_grad_mul * dL_du * t_clipped_x + y_grad_mul * dL_dv * t_clipped_y) / depth_sq + dL_dz;

        const float ray_distance_safe = fmaxf(ray_distance, 1e-10f);
        const float inv_ray_distance = 1.0f / ray_distance_safe;
        const float dL_dt_x = x * depth * inv_ray_distance * dL_dtc;
        const float dL_dt_y = y * depth * inv_ray_distance * dL_dtc;
        const float dL_dt_z = depth * inv_ray_distance * dL_dtc;

        const float3 dL_dmean3d_from_depth = make_float3(
            w2c_r1.x * (dL_dtx + dL_dt_x) + w2c_r2.x * (dL_dty + dL_dt_y) + w2c_r3.x * (dL_dtz + dL_dt_z),
            w2c_r1.y * (dL_dtx + dL_dt_x) + w2c_r2.y * (dL_dty + dL_dt_y) + w2c_r3.y * (dL_dtz + dL_dt_z),
            w2c_r1.z * (dL_dtx + dL_dt_x) + w2c_r2.z * (dL_dty + dL_dt_y) + w2c_r3.z * (dL_dtz + dL_dt_z));
        grad_means[primitive_idx] += clamp_grad3(dL_dmean3d_from_depth);
    }

} // namespace fast_lfs::rasterization::kernels::backward
