/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pseudo_view_reprojection.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

namespace lfs::training::kernels {

    namespace {

        __device__ __forceinline__ float bilinear_sample_hw(
            const float* image,
            const int width,
            const int height,
            const float x,
            const float y) {

            const float x_clamped = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
            const float y_clamped = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
            const int x0 = static_cast<int>(floorf(x_clamped));
            const int y0 = static_cast<int>(floorf(y_clamped));
            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);
            const float tx = x_clamped - static_cast<float>(x0);
            const float ty = y_clamped - static_cast<float>(y0);

            const float v00 = image[y0 * width + x0];
            const float v10 = image[y0 * width + x1];
            const float v01 = image[y1 * width + x0];
            const float v11 = image[y1 * width + x1];
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            return v0 * (1.0f - ty) + v1 * ty;
        }

        __global__ void pseudo_view_reproject_rgb_kernel(
            const float* pseudo_depth,
            const float* pseudo_mesh_mask,
            const int32_t* pseudo_triangle_id,
            const int pseudo_width,
            const int pseudo_height,
            const float pseudo_fx,
            const float pseudo_fy,
            const float pseudo_cx,
            const float pseudo_cy,
            const float* pseudo_c2w_4x4,
            const float* source_depth,
            const float* source_rgb,
            const float* source_mask,
            const int source_width,
            const int source_height,
            const float source_fx,
            const float source_fy,
            const float source_cx,
            const float source_cy,
            const float* source_w2c_4x4,
            const float* face_normals,
            const int face_count,
            const float source_camera_x,
            const float source_camera_y,
            const float source_camera_z,
            const float source_face_angle_min_cos,
            const float scene_radius,
            const float rel_depth_epsilon,
            const float source_mask_threshold,
            float* pseudo_rgb_out,
            float* valid_mask_out,
            float* mesh_mask_out,
            int32_t* counters) {

            const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int pseudo_numel = pseudo_width * pseudo_height;
            if (pixel_idx >= pseudo_numel) {
                return;
            }

            const bool write_outputs = pseudo_rgb_out && valid_mask_out && mesh_mask_out && source_rgb;
            const bool source_angle_gate =
                pseudo_triangle_id && face_normals && face_count > 0 && source_face_angle_min_cos > -0.999999f;
            if (write_outputs) {
                pseudo_rgb_out[pixel_idx] = 0.0f;
                pseudo_rgb_out[pseudo_numel + pixel_idx] = 0.0f;
                pseudo_rgb_out[pseudo_numel * 2 + pixel_idx] = 0.0f;
                valid_mask_out[pixel_idx] = 0.0f;
                mesh_mask_out[pixel_idx] = 0.0f;
            }

            const float z_p = pseudo_depth[pixel_idx];
            if (!(z_p > 0.0f) || !isfinite(z_p)) {
                return;
            }
            if (pseudo_mesh_mask) {
                const float mask_value = pseudo_mesh_mask[pixel_idx];
                if (!(mask_value > 0.5f) || !isfinite(mask_value)) {
                    return;
                }
            }

            if (write_outputs) {
                mesh_mask_out[pixel_idx] = 1.0f;
            }
            atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::MeshValid), 1);

            const int py = pixel_idx / pseudo_width;
            const int px = pixel_idx - py * pseudo_width;
            const float x_p = (static_cast<float>(px) - pseudo_cx) * z_p / pseudo_fx;
            const float y_p = (static_cast<float>(py) - pseudo_cy) * z_p / pseudo_fy;

            const float wx = pseudo_c2w_4x4[0] * x_p + pseudo_c2w_4x4[1] * y_p + pseudo_c2w_4x4[2] * z_p + pseudo_c2w_4x4[3];
            const float wy = pseudo_c2w_4x4[4] * x_p + pseudo_c2w_4x4[5] * y_p + pseudo_c2w_4x4[6] * z_p + pseudo_c2w_4x4[7];
            const float wz = pseudo_c2w_4x4[8] * x_p + pseudo_c2w_4x4[9] * y_p + pseudo_c2w_4x4[10] * z_p + pseudo_c2w_4x4[11];

            if (source_angle_gate) {
                const int32_t face_id = pseudo_triangle_id[pixel_idx];
                if (face_id < 0 || face_id >= face_count) {
                    atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::SourceFaceAngleFail), 1);
                    return;
                }
                const float nx = face_normals[face_id * 3 + 0];
                const float ny = face_normals[face_id * 3 + 1];
                const float nz = face_normals[face_id * 3 + 2];
                const float vx = source_camera_x - wx;
                const float vy = source_camera_y - wy;
                const float vz = source_camera_z - wz;
                const float v_len = sqrtf(vx * vx + vy * vy + vz * vz);
                if (!(v_len > 1e-8f) || !isfinite(v_len)) {
                    atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::SourceFaceAngleFail), 1);
                    return;
                }
                const float source_alignment = (nx * vx + ny * vy + nz * vz) / v_len;
                if (!(source_alignment >= source_face_angle_min_cos) || !isfinite(source_alignment)) {
                    atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::SourceFaceAngleFail), 1);
                    return;
                }
            }

            const float sx = source_w2c_4x4[0] * wx + source_w2c_4x4[1] * wy + source_w2c_4x4[2] * wz + source_w2c_4x4[3];
            const float sy = source_w2c_4x4[4] * wx + source_w2c_4x4[5] * wy + source_w2c_4x4[6] * wz + source_w2c_4x4[7];
            const float sz = source_w2c_4x4[8] * wx + source_w2c_4x4[9] * wy + source_w2c_4x4[10] * wz + source_w2c_4x4[11];

            if (!(sz > 1e-6f) || !isfinite(sz)) {
                atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::ProjectionFail), 1);
                return;
            }

            const float u = source_fx * (sx / sz) + source_cx;
            const float v = source_fy * (sy / sz) + source_cy;
            if (!(u >= 0.0f && v >= 0.0f &&
                  u <= static_cast<float>(source_width - 1) &&
                  v <= static_cast<float>(source_height - 1)) ||
                !isfinite(u) || !isfinite(v)) {
                atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::ProjectionFail), 1);
                return;
            }

            const float source_z = bilinear_sample_hw(source_depth, source_width, source_height, u, v);
            const float abs_eps = fmaxf(1e-6f, 1e-3f * fmaxf(scene_radius, 1e-6f));
            const float depth_eps = fmaxf(abs_eps, rel_depth_epsilon * fabsf(sz));
            if (!(source_z > 0.0f) || !isfinite(source_z) || fabsf(source_z - sz) > depth_eps) {
                atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::OcclusionFail), 1);
                return;
            }

            if (source_mask) {
                const float mask_value = bilinear_sample_hw(source_mask, source_width, source_height, u, v);
                if (!(mask_value >= source_mask_threshold) || !isfinite(mask_value)) {
                    atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::SourceMaskFail), 1);
                    return;
                }
            }

            if (write_outputs) {
                const int source_numel = source_width * source_height;
                const float r = bilinear_sample_hw(source_rgb, source_width, source_height, u, v);
                const float g = bilinear_sample_hw(source_rgb + source_numel, source_width, source_height, u, v);
                const float b = bilinear_sample_hw(source_rgb + source_numel * 2, source_width, source_height, u, v);

                pseudo_rgb_out[pixel_idx] = r;
                pseudo_rgb_out[pseudo_numel + pixel_idx] = g;
                pseudo_rgb_out[pseudo_numel * 2 + pixel_idx] = b;
                valid_mask_out[pixel_idx] = 1.0f;
            }
            atomicAdd(counters + static_cast<int>(PseudoViewReprojectionCounter::Valid), 1);
        }

        __global__ void pseudo_view_front_face_mask_kernel(
            const int32_t* triangle_id,
            const int width,
            const int height,
            const float* face_centers,
            const float* face_normals,
            const int face_count,
            const float camera_x,
            const float camera_y,
            const float camera_z,
            const float front_face_min_cos,
            float* mask_out,
            int32_t* counters) {
            const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int numel = width * height;
            if (pixel_idx >= numel) {
                return;
            }

            mask_out[pixel_idx] = 0.0f;
            const int32_t face_id = triangle_id[pixel_idx];
            if (face_id < 0 || face_id >= face_count) {
                return;
            }

            atomicAdd(counters + static_cast<int>(PseudoViewFrontFaceCounter::Mesh), 1);

            const float cx = face_centers[face_id * 3 + 0];
            const float cy = face_centers[face_id * 3 + 1];
            const float cz = face_centers[face_id * 3 + 2];
            const float nx = face_normals[face_id * 3 + 0];
            const float ny = face_normals[face_id * 3 + 1];
            const float nz = face_normals[face_id * 3 + 2];
            const float vx = camera_x - cx;
            const float vy = camera_y - cy;
            const float vz = camera_z - cz;
            const float v_len = sqrtf(vx * vx + vy * vy + vz * vz);
            float alignment = 1.0f;
            if (v_len > 1e-8f && isfinite(v_len)) {
                alignment = (nx * vx + ny * vy + nz * vz) / v_len;
            }

            if (alignment >= front_face_min_cos && isfinite(alignment)) {
                mask_out[pixel_idx] = 1.0f;
                atomicAdd(counters + static_cast<int>(PseudoViewFrontFaceCounter::Front), 1);
            } else {
                atomicAdd(counters + static_cast<int>(PseudoViewFrontFaceCounter::Back), 1);
            }
        }

        __global__ void pseudo_view_apply_valid_mask_rgb_kernel(
            float* rgb,
            const float* valid_mask,
            const int width,
            const int height) {
            const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int numel = width * height;
            if (pixel_idx >= numel) {
                return;
            }

            const float m = valid_mask[pixel_idx];
            rgb[pixel_idx] *= m;
            rgb[numel + pixel_idx] *= m;
            rgb[numel * 2 + pixel_idx] *= m;
        }
    } // namespace

    void launch_pseudo_view_front_face_mask(
        const int32_t* triangle_id,
        const int width,
        const int height,
        const float* face_centers,
        const float* face_normals,
        const int face_count,
        const float camera_x,
        const float camera_y,
        const float camera_z,
        const float front_face_min_cos,
        float* mask_out,
        int32_t* counters,
        cudaStream_t stream) {
        if (!triangle_id || !face_centers || !face_normals || !mask_out || !counters ||
            width <= 0 || height <= 0 || face_count <= 0) {
            return;
        }

        constexpr int threads = 256;
        const int pixel_count = width * height;
        const int blocks = (pixel_count + threads - 1) / threads;
        pseudo_view_front_face_mask_kernel<<<blocks, threads, 0, stream>>>(
            triangle_id,
            width,
            height,
            face_centers,
            face_normals,
            face_count,
            camera_x,
            camera_y,
            camera_z,
            front_face_min_cos,
            mask_out,
            counters);
    }

    void launch_pseudo_view_apply_valid_mask_rgb(
        float* rgb,
        const float* valid_mask,
        const int width,
        const int height,
        cudaStream_t stream) {
        if (!rgb || !valid_mask || width <= 0 || height <= 0) {
            return;
        }

        constexpr int threads = 256;
        const int pixel_count = width * height;
        const int blocks = (pixel_count + threads - 1) / threads;
        pseudo_view_apply_valid_mask_rgb_kernel<<<blocks, threads, 0, stream>>>(
            rgb,
            valid_mask,
            width,
            height);
    }

    void launch_pseudo_view_reproject_rgb(
        const float* pseudo_depth,
        const float* pseudo_mesh_mask,
        const int32_t* pseudo_triangle_id,
        const int pseudo_width,
        const int pseudo_height,
        const float pseudo_fx,
        const float pseudo_fy,
        const float pseudo_cx,
        const float pseudo_cy,
        const float* pseudo_c2w_4x4,
        const float* source_depth,
        const float* source_rgb,
        const float* source_mask,
        const int source_width,
        const int source_height,
        const float source_fx,
        const float source_fy,
        const float source_cx,
        const float source_cy,
        const float* source_w2c_4x4,
        const float* face_normals,
        const int face_count,
        const float source_camera_x,
        const float source_camera_y,
        const float source_camera_z,
        const float source_face_angle_min_cos,
        const float scene_radius,
        const float rel_depth_epsilon,
        const float source_mask_threshold,
        float* pseudo_rgb_out,
        float* valid_mask_out,
        float* mesh_mask_out,
        int32_t* counters,
        cudaStream_t stream) {

        if (!pseudo_depth || !pseudo_c2w_4x4 || !source_depth || !source_rgb || !source_w2c_4x4 ||
            !pseudo_rgb_out || !valid_mask_out || !mesh_mask_out || !counters ||
            pseudo_width <= 0 || pseudo_height <= 0 || source_width <= 0 || source_height <= 0) {
            return;
        }

        constexpr int threads = 256;
        const int pixel_count = pseudo_width * pseudo_height;
        const int blocks = (pixel_count + threads - 1) / threads;
        pseudo_view_reproject_rgb_kernel<<<blocks, threads, 0, stream>>>(
            pseudo_depth,
            pseudo_mesh_mask,
            pseudo_triangle_id,
            pseudo_width,
            pseudo_height,
            pseudo_fx,
            pseudo_fy,
            pseudo_cx,
            pseudo_cy,
            pseudo_c2w_4x4,
            source_depth,
            source_rgb,
            source_mask,
            source_width,
            source_height,
            source_fx,
            source_fy,
            source_cx,
            source_cy,
            source_w2c_4x4,
            face_normals,
            face_count,
            source_camera_x,
            source_camera_y,
            source_camera_z,
            source_face_angle_min_cos,
            scene_radius,
            rel_depth_epsilon,
            source_mask_threshold,
            pseudo_rgb_out,
            valid_mask_out,
            mesh_mask_out,
            counters);
    }

    void launch_pseudo_view_reproject_score(
        const float* pseudo_depth,
        const float* pseudo_mesh_mask,
        const int32_t* pseudo_triangle_id,
        const int pseudo_width,
        const int pseudo_height,
        const float pseudo_fx,
        const float pseudo_fy,
        const float pseudo_cx,
        const float pseudo_cy,
        const float* pseudo_c2w_4x4,
        const float* source_depth,
        const float* source_mask,
        const int source_width,
        const int source_height,
        const float source_fx,
        const float source_fy,
        const float source_cx,
        const float source_cy,
        const float* source_w2c_4x4,
        const float* face_normals,
        const int face_count,
        const float source_camera_x,
        const float source_camera_y,
        const float source_camera_z,
        const float source_face_angle_min_cos,
        const float scene_radius,
        const float rel_depth_epsilon,
        const float source_mask_threshold,
        int32_t* counters,
        cudaStream_t stream) {

        if (!pseudo_depth || !pseudo_c2w_4x4 || !source_depth || !source_w2c_4x4 || !counters ||
            pseudo_width <= 0 || pseudo_height <= 0 || source_width <= 0 || source_height <= 0) {
            return;
        }

        constexpr int threads = 256;
        const int pixel_count = pseudo_width * pseudo_height;
        const int blocks = (pixel_count + threads - 1) / threads;
        pseudo_view_reproject_rgb_kernel<<<blocks, threads, 0, stream>>>(
            pseudo_depth,
            pseudo_mesh_mask,
            pseudo_triangle_id,
            pseudo_width,
            pseudo_height,
            pseudo_fx,
            pseudo_fy,
            pseudo_cx,
            pseudo_cy,
            pseudo_c2w_4x4,
            source_depth,
            nullptr,
            source_mask,
            source_width,
            source_height,
            source_fx,
            source_fy,
            source_cx,
            source_cy,
            source_w2c_4x4,
            face_normals,
            face_count,
            source_camera_x,
            source_camera_y,
            source_camera_z,
            source_face_angle_min_cos,
            scene_radius,
            rel_depth_epsilon,
            source_mask_threshold,
            nullptr,
            nullptr,
            nullptr,
            counters);
    }

} // namespace lfs::training::kernels
