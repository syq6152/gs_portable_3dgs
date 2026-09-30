/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_init_rgb_color_transfer.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace lfs::training::kernels {

    namespace {
        constexpr int THREADS = 256;

        __device__ __forceinline__ float bilinear_sample_hw(
            const float* image,
            const int width,
            const int height,
            const float x,
            const float y) {
            const float clamped_x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
            const float clamped_y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
            const int x0 = static_cast<int>(floorf(clamped_x));
            const int y0 = static_cast<int>(floorf(clamped_y));
            const int x1 = x0 + 1 < width ? x0 + 1 : width - 1;
            const int y1 = y0 + 1 < height ? y0 + 1 : height - 1;
            const float tx = clamped_x - static_cast<float>(x0);
            const float ty = clamped_y - static_cast<float>(y0);
            const float v00 = image[y0 * width + x0];
            const float v01 = image[y0 * width + x1];
            const float v10 = image[y1 * width + x0];
            const float v11 = image[y1 * width + x1];
            const float top = v00 * (1.0f - tx) + v01 * tx;
            const float bottom = v10 * (1.0f - tx) + v11 * tx;
            return top * (1.0f - ty) + bottom * ty;
        }

        __global__ void mesh_init_rgb_accumulate_kernel(
            const float* means_xyz,
            const int point_count,
            const float* source_depth,
            const float* source_rgb,
            const float* source_mask,
            const int width,
            const int height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float* source_w2c_4x4,
            const float scene_radius,
            const float rel_depth_epsilon,
            const float source_mask_threshold,
            float* rgb_sum,
            float* weight_sum,
            int32_t* counters) {
            const int point = blockIdx.x * blockDim.x + threadIdx.x;
            if (point >= point_count) {
                return;
            }

            // RGB initialization is intentionally coarse: the first source
            // camera that passes all visibility checks wins. Cameras are
            // launched sequentially by the host, so one thread owns this row
            // within a launch and no atomic compare-and-swap is required here.
            if (weight_sum[point] > 0.0f) {
                return;
            }

            const float wx = means_xyz[point * 3 + 0];
            const float wy = means_xyz[point * 3 + 1];
            const float wz = means_xyz[point * 3 + 2];
            const float sx = source_w2c_4x4[0] * wx + source_w2c_4x4[1] * wy +
                             source_w2c_4x4[2] * wz + source_w2c_4x4[3];
            const float sy = source_w2c_4x4[4] * wx + source_w2c_4x4[5] * wy +
                             source_w2c_4x4[6] * wz + source_w2c_4x4[7];
            const float sz = source_w2c_4x4[8] * wx + source_w2c_4x4[9] * wy +
                             source_w2c_4x4[10] * wz + source_w2c_4x4[11];
            if (!(sz > 1e-6f) || !isfinite(sz)) {
                atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::ProjectionFail), 1);
                return;
            }

            const float u = fx * (sx / sz) + cx;
            const float v = fy * (sy / sz) + cy;
            if (!(u >= 0.0f && v >= 0.0f &&
                  u <= static_cast<float>(width - 1) &&
                  v <= static_cast<float>(height - 1)) ||
                !isfinite(u) || !isfinite(v)) {
                atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::ProjectionFail), 1);
                return;
            }
            atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::Projected), 1);

            const float source_z = bilinear_sample_hw(source_depth, width, height, u, v);
            const float abs_eps = fmaxf(1e-6f, 1e-3f * fmaxf(scene_radius, 1e-6f));
            const float depth_eps = fmaxf(abs_eps, rel_depth_epsilon * fabsf(sz));
            if (!(source_z > 0.0f) || !isfinite(source_z) || fabsf(source_z - sz) > depth_eps) {
                atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::OcclusionFail), 1);
                return;
            }

            if (source_mask) {
                const float mask_value = bilinear_sample_hw(source_mask, width, height, u, v);
                if (!(mask_value >= source_mask_threshold) || !isfinite(mask_value)) {
                    atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::SourceMaskFail), 1);
                    return;
                }
            }

            const int pixels = width * height;
            const float r = bilinear_sample_hw(source_rgb, width, height, u, v);
            const float g = bilinear_sample_hw(source_rgb + pixels, width, height, u, v);
            const float b = bilinear_sample_hw(source_rgb + pixels * 2, width, height, u, v);
            if (!isfinite(r) || !isfinite(g) || !isfinite(b)) {
                atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::SourceMaskFail), 1);
                return;
            }

            rgb_sum[point * 3 + 0] = r;
            rgb_sum[point * 3 + 1] = g;
            rgb_sum[point * 3 + 2] = b;
            weight_sum[point] = 1.0f;
            atomicAdd(counters + static_cast<int>(MeshInitRgbCounter::Valid), 1);
        }
    } // namespace

    void launch_mesh_init_rgb_accumulate(
        const float* means_xyz,
        const int point_count,
        const float* source_depth,
        const float* source_rgb,
        const float* source_mask,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float* source_w2c_4x4,
        const float scene_radius,
        const float rel_depth_epsilon,
        const float source_mask_threshold,
        float* rgb_sum,
        float* weight_sum,
        int32_t* counters,
        cudaStream_t stream) {
        if (point_count <= 0 || width <= 0 || height <= 0) {
            return;
        }
        const int blocks = (point_count + THREADS - 1) / THREADS;
        mesh_init_rgb_accumulate_kernel<<<blocks, THREADS, 0, stream>>>(
            means_xyz,
            point_count,
            source_depth,
            source_rgb,
            source_mask,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            source_w2c_4x4,
            scene_radius,
            rel_depth_epsilon,
            source_mask_threshold,
            rgb_sum,
            weight_sum,
            counters);
    }

} // namespace lfs::training::kernels
