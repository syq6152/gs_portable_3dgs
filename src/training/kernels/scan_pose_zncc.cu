/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scan_pose_zncc.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <math_functions.h>

namespace lfs::training::kernels {
    namespace {

        constexpr int MAX_SAMPLES_PER_REGION = 128;
        constexpr int PATCH_SIZE = 11;

        __device__ __forceinline__ bool finite_positive(const float value) {
            return isfinite(value) && value > 0.0f;
        }

        __device__ __forceinline__ bool inside_index_space(
            const float x,
            const float y,
            const int width,
            const int height) {
            return isfinite(x) && isfinite(y) &&
                   x >= 0.0f && y >= 0.0f &&
                   x <= static_cast<float>(width - 1) &&
                   y <= static_cast<float>(height - 1);
        }

        __device__ __forceinline__ float bilinear_plane(
            const float* plane,
            const int width,
            const int height,
            const float x,
            const float y) {
            const int x0 = static_cast<int>(floorf(x));
            const int y0 = static_cast<int>(floorf(y));
            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);
            const float tx = x - static_cast<float>(x0);
            const float ty = y - static_cast<float>(y0);
            const float v00 = plane[y0 * width + x0];
            const float v10 = plane[y0 * width + x1];
            const float v01 = plane[y1 * width + x0];
            const float v11 = plane[y1 * width + x1];
            const float v0 = v00 * (1.0f - tx) + v10 * tx;
            const float v1 = v01 * (1.0f - tx) + v11 * tx;
            return v0 * (1.0f - ty) + v1 * ty;
        }

        __device__ __forceinline__ float luminance_at_pixel(
            const float* rgb,
            const int plane_size,
            const int index) {
            return 0.2126f * rgb[index] +
                   0.7152f * rgb[plane_size + index] +
                   0.0722f * rgb[2 * plane_size + index];
        }

        __device__ __forceinline__ float luminance_bilinear(
            const float* rgb,
            const int plane_size,
            const int width,
            const int height,
            const float x,
            const float y) {
            return 0.2126f * bilinear_plane(rgb, width, height, x, y) +
                   0.7152f * bilinear_plane(rgb + plane_size, width, height, x, y) +
                   0.0722f * bilinear_plane(rgb + 2 * plane_size, width, height, x, y);
        }

        __global__ void scan_pose_zncc_regions_kernel(
            const float* target_rgb,
            const float* target_depth,
            const float* target_c2w,
            const int target_width,
            const int target_height,
            const float target_fx,
            const float target_fy,
            const float target_cx,
            const float target_cy,
            const float* source_rgb,
            const float* source_w2c,
            const int source_width,
            const int source_height,
            const float source_fx,
            const float source_fy,
            const float source_cx,
            const float source_cy,
            const int region_count_x,
            const int region_count_y,
            const int samples_per_axis,
            const int min_valid_samples,
            const float variance_epsilon,
            float* region_zncc,
            int32_t* region_photometric_samples,
            int32_t* region_target_samples,
            int32_t* region_projected_samples,
            int32_t* region_depth_consistent_samples) {

            __shared__ float target_values[MAX_SAMPLES_PER_REGION];
            __shared__ float source_values[MAX_SAMPLES_PER_REGION];
            __shared__ int32_t target_supported_flags[MAX_SAMPLES_PER_REGION];
            __shared__ int32_t projected_flags[MAX_SAMPLES_PER_REGION];
            __shared__ int32_t photo_valid_flags[MAX_SAMPLES_PER_REGION];
            __shared__ int32_t depth_consistent_flags[MAX_SAMPLES_PER_REGION];

            const int region_index = static_cast<int>(blockIdx.x);
            const int region_count = region_count_x * region_count_y;
            if (region_index >= region_count) {
                return;
            }

            const int thread_index = static_cast<int>(threadIdx.x);
            if (thread_index < MAX_SAMPLES_PER_REGION) {
                target_values[thread_index] = 0.0f;
                source_values[thread_index] = 0.0f;
                target_supported_flags[thread_index] = 0;
                projected_flags[thread_index] = 0;
                photo_valid_flags[thread_index] = 0;
                depth_consistent_flags[thread_index] = 0;
            }
            __syncthreads();

            const int sample_count = samples_per_axis * samples_per_axis;
            if (thread_index < sample_count) {
                const int region_x = region_index % region_count_x;
                const int region_y = region_index / region_count_x;
                const int x_begin = (region_x * target_width) / region_count_x;
                const int x_end = ((region_x + 1) * target_width) / region_count_x;
                const int y_begin = (region_y * target_height) / region_count_y;
                const int y_end = ((region_y + 1) * target_height) / region_count_y;
                const int center_x = (x_begin + x_end - 1) / 2;
                const int center_y = (y_begin + y_end - 1) / 2;
                // Keep the local window at its full size whenever the working
                // image is large enough. Merely clipping the window at an edge
                // would make the regular lattice repeat integer pixels.
                const int patch_begin_x = min(
                    max(0, center_x - PATCH_SIZE / 2),
                    target_width - PATCH_SIZE);
                const int patch_begin_y = min(
                    max(0, center_y - PATCH_SIZE / 2),
                    target_height - PATCH_SIZE);
                const int patch_width = PATCH_SIZE;
                const int patch_height = PATCH_SIZE;

                const int sample_x = thread_index % samples_per_axis;
                const int sample_y = thread_index / samples_per_axis;
                const int px = patch_begin_x + min(
                                                   patch_width - 1,
                                                   samples_per_axis > 1
                                                       ? (sample_x * (patch_width - 1) + (samples_per_axis - 1) / 2) /
                                                             (samples_per_axis - 1)
                                                       : patch_width / 2);
                const int py = patch_begin_y + min(
                                                   patch_height - 1,
                                                   samples_per_axis > 1
                                                       ? (sample_y * (patch_height - 1) + (samples_per_axis - 1) / 2) /
                                                             (samples_per_axis - 1)
                                                       : patch_height / 2);

                const int target_plane_size = target_width * target_height;
                const int target_index = py * target_width + px;
                const float z = target_depth[target_index];
                const bool target_supported =
                    finite_positive(z) &&
                    isfinite(target_fx) && target_fx > 0.0f &&
                    isfinite(target_fy) && target_fy > 0.0f &&
                    isfinite(source_fx) && source_fx > 0.0f &&
                    isfinite(source_fy) && source_fy > 0.0f;
                bool projected = false;
                bool photo_valid = false;

                float source_luma = 0.0f;
                if (target_supported) {
                    // The depth rasterizer uses pixel centers. Keep this offset
                    // explicit so an identity c2w/w2c pair maps to the same RGB
                    // texel after the source array's index-space shift below.
                    const float camera_x =
                        (static_cast<float>(px) + 0.5f - target_cx) * z / target_fx;
                    const float camera_y =
                        (static_cast<float>(py) + 0.5f - target_cy) * z / target_fy;

                    const float world_x = target_c2w[0] * camera_x +
                                          target_c2w[1] * camera_y +
                                          target_c2w[2] * z + target_c2w[3];
                    const float world_y = target_c2w[4] * camera_x +
                                          target_c2w[5] * camera_y +
                                          target_c2w[6] * z + target_c2w[7];
                    const float world_z = target_c2w[8] * camera_x +
                                          target_c2w[9] * camera_y +
                                          target_c2w[10] * z + target_c2w[11];

                    const float source_x = source_w2c[0] * world_x +
                                           source_w2c[1] * world_y +
                                           source_w2c[2] * world_z + source_w2c[3];
                    const float source_y = source_w2c[4] * world_x +
                                           source_w2c[5] * world_y +
                                           source_w2c[6] * world_z + source_w2c[7];
                    const float source_z = source_w2c[8] * world_x +
                                           source_w2c[9] * world_y +
                                           source_w2c[10] * world_z + source_w2c[11];

                    projected = isfinite(source_x) && isfinite(source_y) &&
                                finite_positive(source_z);
                    if (projected) {
                        const float source_u = source_fx * (source_x / source_z) + source_cx;
                        const float source_v = source_fy * (source_y / source_z) + source_cy;
                        // Convert raster coordinates (whose centers are n+0.5)
                        // to array index coordinates (whose first sample is n).
                        const float source_index_x = source_u - 0.5f;
                        const float source_index_y = source_v - 0.5f;
                        projected = inside_index_space(
                            source_index_x,
                            source_index_y,
                            source_width,
                            source_height);

                        if (projected) {
                            source_luma = luminance_bilinear(
                                source_rgb,
                                source_width * source_height,
                                source_width,
                                source_height,
                                source_index_x,
                                source_index_y);
                            photo_valid = isfinite(source_luma);

                        }
                    }
                }

                target_supported_flags[thread_index] = target_supported ? 1 : 0;
                projected_flags[thread_index] = projected ? 1 : 0;
                // Source mesh depth is deliberately not evaluated. Keep the
                // legacy diagnostic slot zeroed for API/report compatibility.
                depth_consistent_flags[thread_index] = 0;
                if (photo_valid) {
                    target_values[thread_index] = luminance_at_pixel(
                        target_rgb,
                        target_plane_size,
                        target_index);
                    source_values[thread_index] = source_luma;
                    photo_valid_flags[thread_index] =
                        isfinite(target_values[thread_index]) ? 1 : 0;
                }
            }
            __syncthreads();

            if (thread_index == 0) {
                float sum_target = 0.0f;
                float sum_source = 0.0f;
                float sum_target_sq = 0.0f;
                float sum_source_sq = 0.0f;
                float sum_cross = 0.0f;
                int32_t target_count = 0;
                int32_t projected_count = 0;
                int32_t photo_count = 0;
                int32_t depth_consistent_count = 0;
                for (int i = 0; i < sample_count; ++i) {
                    target_count += target_supported_flags[i];
                    projected_count += projected_flags[i];
                    depth_consistent_count += depth_consistent_flags[i];
                    if (photo_valid_flags[i] != 0) {
                        const float target_value = target_values[i];
                        const float source_value = source_values[i];
                        sum_target += target_value;
                        sum_source += source_value;
                        sum_target_sq += target_value * target_value;
                        sum_source_sq += source_value * source_value;
                        sum_cross += target_value * source_value;
                        ++photo_count;
                    }
                }

                region_photometric_samples[region_index] = photo_count;
                if (region_target_samples) {
                    region_target_samples[region_index] = target_count;
                }
                if (region_projected_samples) {
                    region_projected_samples[region_index] = projected_count;
                }
                if (region_depth_consistent_samples) {
                    region_depth_consistent_samples[region_index] = depth_consistent_count;
                }
                float score = CUDART_NAN_F;
                if (photo_count >= max(1, min_valid_samples)) {
                    const float inv_count = 1.0f / static_cast<float>(photo_count);
                    const float target_variance =
                        fmaxf(0.0f, sum_target_sq - sum_target * sum_target * inv_count);
                    const float source_variance =
                        fmaxf(0.0f, sum_source_sq - sum_source * sum_source * inv_count);
                    const float variance_floor = fmaxf(0.0f, variance_epsilon);
                    if (target_variance > variance_floor && source_variance > variance_floor) {
                        const float covariance =
                            sum_cross - sum_target * sum_source * inv_count;
                        const float denominator = sqrtf(target_variance * source_variance);
                        if (denominator > 1.0e-12f && isfinite(denominator)) {
                            score = covariance / denominator;
                            score = fminf(1.0f, fmaxf(-1.0f, score));
                            if (!isfinite(score)) {
                                score = CUDART_NAN_F;
                            }
                        }
                    }
                }
                region_zncc[region_index] = score;
            }
        }

    } // namespace

    void launch_scan_pose_zncc_regions(
        const float* target_rgb_chw,
        const float* target_depth_hw,
        const float* target_c2w_4x4,
        const int target_width,
        const int target_height,
        const float target_fx,
        const float target_fy,
        const float target_cx,
        const float target_cy,
        const float* source_rgb_chw,
        const float* source_depth_hw,
        const float* source_w2c_4x4,
        const int source_width,
        const int source_height,
        const float source_fx,
        const float source_fy,
        const float source_cx,
        const float source_cy,
        const int region_count_x,
        const int region_count_y,
        const int samples_per_axis,
        const int min_valid_samples,
        const float relative_depth_epsilon,
        const float variance_epsilon,
        float* region_zncc_out,
        int32_t* region_photometric_samples_out,
        cudaStream_t stream,
        int32_t* region_target_samples_out,
        int32_t* region_projected_samples_out,
        int32_t* region_depth_consistent_samples_out) {

        // Kept in the public wrapper for compatibility with existing callers;
        // source depth and its tolerance are no longer used by the filter.
        (void)source_depth_hw;
        (void)relative_depth_epsilon;

        // The kernel computes a fixed-size patch window; reject undersized
        // target images before forming `target_width - PATCH_SIZE` below.
        if (!target_rgb_chw || !target_depth_hw || !target_c2w_4x4 ||
            !source_rgb_chw || !source_depth_hw || !source_w2c_4x4 ||
            !region_zncc_out || !region_photometric_samples_out ||
            target_width < PATCH_SIZE || target_height < PATCH_SIZE ||
            source_width <= 0 || source_height <= 0 ||
            region_count_x <= 0 || region_count_y <= 0 ||
            samples_per_axis <= 0 || samples_per_axis > PATCH_SIZE) {
            return;
        }

        const int region_count = region_count_x * region_count_y;
        scan_pose_zncc_regions_kernel<<<region_count, MAX_SAMPLES_PER_REGION, 0, stream>>>(
            target_rgb_chw,
            target_depth_hw,
            target_c2w_4x4,
            target_width,
            target_height,
            target_fx,
            target_fy,
            target_cx,
            target_cy,
            source_rgb_chw,
            source_w2c_4x4,
            source_width,
            source_height,
            source_fx,
            source_fy,
            source_cx,
            source_cy,
            region_count_x,
            region_count_y,
            samples_per_axis,
            min_valid_samples,
            variance_epsilon,
            region_zncc_out,
            region_photometric_samples_out,
            region_target_samples_out,
            region_projected_samples_out,
            region_depth_consistent_samples_out);
    }

} // namespace lfs::training::kernels
