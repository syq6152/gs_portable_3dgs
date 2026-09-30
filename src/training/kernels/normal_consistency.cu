/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/normal_consistency.cuh"
#include <cfloat>
#include <cmath>

namespace lfs::training::kernels {
    namespace {

        constexpr int BLOCK_SIZE = 256;
        constexpr float NORMAL_EPS = 1e-6f;

        /**
         * Pass 1: Per-pixel loss and unscaled gradient.
         * Also computes per-block partial sum and valid count via warp reduction.
         */
        __global__ void normal_consistency_kernel(
            const float* __restrict__ render_normal,  // [3, H, W]
            const float* __restrict__ depth_normal,   // [3, H, W]
            const float* __restrict__ depth,          // [H, W]
            float* __restrict__ grad_render_normal,   // [3, H, W]
            float* __restrict__ grad_depth_normal,    // [3, H, W]
            float* __restrict__ block_sums,
            int* __restrict__ block_counts,
            const int HW) {

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            float local_error = 0.0f;
            int local_valid = 0;

            if (idx < HW) {
                const float d = depth[idx];
                const float rn_x = render_normal[idx];
                const float rn_y = render_normal[idx + HW];
                const float rn_z = render_normal[idx + 2 * HW];
                const float dn_x = depth_normal[idx];
                const float dn_y = depth_normal[idx + HW];
                const float dn_z = depth_normal[idx + 2 * HW];

                const float rn_len = sqrtf(rn_x * rn_x + rn_y * rn_y + rn_z * rn_z);
                const float dn_len = sqrtf(dn_x * dn_x + dn_y * dn_y + dn_z * dn_z);

                const bool valid = (d > 0.0f) && (rn_len > NORMAL_EPS) && (dn_len > NORMAL_EPS);

                if (valid) {
                    const float dot_val = rn_x * dn_x + rn_y * dn_y + rn_z * dn_z;
                    local_error = 1.0f - dot_val;
                    local_valid = 1;

                    // Unscaled gradient: -depth_normal (will be divided by N_valid later)
                    grad_render_normal[idx] = -dn_x;
                    grad_render_normal[idx + HW] = -dn_y;
                    grad_render_normal[idx + 2 * HW] = -dn_z;

                    // Symmetric: dL/d(depth_normal) = -render_normal / N_valid
                    grad_depth_normal[idx] = -rn_x;
                    grad_depth_normal[idx + HW] = -rn_y;
                    grad_depth_normal[idx + 2 * HW] = -rn_z;
                } else {
                    grad_render_normal[idx] = 0.0f;
                    grad_render_normal[idx + HW] = 0.0f;
                    grad_render_normal[idx + 2 * HW] = 0.0f;

                    grad_depth_normal[idx] = 0.0f;
                    grad_depth_normal[idx + HW] = 0.0f;
                    grad_depth_normal[idx + 2 * HW] = 0.0f;
                }
            }

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_error += __shfl_down_sync(0xffffffffu, local_error, offset);
                local_valid += __shfl_down_sync(0xffffffffu, local_valid, offset);
            }

            // Lane 0 of each warp writes to shared memory
            __shared__ float s_sum[8];  // max 256/32 = 8 warps
            __shared__ int s_count[8];
            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            if (lane_id == 0) {
                s_sum[warp_id] = local_error;
                s_count[warp_id] = local_valid;
            }
            __syncthreads();

            // First warp reduces across warps
            if (warp_id == 0) {
                const int n_warps = (blockDim.x + 31) / 32;
                float val = (lane_id < n_warps) ? s_sum[lane_id] : 0.0f;
                int cnt = (lane_id < n_warps) ? s_count[lane_id] : 0;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffffu, val, offset);
                    cnt += __shfl_down_sync(0xffffffffu, cnt, offset);
                }
                if (lane_id == 0) {
                    block_sums[blockIdx.x] = val;
                    block_counts[blockIdx.x] = cnt;
                }
            }
        }

        /**
         * Pass 2: Reduce block-level sums to scalar loss.
         * Also scales the gradient by 1/N_valid.
         */
        __global__ void normal_consistency_reduce_and_scale_kernel(
            const float* __restrict__ block_sums,
            const int* __restrict__ block_counts,
            float* __restrict__ loss_out,
            float* __restrict__ grad_render_normal,  // [3*HW]
            float* __restrict__ grad_depth_normal,    // [3*HW]
            const int num_blocks,
            const int HW) {

            // Single-block reduction
            float total_sum = 0.0f;
            int total_count = 0;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
                total_sum += block_sums[i];
                total_count += block_counts[i];
            }

            // Warp reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                total_sum += __shfl_down_sync(0xffffffffu, total_sum, offset);
                total_count += __shfl_down_sync(0xffffffffu, total_count, offset);
            }

            __shared__ float s_sum[8];
            __shared__ int s_count[8];
            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            if (lane_id == 0) {
                s_sum[warp_id] = total_sum;
                s_count[warp_id] = total_count;
            }
            __syncthreads();

            if (warp_id == 0) {
                const int n_warps = (blockDim.x + 31) / 32;
                float val = (lane_id < n_warps) ? s_sum[lane_id] : 0.0f;
                int cnt = (lane_id < n_warps) ? s_count[lane_id] : 0;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffffu, val, offset);
                    cnt += __shfl_down_sync(0xffffffffu, cnt, offset);
                }
                if (lane_id == 0) {
                    const float n_valid = fmaxf(static_cast<float>(cnt), 1.0f);
                    loss_out[0] = val / n_valid;
                    // Store 1/N_valid for gradient scaling (using loss_out[1] as scratch)
                    // We'll use a separate scale kernel for the gradient
                }
            }

            // Scale all gradient values by 1/N_valid
            // Recompute total_count for all threads
            __syncthreads();
            int final_count = 0;
            for (int i = 0; i < num_blocks; i++) {
                final_count += block_counts[i];
            }
            const float inv_n = (final_count > 0) ? (1.0f / static_cast<float>(final_count)) : 0.0f;

            for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < 3 * HW; i += blockDim.x * gridDim.x) {
                grad_render_normal[i] *= inv_n;
                grad_depth_normal[i] *= inv_n;
            }
        }

    } // anonymous namespace

    void launch_normal_consistency_loss(
        const float* render_normal,
        const float* depth_normal,
        const float* depth,
        float* grad_render_normal,
        float* grad_depth_normal,
        float* loss_out,
        float* temp_sum_buffer,
        int* temp_count_buffer,
        int H,
        int W,
        cudaStream_t stream) {

        const int HW = H * W;
        const int num_blocks = (HW + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Pass 1: per-pixel computation + block-level reduction
        normal_consistency_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            render_normal, depth_normal, depth,
            grad_render_normal,
            grad_depth_normal,
            temp_sum_buffer, temp_count_buffer,
            HW);

        // Pass 2: cross-block reduction + gradient scaling
        // Launch single block with enough threads to reduce all blocks
        const int reduce_threads = min(num_blocks, BLOCK_SIZE);
        normal_consistency_reduce_and_scale_kernel<<<1, reduce_threads, 0, stream>>>(
            temp_sum_buffer, temp_count_buffer,
            loss_out,
            grad_render_normal,
            grad_depth_normal,
            num_blocks,
            HW);
    }

} // namespace lfs::training::kernels
