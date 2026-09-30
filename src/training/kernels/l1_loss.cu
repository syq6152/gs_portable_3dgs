/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/l1_loss.cuh"

namespace lfs::training::kernels {

    // Fused kernel: computes gradient and accumulates sum(abs(diff)) in single pass
    // OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB BlockReduce!)
    __global__ void fused_l1_kernel(
        const float* __restrict__ img1,
        const float* __restrict__ img2,
        float* __restrict__ grad_out,
        float* __restrict__ partial_sums,
        size_t N,
        float grad_scale) {

        // Thread-local sum
        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < N;
             idx += blockDim.x * gridDim.x) {

            float diff = img1[idx] - img2[idx];
            float abs_diff = fabsf(diff);

            // Accumulate for loss
            local_sum += abs_diff;

            // Store gradient: sign(diff) * grad_scale
            // NOTE: sign(0) = 0 to match PyTorch behavior
            // copysignf returns ±grad_scale even for 0.0, which is wrong
            float grad = (diff > 0.0f) ? grad_scale : ((diff < 0.0f) ? -grad_scale : 0.0f);
            grad_out[idx] = grad;
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        // First thread writes block result
        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    // Final reduction kernel (handles any number of blocks)
    // OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB BlockReduce!)
    __global__ void final_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float norm_factor) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = sum * norm_factor;
        }
    }

    void launch_fused_l1_loss(
        const float* img1,
        const float* img2,
        float* grad_out,
        float* loss_out,
        float* temp_buffer,
        size_t N,
        cudaStream_t stream) {

        const int block_size = 256;
        const int num_blocks = std::min((N + block_size - 1) / block_size, size_t(1024));

        float grad_scale = 1.0f / static_cast<float>(N);

        // Launch fused kernel
        fused_l1_kernel<<<num_blocks, block_size, 0, stream>>>(
            img1, img2, grad_out, temp_buffer, N, grad_scale);

        // Launch final reduction (normalize by N for mean)
        float norm_factor = 1.0f / static_cast<float>(N);
        final_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, norm_factor);
    }

} // namespace lfs::training::kernels
