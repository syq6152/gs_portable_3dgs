/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pruning_kernels.hpp"

#include "core/cuda_debug.hpp"

namespace lfs::training::kernels {

    namespace {
        __global__ void near_zero_quaternion_mask_kernel(
            const float* __restrict__ rotations,
            uint8_t* __restrict__ mask,
            const size_t count) {
            const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (index >= count) {
                return;
            }

            const size_t offset = index * 4;
            const float q0 = rotations[offset];
            const float q1 = rotations[offset + 1];
            const float q2 = rotations[offset + 2];
            const float q3 = rotations[offset + 3];
            const float norm_squared = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
            mask[index] = static_cast<uint8_t>(norm_squared < 1e-8f);
        }
    } // namespace

    void launch_near_zero_quaternion_mask(
        const float* rotations,
        uint8_t* mask,
        const size_t count,
        cudaStream_t stream) {
        if (count == 0) {
            return;
        }

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        near_zero_quaternion_mask_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            rotations, mask, count);
        CUDA_KERNEL_CHECK("near_zero_quaternion_mask_kernel");
    }

} // namespace lfs::training::kernels
