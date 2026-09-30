/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/per_frame_observation_blur.cuh"

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int PARAMS_PER_FRAME = 8;

        constexpr int div_up(const int value, const int divisor) {
            return value / divisor + static_cast<int>(value % divisor != 0);
        }

        __global__ void regularization_forward_kernel(
            const float* __restrict__ parameters,
            float* __restrict__ loss,
            const int num_frames,
            const bool motion_active,
            const bool defocus_active,
            const float rotation_weight,
            const float translation_weight,
            const float defocus_weight) {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int count = num_frames * PARAMS_PER_FRAME;
            if (index >= count) {
                return;
            }

            const int parameter = index % PARAMS_PER_FRAME;
            float weight = 0.0f;
            if (motion_active && parameter < 3) {
                weight = rotation_weight;
            } else if (motion_active && parameter < 6) {
                weight = translation_weight;
            } else if (defocus_active && parameter == 6) {
                weight = defocus_weight;
            }
            if (weight > 0.0f) {
                atomicAdd(loss, weight * parameters[index] / static_cast<float>(num_frames));
            }
        }

        __global__ void regularization_backward_kernel(
            float* __restrict__ gradients,
            const int num_frames,
            const bool motion_active,
            const bool defocus_active,
            const float rotation_weight,
            const float translation_weight,
            const float defocus_weight) {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int count = num_frames * PARAMS_PER_FRAME;
            if (index >= count) {
                return;
            }

            const int parameter = index % PARAMS_PER_FRAME;
            float weight = 0.0f;
            if (motion_active && parameter < 3) {
                weight = rotation_weight;
            } else if (motion_active && parameter < 6) {
                weight = translation_weight;
            } else if (defocus_active && parameter == 6) {
                weight = defocus_weight;
            }
            gradients[index] += weight / static_cast<float>(num_frames);
        }

        __global__ void adam_update_kernel(
            float* __restrict__ parameters,
            float* __restrict__ exp_avg,
            float* __restrict__ exp_avg_sq,
            const float* __restrict__ gradients,
            const int count,
            const bool motion_active,
            const bool defocus_active,
            const float rotation_lr,
            const float translation_lr,
            const float defocus_scale_lr,
            const float defocus_focus_lr,
            const float beta1,
            const float beta2,
            const float motion_bias_corr1_rcp,
            const float motion_bias_corr2_sqrt_rcp,
            const float defocus_bias_corr1_rcp,
            const float defocus_bias_corr2_sqrt_rcp,
            const float eps,
            const float max_rotation_variance,
            const float max_translation_variance,
            const float focus_inverse_depth_abs_max) {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= count) {
                return;
            }

            const int parameter = index % PARAMS_PER_FRAME;
            const bool is_motion = parameter < 6;
            if ((is_motion && !motion_active) || (!is_motion && !defocus_active)) {
                return;
            }

            const float lr = parameter < 3 ? rotation_lr
                            : parameter < 6 ? translation_lr
                            : parameter == 6 ? defocus_scale_lr
                                             : defocus_focus_lr;
            if (lr <= 0.0f) {
                return;
            }

            const float gradient = gradients[index];
            const float first_moment =
                beta1 * exp_avg[index] + (1.0f - beta1) * gradient;
            const float second_moment =
                beta2 * exp_avg_sq[index] + (1.0f - beta2) * gradient * gradient;
            exp_avg[index] = first_moment;
            exp_avg_sq[index] = second_moment;

            const float bias_corr1_rcp = is_motion
                                             ? motion_bias_corr1_rcp
                                             : defocus_bias_corr1_rcp;
            const float bias_corr2_sqrt_rcp = is_motion
                                                  ? motion_bias_corr2_sqrt_rcp
                                                  : defocus_bias_corr2_sqrt_rcp;
            const float numerator = first_moment * bias_corr1_rcp;
            const float denominator = sqrtf(second_moment) * bias_corr2_sqrt_rcp + eps;
            float value = parameters[index] - lr * numerator / denominator;

            if (parameter < 3) {
                value = fminf(fmaxf(value, 0.0f), max_rotation_variance);
            } else if (parameter < 6) {
                value = fminf(fmaxf(value, 0.0f), max_translation_variance);
            } else if (parameter == 6) {
                // beta = A^2 is not a screen-space radius. The configured
                // radius cap is applied to beta * (rho - 1 / depth)^2 in the
                // rasterizer, where its units and gradient saturation are
                // well-defined.
                value = fmaxf(value, 0.0f);
            } else {
                value = fminf(fmaxf(value, 0.0f), focus_inverse_depth_abs_max);
            }
            parameters[index] = value;
        }

        void assert_launch_success() {
            const cudaError_t error = cudaGetLastError();
            assert(error == cudaSuccess && "Per-frame observation blur CUDA kernel launch failed");
        }
    } // namespace

    void launch_per_frame_observation_blur_regularization_forward(
        const float* const parameters,
        float* const loss,
        const int num_frames,
        const bool motion_active,
        const bool defocus_active,
        const float rotation_weight,
        const float translation_weight,
        const float defocus_weight,
        const cudaStream_t stream) {
        const int count = num_frames * PARAMS_PER_FRAME;
        regularization_forward_kernel<<<div_up(count, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            parameters, loss, num_frames, motion_active, defocus_active,
            rotation_weight, translation_weight, defocus_weight);
        assert_launch_success();
    }

    void launch_per_frame_observation_blur_regularization_backward(
        float* const gradients,
        const int num_frames,
        const bool motion_active,
        const bool defocus_active,
        const float rotation_weight,
        const float translation_weight,
        const float defocus_weight,
        const cudaStream_t stream) {
        const int count = num_frames * PARAMS_PER_FRAME;
        regularization_backward_kernel<<<div_up(count, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            gradients, num_frames, motion_active, defocus_active,
            rotation_weight, translation_weight, defocus_weight);
        assert_launch_success();
    }

    void launch_per_frame_observation_blur_adam_update(
        float* const parameters,
        float* const exp_avg,
        float* const exp_avg_sq,
        const float* const gradients,
        const int num_frames,
        const bool motion_active,
        const bool defocus_active,
        const float rotation_lr,
        const float translation_lr,
        const float defocus_scale_lr,
        const float defocus_focus_lr,
        const float beta1,
        const float beta2,
        const float motion_bias_corr1_rcp,
        const float motion_bias_corr2_sqrt_rcp,
        const float defocus_bias_corr1_rcp,
        const float defocus_bias_corr2_sqrt_rcp,
        const float eps,
        const float max_rotation_variance,
        const float max_translation_variance,
        const float focus_inverse_depth_abs_max,
        const cudaStream_t stream) {
        const int count = num_frames * PARAMS_PER_FRAME;
        adam_update_kernel<<<div_up(count, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            parameters, exp_avg, exp_avg_sq, gradients, count,
            motion_active, defocus_active,
            rotation_lr, translation_lr, defocus_scale_lr, defocus_focus_lr,
            beta1, beta2,
            motion_bias_corr1_rcp, motion_bias_corr2_sqrt_rcp,
            defocus_bias_corr1_rcp, defocus_bias_corr2_sqrt_rcp,
            eps, max_rotation_variance, max_translation_variance,
            focus_inverse_depth_abs_max);
        assert_launch_success();
    }

} // namespace lfs::training::kernels
