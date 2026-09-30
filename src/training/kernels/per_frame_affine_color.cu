/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/per_frame_affine_color.cuh"

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr int PARAMS_PER_FRAME = 12;

        constexpr int div_up(const int value, const int divisor) {
            return value / divisor + static_cast<int>(value % divisor != 0);
        }

        __device__ __forceinline__ float affine_weight(
            const float* const parameters,
            const int output_channel,
            const int input_channel) {
            const float delta = parameters[output_channel * 3 + input_channel];
            return delta + static_cast<float>(output_channel == input_channel);
        }

        __global__ void per_frame_affine_color_forward_kernel(
            const float* __restrict__ parameters,
            const float* __restrict__ rgb,
            float* __restrict__ output,
            const int num_pixels) {

            const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
            if (pixel >= num_pixels) {
                return;
            }

            const float input[3] = {
                rgb[pixel],
                rgb[num_pixels + pixel],
                rgb[2 * num_pixels + pixel]};

#pragma unroll
            for (int out_channel = 0; out_channel < 3; ++out_channel) {
                float value = parameters[9 + out_channel];
#pragma unroll
                for (int in_channel = 0; in_channel < 3; ++in_channel) {
                    value += affine_weight(parameters, out_channel, in_channel) * input[in_channel];
                }
                output[out_channel * num_pixels + pixel] = value;
            }
        }

        __global__ void per_frame_affine_color_backward_kernel(
            const float* __restrict__ parameters,
            const float* __restrict__ rgb,
            const float* __restrict__ grad_output,
            float* __restrict__ grad_parameters,
            float* __restrict__ grad_rgb,
            const int num_pixels,
            const bool accumulate_parameters) {

            const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
            float input[3] = {0.0f, 0.0f, 0.0f};
            float grad[3] = {0.0f, 0.0f, 0.0f};

            if (pixel < num_pixels) {
#pragma unroll
                for (int channel = 0; channel < 3; ++channel) {
                    input[channel] = rgb[channel * num_pixels + pixel];
                    grad[channel] = grad_output[channel * num_pixels + pixel];
                }

#pragma unroll
                for (int in_channel = 0; in_channel < 3; ++in_channel) {
                    float value = 0.0f;
#pragma unroll
                    for (int out_channel = 0; out_channel < 3; ++out_channel) {
                        value += affine_weight(parameters, out_channel, in_channel) * grad[out_channel];
                    }
                    grad_rgb[in_channel * num_pixels + pixel] = value;
                }
            }

            if (!accumulate_parameters) {
                return;
            }

            extern __shared__ float reduction[];
            const int thread = threadIdx.x;

#pragma unroll
            for (int out_channel = 0; out_channel < 3; ++out_channel) {
#pragma unroll
                for (int in_channel = 0; in_channel < 3; ++in_channel) {
                    const int parameter = out_channel * 3 + in_channel;
                    reduction[parameter * blockDim.x + thread] =
                        grad[out_channel] * input[in_channel];
                }
                reduction[(9 + out_channel) * blockDim.x + thread] = grad[out_channel];
            }
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (thread < stride) {
#pragma unroll
                    for (int parameter = 0; parameter < PARAMS_PER_FRAME; ++parameter) {
                        reduction[parameter * blockDim.x + thread] +=
                            reduction[parameter * blockDim.x + thread + stride];
                    }
                }
                __syncthreads();
            }

            if (thread == 0) {
#pragma unroll
                for (int parameter = 0; parameter < PARAMS_PER_FRAME; ++parameter) {
                    atomicAdd(&grad_parameters[parameter], reduction[parameter * blockDim.x]);
                }
            }
        }

        __global__ void per_frame_affine_color_regularization_stats_kernel(
            const float* __restrict__ parameters,
            float* __restrict__ parameter_means,
            float* __restrict__ loss,
            const int num_frames,
            const float identity_weight,
            const float gauge_weight,
            const float bias_weight) {

            const int parameter = threadIdx.x;
            if (parameter >= PARAMS_PER_FRAME) {
                return;
            }

            float sum = 0.0f;
            float squared_sum = 0.0f;
            for (int frame = 0; frame < num_frames; ++frame) {
                const float value = parameters[frame * PARAMS_PER_FRAME + parameter];
                sum += value;
                squared_sum += value * value;
            }

            const float inv_frames = 1.0f / static_cast<float>(num_frames);
            const float mean = sum * inv_frames;
            parameter_means[parameter] = mean;

            if (loss != nullptr) {
                const float entry_weight = parameter < 9 ? 1.0f : bias_weight;
                const float entry_loss = entry_weight *
                                         (identity_weight * squared_sum * inv_frames +
                                          gauge_weight * mean * mean);
                atomicAdd(loss, entry_loss);
            }
        }

        __global__ void per_frame_affine_color_regularization_backward_kernel(
            const float* __restrict__ parameters,
            const float* __restrict__ parameter_means,
            float* __restrict__ grad_parameters,
            const int num_frames,
            const float identity_weight,
            const float gauge_weight,
            const float bias_weight) {

            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int num_parameters = num_frames * PARAMS_PER_FRAME;
            if (index >= num_parameters) {
                return;
            }

            const int parameter = index % PARAMS_PER_FRAME;
            const float entry_weight = parameter < 9 ? 1.0f : bias_weight;
            const float inv_frames = 1.0f / static_cast<float>(num_frames);
            const float grad_identity =
                2.0f * identity_weight * entry_weight * parameters[index] * inv_frames;
            const float grad_gauge =
                2.0f * gauge_weight * entry_weight * parameter_means[parameter] * inv_frames;
            grad_parameters[index] += grad_identity + grad_gauge;
        }

        __global__ void per_frame_affine_color_adam_update_kernel(
            float* __restrict__ parameters,
            float* __restrict__ exp_avg,
            float* __restrict__ exp_avg_sq,
            const float* __restrict__ gradients,
            const int num_elements,
            const float lr,
            const float beta1,
            const float beta2,
            const float bias_corr1_rcp,
            const float bias_corr2_sqrt_rcp,
            const float eps) {

            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= num_elements) {
                return;
            }

            const float gradient = gradients[index];
            const float first_moment =
                beta1 * exp_avg[index] + (1.0f - beta1) * gradient;
            const float second_moment =
                beta2 * exp_avg_sq[index] + (1.0f - beta2) * gradient * gradient;

            exp_avg[index] = first_moment;
            exp_avg_sq[index] = second_moment;

            const float corrected_first_moment = first_moment * bias_corr1_rcp;
            const float corrected_second_moment_sqrt = sqrtf(second_moment) * bias_corr2_sqrt_rcp;
            parameters[index] -=
                lr * corrected_first_moment / (corrected_second_moment_sqrt + eps);
        }

        void assert_launch_success() {
            const cudaError_t error = cudaGetLastError();
            assert(error == cudaSuccess && "Per-frame affine color CUDA kernel launch failed");
        }
    } // namespace

    void launch_per_frame_affine_color_forward(
        const float* const parameters,
        const float* const rgb,
        float* const output,
        const int height,
        const int width,
        const cudaStream_t stream) {
        const int num_pixels = height * width;
        per_frame_affine_color_forward_kernel<<<div_up(num_pixels, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            parameters, rgb, output, num_pixels);
        assert_launch_success();
    }

    void launch_per_frame_affine_color_backward(
        const float* const parameters,
        const float* const rgb,
        const float* const grad_output,
        float* const grad_parameters,
        float* const grad_rgb,
        const int height,
        const int width,
        const bool accumulate_parameters,
        const cudaStream_t stream) {
        const int num_pixels = height * width;
        constexpr size_t SHARED_BYTES =
            static_cast<size_t>(PARAMS_PER_FRAME) * BLOCK_SIZE * sizeof(float);
        per_frame_affine_color_backward_kernel<<<
            div_up(num_pixels, BLOCK_SIZE), BLOCK_SIZE, SHARED_BYTES, stream>>>(
            parameters, rgb, grad_output, grad_parameters, grad_rgb,
            num_pixels, accumulate_parameters);
        assert_launch_success();
    }

    void launch_per_frame_affine_color_regularization_forward(
        const float* const parameters,
        float* const parameter_means,
        float* const loss,
        const int num_frames,
        const float identity_weight,
        const float gauge_weight,
        const float bias_weight,
        const cudaStream_t stream) {
        per_frame_affine_color_regularization_stats_kernel<<<1, 32, 0, stream>>>(
            parameters, parameter_means, loss, num_frames,
            identity_weight, gauge_weight, bias_weight);
        assert_launch_success();
    }

    void launch_per_frame_affine_color_parameter_means(
        const float* const parameters,
        float* const parameter_means,
        const int num_frames,
        const cudaStream_t stream) {
        per_frame_affine_color_regularization_stats_kernel<<<1, 32, 0, stream>>>(
            parameters, parameter_means, nullptr, num_frames,
            0.0f, 0.0f, 1.0f);
        assert_launch_success();
    }

    void launch_per_frame_affine_color_regularization_backward(
        const float* const parameters,
        const float* const parameter_means,
        float* const grad_parameters,
        const int num_frames,
        const float identity_weight,
        const float gauge_weight,
        const float bias_weight,
        const cudaStream_t stream) {
        const int num_parameters = num_frames * PARAMS_PER_FRAME;
        per_frame_affine_color_regularization_backward_kernel<<<
            div_up(num_parameters, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            parameters, parameter_means, grad_parameters, num_frames,
            identity_weight, gauge_weight, bias_weight);
        assert_launch_success();
    }

    void launch_per_frame_affine_color_adam_update(
        float* const parameters,
        float* const exp_avg,
        float* const exp_avg_sq,
        const float* const gradients,
        const int num_elements,
        const float lr,
        const float beta1,
        const float beta2,
        const float bias_corr1_rcp,
        const float bias_corr2_sqrt_rcp,
        const float eps,
        const cudaStream_t stream) {
        per_frame_affine_color_adam_update_kernel<<<
            div_up(num_elements, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
            parameters, exp_avg, exp_avg_sq, gradients, num_elements,
            lr, beta1, beta2, bias_corr1_rcp, bias_corr2_sqrt_rcp, eps);
        assert_launch_success();
    }

} // namespace lfs::training::kernels
