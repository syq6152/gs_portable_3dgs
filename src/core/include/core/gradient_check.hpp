/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>

namespace lfs::core::debug {

    // Default tolerances for gradient checking
    constexpr float DEFAULT_EPS = 1e-4f;
    constexpr float DEFAULT_RTOL = 1e-3f;
    constexpr float DEFAULT_ATOL = 1e-5f;
    constexpr float REL_ERR_EPSILON = 1e-8f;
    constexpr size_t DEFAULT_SAMPLES = 100;

    struct GradientCheckResult {
        bool passed = false;
        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f;
        size_t num_checked = 0;
        size_t num_failed = 0;

        [[nodiscard]] std::string to_string() const {
            return std::format("{}: max_abs={:.2e}, max_rel={:.2e}, failed={}/{}",
                               passed ? "PASSED" : "FAILED",
                               max_abs_error, max_rel_error, num_failed, num_checked);
        }
    };

    // Numerical gradient checker for verifying backward pass implementations
    // Note: This tensor library doesn't support autograd, so analytical gradients must be provided
    class GradientChecker {
    public:
        // Forward function type: takes input tensor, returns scalar output
        using ForwardFn = std::function<Tensor(const Tensor&)>;

        // Check gradients numerically against provided analytical gradient
        // forward: function that takes input and returns scalar output
        // input: input tensor (will be cloned, original not modified)
        // analytical_grad: gradient computed by your backward pass (required)
        // eps: perturbation size for numerical gradient
        // rtol: relative tolerance
        // atol: absolute tolerance
        static GradientCheckResult check(
            ForwardFn forward,
            const Tensor& input,
            const Tensor& analytical_grad,
            const float eps = DEFAULT_EPS,
            const float rtol = DEFAULT_RTOL,
            const float atol = DEFAULT_ATOL) {
            GradientCheckResult result;

            if (input.dtype() != DataType::Float32) {
                LOG_ERROR("GradientChecker only supports Float32 tensors");
                return result;
            }

            if (input.shape() != analytical_grad.shape()) {
                LOG_ERROR("GradientChecker: input and analytical_grad shapes don't match");
                return result;
            }

            // Compute numerical gradient
            Tensor input_cpu = input.device() == Device::CUDA ? input.cpu() : input.clone();
            Tensor num_grad = Tensor::zeros_like(input_cpu);

            const size_t n = input.numel();
            float* input_data = input_cpu.ptr<float>();
            float* num_grad_data = num_grad.ptr<float>();

            result.num_checked = n;

            for (size_t i = 0; i < n; ++i) {
                const float orig = input_data[i];

                // f(x + eps)
                input_data[i] = orig + eps;
                Tensor input_plus = input.device() == Device::CUDA ? input_cpu.cuda() : input_cpu;
                Tensor out_plus = forward(input_plus);
                if (out_plus.numel() > 1)
                    out_plus = out_plus.sum();
                const float f_plus = out_plus.cpu().ptr<float>()[0];

                // f(x - eps)
                input_data[i] = orig - eps;
                Tensor input_minus = input.device() == Device::CUDA ? input_cpu.cuda() : input_cpu;
                Tensor out_minus = forward(input_minus);
                if (out_minus.numel() > 1)
                    out_minus = out_minus.sum();
                const float f_minus = out_minus.cpu().ptr<float>()[0];

                // Restore
                input_data[i] = orig;

                // Central difference
                num_grad_data[i] = (f_plus - f_minus) / (2.0f * eps);
            }

            // Compare analytical vs numerical
            const Tensor anal_cpu = analytical_grad.device() == Device::CUDA ? analytical_grad.cpu() : analytical_grad;
            const float* anal_data = anal_cpu.ptr<float>();

            for (size_t i = 0; i < n; ++i) {
                const float anal = anal_data[i];
                const float num = num_grad_data[i];
                const float abs_err = std::abs(anal - num);
                const float rel_err = abs_err / (std::max(std::abs(anal), std::abs(num)) + REL_ERR_EPSILON);

                result.max_abs_error = std::max(result.max_abs_error, abs_err);
                result.max_rel_error = std::max(result.max_rel_error, rel_err);

                // Check if within tolerance
                if (abs_err > atol + rtol * std::abs(num)) {
                    ++result.num_failed;
                }
            }

            result.passed = result.num_failed == 0;
            return result;
        }

        // Check gradient for a subset of elements (faster for large tensors)
        static GradientCheckResult check_sampled(
            ForwardFn forward,
            const Tensor& input,
            const Tensor& analytical_grad,
            const size_t num_samples = DEFAULT_SAMPLES,
            const float eps = DEFAULT_EPS,
            const float rtol = DEFAULT_RTOL,
            const float atol = DEFAULT_ATOL) {
            if (input.numel() <= num_samples) {
                return check(forward, input, analytical_grad, eps, rtol, atol);
            }

            GradientCheckResult result;

            if (input.dtype() != DataType::Float32) {
                LOG_ERROR("GradientChecker only supports Float32 tensors");
                return result;
            }

            // Random sampling indices
            std::vector<size_t> indices(input.numel());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
            indices.resize(num_samples);

            // Get analytical gradient on CPU
            const Tensor anal_cpu = analytical_grad.device() == Device::CUDA ? analytical_grad.cpu() : analytical_grad;
            const float* anal_data = anal_cpu.ptr<float>();

            // Compute numerical gradient for sampled indices
            Tensor input_cpu = input.device() == Device::CUDA ? input.cpu() : input.clone();
            float* input_data = input_cpu.ptr<float>();

            result.num_checked = num_samples;

            for (const size_t i : indices) {
                const float orig = input_data[i];

                input_data[i] = orig + eps;
                Tensor input_plus = input.device() == Device::CUDA ? input_cpu.cuda() : input_cpu;
                Tensor out_plus = forward(input_plus);
                if (out_plus.numel() > 1)
                    out_plus = out_plus.sum();
                const float f_plus = out_plus.cpu().ptr<float>()[0];

                input_data[i] = orig - eps;
                Tensor input_minus = input.device() == Device::CUDA ? input_cpu.cuda() : input_cpu;
                Tensor out_minus = forward(input_minus);
                if (out_minus.numel() > 1)
                    out_minus = out_minus.sum();
                const float f_minus = out_minus.cpu().ptr<float>()[0];

                input_data[i] = orig;

                const float num_grad = (f_plus - f_minus) / (2.0f * eps);
                const float anal = anal_data[i];
                const float abs_err = std::abs(anal - num_grad);
                const float rel_err = abs_err / (std::max(std::abs(anal), std::abs(num_grad)) + REL_ERR_EPSILON);

                result.max_abs_error = std::max(result.max_abs_error, abs_err);
                result.max_rel_error = std::max(result.max_rel_error, rel_err);

                if (abs_err > atol + rtol * std::abs(num_grad)) {
                    ++result.num_failed;
                }
            }

            result.passed = result.num_failed == 0;
            return result;
        }
    };

} // namespace lfs::core::debug

// Convenience macro for gradient checking
#define CHECK_GRADIENT(forward_fn, input, analytical_grad)                                            \
    do {                                                                                              \
        auto _result = lfs::core::debug::GradientChecker::check(forward_fn, input, analytical_grad);  \
        if (!_result.passed) {                                                                        \
            LOG_WARN("Gradient check failed at {}:{} - {}", __FILE__, __LINE__, _result.to_string()); \
        }                                                                                             \
    } while (0)

#define CHECK_GRADIENT_SAMPLED(forward_fn, input, analytical_grad, samples)                                           \
    do {                                                                                                              \
        auto _result = lfs::core::debug::GradientChecker::check_sampled(forward_fn, input, analytical_grad, samples); \
        if (!_result.passed) {                                                                                        \
            LOG_WARN("Gradient check failed at {}:{} - {}", __FILE__, __LINE__, _result.to_string());                 \
        }                                                                                                             \
    } while (0)
