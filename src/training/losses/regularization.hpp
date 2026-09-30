/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief L1 regularization on exp(scaling_raw) with fused CUDA kernel
     *
     * Forward:  scaling = exp(scaling_raw)
     * Loss:     L = weight * mean(scaling)
     * Gradient: ∂L/∂scaling_raw = (weight / N) * exp(scaling_raw)
     *
     * NOTE: This loss writes gradients directly to scaling_raw_grad in-place
     */
    struct ScaleRegularization {
        struct Params {
            float weight; ///< Regularization weight
        };

        /**
         * @brief Compute scale regularization loss and accumulate gradients
         * @param scaling_raw [N, 3] raw scaling parameters
         * @param scaling_raw_grad [N, 3] gradient tensor (will be accumulated to)
         * @param params Loss parameters
         * @return loss_tensor (GPU) or error - loss stays on GPU!
         * @note Accumulates gradients directly to scaling_raw_grad
         */
        static std::expected<lfs::core::Tensor, std::string> forward(
            const lfs::core::Tensor& scaling_raw,
            lfs::core::Tensor& scaling_raw_grad,
            const Params& params);
    };

    /**
     * @brief L1 regularization on sigmoid(opacity_raw) with fused CUDA kernel
     *
     * Forward:  opacity = sigmoid(opacity_raw)
     * Loss:     L = weight * mean(opacity)
     * Gradient: ∂L/∂opacity_raw = (weight / N) * sigmoid(x) * (1 - sigmoid(x))
     *
     * NOTE: This loss writes gradients directly to opacity_raw_grad in-place
     */
    struct OpacityRegularization {
        struct Params {
            float weight; ///< Regularization weight
        };

        /**
         * @brief Compute opacity regularization loss and accumulate gradients
         * @param opacity_raw [N, 1] raw opacity parameters
         * @param opacity_raw_grad [N, 1] gradient tensor (will be accumulated to)
         * @param params Loss parameters
         * @return loss_tensor (GPU) or error - loss stays on GPU!
         * @note Accumulates gradients directly to opacity_raw_grad
         */
        static std::expected<lfs::core::Tensor, std::string> forward(
            const lfs::core::Tensor& opacity_raw,
            lfs::core::Tensor& opacity_raw_grad,
            const Params& params);
    };

    struct MeshSurfaceRegularization {
        struct Params {
            float lambda_project = 0.0f;
            float lambda_outside_barrier = 0.0f;
            float outside_distance_threshold = 0.0f;
            float inside_constraint_distance = 0.0f;
            float inside_constraint_fade_ratio = 0.2f;
            float lambda_scale_min = 0.0f;
            float lambda_scale_max = 0.0f;
            float rho = 0.01f;
            float lambda_normal = 0.0f;
            int walk_steps = 1;
        };

        static std::expected<void, std::string> mark_visible_primitives(
            const uint32_t* visible_primitive_indices,
            size_t n_visible_primitives,
            size_t n_primitives,
            const lfs::core::Tensor& current_faces,
            lfs::core::Tensor& visible_mask,
            lfs::core::Tensor& visible_count);

        static std::expected<lfs::core::Tensor, std::string> forward(
            const lfs::core::Tensor& means,
            const lfs::core::Tensor& scaling_raw,
            const lfs::core::Tensor& rotation_raw,
            lfs::core::Tensor& means_grad,
            lfs::core::Tensor& scaling_raw_grad,
            lfs::core::Tensor& rotation_raw_grad,
            const lfs::core::Tensor* visible_mask,
            const lfs::core::Tensor* visible_count,
            lfs::core::Tensor& current_faces,
            const lfs::core::Tensor& edge_neighbors,
            const lfs::core::Tensor& mesh_vertices,
            const lfs::core::Tensor& mesh_indices,
            const lfs::core::Tensor* face_normals,
            const Params& params);
    };

} // namespace lfs::training::losses
