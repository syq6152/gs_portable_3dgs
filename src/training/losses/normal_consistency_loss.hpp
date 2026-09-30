/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <string>

namespace lfs::training::losses {

    /**
     * @brief GGGS Equation 25: Normal consistency loss
     *
     * Computes L = mean(1 - dot(render_normal, depth_normal)) over valid pixels.
     *
     * When mesh_gt_depth is used for depth_normal computation, no gradients
     * flow through the depth-to-normal branch (the target is constant).
     * Only gradients through render_normal are computed.
     *
     * Forward:
     *   depth_normal = depth_to_normal_map(mesh_gt_depth, fx, fy, cx, cy)  [constant]
     *   error_map = 1 - vecdot(render_normal, depth_normal)
     *   loss = mean(error_map) over valid pixels
     *
     * Backward:
     *   dL/d(render_normal) = -depth_normal / N_valid   (per valid pixel)
     *   dL/d(normal_accum) = normalize_backward(dL/d(render_normal), render_normal, nlen)
     */
    struct NormalConsistencyLoss {
        struct Params {
            float lambda = 0.05f;
        };

        struct Context {
            lfs::core::Tensor grad_render_normal; // [3, H, W] gradient w.r.t. rendered normal
            lfs::core::Tensor grad_depth_normal;  // [3, H, W] gradient w.r.t. depth-derived normal
            float loss_value = 0.0f;
        };

        /**
         * @brief Compute normal consistency loss and gradients
         * @param render_normal [3, H, W] alpha-blended normal from rasterizer (CUDA)
         * @param depth_normal [3, H, W] normal derived from mesh GT depth (CUDA or CPU)
         * @param depth [H, W] or [1, H, W] depth map for validity checking (CUDA or CPU)
         * @param params Loss parameters
         * @return {loss_tensor [1] on GPU, context with gradients} or error
         */
        static std::expected<std::pair<lfs::core::Tensor, Context>, std::string> forward(
            const lfs::core::Tensor& render_normal,
            const lfs::core::Tensor& depth_normal,
            const lfs::core::Tensor& depth,
            const Params& params);
    };

} // namespace lfs::training::losses
