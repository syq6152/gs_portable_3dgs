/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <memory>
#include <rasterization_api.h>
#include <string>

namespace lfs::training {
    class FastRasterizeFrameLease;

    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        lfs::core::Tensor image;
        lfs::core::Tensor alpha;
        lfs::core::Tensor bg_color; // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        lfs::core::Tensor means;
        lfs::core::Tensor raw_scales;
        lfs::core::Tensor raw_rotations;
        lfs::core::Tensor raw_opacities;
        lfs::core::Tensor shN;

        const float* w2c_ptr = nullptr;
        const float* cam_position_ptr = nullptr;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_lfs::rasterization::ForwardContext forward_ctx;

        // Per-frame Robust-GS observation blur settings borrowed from the
        // trainer's persistent nuisance-parameter tensor.  The row remains
        // alive until backward consumes this context; disabled renders carry
        // an all-default settings value and follow the legacy path.
        fast_lfs::rasterization::ObservationBlurSettings observation_blur;

        int active_sh_bases;
        int total_bases_sh_rest;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
        bool mip_filter = false;

        // Tile information (for tile-based training)
        int tile_x_offset = 0; // Horizontal offset of this tile
        int tile_y_offset = 0; // Vertical offset of this tile
        int tile_width = 0;    // Width of this tile (0 = full image)
        int tile_height = 0;   // Height of this tile (0 = full image)

        // Background image for per-pixel blending (optional, empty = use bg_color)
        lfs::core::Tensor bg_image;

        // GGGS: whether normal backward buffers are allocated
        bool require_normal_backward = false;

        // Shared ownership ensures inference/discarded contexts release the
        // arena frame exactly once, while backward can atomically claim it.
        std::shared_ptr<FastRasterizeFrameLease> frame_lease;
    };

    // Explicit forward pass - returns render output and context for backward
    // Optional tile parameters for memory-efficient training (tile_width/height=0 means full image)
    // bg_image is optional - if provided, uses per-pixel background blending instead of solid color
    std::expected<std::pair<RenderOutput, FastRasterizeContext>, std::string> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {},
        bool require_depth_normal = false,
        bool require_normal_backward = false,
        const lfs::core::Tensor& mesh_depth_cull = {},
        bool enable_mesh_depth_cull = false,
        const fast_lfs::rasterization::ObservationBlurSettings& observation_blur = {});

    // Backward pass with optional extra alpha gradient for masked training
    // grad_render_normal: optional [3,H,W] gradient from GGGS normal losses
    // grad_depth_normal: optional [3,H,W] gradient w.r.t. depth-derived normals (triggers depth backward via depth_to_normal)
    // depth: optional [H,W] or [1,H,W] depth map from forward (required when grad_depth_normal or grad_depth_direct is provided)
    // grad_depth_direct: optional [H,W] direct gradient w.r.t. rendered depth (e.g. from inverse depth loss)
    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra = {},
        const lfs::core::Tensor& pixel_error_map = {},
        const lfs::core::Tensor& grad_render_normal = {},
        const lfs::core::Tensor& render_normal = {},
        const lfs::core::Tensor& grad_depth_normal = {},
        const lfs::core::Tensor& depth_map = {},
        const lfs::core::Tensor& grad_depth_direct = {},
        lfs::core::Tensor* grad_w2c_out = nullptr);

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color) {
        auto result = fast_rasterize_forward(viewpoint_camera, gaussian_model, bg_color);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        return result->first;
    }
} // namespace lfs::training
