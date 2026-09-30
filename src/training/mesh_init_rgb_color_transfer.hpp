/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"

#include <expected>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::training {

    struct MeshInitRgbColorTransferStats {
        size_t cameras_considered = 0;
        size_t cameras_rendered = 0;
        size_t cameras_failed = 0;
        size_t transferred_gaussians = 0;
        size_t fallback_gaussians = 0;
        size_t projected_points = 0;
        // Number of first-hit RGB assignments (not a multi-view observation sum).
        size_t valid_observations = 0;
        size_t projection_failures = 0;
        size_t occlusion_failures = 0;
        size_t source_mask_failures = 0;
    };

    std::expected<MeshInitRgbColorTransferStats, std::string>
    transfer_mesh_init_rgb_colors(
        lfs::core::SplatData& model,
        const lfs::core::MeshData& normalized_mesh,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& train_cameras,
        const lfs::core::param::TrainingParameters& params);

} // namespace lfs::training
