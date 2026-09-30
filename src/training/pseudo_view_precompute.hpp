/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::training {

    struct PseudoViewPrecomputeStats {
        size_t total_train_cameras = 0;
        size_t sphere_cells = 0;
        size_t occupied_sphere_cells = 0;
        size_t empty_sphere_cells = 0;
        size_t supported_sphere_cells = 0;
        size_t pseudo_candidates = 0;        // candidate poses rendered and source-scored
        size_t pseudo_views_written = 0;     // pseudo views with valid_ratio >= threshold
        size_t pseudo_views_unobservable = 0;
        std::filesystem::path output_dir;
        std::filesystem::path manifest_path;
    };

    // Run the pseudo-view RGB supervision precompute pipeline:
    //   1. Map real camera poses to sphere cells around the mesh center; occupied cells are skipped.
    //   2. For each empty cell, compute a mesh-surface target from face normals aligned with the
    //      cell direction and generate one look-at pseudo pose at the median real-camera radius.
    //   3. Hard-filter poses that face away from the mesh or view the supported surface from the back side.
    //   4. For each accepted pseudo camera, evaluate top-K real cameras through mesh-depth
    //      reprojection and use the single source with the highest valid ratio for RGB.
    //   5. Save per-pseudo-view rgb / valid mask / mesh mask / pose JSON, plus a top-level manifest.
    //
    // This step is read-only with respect to training data (it only needs the loaded Scene + train
    // camera list); P3/P4 (training-side ingestion + low-weight loss) are not handled here.
    std::expected<PseudoViewPrecomputeStats, std::string> run_pseudo_view_precompute(
        lfs::core::Scene& scene,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& train_cameras,
        const lfs::core::param::TrainingParameters& params);

} // namespace lfs::training
