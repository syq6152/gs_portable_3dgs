/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "mesh_supervision_renderer.hpp"

#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::training {

    struct ScanPoseFilterPairScore {
        std::size_t peer_index = 0;
        float zncc = 0.0f;
        // Raw photometric patch ZNCC. Source mesh depth is not used to reject
        // a patch; target mesh depth is only used to establish correspondence.
        std::optional<float> photometric_zncc;
        std::size_t informative_regions = 0;
        // Number of projected samples whose source/target RGB values were
        // finite; retained as `valid_samples` in the JSON report for backward
        // compatibility with reports from the first implementation.
        std::size_t valid_samples = 0;
        std::size_t target_samples = 0;
        std::size_t projected_samples = 0;
        std::size_t depth_consistent_samples = 0;
        // Retained for report/API compatibility. These values no longer affect
        // filtering after source mesh-depth rejection was removed.
        std::size_t geometry_consistent_regions = 0;
    };

    struct ScanPoseFilterFrameScore {
        // A missing score means that the frame had no admissible correspondence
        // evidence.  Callers must treat it as uncertain and retain the frame.
        std::optional<float> zncc;
        std::optional<float> photometric_zncc;
        bool keep = true;
        std::vector<ScanPoseFilterPairScore> pair_scores;
    };

    struct ScanPoseFilterResult {
        std::vector<ScanPoseFilterFrameScore> frames;
        std::size_t kept_count = 0;
        std::size_t dropped_count = 0;
    };

    // A missing frame score means that no target-mesh-masked patch produced
    // admissible photometric evidence (for example, all patches were outside
    // the mask, out of view, or low-texture). Such a frame is retained as
    // uncertain because it has no valid ZNCC evidence.
    [[nodiscard]] inline bool keep_scan_pose_frame_by_zncc(
        const std::optional<float> zncc,
        const float zncc_threshold) noexcept {
        return !zncc || *zncc >= zncc_threshold;
    }

    // Score scanner-camera poses against one another through the prepared mesh.
    //
    // The cameras are expected to contain scanner c2w poses converted to the
    // Camera class's w2c representation and undistorted pinhole intrinsics. RGB
    // patches are compared at pose-derived correspondences. Target mesh depth
    // establishes those correspondences; source mesh depth is not an additional
    // rejection criterion. Pixels outside the target mesh projection mask and
    // patches without enough valid RGB correspondences are omitted from ZNCC.
    [[nodiscard]] std::expected<ScanPoseFilterResult, std::string>
    score_scan_pose_frames_zncc(
        const PreparedMesh& prepared_mesh,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        float zncc_threshold);

    // Descriptive alias used by standalone command runners.
    [[nodiscard]] inline std::expected<ScanPoseFilterResult, std::string>
    filter_error_scan_poses_zncc(
        const PreparedMesh& prepared_mesh,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const float zncc_threshold) {
        return score_scan_pose_frames_zncc(prepared_mesh, cameras, zncc_threshold);
    }

} // namespace lfs::training
