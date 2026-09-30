/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "binary_mask_ops.hpp"

#include <glm/glm.hpp>

#include <cstddef>
#include <vector>

namespace lfs::training {

    // Coverage-mask construction for project_mesh masking in external_pointcloud mode.
    //
    // The mesh depth buffer alone leaves the scan's holes uncovered, so the Gaussians the
    // external point cloud seeded there never receive photometric gradient and keep their
    // white initialization forever. This fills those holes.
    //
    // The point cloud is evidence, not a brush: 2D topology decides where a hole is (a
    // background component that does not reach the image border must be spanned by the
    // surface), and the projected points only decide whether that hole is a scan gap worth
    // filling or a genuine through-hole such as a handle's opening. That keeps the filled
    // region exactly bounded by the silhouette and free of radius tuning.
    //
    // Everything here is pure CPU arithmetic on a depth buffer and camera-space points, so
    // the geometry decisions are unit-testable without CUDA, a Camera, or file IO.

    struct PinholeIntrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct ProjectMaskHoleFillOptions {
        // Interior-hole path. This is the only knob that affects it.
        int min_points_per_hole = 4;
        // Pre-clean applied to mesh coverage before component labeling only. Without it a
        // single-pixel notch at a hole's rim lets the hole leak to the image border, which
        // reclassifies it as open and silently skips the fill.
        int rim_close_pixels = 1;

        // Fallback path for holes that open onto the silhouette border, where 2D topology
        // cannot say where the object's outline should be. 0 disables it.
        float splat_radius_scale = 1.5f;
        int close_pixels = 0;    // 0 derives the radius from the projected point spacing
        float point_spacing = 0.0f; // scene-space spacing the cloud was downsampled to
    };

    struct ProjectMaskHoleFillStats {
        size_t points_in_frustum = 0; // in front of the camera and inside the image
        size_t points_accepted = 0;   // also survived the occlusion test
        size_t enclosed_holes = 0;    // background components not touching the border
        size_t filled_holes = 0;      // of those, the ones with enough point evidence
        size_t filled_pixels = 0;
        size_t fallback_pixels = 0; // extra pixels the disc + closing path contributed
        int fallback_close_radius = 0;
        bool fallback_radius_capped = false;
    };

    // `depth` is [height * width] row-major; <= 0 means the mesh did not cover that pixel.
    // `camera_space_points` are already transformed by the camera's world-to-camera matrix.
    // Returns the coverage mask BEFORE the caller's erosion step.
    [[nodiscard]] BinaryMask build_project_mesh_coverage(
        const float* depth,
        int width,
        int height,
        const std::vector<glm::vec3>& camera_space_points,
        const PinholeIntrinsics& intrinsics,
        const ProjectMaskHoleFillOptions& options,
        ProjectMaskHoleFillStats& stats);

} // namespace lfs::training
