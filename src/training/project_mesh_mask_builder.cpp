/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "project_mesh_mask_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lfs::training {

    namespace {

        // Matches NEAR_Z in mesh_supervision_renderer.cpp, so a point is rejected on the
        // same side of the near plane as the triangle rasterizer would reject it.
        constexpr float kNearZ = 1e-4f;

        // Occlusion tolerance. Scene space is the mesh divided by
        // MESH_SCENE_POSITION_DIVISOR, so an absolute threshold has no natural value here;
        // the relative term carries the decision and the absolute term only guards against
        // a degenerate depth of nearly zero.
        constexpr float kOcclusionRelativeTolerance = 1e-3f;
        constexpr float kOcclusionAbsoluteTolerance = 1e-5f;

        // A point very close to the camera projects to an enormous disc. Capping keeps one
        // outlier from stamping over the whole image.
        constexpr int kMaxDiscRadiusPixels = 64;

        // Ceiling on the auto-derived closing radius. Closing seals every gap narrower than
        // twice the radius regardless of evidence, so an unbounded radius would quietly
        // erase the interior path's "no evidence, no fill" guarantee.
        constexpr int kMaxAutoCloseRadiusPixels = 16;

        struct AcceptedPoint {
            int x = 0;
            int y = 0;
            float z = 0.0f;
        };

        void stamp_disc(BinaryMask& mask, const int cx, const int cy, const int radius) {
            const int r = std::max(radius, 0);
            const int r_squared = r * r;
            const int min_y = std::max(0, cy - r);
            const int max_y = std::min(mask.height - 1, cy + r);
            const int min_x = std::max(0, cx - r);
            const int max_x = std::min(mask.width - 1, cx + r);
            for (int y = min_y; y <= max_y; ++y) {
                const int dy = y - cy;
                for (int x = min_x; x <= max_x; ++x) {
                    const int dx = x - cx;
                    if (dx * dx + dy * dy <= r_squared) {
                        mask.set(x, y, true);
                    }
                }
            }
        }

    } // namespace

    BinaryMask build_project_mesh_coverage(
        const float* depth,
        const int width,
        const int height,
        const std::vector<glm::vec3>& camera_space_points,
        const PinholeIntrinsics& intrinsics,
        const ProjectMaskHoleFillOptions& options,
        ProjectMaskHoleFillStats& stats) {

        stats = ProjectMaskHoleFillStats{};

        BinaryMask coverage = make_binary_mask(width, height);
        if (!coverage.is_valid() || depth == nullptr) {
            return coverage;
        }
        for (size_t i = 0; i < coverage.values.size(); ++i) {
            coverage.values[i] = depth[i] > 0.0f ? uint8_t{1} : uint8_t{0};
        }

        if (camera_space_points.empty()) {
            // No evidence for this camera, so no hole may be filled. Plain mesh coverage.
            return coverage;
        }

        // --- Project the cloud and reject points the mesh occludes -------------------
        BinaryMask point_pixels = make_binary_mask(width, height);
        std::vector<AcceptedPoint> accepted;
        accepted.reserve(camera_space_points.size());

        for (const auto& p : camera_space_points) {
            if (!(p.z > kNearZ)) {
                continue;
            }
            const float u = intrinsics.fx * (p.x / p.z) + intrinsics.cx;
            const float v = intrinsics.fy * (p.y / p.z) + intrinsics.cy;
            if (!std::isfinite(u) || !std::isfinite(v)) {
                continue;
            }
            // The rasterizer samples pixel centres at (px + 0.5, py + 0.5), so the pixel a
            // projected point lands in is the floor of its screen coordinate.
            const int px = static_cast<int>(std::floor(u));
            const int py = static_cast<int>(std::floor(v));
            if (px < 0 || py < 0 || px >= width || py >= height) {
                continue;
            }
            ++stats.points_in_frustum;

            const float mesh_depth = depth[static_cast<size_t>(py) * static_cast<size_t>(width) +
                                           static_cast<size_t>(px)];
            if (mesh_depth > 0.0f &&
                p.z > mesh_depth * (1.0f + kOcclusionRelativeTolerance) + kOcclusionAbsoluteTolerance) {
                // The mesh is in front of this point: it belongs to a surface hidden from
                // this view, not to a hole visible in it.
                continue;
            }
            ++stats.points_accepted;
            point_pixels.set(px, py, true);
            accepted.push_back({px, py, p.z});
        }

        // --- Interior holes: 2D topology draws the boundary, points authorize the fill --
        const BinaryMask rim_closed = close_binary_mask(coverage, options.rim_close_pixels);
        const auto components = label_background_components(rim_closed);
        const size_t component_count = components.component_count();

        std::vector<size_t> evidence(component_count, 0);
        if (component_count > 0) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (!point_pixels.at(x, y)) {
                        continue;
                    }
                    const int32_t label = components.label_at(x, y);
                    if (label >= 0) {
                        ++evidence[static_cast<size_t>(label)];
                    }
                }
            }
        }

        const auto min_points = static_cast<size_t>(std::max(1, options.min_points_per_hole));
        std::vector<uint8_t> authorized(component_count, 0);
        for (size_t label = 0; label < component_count; ++label) {
            if (!components.is_enclosed(static_cast<int32_t>(label))) {
                continue;
            }
            ++stats.enclosed_holes;
            if (evidence[label] >= min_points) {
                authorized[label] = 1;
                ++stats.filled_holes;
            }
        }

        BinaryMask fill = make_binary_mask(width, height);
        if (stats.filled_holes > 0) {
            for (size_t i = 0; i < fill.values.size(); ++i) {
                const int32_t label = components.labels[i];
                if (label >= 0 && authorized[static_cast<size_t>(label)] != 0) {
                    fill.values[i] = uint8_t{1};
                    ++stats.filled_pixels;
                }
            }
        }

        // Union against the ORIGINAL coverage, not the rim-closed one: the pre-clean exists
        // only to make labeling robust and must not leak its dilation into the saved mask.
        BinaryMask result = union_binary_mask(coverage, fill);
        if (stats.filled_pixels > 0 && options.rim_close_pixels > 0) {
            // The rim notches the pre-clean sealed sit between the mesh coverage and the
            // filled block, and belong to neither. Without them the two are separated by a
            // one-pixel unset ring that gates the loss off exactly at the hole boundary.
            // Restricting to a dilation of the fill keeps unauthorized holes untouched.
            result = union_binary_mask(
                result,
                intersect_binary_mask(
                    rim_closed, dilate_binary_mask(fill, options.rim_close_pixels)));
        }

        // --- Fallback for holes open to the silhouette border ------------------------
        // A gap on the object's outline reaches the image border, so it is never enclosed
        // and the path above cannot see it. Only here do the points have to draw coverage
        // themselves; closing turns "seamless stamping", a 3D sampling property, into a 2D
        // one, and dilate-then-erode restores the outline so it cannot bleed outward.
        if (options.splat_radius_scale > 0.0f && options.point_spacing > 0.0f && !accepted.empty()) {
            const float focal = 0.5f * (intrinsics.fx + intrinsics.fy);
            BinaryMask discs = make_binary_mask(width, height);
            std::vector<float> projected_spacing;
            projected_spacing.reserve(accepted.size());

            for (const auto& point : accepted) {
                const float spacing_px = focal * options.point_spacing / point.z;
                if (!std::isfinite(spacing_px) || spacing_px <= 0.0f) {
                    continue;
                }
                projected_spacing.push_back(spacing_px);
                const int radius = std::clamp(
                    static_cast<int>(std::lround(options.splat_radius_scale * spacing_px)),
                    0,
                    kMaxDiscRadiusPixels);
                stamp_disc(discs, point.x, point.y, radius);
            }

            int close_radius = options.close_pixels;
            if (close_radius <= 0 && !projected_spacing.empty()) {
                // Derive from the projected spacing. The 95th percentile rather than the
                // maximum: one point near the camera would otherwise dictate a radius large
                // enough to seal every hole in the image.
                const size_t rank = std::min(
                    projected_spacing.size() - 1,
                    static_cast<size_t>(projected_spacing.size() * 95 / 100));
                std::nth_element(projected_spacing.begin(),
                                 projected_spacing.begin() + static_cast<std::ptrdiff_t>(rank),
                                 projected_spacing.end());
                close_radius = static_cast<int>(std::ceil(projected_spacing[rank]));
                if (close_radius > kMaxAutoCloseRadiusPixels) {
                    close_radius = kMaxAutoCloseRadiusPixels;
                    stats.fallback_radius_capped = true;
                }
            }
            close_radius = std::max(close_radius, 0);
            stats.fallback_close_radius = close_radius;

            const size_t before = count_binary_mask(result);
            BinaryMask widened = close_binary_mask(union_binary_mask(result, discs), close_radius);
            const size_t after = count_binary_mask(widened);
            stats.fallback_pixels = after > before ? after - before : 0;
            result = std::move(widened);
        }

        return result;
    }

} // namespace lfs::training
