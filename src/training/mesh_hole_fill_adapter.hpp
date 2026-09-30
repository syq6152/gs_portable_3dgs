/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh_data.hpp"
#include "geometry/mesh_hole_fill.hpp"
#include "rendering/mesh2splat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace lfs::training::mesh_hole_fill_adapter {

    inline constexpr double DENSITY_RETRY_MIN_FACTOR_MULTIPLIER = 0.25;
    inline constexpr double DENSITY_RETRY_MAX_FACTOR_MULTIPLIER = 4.0;

    // Keep texture-color transfer consistent with converterGS.glsl: a face
    // crossing a wrapped UV seam is not a valid source of albedo color.
    [[nodiscard]] inline bool is_transferable_uv_triangle(
        const glm::vec2& uv0,
        const glm::vec2& uv1,
        const glm::vec2& uv2) {
        const float min_u = std::min({uv0.x, uv1.x, uv2.x});
        const float max_u = std::max({uv0.x, uv1.x, uv2.x});
        const float min_v = std::min({uv0.y, uv1.y, uv2.y});
        const float max_v = std::max({uv0.y, uv1.y, uv2.y});
        return std::isfinite(min_u) && std::isfinite(max_u) &&
               std::isfinite(min_v) && std::isfinite(max_v) &&
               max_u - min_u <= 0.5f && max_v - min_v <= 0.5f;
    }

    [[nodiscard]] inline bool has_transferable_albedo_alpha(
        const float sampled_alpha,
        const float material_alpha) {
        constexpr float MIN_TRANSFERABLE_ALPHA = 1.0f / 255.0f;
        return std::isfinite(sampled_alpha) && std::isfinite(material_alpha) &&
               sampled_alpha * material_alpha > MIN_TRANSFERABLE_ALPHA;
    }

    struct HoleDensityRecord {
        std::size_t hole_id = 0;
        std::size_t patch_gaussian_count = 0;
        std::size_t neighbor_gaussian_count = 0;
        double patch_density = 0.0;
        double neighbor_density = 0.0;
        double density_ratio = 0.0;
        bool accepted = false;
    };

    struct HoleDensityCheck {
        std::vector<HoleDensityRecord> records;
        std::unordered_set<std::size_t> rejected_hole_ids;
    };

    [[nodiscard]] std::expected<core::MeshData, std::string>
    assemble_hole_fill_render_mesh(
        const core::MeshData& original_mesh,
        const geometry::MeshHoleFillResult& fill_result);

    [[nodiscard]] std::expected<HoleDensityCheck, std::string>
    check_hole_fill_density(
        const geometry::MeshHoleFillResult& fill_result,
        const rendering::Mesh2SplatFaceStatistics& statistics,
        double min_density_ratio,
        double max_density_ratio);

    // Builds the one-shot, non-config density overrides indexed by stable hole_id.
    // The correction uses factor * sqrt(nearest_allowed_ratio / observed_ratio),
    // then clamps the multiplier to [1/4, 4]. The square root is a damped update
    // for an area density controlled by a linear refinement factor.
    [[nodiscard]] std::expected<std::vector<std::optional<double>>, std::string>
    build_density_retry_overrides(
        const geometry::MeshHoleFillResult& fill_result,
        const HoleDensityCheck& density_check,
        double base_density_control_factor,
        double min_density_ratio,
        double max_density_ratio);

    // Preserves first-pass accepted patches byte-for-byte and takes only the
    // retry result for holes named in retry_hole_ids. Appended vertices are
    // remapped transactionally into a shared output index space.
    [[nodiscard]] std::expected<geometry::MeshHoleFillResult, std::string>
    merge_density_retry_fill_results(
        const geometry::MeshHoleFillResult& first_fill_result,
        const geometry::MeshHoleFillResult& retry_fill_result,
        const std::unordered_set<std::size_t>& retry_hole_ids);

    [[nodiscard]] geometry::MeshHoleFillResult filter_rejected_holes(
        const geometry::MeshHoleFillResult& fill_result,
        const std::unordered_set<std::size_t>& rejected_hole_ids);

    // Returns one keep bit per converted Gaussian. Original-face rows and rows
    // belonging to accepted holes are retained; rows from rejected holes are removed.
    [[nodiscard]] std::expected<std::vector<bool>, std::string>
    build_density_row_keep_mask(
        const geometry::MeshHoleFillResult& fill_result,
        const rendering::Mesh2SplatFaceStatistics& statistics,
        const std::unordered_set<std::size_t>& rejected_hole_ids);

} // namespace lfs::training::mesh_hole_fill_adapter
