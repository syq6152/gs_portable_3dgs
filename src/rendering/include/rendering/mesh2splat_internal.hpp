/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace lfs::rendering::detail {

    struct Mesh2SplatMappedFace {
        int32_t constraint_face = -1;
        bool hole_fill = false;
    };

    // Kept as a pure CPU boundary so raw shader face IDs and the legacy
    // no-mapping behavior can be regression-tested without an OpenGL context.
    [[nodiscard]] std::expected<Mesh2SplatMappedFace, std::string>
    map_mesh2splat_render_face(
        int32_t raw_render_face,
        std::span<const int32_t> render_face_to_constraint_face);

    [[nodiscard]] std::expected<void, std::string>
    validate_mesh2splat_face_mapping_contract(
        std::size_t render_face_count,
        std::size_t constraint_face_count,
        bool uses_separate_constraint_mesh,
        std::span<const int32_t> render_face_to_constraint_face);

    [[nodiscard]] constexpr bool mesh2splat_uses_hole_fill_provenance(
        const std::span<const int32_t> render_face_to_constraint_face) noexcept {
        return !render_face_to_constraint_face.empty();
    }

    // The historical single-mesh overload may exceed target_max_gaussians at
    // the minimum resolution. Preserve that behavior unless an explicit face
    // mapping identifies the combined render/constraint conversion path.
    [[nodiscard]] constexpr bool mesh2splat_enforces_combined_hard_cap(
        const std::span<const int32_t> render_face_to_constraint_face) noexcept {
        return !render_face_to_constraint_face.empty();
    }

    // Deterministic midpoint sampling across the complete converter output;
    // unlike prefix truncation, this does not systematically drop later submeshes.
    [[nodiscard]] std::vector<bool> build_mesh2splat_uniform_cap_keep_mask(
        std::size_t row_count,
        std::size_t target_row_count);

} // namespace lfs::rendering::detail
