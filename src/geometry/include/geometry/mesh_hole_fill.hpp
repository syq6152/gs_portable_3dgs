/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

namespace lfs::geometry {

    // LibTorch- and CGAL-free input view. triangle_indices is a flat array with
    // exactly three indices per face.
    struct MeshHoleFillInput {
        std::span<const glm::vec3> vertices;
        std::span<const uint32_t> triangle_indices;
    };

    struct MeshHoleFillOptions {
        std::size_t max_boundary_edges = 128;
        double max_boundary_perimeter_bbox_ratio = 0.10;
        double max_boundary_area_bbox_ratio = 0.0025;
        double max_boundary_bbox_diagonal_ratio = 0.05;
        double weld_epsilon_bbox_ratio = 1e-6;
        double density_control_factor = 1.41421356;

        // Internal retry input, not a user-facing configuration surface. An
        // engaged entry overrides density_control_factor for the boundary whose
        // stable hole_id equals the entry index.
        std::vector<std::optional<double>> density_control_factor_overrides;
        // Retry-only mode keeps unselected holes out of the retry result so the
        // caller can retain their already accepted first-pass patches verbatim.
        bool fill_only_density_control_factor_overrides = false;
    };

    enum class HoleFillStatus {
        Filled,
        Skipped,
        Failed,
    };

    struct HoleFillDiagnostic {
        std::size_t hole_id = 0;
        HoleFillStatus status = HoleFillStatus::Failed;
        std::string reason;

        std::size_t boundary_edge_count = 0;
        double boundary_perimeter = 0.0;
        double boundary_projected_area = 0.0;
        double boundary_bbox_diagonal = 0.0;
        double boundary_perimeter_bbox_ratio = 0.0;
        double boundary_area_bbox_ratio = 0.0;
        double boundary_bbox_diagonal_ratio = 0.0;

        std::size_t neighbor_face_count = 0;
        // Original input face IDs in the one-ring adjacent to this boundary.
        std::vector<uint32_t> neighbor_face_indices;
        double neighbor_area = 0.0;
        double neighbor_median_edge_length = 0.0;
        double neighbor_median_face_area = 0.0;

        std::size_t appended_vertex_count = 0;
        std::size_t appended_face_count = 0;
        double patch_area = 0.0;
        double patch_median_edge_length = 0.0;
        double density_control_factor = 0.0;

        bool topology_valid = false;
        bool boundary_self_intersects = false;
        bool normals_consistent = false;
        bool patch_self_intersects = false;
        bool patch_manifold = false;
        bool patch_normals_consistent = false;
    };

    enum class MeshHoleFillErrorCode {
        InvalidOptions,
        InvalidInput,
        InvalidTopology,
        SelfIntersection,
    };

    struct MeshHoleFillError {
        MeshHoleFillErrorCode code = MeshHoleFillErrorCode::InvalidInput;
        std::string message;
        std::size_t face_index = 0;
        bool has_face_index = false;
    };

    struct MeshHoleFillResult {
        std::size_t original_vertex_count = 0;
        std::size_t original_face_count = 0;
        std::size_t welded_vertex_count = 0;
        std::size_t appended_face_count = 0;
        std::size_t successful_hole_count = 0;
        std::size_t skipped_hole_count = 0;
        std::size_t failed_hole_count = 0;
        double mesh_bbox_diagonal = 0.0;
        double weld_epsilon = 0.0;

        // Append these vertices to the original render-vertex array. Every
        // appended triangle index addresses that combined array, so new vertex
        // indices start at original_vertex_count. Patch triangles can reference
        // original boundary vertices; an adapter that needs patch-only vertex
        // attributes may duplicate/remap those boundary corners while assembling
        // its render mesh.
        std::vector<glm::vec3> appended_vertices;
        std::vector<uint32_t> appended_triangle_indices;
        std::vector<std::size_t> appended_face_hole_ids;

        // Each entry maps an input vertex (including UV-seam duplicates) to the
        // representative original input vertex used by the welded topology.
        // No welded-only vertex is exposed in the output index space.
        std::vector<uint32_t> input_vertex_to_welded_representative;

        std::vector<HoleFillDiagnostic> diagnostics;
    };

    // The input is never mutated. A rejected or failed individual hole cannot
    // leave partial geometry in the returned patch arrays.
    [[nodiscard]] std::expected<MeshHoleFillResult, MeshHoleFillError>
    fill_mesh_holes(const MeshHoleFillInput& input,
                    const MeshHoleFillOptions& options = {});

} // namespace lfs::geometry
