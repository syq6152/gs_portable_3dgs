/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/mesh2splat.hpp"
#include "core/splat_data.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

namespace lfs::core {
    struct MeshData;
}

namespace lfs::rendering {

    struct Mesh2SplatFaceStatistics {
        std::vector<std::size_t> gaussian_count_per_render_face;
        // Raw shader/render-face ID for every output row, before mapping to the
        // original constraint mesh. Populated only when statistics are requested.
        std::vector<int32_t> raw_render_face_per_gaussian;
        std::size_t gaussian_count = 0;
        int base_resolution = 0;
        int requested_resolution = 0;
        int final_resolution = 0;
    };

    struct Mesh2SplatInput {
        const core::MeshData& render_mesh;
        const core::MeshData* constraint_mesh = nullptr;
        std::span<const int32_t> render_face_to_constraint_face;
        Mesh2SplatFaceStatistics* face_statistics = nullptr;
    };

    [[nodiscard]] std::expected<std::unique_ptr<core::SplatData>, std::string>
    mesh_to_splat(const Mesh2SplatInput& input,
                  const core::Mesh2SplatOptions& options = {},
                  core::Mesh2SplatProgressCallback progress = nullptr);

    [[nodiscard]] std::expected<std::unique_ptr<core::SplatData>, std::string>
    mesh_to_splat(const core::MeshData& mesh,
                  const core::Mesh2SplatOptions& options = {},
                  core::Mesh2SplatProgressCallback progress = nullptr);

} // namespace lfs::rendering
