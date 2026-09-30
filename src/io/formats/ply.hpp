/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "io/exporter.hpp"

namespace lfs::io {

    namespace ply_constants {
        constexpr float DEFAULT_LOG_SCALE = -5.0f;
        constexpr float SCENE_SCALE_FACTOR = 0.5f;
        constexpr float SH_C0 = 0.28209479177387814f;
    } // namespace ply_constants

    // Check if PLY contains triangle face elements
    bool ply_has_faces(const std::filesystem::path& filepath);

    // Check if PLY contains Gaussian splat properties (opacity, scaling, rotation)
    bool is_gaussian_splat_ply(const std::filesystem::path& filepath);

    // Load PLY as Gaussian splat (with opacity, scaling, rotation, SH)
    std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath);

    // Load PLY as simple point cloud (xyz + optional colors)
    std::expected<lfs::core::PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath);

    // Load PLY mesh with per-face texcoord list properties, splitting vertices at UV seams.
    // Returns nullopt if the PLY has no per-face texcoord list property.
    std::expected<lfs::core::MeshData, std::string> load_ply_mesh(const std::filesystem::path& filepath);

    // Resolve a PLY mesh texture from a header comment of the form:
    //   comment TextureFile relative_or_absolute_path
    std::expected<std::filesystem::path, std::string> resolve_ply_texture_file(const std::filesystem::path& filepath);

} // namespace lfs::io
