/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/colmap_database.hpp"
#include "preprocessing/image_pipeline.hpp"

namespace lfs::preprocess {
    struct CpuMesh {
        std::vector<std::array<double, 3>> vertices;
        std::vector<std::array<uint32_t, 3>> triangles;
    };
    // Frozen scanner meshes already use meters. Scaling is explicit; never infer units.
    std::expected<CpuMesh, Error> read_mesh_geometry(const std::filesystem::path&, double coordinate_scale = 1.0,
                                                     const ExecutionContext& = {});
    // Legacy silhouette: convex hull of visible, in-bounds projected vertices, then erosion.
    // This deliberately is not a triangle/z-buffer silhouette (which changes the Python output).
    std::expected<CpuImage, Error> project_mesh_mask(const CpuMesh&, const ColmapCamera&, const ColmapImage&,
                                                     int erosion_pixels = 0, const ExecutionContext& = {});
    // Frozen debug visualization: visible vertex pixels only, RGB green at alpha 160.
    // In-bounds clipping precedes ties-to-even rounding; duplicate hits blend once.
    // The RGB input must match the camera size. This never modifies training RGB/masks.
    std::expected<CpuImage, Error> project_mesh_overlay(const CpuMesh&, const CpuImage&, const ColmapCamera&,
                                                        const ColmapImage&, const ExecutionContext& = {});
    struct MeshTriangulationOptions {
        double match_3d_threshold = 0.0004;
        std::size_t min_matches_per_pair = 15;
        bool mutual_nearest = true;
        bool mean_colors = true;
    };
    struct MeshPairMatches {
        uint32_t first = 0, second = 0;
        Matches matches;
    };
    struct MeshTriangulationResult {
        ColmapModel model;
        std::vector<MeshPairMatches> pairs;
        std::size_t hit_count = 0;
        std::vector<std::string> dropped_names;
    };
    // scan_model IDs must have been synchronized with the private database. Returns matches
    // for the caller to write transactionally to matches and two_view_geometries (config=2).
    std::expected<MeshTriangulationResult, Error> triangulate_mesh(
        const CpuMesh&, const ColmapModel& scan_model, const std::map<uint32_t, Keypoints>&,
        const std::filesystem::path& image_root, const MeshTriangulationOptions& = {}, const ExecutionContext& = {});
} // namespace lfs::preprocess
