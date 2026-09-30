/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    struct PreparedMesh {
        lfs::core::Tensor world_vertices_xyz;       // [V,3], CUDA float32
        lfs::core::Tensor world_vertex_normals_xyz; // [V,3], CUDA float32 (optional for depth-only)
        lfs::core::Tensor triangle_indices;         // [F,3], CUDA int32
        int vertex_count = 0;
        int face_count = 0;

        bool is_valid() const { return vertex_count > 0 && face_count > 0; }
    };

    std::expected<PreparedMesh, std::string> prepare_mesh_geometry(
        lfs::core::Scene& scene);

    std::expected<PreparedMesh, std::string> prepare_mesh_geometry(
        const lfs::core::MeshData& mesh);

    struct ProjectMaskMeshSource {
        const lfs::core::MeshData* mesh_override = nullptr;
        bool use_visible_scene_meshes = false;

        // external_pointcloud mode only. The mesh stays the coverage base; this cloud is
        // the evidence deciding which silhouette-interior holes in that coverage get
        // filled. Null in every other hole-fill mode, which keeps their masks unchanged.
        const lfs::core::PointCloud* point_cloud_override = nullptr;
        float point_cloud_spacing = 0.0f; // scene-space spacing the cloud was downsampled to

        [[nodiscard]] explicit operator bool() const noexcept {
            return mesh_override != nullptr || use_visible_scene_meshes;
        }
    };

    // Pure source selection used by project_mesh masking. A training-only
    // override is valid even when the scene has no visible mesh node.
    [[nodiscard]] ProjectMaskMeshSource select_project_mask_mesh_source(
        const lfs::core::Scene& scene);

    struct MeshSupervisionTargets {
        lfs::core::Tensor depth;       // [1,H,W], float32 CUDA
        lfs::core::Tensor normal;      // [3,H,W], float32 CUDA (optional)
        lfs::core::Tensor mask;        // [H,W], float32 CUDA (optional)
        lfs::core::Tensor triangle_id; // [H,W], int32 CUDA (optional, -1 == invalid)
        bool masked_camera = false;
    };

    // Select only the mesh outputs a caller consumes. Depth is always produced;
    // normals and triangle ids are opt-in so lightweight callers do not pay for
    // full-resolution normal buffers or transformed camera-space normals.
    struct MeshSupervisionOutputRequest {
        bool normal = true;
        bool triangle_id = false;

        [[nodiscard]] static constexpr MeshSupervisionOutputRequest depth_only() {
            return {.normal = false, .triangle_id = false};
        }

        [[nodiscard]] static constexpr MeshSupervisionOutputRequest depth_and_triangle_id() {
            return {.normal = false, .triangle_id = true};
        }

        [[nodiscard]] static constexpr MeshSupervisionOutputRequest depth_and_normal(
            const bool request_triangle_id = false) {
            return {.normal = true, .triangle_id = request_triangle_id};
        }
    };

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        lfs::core::Scene& scene,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionOutputRequest& output_request,
        bool apply_camera_mask = true);

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        const PreparedMesh& prepared_mesh,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionOutputRequest& output_request,
        bool apply_camera_mask = true);

    // Compatibility wrappers for existing callers. The historical bool API
    // always requests normals and optionally requests triangle ids.
    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        lfs::core::Scene& scene,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        bool apply_camera_mask = true,
        bool request_triangle_id = false);

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        const PreparedMesh& prepared_mesh,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        bool apply_camera_mask = true,
        bool request_triangle_id = false);

    struct MeshSupervisionRenderOptions {
        std::filesystem::path output_root;
        // Save debug preview PNGs for depth/normal alongside cache binaries.
        bool save_preview_png = false;
        // Optional subdirectory under output_root/mesh_supervision for per-level caches.
        std::string cache_tag;
    };

    struct MeshSupervisionRenderStats {
        size_t total_cameras = 0;
        size_t rendered_cameras = 0;
        size_t masked_cameras = 0;
        std::filesystem::path cache_dir;
    };

    struct MeshSupervisionCameraCacheEntry {
        uint32_t camera_uid = 0;
        std::string image_name;
        int width = 0;
        int height = 0;
        std::filesystem::path depth_path;
        std::filesystem::path normal_path;
        std::optional<std::filesystem::path> mask_path;
    };

    struct MeshSupervisionRenderResult {
        MeshSupervisionRenderStats stats;
        std::unordered_map<uint32_t, MeshSupervisionCameraCacheEntry> camera_cache_entries;
    };

    // Persist/restore float32 tensors used by mesh supervision cache.
    std::expected<void, std::string> write_mesh_supervision_tensor_file(
        const std::filesystem::path& path,
        const lfs::core::Tensor& tensor);

    std::expected<lfs::core::Tensor, std::string> decode_mesh_supervision_tensor_bytes(
        const std::vector<uint8_t>& bytes);

    std::filesystem::path mesh_supervision_cache_dir(
        const std::filesystem::path& output_root,
        const std::string& cache_tag = {});

    std::expected<MeshSupervisionRenderResult, std::string> render_mesh_supervision_cache(
        lfs::core::Scene& scene,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionRenderOptions& options);

} // namespace lfs::training
