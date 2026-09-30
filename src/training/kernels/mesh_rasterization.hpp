/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_mesh_depth_key_init(
        int64_t* depth_keys,
        int pixel_count,
        cudaStream_t stream = nullptr);

    void launch_mesh_transform_vertices_normals(
        const float* world_vertices_xyz,
        const float* world_normals_xyz, // optional when cam_normals_xyz is null
        int vertex_count,
        const float* w2c_4x4,
        float* cam_vertices_xyz,
        float* cam_normals_xyz, // optional for depth-only rendering
        cudaStream_t stream = nullptr);

    void launch_mesh_rasterize_triangles(
        const float* cam_vertices_xyz,
        const float* cam_normals_xyz,
        const int32_t* triangle_indices,
        int face_count,
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float near_z,
        int64_t* depth_keys,
        cudaStream_t stream = nullptr);

    void launch_mesh_finalize_depth_normal(
        const int64_t* depth_keys,
        const float* cam_vertices_xyz,
        const float* cam_normals_xyz,
        const int32_t* triangle_indices,
        int face_count,
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float* depth_out,
        float* normal_out, // optional for depth-only rendering
        cudaStream_t stream = nullptr);

    void launch_mesh_apply_mask_depth_normal(
        float* depth,
        float* normal, // optional for depth-only rendering
        const float* mask,
        int width,
        int height,
        cudaStream_t stream = nullptr);

    // Extract per-pixel triangle id from the packed depth_keys produced by
    // launch_mesh_rasterize_triangles. Pixels with no triangle hit are written as -1.
    void launch_mesh_extract_tri_id(
        const int64_t* depth_keys,
        int face_count,
        int width,
        int height,
        int32_t* tri_id_out,
        cudaStream_t stream = nullptr);

    // Invalidate triangle id (-1) on pixels where mask <= 0 or non-finite, matching
    // the depth/normal masking semantics in launch_mesh_apply_mask_depth_normal.
    void launch_mesh_apply_mask_tri_id(
        int32_t* tri_id,
        const float* mask,
        int width,
        int height,
        cudaStream_t stream = nullptr);

    void launch_mesh_compute_vertex_normals(
        const float* vertices_xyz,
        const int32_t* triangle_indices,
        int vertex_count,
        int face_count,
        float* vertex_normals_xyz,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
