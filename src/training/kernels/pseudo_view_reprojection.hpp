/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    enum class PseudoViewReprojectionCounter : int {
        MeshValid = 0,
        Valid = 1,
        OcclusionFail = 2,
        ProjectionFail = 3,
        SourceMaskFail = 4,
        SourceFaceAngleFail = 5,
        Count = 6
    };

    enum class PseudoViewFrontFaceCounter : int {
        Mesh = 0,
        Front = 1,
        Back = 2,
        Count = 3
    };

    void launch_pseudo_view_front_face_mask(
        const int32_t* triangle_id,
        int width,
        int height,
        const float* face_centers,
        const float* face_normals,
        int face_count,
        float camera_x,
        float camera_y,
        float camera_z,
        float front_face_min_cos,
        float* mask_out,
        int32_t* counters,
        cudaStream_t stream = nullptr);

    void launch_pseudo_view_apply_valid_mask_rgb(
        float* rgb,
        const float* valid_mask,
        int width,
        int height,
        cudaStream_t stream = nullptr);

    void launch_pseudo_view_reproject_rgb(
        const float* pseudo_depth,
        const float* pseudo_mesh_mask,
        const int32_t* pseudo_triangle_id,
        int pseudo_width,
        int pseudo_height,
        float pseudo_fx,
        float pseudo_fy,
        float pseudo_cx,
        float pseudo_cy,
        const float* pseudo_c2w_4x4,
        const float* source_depth,
        const float* source_rgb,
        const float* source_mask,
        int source_width,
        int source_height,
        float source_fx,
        float source_fy,
        float source_cx,
        float source_cy,
        const float* source_w2c_4x4,
        const float* face_normals,
        int face_count,
        float source_camera_x,
        float source_camera_y,
        float source_camera_z,
        float source_face_angle_min_cos,
        float scene_radius,
        float rel_depth_epsilon,
        float source_mask_threshold,
        float* pseudo_rgb_out,
        float* valid_mask_out,
        float* mesh_mask_out,
        int32_t* counters,
        cudaStream_t stream = nullptr);

    void launch_pseudo_view_reproject_score(
        const float* pseudo_depth,
        const float* pseudo_mesh_mask,
        const int32_t* pseudo_triangle_id,
        int pseudo_width,
        int pseudo_height,
        float pseudo_fx,
        float pseudo_fy,
        float pseudo_cx,
        float pseudo_cy,
        const float* pseudo_c2w_4x4,
        const float* source_depth,
        const float* source_mask,
        int source_width,
        int source_height,
        float source_fx,
        float source_fy,
        float source_cx,
        float source_cy,
        const float* source_w2c_4x4,
        const float* face_normals,
        int face_count,
        float source_camera_x,
        float source_camera_y,
        float source_camera_z,
        float source_face_angle_min_cos,
        float scene_radius,
        float rel_depth_epsilon,
        float source_mask_threshold,
        int32_t* counters,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
