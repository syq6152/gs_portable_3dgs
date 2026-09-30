/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /**
     * @brief Fused scale regularization: loss = weight * mean(exp(scaling_raw))
     *
     * Computes loss and gradients in a single pass using warp-level reductions.
     *
     * @param params Input parameters (scaling_raw) [N*3]
     * @param param_grads Output gradients (accumulated) [N*3]
     * @param loss_out Device buffer for scalar loss (1 element)
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (n+255)/256) elements)
     * @param n Number of elements
     * @param weight Regularization weight
     * @param stream CUDA stream
     */
    void launch_fused_scale_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream = nullptr);

    /**
     * @brief Fused opacity regularization: loss = weight * mean(sigmoid(opacity_raw))
     *
     * Computes loss and gradients in a single pass using warp-level reductions.
     *
     * @param params Input parameters (opacity_raw) [N]
     * @param param_grads Output gradients (accumulated) [N]
     * @param loss_out Device buffer for scalar loss (1 element)
     * @param temp_buffer Temporary buffer for partial sums (min(1024, (n+255)/256) elements)
     * @param n Number of elements
     * @param weight Regularization weight
     * @param stream CUDA stream
     */
    void launch_fused_opacity_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream = nullptr);

    void launch_mark_visible_mesh_primitives(
        const uint32_t* visible_primitive_indices,
        size_t n_visible_primitives,
        const int32_t* current_faces,
        int32_t* visible_mask,
        int32_t* visible_count,
        size_t n_primitives,
        cudaStream_t stream = nullptr);

    void launch_count_mesh_surface_primitives(
        const int32_t* current_faces,
        const int32_t* visible_mask,
        int32_t* count_out,
        size_t n_primitives,
        size_t n_faces,
        cudaStream_t stream = nullptr);

    void launch_compute_mesh_face_normals(
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        float* face_normals,
        size_t n_faces,
        cudaStream_t stream = nullptr);

    void launch_prepare_mesh_constraint_regions(
        const float* means,
        const int32_t* visible_mask,
        int32_t* current_faces,
        const int32_t* edge_neighbors,
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        const float* face_normals,
        float* surface_points,
        float* spatial_weights,
        float* soft_weight_sum,
        size_t n_primitives,
        size_t n_faces,
        int walk_steps,
        float inside_constraint_distance,
        float inside_constraint_fade_ratio,
        cudaStream_t stream = nullptr);

    void launch_mesh_surface_regularization(
        const float* means,
        const float* scaling_raw,
        const float* rotation_raw,
        float* grad_means,
        float* grad_scaling_raw,
        float* grad_rotation_raw,
        const int32_t* visible_mask,
        const int32_t* soft_count,
        const float* soft_weight_sum,
        const int32_t* tracked_count,
        int32_t* current_faces,
        const int32_t* edge_neighbors,
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        const float* face_normals,
        const float* surface_points,
        const float* spatial_weights,
        float* loss_out,
        float* temp_buffer,
        size_t n_primitives,
        size_t n_faces,
        int walk_steps,
        float lambda_project,
        float lambda_outside_barrier,
        float outside_distance_threshold,
        float lambda_scale_min,
        float lambda_scale_max,
        float rho,
        float lambda_normal,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
