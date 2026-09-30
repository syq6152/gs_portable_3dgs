/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "regularization.hpp"
#include "lfs/kernels/regularization.cuh" // LibTorch-free CUDA kernels
#include <algorithm>
#include <cmath>
#include <format>

namespace lfs::training::losses {

    std::expected<lfs::core::Tensor, std::string> ScaleRegularization::forward(
        const lfs::core::Tensor& scaling_raw,
        lfs::core::Tensor& scaling_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Validate inputs
            if (scaling_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw must be on CUDA device");
            }
            if (scaling_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("scaling_raw_grad must be on CUDA device");
            }
            if (scaling_raw.shape() != scaling_raw_grad.shape()) {
                return std::unexpected("scaling_raw and scaling_raw_grad must have same shape");
            }

            size_t n = scaling_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_scale_regularization(
                scaling_raw.ptr<float>(),
                scaling_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in ScaleRegularization::forward: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> OpacityRegularization::forward(
        const lfs::core::Tensor& opacity_raw,
        lfs::core::Tensor& opacity_raw_grad,
        const Params& params) {
        try {
            if (params.weight <= 0.0f) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Validate inputs
            if (opacity_raw.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw must be on CUDA device");
            }
            if (opacity_raw_grad.device() != lfs::core::Device::CUDA) {
                return std::unexpected("opacity_raw_grad must be on CUDA device");
            }
            if (opacity_raw.shape() != opacity_raw_grad.shape()) {
                return std::unexpected("opacity_raw and opacity_raw_grad must have same shape");
            }

            size_t n = opacity_raw.numel();
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
            }

            // Allocate temporary buffers
            size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Launch LibTorch-free fused kernel with warp reductions
            lfs::training::kernels::launch_fused_opacity_regularization(
                opacity_raw.ptr<float>(),
                opacity_raw_grad.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                params.weight,
                nullptr);

            // NO .item<float>() - keep on GPU!
            return loss_tensor;

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in OpacityRegularization::forward: {}", e.what()));
        }
    }

    std::expected<void, std::string> MeshSurfaceRegularization::mark_visible_primitives(
        const uint32_t* visible_primitive_indices,
        size_t n_visible_primitives,
        size_t n_primitives,
        const lfs::core::Tensor& current_faces,
        lfs::core::Tensor& visible_mask,
        lfs::core::Tensor& visible_count) {
        try {
            if (n_visible_primitives == 0 || n_primitives == 0) {
                return {};
            }
            if (visible_primitive_indices == nullptr) {
                return std::unexpected("visible_primitive_indices must not be null");
            }
            if (current_faces.device() != lfs::core::Device::CUDA ||
                visible_mask.device() != lfs::core::Device::CUDA ||
                visible_count.device() != lfs::core::Device::CUDA) {
                return std::unexpected("mesh visible marking tensors must be on CUDA device");
            }
            if (current_faces.dtype() != lfs::core::DataType::Int32) {
                return std::unexpected("current_faces must be Int32");
            }
            if (visible_mask.dtype() != lfs::core::DataType::Int32) {
                return std::unexpected("visible_mask must be Int32");
            }
            if (visible_count.dtype() != lfs::core::DataType::Int32 ||
                visible_count.numel() < 1) {
                return std::unexpected("visible_count must be Int32[1]");
            }
            if (current_faces.numel() < n_primitives || visible_mask.numel() < n_primitives) {
                return std::unexpected("mesh visible marking tensors are smaller than n_primitives");
            }

            lfs::training::kernels::launch_mark_visible_mesh_primitives(
                visible_primitive_indices,
                n_visible_primitives,
                current_faces.ptr<int32_t>(),
                visible_mask.ptr<int32_t>(),
                visible_count.ptr<int32_t>(),
                n_primitives,
                nullptr);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in MeshSurfaceRegularization::mark_visible_primitives: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string> MeshSurfaceRegularization::forward(
        const lfs::core::Tensor& means,
        const lfs::core::Tensor& scaling_raw,
        const lfs::core::Tensor& rotation_raw,
        lfs::core::Tensor& means_grad,
        lfs::core::Tensor& scaling_raw_grad,
        lfs::core::Tensor& rotation_raw_grad,
        const lfs::core::Tensor* visible_mask,
        const lfs::core::Tensor* visible_count,
        lfs::core::Tensor& current_faces,
        const lfs::core::Tensor& edge_neighbors,
        const lfs::core::Tensor& mesh_vertices,
        const lfs::core::Tensor& mesh_indices,
        const lfs::core::Tensor* face_normals,
        const Params& params) {
        try {
            const auto finite_nonnegative = [](const float value) {
                return value >= 0.0f && std::isfinite(value);
            };
            if (!finite_nonnegative(params.lambda_project) ||
                !finite_nonnegative(params.lambda_outside_barrier) ||
                !finite_nonnegative(params.lambda_scale_min) ||
                !finite_nonnegative(params.lambda_scale_max) ||
                !finite_nonnegative(params.lambda_normal)) {
                return std::unexpected("Mesh surface regularization weights must be finite and non-negative");
            }
            if (params.lambda_outside_barrier > 0.0f &&
                (!(params.outside_distance_threshold > 0.0f) ||
                 !std::isfinite(params.outside_distance_threshold))) {
                return std::unexpected("Mesh outside barrier requires a positive finite distance threshold");
            }
            if (!finite_nonnegative(params.inside_constraint_distance)) {
                return std::unexpected("Mesh inside constraint distance must be finite and non-negative");
            }
            if (params.inside_constraint_fade_ratio < 0.0f ||
                params.inside_constraint_fade_ratio >= 1.0f ||
                !std::isfinite(params.inside_constraint_fade_ratio)) {
                return std::unexpected("Mesh inside constraint fade ratio must be finite and in [0, 1)");
            }
            const bool use_spatial_band = params.inside_constraint_distance > 0.0f;

            const bool enabled = params.lambda_project > 0.0f ||
                                 params.lambda_outside_barrier > 0.0f ||
                                 params.lambda_scale_min > 0.0f ||
                                 params.lambda_scale_max > 0.0f ||
                                 params.lambda_normal > 0.0f;
            if (!enabled) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            }

            if (means.device() != lfs::core::Device::CUDA ||
                scaling_raw.device() != lfs::core::Device::CUDA ||
                rotation_raw.device() != lfs::core::Device::CUDA ||
                means_grad.device() != lfs::core::Device::CUDA ||
                scaling_raw_grad.device() != lfs::core::Device::CUDA ||
                rotation_raw_grad.device() != lfs::core::Device::CUDA ||
                current_faces.device() != lfs::core::Device::CUDA ||
                edge_neighbors.device() != lfs::core::Device::CUDA ||
                mesh_vertices.device() != lfs::core::Device::CUDA ||
                mesh_indices.device() != lfs::core::Device::CUDA) {
                return std::unexpected("MeshSurfaceRegularization inputs must be on CUDA device");
            }
            const bool use_visible_mask = visible_mask != nullptr || visible_count != nullptr;
            if (use_visible_mask) {
                if (visible_mask == nullptr || visible_count == nullptr) {
                    return std::unexpected("Mesh surface visible mask and count must be provided together");
                }
                if (visible_mask->device() != lfs::core::Device::CUDA ||
                    visible_count->device() != lfs::core::Device::CUDA) {
                    return std::unexpected("Mesh surface visible tensors must be on CUDA device");
                }
            }

            const size_t n = static_cast<size_t>(means.shape()[0]);
            if (n == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            }
            if (means.ndim() != 2 || means.shape()[1] != 3 ||
                scaling_raw.ndim() != 2 || scaling_raw.shape()[0] < n || scaling_raw.shape()[1] != 3 ||
                rotation_raw.ndim() != 2 || rotation_raw.shape()[0] < n || rotation_raw.shape()[1] != 4) {
                return std::unexpected("Invalid Gaussian parameter shapes for mesh surface regularization");
            }
            if (means_grad.shape()[0] < n || scaling_raw_grad.shape()[0] < n || rotation_raw_grad.shape()[0] < n) {
                return std::unexpected("Mesh surface regularization gradient tensors are smaller than active Gaussian count");
            }
            if (current_faces.numel() < n) {
                return std::unexpected("Mesh surface regularization current face tensor is smaller than active Gaussian count");
            }
            if (use_visible_mask && visible_mask->numel() < n) {
                return std::unexpected("Mesh surface regularization visible mask is smaller than active Gaussian count");
            }
            if (use_visible_mask &&
                (visible_mask->dtype() != lfs::core::DataType::Int32 ||
                 visible_count->dtype() != lfs::core::DataType::Int32 ||
                 visible_count->numel() < 1)) {
                return std::unexpected("Mesh surface regularization visible tensors must be Int32");
            }
            if (current_faces.dtype() != lfs::core::DataType::Int32 ||
                edge_neighbors.dtype() != lfs::core::DataType::Int32 ||
                mesh_indices.dtype() != lfs::core::DataType::Int32) {
                return std::unexpected("Mesh surface regularization face/index tensors must be Int32");
            }
            if (mesh_vertices.ndim() != 2 || mesh_vertices.shape()[1] != 3 ||
                mesh_indices.ndim() != 2 || mesh_indices.shape()[1] != 3 ||
                edge_neighbors.ndim() != 2 || edge_neighbors.shape()[1] != 3 ||
                edge_neighbors.shape()[0] < mesh_indices.shape()[0]) {
                return std::unexpected("Invalid mesh tensor shapes for mesh surface regularization");
            }
            if (mesh_indices.shape()[0] == 0) {
                return lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            }
            const bool need_face_normals = params.lambda_normal > 0.0f ||
                                           params.lambda_outside_barrier > 0.0f ||
                                           use_spatial_band;
            if (need_face_normals) {
                if (face_normals == nullptr || !face_normals->is_valid()) {
                    return std::unexpected(
                        "Mesh normal, outside barrier, or inside constraint band requires precomputed mesh face normals");
                }
                if (face_normals->device() != lfs::core::Device::CUDA ||
                    face_normals->dtype() != lfs::core::DataType::Float32 ||
                    face_normals->ndim() != 2 ||
                    face_normals->shape()[0] < mesh_indices.shape()[0] ||
                    face_normals->shape()[1] != 3) {
                    return std::unexpected("Mesh face normals must be Float32 CUDA [F,3]");
                }
            }

            const size_t num_blocks = std::min((n + 255) / 256, size_t(1024));
            auto temp_buffer = lfs::core::Tensor::empty({num_blocks * 2}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            lfs::core::Tensor active_count;
            lfs::core::Tensor tracked_count;
            const int32_t* visible_mask_ptr = use_visible_mask ? visible_mask->ptr<int32_t>() : nullptr;
            const int32_t* count_ptr = nullptr;
            if (use_visible_mask) {
                count_ptr = visible_count->ptr<int32_t>();
            } else {
                active_count = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
                lfs::training::kernels::launch_count_mesh_surface_primitives(
                    current_faces.ptr<int32_t>(),
                    nullptr,
                    active_count.ptr<int32_t>(),
                    n,
                    mesh_indices.shape()[0],
                    nullptr);
                count_ptr = active_count.ptr<int32_t>();
            }

            const int32_t* tracked_count_ptr = count_ptr;
            if (params.lambda_outside_barrier > 0.0f && use_visible_mask) {
                tracked_count = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
                lfs::training::kernels::launch_count_mesh_surface_primitives(
                    current_faces.ptr<int32_t>(),
                    nullptr,
                    tracked_count.ptr<int32_t>(),
                    n,
                    mesh_indices.shape()[0],
                    nullptr);
                tracked_count_ptr = tracked_count.ptr<int32_t>();
            }

            lfs::core::Tensor surface_points;
            lfs::core::Tensor spatial_weights;
            lfs::core::Tensor soft_weight_sum;
            const float* surface_points_ptr = nullptr;
            const float* spatial_weights_ptr = nullptr;
            const float* soft_weight_sum_ptr = nullptr;
            if (use_spatial_band) {
                surface_points = lfs::core::Tensor::empty(
                    {n, 3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                spatial_weights = lfs::core::Tensor::zeros(
                    {n}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                soft_weight_sum = lfs::core::Tensor::zeros(
                    {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                lfs::training::kernels::launch_prepare_mesh_constraint_regions(
                    means.ptr<float>(),
                    visible_mask_ptr,
                    current_faces.ptr<int32_t>(),
                    edge_neighbors.ptr<int32_t>(),
                    mesh_vertices.ptr<float>(),
                    mesh_indices.ptr<int32_t>(),
                    face_normals->ptr<float>(),
                    surface_points.ptr<float>(),
                    spatial_weights.ptr<float>(),
                    soft_weight_sum.ptr<float>(),
                    n,
                    mesh_indices.shape()[0],
                    std::max(1, params.walk_steps),
                    params.inside_constraint_distance,
                    params.inside_constraint_fade_ratio,
                    nullptr);
                surface_points_ptr = surface_points.ptr<float>();
                spatial_weights_ptr = spatial_weights.ptr<float>();
                soft_weight_sum_ptr = soft_weight_sum.ptr<float>();
            }

            lfs::training::kernels::launch_mesh_surface_regularization(
                means.ptr<float>(),
                scaling_raw.ptr<float>(),
                rotation_raw.ptr<float>(),
                means_grad.ptr<float>(),
                scaling_raw_grad.ptr<float>(),
                rotation_raw_grad.ptr<float>(),
                visible_mask_ptr,
                count_ptr,
                soft_weight_sum_ptr,
                tracked_count_ptr,
                current_faces.ptr<int32_t>(),
                edge_neighbors.ptr<int32_t>(),
                mesh_vertices.ptr<float>(),
                mesh_indices.ptr<int32_t>(),
                need_face_normals ? face_normals->ptr<float>() : nullptr,
                surface_points_ptr,
                spatial_weights_ptr,
                loss_tensor.ptr<float>(),
                temp_buffer.ptr<float>(),
                n,
                mesh_indices.shape()[0],
                std::max(1, params.walk_steps),
                params.lambda_project,
                params.lambda_outside_barrier,
                params.outside_distance_threshold,
                params.lambda_scale_min,
                params.lambda_scale_max,
                params.rho,
                params.lambda_normal,
                nullptr);

            return loss_tensor;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error in MeshSurfaceRegularization::forward: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
