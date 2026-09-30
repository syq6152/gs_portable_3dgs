/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_init_rgb_color_transfer.hpp"

#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "mesh_supervision_renderer.hpp"
#include "kernels/mesh_init_rgb_color_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <format>
#include <glm/glm.hpp>
#include <exception>

namespace lfs::training {

    namespace {
        constexpr size_t SH_CHANNELS = 3;
        constexpr float SH_C0 = 0.28209479177387814f;
        constexpr float WHITE_SH0 = 0.5f / SH_C0;
        constexpr float REL_DEPTH_EPSILON = 1.0e-3f;
        constexpr float SOURCE_MASK_THRESHOLD = 0.5f;

        float mesh_scene_radius(const lfs::core::MeshData& mesh) {
            if (!mesh.vertices.is_valid() || mesh.vertices.numel() == 0) {
                return 1.0f;
            }
            auto vertices = mesh.vertices.to(lfs::core::Device::CPU)
                                .to(lfs::core::DataType::Float32)
                                .contiguous();
            if (vertices.ndim() != 2 || vertices.shape()[1] != 3) {
                return 1.0f;
            }
            const float* ptr = vertices.ptr<float>();
            glm::vec3 bbox_min(ptr[0], ptr[1], ptr[2]);
            glm::vec3 bbox_max = bbox_min;
            for (size_t i = 1; i < vertices.shape()[0]; ++i) {
                const glm::vec3 point(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
                bbox_min = glm::min(bbox_min, point);
                bbox_max = glm::max(bbox_max, point);
            }
            const float radius = 0.5f * glm::length(bbox_max - bbox_min);
            return std::isfinite(radius) && radius > 1.0e-6f ? radius : 1.0f;
        }

        bool valid_image_tensor(const lfs::core::Tensor& image) {
            return image.is_valid() && !image.is_empty() && image.ndim() == 3 &&
                   image.shape()[0] >= 3 && image.shape()[1] > 0 && image.shape()[2] > 0;
        }

        bool valid_mask_tensor(const lfs::core::Tensor& mask, const int width, const int height) {
            return mask.is_valid() && !mask.is_empty() && mask.ndim() == 2 &&
                   static_cast<int>(mask.shape()[1]) == width &&
                   static_cast<int>(mask.shape()[0]) == height;
        }
    } // namespace

    std::expected<MeshInitRgbColorTransferStats, std::string>
    transfer_mesh_init_rgb_colors(
        lfs::core::SplatData& model,
        const lfs::core::MeshData& normalized_mesh,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& train_cameras,
        const lfs::core::param::TrainingParameters& params) {
        if (model.size() == 0) {
            return std::unexpected("Mesh-init RGB color transfer received an empty Gaussian model");
        }
        if (train_cameras.empty()) {
            LOG_WARN("mesh_init RGB color transfer: no train cameras; retaining white Gaussian colors");
            model.sh0().fill_(WHITE_SH0);
            return MeshInitRgbColorTransferStats{
                .fallback_gaussians = model.size()};
        }

        auto prepared_result = prepare_mesh_geometry(normalized_mesh);
        if (!prepared_result) {
            return std::unexpected(std::format(
                "mesh_init RGB color transfer: mesh preparation failed: {}",
                prepared_result.error()));
        }
        const auto& prepared_mesh = *prepared_result;

        auto means = model.means().to(lfs::core::Device::CUDA)
                          .to(lfs::core::DataType::Float32)
                          .contiguous();
        const int point_count = static_cast<int>(model.size());
        auto rgb_sum = lfs::core::Tensor::zeros(
            {model.size(), size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto weight_sum = lfs::core::Tensor::zeros(
            {model.size()}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto counters = lfs::core::Tensor::zeros(
            {static_cast<size_t>(kernels::MeshInitRgbCounter::Count)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Int32);

        // RGB-view mode must have a deterministic fallback independent of whatever
        // converter-side vertex-color path happened to be available.
        model.sh0().fill_(WHITE_SH0);
        const float scene_radius = mesh_scene_radius(normalized_mesh);
        MeshInitRgbColorTransferStats stats;

        bool all_gaussians_resolved = false;
        for (const auto& camera_ptr : train_cameras) {
            if (!camera_ptr) {
                continue;
            }
            ++stats.cameras_considered;
            auto& camera = *camera_ptr;
            try {
                camera.load_image_size(params.dataset.resize_factor, params.dataset.max_width);
                const int width = camera.image_width();
                const int height = camera.image_height();
                if (width <= 0 || height <= 0) {
                    ++stats.cameras_failed;
                    LOG_WARN("mesh_init RGB color transfer: skipping '{}' with invalid image size {}x{}",
                             camera.image_name(), width, height);
                    continue;
                }

                auto depth_result = render_mesh_supervision_targets_for_camera(
                    prepared_mesh,
                    camera,
                    params,
                    MeshSupervisionOutputRequest::depth_only(),
                    false);
                if (!depth_result) {
                    ++stats.cameras_failed;
                    LOG_WARN("mesh_init RGB color transfer: depth render failed for '{}': {}",
                             camera.image_name(), depth_result.error());
                    continue;
                }
                auto depth = depth_result->depth.to(lfs::core::Device::CUDA)
                                 .to(lfs::core::DataType::Float32)
                                 .contiguous();
                if (depth.numel() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
                    ++stats.cameras_failed;
                    LOG_WARN("mesh_init RGB color transfer: depth size mismatch for '{}'",
                             camera.image_name());
                    continue;
                }

                auto rgb = camera.load_and_get_image(params.dataset.resize_factor, params.dataset.max_width)
                                .to(lfs::core::Device::CUDA)
                                .to(lfs::core::DataType::Float32)
                                .contiguous();
                if (!valid_image_tensor(rgb) || static_cast<int>(rgb.shape()[2]) != width ||
                    static_cast<int>(rgb.shape()[1]) != height) {
                    ++stats.cameras_failed;
                    LOG_WARN("mesh_init RGB color transfer: RGB size mismatch or invalid image for '{}'",
                             camera.image_name());
                    continue;
                }

                lfs::core::Tensor mask;
                const float* mask_ptr = nullptr;
                if (camera.has_mask()) {
                    mask = camera.load_and_get_mask(
                        params.dataset.resize_factor,
                        params.dataset.max_width,
                        params.optimization.invert_masks,
                        params.optimization.mask_threshold)
                               .to(lfs::core::Device::CUDA)
                               .to(lfs::core::DataType::Float32)
                               .contiguous();
                    if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                        mask = mask.squeeze(0).contiguous();
                    }
                    if (valid_mask_tensor(mask, width, height)) {
                        mask_ptr = mask.ptr<float>();
                    } else {
                        mask = lfs::core::Tensor();
                    }
                    camera.clear_cached_mask();
                }

                if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                    return std::unexpected(std::format(
                        "mesh_init RGB color transfer source preparation failed for '{}': {}",
                        camera.image_name(), cudaGetErrorString(error)));
                }
                const auto [fx, fy, cx, cy] = camera.get_intrinsics();
                kernels::launch_mesh_init_rgb_accumulate(
                    means.ptr<float>(),
                    point_count,
                    depth.ptr<float>(),
                    rgb.ptr<float>(),
                    mask_ptr,
                    width,
                    height,
                    fx,
                    fy,
                    cx,
                    cy,
                    camera.world_view_transform().ptr<float>(),
                    scene_radius,
                    REL_DEPTH_EPSILON,
                    SOURCE_MASK_THRESHOLD,
                    rgb_sum.ptr<float>(),
                    weight_sum.ptr<float>(),
                    counters.ptr<int32_t>());
                if (const auto error = cudaGetLastError(); error != cudaSuccess) {
                    return std::unexpected(std::format(
                        "mesh_init RGB color transfer kernel launch failed for '{}': {}",
                        camera.image_name(), cudaGetErrorString(error)));
                }
                if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                    return std::unexpected(std::format(
                        "mesh_init RGB color transfer kernel failed for '{}': {}",
                        camera.image_name(), cudaGetErrorString(error)));
                }
                ++stats.cameras_rendered;

                // The kernel's Valid counter is cumulative and now counts
                // first-hit assignments. Avoid loading/depth-rendering any
                // additional cameras once every Gaussian has a color.
                auto counters_cpu = counters.to(lfs::core::Device::CPU)
                                        .to(lfs::core::DataType::Int32)
                                        .contiguous();
                const auto resolved = static_cast<size_t>(std::max<int32_t>(
                    0, counters_cpu.ptr<int32_t>()[static_cast<int>(kernels::MeshInitRgbCounter::Valid)]));
                if (resolved >= model.size()) {
                    all_gaussians_resolved = true;
                    break;
                }
            } catch (const std::exception& error) {
                ++stats.cameras_failed;
                LOG_WARN("mesh_init RGB color transfer: skipping '{}': {}",
                         camera.image_name(), error.what());
            }
        }

        auto counters_cpu = counters.to(lfs::core::Device::CPU)
                                .to(lfs::core::DataType::Int32)
                                .contiguous();
        const int32_t* counter_ptr = counters_cpu.ptr<int32_t>();
        stats.projected_points = static_cast<size_t>(std::max<int32_t>(
            0, counter_ptr[static_cast<int>(kernels::MeshInitRgbCounter::Projected)]));
        stats.valid_observations = static_cast<size_t>(std::max<int32_t>(
            0, counter_ptr[static_cast<int>(kernels::MeshInitRgbCounter::Valid)]));
        stats.projection_failures = static_cast<size_t>(std::max<int32_t>(
            0, counter_ptr[static_cast<int>(kernels::MeshInitRgbCounter::ProjectionFail)]));
        stats.occlusion_failures = static_cast<size_t>(std::max<int32_t>(
            0, counter_ptr[static_cast<int>(kernels::MeshInitRgbCounter::OcclusionFail)]));
        stats.source_mask_failures = static_cast<size_t>(std::max<int32_t>(
            0, counter_ptr[static_cast<int>(kernels::MeshInitRgbCounter::SourceMaskFail)]));

        auto sums_cpu = rgb_sum.to(lfs::core::Device::CPU).contiguous();
        auto weights_cpu = weight_sum.to(lfs::core::Device::CPU).contiguous();
        auto sh0_cpu = model.sh0().to(lfs::core::Device::CPU)
                           .to(lfs::core::DataType::Float32)
                           .contiguous();
        if (sh0_cpu.numel() != model.size() * SH_CHANNELS) {
            return std::unexpected("mesh_init RGB color transfer: unexpected sh0 tensor shape");
        }

        const float* sums = sums_cpu.ptr<float>();
        const float* weights = weights_cpu.ptr<float>();
        float* sh0 = sh0_cpu.ptr<float>();
        for (size_t point = 0; point < model.size(); ++point) {
            const float weight = weights[point];
            if (std::isfinite(weight) && weight > 0.0f) {
                for (size_t channel = 0; channel < SH_CHANNELS; ++channel) {
                    const float rgb = std::clamp(sums[point * SH_CHANNELS + channel] / weight, 0.0f, 1.0f);
                    sh0[point * SH_CHANNELS + channel] = (rgb - 0.5f) / SH_C0;
                }
                ++stats.transferred_gaussians;
            } else {
                ++stats.fallback_gaussians;
            }
        }
        model.sh0() = sh0_cpu.to(model.sh0().device());

        LOG_INFO("mesh_init RGB color transfer: mode=first_valid, cameras={}/{}, projected={}, valid_observations={}, "
                 "transferred={}, white_fallback={}, projection_fail={}, occlusion_fail={}, mask_fail={}, early_stop={}",
                 stats.cameras_rendered,
                 stats.cameras_considered,
                 stats.projected_points,
                 stats.valid_observations,
                 stats.transferred_gaussians,
                 stats.fallback_gaussians,
                 stats.projection_failures,
                 stats.occlusion_failures,
                 stats.source_mask_failures,
                 all_gaussians_resolved);
        return stats;
    }

} // namespace lfs::training
