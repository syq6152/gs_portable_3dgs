/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "kernels/pruning_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lfs::training {

    namespace {
        void validate_mesh_row_tensor_for_save(
            const lfs::core::Tensor& tensor,
            const lfs::core::DataType expected_dtype,
            const size_t model_size,
            const std::string_view field_name,
            const std::string_view strategy_name) {
            if (tensor.ndim() != 1 || tensor.dtype() != expected_dtype || tensor.numel() < model_size) {
                throw std::runtime_error(
                    std::string(strategy_name) + " checkpoint has invalid " + std::string(field_name) +
                    "; expected a 1D tensor with at least " + std::to_string(model_size) + " rows");
            }
        }

        void write_optional_mesh_row_tensor(
            std::ostream& os,
            const lfs::core::Tensor& tensor,
            const lfs::core::DataType expected_dtype,
            const size_t model_size,
            const std::string_view field_name,
            const std::string_view strategy_name) {
            const uint8_t present = tensor.is_valid() ? 1 : 0;
            os.write(reinterpret_cast<const char*>(&present), sizeof(present));
            if (!present) {
                return;
            }

            validate_mesh_row_tensor_for_save(
                tensor, expected_dtype, model_size, field_name, strategy_name);
            os << tensor.slice(0, 0, model_size).contiguous();
        }

        lfs::core::Tensor read_optional_mesh_row_tensor(
            std::istream& is,
            const lfs::core::DataType expected_dtype,
            const size_t model_size,
            const std::string_view field_name,
            const std::string_view strategy_name) {
            uint8_t present = 0;
            lfs::core::serialization_detail::read_exact(
                is,
                &present,
                sizeof(present),
                std::string(strategy_name) + " " + std::string(field_name) + " presence flag");
            if (present > 1) {
                throw std::runtime_error(
                    std::string(strategy_name) + " checkpoint has an invalid " +
                    std::string(field_name) + " presence flag");
            }
            if (!present) {
                return {};
            }

            lfs::core::Tensor tensor;
            is >> tensor;
            if (tensor.ndim() != 1 || tensor.dtype() != expected_dtype || tensor.numel() != model_size) {
                throw std::runtime_error(
                    std::string(strategy_name) + " checkpoint has invalid " + std::string(field_name) +
                    "; expected shape [" + std::to_string(model_size) + "]");
            }
            return tensor;
        }

        [[nodiscard]] bool has_compatible_constraint_mesh(
            const lfs::core::Tensor& constraint_mesh_verts,
            const lfs::core::Tensor& constraint_mesh_indices,
            const lfs::core::Tensor& tri_edge_neighbors) {
            if (!constraint_mesh_verts.is_valid() || constraint_mesh_verts.ndim() != 2 ||
                constraint_mesh_verts.shape()[1] != 3 ||
                constraint_mesh_verts.dtype() != lfs::core::DataType::Float32 ||
                !constraint_mesh_indices.is_valid() || constraint_mesh_indices.ndim() != 2 ||
                constraint_mesh_indices.shape()[1] != 3 ||
                constraint_mesh_indices.dtype() != lfs::core::DataType::Int32 ||
                !tri_edge_neighbors.is_valid() || tri_edge_neighbors.ndim() != 2 ||
                tri_edge_neighbors.shape()[1] != 3 ||
                tri_edge_neighbors.dtype() != lfs::core::DataType::Int32) {
                return false;
            }
            return tri_edge_neighbors.shape()[0] == constraint_mesh_indices.shape()[0];
        }

        void validate_birth_tri_values(
            const lfs::core::Tensor& birth_tri,
            const lfs::core::Tensor& constraint_mesh_indices,
            const std::string_view strategy_name) {
            if (!birth_tri.is_valid()) {
                return;
            }

            const auto face_ids = birth_tri.to_vector_int();
            if (!constraint_mesh_indices.is_valid() || constraint_mesh_indices.ndim() != 2 ||
                constraint_mesh_indices.shape()[1] != 3 ||
                constraint_mesh_indices.dtype() != lfs::core::DataType::Int32) {
                throw std::runtime_error(
                    std::string(strategy_name) +
                    " checkpoint contains birth_tri state without a compatible constraint-mesh face table");
            }

            const size_t face_count = constraint_mesh_indices.is_valid() && constraint_mesh_indices.ndim() == 2
                                          ? constraint_mesh_indices.shape()[0]
                                          : 0;
            for (const int face_id : face_ids) {
                if (face_id < -1 || (face_id >= 0 && static_cast<size_t>(face_id) >= face_count)) {
                    throw std::runtime_error(
                        std::string(strategy_name) + " checkpoint contains an out-of-range birth_tri face id: " +
                        std::to_string(face_id));
                }
            }
        }

        void restore_birth_tri_tensor(
            lfs::core::Tensor& destination,
            const lfs::core::Tensor& prefix,
            const size_t capacity,
            const lfs::core::Device target_device) {
            if (!prefix.is_valid()) {
                destination = lfs::core::Tensor();
                return;
            }
            destination = lfs::core::Tensor::full(
                {capacity}, -1.0f, target_device, lfs::core::DataType::Int32);
            if (prefix.numel() > 0) {
                destination.slice(0, 0, prefix.numel()).copy_(prefix.to(target_device));
            }
        }

        void restore_mesh_init_mask_tensor(
            lfs::core::Tensor& destination,
            const lfs::core::Tensor& prefix,
            const size_t capacity,
            const lfs::core::Device target_device) {
            if (!prefix.is_valid()) {
                destination = lfs::core::Tensor();
                return;
            }
            destination = lfs::core::Tensor::zeros_bool({capacity}, target_device);
            if (prefix.numel() > 0) {
                destination.slice(0, 0, prefix.numel()).copy_(prefix.to(target_device));
            }
        }

        void restore_mesh_hole_fill_mask_tensor(
            lfs::core::Tensor& destination,
            const lfs::core::Tensor& prefix,
            const size_t capacity,
            const lfs::core::Device target_device) {
            // Hole-fill provenance is a total row property. A checkpoint that
            // predates the field (or explicitly omits it) means "not hole-fill"
            // for every active row and every unused capacity slot.
            destination = lfs::core::Tensor::zeros_bool({capacity}, target_device);
            if (prefix.is_valid() && prefix.numel() > 0) {
                destination.slice(0, 0, prefix.numel()).copy_(prefix.to(target_device));
            }
        }

        void validate_hole_fill_birth_tri_consistency(
            const lfs::core::Tensor& birth_tri,
            const lfs::core::Tensor& mesh_hole_fill_mask,
            const std::string_view strategy_name) {
            if (!mesh_hole_fill_mask.is_valid()) {
                return;
            }

            const auto hole_rows = mesh_hole_fill_mask.to_vector_bool();
            const bool has_hole_rows = std::any_of(
                hole_rows.begin(), hole_rows.end(), [](const bool value) { return value; });
            if (!has_hole_rows) {
                return;
            }
            if (!birth_tri.is_valid() || birth_tri.numel() != mesh_hole_fill_mask.numel()) {
                throw std::runtime_error(
                    std::string(strategy_name) +
                    " checkpoint contains mesh_hole_fill_mask rows without matching birth_tri state");
            }

            const auto face_ids = birth_tri.to_vector_int();
            for (size_t row = 0; row < hole_rows.size(); ++row) {
                if (hole_rows[row] && face_ids[row] != -1) {
                    throw std::runtime_error(
                        std::string(strategy_name) +
                        " checkpoint contains a hole-fill row whose birth_tri is not -1");
                }
            }
        }
    } // namespace

    namespace detail {
        lfs::core::Tensor near_zero_quaternion_mask(
            const lfs::core::Tensor& rotation_raw) {
            if (!rotation_raw.is_valid() ||
                rotation_raw.device() != lfs::core::Device::CUDA ||
                rotation_raw.dtype() != lfs::core::DataType::Float32 ||
                rotation_raw.ndim() != 2 || rotation_raw.shape()[1] != 4) {
                throw std::invalid_argument(
                    "near_zero_quaternion_mask expects a CUDA Float32 tensor with shape [N, 4]");
            }

            if (!rotation_raw.is_contiguous()) {
                throw std::invalid_argument(
                    "near_zero_quaternion_mask expects contiguous rotation rows");
            }

            auto mask = lfs::core::Tensor::empty(
                {rotation_raw.shape()[0]},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Bool);
            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
            mask.set_stream(stream);
            lfs::training::kernels::launch_near_zero_quaternion_mask(
                rotation_raw.ptr<float>(),
                mask.ptr<uint8_t>(),
                rotation_raw.shape()[0],
                stream);
            return mask;
        }

        lfs::core::Tensor build_igs_plus_prune_mask(
            const lfs::core::Tensor& opacity,
            const lfs::core::Tensor& rotation_raw,
            const lfs::core::Tensor& free_mask,
            const size_t model_size,
            const float prune_opacity) {
            if (!opacity.is_valid() || opacity.device() != lfs::core::Device::CUDA ||
                opacity.dtype() != lfs::core::DataType::Float32 || opacity.ndim() != 1 ||
                opacity.numel() != model_size) {
                throw std::invalid_argument(
                    "build_igs_plus_prune_mask expects CUDA Float32 opacity with shape [N]");
            }
            if (!rotation_raw.is_valid() || rotation_raw.ndim() != 2 ||
                rotation_raw.shape()[0] != model_size || rotation_raw.shape()[1] != 4) {
                throw std::invalid_argument(
                    "build_igs_plus_prune_mask expects rotation_raw with shape [N, 4]");
            }
            if (!free_mask.is_valid() || free_mask.device() != lfs::core::Device::CUDA ||
                free_mask.dtype() != lfs::core::DataType::Bool || free_mask.ndim() != 1 ||
                free_mask.numel() < model_size) {
                throw std::invalid_argument(
                    "build_igs_plus_prune_mask expects a CUDA Bool free_mask covering N physical rows");
            }

            auto prune_mask = (opacity < prune_opacity).logical_or(near_zero_quaternion_mask(rotation_raw));
            if (model_size == 0) {
                return prune_mask;
            }
            const auto active_rows = free_mask.slice(0, 0, model_size).logical_not();
            return prune_mask.logical_and(active_rows);
        }

        void serialize_mesh_row_state(
            std::ostream& os,
            const lfs::core::Tensor& birth_tri,
            const lfs::core::Tensor& mesh_init_mask,
            const lfs::core::Tensor& mesh_hole_fill_mask,
            const lfs::core::Tensor& constraint_mesh_indices,
            const size_t model_size,
            const std::string_view strategy_name) {
            if (birth_tri.is_valid()) {
                validate_mesh_row_tensor_for_save(
                    birth_tri, lfs::core::DataType::Int32, model_size, "birth_tri", strategy_name);
                validate_birth_tri_values(
                    birth_tri.slice(0, 0, model_size).cpu().contiguous(),
                    constraint_mesh_indices,
                    strategy_name);
            }
            if (mesh_hole_fill_mask.is_valid()) {
                validate_mesh_row_tensor_for_save(
                    mesh_hole_fill_mask,
                    lfs::core::DataType::Bool,
                    model_size,
                    "mesh_hole_fill_mask",
                    strategy_name);
                const auto active_birth_tri = birth_tri.is_valid()
                                                  ? birth_tri.slice(0, 0, model_size).cpu().contiguous()
                                                  : lfs::core::Tensor{};
                validate_hole_fill_birth_tri_consistency(
                    active_birth_tri,
                    mesh_hole_fill_mask.slice(0, 0, model_size).cpu().contiguous(),
                    strategy_name);
            }
            write_optional_mesh_row_tensor(
                os, birth_tri, lfs::core::DataType::Int32, model_size, "birth_tri", strategy_name);
            write_optional_mesh_row_tensor(
                os, mesh_init_mask, lfs::core::DataType::Bool, model_size, "mesh_init_mask", strategy_name);
            write_optional_mesh_row_tensor(
                os,
                mesh_hole_fill_mask,
                lfs::core::DataType::Bool,
                model_size,
                "mesh_hole_fill_mask",
                strategy_name);
        }

        void deserialize_mesh_row_state(
            std::istream& is,
            const size_t model_size,
            const size_t capacity,
            const lfs::core::Device target_device,
            const lfs::core::Tensor& constraint_mesh_verts,
            const lfs::core::Tensor& constraint_mesh_indices,
            const lfs::core::Tensor& tri_edge_neighbors,
            lfs::core::Tensor& birth_tri,
            lfs::core::Tensor& mesh_init_mask,
            lfs::core::Tensor& mesh_hole_fill_mask,
            const bool checkpoint_has_hole_fill_mask,
            const std::string_view strategy_name) {
            auto restored_birth_tri = read_optional_mesh_row_tensor(
                is, lfs::core::DataType::Int32, model_size, "birth_tri", strategy_name);
            auto restored_mesh_init_mask = read_optional_mesh_row_tensor(
                is, lfs::core::DataType::Bool, model_size, "mesh_init_mask", strategy_name);
            auto restored_mesh_hole_fill_mask = checkpoint_has_hole_fill_mask
                                                    ? read_optional_mesh_row_tensor(
                                                          is,
                                                          lfs::core::DataType::Bool,
                                                          model_size,
                                                          "mesh_hole_fill_mask",
                                                          strategy_name)
                                                    : lfs::core::Tensor{};

            if (restored_birth_tri.is_valid() &&
                !has_compatible_constraint_mesh(
                    constraint_mesh_verts, constraint_mesh_indices, tri_edge_neighbors)) {
                throw std::runtime_error(
                    std::string(strategy_name) +
                    " checkpoint contains birth_tri state, but the current scene does not provide a compatible constraint mesh and triangle adjacency");
            }
            validate_birth_tri_values(restored_birth_tri, constraint_mesh_indices, strategy_name);
            validate_hole_fill_birth_tri_consistency(
                restored_birth_tri, restored_mesh_hole_fill_mask, strategy_name);

            const size_t target_capacity = std::max(model_size, capacity);
            restore_birth_tri_tensor(
                birth_tri, restored_birth_tri, target_capacity, target_device);
            restore_mesh_init_mask_tensor(
                mesh_init_mask, restored_mesh_init_mask, target_capacity, target_device);
            restore_mesh_hole_fill_mask_tensor(
                mesh_hole_fill_mask,
                restored_mesh_hole_fill_mask,
                target_capacity,
                target_device);
        }

        void restore_legacy_mesh_row_state(
            const size_t model_size,
            const size_t capacity,
            const lfs::core::Device target_device,
            lfs::core::Tensor& birth_tri,
            lfs::core::Tensor& mesh_init_mask,
            lfs::core::Tensor& mesh_hole_fill_mask,
            const std::string_view strategy_name) {
            const size_t target_capacity = std::max(model_size, capacity);
            const bool has_mesh_state = birth_tri.is_valid() || mesh_init_mask.is_valid();

            if (birth_tri.is_valid() && birth_tri.ndim() == 1 &&
                birth_tri.dtype() == lfs::core::DataType::Int32) {
                const size_t copy_rows = std::min(model_size, static_cast<size_t>(birth_tri.numel()));
                auto prefix = birth_tri.slice(0, 0, copy_rows).contiguous();
                restore_birth_tri_tensor(birth_tri, prefix, target_capacity, target_device);
            }
            if (mesh_init_mask.is_valid() && mesh_init_mask.ndim() == 1 &&
                mesh_init_mask.dtype() == lfs::core::DataType::Bool) {
                const size_t copy_rows = std::min(model_size, static_cast<size_t>(mesh_init_mask.numel()));
                auto prefix = mesh_init_mask.slice(0, 0, copy_rows).contiguous();
                restore_mesh_init_mask_tensor(mesh_init_mask, prefix, target_capacity, target_device);
            }

            // Legacy checkpoints cannot carry immutable hole-fill provenance. Do
            // not infer it from the mutable face cache or fresh scene row state.
            restore_mesh_hole_fill_mask_tensor(
                mesh_hole_fill_mask, lfs::core::Tensor{}, target_capacity, target_device);

            if (has_mesh_state) {
                LOG_WARN("{}: loading a legacy strategy checkpoint without mesh row state; fresh initialization was retained where available, but densified checkpoint rows cannot recover their mesh face cache",
                         strategy_name);
            }
        }
    } // namespace detail

    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap) {
        // Tensors are already on GPU in the new framework (created with Device::CUDA by default)
        // Gradients are now owned by AdamOptimizer, not SplatData

        // Pre-allocate tensor capacity to avoid reallocations during MCMC operations
        // This eliminates memory fragmentation from varying tensor sizes
        if (max_cap > 0) {
            const size_t capacity = static_cast<size_t>(max_cap);
            LOG_INFO("Pre-allocating tensor capacity for {} Gaussians (parameters)", capacity);

            // Reserve capacity for all parameters (skip empty tensors like shN at sh_degree=0)
            auto reserve_if_valid = [capacity](lfs::core::Tensor& t) {
                if (t.is_valid() && t.numel() > 0)
                    t.reserve(capacity);
            };
            reserve_if_valid(splat_data.means());
            reserve_if_valid(splat_data.sh0());
            reserve_if_valid(splat_data.shN());
            reserve_if_valid(splat_data.scaling_raw());
            reserve_if_valid(splat_data.rotation_raw());
            reserve_if_valid(splat_data.opacity_raw());
        }
    }

    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params) {

        // Create Adam config with per-parameter learning rates
        AdamConfig config;
        config.lr = params.means_lr * splat_data.get_scene_scale(); // Default LR (for means)
        // Use double literals (not float!) to match legacy precision
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.eps = 1e-15;

        // Set per-parameter learning rates (matching legacy MCMC strategy)
        config.param_lrs["means"] = params.means_lr * splat_data.get_scene_scale();
        config.param_lrs["sh0"] = params.shs_lr;
        config.param_lrs["shN"] = params.shs_lr / 20.0f; // ShN uses reduced LR (1/20 of SH0)
        config.param_lrs["scaling"] = params.scaling_lr;
        config.param_lrs["rotation"] = params.rotation_lr;
        config.param_lrs["opacity"] = params.opacity_lr;

        // Pre-allocate optimizer state capacity to avoid reallocations during training
        // This dramatically reduces peak memory usage by avoiding double-buffering during growth
        if (params.max_cap > 0) {
            config.initial_capacity = static_cast<size_t>(params.max_cap);
            config.growth_factor = 1.5f; // Still allow growth beyond max_cap if needed
            LOG_INFO("AdamOptimizer: pre-allocating capacity for {} Gaussians (optimizer states)", config.initial_capacity);
        }

        LOG_DEBUG("Creating optimizer with per-parameter LRs:");
        LOG_DEBUG("  means: {:.2e}", config.param_lrs["means"]);
        LOG_DEBUG("  sh0: {:.2e}", config.param_lrs["sh0"]);
        LOG_DEBUG("  shN: {:.2e}", config.param_lrs["shN"]);
        LOG_DEBUG("  scaling: {:.2e}", config.param_lrs["scaling"]);
        LOG_DEBUG("  rotation: {:.2e}", config.param_lrs["rotation"]);
        LOG_DEBUG("  opacity: {:.2e}", config.param_lrs["opacity"]);

        auto optimizer = std::make_unique<AdamOptimizer>(splat_data, config);

        return optimizer;
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer) {

        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);

        return std::make_unique<ExponentialLR>(optimizer, gamma, std::vector<ParamType>{ParamType::Means});
    }

    float compute_mesh_scene_radius(const lfs::core::Tensor& mesh_vertices) {
        if (!mesh_vertices.is_valid() || mesh_vertices.ndim() != 2 || mesh_vertices.shape()[1] != 3 ||
            mesh_vertices.shape()[0] == 0) {
            return 1.0f;
        }

        auto verts = mesh_vertices.cpu().contiguous().to_vector();
        if (verts.size() < 3) {
            return 1.0f;
        }

        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        for (size_t i = 0; i + 2 < verts.size(); i += 3) {
            const float x = verts[i + 0];
            const float y = verts[i + 1];
            const float z = verts[i + 2];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            min_z = std::min(min_z, z);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            max_z = std::max(max_z, z);
        }

        const float dx = max_x - min_x;
        const float dy = max_y - min_y;
        const float dz = max_z - min_z;
        const float radius = 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz);
        return (radius > 1e-5f && std::isfinite(radius)) ? radius : 1.0f;
    }

    bool mesh_surface_soft_constraint_enabled(
        const lfs::core::param::OptimizationParameters& params) {
        return params.lambda_mesh_project > 0.0f ||
               params.lambda_mesh_outside_barrier > 0.0f ||
               params.lambda_mesh_scale_min > 0.0f ||
               params.lambda_mesh_scale_max > 0.0f ||
               params.lambda_mesh_normal > 0.0f;
    }

    bool mesh_surface_hard_projection_enabled(
        const lfs::core::param::OptimizationParameters& params) {
        // Hard projection is the legacy replacement for the mean distance loss
        // only. Scale and normal soft losses can still run at the same time.
        return params.mesh_surface_hard_projection_enabled &&
               params.lambda_mesh_project <= 0.0f &&
               params.lambda_mesh_outside_barrier <= 0.0f;
    }

    MeshTopologyDiagnostics diagnose_mesh_topology(
        const lfs::core::Tensor& mesh_vertices,
        const lfs::core::Tensor& mesh_indices) {
        MeshTopologyDiagnostics result;
        if (!mesh_vertices.is_valid() || !mesh_indices.is_valid() ||
            mesh_vertices.dtype() != lfs::core::DataType::Float32 ||
            mesh_indices.dtype() != lfs::core::DataType::Int32 ||
            mesh_vertices.ndim() != 2 || mesh_vertices.shape()[1] != 3 ||
            mesh_indices.ndim() != 2 || mesh_indices.shape()[1] != 3) {
            result.invalid_faces = mesh_indices.is_valid() && mesh_indices.ndim() > 0
                                       ? mesh_indices.shape()[0]
                                       : 1;
            return result;
        }

        auto vertices_cpu = mesh_vertices.cpu().contiguous();
        auto indices_cpu = mesh_indices.cpu().contiguous();
        const auto* vertices = vertices_cpu.ptr<float>();
        const auto* indices = indices_cpu.ptr<int32_t>();
        const size_t vertex_count = vertices_cpu.shape()[0];
        const size_t face_count = indices_cpu.shape()[0];

        struct EdgeUse {
            size_t face;
            bool low_to_high;
        };
        auto edge_key = [](const uint32_t low, const uint32_t high) {
            return (static_cast<uint64_t>(low) << 32U) | static_cast<uint64_t>(high);
        };

        std::unordered_map<uint64_t, std::vector<EdgeUse>> edges;
        edges.reserve(face_count * 3);
        std::vector<bool> valid_faces(face_count, false);
        std::vector<std::vector<size_t>> adjacency(face_count);

        for (size_t face = 0; face < face_count; ++face) {
            const int32_t a = indices[face * 3 + 0];
            const int32_t b = indices[face * 3 + 1];
            const int32_t c = indices[face * 3 + 2];
            if (a < 0 || b < 0 || c < 0 ||
                static_cast<size_t>(a) >= vertex_count ||
                static_cast<size_t>(b) >= vertex_count ||
                static_cast<size_t>(c) >= vertex_count ||
                a == b || b == c || c == a) {
                ++result.invalid_faces;
                continue;
            }
            const double ax = vertices[a * 3 + 0];
            const double ay = vertices[a * 3 + 1];
            const double az = vertices[a * 3 + 2];
            const double bx = vertices[b * 3 + 0];
            const double by = vertices[b * 3 + 1];
            const double bz = vertices[b * 3 + 2];
            const double cx = vertices[c * 3 + 0];
            const double cy = vertices[c * 3 + 1];
            const double cz = vertices[c * 3 + 2];
            const double abx = bx - ax;
            const double aby = by - ay;
            const double abz = bz - az;
            const double acx = cx - ax;
            const double acy = cy - ay;
            const double acz = cz - az;
            const double nx = aby * acz - abz * acy;
            const double ny = abz * acx - abx * acz;
            const double nz = abx * acy - aby * acx;
            if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(az) ||
                !std::isfinite(bx) || !std::isfinite(by) || !std::isfinite(bz) ||
                !std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz) ||
                nx * nx + ny * ny + nz * nz <= std::numeric_limits<double>::epsilon()) {
                ++result.invalid_faces;
                continue;
            }
            valid_faces[face] = true;
            const int32_t directed_edges[3][2] = {{a, b}, {b, c}, {c, a}};
            for (const auto& directed : directed_edges) {
                const uint32_t from = static_cast<uint32_t>(directed[0]);
                const uint32_t to = static_cast<uint32_t>(directed[1]);
                const uint32_t low = std::min(from, to);
                const uint32_t high = std::max(from, to);
                edges[edge_key(low, high)].push_back(EdgeUse{
                    .face = face,
                    .low_to_high = from == low});
            }
        }

        for (const auto& [key, uses] : edges) {
            (void)key;
            if (uses.size() == 1) {
                ++result.boundary_edges;
            } else if (uses.size() > 2) {
                ++result.non_manifold_edges;
            }
            if (uses.size() == 2 && uses[0].low_to_high == uses[1].low_to_high) {
                ++result.same_direction_shared_edges;
            }
            for (size_t i = 1; i < uses.size(); ++i) {
                adjacency[uses[0].face].push_back(uses[i].face);
                adjacency[uses[i].face].push_back(uses[0].face);
            }
        }

        std::vector<int32_t> component_ids(face_count, -1);
        for (size_t seed = 0; seed < face_count; ++seed) {
            if (!valid_faces[seed] || component_ids[seed] >= 0) {
                continue;
            }
            const int32_t component = static_cast<int32_t>(result.connected_components++);
            std::queue<size_t> pending;
            pending.push(seed);
            component_ids[seed] = component;
            while (!pending.empty()) {
                const size_t face = pending.front();
                pending.pop();
                for (const size_t neighbor : adjacency[face]) {
                    if (component_ids[neighbor] < 0) {
                        component_ids[neighbor] = component;
                        pending.push(neighbor);
                    }
                }
            }
        }

        std::vector<bool> closed(result.connected_components, true);
        std::vector<bool> consistent(result.connected_components, true);
        std::vector<double> signed_volume(result.connected_components, 0.0);
        for (const auto& [key, uses] : edges) {
            (void)key;
            if (uses.empty()) {
                continue;
            }
            const int32_t component = component_ids[uses[0].face];
            if (component < 0) {
                continue;
            }
            if (uses.size() != 2) {
                closed[component] = false;
            }
            if (uses.size() > 2 ||
                (uses.size() == 2 && uses[0].low_to_high == uses[1].low_to_high)) {
                consistent[component] = false;
            }
        }
        for (size_t face = 0; face < face_count; ++face) {
            const int32_t component = component_ids[face];
            if (component < 0) {
                continue;
            }
            const int32_t ia = indices[face * 3 + 0];
            const int32_t ib = indices[face * 3 + 1];
            const int32_t ic = indices[face * 3 + 2];
            const double ax = vertices[ia * 3 + 0];
            const double ay = vertices[ia * 3 + 1];
            const double az = vertices[ia * 3 + 2];
            const double bx = vertices[ib * 3 + 0];
            const double by = vertices[ib * 3 + 1];
            const double bz = vertices[ib * 3 + 2];
            const double cx = vertices[ic * 3 + 0];
            const double cy = vertices[ic * 3 + 1];
            const double cz = vertices[ic * 3 + 2];
            signed_volume[component] +=
                (ax * (by * cz - bz * cy) +
                 ay * (bz * cx - bx * cz) +
                 az * (bx * cy - by * cx)) /
                6.0;
        }
        for (size_t component = 0; component < result.connected_components; ++component) {
            if (closed[component] && consistent[component]) {
                ++result.closed_consistent_components;
                if (signed_volume[component] < 0.0) {
                    ++result.inward_components;
                }
            }
        }
        return result;
    }

    MeshTopologyDiagnostics log_mesh_constraint_topology_diagnostics(
        const lfs::core::Tensor& mesh_vertices,
        const lfs::core::Tensor& mesh_indices) {
        const auto diagnostics = diagnose_mesh_topology(mesh_vertices, mesh_indices);
        LOG_INFO("Mesh constraint topology: components={}, closed_consistent={}, inward_components={}, boundary_edges={}, non_manifold_edges={}, same_direction_shared_edges={}, invalid_faces={}",
                 diagnostics.connected_components,
                 diagnostics.closed_consistent_components,
                 diagnostics.inward_components,
                 diagnostics.boundary_edges,
                 diagnostics.non_manifold_edges,
                 diagnostics.same_direction_shared_edges,
                 diagnostics.invalid_faces);
        if (diagnostics.has_local_inside_outside_risks() || diagnostics.inward_components > 0) {
            LOG_WARN("Mesh constraint topology is open, non-manifold, invalid, inconsistently wound, or inward-facing. Training will continue using nearest-face local half-space inside/outside classification; results may be unreliable near holes, boundaries, or reversed normals.");
        }
        return diagnostics;
    }

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // CRITICAL: Ensure CUDA device is set for this thread
        // Some operations might spawn TBB threads, and those need CUDA context
        cudaSetDevice(0);

        // Map param index to ParamType
        auto index_to_param_type = [](size_t idx) -> ParamType {
            switch (idx) {
            case 0: return ParamType::Means;
            case 1: return ParamType::Sh0;
            case 2: return ParamType::ShN;
            case 3: return ParamType::Scaling;
            case 4: return ParamType::Rotation;
            case 5: return ParamType::Opacity;
            default:
                LOG_ERROR("Invalid parameter index: {}", idx);
                return ParamType::Means;
            }
        };

        // Get references to all parameters
        // (Gradients are now owned by AdamOptimizer, not SplatData)
        std::array<lfs::core::Tensor*, 6> params = {
            &splat_data.means(),
            &splat_data.sh0(),
            &splat_data.shN(),
            &splat_data.scaling_raw(),
            &splat_data.rotation_raw(),
            &splat_data.opacity_raw()};

        std::array<lfs::core::Tensor, 6> new_params;

        // First pass: Compute new parameters and update optimizer state
        for (auto i : param_idxs) {
            auto param = params[i];
            cudaError_t err_before = cudaGetLastError();
            if (err_before != cudaSuccess) {
                LOG_ERROR("CUDA error before param_fn: {}", cudaGetErrorString(err_before));
            }

            auto param_type = index_to_param_type(i);
            LOG_DEBUG("Calling param_fn for param {}", i);

            auto new_param = param_fn(i, *param);

            cudaError_t err_after = cudaGetLastError();
            if (err_after != cudaSuccess) {
                LOG_ERROR("CUDA error after param_fn({}) [param_type={}]: {}", i, static_cast<int>(param_type), cudaGetErrorString(err_after));
                throw std::runtime_error(std::string("CUDA error in param_fn (param ") + std::to_string(i) + "): " + cudaGetErrorString(err_after));
            }
            new_params[i] = new_param;

            // Modify state in-place (preserves capacity)
            AdamParamState* state = optimizer->get_state_mutable(param_type);
            if (state) {
                optimizer_fn(*state, new_param);
            }
        }

        // Second pass: Update parameters in SplatData
        // (Gradient updates are handled by the optimizer_fn callback which updates optimizer state)
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];
            } else if (i == 2) {
                splat_data.shN() = new_params[i];
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i];
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];
            }
        }
    }

} // namespace lfs::training
