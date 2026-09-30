/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string_view>
#include <vector>

namespace lfs::training {

    // Initialize Gaussians (move to GPU, pre-allocate capacity, etc.)
    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap = 0);

    namespace detail {
        // IGS+ pruning helpers. These intentionally operate only on Gaussian
        // row state and do not introduce mesh-surface strategy state.
        lfs::core::Tensor near_zero_quaternion_mask(
            const lfs::core::Tensor& rotation_raw);

        lfs::core::Tensor build_igs_plus_prune_mask(
            const lfs::core::Tensor& opacity,
            const lfs::core::Tensor& rotation_raw,
            const lfs::core::Tensor& free_mask,
            size_t model_size,
            float prune_opacity);

        void serialize_mesh_row_state(
            std::ostream& os,
            const lfs::core::Tensor& birth_tri,
            const lfs::core::Tensor& mesh_init_mask,
            const lfs::core::Tensor& mesh_hole_fill_mask,
            const lfs::core::Tensor& constraint_mesh_indices,
            size_t model_size,
            std::string_view strategy_name);

        void deserialize_mesh_row_state(
            std::istream& is,
            size_t model_size,
            size_t capacity,
            lfs::core::Device target_device,
            const lfs::core::Tensor& constraint_mesh_verts,
            const lfs::core::Tensor& constraint_mesh_indices,
            const lfs::core::Tensor& tri_edge_neighbors,
            lfs::core::Tensor& birth_tri,
            lfs::core::Tensor& mesh_init_mask,
            lfs::core::Tensor& mesh_hole_fill_mask,
            bool checkpoint_has_hole_fill_mask,
            std::string_view strategy_name);

        void restore_legacy_mesh_row_state(
            size_t model_size,
            size_t capacity,
            lfs::core::Device target_device,
            lfs::core::Tensor& birth_tri,
            lfs::core::Tensor& mesh_init_mask,
            lfs::core::Tensor& mesh_hole_fill_mask,
            std::string_view strategy_name);
    } // namespace detail

    // Create optimizer for splat data
    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params);

    // Create exponential LR scheduler
    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer);

    // Mesh scene scale follows pseudo-view precompute semantics:
    // half the diagonal length of the mesh vertex AABB, with 1.0 fallback.
    float compute_mesh_scene_radius(const lfs::core::Tensor& mesh_vertices);

    bool mesh_surface_soft_constraint_enabled(
        const lfs::core::param::OptimizationParameters& params);

    bool mesh_surface_hard_projection_enabled(
        const lfs::core::param::OptimizationParameters& params);

    struct MeshTopologyDiagnostics {
        size_t boundary_edges = 0;
        size_t non_manifold_edges = 0;
        size_t same_direction_shared_edges = 0;
        size_t connected_components = 0;
        size_t closed_consistent_components = 0;
        size_t inward_components = 0;
        size_t invalid_faces = 0;

        [[nodiscard]] bool has_local_inside_outside_risks() const {
            return boundary_edges > 0 || non_manifold_edges > 0 ||
                   same_direction_shared_edges > 0 || invalid_faces > 0;
        }

        [[nodiscard]] bool is_closed_manifold_outward() const {
            return connected_components > 0 &&
                   closed_consistent_components == connected_components &&
                   inward_components == 0 &&
                   !has_local_inside_outside_risks();
        }
    };

    MeshTopologyDiagnostics diagnose_mesh_topology(
        const lfs::core::Tensor& mesh_vertices,
        const lfs::core::Tensor& mesh_indices);

    MeshTopologyDiagnostics log_mesh_constraint_topology_diagnostics(
        const lfs::core::Tensor& mesh_vertices,
        const lfs::core::Tensor& mesh_indices);

    // Function types for parameter and optimizer state updates
    using ParamUpdateFn = std::function<lfs::core::Tensor(const int, const lfs::core::Tensor&)>;
    using OptimizerUpdateFn = std::function<void(
        AdamParamState& state,
        const lfs::core::Tensor& new_param)>;

    // Update parameter with optimizer state synchronization
    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs = {0, 1, 2, 3, 4, 5});

} // namespace lfs::training
