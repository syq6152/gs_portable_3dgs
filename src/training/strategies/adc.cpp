/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adc.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "lfs/kernels/regularization.cuh"
#include "optimizer/render_output.hpp"
#include "strategy_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

namespace lfs::training {

    namespace {
        // Returns true if shape has any zero dimension (e.g., ShN at sh-degree 0)
        [[nodiscard]] inline bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0)
                    return true;
            }
            return false;
        }

        // Returns true if shN tensor has non-zero coefficients
        [[nodiscard]] inline bool has_shN_coefficients(const lfs::core::Tensor& shN) {
            return shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > 0;
        }

        [[nodiscard]] size_t deleted_mask_capacity(
            const lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            return free_mask.is_valid() ? static_cast<size_t>(free_mask.numel())
                                        : static_cast<size_t>(splat_data.size());
        }

        void sync_deleted_mask_from_free_mask(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);

            if (!free_mask.is_valid()) {
                splat_data.deleted() = lfs::core::Tensor::zeros_bool(
                    {current_size}, splat_data.means().device());
            } else {
                splat_data.deleted() = free_mask.slice(0, 0, current_size).clone();
            }
            splat_data.deleted().reserve(desired_capacity);
        }

        void ensure_deleted_mask_size(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const auto& deleted = splat_data.deleted();
            if (!deleted.is_valid() || deleted.ndim() != 1 ||
                deleted.numel() != current_size ||
                deleted.dtype() != lfs::core::DataType::Bool) {
                sync_deleted_mask_from_free_mask(splat_data, free_mask);
                return;
            }
            splat_data.deleted().reserve(deleted_mask_capacity(splat_data, free_mask));
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool(
                                    {static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool(
                                    {static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
        }

        void append_live_deleted_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const size_t n_rows) {
            if (n_rows == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            splat_data.deleted().append_zeros(n_rows);
        }

        void ensure_bool_row_state_capacity(
            lfs::core::Tensor& state,
            const size_t required,
            const lfs::core::Device device,
            const bool create_if_missing) {
            if (!state.is_valid()) {
                if (create_if_missing) {
                    state = lfs::core::Tensor::zeros_bool({required}, device);
                }
                return;
            }
            if (state.ndim() != 1 || state.dtype() != lfs::core::DataType::Bool) {
                throw std::runtime_error("ADC bool row state has an incompatible schema");
            }
            if (state.numel() >= required) {
                return;
            }

            auto expanded = lfs::core::Tensor::zeros_bool({required}, device);
            expanded.slice(0, 0, state.numel()).copy_(state.to(device));
            state = std::move(expanded);
        }

        void ensure_birth_tri_capacity(
            lfs::core::Tensor& state,
            const size_t required,
            const lfs::core::Device device) {
            if (!state.is_valid()) {
                return;
            }
            if (state.ndim() != 1 || state.dtype() != lfs::core::DataType::Int32) {
                throw std::runtime_error("ADC birth_tri row state has an incompatible schema");
            }
            if (state.numel() >= required) {
                return;
            }

            auto expanded = lfs::core::Tensor::full(
                {required}, -1.0f, device, lfs::core::DataType::Int32);
            expanded.slice(0, 0, state.numel()).copy_(state.to(device));
            state = std::move(expanded);
        }
    } // anonymous namespace

    ADC::ADC(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void ADC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        initialize_gaussians(*_splat_data, _params->max_cap);

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data->_densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data->size())},
            _splat_data->means().device());

        // Initialize free mask: all slots are active (not free)
        const size_t init_n = static_cast<size_t>(_splat_data->size());
        const size_t configured_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                                : init_n;
        const size_t capacity = std::max(configured_capacity, init_n);
        _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        if (_splat_data->has_mesh_init_mask()) {
            _mesh_init_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
            auto src = _splat_data->mesh_init_mask().slice(0, 0, init_n);
            _mesh_init_mask.slice(0, 0, init_n).copy_(src);
            LOG_INFO("ADC: Mesh-init marker preserved for {} Gaussians", init_n);
        }

        _mesh_hole_fill_mask = lfs::core::Tensor::zeros_bool(
            {capacity}, _splat_data->means().device());
        if (_splat_data->has_mesh_hole_fill_mask()) {
            const auto& source = _splat_data->mesh_hole_fill_mask();
            if (source.ndim() != 1 || source.dtype() != lfs::core::DataType::Bool ||
                source.numel() != init_n) {
                throw std::runtime_error(
                    "ADC: mesh_hole_fill_mask must be a Bool tensor matching the initial model size");
            }
            _mesh_hole_fill_mask.slice(0, 0, init_n).copy_(source);
            LOG_INFO("ADC: Hole-fill provenance preserved for {} Gaussians", init_n);
        }

        if (_splat_data->has_constraint_mesh()) {
            _constraint_mesh_verts = _splat_data->constraint_mesh_verts();
            _constraint_mesh_indices = _splat_data->constraint_mesh_indices();
            _mesh_scene_radius = compute_mesh_scene_radius(_constraint_mesh_verts);
            _surface_walk_steps = std::max(1, _params->mesh_surface_walk_steps);

            _birth_tri = lfs::core::Tensor::full(
                {capacity}, -1.0f, _splat_data->means().device(), lfs::core::DataType::Int32);
            auto src_bt = _splat_data->birth_tri().slice(0, 0, init_n);
            _birth_tri.slice(0, 0, init_n).copy_(src_bt);

            LOG_INFO("ADC: Mesh-surface state enabled: {} verts, {} faces, {} tracked Gaussians with {} max edge-neighbor steps",
                     _constraint_mesh_verts.shape()[0],
                     _constraint_mesh_indices.shape()[0],
                     init_n,
                     _surface_walk_steps);

            if (_splat_data->has_tri_edge_neighbors()) {
                _tri_edge_neighbors = _splat_data->tri_edge_neighbors();
            } else {
                LOG_WARN("ADC: mesh-surface walk requested but tri_edge_neighbors are missing; projection is disabled for this SplatData.");
            }
        }
    }

    bool ADC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void ADC::remove_gaussians(const lfs::core::Tensor& mask) {
        if (!mask.is_valid() || mask.numel() == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();
        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void ADC::duplicate(const lfs::core::Tensor& is_duplicated) {
        const lfs::core::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);
        const int64_t num_duplicated = sampled_idxs.shape()[0];

        if (num_duplicated == 0) {
            return; // Nothing to duplicate
        }

        // Try to fill free slots first (in-place with index_put_)
        auto [filled_indices, remaining] = fill_free_slots(sampled_idxs, num_duplicated);
        const int64_t num_filled = num_duplicated - remaining;

        LOG_DEBUG("duplicate(): {} total, {} filled free slots, {} to append", num_duplicated, num_filled, remaining);

        // Append remaining Gaussians in-place
        if (remaining > 0) {
            const size_t old_size = static_cast<size_t>(_splat_data->size());
            const auto append_src_indices = sampled_idxs.slice(0, num_filled, num_duplicated);
            const size_t n_new = static_cast<size_t>(remaining);
            const size_t required_rows = old_size + n_new;

            ensure_bool_row_state_capacity(
                _free_mask, required_rows, _splat_data->means().device(), true);
            ensure_bool_row_state_capacity(
                _mesh_init_mask, required_rows, _splat_data->means().device(), false);
            ensure_bool_row_state_capacity(
                _mesh_hole_fill_mask, required_rows, _splat_data->means().device(), true);
            ensure_birth_tri_capacity(
                _birth_tri, required_rows, _splat_data->means().device());

            // In-place append
            append_live_deleted_rows(*_splat_data, _free_mask, n_new);
            _splat_data->means().append_gather(append_src_indices);
            _splat_data->rotation_raw().append_gather(append_src_indices);
            _splat_data->scaling_raw().append_gather(append_src_indices);
            _splat_data->sh0().append_gather(append_src_indices);
            _splat_data->opacity_raw().append_gather(append_src_indices);

            auto& shN = _splat_data->shN();
            if (has_shN_coefficients(shN)) {
                shN.append_gather(append_src_indices);
            }

            // Initialize optimizer states with zeros
            _optimizer->extend_state_for_new_params(ParamType::Means, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Rotation, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Scaling, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Sh0, n_new);
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Opacity, n_new);

            if (_birth_tri.is_valid()) {
                auto gathered = _birth_tri.slice(0, 0, old_size)
                                    .index_select(0, append_src_indices)
                                    .contiguous();
                _birth_tri.slice(0, old_size, old_size + n_new).copy_(gathered);
            }

            {
                auto gathered = _mesh_hole_fill_mask.slice(0, 0, old_size)
                                    .index_select(0, append_src_indices)
                                    .contiguous();
                _mesh_hole_fill_mask.slice(0, old_size, old_size + n_new).copy_(gathered);
            }

            if (_surface_prev_means.is_valid() && _surface_prev_means.numel() >= (old_size + n_new) * 3) {
                auto gathered_prev = _surface_prev_means.slice(0, 0, old_size)
                                         .index_select(0, append_src_indices)
                                         .contiguous();
                _surface_prev_means.slice(0, old_size, old_size + n_new).copy_(gathered_prev);
            }
        }
    }

    void ADC::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor split_idxs = is_split.nonzero().squeeze(-1);
        const int64_t num_split = split_idxs.shape()[0];

        if (num_split == 0) {
            return; // Nothing to split
        }

        LOG_DEBUG("split(): {} Gaussians to split", num_split);

        // Get SH dimensions
        const bool has_shN = _splat_data->shN().is_valid();
        int shN_dim = 0;
        if (has_shN) {
            const auto& shN_shape = _splat_data->shN().shape();
            if (shN_shape.rank() == 2) {
                shN_dim = shN_shape[1];
            } else if (shN_shape.rank() == 3) {
                shN_dim = shN_shape[1] * shN_shape[2];
            }
        }

        const auto device = _splat_data->means().device();

        // Generate random noise [2, num_split, 3]
        const lfs::core::Tensor random_noise = lfs::core::Tensor::randn(
            {2, static_cast<size_t>(num_split), 3}, device);

        // Allocate temporary tensors for second split results [num_split, ...]
        auto second_positions = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        auto second_rotations = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 4}, device);
        auto second_scales = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        auto second_sh0 = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        lfs::core::Tensor second_shN;
        if (has_shN) {
            second_shN = lfs::core::Tensor::empty({static_cast<size_t>(num_split), static_cast<size_t>(shN_dim)}, device);
        }
        auto second_opacities = lfs::core::Tensor::empty({static_cast<size_t>(num_split)}, device);

        // Skip shN pointers when shN_dim=0 (sh-degree 0)
        const bool use_shN = has_shN && shN_dim > 0;

        // First result modifies in-place, second goes to temporaries
        kernels::launch_split_gaussians_inplace(
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            use_shN ? _splat_data->shN().ptr<float>() : nullptr,
            _splat_data->opacity_raw().ptr<float>(),
            second_positions.ptr<float>(),
            second_rotations.ptr<float>(),
            second_scales.ptr<float>(),
            second_sh0.ptr<float>(),
            use_shN ? second_shN.ptr<float>() : nullptr,
            second_opacities.ptr<float>(),
            split_idxs.ptr<int64_t>(),
            random_noise.ptr<float>(),
            static_cast<int>(num_split),
            shN_dim,
            _params->revised_opacity,
            nullptr);

        // Reset optimizer states for split indices
        auto reset_optimizer_state_at_indices = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(num_split)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(split_idxs, zeros);
            state->exp_avg_sq.index_put_(split_idxs, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(split_idxs, zeros);
            }
        };

        reset_optimizer_state_at_indices(ParamType::Means);
        reset_optimizer_state_at_indices(ParamType::Rotation);
        reset_optimizer_state_at_indices(ParamType::Scaling);
        reset_optimizer_state_at_indices(ParamType::Sh0);
        reset_optimizer_state_at_indices(ParamType::ShN);
        reset_optimizer_state_at_indices(ParamType::Opacity);

        // Now place second split results: fill free slots first, then append
        // Try to fill free slots first
        const size_t source_region_size = static_cast<size_t>(_splat_data->size());
        auto [filled_indices, remaining] = fill_free_slots_with_data(
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities, num_split);

        const int64_t num_filled = num_split - remaining;

        if (_birth_tri.is_valid() && num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = split_idxs.slice(0, 0, num_filled);
            auto gathered = _birth_tri.slice(0, 0, source_region_size)
                                .index_select(0, filled_sources)
                                .contiguous();
            _birth_tri.index_put_(filled_indices, gathered);
        }

        if (num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = split_idxs.slice(0, 0, num_filled);
            auto gathered = _mesh_hole_fill_mask.slice(0, 0, source_region_size)
                                .index_select(0, filled_sources)
                                .contiguous();
            _mesh_hole_fill_mask.index_put_(filled_indices, gathered);
        }

        if (_surface_prev_means.is_valid() && num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = split_idxs.slice(0, 0, num_filled);
            auto gathered_prev = _surface_prev_means.slice(0, 0, source_region_size)
                                     .index_select(0, filled_sources)
                                     .contiguous();
            _surface_prev_means.index_put_(filled_indices, gathered_prev);
        }

        // Append remaining second results
        if (remaining > 0) {
            const size_t old_size = static_cast<size_t>(_splat_data->size());
            const size_t n_remaining = static_cast<size_t>(remaining);
            const size_t required_rows = old_size + n_remaining;

            ensure_bool_row_state_capacity(
                _free_mask, required_rows, _splat_data->means().device(), true);
            ensure_bool_row_state_capacity(
                _mesh_init_mask, required_rows, _splat_data->means().device(), false);
            ensure_bool_row_state_capacity(
                _mesh_hole_fill_mask, required_rows, _splat_data->means().device(), true);
            ensure_birth_tri_capacity(
                _birth_tri, required_rows, _splat_data->means().device());

            // Get the remaining data
            const auto append_positions = second_positions.slice(0, num_filled, num_split);
            const auto append_rotations = second_rotations.slice(0, num_filled, num_split);
            const auto append_scales = second_scales.slice(0, num_filled, num_split);
            const auto append_sh0_flat = second_sh0.slice(0, num_filled, num_split);
            const auto append_opacities = second_opacities.slice(0, num_filled, num_split);

            // Create indices for new rows
            std::vector<int> new_indices_vec(n_remaining);
            for (size_t i = 0; i < n_remaining; ++i) {
                new_indices_vec[i] = static_cast<int>(old_size + i);
            }
            const auto new_indices = lfs::core::Tensor::from_vector(
                new_indices_vec, lfs::core::TensorShape({n_remaining}), device);

            // Extend and write data
            append_live_deleted_rows(*_splat_data, _free_mask, n_remaining);
            _splat_data->means().append_zeros(n_remaining);
            _splat_data->means().index_put_(new_indices, append_positions);

            _splat_data->rotation_raw().append_zeros(n_remaining);
            _splat_data->rotation_raw().index_put_(new_indices, append_rotations);

            _splat_data->scaling_raw().append_zeros(n_remaining);
            _splat_data->scaling_raw().index_put_(new_indices, append_scales);

            const auto append_sh0_reshaped = append_sh0_flat.reshape(
                lfs::core::TensorShape({n_remaining, 1, 3}));
            _splat_data->sh0().append_zeros(n_remaining);
            _splat_data->sh0().index_put_(new_indices, append_sh0_reshaped);

            _splat_data->opacity_raw().append_zeros(n_remaining);
            _splat_data->opacity_raw().index_put_(new_indices, append_opacities);

            if (use_shN) {
                auto append_shN = second_shN.slice(0, num_filled, num_split);
                const auto& shN_shape = _splat_data->shN().shape();
                if (shN_shape.rank() == 3) {
                    append_shN = append_shN.reshape(
                        lfs::core::TensorShape({n_remaining, shN_shape[1], shN_shape[2]}));
                }
                _splat_data->shN().append_zeros(n_remaining);
                _splat_data->shN().index_put_(new_indices, append_shN);
            }

            // Update optimizer states
            _optimizer->extend_state_for_new_params(ParamType::Means, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Rotation, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Scaling, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Sh0, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Opacity, n_remaining);

            if (_birth_tri.is_valid()) {
                auto append_sources = split_idxs.slice(0, num_filled, num_split);
                auto gathered = _birth_tri.slice(0, 0, old_size)
                                    .index_select(0, append_sources)
                                    .contiguous();
                _birth_tri.slice(0, old_size, old_size + n_remaining).copy_(gathered);
            }

            {
                auto append_sources = split_idxs.slice(0, num_filled, num_split);
                auto gathered = _mesh_hole_fill_mask.slice(0, 0, old_size)
                                    .index_select(0, append_sources)
                                    .contiguous();
                _mesh_hole_fill_mask.slice(0, old_size, old_size + n_remaining).copy_(gathered);
            }

            if (_surface_prev_means.is_valid() && _surface_prev_means.numel() >= (old_size + n_remaining) * 3) {
                auto append_sources = split_idxs.slice(0, num_filled, num_split);
                auto gathered_prev = _surface_prev_means.slice(0, 0, old_size)
                                         .index_select(0, append_sources)
                                         .contiguous();
                _surface_prev_means.slice(0, old_size, old_size + n_remaining).copy_(gathered_prev);
            }
        }

        LOG_DEBUG("split(): done, {} filled free slots, {} appended", num_filled, remaining);
    }

    std::pair<lfs::core::Tensor, int64_t> ADC::fill_free_slots_with_data(
        const lfs::core::Tensor& positions,
        const lfs::core::Tensor& rotations,
        const lfs::core::Tensor& scales,
        const lfs::core::Tensor& sh0,
        const lfs::core::Tensor& shN,
        const lfs::core::Tensor& opacities,
        int64_t count) {

        if (!_free_mask.is_valid() || count == 0) {
            return {lfs::core::Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());

        // Find free slot indices within current size
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            return {lfs::core::Tensor(), count};
        }

        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);

        // Copy data to free slots
        _splat_data->means().index_put_(target_indices, positions.slice(0, 0, slots_to_fill));
        _splat_data->rotation_raw().index_put_(target_indices, rotations.slice(0, 0, slots_to_fill));
        _splat_data->scaling_raw().index_put_(target_indices, scales.slice(0, 0, slots_to_fill));

        // sh0 needs reshape from [slots_to_fill, 3] to [slots_to_fill, 1, 3]
        auto sh0_reshaped = sh0.slice(0, 0, slots_to_fill).reshape(lfs::core::TensorShape({static_cast<size_t>(slots_to_fill), 1, 3}));
        _splat_data->sh0().index_put_(target_indices, sh0_reshaped);

        _splat_data->opacity_raw().index_put_(target_indices, opacities.slice(0, 0, slots_to_fill));

        if (shN.is_valid() && has_shN_coefficients(_splat_data->shN())) {
            const auto& shN_shape = _splat_data->shN().shape();
            const auto n = static_cast<int>(slots_to_fill);
            const auto shN_slice = (shN_shape.rank() == 3)
                                       ? shN.slice(0, 0, slots_to_fill).reshape({n, static_cast<int>(shN_shape[1]), static_cast<int>(shN_shape[2])})
                                       : shN.slice(0, 0, slots_to_fill).reshape({n, static_cast<int>(shN_shape[1])});
            _splat_data->shN().index_put_(target_indices, shN_slice);
        }

        // Reset optimizer states for filled slots
        auto reset_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(slots_to_fill)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(target_indices, zeros);
            state->exp_avg_sq.index_put_(target_indices, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(target_indices, zeros);
            }
        };

        reset_optimizer_state(ParamType::Means);
        reset_optimizer_state(ParamType::Rotation);
        reset_optimizer_state(ParamType::Scaling);
        reset_optimizer_state(ParamType::Sh0);
        reset_optimizer_state(ParamType::ShN);
        reset_optimizer_state(ParamType::Opacity);

        // Mark filled slots as active
        auto false_vals = lfs::core::Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, target_indices.device());
        _free_mask.index_put_(target_indices, false_vals);
        set_deleted_mask_rows(*_splat_data, _free_mask, target_indices, false);

        // Clear mesh-init flag for reused slots (new Gaussians are not mesh-init)
        if (_mesh_init_mask.is_valid()) {
            _mesh_init_mask.index_put_(target_indices, false_vals);
        }
        _mesh_hole_fill_mask.index_put_(target_indices, false_vals);

        return {target_indices, count - slots_to_fill};
    }

    void ADC::grow_gs(int iter) {
        lfs::core::Tensor numer = _splat_data->_densification_info[1];
        lfs::core::Tensor denom = _splat_data->_densification_info[0];
        const lfs::core::Tensor grads = numer / denom.clamp_min(1.0f);

        lfs::core::Tensor is_grad_high = grads > _params->grad_threshold;

        // Exclude free slots from consideration
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (_free_mask.is_valid() && current_size > 0) {
            auto active_free_mask = _free_mask.slice(0, 0, current_size);
            auto is_active = active_free_mask.logical_not(); // true = slot is active (not free)
            is_grad_high = is_grad_high.logical_and(is_active);
        }

        // // Skip all growth when scene was initialized from mesh (mesh2splat)
        // if (_mesh_init_mask.is_valid()) {
        //     return;
        // }

        // Get max along last dimension
        const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
        const lfs::core::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data->get_scene_scale();
        lfs::core::Tensor is_duplicated = is_grad_high.logical_and(is_small);

        auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

        const lfs::core::Tensor is_large = is_small.logical_not();
        lfs::core::Tensor is_split = is_grad_high.logical_and(is_large);
        auto num_split = static_cast<int64_t>(is_split.sum_scalar());

        // Enforce max_cap: limit growth to stay within capacity
        if (_params->max_cap > 0) {
            const int current_n = _splat_data->size();
            // Duplication adds num_duplicates, split replaces num_split with 2*num_split (net +num_split)
            const int64_t potential_new = num_duplicates + num_split;
            const int64_t active_n = static_cast<int64_t>(active_count());
            const int64_t available = std::max<int64_t>(
                0, static_cast<int64_t>(_params->max_cap) - active_n);

            if (potential_new > available) {
                // Need to limit - prioritize duplication over split (duplicates small Gaussians)
                if (num_duplicates >= available) {
                    // Can only do partial duplication, no split
                    num_duplicates = available;
                    num_split = 0;
                    // Limit is_duplicated to first 'available' true values
                    auto indices = is_duplicated.nonzero().squeeze(-1);
                    if (indices.numel() > available) {
                        auto keep_indices = indices.slice(0, 0, available);
                        is_duplicated = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_duplicated.device());
                        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(available)}, is_duplicated.device());
                        is_duplicated.index_put_(keep_indices, true_vals);
                    }
                    is_split = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_split.device());
                } else {
                    // Do all duplications, limit splits
                    const int64_t remaining = available - num_duplicates;
                    num_split = remaining;
                    // Limit is_split to first 'remaining' true values
                    auto indices = is_split.nonzero().squeeze(-1);
                    if (indices.numel() > remaining) {
                        auto keep_indices = indices.slice(0, 0, remaining);
                        is_split = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_split.device());
                        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(remaining)}, is_split.device());
                        is_split.index_put_(keep_indices, true_vals);
                    }
                }
                LOG_DEBUG("max_cap enforcement: limited growth from {} to {} new Gaussians", potential_new, available);
            }
        }

        LOG_DEBUG("grow_gs(): {} duplicates, {} splits", num_duplicates, num_split);

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        auto zeros_to_concat = lfs::core::Tensor::zeros_bool({static_cast<size_t>(num_duplicates)}, is_split.device());
        is_split = is_split.cat(zeros_to_concat, 0);

        if (num_split > 0) {
            split(is_split);
        }

        assert(_params->max_cap <= 0 ||
               static_cast<size_t>(_splat_data->size()) <=
                   std::max(current_size, static_cast<size_t>(_params->max_cap)));
    }

    void ADC::remove(const lfs::core::Tensor& is_prune) {
        // Soft deletion: mark slots as free instead of resizing tensors
        // This avoids expensive tensor reallocations during training
        const lfs::core::Tensor prune_indices = is_prune.nonzero().squeeze(-1);
        const int64_t num_pruned = prune_indices.numel();

        if (num_pruned == 0) {
            return;
        }

        // Mark pruned slots as free
        mark_as_free(prune_indices);
        set_deleted_mask_rows(*_splat_data, _free_mask, prune_indices, true);

        if (_birth_tri.is_valid()) {
            auto unconstrained = lfs::core::Tensor::full(
                {static_cast<size_t>(num_pruned)}, -1.0f,
                _birth_tri.device(), lfs::core::DataType::Int32);
            _birth_tri.index_put_(prune_indices, unconstrained);
        }

        if (_mesh_init_mask.is_valid()) {
            auto false_vals = lfs::core::Tensor::zeros_bool(
                {static_cast<size_t>(num_pruned)}, _mesh_init_mask.device());
            _mesh_init_mask.index_put_(prune_indices, false_vals);
        }
        {
            auto false_vals = lfs::core::Tensor::zeros_bool(
                {static_cast<size_t>(num_pruned)}, _mesh_hole_fill_mask.device());
            _mesh_hole_fill_mask.index_put_(prune_indices, false_vals);
        }

        // SplatData::deleted() is the canonical inactive state used by export and
        // scene consumers. Keep zero quaternion as a secondary fast-rasterizer sentinel.
        // Zero out quaternion to trigger early exit in preprocessing kernel.
        // The rasterizer checks: if (q_norm_sq < 1e-8f) active = false
        // This happens BEFORE expensive covariance computation and gradient computation
        auto zero_rotation = lfs::core::Tensor::zeros(
            {static_cast<size_t>(num_pruned), 4},
            _splat_data->rotation_raw().device());
        _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

        // Zero optimizer states in-place (preserves capacity)
        auto zero_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(num_pruned)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            // Modify in-place to preserve capacity
            state->exp_avg.index_put_(prune_indices, zeros);
            state->exp_avg_sq.index_put_(prune_indices, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(prune_indices, zeros);
            }
        };

        zero_optimizer_state(ParamType::Means);
        zero_optimizer_state(ParamType::Rotation);
        zero_optimizer_state(ParamType::Scaling);
        zero_optimizer_state(ParamType::Sh0);
        zero_optimizer_state(ParamType::ShN);
        zero_optimizer_state(ParamType::Opacity);

        LOG_DEBUG("remove(): soft-deleted {} Gaussians (marked as free, rotation & gradients zeroed)", num_pruned);
    }

    void ADC::prune_gs(int iter) {
        // Check for low opacity
        lfs::core::Tensor is_prune = _splat_data->get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data->rotation_raw();
        is_prune = is_prune.logical_or(rotation_raw.square().sum(-1, false) < 1e-8f);

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
            lfs::core::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data->get_scene_scale();
            is_prune = is_prune.logical_or(is_too_big);
        }

        // Exclude already-free slots from pruning (they're already soft-deleted)
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (_free_mask.is_valid() && current_size > 0) {
            auto active_free_mask = _free_mask.slice(0, 0, current_size);
            auto is_active = active_free_mask.logical_not(); // true = slot is active
            is_prune = is_prune.logical_and(is_active);      // only prune active slots
        }

        // // Exclude mesh-init Gaussians from pruning
        // if (_mesh_init_mask.is_valid() && current_size > 0) {
        //     auto is_not_mesh_init = _mesh_init_mask.slice(0, 0, current_size).logical_not();
        //     is_prune = is_prune.logical_and(is_not_mesh_init);
        // }

        const auto num_prunes = static_cast<int64_t>(is_prune.sum_scalar());
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void ADC::zero_mesh_init_gradients() {
        if (!_params || !_params->mesh2splat_opacity_no_grad) {
            return;
        }

        if (!_mesh_init_mask.is_valid()) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        auto active_mask = _mesh_init_mask.slice(0, 0, n);
        auto indices = active_mask.nonzero().squeeze(-1);
        if (indices.numel() == 0) {
            return;
        }

        auto zero_grad_at_indices = [&](ParamType type) {
            auto* state = _optimizer->get_state_mutable(type);
            if (!state || !state->grad.is_valid()) {
                return;
            }
            const auto& shape = state->grad.shape();
            std::vector<size_t> dims = {static_cast<size_t>(indices.numel())};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->grad.device());
            state->grad.index_put_(indices, zeros);
        };

        // Only opacity is locked by mesh2splat_opacity_no_grad. The older
        // mesh-init parameter freeze paths remain here as reference.
        // zero_grad_at_indices(ParamType::Means);
        // zero_grad_at_indices(ParamType::Rotation);
        zero_grad_at_indices(ParamType::Opacity);
        // zero_grad_at_indices(ParamType::Scaling);

        // // Zero only x,y components of scaling gradient (keep z for optimization)
        // {
        //     auto* state = _optimizer->get_state_mutable(ParamType::Scaling);
        //     if (state && state->grad.is_valid()) {
        //         auto grad_xy = state->grad.slice(1, 0, 2); // [N, 2] view of x,y columns
        //         auto zeros_xy = lfs::core::Tensor::zeros(
        //             {static_cast<size_t>(indices.numel()), 2}, state->grad.device());
        //         grad_xy.index_put_(indices, zeros_xy);
        //     }
        // }
    }

    void ADC::snapshot_mesh_surface_means() {
        if (!_constraint_mesh_verts.is_valid() ||
            !_constraint_mesh_indices.is_valid() ||
            !_birth_tri.is_valid())
            return;

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const size_t configured_capacity = _params && _params->max_cap > 0
                                               ? static_cast<size_t>(_params->max_cap)
                                               : n;
        const size_t capacity = std::max(configured_capacity, n);
        if (!_surface_prev_means.is_valid() ||
            _surface_prev_means.numel() < n * 3) {
            _surface_prev_means = lfs::core::Tensor::zeros_direct(
                lfs::core::TensorShape({capacity, 3}), capacity);
        }

        cudaMemcpyAsync(_surface_prev_means.ptr<float>(),
                        _splat_data->means().ptr<float>(),
                        n * 3 * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        nullptr);
    }

    void ADC::project_mesh_init_means_to_surface() {
        if (!_constraint_mesh_verts.is_valid() ||
            !_constraint_mesh_indices.is_valid() ||
            !_birth_tri.is_valid())
            return;

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const size_t F = static_cast<size_t>(_constraint_mesh_indices.shape()[0]);
        if (F == 0)
            return;

        if (!_surface_prev_means.is_valid() ||
            _surface_prev_means.numel() < n * 3 ||
            !_tri_edge_neighbors.is_valid())
            return;

        const float inside_constraint_distance =
            _params && _params->mesh_inside_constraint_distance_avg_max_scale_multiplier > 0.0f
                ? _splat_data->get_mesh2splat_mean_max_scale() *
                      _params->mesh_inside_constraint_distance_avg_max_scale_multiplier
                : 0.0f;

        mcmc::launch_surface_walk_project(
            _surface_prev_means.ptr<float>(),
            _splat_data->means().ptr<float>(),
            _birth_tri.ptr<int32_t>(),
            _tri_edge_neighbors.ptr<int32_t>(),
            _constraint_mesh_verts.ptr<float>(),
            _constraint_mesh_indices.ptr<int32_t>(),
            n,
            F,
            std::max(1, _surface_walk_steps),
            inside_constraint_distance,
            _params ? _params->mesh_inside_constraint_fade_ratio : 0.2f);
    }

    void ADC::ensure_mesh_face_normals() {
        if (!_params || (_params->lambda_mesh_normal <= 0.0f &&
                         _params->lambda_mesh_outside_barrier <= 0.0f &&
                         _params->mesh_inside_constraint_distance_avg_max_scale_multiplier <= 0.0f)) {
            return;
        }
        if (!_constraint_mesh_verts.is_valid() || !_constraint_mesh_indices.is_valid()) {
            return;
        }

        const size_t F = static_cast<size_t>(_constraint_mesh_indices.shape()[0]);
        if (F == 0) {
            return;
        }

        const bool needs_rebuild =
            !_constraint_face_normals.is_valid() ||
            _constraint_face_normals.ndim() != 2 ||
            _constraint_face_normals.shape()[0] != F ||
            _constraint_face_normals.shape()[1] != 3;
        if (!needs_rebuild) {
            return;
        }

        _constraint_face_normals = lfs::core::Tensor::empty(
            {F, 3}, _constraint_mesh_verts.device(), lfs::core::DataType::Float32);
        kernels::launch_compute_mesh_face_normals(
            _constraint_mesh_verts.ptr<float>(),
            _constraint_mesh_indices.ptr<int32_t>(),
            _constraint_face_normals.ptr<float>(),
            F,
            nullptr);
    }

    MeshSurfaceState ADC::mesh_surface_state() {
        ensure_mesh_face_normals();
        return MeshSurfaceState{
            .constraint_mesh_verts = &_constraint_mesh_verts,
            .constraint_mesh_indices = &_constraint_mesh_indices,
            .tri_edge_neighbors = &_tri_edge_neighbors,
            .constraint_face_normals = _constraint_face_normals.is_valid() ? &_constraint_face_normals : nullptr,
            .current_faces = &_birth_tri,
            .scene_radius = _mesh_scene_radius,
            .walk_steps = std::max(1, _surface_walk_steps)};
    }

    void ADC::reset_opacity() {
        const float threshold = 2.0f * _params->prune_opacity;
        const float logit_threshold = std::log(threshold / (1.0f - threshold));

        // In-place ops preserve capacity
        _splat_data->opacity_raw().clamp_max_(logit_threshold);

        auto* state = _optimizer->get_state_mutable(ParamType::Opacity);
        if (state) {
            state->exp_avg.zero_();
            state->exp_avg_sq.zero_();
        }
    }

    void ADC::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement. Saves memory and processing time.
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            // Reinit if invalid (e.g., checkpoint resume with extended stop_refine)
            const auto& info = _splat_data->_densification_info;
            const size_t n = static_cast<size_t>(_splat_data->size());
            if (!info.is_valid() || info.ndim() != 2 || info.shape()[1] != n) {
                _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
            }

            if (mesh_surface_hard_projection_enabled(*_params)) {
                snapshot_mesh_surface_means();
            }
            grow_gs(iter);
            prune_gs(iter);
            if (mesh_surface_hard_projection_enabled(*_params)) {
                project_mesh_init_means_to_surface();
            }

            // Trim memory pools after densification to release temporary allocations
            lfs::core::Tensor::trim_memory_pool();

            _splat_data->_densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data->size())},
                _splat_data->means().device());
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }
    }

    void ADC::step(int iter) {
        if (iter < _params->iterations) {
            if (mesh_surface_hard_projection_enabled(*_params)) {
                snapshot_mesh_surface_means();
            }
            zero_mesh_init_gradients();
            _optimizer->step(iter);
            if (mesh_surface_hard_projection_enabled(*_params)) {
                project_mesh_init_means_to_surface();
            }
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t DEFAULT_MAGIC = 0x4C464446; // "LFDF"
        constexpr uint32_t DEFAULT_VERSION = 4;        // v4 adds mesh_hole_fill_mask provenance
    } // namespace

    void ADC::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&DEFAULT_MAGIC), sizeof(DEFAULT_MAGIC));
        os.write(reinterpret_cast<const char*>(&DEFAULT_VERSION), sizeof(DEFAULT_VERSION));

        // Serialize optimizer state
        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        // Serialize scheduler state
        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        // Serialize free mask (v2+)
        if (_free_mask.is_valid()) {
            uint8_t has_free_mask = 1;
            os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
            os << _free_mask;
        } else {
            uint8_t has_free_mask = 0;
            os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
        }

        detail::serialize_mesh_row_state(
            os,
            _birth_tri,
            _mesh_init_mask,
            _mesh_hole_fill_mask,
            _constraint_mesh_indices,
            static_cast<size_t>(_splat_data->size()),
            "ADC");

        LOG_DEBUG("Serialized AdcStrategy");
    }

    void ADC::deserialize(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        lfs::core::serialization_detail::read_exact(
            is, &magic, sizeof(magic), "ADC magic");
        lfs::core::serialization_detail::read_exact(
            is, &version, sizeof(version), "ADC version");

        if (magic != DEFAULT_MAGIC) {
            throw std::runtime_error("Invalid AdcStrategy checkpoint: wrong magic");
        }
        if (version < 1 || version > DEFAULT_VERSION) {
            throw std::runtime_error("Unsupported AdcStrategy checkpoint version: " + std::to_string(version));
        }

        // Deserialize optimizer state
        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_optimizer, sizeof(has_optimizer), "ADC optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer)) {
            throw std::runtime_error("Invalid AdcStrategy checkpoint: optimizer flag/state mismatch");
        }
        if (has_optimizer) {
            _optimizer->deserialize(is);
        }

        // Deserialize scheduler state
        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_scheduler, sizeof(has_scheduler), "ADC scheduler flag");
        if (has_scheduler > 1 || (has_scheduler && !_scheduler)) {
            throw std::runtime_error("Invalid AdcStrategy checkpoint: scheduler flag/state mismatch");
        }
        if (has_scheduler) {
            _scheduler->deserialize(is);
        }

        const size_t model_size = static_cast<size_t>(_splat_data->size());
        const size_t configured_capacity = _params && _params->max_cap > 0
                                               ? static_cast<size_t>(_params->max_cap)
                                               : model_size;
        const size_t target_capacity = std::max(model_size, configured_capacity);
        lfs::core::Tensor restored_free_mask = lfs::core::Tensor::zeros_bool(
            {target_capacity}, _splat_data->means().device());

        // Deserialize free mask (v2+)
        if (version >= 2) {
            uint8_t has_free_mask = 0;
            lfs::core::serialization_detail::read_exact(
                is, &has_free_mask, sizeof(has_free_mask), "ADC free-mask flag");
            if (has_free_mask > 1) {
                throw std::runtime_error("Invalid AdcStrategy checkpoint: free-mask flag must be boolean");
            }
            if (has_free_mask) {
                lfs::core::Tensor serialized_free_mask;
                is >> serialized_free_mask;
                if (!serialized_free_mask.is_valid() ||
                    !lfs::core::is_bool_like(serialized_free_mask.dtype()) ||
                    serialized_free_mask.ndim() != 1 ||
                    serialized_free_mask.numel() < model_size ||
                    serialized_free_mask.numel() > target_capacity) {
                    throw std::runtime_error(
                        "Invalid AdcStrategy checkpoint: free mask has incompatible schema");
                }

                if (model_size > 0) {
                    if (serialized_free_mask.dtype() != lfs::core::DataType::Bool) {
                        serialized_free_mask = serialized_free_mask.to(lfs::core::DataType::Bool);
                    }
                    auto active_prefix = serialized_free_mask.slice(0, 0, model_size)
                                             .to(_splat_data->means().device());
                    restored_free_mask.slice(0, 0, model_size).copy_(active_prefix);
                }
            }
        }

        if (version >= 3) {
            detail::deserialize_mesh_row_state(
                is,
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                _constraint_mesh_verts,
                _constraint_mesh_indices,
                _tri_edge_neighbors,
                _birth_tri,
                _mesh_init_mask,
                _mesh_hole_fill_mask,
                version >= 4,
                "ADC");
        } else {
            detail::restore_legacy_mesh_row_state(
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                _birth_tri,
                _mesh_init_mask,
                _mesh_hole_fill_mask,
                "ADC");
        }

        _free_mask = std::move(restored_free_mask);
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        LOG_DEBUG("Deserialized AdcStrategy (version {})", version);
    }

    void ADC::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    size_t ADC::active_count() const {
        if (!_free_mask.is_valid()) {
            return static_cast<size_t>(_splat_data->size());
        }
        // Count slots that are NOT free (i.e., active)
        // Only count up to the current size (not full capacity)
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_count_val = static_cast<size_t>(active_region.sum_scalar());
        return current_size - free_count_val;
    }

    size_t ADC::free_count() const {
        if (!_free_mask.is_valid()) {
            return 0;
        }
        // Count free slots within current size
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        return static_cast<size_t>(active_region.sum_scalar());
    }

    lfs::core::Tensor ADC::get_active_indices() const {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return lfs::core::Tensor();
        }

        if (!_free_mask.is_valid() || free_count() == 0) {
            // No free mask or no free slots means all slots are active
            // Create all indices using ones_bool -> nonzero
            auto all_active = lfs::core::Tensor::ones_bool({current_size}, _splat_data->means().device());
            return all_active.nonzero().squeeze(-1);
        }

        // Return indices where free_mask is false (i.e., active)
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto is_active = active_region.logical_not();
        return is_active.nonzero().squeeze(-1);
    }

    void ADC::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }
        // Mark the given indices as free
        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device());
        _free_mask.index_put_(indices, true_vals);
    }

    std::pair<lfs::core::Tensor, int64_t> ADC::fill_free_slots(
        const lfs::core::Tensor& source_indices, int64_t count) {

        if (!_free_mask.is_valid() || count == 0) {
            // No free slot tracking, all need to be appended
            return {lfs::core::Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());

        // Find free slot indices within current size
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            // No free slots available
            return {lfs::core::Tensor(), count};
        }

        // Use min(count, num_free) slots
        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);
        auto src_indices = source_indices.slice(0, 0, slots_to_fill);

        // Copy data from source to target slots
        _splat_data->means().index_put_(target_indices, _splat_data->means().index_select(0, src_indices));
        _splat_data->rotation_raw().index_put_(target_indices, _splat_data->rotation_raw().index_select(0, src_indices));
        _splat_data->scaling_raw().index_put_(target_indices, _splat_data->scaling_raw().index_select(0, src_indices));
        _splat_data->sh0().index_put_(target_indices, _splat_data->sh0().index_select(0, src_indices));
        _splat_data->opacity_raw().index_put_(target_indices, _splat_data->opacity_raw().index_select(0, src_indices));

        auto& shN = _splat_data->shN();
        if (has_shN_coefficients(shN)) {
            shN.index_put_(target_indices, shN.index_select(0, src_indices));
        }

        if (_birth_tri.is_valid()) {
            auto gathered = _birth_tri.slice(0, 0, current_size)
                                .index_select(0, src_indices)
                                .contiguous();
            _birth_tri.index_put_(target_indices, gathered);
        }

        {
            auto gathered = _mesh_hole_fill_mask.slice(0, 0, current_size)
                                .index_select(0, src_indices)
                                .contiguous();
            _mesh_hole_fill_mask.index_put_(target_indices, gathered);
        }

        if (_surface_prev_means.is_valid()) {
            auto gathered_prev = _surface_prev_means.slice(0, 0, current_size)
                                     .index_select(0, src_indices)
                                     .contiguous();
            _surface_prev_means.index_put_(target_indices, gathered_prev);
        }

        // Reset optimizer states in-place (preserves capacity)
        auto update_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(slots_to_fill)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(target_indices, zeros);
            state->exp_avg_sq.index_put_(target_indices, zeros);
        };

        update_optimizer_state(ParamType::Means);
        update_optimizer_state(ParamType::Rotation);
        update_optimizer_state(ParamType::Scaling);
        update_optimizer_state(ParamType::Sh0);
        update_optimizer_state(ParamType::ShN);
        update_optimizer_state(ParamType::Opacity);

        // Mark filled slots as active (not free)
        auto false_vals = lfs::core::Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, target_indices.device());
        _free_mask.index_put_(target_indices, false_vals);
        set_deleted_mask_rows(*_splat_data, _free_mask, target_indices, false);

        // Clear mesh-init flag for reused slots (new Gaussians are not mesh-init)
        if (_mesh_init_mask.is_valid()) {
            _mesh_init_mask.index_put_(target_indices, false_vals);
        }

        const int64_t remaining = count - slots_to_fill;
        LOG_DEBUG("fill_free_slots: filled {} slots, {} remaining to append", slots_to_fill, remaining);

        return {target_indices, remaining};
    }

} // namespace lfs::training
