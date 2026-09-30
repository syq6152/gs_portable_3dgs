/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "lfs/kernels/regularization.cuh"
#include "strategy_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace lfs::training {

    namespace {
        [[nodiscard]] size_t deleted_mask_capacity(const lfs::core::SplatData& splat_data) {
            const size_t means_capacity = splat_data.means().capacity();
            return means_capacity > 0 ? means_capacity
                                      : static_cast<size_t>(splat_data.size());
        }

        void ensure_deleted_mask_size(lfs::core::SplatData& splat_data) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            auto& deleted = splat_data.deleted();
            if (!deleted.is_valid() || deleted.ndim() != 1 ||
                deleted.numel() != current_size ||
                deleted.dtype() != lfs::core::DataType::Bool) {
                deleted = lfs::core::Tensor::zeros_bool(
                    {current_size}, splat_data.means().device());
            }
            deleted.reserve(deleted_mask_capacity(splat_data));
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool(
                                    {static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool(
                                    {static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
        }

        void append_live_deleted_rows(
            lfs::core::SplatData& splat_data,
            const size_t n_rows) {
            if (n_rows == 0 || !splat_data.has_deleted_mask()) {
                return;
            }

            ensure_deleted_mask_size(splat_data);
            splat_data.deleted().append_zeros(n_rows);
        }

        void ensure_bool_row_state_capacity(
            lfs::core::Tensor& state,
            const size_t required,
            const lfs::core::Device device) {
            if (!state.is_valid()) {
                state = lfs::core::Tensor::zeros_bool({required}, device);
                return;
            }
            if (state.ndim() != 1 || state.dtype() != lfs::core::DataType::Bool) {
                throw std::runtime_error("MCMC bool row state has an incompatible schema");
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
                throw std::runtime_error("MCMC birth_tri row state has an incompatible schema");
            }
            if (state.numel() >= required) {
                return;
            }
            auto expanded = lfs::core::Tensor::full(
                {required}, -1.0f, device, lfs::core::DataType::Int32);
            expanded.slice(0, 0, state.numel()).copy_(state.to(device));
            state = std::move(expanded);
        }
    } // namespace

    MCMC::MCMC(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    lfs::core::Tensor MCMC::multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement) {
        // Use the tensor library's built-in multinomial sampling
        return lfs::core::Tensor::multinomial(weights, n, replacement);
    }

    void MCMC::update_optimizer_for_relocate(
        const lfs::core::Tensor& sampled_indices,
        const lfs::core::Tensor& dead_indices,
        ParamType param_type) {

        // Source rows have adjusted opacity/scaling, while destination rows receive
        // copied parameters. Both must start with optimizer moments consistent with
        // their new values.
        _optimizer->relocate_params_at_indices_gpu(
            param_type,
            sampled_indices.ptr<int64_t>(),
            sampled_indices.numel());
        _optimizer->relocate_params_at_indices_gpu(
            param_type,
            dead_indices.ptr<int64_t>(),
            dead_indices.numel());
    }

    void MCMC::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;
        if (!info.is_valid() ||
            info.ndim() != 2 ||
            info.shape()[0] < 2 ||
            info.shape()[1] != n) {
            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }

        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            _error_score_max = lfs::core::Tensor::zeros({n}, _splat_data->means().device());
            _error_score_windows = 0;
        }
    }

    lfs::core::Tensor MCMC::get_sampling_weights() const {
        using namespace lfs::core;

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            return Tensor::ones({n}, _splat_data->means().device());
        }

        return _error_score_max.clamp_min(1e-12f);
    }

    void MCMC::ensure_ratio_workspace_size(const size_t required) {
        if (!_ones_int32.is_valid() || _ones_int32.numel() < required) {
            _ones_int32 = lfs::core::Tensor::ones(
                {required}, _splat_data->means().device(), lfs::core::DataType::Int32);
        }
    }

    int MCMC::relocate_gs() {
        LOG_TIMER("MCMC::relocate_gs");
        using namespace lfs::core;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("relocate_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        // Find dead Gaussians: opacity <= min_opacity OR rotation magnitude near zero
        Tensor dead_mask, dead_indices;
        size_t n_dead;
        {
            LOG_TIMER("relocate_find_dead");
            // Fully fused kernel - no intermediate allocations
            const size_t N = opacities.numel();
            dead_mask = Tensor::empty({N}, Device::CUDA, DataType::Bool);
            mcmc::launch_compute_dead_mask(
                opacities.ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                dead_mask.ptr<uint8_t>(),
                N,
                _params->min_opacity);
            dead_indices = dead_mask.nonzero().squeeze(-1);
            n_dead = dead_indices.numel();
        }

        // [mesh-surface-constraint] Mesh-init Gaussians are NOT excluded from being
        // considered dead. After relocation, dead slots inherit birth_tri from their
        // sampled source (see below) so the projection step keeps them on the mesh.

        if (n_dead == 0)
            return 0;

        Tensor alive_indices;
        {
            LOG_TIMER("relocate_find_alive");
            Tensor alive_mask = dead_mask.logical_not();
            alive_indices = alive_mask.nonzero().squeeze(-1);
        }

        if (alive_indices.numel() == 0)
            return 0;

        Tensor sampled_idxs, sampled_opacities, sampled_scales;
        {
            LOG_TIMER("relocate_multinomial_sample_and_gather_FUSED");
            const size_t N = opacities.numel();

            // Get source tensors (contiguous)
            Tensor opacities_contig = opacities.contiguous();
            const Tensor sampling_weights = get_sampling_weights();
            Tensor scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()

            // Allocate outputs
            sampled_idxs = Tensor::empty({n_dead}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_dead}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_dead, 3}, Device::CUDA, DataType::Float32);

            static thread_local uint64_t seed_counter = 0;
            const uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + seed_counter++;

            // does multinomial sampling + gathering in one pass
            mcmc::launch_multinomial_sample_and_gather(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                alive_indices.ptr<int64_t>(),
                alive_indices.numel(),
                n_dead,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                N);
        }

        // Count occurrences of each sampled index (how many times each was sampled)
        Tensor ratios;
        {
            LOG_TIMER("relocate_count_occurrences");
            ensure_ratio_workspace_size(std::max(opacities.numel(), sampled_idxs.numel()));
            auto ones_N = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ones_N.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs).contiguous();

            // Clamp ratios to [1, n_max]
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("relocate_cuda_kernel");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and compute raw values
        Tensor new_opacity_raw;
        {
            LOG_TIMER("relocate_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update parameters
        {
            LOG_TIMER("relocate_update_params");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0]; // Total number of Gaussians

            // Compute log(scales) for the new scales
            Tensor new_scales_log = new_scales.log();

            // Update sampled indices with new opacity/scaling using direct CUDA kernel
            // This preserves tensor capacity (unlike index_put_ which creates new tensors)
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scales_log.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);

            // Copy sampled params to dead slots
            const size_t sh_coeffs = (_splat_data->shN().is_valid() && _splat_data->shN().ndim() >= 2)
                                         ? _splat_data->shN().shape()[1]
                                         : 0;
            mcmc::launch_copy_gaussian_params(
                sampled_idxs.ptr<int64_t>(),
                dead_indices.ptr<int64_t>(),
                _splat_data->means().ptr<float>(),
                _splat_data->sh0().ptr<float>(),
                _splat_data->shN().ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                dead_indices.numel(),
                sh_coeffs,
                opacity_dim,
                N);

            // [mesh-surface-constraint] Inherit birth_tri from source into dead slots so
            // relocated Gaussians stay constrained to the mesh patch around the source.
            if (_birth_tri.is_valid()) {
                auto src_bt = _birth_tri.slice(0, 0, N).index_select(0, sampled_idxs).contiguous();
                _birth_tri.slice(0, 0, N).index_put_(dead_indices, src_bt);
            }

            if (_mesh_init_mask.is_valid()) {
                auto src_mesh_init = _mesh_init_mask.slice(0, 0, N).index_select(0, sampled_idxs).contiguous();
                _mesh_init_mask.slice(0, 0, N).index_put_(dead_indices, src_mesh_init);
            }

            auto src_hole_fill = _mesh_hole_fill_mask.slice(0, 0, N)
                                     .index_select(0, sampled_idxs)
                                     .contiguous();
            _mesh_hole_fill_mask.slice(0, 0, N).index_put_(dead_indices, src_hole_fill);
        }

        // Update optimizer states for all parameters
        {
            LOG_TIMER("relocate_update_optimizer");
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Means);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Sh0);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::ShN);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Scaling);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Rotation);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Opacity);
        }

        if (_splat_data->has_deleted_mask()) {
            set_deleted_mask_rows(*_splat_data, dead_indices, false);
        }

        return n_dead;
    }

    int MCMC::add_new_gs() {
        LOG_TIMER("MCMC::add_new_gs");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("MCMC::add_new_gs: optimizer not initialized");
            return 0;
        }

        const int current_n = _splat_data->size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const size_t n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("add_new_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        Tensor sampled_idxs;
        Tensor sampled_opacities;
        Tensor sampled_scales;
        {
            LOG_TIMER("add_new_multinomial_sample_and_gather");

            const size_t N = opacities.numel();

            // Get raw scaling and ensure contiguity
            auto scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()
            auto opacities_contig = opacities.contiguous();
            const auto sampling_weights = get_sampling_weights();

            // Allocate output tensors
            sampled_idxs = Tensor::empty({n_new}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_new}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_new, 3}, Device::CUDA, DataType::Float32);

            // Generate random seed
            auto seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            // Call fused CUDA kernel
            mcmc::launch_multinomial_sample_all(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                N,
                n_new,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>());
        }

        // Count occurrences as int32 to avoid float->int conversions in the hot path.
        Tensor ratios;
        {
            LOG_TIMER("add_new_count_occurrences");
            ensure_ratio_workspace_size(std::max(opacities.numel(), sampled_idxs.numel()));
            ratios = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ratios.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs);

            // Clamp in int32 domain
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
            ratios = ratios.contiguous();
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation_kernel");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update existing Gaussians first (before concatenation)
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);
        }

        // Use add_new_params_gather() to leverage reserved capacity
        {
            LOG_TIMER("add_new_append_gather");
            // Gather and append parameters for new Gaussians (done after updating opacity/scaling)
            append_live_deleted_rows(*_splat_data, n_new);
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs);
        }

        // Inherit the current face cache from the source Gaussians; surface-walk can
        // migrate each copy to adjacent faces in subsequent projections.
        const size_t required_rows = static_cast<size_t>(current_n) + n_new;
        ensure_birth_tri_capacity(_birth_tri, required_rows, _splat_data->means().device());
        ensure_bool_row_state_capacity(
            _mesh_hole_fill_mask, required_rows, _splat_data->means().device());
        if (_birth_tri.is_valid()) {
            LOG_TIMER("add_new_extend_birth_tri");
            auto gathered = _birth_tri.slice(0, 0, static_cast<size_t>(current_n))
                                .index_select(0, sampled_idxs)
                                .contiguous();
            _birth_tri.slice(0, static_cast<size_t>(current_n),
                             static_cast<size_t>(current_n) + n_new)
                .copy_(gathered);
        }

        auto gathered_hole_fill = _mesh_hole_fill_mask
                                      .slice(0, 0, static_cast<size_t>(current_n))
                                      .index_select(0, sampled_idxs)
                                      .contiguous();
        _mesh_hole_fill_mask
            .slice(0, static_cast<size_t>(current_n), required_rows)
            .copy_(gathered_hole_fill);

        return n_new;
    }

    // Test helper: add_new_gs with explicitly specified indices (no multinomial sampling)
    int MCMC::add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs) {
        LOG_TIMER("MCMC::add_new_gs_with_indices_test");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("add_new_gs_with_indices_test called but optimizer not initialized");
            return 0;
        }

        const int n_new = sampled_idxs.numel();
        if (n_new == 0)
            return 0;

        // Ensure indices are Int64 (test may pass Int32)
        Tensor sampled_idxs_i64 = (sampled_idxs.dtype() == DataType::Int64) ? sampled_idxs : sampled_idxs.to(DataType::Int64);

        // Get opacities
        auto opacities = _splat_data->get_opacity();

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs_i64);
        auto sampled_scales = _splat_data->get_scaling().index_select(0, sampled_idxs_i64);

        const size_t model_rows = static_cast<size_t>(_splat_data->size());
        const size_t required = std::max(model_rows, sampled_idxs_i64.numel());
        ensure_ratio_workspace_size(required);

        // Count occurrences in int32 and keep +1 baseline.
        auto ratios = _ones_int32.slice(0, 0, model_rows).clone();
        ratios.index_add_(0, sampled_idxs_i64, _ones_int32.slice(0, 0, sampled_idxs_i64.numel()));
        ratios = ratios.index_select(0, sampled_idxs_i64);

        // Clamp in int32 domain
        const int n_max = _n_max;
        ratios = ratios.clamp(1, n_max);
        ratios = ratios.contiguous();

        // Call the CUDA relocation function
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                _params->min_opacity,
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update existing Gaussians first
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
            mcmc::launch_update_scaling_opacity(
                sampled_idxs_i64.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs_i64.numel(),
                opacity_dim,
                N);
        }

        // Use fused append_gather() operation
        {
            LOG_TIMER("add_new_params_gather");
            // Gather opacity/scaling after updating them
            append_live_deleted_rows(*_splat_data, static_cast<size_t>(n_new));
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs_i64);
        }

        const size_t required_rows = model_rows + static_cast<size_t>(n_new);
        ensure_birth_tri_capacity(_birth_tri, required_rows, _splat_data->means().device());
        ensure_bool_row_state_capacity(
            _mesh_hole_fill_mask, required_rows, _splat_data->means().device());
        if (_birth_tri.is_valid()) {
            auto gathered = _birth_tri.slice(0, 0, model_rows)
                                .index_select(0, sampled_idxs_i64)
                                .contiguous();
            _birth_tri.slice(0, model_rows, required_rows).copy_(gathered);
        }
        auto gathered_hole_fill = _mesh_hole_fill_mask.slice(0, 0, model_rows)
                                      .index_select(0, sampled_idxs_i64)
                                      .contiguous();
        _mesh_hole_fill_mask.slice(0, model_rows, required_rows).copy_(gathered_hole_fill);

        return n_new;
    }

    void MCMC::inject_noise() {
        LOG_TIMER("MCMC::inject_noise");
        using namespace lfs::core;

        const size_t required_rows = static_cast<size_t>(_splat_data->size());
        if (_noise_buffer.is_valid() && _noise_buffer.capacity() > 0 &&
            (_noise_buffer.ndim() != 2 || _noise_buffer.shape()[1] != 3 ||
             _noise_buffer.shape()[0] < required_rows ||
             _noise_buffer.capacity() < required_rows)) {
            _noise_buffer = Tensor::zeros_direct(
                TensorShape({required_rows, 3}), required_rows);
        }

        // Get current learning rate from optimizer (after scheduler has updated it)
        const float current_lr = _optimizer->get_lr() * NOISE_LR;

        // Generate noise in pre-allocated buffer
        {
            LOG_TIMER("inject_noise_generate");
            if (_noise_buffer.is_valid() && _noise_buffer.capacity() > 0) {
                // Fill pre-allocated buffer with random values (kernel will use first size() elements)
                _noise_buffer.normal_(0.0f, 1.0f);
            } else {
                // Fallback for non-capacity mode
                _noise_buffer = Tensor::randn(_splat_data->means().shape(), Device::CUDA, DataType::Float32);
            }
        }

        // Call CUDA add_noise kernel (uses first size() elements of buffer)
        {
            LOG_TIMER("inject_noise_cuda_kernel");

            if (mesh_surface_hard_projection_enabled(*_params)) {
                snapshot_mesh_surface_means();
            }

            mcmc::launch_add_noise_kernel(
                _splat_data->opacity_raw().ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                _noise_buffer.ptr<float>(),
                _splat_data->means().ptr<float>(),
                current_lr,
                _splat_data->size());

            if (mesh_surface_hard_projection_enabled(*_params)) {
                project_mesh_init_means_to_surface();
            }
        }
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        LOG_TIMER("MCMC::post_backward");

        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
            _error_score_max = lfs::core::Tensor::empty({0});
            _error_score_windows = 0;
        }

        if (iter < _params->stop_refine) {
            ensure_densification_info_shape();

            // One training iteration corresponds to one camera view, so info[1] is E_k^pi.
            // Keep the max over views as the densification priority.
            const auto& info = _splat_data->_densification_info;
            if (info.is_valid() &&
                info.ndim() == 2 &&
                info.shape()[0] >= 2 &&
                info.shape()[1] == _error_score_max.numel()) {
                const float* error_row = info.ptr<float>() + info.shape()[1];
                lfs::training::mcmc::launch_elementwise_max_inplace(
                    _error_score_max.ptr<float>(),
                    error_row,
                    _error_score_max.numel());
            }

            // Clear per-view accumulators; they are rebuilt by the next backward pass.
            _splat_data->_densification_info.zero_();
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            const int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("MCMC: Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            // New copies inherit their source's current face cache in add_new_gs(), then keep
            // moving with the same surface-walk projection path as existing constrained points.
            const int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("MCMC: Added {} new Gaussians at iteration {} (total: {})",
                          n_added, iter, _splat_data->size());
            }
            // Release cached pool memory to avoid bloat (important after add_new_gs)
            lfs::core::Tensor::trim_memory_pool();

            const size_t n = static_cast<size_t>(_splat_data->size());

            if (_error_score_max.numel() < n) {
                const size_t n_new = n - _error_score_max.numel();
                _error_score_max = _error_score_max.cat(
                    lfs::core::Tensor::zeros({n_new}, _splat_data->means().device()),
                    0);
            }

            ++_error_score_windows;
            if (_error_score_windows >= 2) {
                _error_score_max = lfs::core::Tensor::zeros({n}, _splat_data->means().device());
                _error_score_windows = 0;
            }

            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }

        // Inject noise to positions every iteration
        inject_noise();
    }

    void MCMC::zero_mesh_init_gradients() {
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
    }

    void MCMC::snapshot_mesh_surface_means() {
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

    void MCMC::project_mesh_init_means_to_surface() {
        // [mesh-surface-constraint] Project ALL Gaussians whose birth_tri >= 0 back
        // onto the mesh surface. birth_tri is a per-Gaussian current-face cache, so
        // constrained points can walk across edge-adjacent faces when adjacency is available.
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

    void MCMC::ensure_mesh_face_normals() {
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

    MeshSurfaceState MCMC::mesh_surface_state() {
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

    void MCMC::step(int iter) {
        LOG_TIMER("MCMC::step");
        if (iter < _params->iterations) {
            {
                LOG_TIMER("step_optimizer_step");
                if (mesh_surface_hard_projection_enabled(*_params)) {
                    snapshot_mesh_surface_means();
                }
                zero_mesh_init_gradients();
                _optimizer->step(iter);
            }
            if (mesh_surface_hard_projection_enabled(*_params)) {
                project_mesh_init_means_to_surface();
            }
            {
                LOG_TIMER("step_zero_grad");
                _optimizer->zero_grad(iter);
            }
            {
                LOG_TIMER("step_scheduler");
                _scheduler->step();
            }
        }
    }

    void MCMC::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        const size_t old_size = static_cast<size_t>(_splat_data->size());
        if (!mask.is_valid() || mask.ndim() != 1 || mask.numel() != old_size ||
            !is_bool_like(mask.dtype())) {
            throw std::runtime_error(
                "MCMC::remove_gaussians requires a 1D Bool/UInt8 mask matching model size");
        }

        // Get indices to keep
        Tensor normalized_mask = mask.dtype() == DataType::Bool
                                     ? mask
                                     : mask.to(DataType::Bool);
        if (normalized_mask.device() != _splat_data->means().device()) {
            normalized_mask = normalized_mask.to(_splat_data->means().device());
        }
        Tensor keep_mask = normalized_mask.logical_not();
        Tensor keep_indices = keep_mask.nonzero().squeeze(-1);
        const size_t n_remove = old_size - keep_indices.numel();

        LOG_INFO("MCMC::remove_gaussians called: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", n_remove);

        Tensor kept_deleted;
        bool clear_deleted = false;
        if (_splat_data->has_deleted_mask()) {
            const auto& deleted = _splat_data->deleted();
            if (deleted.ndim() == 1 && deleted.numel() == old_size &&
                deleted.dtype() == DataType::Bool) {
                kept_deleted = deleted.index_select(0, keep_indices).contiguous();
            } else {
                LOG_WARN("MCMC: clearing malformed deleted mask during compaction (mask size={}, model size={})",
                         deleted.numel(), old_size);
                clear_deleted = true;
            }
        }

        // Validate all six parameter/state groups before committing the first
        // group so schema errors cannot leave a partially compacted model.
        _optimizer->validate_compaction(keep_indices, old_size);

        // Compact each parameter together with its gradient and Adam moments. Doing
        // this one group at a time bounds peak memory and preserves optimizer state,
        // learning rates, step counts, and the scheduler instance.
        for (const auto type : AdamOptimizer::all_param_types()) {
            _optimizer->compact_params_and_state(type, keep_indices);
        }
        const auto& info = _splat_data->_densification_info;
        if (info.is_valid() && info.ndim() == 2 && info.shape()[1] == old_size) {
            _splat_data->_densification_info = info.index_select(1, keep_indices).contiguous();
        }
        if (_error_score_max.is_valid() && _error_score_max.ndim() == 1 && _error_score_max.numel() == old_size) {
            _error_score_max = _error_score_max.index_select(0, keep_indices).contiguous();
        }

        // Compact current-face cache identically to the active Gaussian arrays.
        if (_birth_tri.is_valid() && _birth_tri.numel() >= old_size) {
            auto kept = _birth_tri.slice(0, 0, old_size).index_select(0, keep_indices).contiguous();
            const size_t configured_cap =
                _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
            const size_t new_cap = std::max(configured_cap, kept.numel());
            _birth_tri = lfs::core::Tensor::full(
                {new_cap}, -1.0f, kept.device(), lfs::core::DataType::Int32);
            _birth_tri.slice(0, 0, kept.numel()).copy_(kept);
        }

        if (_mesh_init_mask.is_valid() && _mesh_init_mask.numel() >= old_size) {
            auto kept = _mesh_init_mask.slice(0, 0, old_size).index_select(0, keep_indices).contiguous();
            const size_t configured_cap =
                _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
            const size_t new_cap = std::max(configured_cap, kept.numel());
            _mesh_init_mask = lfs::core::Tensor::zeros_bool({new_cap}, kept.device());
            _mesh_init_mask.slice(0, 0, kept.numel()).copy_(kept);
        }

        if (_mesh_hole_fill_mask.is_valid() && _mesh_hole_fill_mask.numel() >= old_size) {
            auto kept = _mesh_hole_fill_mask.slice(0, 0, old_size)
                            .index_select(0, keep_indices)
                            .contiguous();
            const size_t configured_cap =
                _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
            const size_t new_cap = std::max(configured_cap, kept.numel());
            _mesh_hole_fill_mask = lfs::core::Tensor::zeros_bool({new_cap}, kept.device());
            _mesh_hole_fill_mask.slice(0, 0, kept.numel()).copy_(kept);
        }

        if (kept_deleted.is_valid()) {
            _splat_data->deleted() = std::move(kept_deleted);
        } else if (clear_deleted) {
            _splat_data->deleted() = Tensor();
        }
    }

    void MCMC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Pre-allocate tensor capacity if max_cap is specified
        if (_params->max_cap > 0) {
            const size_t current_size = _splat_data->size();
            const size_t capacity = std::max(
                current_size, static_cast<size_t>(_params->max_cap));
            LOG_INFO("Pre-allocating capacity for {} Gaussians (current size: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            try {
                initialize_gaussians(*_splat_data, _params->max_cap);

                // Pre-allocate noise buffer [max_cap, 3]
                _noise_buffer = Tensor::zeros_direct(TensorShape({capacity, 3}), capacity);

                LOG_INFO("Pre-allocated capacity: {}/{} Gaussians ({:.1f}%)",
                         current_size, capacity, 100.0f * current_size / capacity);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to pre-allocate capacity: {}. Continuing without pre-allocation.", e.what());
            }
        }

        _n_max = 51;
        mcmc::init_relocation_coefficients(_n_max);

        if (_params->max_cap > 0) {
            _ones_int32 = Tensor::ones({static_cast<size_t>(_params->max_cap)}, Device::CUDA, DataType::Int32);
        }

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        ensure_densification_info_shape();
        _error_score_windows = 0;

        if (_splat_data->has_mesh_init_mask()) {
            const size_t init_n = static_cast<size_t>(_splat_data->size());
            const size_t configured_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                                    : init_n;
            const size_t capacity = std::max(configured_capacity, init_n);
            _mesh_init_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
            auto src = _splat_data->mesh_init_mask().slice(0, 0, init_n);
            _mesh_init_mask.slice(0, 0, init_n).copy_(src);
            LOG_INFO("MCMC: Mesh-init marker preserved for {} Gaussians", init_n);
        }

        {
            const size_t init_n = static_cast<size_t>(_splat_data->size());
            const size_t configured_capacity = _params->max_cap > 0
                                                   ? static_cast<size_t>(_params->max_cap)
                                                   : init_n;
            const size_t capacity = std::max(configured_capacity, init_n);
            _mesh_hole_fill_mask = lfs::core::Tensor::zeros_bool(
                {capacity}, _splat_data->means().device());
            if (_splat_data->has_mesh_hole_fill_mask()) {
                const auto& source = _splat_data->mesh_hole_fill_mask();
                if (source.ndim() != 1 || source.dtype() != lfs::core::DataType::Bool ||
                    source.numel() != init_n) {
                    throw std::runtime_error(
                        "MCMC: mesh_hole_fill_mask must be a Bool tensor matching the initial model size");
                }
                _mesh_hole_fill_mask.slice(0, 0, init_n).copy_(source);
                LOG_INFO("MCMC: Hole-fill provenance preserved for {} Gaussians", init_n);
            }
        }

        // [mesh-surface-constraint] Load reference mesh + per-Gaussian current face cache.
        if (_splat_data->has_constraint_mesh()) {
            _constraint_mesh_verts = _splat_data->constraint_mesh_verts();
            _constraint_mesh_indices = _splat_data->constraint_mesh_indices();
            _mesh_scene_radius = compute_mesh_scene_radius(_constraint_mesh_verts);
            _surface_walk_steps = std::max(1, _params->mesh_surface_walk_steps);

            const size_t init_n = static_cast<size_t>(_splat_data->size());
            const size_t configured_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                                    : init_n;
            const size_t capacity = std::max(configured_capacity, init_n);
            _birth_tri = lfs::core::Tensor::full(
                {capacity}, -1.0f, _splat_data->means().device(), lfs::core::DataType::Int32);
            auto src_bt = _splat_data->birth_tri().slice(0, 0, init_n);
            _birth_tri.slice(0, 0, init_n).copy_(src_bt);
            LOG_INFO("MCMC: Mesh-surface state enabled: {} verts, {} faces, {} tracked Gaussians with {} max edge-neighbor steps",
                     _constraint_mesh_verts.shape()[0],
                     _constraint_mesh_indices.shape()[0],
                     init_n,
                     _surface_walk_steps);

            if (_splat_data->has_tri_edge_neighbors()) {
                _tri_edge_neighbors = _splat_data->tri_edge_neighbors();
            } else {
                LOG_WARN("MCMC: mesh-surface walk requested but tri_edge_neighbors are missing; projection is disabled for this SplatData.");
            }
        }

        LOG_INFO("MCMC strategy initialized with {} Gaussians", _splat_data->size());
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t MCMC_MAGIC = 0x4C464D43; // "LFMC"
        constexpr uint32_t MCMC_VERSION = 3;        // v3 adds mesh_hole_fill_mask provenance
    } // namespace

    void MCMC::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&MCMC_MAGIC), sizeof(MCMC_MAGIC));
        os.write(reinterpret_cast<const char*>(&MCMC_VERSION), sizeof(MCMC_VERSION));

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

        detail::serialize_mesh_row_state(
            os,
            _birth_tri,
            _mesh_init_mask,
            _mesh_hole_fill_mask,
            _constraint_mesh_indices,
            static_cast<size_t>(_splat_data->size()),
            "MCMC");

        LOG_DEBUG("Serialized MCMC strategy");
    }

    void MCMC::deserialize(std::istream& is) {
        uint32_t magic = 0;
        uint32_t version = 0;
        lfs::core::serialization_detail::read_exact(
            is, &magic, sizeof(magic), "MCMC magic");
        lfs::core::serialization_detail::read_exact(
            is, &version, sizeof(version), "MCMC version");

        if (magic != MCMC_MAGIC) {
            throw std::runtime_error("Invalid MCMC checkpoint: wrong magic");
        }
        if (version < 1 || version > MCMC_VERSION) {
            throw std::runtime_error("Unsupported MCMC checkpoint version: " + std::to_string(version));
        }

        // Deserialize optimizer state
        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_optimizer, sizeof(has_optimizer), "MCMC optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer)) {
            throw std::runtime_error("Invalid MCMC checkpoint: optimizer flag/state mismatch");
        }
        if (has_optimizer) {
            _optimizer->deserialize(is);
        }

        // Deserialize scheduler state
        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_scheduler, sizeof(has_scheduler), "MCMC scheduler flag");
        if (has_scheduler > 1 || (has_scheduler && !_scheduler)) {
            throw std::runtime_error("Invalid MCMC checkpoint: scheduler flag/state mismatch");
        }
        if (has_scheduler) {
            _scheduler->deserialize(is);
        }

        const size_t model_size = static_cast<size_t>(_splat_data->size());
        const size_t configured_capacity = _params && _params->max_cap > 0
                                               ? static_cast<size_t>(_params->max_cap)
                                               : model_size;
        if (version >= 2) {
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
                version >= 3,
                "MCMC");
        } else {
            detail::restore_legacy_mesh_row_state(
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                _birth_tri,
                _mesh_init_mask,
                _mesh_hole_fill_mask,
                "MCMC");
        }

        LOG_DEBUG("Deserialized MCMC strategy (version {})", version);
    }

    void MCMC::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

} // namespace lfs::training
