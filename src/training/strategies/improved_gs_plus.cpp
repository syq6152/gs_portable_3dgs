/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "improved_gs_plus.hpp"

#include "core/igs_failure_diagnostics.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "edge_rasterizer.hpp"
#include "gsplat_rasterizer.hpp"
#include "strategy_utils.hpp"

#include "core/tensor/internal/memory_pool.hpp"
#include "io/pipelined_image_loader.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/image_kernels.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "lfs/kernels/regularization.cuh"
#include "optimizer/adam_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <random>

namespace lfs::training {

    namespace {
        constexpr uint32_t IGS_PLUS_MAGIC = 0x4C464947; // "LFIG"
        constexpr uint32_t IGS_PLUS_VERSION = 3;        // v3 adds mesh_hole_fill_mask provenance
        constexpr uint32_t MAX_BUDGET_SCHEDULE_ENTRIES = 10'000'000;
        constexpr int64_t ERROR_CANDIDATE_FACTOR = 4;
        constexpr float EDGE_SCORE_WEIGHT = 0.25f;

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

        const float get_percentil_value(const float q_percent, const lfs::core::Tensor tensor) {
            auto [sorted_val, sorted_idx] = tensor.sort();

            const int num_gaussians = static_cast<int>(tensor.shape()[0]);
            const int q_index = std::clamp(static_cast<int>(num_gaussians * q_percent), 0, num_gaussians - 1);
            const float quantile_threshold = sorted_val[q_index].item_as<float>();

            return quantile_threshold;
        }

        struct CannyWorkspace {
            lfs::core::Tensor grayscale;
            lfs::core::Tensor blurred;
            lfs::core::Tensor magnitude;
            lfs::core::Tensor angle;
            lfs::core::Tensor nms_output;
        };

        CannyWorkspace create_canny_workspace(int height, int width) {
            const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
            const auto dev = lfs::core::Device::CUDA;
            const auto dt = lfs::core::DataType::Float32;
            return {
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({static_cast<size_t>(height), static_cast<size_t>(width)}, dev, dt)};
        }

        void apply_canny_filter(const lfs::core::Tensor& input_data, CannyWorkspace& ws) {
            assert(input_data.dtype() == lfs::core::DataType::Float32);
            assert(input_data.device() == lfs::core::Device::CUDA);
            assert(input_data.ndim() == 3);

            const int width = input_data.shape()[2];
            const int height = input_data.shape()[1];

            ws.grayscale.zero_();
            ws.blurred.zero_();
            ws.magnitude.zero_();
            ws.angle.zero_();
            ws.nms_output.zero_();

            auto input_contig = input_data.contiguous();
            kernels::launch_grayscale_filter(input_contig.ptr<float>(), ws.grayscale.ptr<float>(), height, width);
            kernels::launch_gausssian_blur(ws.grayscale.ptr<float>(), ws.blurred.ptr<float>(), 3, height, width);
            kernels::launch_sobel_gradient_filter(ws.blurred.ptr<float>(), ws.magnitude.ptr<float>(), ws.angle.ptr<float>(), height, width);
            kernels::launch_nms_kernel(ws.magnitude.ptr<float>(), ws.angle.ptr<float>(), ws.nms_output.ptr<float>(), height, width);
        }

        void normalize_by_positive_median_inplace(lfs::core::Tensor& tensor) {
            tensor.masked_fill_(tensor.isnan(), 0.0f);
            auto valid = tensor.masked_select(tensor > 0.0f);
            if (valid.numel() == 0) {
                tensor.zero_();
                return;
            }
            auto [sorted, _] = valid.sort();
            float median = sorted[valid.numel() / 2].item_as<float>();
            tensor.div_(std::max(median, 1e-9f));
        }

        lfs::core::Tensor normalized_by_positive_median(const lfs::core::Tensor& tensor) {
            auto normalized = tensor.clone();
            normalize_by_positive_median_inplace(normalized);
            return normalized;
        }

        [[nodiscard]] size_t deleted_mask_capacity(const lfs::core::SplatData& splat_data, const lfs::core::Tensor& free_mask) {
            return free_mask.is_valid() ? static_cast<size_t>(free_mask.numel())
                                        : static_cast<size_t>(splat_data.size());
        }

        void ensure_deleted_mask_size(lfs::core::SplatData& splat_data, const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);
            auto& deleted = splat_data.deleted();
            if (!deleted.is_valid() || deleted.ndim() != 1 || deleted.numel() != current_size) {
                deleted = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
            }
            deleted.reserve(desired_capacity);
        }

        void sync_deleted_mask_from_free_mask(lfs::core::SplatData& splat_data, const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);

            if (!free_mask.is_valid()) {
                splat_data.deleted() = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
                splat_data.deleted().reserve(desired_capacity);
                return;
            }

            splat_data.deleted() = free_mask.slice(0, 0, current_size).clone();
            splat_data.deleted().reserve(desired_capacity);
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const lfs::core::Tensor& indices,
            bool deleted) {
            if (indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool({static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
        }

        void append_live_deleted_rows(lfs::core::SplatData& splat_data, const lfs::core::Tensor& free_mask, size_t n_rows) {
            if (n_rows == 0) {
                return;
            }

            auto& deleted = splat_data.deleted();
            if (!deleted.is_valid()) {
                deleted = lfs::core::Tensor::zeros_bool({static_cast<size_t>(splat_data.size())}, splat_data.means().device());
            }
            deleted.reserve(deleted_mask_capacity(splat_data, free_mask));
            deleted.append_zeros(n_rows);
        }

        struct SampledScaleSummary {
            float p95 = 0.0f;
            float max = 0.0f;
            float exp_max = 0.0f;
        };

        [[nodiscard]] SampledScaleSummary summarize_sampled_scales(
            const lfs::core::Tensor& scaling_raw,
            const lfs::core::Tensor& sampled_idxs) {
            if (sampled_idxs.numel() == 0) {
                return {};
            }

            const int sampled_scale_count = static_cast<int>(sampled_idxs.numel() * 3);
            auto sampled_scales = scaling_raw.index_select(0, sampled_idxs).reshape({sampled_scale_count});
            if (sampled_scales.numel() == 0) {
                return {};
            }

            const float p95 = get_percentil_value(0.95f, sampled_scales);
            const float max_raw = get_percentil_value(1.0f, sampled_scales);
            return {
                .p95 = p95,
                .max = max_raw,
                .exp_max = std::exp(max_raw),
            };
        }

    } // namespace

    ImprovedGSPlus::ImprovedGSPlus(lfs::core::SplatData& splat_data)
        : _splat_data(&splat_data) {}

    std::vector<int64_t> ImprovedGSPlus::get_count_array() {

        const int64_t budget = _params->max_cap;
        this->_initial_points = _splat_data->size();

        this->_total_steps = static_cast<int>((_params->stop_refine - _params->start_refine) / _params->refine_every) + 2;

        std::vector<int64_t> values;
        values.reserve(_total_steps);

        // Equation (2) Taming paper
        const float slope_lower_bound = (budget - _initial_points) / _total_steps;

        const float k = 2 * slope_lower_bound;
        const float a = (budget - _initial_points - k * _total_steps) / (_total_steps * _total_steps);
        const float b = k;
        const float c = static_cast<float>(_initial_points);

        // Set the total number of primitives up to add in each step
        for (int i = 1; i <= _total_steps; i++) {
            values.push_back(static_cast<int64_t>(1 * a * pow(i, 2) + (b * i) + c));
        }

        return values;
    }

    void ImprovedGSPlus::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Initialize Gaussians
        initialize_gaussians(*(_splat_data), _params->max_cap);

        // Create optimizer and scheduler
        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);

        const double gamma = std::pow(0.1, 1.0 / optimParams.iterations);
        _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, std::vector<ParamType>{ParamType::Means, ParamType::Scaling});

        // Initialize densification info: [2, N] tensor for tracking per-view densification statistics
        _splat_data->_densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data->size())},
            _splat_data->means().device());

        // Initialize free mask: all slots are active (not free)
        const size_t init_n = static_cast<size_t>(_splat_data->size());
        const size_t configured_capacity = _params->max_cap > 0
                                               ? static_cast<size_t>(_params->max_cap)
                                               : init_n;
        const size_t capacity = std::max(configured_capacity, init_n);
        _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        _mesh_init_mask = lfs::core::Tensor();
        if (_splat_data->has_mesh_init_mask()) {
            _mesh_init_mask = lfs::core::Tensor::zeros_bool(
                {capacity}, _splat_data->means().device());
            auto src = _splat_data->mesh_init_mask().slice(0, 0, init_n);
            _mesh_init_mask.slice(0, 0, init_n).copy_(src);
            LOG_INFO("ImprovedGSPlus: Mesh-init marker preserved for {} Gaussians", init_n);
        }

        _mesh_hole_fill_mask = lfs::core::Tensor::zeros_bool(
            {capacity}, _splat_data->means().device());
        if (_splat_data->has_mesh_hole_fill_mask()) {
            const auto& source = _splat_data->mesh_hole_fill_mask();
            if (source.ndim() != 1 || source.dtype() != lfs::core::DataType::Bool ||
                source.numel() != init_n) {
                throw std::runtime_error(
                    "ImprovedGSPlus: mesh_hole_fill_mask must be a Bool tensor matching the initial model size");
            }
            _mesh_hole_fill_mask.slice(0, 0, init_n).copy_(source);
            LOG_INFO("ImprovedGSPlus: Hole-fill provenance preserved for {} Gaussians", init_n);
        }

        _constraint_mesh_verts = lfs::core::Tensor();
        _constraint_mesh_indices = lfs::core::Tensor();
        _birth_tri = lfs::core::Tensor();
        _tri_edge_neighbors = lfs::core::Tensor();
        _constraint_face_normals = lfs::core::Tensor();
        _surface_prev_means = lfs::core::Tensor();
        _mesh_scene_radius = 1.0f;
        _surface_walk_steps = std::max(1, _params->mesh_surface_walk_steps);

        if (_splat_data->has_constraint_mesh()) {
            _constraint_mesh_verts = _splat_data->constraint_mesh_verts();
            _constraint_mesh_indices = _splat_data->constraint_mesh_indices();
            _mesh_scene_radius = compute_mesh_scene_radius(_constraint_mesh_verts);

            _birth_tri = lfs::core::Tensor::full(
                {capacity}, -1.0f, _splat_data->means().device(), lfs::core::DataType::Int32);
            auto src_bt = _splat_data->birth_tri().slice(0, 0, init_n);
            _birth_tri.slice(0, 0, init_n).copy_(src_bt);

            LOG_INFO("ImprovedGSPlus: Mesh-surface state enabled: {} verts, {} faces, {} tracked Gaussians with {} max edge-neighbor steps",
                     _constraint_mesh_verts.shape()[0],
                     _constraint_mesh_indices.shape()[0],
                     init_n,
                     _surface_walk_steps);

            if (_splat_data->has_tri_edge_neighbors()) {
                _tri_edge_neighbors = _splat_data->tri_edge_neighbors();
            } else {
                LOG_WARN("ImprovedGSPlus: mesh-surface walk requested but tri_edge_neighbors are missing; projection is disabled for this SplatData.");
            }
        }

        // Initialize I-GS+ specifics
        this->_current_step = 0;

        this->_budget_schedule = get_count_array();
        ensure_error_score_shape();
    }

    const lfs::core::Tensor ImprovedGSPlus::compute_gaussian_score() {
        const int64_t N = _splat_data->size();

        auto view_indices = random_cam_indices();
        const int num_views = static_cast<int>(view_indices.size());
        assert(num_views > 0);

        CannyWorkspace canny_ws;
        auto gaussian_scores = lfs::core::Tensor::zeros(
            {static_cast<size_t>(N)}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        for (int view = 0; view < num_views; view++) {
            const int idx = view_indices[view];
            lfs::core::Camera* cam = _views->get_camera(idx);

            assert(_image_loader && "set_image_loader() must be called before training");
            lfs::io::LoadParams params;
            params.resize_factor = _views->get_resize_factor();
            params.max_width = _views->get_max_width();
            if (cam->is_undistort_prepared()) {
                params.undistort = &cam->undistort_params();
            }
            lfs::core::Tensor image = _image_loader->load_image_immediate(cam->image_path(), params);

            const int img_h = image.shape()[1];
            const int img_w = image.shape()[2];

            if (cam->image_width() != img_w || cam->image_height() != img_h) {
                cam->set_image_dimensions(img_w, img_h);
            }
            if (view == 0 ||
                img_h != static_cast<int>(canny_ws.nms_output.shape()[0]) ||
                img_w != static_cast<int>(canny_ws.nms_output.shape()[1])) {
                canny_ws = create_canny_workspace(img_h, img_w);
            }

            apply_canny_filter(image, canny_ws);
            normalize_by_positive_median_inplace(canny_ws.nms_output);

            lfs::core::Tensor bg;
            auto score_render = edge_rasterize(*cam, this->get_model(), bg, canny_ws.nms_output);

            normalize_by_positive_median_inplace(score_render.edges_score);
            gaussian_scores.add_(score_render.edges_score);
        }

        gaussian_scores.div_(static_cast<float>(num_views));
        return gaussian_scores;
    }

    void ImprovedGSPlus::ensure_error_score_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            _error_score_max = lfs::core::Tensor::zeros({n}, _splat_data->means().device());
        }
    }

    void ImprovedGSPlus::densify_with_score(const lfs::core::Tensor& edge_scores, const lfs::core::Tensor& error_scores, const int64_t budget) {
        const int64_t current_size = static_cast<int64_t>(_splat_data->size());
        const int64_t curr_points = static_cast<int64_t>(active_count());
        const int64_t free_slots = static_cast<int64_t>(free_count());
        const int64_t budget_for_alloc = std::max<int64_t>(0, budget - curr_points);
        if (budget_for_alloc <= 0) {
            return;
        }

        const auto active_indices = get_active_indices();
        const int64_t total_active = static_cast<int64_t>(active_indices.numel());
        if (total_active == 0) {
            return;
        }

        const int64_t candidate_budget = std::min<int64_t>(
            total_active,
            std::max<int64_t>(budget_for_alloc, budget_for_alloc * ERROR_CANDIDATE_FACTOR));

        const auto normalized_error = normalized_by_positive_median(error_scores);
        const auto normalized_edge = normalized_by_positive_median(edge_scores);
        const auto device = _splat_data->means().device();

        auto active_mask = lfs::core::Tensor::zeros_bool({static_cast<size_t>(_splat_data->size())}, device);
        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(active_indices.numel())}, device);
        active_mask.index_put_(active_indices, true_vals);

        lfs::core::Tensor candidate_mask = active_mask;
        if (candidate_budget < total_active) {
            const auto active_error = normalized_error.index_select(0, active_indices);
            auto [sorted_error, _] = active_error.sort(0, true);
            const float threshold = sorted_error[candidate_budget - 1].item_as<float>();
            candidate_mask = active_mask.logical_and(normalized_error >= threshold);
        }

        auto sampling_scores = normalized_error * (normalized_edge * EDGE_SCORE_WEIGHT + 1.0f);
        sampling_scores = sampling_scores.masked_fill(~candidate_mask, 0.0f);

        int64_t selectable = static_cast<int64_t>(sampling_scores.count_nonzero());
        if (selectable < budget_for_alloc) {
            auto edge_fallback = normalized_edge.masked_fill(~active_mask, 0.0f);
            selectable = static_cast<int64_t>(edge_fallback.count_nonzero());
            if (selectable > 0) {
                sampling_scores = std::move(edge_fallback);
            } else {
                auto active_weights = lfs::core::Tensor::zeros({static_cast<size_t>(_splat_data->size())}, device);
                auto active_weight_vals = lfs::core::Tensor::ones({static_cast<size_t>(active_indices.numel())}, device);
                active_weights.index_put_(active_indices, active_weight_vals);
                sampling_scores = std::move(active_weights);
                selectable = total_active;
            }
        }

        if (selectable <= 0) {
            return;
        }

        _pending_failure_snapshot.valid = true;
        _pending_failure_snapshot.size_before = current_size;
        _pending_failure_snapshot.active_before = curr_points;
        _pending_failure_snapshot.free_before = free_slots;
        _pending_failure_snapshot.budget = budget;
        _pending_failure_snapshot.budget_for_alloc = budget_for_alloc;
        _pending_failure_snapshot.candidate_budget = candidate_budget;
        _pending_failure_snapshot.selectable = selectable;
        _pending_failure_snapshot.selected = std::min<int64_t>(budget_for_alloc, selectable);
        _pending_failure_snapshot.num_filled = 0;
        _pending_failure_snapshot.num_appended = 0;
        _pending_failure_snapshot.sampled_scale_p95 = 0.0f;
        _pending_failure_snapshot.sampled_scale_max = 0.0f;
        _pending_failure_snapshot.sampled_scale_exp_max = 0.0f;

        LAS_densify(sampling_scores.clamp_min(1e-12f), std::min<int64_t>(budget_for_alloc, selectable));
    }

    void ImprovedGSPlus::LAS_densify(const lfs::core::Tensor& scores, const int64_t budget_for_alloc) {
        const lfs::core::Tensor sampled_idxs = lfs::core::Tensor::multinomial(scores, budget_for_alloc, false);
        const auto sampled_scale_summary = summarize_sampled_scales(_splat_data->scaling_raw(), sampled_idxs);
        if (_pending_failure_snapshot.valid) {
            _pending_failure_snapshot.sampled_scale_p95 = sampled_scale_summary.p95;
            _pending_failure_snapshot.sampled_scale_max = sampled_scale_summary.max;
            _pending_failure_snapshot.sampled_scale_exp_max = sampled_scale_summary.exp_max;
        }

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

        const lfs::core::Device device = _splat_data->means().device();

        // Allocate temporary tensors for split results [budget_for_alloc, ...]
        auto second_positions = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        auto second_rotations = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 4}, device);
        auto second_scales = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        auto second_sh0 = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        lfs::core::Tensor second_shN;
        if (has_shN) {
            second_shN = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), static_cast<size_t>(shN_dim)}, device);
        }
        auto second_opacities = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc)}, device);

        // Skip shN pointer when shN_dim = 0 (sh-degree 0)
        const bool use_shN = has_shN && shN_dim > 0;

        // Kernel launch: First result modifies in-place, seconds will go to temporaries:
        kernels::launch_long_axis_split_gaussians_inplace(
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
            sampled_idxs.ptr<int64_t>(),
            static_cast<int>(budget_for_alloc),
            shN_dim,
            nullptr);

        // Reset optimizer states for long-axis-split indices
        auto reset_optimizer_state_at_indices = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(budget_for_alloc)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(sampled_idxs, zeros);
            state->exp_avg_sq.index_put_(sampled_idxs, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(sampled_idxs, zeros);
            }
        };

        reset_optimizer_state_at_indices(ParamType::Means);
        reset_optimizer_state_at_indices(ParamType::Rotation);
        reset_optimizer_state_at_indices(ParamType::Scaling);
        reset_optimizer_state_at_indices(ParamType::Sh0);
        reset_optimizer_state_at_indices(ParamType::ShN);
        reset_optimizer_state_at_indices(ParamType::Opacity);

        // Now place second split results: fill free slots first, then append.
        // The in-place first child keeps the parent's mesh row state. The second
        // child inherits the mutable face cache, immutable hole-fill provenance,
        // and projection snapshot; mesh_init_mask keeps its existing newborn-row
        // semantics.
        const size_t source_region_size = static_cast<size_t>(_splat_data->size());
        auto [filled_indices, remaining] = fill_free_slots_with_data(
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities, budget_for_alloc);

        const int64_t num_filled = budget_for_alloc - remaining;

        if (_birth_tri.is_valid() && num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = sampled_idxs.slice(0, 0, num_filled);
            auto gathered = _birth_tri.slice(0, 0, source_region_size)
                                .index_select(0, filled_sources)
                                .contiguous();
            _birth_tri.index_put_(filled_indices, gathered);
        }

        if (num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = sampled_idxs.slice(0, 0, num_filled);
            auto gathered = _mesh_hole_fill_mask.slice(0, 0, source_region_size)
                                .index_select(0, filled_sources)
                                .contiguous();
            _mesh_hole_fill_mask.index_put_(filled_indices, gathered);
        }

        if (_surface_prev_means.is_valid() && num_filled > 0 && filled_indices.is_valid()) {
            auto filled_sources = sampled_idxs.slice(0, 0, num_filled);
            auto gathered_prev = _surface_prev_means.slice(0, 0, source_region_size)
                                     .index_select(0, filled_sources)
                                     .contiguous();
            _surface_prev_means.index_put_(filled_indices, gathered_prev);
        }

        if (_pending_failure_snapshot.valid) {
            _pending_failure_snapshot.num_filled = num_filled;
            _pending_failure_snapshot.num_appended = remaining;
        }

        // Append remaining second results
        if (remaining > 0) {
            const size_t old_size = static_cast<size_t>(_splat_data->size());
            const size_t n_remaining = static_cast<size_t>(remaining);

            // Get the remaining data
            const auto append_positions = second_positions.slice(0, num_filled, budget_for_alloc);
            const auto append_rotations = second_rotations.slice(0, num_filled, budget_for_alloc);
            const auto append_scales = second_scales.slice(0, num_filled, budget_for_alloc);
            const auto append_sh0_flat = second_sh0.slice(0, num_filled, budget_for_alloc);
            const auto append_opacities = second_opacities.slice(0, num_filled, budget_for_alloc);

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
                auto append_shN = second_shN.slice(0, num_filled, budget_for_alloc);
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

            if (_birth_tri.is_valid() && old_size + n_remaining <= _birth_tri.numel()) {
                auto append_sources = sampled_idxs.slice(0, num_filled, budget_for_alloc);
                auto gathered = _birth_tri.slice(0, 0, old_size)
                                    .index_select(0, append_sources)
                                    .contiguous();
                _birth_tri.slice(0, old_size, old_size + n_remaining).copy_(gathered);
            }

            if (_mesh_init_mask.is_valid() && old_size + n_remaining <= _mesh_init_mask.numel()) {
                _mesh_init_mask.slice(0, old_size, old_size + n_remaining).zero_();
            }

            if (old_size + n_remaining <= _mesh_hole_fill_mask.numel()) {
                auto append_sources = sampled_idxs.slice(0, num_filled, budget_for_alloc);
                auto gathered = _mesh_hole_fill_mask.slice(0, 0, old_size)
                                    .index_select(0, append_sources)
                                    .contiguous();
                _mesh_hole_fill_mask.slice(0, old_size, old_size + n_remaining).copy_(gathered);
            }

            if (_surface_prev_means.is_valid() &&
                _surface_prev_means.numel() >= (old_size + n_remaining) * 3) {
                auto append_sources = sampled_idxs.slice(0, num_filled, budget_for_alloc);
                auto gathered_prev = _surface_prev_means.slice(0, 0, old_size)
                                         .index_select(0, append_sources)
                                         .contiguous();
                _surface_prev_means.slice(0, old_size, old_size + n_remaining).copy_(gathered_prev);
            }
        }
    }

    void ImprovedGSPlus::zero_mesh_init_gradients() {
        if (!_params || !_params->mesh2splat_opacity_no_grad ||
            !_mesh_init_mask.is_valid()) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        auto active_mask = _mesh_init_mask.slice(0, 0, n);
        auto indices = active_mask.nonzero().squeeze(-1);
        if (indices.numel() == 0) {
            return;
        }

        auto* state = _optimizer->get_state_mutable(ParamType::Opacity);
        if (!state || !state->grad.is_valid()) {
            return;
        }

        const auto& shape = state->grad.shape();
        std::vector<size_t> dims = {static_cast<size_t>(indices.numel())};
        for (size_t i = 1; i < shape.rank(); ++i) {
            dims.push_back(shape[i]);
        }
        auto zeros = lfs::core::Tensor::zeros(
            lfs::core::TensorShape(dims), state->grad.device());
        state->grad.index_put_(indices, zeros);
    }

    void ImprovedGSPlus::snapshot_mesh_surface_means() {
        if (!_constraint_mesh_verts.is_valid() ||
            !_constraint_mesh_indices.is_valid() ||
            !_birth_tri.is_valid()) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0) {
            return;
        }

        const size_t configured_capacity = _params && _params->max_cap > 0
                                               ? static_cast<size_t>(_params->max_cap)
                                               : n;
        const size_t capacity = std::max(configured_capacity, n);
        if (!_surface_prev_means.is_valid() ||
            _surface_prev_means.numel() < n * 3) {
            _surface_prev_means = lfs::core::Tensor::zeros_direct(
                lfs::core::TensorShape({capacity, 3}), capacity);
        }

        cudaMemcpyAsync(
            _surface_prev_means.ptr<float>(),
            _splat_data->means().ptr<float>(),
            n * 3 * sizeof(float),
            cudaMemcpyDeviceToDevice,
            nullptr);
    }

    void ImprovedGSPlus::project_mesh_init_means_to_surface() {
        if (!_constraint_mesh_verts.is_valid() ||
            !_constraint_mesh_indices.is_valid() ||
            !_birth_tri.is_valid()) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0) {
            return;
        }

        const size_t face_count = static_cast<size_t>(_constraint_mesh_indices.shape()[0]);
        if (face_count == 0 ||
            !_surface_prev_means.is_valid() ||
            _surface_prev_means.numel() < n * 3 ||
            !_tri_edge_neighbors.is_valid()) {
            return;
        }

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
            face_count,
            std::max(1, _surface_walk_steps),
            inside_constraint_distance,
            _params ? _params->mesh_inside_constraint_fade_ratio : 0.2f);
    }

    void ImprovedGSPlus::ensure_mesh_face_normals() {
        if (!_params || (_params->lambda_mesh_normal <= 0.0f &&
                         _params->lambda_mesh_outside_barrier <= 0.0f &&
                         _params->mesh_inside_constraint_distance_avg_max_scale_multiplier <= 0.0f)) {
            return;
        }
        if (!_constraint_mesh_verts.is_valid() ||
            !_constraint_mesh_indices.is_valid()) {
            return;
        }

        const size_t face_count = static_cast<size_t>(_constraint_mesh_indices.shape()[0]);
        if (face_count == 0) {
            return;
        }

        const bool needs_rebuild =
            !_constraint_face_normals.is_valid() ||
            _constraint_face_normals.ndim() != 2 ||
            _constraint_face_normals.shape()[0] != face_count ||
            _constraint_face_normals.shape()[1] != 3;
        if (!needs_rebuild) {
            return;
        }

        _constraint_face_normals = lfs::core::Tensor::empty(
            {face_count, 3},
            _constraint_mesh_verts.device(),
            lfs::core::DataType::Float32);
        kernels::launch_compute_mesh_face_normals(
            _constraint_mesh_verts.ptr<float>(),
            _constraint_mesh_indices.ptr<int32_t>(),
            _constraint_face_normals.ptr<float>(),
            face_count,
            nullptr);
    }

    MeshSurfaceState ImprovedGSPlus::mesh_surface_state() {
        ensure_mesh_face_normals();
        return MeshSurfaceState{
            .constraint_mesh_verts = &_constraint_mesh_verts,
            .constraint_mesh_indices = &_constraint_mesh_indices,
            .tri_edge_neighbors = &_tri_edge_neighbors,
            .constraint_face_normals = _constraint_face_normals.is_valid()
                                           ? &_constraint_face_normals
                                           : nullptr,
            .current_faces = &_birth_tri,
            .scene_radius = _mesh_scene_radius,
            .walk_steps = std::max(1, _surface_walk_steps)};
    }

    void ImprovedGSPlus::reset_opacity() {
        const float reset_value = 0.1;
        const float logit_reset_value = std::log(reset_value / (1.0f - reset_value));

        _splat_data->opacity_raw().clamp_max_(logit_reset_value);

        auto* state = _optimizer->get_state_mutable(ParamType::Opacity);
        if (state) {
            state->exp_avg.zero_();
            state->exp_avg_sq.zero_();
        }
    }

    void ImprovedGSPlus::pre_step(int iter, RenderOutput& render_output) {
        if (iter > _params->stop_refine)
            return;
        if (!is_refining(iter))
            return;

        assert(_views && "set_views() must be called before training");

        _precomputed_scores = compute_gaussian_score();
        _precompute_valid = true;
    }

    void ImprovedGSPlus::post_backward(int iter, RenderOutput& render_output) {

        if (iter % _params->sh_degree_interval == 0) {
            this->_splat_data->increment_sh_degree();
        }

        if (iter > _params->stop_refine) {
            return;
        }

        {
            const size_t n = static_cast<size_t>(_splat_data->size());
            const auto& info = _splat_data->_densification_info;
            if (!info.is_valid() || info.ndim() != 2 || info.shape()[0] < 2 || info.shape()[1] != n) {
                _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
            }
            ensure_error_score_shape();

            const auto& accum = _splat_data->_densification_info;
            if (accum.is_valid() &&
                accum.ndim() == 2 &&
                accum.shape()[0] >= 2 &&
                accum.shape()[1] == _error_score_max.numel()) {
                const float* error_row = accum.ptr<float>() + accum.shape()[1];
                lfs::training::mcmc::launch_elementwise_max_inplace(
                    _error_score_max.ptr<float>(),
                    error_row,
                    _error_score_max.numel());
            }

            _splat_data->_densification_info.zero_();
        }

        if (is_refining(iter)) {
            assert(_precompute_valid);

            _pending_failure_snapshot = {};
            if (mesh_surface_hard_projection_enabled(*_params)) {
                snapshot_mesh_surface_means();
            }
            densify_with_score(_precomputed_scores, _error_score_max, get_current_budget());

            opacity_prune(iter);
            if (mesh_surface_hard_projection_enabled(*_params)) {
                project_mesh_init_means_to_surface();
            }

            if (_pending_failure_snapshot.valid) {
                _pending_failure_snapshot.iter = iter;
                _pending_failure_snapshot.active_after = static_cast<int64_t>(active_count());
                _pending_failure_snapshot.free_after = static_cast<int64_t>(free_count());

                lfs::core::update_igs_plus_failure_snapshot({
                    .valid = true,
                    .iter = _pending_failure_snapshot.iter,
                    .size_before = _pending_failure_snapshot.size_before,
                    .active_before = _pending_failure_snapshot.active_before,
                    .free_before = _pending_failure_snapshot.free_before,
                    .budget = _pending_failure_snapshot.budget,
                    .budget_for_alloc = _pending_failure_snapshot.budget_for_alloc,
                    .candidate_budget = _pending_failure_snapshot.candidate_budget,
                    .selectable = _pending_failure_snapshot.selectable,
                    .selected = _pending_failure_snapshot.selected,
                    .num_filled = _pending_failure_snapshot.num_filled,
                    .num_appended = _pending_failure_snapshot.num_appended,
                    .active_after = _pending_failure_snapshot.active_after,
                    .free_after = _pending_failure_snapshot.free_after,
                    .sampled_scale_p95 = _pending_failure_snapshot.sampled_scale_p95,
                    .sampled_scale_max = _pending_failure_snapshot.sampled_scale_max,
                    .sampled_scale_exp_max = _pending_failure_snapshot.sampled_scale_exp_max,
                });
            }

            lfs::core::Tensor::trim_memory_pool();

            _splat_data->_densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data->size())},
                _splat_data->means().device());
            ensure_error_score_shape();
            _error_score_max.zero_();

            this->_current_step++;

            _precomputed_scores = lfs::core::Tensor();
            _precompute_valid = false;
        }

        if (((iter % _params->reset_every) == 0) && (iter < _params->stop_refine) && (iter > 0)) {
            reset_opacity();
        }

        if (iter == _params->stop_refine) {
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
            _error_score_max = lfs::core::Tensor::empty({0});

            lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        }
    }

    bool ImprovedGSPlus::is_refining(int iter) const {
        if (!_params ||
            _current_step < 0 ||
            static_cast<size_t>(_current_step) + 1 >= _budget_schedule.size()) {
            return false;
        }

        return (iter >= _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter <= _params->stop_refine);
    }

    void ImprovedGSPlus::step(int iter) {
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

    void ImprovedGSPlus::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void ImprovedGSPlus::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    size_t ImprovedGSPlus::active_count() const {
        if (!_free_mask.is_valid()) {
            return static_cast<size_t>(_splat_data->size());
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return 0;
        }

        auto active_region = _free_mask.slice(0, 0, current_size);
        const auto free_count_val = static_cast<size_t>(active_region.sum_scalar());
        return current_size - free_count_val;
    }

    size_t ImprovedGSPlus::free_count() const {
        if (!_free_mask.is_valid()) {
            return 0;
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return 0;
        }

        auto active_region = _free_mask.slice(0, 0, current_size);
        return static_cast<size_t>(active_region.sum_scalar());
    }

    lfs::core::Tensor ImprovedGSPlus::get_active_indices() const {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return lfs::core::Tensor();
        }

        if (!_free_mask.is_valid() || free_count() == 0) {
            auto all_active = lfs::core::Tensor::ones_bool({current_size}, _splat_data->means().device());
            return all_active.nonzero().squeeze(-1);
        }

        auto active_region = _free_mask.slice(0, 0, current_size);
        auto is_active = active_region.logical_not();
        return is_active.nonzero().squeeze(-1);
    }

    std::vector<int> ImprovedGSPlus::random_cam_indices(const int N) const {
        const int num_cam_dataset = _views->size();
        int num_samples = 0;

        if (num_cam_dataset < N) {
            num_samples = num_cam_dataset;
        } else {
            const int min_cam_dataset = 0.08 * num_cam_dataset;
            num_samples = std::max(N, min_cam_dataset);
        }

        std::vector<int> all_indices(num_cam_dataset);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        std::default_random_engine rng(global_seed());
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        all_indices.resize(num_samples);
        return all_indices;
    }

    // From ImprovedGS but not used
    [[maybe_unused]] void ImprovedGSPlus::prune_post_reset() {
        const float q = 0.2f;
        const lfs::core::Tensor opacity = _splat_data->get_opacity();

        auto [sorted_val, sorted_idx] = opacity.sort();

        int num_gaussians = opacity.shape()[0];
        int q_index = static_cast<int>(num_gaussians * q);

        float quantile_threshold = sorted_val[q_index].item_as<float>();

        const lfs::core::Tensor prune_mask = (opacity < quantile_threshold);

        lfs::training::ImprovedGSPlus::remove(prune_mask);
    }

    void ImprovedGSPlus::opacity_prune(const int iter) {
        if (iter >= _params->stop_refine) {
            return;
        }
        const auto prune_mask = detail::build_igs_plus_prune_mask(
            _splat_data->get_opacity(),
            _splat_data->rotation_raw(),
            _free_mask,
            static_cast<size_t>(_splat_data->size()),
            _params->prune_opacity);
        remove(prune_mask);
    }

    void ImprovedGSPlus::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }
        // Mark the given indices as free
        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device());
        _free_mask.index_put_(indices, true_vals);
    }

    void ImprovedGSPlus::remove(const lfs::core::Tensor& is_prune) {
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

        // Zero out quaternion to trigger early exit in preprocessing kernel
        // Deleted rows are now also tracked explicitly via splat_data.deleted().
        // Zero quaternion remains a secondary inactive sentinel for preprocessing kernels.
        auto zero_rotation = lfs::core::Tensor::zeros(
            {static_cast<size_t>(num_pruned), 4},
            _splat_data->rotation_raw().device());
        _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

        if (_birth_tri.is_valid()) {
            auto unconstrained = lfs::core::Tensor::full(
                {static_cast<size_t>(num_pruned)},
                -1.0f,
                _birth_tri.device(),
                lfs::core::DataType::Int32);
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

        if (_error_score_max.is_valid() && _error_score_max.ndim() == 1 && _error_score_max.numel() >= _splat_data->size()) {
            auto zeros = lfs::core::Tensor::zeros({static_cast<size_t>(num_pruned)}, _error_score_max.device());
            _error_score_max.index_put_(prune_indices, zeros);
        }

        LOG_DEBUG("remove(): soft-deleted {} Gaussians (marked as free, rotation & gradients zeroed)", num_pruned);
    }

    std::pair<lfs::core::Tensor, int64_t> ImprovedGSPlus::fill_free_slots_with_data(
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

        // Reused rows are newborn split children, not original mesh2splat rows.
        if (_mesh_init_mask.is_valid()) {
            _mesh_init_mask.index_put_(target_indices, false_vals);
        }
        _mesh_hole_fill_mask.index_put_(target_indices, false_vals);

        if (_error_score_max.is_valid() && _error_score_max.ndim() == 1 && _error_score_max.numel() >= current_size) {
            auto zeros = lfs::core::Tensor::zeros({static_cast<size_t>(slots_to_fill)}, _error_score_max.device());
            _error_score_max.index_put_(target_indices, zeros);
        }

        return {target_indices, count - slots_to_fill};
    }

    // ===== Serialization =====
    void ImprovedGSPlus::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&IGS_PLUS_MAGIC), sizeof(IGS_PLUS_MAGIC));
        os.write(reinterpret_cast<const char*>(&IGS_PLUS_VERSION), sizeof(IGS_PLUS_VERSION));

        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        os.write(reinterpret_cast<const char*>(&_initial_points), sizeof(_initial_points));
        os.write(reinterpret_cast<const char*>(&_current_step), sizeof(_current_step));
        os.write(reinterpret_cast<const char*>(&_total_steps), sizeof(_total_steps));

        const auto budget_size = static_cast<uint32_t>(_budget_schedule.size());
        os.write(reinterpret_cast<const char*>(&budget_size), sizeof(budget_size));
        if (budget_size > 0) {
            os.write(reinterpret_cast<const char*>(_budget_schedule.data()),
                     static_cast<std::streamsize>(budget_size * sizeof(_budget_schedule.front())));
        }

        const uint8_t has_free_mask = _free_mask.is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
        if (has_free_mask) {
            os << _free_mask;
        }

        detail::serialize_mesh_row_state(
            os,
            _birth_tri,
            _mesh_init_mask,
            _mesh_hole_fill_mask,
            _constraint_mesh_indices,
            static_cast<size_t>(_splat_data->size()),
            "ImprovedGSPlus");

        LOG_DEBUG("Serialized ImprovedGSPlus");
    }

    void ImprovedGSPlus::deserialize(std::istream& is) {
        const auto start = is.tellg();

        const auto restore_legacy_state = [&]() {
            if (!_params) {
                throw std::runtime_error(
                    "Cannot restore legacy ImprovedGSPlus checkpoint before strategy initialization");
            }

            LOG_WARN("Legacy igs+ checkpoint without strategy state; rebuilding optimizer state");

            _optimizer = create_optimizer(*_splat_data, *_params);
            _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);

            const double gamma = std::pow(0.1, 1.0 / _params->iterations);
            _scheduler = std::make_unique<ExponentialLR>(
                *_optimizer, gamma, std::vector<ParamType>{ParamType::Means, ParamType::Scaling});

            const size_t model_size = static_cast<size_t>(_splat_data->size());
            const size_t configured_capacity = _params->max_cap > 0
                                                   ? static_cast<size_t>(_params->max_cap)
                                                   : model_size;
            const size_t capacity = std::max(model_size, configured_capacity);
            _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
            sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);
            detail::restore_legacy_mesh_row_state(
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                _birth_tri,
                _mesh_init_mask,
                _mesh_hole_fill_mask,
                "ImprovedGSPlus");
            _precomputed_scores = lfs::core::Tensor();
            _error_score_max = lfs::core::Tensor::zeros({model_size}, _splat_data->means().device());
            _precompute_valid = false;
            _current_step = 0;
            _budget_schedule = get_count_array();
        };

        const auto rewind_for_legacy = [&](const std::streamsize consumed) {
            is.clear();
            if (start != std::streampos(-1)) {
                is.seekg(start);
                if (!is) {
                    throw std::runtime_error(
                        "Failed to rewind legacy ImprovedGSPlus checkpoint stream");
                }
            } else if (consumed != 0) {
                throw std::runtime_error(
                    "Cannot rewind non-seekable legacy ImprovedGSPlus checkpoint stream");
            }
        };

        uint32_t magic = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        const auto magic_bytes = is.gcount();
        if (magic_bytes == 0) {
            rewind_for_legacy(magic_bytes);
            restore_legacy_state();
            return;
        }
        if (magic_bytes != static_cast<std::streamsize>(sizeof(magic))) {
            throw std::runtime_error("Truncated serialized ImprovedGSPlus magic");
        }
        if (magic != IGS_PLUS_MAGIC) {
            rewind_for_legacy(magic_bytes);
            restore_legacy_state();
            return;
        }

        uint32_t version = 0;
        lfs::core::serialization_detail::read_exact(
            is, &version, sizeof(version), "ImprovedGSPlus version");
        if (version < 1 || version > IGS_PLUS_VERSION) {
            throw std::runtime_error("Unsupported ImprovedGSPlus checkpoint version: " + std::to_string(version));
        }

        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_optimizer, sizeof(has_optimizer), "ImprovedGSPlus optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer)) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: optimizer flag/state mismatch");
        }
        if (has_optimizer) {
            _optimizer->deserialize(is);
        }

        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_scheduler, sizeof(has_scheduler), "ImprovedGSPlus scheduler flag");
        if (has_scheduler > 1 || (has_scheduler && !_scheduler)) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: scheduler flag/state mismatch");
        }
        if (has_scheduler) {
            _scheduler->deserialize(is);
        }

        int64_t restored_initial_points = 0;
        int restored_current_step = 0;
        int restored_total_steps = 0;
        lfs::core::serialization_detail::read_exact(
            is,
            &restored_initial_points,
            sizeof(restored_initial_points),
            "ImprovedGSPlus initial point count");
        lfs::core::serialization_detail::read_exact(
            is,
            &restored_current_step,
            sizeof(restored_current_step),
            "ImprovedGSPlus current schedule step");
        lfs::core::serialization_detail::read_exact(
            is,
            &restored_total_steps,
            sizeof(restored_total_steps),
            "ImprovedGSPlus total schedule steps");

        if (restored_initial_points < 0) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: initial point count must be non-negative");
        }
        if (restored_total_steps <= 0) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: total schedule steps must be positive");
        }
        if (restored_current_step < 0 || restored_current_step >= restored_total_steps) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: current schedule step is out of range");
        }

        uint32_t budget_size = 0;
        lfs::core::serialization_detail::read_exact(
            is, &budget_size, sizeof(budget_size), "ImprovedGSPlus budget schedule length");
        if (budget_size > MAX_BUDGET_SCHEDULE_ENTRIES) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: budget schedule exceeds entry limit");
        }
        if (budget_size != static_cast<uint32_t>(restored_total_steps)) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: budget schedule length does not match total steps");
        }

        std::vector<int64_t> restored_budget_schedule(budget_size);
        if (budget_size > 0) {
            lfs::core::serialization_detail::read_exact(
                is,
                restored_budget_schedule.data(),
                static_cast<size_t>(budget_size) * sizeof(restored_budget_schedule.front()),
                "ImprovedGSPlus budget schedule");
        }

        const int64_t configured_max_cap = _params ? _params->max_cap : 0;
        for (const int64_t budget : restored_budget_schedule) {
            if (budget < 0 || (configured_max_cap > 0 && budget > configured_max_cap)) {
                throw std::runtime_error(
                    "Invalid ImprovedGSPlus checkpoint: budget schedule entry is out of range");
            }
        }

        uint8_t has_free_mask = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_free_mask, sizeof(has_free_mask), "ImprovedGSPlus free-mask flag");
        if (has_free_mask > 1) {
            throw std::runtime_error(
                "Invalid ImprovedGSPlus checkpoint: free-mask flag must be boolean");
        }

        const size_t model_size = static_cast<size_t>(_splat_data->size());
        const size_t configured_capacity = configured_max_cap > 0
                                               ? static_cast<size_t>(configured_max_cap)
                                               : model_size;
        const size_t target_capacity = std::max(model_size, configured_capacity);
        lfs::core::Tensor restored_free_mask = lfs::core::Tensor::zeros_bool(
            {target_capacity}, _splat_data->means().device());

        if (has_free_mask) {
            lfs::core::Tensor serialized_free_mask;
            is >> serialized_free_mask;
            if (!serialized_free_mask.is_valid() ||
                !lfs::core::is_bool_like(serialized_free_mask.dtype()) ||
                serialized_free_mask.ndim() != 1 ||
                serialized_free_mask.numel() < model_size ||
                serialized_free_mask.numel() > target_capacity) {
                throw std::runtime_error(
                    "Invalid ImprovedGSPlus checkpoint: free mask has incompatible schema");
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

        lfs::core::Tensor restored_birth_tri;
        lfs::core::Tensor restored_mesh_init_mask;
        lfs::core::Tensor restored_mesh_hole_fill_mask;
        if (version >= 2) {
            detail::deserialize_mesh_row_state(
                is,
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                _constraint_mesh_verts,
                _constraint_mesh_indices,
                _tri_edge_neighbors,
                restored_birth_tri,
                restored_mesh_init_mask,
                restored_mesh_hole_fill_mask,
                version >= 3,
                "ImprovedGSPlus");
        } else {
            // Legacy checkpoints have no mesh bytes. Preserve the fresh scene
            // prefix while rebuilding capacity, but do so transactionally so a
            // failed restore cannot partially replace the live row state.
            restored_birth_tri = _birth_tri;
            restored_mesh_init_mask = _mesh_init_mask;
            detail::restore_legacy_mesh_row_state(
                model_size,
                configured_capacity,
                _splat_data->means().device(),
                restored_birth_tri,
                restored_mesh_init_mask,
                restored_mesh_hole_fill_mask,
                "ImprovedGSPlus");
        }

        auto restored_deleted_mask = restored_free_mask.slice(0, 0, model_size).clone();
        restored_deleted_mask.reserve(target_capacity);

        _initial_points = restored_initial_points;
        _current_step = restored_current_step;
        _total_steps = restored_total_steps;
        _budget_schedule = std::move(restored_budget_schedule);
        _free_mask = std::move(restored_free_mask);
        _birth_tri = std::move(restored_birth_tri);
        _mesh_init_mask = std::move(restored_mesh_init_mask);
        _mesh_hole_fill_mask = std::move(restored_mesh_hole_fill_mask);
        _splat_data->deleted() = std::move(restored_deleted_mask);

        _precomputed_scores = lfs::core::Tensor();
        _error_score_max = lfs::core::Tensor::zeros({static_cast<size_t>(_splat_data->size())}, _splat_data->means().device());
        _precompute_valid = false;

        LOG_DEBUG("Deserialized ImprovedGSPlus (version {})", version);
    }

} // namespace lfs::training
