/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/per_frame_affine_color.hpp"
#include "components/per_frame_observation_blur.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "components/sparsity_optimizer.hpp"
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "depth_normal_utils.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "mesh_supervision_renderer.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "pose_refiner.hpp"
#include "project_mesh_mask_builder.hpp"
#include "pseudo_view_precompute.hpp"
#include "pseudo_view_loader.hpp"
// #include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "strategies/strategy_utils.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "training/kernels/image_kernels.hpp"

#include <filesystem>
#include <fstream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        bool has_explicit_eval_split(const lfs::core::param::OptimizationParameters& params) {
            return !params.train_images.empty() && !params.test_images.empty();
        }

        class ScopedPseudoViewCpuCacheBypass {
        public:
            ScopedPseudoViewCpuCacheBypass(
                lfs::io::CacheLoader& cache_loader,
                const lfs::core::param::DatasetConfig& dataset,
                const int expected_images)
                : cache_loader_(cache_loader),
                  loading_params_(dataset.loading_params),
                  expected_images_(expected_images) {
                cache_loader_.update_cache_params(
                    false,
                    loading_params_.use_fs_cache,
                    expected_images_,
                    loading_params_.min_cpu_free_GB,
                    loading_params_.min_cpu_free_memory_ratio,
                    loading_params_.print_cache_status,
                    loading_params_.print_status_freq_num);
            }

            ~ScopedPseudoViewCpuCacheBypass() {
                cache_loader_.clear_cpu_cache();
                cache_loader_.update_cache_params(
                    loading_params_.use_cpu_memory,
                    loading_params_.use_fs_cache,
                    expected_images_,
                    loading_params_.min_cpu_free_GB,
                    loading_params_.min_cpu_free_memory_ratio,
                    loading_params_.print_cache_status,
                    loading_params_.print_status_freq_num);
            }

            ScopedPseudoViewCpuCacheBypass(const ScopedPseudoViewCpuCacheBypass&) = delete;
            ScopedPseudoViewCpuCacheBypass& operator=(const ScopedPseudoViewCpuCacheBypass&) = delete;

        private:
            lfs::io::CacheLoader& cache_loader_;
            lfs::core::param::LoadingParams loading_params_;
            int expected_images_ = 0;
        };

        std::expected<int, std::string> resolve_mesh2splat_max_cap(
            const lfs::core::param::OptimizationParameters& params,
            const lfs::core::SplatData& splat) {
            if (params.mesh2splat_max_cap_extra_ratio <= 0.0f) {
                return params.max_cap;
            }

            const bool mesh2splat_initialized =
                splat.has_mesh_init_mask() ||
                splat.has_constraint_mesh() ||
                splat.get_mesh2splat_mean_max_scale() > 0.0f;
            if (!mesh2splat_initialized) {
                return params.max_cap;
            }

            const auto initial_count = splat.size();
            if (initial_count == 0) {
                return params.max_cap;
            }
            if (initial_count > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
                return std::unexpected("mesh2splat initial Gaussian count exceeds int max_cap range");
            }

            const long double multiplier =
                1.0L + static_cast<long double>(params.mesh2splat_max_cap_extra_ratio);
            const long double target = std::ceil(static_cast<long double>(initial_count) * multiplier);
            if (target > static_cast<long double>(std::numeric_limits<int>::max())) {
                return std::unexpected("mesh2splat-derived max_cap exceeds int range");
            }

            const int dynamic_cap = std::max(static_cast<int>(initial_count), static_cast<int>(target));
            LOG_INFO("mesh2splat dynamic max_cap: initial={}, extra_ratio={:.3f}, configured={}, effective={}",
                     initial_count,
                     params.mesh2splat_max_cap_extra_ratio,
                     params.max_cap,
                     dynamic_cap);
            return dynamic_cap;
        }

        std::expected<float, std::string> resolve_mesh_inside_constraint_distance(
            const lfs::core::param::OptimizationParameters& params,
            const lfs::core::SplatData& splat) {
            if (params.mesh_inside_constraint_distance_avg_max_scale_multiplier <= 0.0f) {
                return 0.0f;
            }

            const float mean_max_scale = splat.get_mesh2splat_mean_max_scale();
            if (!(mean_max_scale > 0.0f) || !std::isfinite(mean_max_scale)) {
                return std::unexpected(
                    "mesh_inside_constraint_distance_avg_max_scale_multiplier requires positive finite mesh2splat mean max scale metadata");
            }
            const float distance =
                mean_max_scale * params.mesh_inside_constraint_distance_avg_max_scale_multiplier;
            if (!(distance > 0.0f) || !std::isfinite(distance)) {
                return std::unexpected("Mesh inside constraint distance must be positive and finite");
            }
            return distance;
        }

        template <typename Fn>
        class ScopeGuard {
        public:
            explicit ScopeGuard(Fn fn)
                : fn_(std::move(fn)) {}

            ScopeGuard(const ScopeGuard&) = delete;
            ScopeGuard& operator=(const ScopeGuard&) = delete;

            ScopeGuard(ScopeGuard&& other) noexcept
                : fn_(std::move(other.fn_)),
                  active_(other.active_) {
                other.active_ = false;
            }

            ScopeGuard& operator=(ScopeGuard&&) = delete;

            ~ScopeGuard() {
                if (active_) {
                    fn_();
                }
            }

            void release() noexcept { active_ = false; }

        private:
            Fn fn_;
            bool active_ = true;
        };

        template <typename Fn>
        ScopeGuard<Fn> makeScopeGuard(Fn fn) {
            return ScopeGuard<Fn>(std::move(fn));
        }

        PPISPRenderOverrides toRenderOverrides(const PPISPViewportOverrides& ov) {
            PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }

        lfs::core::Tensor resize_mask_to_resolution(const lfs::core::Tensor& mask, int dst_h, int dst_w) {
            if (!mask.is_valid() || mask.is_empty()) {
                return mask;
            }

            lfs::core::Tensor mask_hw = (mask.ndim() == 3 && mask.shape()[0] == 1)
                                            ? mask.squeeze(0)
                                            : mask;

            if (mask_hw.ndim() != 2) {
                return mask_hw;
            }

            if (mask_hw.device() != lfs::core::Device::CUDA) {
                mask_hw = mask_hw.to(lfs::core::Device::CUDA);
            }

            const int src_h = static_cast<int>(mask_hw.shape()[0]);
            const int src_w = static_cast<int>(mask_hw.shape()[1]);
            if (src_h == dst_h && src_w == dst_w) {
                return mask_hw;
            }

            auto dst = lfs::core::Tensor::empty(
                {1, static_cast<size_t>(dst_h), static_cast<size_t>(dst_w)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);

            lfs::training::kernels::launch_bilinear_resize_chw(
                mask_hw.ptr<float>(),
                dst.ptr<float>(),
                1,
                src_h,
                src_w,
                dst_h,
                dst_w,
                nullptr);

            return dst.reshape({dst_h, dst_w});
        }

        std::expected<std::vector<uint8_t>, std::string> read_binary_file(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected(std::format("Failed to open '{}'", lfs::core::path_to_utf8(path)));
            }

            file.seekg(0, std::ios::end);
            const auto end_pos = file.tellg();
            if (end_pos <= 0) {
                return std::unexpected(std::format("File '{}' is empty", lfs::core::path_to_utf8(path)));
            }
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> data(static_cast<size_t>(end_pos));
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!file.good() && !file.eof()) {
                return std::unexpected(std::format("Failed to read '{}'", lfs::core::path_to_utf8(path)));
            }

            return data;
        }

    } // namespace

    // Tile configuration for memory-efficient training
    enum class TileMode {
        One = 1, // 1 tile  - 1x1 - Render full image (no tiling)
        Two = 2, // 2 tiles - 2x1 - Two horizontal tiles
        Four = 4 // 4 tiles - 2x2 - Four tiles in a grid
    };

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        per_frame_affine_color_.reset();
        per_frame_observation_blur_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        pose_refiner_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;
        train_dataset_size_ = 0;
        total_cameras_count_ = 0;
        mesh_supervision_cache_dir_.clear();
        mesh_supervision_cache_dirty_.store(false);
        stop_mesh_supervision_prefetch_thread();
        mesh_supervision_disk_targets_.clear();
        mesh_supervision_hot_targets_.clear();
        mesh_supervision_hot_lru_.clear();
        mesh_supervision_uid_order_.clear();
        mesh_supervision_uid_order_index_.clear();
        mesh_supervision_gpu_targets_.clear();
        mesh_supervision_gpu_lru_.clear();
        mesh_supervision_prepared_mesh_.reset();

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            // BilateralGrid is indexed with cam->uid() in the training loop. Those UIDs stay
            // in the original camera space even when train/val splits are enabled, so the grid
            // must be sized for the full camera set rather than only the training subset.
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(total_cameras_count_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                params_.optimization.iterations,
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} camera slots ({} train images)",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     total_cameras_count_,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_per_frame_affine_color(
        const lfs::core::param::OptimizationParameters* optimization_override) {
        per_frame_affine_color_.reset();
        const auto& optimization = optimization_override ? *optimization_override : params_.optimization;
        if (!optimization.use_per_frame_affine_color) {
            return {};
        }
        if (!train_dataset_) {
            return std::unexpected("Per-frame affine color requires an initialized training dataset");
        }

        try {
            PerFrameAffineColor::Config config;
            config.lr = optimization.per_frame_affine_color_lr;

            auto affine = std::make_unique<PerFrameAffineColor>(
                static_cast<int>(optimization.iterations), config);

            size_t real_frames = 0;
            size_t pseudo_frames = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (!cam) {
                    continue;
                }
                if (cam->is_pseudo()) {
                    ++pseudo_frames;
                    continue;
                }
                affine->register_frame(cam->uid(), cam->image_name(), cam->camera_id());
                ++real_frames;
            }

            if (real_frames == 0) {
                return std::unexpected("Per-frame affine color requires at least one real training frame");
            }
            affine->finalize();

            // A pseudo view contains reprojected RGB from one real source frame.
            // Require the stable source key rather than guessing from the pose base
            // or the pseudo camera template, which may both identify other cameras.
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (!cam || !cam->is_pseudo()) {
                    continue;
                }
                if (!cam->has_pseudo_rgb_source()) {
                    return std::unexpected(std::format(
                        "Per-frame affine color requires RGB source metadata for pseudo view '{}'. "
                        "Rebuild the pseudo-view cache with source_image_name and source_camera_id.",
                        cam->image_name()));
                }
                if (!affine->has_frame_key(
                        cam->pseudo_rgb_source_image_name(),
                        cam->pseudo_rgb_source_camera_id())) {
                    return std::unexpected(std::format(
                        "Pseudo view '{}' references RGB source '{}' (camera {}) which is not a real training frame. "
                        "Rebuild the pseudo-view cache for the current dataset.",
                        cam->image_name(),
                        cam->pseudo_rgb_source_image_name(),
                        cam->pseudo_rgb_source_camera_id()));
                }
            }

            per_frame_affine_color_ = std::move(affine);
            LOG_INFO("Per-frame affine color initialized: {} real frames, {} pseudo views, lr={:.2e}",
                     real_frames,
                     pseudo_frames,
                     optimization.per_frame_affine_color_lr);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init per-frame affine color: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_per_frame_observation_blur(
        const lfs::core::param::OptimizationParameters* optimization_override) {
        per_frame_observation_blur_.reset();
        const auto& optimization =
            optimization_override ? *optimization_override : params_.optimization;
        const bool motion_enabled = optimization.use_per_frame_motion_blur;
        const bool defocus_enabled = optimization.use_per_frame_defocus_blur;
        if (!motion_enabled && !defocus_enabled) {
            return {};
        }
        if (!train_dataset_) {
            return std::unexpected(
                "Per-frame observation blur requires an initialized training dataset");
        }
        if (optimization.iterations >
                static_cast<size_t>(std::numeric_limits<int>::max()) ||
            optimization.per_frame_motion_blur_start_iter >
                static_cast<size_t>(std::numeric_limits<int>::max()) ||
            optimization.per_frame_defocus_blur_start_iter >
                static_cast<size_t>(std::numeric_limits<int>::max())) {
            return std::unexpected(
                "Per-frame observation blur iteration counts exceed the supported int range");
        }

        try {
            // Translation variance is expressed as a fraction of an explicit
            // real-training-camera scene radius.  Use half of the camera-center
            // AABB diagonal; unlike SplatData::scene_scale this has a direct
            // pose-space interpretation and is stable across Gaussian updates.
            float bbox_min[3] = {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};
            float bbox_max[3] = {
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity()};
            size_t real_frames = 0;
            for (size_t index = 0; index < train_dataset_->size(); ++index) {
                const auto cam = train_dataset_->get_camera_shared(index);
                if (!cam || cam->is_pseudo()) {
                    continue;
                }
                const std::vector<float> center = cam->cam_position().cpu().to_vector();
                if (center.size() < 3 ||
                    !std::isfinite(center[0]) ||
                    !std::isfinite(center[1]) ||
                    !std::isfinite(center[2])) {
                    return std::unexpected(std::format(
                        "Per-frame observation blur found an invalid camera center for '{}'",
                        cam->image_name()));
                }
                for (int axis = 0; axis < 3; ++axis) {
                    bbox_min[axis] = std::min(bbox_min[axis], center[axis]);
                    bbox_max[axis] = std::max(bbox_max[axis], center[axis]);
                }
                ++real_frames;
            }
            if (real_frames == 0) {
                return std::unexpected(
                    "Per-frame observation blur requires at least one real training frame");
            }

            const double dx = static_cast<double>(bbox_max[0]) - bbox_min[0];
            const double dy = static_cast<double>(bbox_max[1]) - bbox_min[1];
            const double dz = static_cast<double>(bbox_max[2]) - bbox_min[2];
            const double camera_scene_radius = 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
            if (motion_enabled &&
                (!(camera_scene_radius > 1e-8) || !std::isfinite(camera_scene_radius))) {
                return std::unexpected(
                    "Per-frame motion blur requires a non-degenerate real-camera scene radius");
            }

            PerFrameObservationBlur::Config config;
            config.motion_enabled = motion_enabled;
            config.defocus_enabled = defocus_enabled;
            config.motion_rot_lr = optimization.per_frame_motion_blur_rot_lr;
            config.motion_trans_lr = optimization.per_frame_motion_blur_trans_lr;
            config.defocus_scale_lr = optimization.per_frame_defocus_blur_scale_lr;
            config.defocus_focus_lr = optimization.per_frame_defocus_blur_focus_lr;
            config.motion_rot_reg_weight =
                optimization.per_frame_motion_blur_rot_reg_weight;
            config.motion_trans_reg_weight =
                optimization.per_frame_motion_blur_trans_reg_weight;
            config.defocus_reg_weight =
                optimization.per_frame_defocus_blur_reg_weight;
            config.motion_start_iter =
                static_cast<int>(optimization.per_frame_motion_blur_start_iter);
            config.defocus_start_iter =
                static_cast<int>(optimization.per_frame_defocus_blur_start_iter);

            const double max_rotation_std_rad =
                static_cast<double>(optimization.per_frame_motion_blur_max_rot_std_deg) *
                std::numbers::pi / 180.0;
            const double max_translation_std =
                static_cast<double>(optimization.per_frame_motion_blur_max_trans_std_scene_ratio) *
                camera_scene_radius;
            const double max_defocus_radius =
                static_cast<double>(optimization.per_frame_defocus_blur_max_radius_px);
            config.max_rot_variance = max_rotation_std_rad * max_rotation_std_rad;
            config.max_trans_variance = max_translation_std * max_translation_std;
            config.max_defocus_radius_sq = max_defocus_radius * max_defocus_radius;
            // No dataset focus metadata is available. beta starts at zero and
            // rho=0 represents an infinity-focus neutral initialization; rho
            // begins receiving data gradients once beta leaves zero.
            config.initial_focus_inverse_depth = 0.0;

            auto blur = std::make_unique<PerFrameObservationBlur>(
                static_cast<int>(optimization.iterations), config);
            for (size_t index = 0; index < train_dataset_->size(); ++index) {
                const auto cam = train_dataset_->get_camera_shared(index);
                if (!cam || cam->is_pseudo()) {
                    continue;
                }
                blur->register_frame(cam->uid(), cam->image_name(), cam->camera_id());
            }
            blur->finalize();
            per_frame_observation_blur_ = std::move(blur);

            LOG_INFO(
                "Per-frame observation blur initialized: {} real frames, motion={}, defocus={}, "
                "camera_scene_radius={:.6g}, max_rot_std_deg={:.3g}, "
                "max_trans_std={:.6g}, max_defocus_radius_px={:.3g}, "
                "max_observation_radius_px={:.3g}, rho_init=0 (infinity focus)",
                real_frames,
                motion_enabled,
                defocus_enabled,
                camera_scene_radius,
                optimization.per_frame_motion_blur_max_rot_std_deg,
                max_translation_std,
                optimization.per_frame_defocus_blur_max_radius_px,
                optimization.per_frame_observation_blur_max_radius_px);
            return {};
        } catch (const std::exception& error) {
            return std::unexpected(std::format(
                "Failed to init per-frame observation blur: {}", error.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp() {
        if (!params_.optimization.use_ppisp) {
            return {};
        }

        try {
            PPISPConfig config;
            config.lr = params_.optimization.ppisp_lr;
            config.warmup_steps = params_.optimization.ppisp_warmup_steps;
            config.exposure_only = params_.optimization.ppisp_exposure_only;
            const float reg_weight = params_.optimization.ppisp_reg_weight;
            config.exposure_mean *= reg_weight;
            config.vig_center *= reg_weight;
            config.vig_channel *= reg_weight;
            config.vig_non_pos *= reg_weight;
            config.color_mean *= reg_weight;
            config.crf_channel *= reg_weight;

            ppisp_ = std::make_unique<PPISP>(params_.optimization.iterations, config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}, reg_weight={:.2e}, mode={}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr, config.warmup_steps,
                     reg_weight, config.exposure_only ? "exposure-only" : "full");

            if (auto result = apply_ppisp_sidecar_if_configured(); !result) {
                return result;
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP: {}", e.what()));
        }
    }

    std::expected<PPISPFileMetadata, std::string> Trainer::build_ppisp_sidecar_metadata() const {
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot build PPISP sidecar metadata before PPISP is initialized");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot build PPISP sidecar metadata without an active training dataset");
        }

        PPISPFileMetadata metadata;
        metadata.dataset_path_utf8 = lfs::core::path_to_utf8(params_.dataset.data_path);
        metadata.images_folder = params_.dataset.images;
        metadata.camera_ids = ppisp_->ordered_camera_ids();

        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            metadata.frame_image_names.push_back(cam->image_name());
            metadata.frame_camera_ids.push_back(cam->camera_id());
        }

        if (static_cast<int>(metadata.frame_image_names.size()) != ppisp_->num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "PPISP metadata frame mismatch: metadata has {} names / {} camera ids but PPISP has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "PPISP metadata camera mismatch: metadata has {} camera ids but PPISP has {} cameras",
                metadata.camera_ids.size(),
                ppisp_->num_cameras()));
        }

        return metadata;
    }

    std::expected<Trainer::PPISPSidecarMappings, std::string> Trainer::build_ppisp_sidecar_mappings(
        const PPISP& loaded_ppisp,
        const PPISPFileMetadata& metadata,
        const std::filesystem::path& sidecar_path) const {

        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot apply PPISP sidecar without an active training dataset");
        }
        if (metadata.empty()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar '{}' has no dataset metadata. Older sidecars cannot be verified against the current dataset; resave the source model with sidecar metadata first.",
                lfs::core::path_to_utf8(sidecar_path)));
        }
        if (static_cast<int>(metadata.frame_image_names.size()) != loaded_ppisp.num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != loaded_ppisp.num_frames()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata frame count mismatch: metadata has {} names / {} camera ids but sidecar has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                loaded_ppisp.num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != loaded_ppisp.num_cameras()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata camera count mismatch: metadata has {} camera ids but sidecar has {} cameras",
                metadata.camera_ids.size(),
                loaded_ppisp.num_cameras()));
        }

        const auto current_dataset_path = lfs::core::path_to_utf8(params_.dataset.data_path);
        if (!metadata.dataset_path_utf8.empty() && metadata.dataset_path_utf8 != current_dataset_path) {
            LOG_INFO("Frozen PPISP sidecar dataset path differs from current dataset path: '{}' vs '{}'",
                     metadata.dataset_path_utf8, current_dataset_path);
        }
        if (!metadata.images_folder.empty() && metadata.images_folder != params_.dataset.images) {
            LOG_INFO("Frozen PPISP sidecar images folder differs from current training config: '{}' vs '{}'",
                     metadata.images_folder, params_.dataset.images);
        }

        auto make_frame_key = [](std::string_view image_name, int camera_id) {
            return std::format("{}\n{}", image_name, camera_id);
        };

        std::unordered_map<std::string, int> source_frame_index_by_key;
        source_frame_index_by_key.reserve(metadata.frame_image_names.size());
        for (size_t i = 0; i < metadata.frame_image_names.size(); ++i) {
            auto [_, inserted] = source_frame_index_by_key.emplace(
                make_frame_key(metadata.frame_image_names[i], metadata.frame_camera_ids[i]),
                static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate frame key for image '{}' and camera {}",
                    metadata.frame_image_names[i],
                    metadata.frame_camera_ids[i]));
            }
        }

        PPISPSidecarMappings mappings;
        mappings.frame_mapping.reserve(static_cast<size_t>(ppisp_->num_frames()));
        std::unordered_set<std::string> seen_target_frames;
        seen_target_frames.reserve(static_cast<size_t>(ppisp_->num_frames()));
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            const auto key = make_frame_key(cam->image_name(), cam->camera_id());
            if (!seen_target_frames.insert(key).second) {
                return std::unexpected(std::format(
                    "Current training dataset contains duplicate frame key for image '{}' and camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            const auto it = source_frame_index_by_key.find(key);
            if (it == source_frame_index_by_key.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing frame '{}' for camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            mappings.frame_mapping.push_back(it->second);
        }
        if (seen_target_frames.size() != source_frame_index_by_key.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} frame keys but current training dataset has {}",
                source_frame_index_by_key.size(),
                seen_target_frames.size()));
        }

        std::unordered_map<int, int> source_camera_index_by_id;
        source_camera_index_by_id.reserve(metadata.camera_ids.size());
        for (size_t i = 0; i < metadata.camera_ids.size(); ++i) {
            auto [_, inserted] = source_camera_index_by_id.emplace(metadata.camera_ids[i], static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate camera id {}",
                    metadata.camera_ids[i]));
            }
        }

        const auto target_camera_ids = ppisp_->ordered_camera_ids();
        mappings.camera_mapping.reserve(target_camera_ids.size());
        for (const int camera_id : target_camera_ids) {
            const auto it = source_camera_index_by_id.find(camera_id);
            if (it == source_camera_index_by_id.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing camera id {} required by the current dataset",
                    camera_id));
            }
            mappings.camera_mapping.push_back(it->second);
        }
        if (target_camera_ids.size() != source_camera_index_by_id.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} camera ids but current training dataset uses {}",
                source_camera_index_by_id.size(),
                target_camera_ids.size()));
        }

        return mappings;
    }

    std::expected<void, std::string> Trainer::apply_ppisp_sidecar_if_configured() {
        if (!should_apply_ppisp_sidecar_on_init()) {
            return {};
        }
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }

        PPISP loaded_ppisp(1);
        PPISPFileMetadata metadata;
        const auto sidecar_path = params_.optimization.ppisp_sidecar_path;

        if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, nullptr, &metadata); !result) {
            return std::unexpected(std::format(
                "Failed to load frozen PPISP sidecar '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
        if (!mappings_result) {
            return std::unexpected(mappings_result.error());
        }
        auto& mappings = *mappings_result;

        if (static_cast<int>(mappings.frame_mapping.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar frame mapping size mismatch: {} mappings for {} target frames",
                mappings.frame_mapping.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(mappings.camera_mapping.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar camera mapping size mismatch: {} mappings for {} target cameras",
                mappings.camera_mapping.size(),
                ppisp_->num_cameras()));
        }

        if (auto result = ppisp_->copy_inference_weights_from(
                loaded_ppisp, mappings.frame_mapping, mappings.camera_mapping);
            !result) {
            return std::unexpected(std::format(
                "Failed to import frozen PPISP weights from '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        LOG_INFO("Loaded frozen PPISP sidecar '{}' ({} cameras, {} frames{})",
                 lfs::core::path_to_utf8(sidecar_path),
                 loaded_ppisp.num_cameras(),
                 loaded_ppisp.num_frames(),
                 ", metadata-mapped");
        return {};
    }

    std::expected<void, std::string> Trainer::initialize_ppisp_controller() {
        if (!params_.optimization.ppisp_use_controller || !params_.optimization.use_ppisp) {
            return {};
        }

        if (!ppisp_) {
            return std::unexpected("PPISP must be initialized before controller");
        }

        try {
            const bool import_frozen_sidecar_controller = should_apply_ppisp_sidecar_on_init();
            const auto sidecar_path = params_.optimization.ppisp_sidecar_path;
            PPISPFileHeader sidecar_header{};
            if (import_frozen_sidecar_controller) {
                std::ifstream file;
                if (!lfs::core::open_file_for_read(sidecar_path, std::ios::binary, file)) {
                    return std::unexpected("Failed to open frozen PPISP sidecar: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                file.read(reinterpret_cast<char*>(&sidecar_header), sizeof(sidecar_header));
                if (!file) {
                    return std::unexpected("Failed to read frozen PPISP sidecar header: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                if (sidecar_header.magic != PPISP_FILE_MAGIC) {
                    return std::unexpected("Invalid frozen PPISP sidecar: wrong magic number");
                }
                if (sidecar_header.version > PPISP_FILE_VERSION) {
                    return std::unexpected("Unsupported frozen PPISP sidecar version: " +
                                           std::to_string(sidecar_header.version));
                }
                if (!has_flag(sidecar_header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                    LOG_INFO("Frozen PPISP sidecar '{}' has no controller pool; controller inference will remain disabled",
                             lfs::core::path_to_utf8(sidecar_path));
                    return {};
                }
            }

            PPISPControllerPool::Config config;
            config.lr = params_.optimization.ppisp_controller_lr;

            const int activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step = activation_step;
            }
            int distillation_iters = static_cast<int>(params_.optimization.iterations) - activation_step;
            int num_cameras = ppisp_->num_cameras();

            ppisp_controller_pool_ = std::make_unique<PPISPControllerPool>(num_cameras, distillation_iters, config);

            size_t max_h = 0, max_w = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    max_h = std::max(max_h, static_cast<size_t>(cam->image_height()));
                    max_w = std::max(max_w, static_cast<size_t>(cam->image_width()));
                }
            }
            ppisp_controller_pool_->allocate_buffers(max_h, max_w);

            LOG_INFO("PPISP controller pool initialized: num_cameras={}, activation_step={}, lr={:.2e}, max_image={}x{}",
                     num_cameras, activation_step,
                     params_.optimization.ppisp_controller_lr, static_cast<int>(max_h), static_cast<int>(max_w));

            if (import_frozen_sidecar_controller) {
                PPISP loaded_ppisp(1);
                auto loaded_controller = std::make_unique<PPISPControllerPool>(
                    static_cast<int>(sidecar_header.num_cameras),
                    1);
                PPISPFileMetadata metadata;
                if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, loaded_controller.get(), &metadata); !result) {
                    return std::unexpected(std::format(
                        "Failed to load frozen PPISP controller sidecar '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        result.error()));
                }
                auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
                if (!mappings_result) {
                    return std::unexpected(mappings_result.error());
                }
                if (const auto error = ppisp_controller_pool_->copy_inference_weights_from(
                        *loaded_controller, mappings_result->camera_mapping);
                    !error.empty()) {
                    return std::unexpected(std::format(
                        "Failed to import frozen PPISP controller weights from '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        error));
                }
                LOG_INFO("Loaded frozen PPISP controller from '{}' ({} cameras)",
                         lfs::core::path_to_utf8(sidecar_path),
                         loaded_controller->num_cameras());
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP controller pool: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss_.forward(rendered, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return std::make_pair(loss_tensor, ctx.grad_image);
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        size_t alpha_count = 0;
        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_alpha())
                ++alpha_count;
            if (cam && cam->has_mask())
                ++masks_found;
        }

        if (opt.use_alpha_as_mask && alpha_count > 0) {
            LOG_INFO("Using alpha channel as mask source ({}/{} cameras){}",
                     alpha_count, train_dataset_->get_cameras().size(),
                     opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        if (masks_found == 0) {
            const auto path_str = lfs::core::path_to_utf8(params_.dataset.data_path);
            if (opt.use_alpha_as_mask) {
                return std::unexpected(std::format(
                    "Mask mode enabled with use_alpha_as_mask but no images have alpha and no mask files found in {}/masks/",
                    path_str));
            }
            return std::unexpected(std::format(
                "Mask mode enabled but no masks found in {}/masks/",
                path_str));
        }

        LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
        return {};
    }

    std::expected<void, std::string> Trainer::generate_project_mesh_masks() {
        if (params_.optimization.mask_mode != lfs::core::param::MaskMode::ProjectMesh) {
            return {};
        }
        if (!train_dataset_) {
            return std::unexpected("project_mesh mask mode requires a training dataset");
        }
        if (!scene_) {
            return std::unexpected("project_mesh mask mode requires a Scene with a visible mesh");
        }

        const auto project_mask_source = select_project_mask_mesh_source(*scene_);
        if (!project_mask_source) {
            return std::unexpected(
                "project_mesh mask mode requires a visible mesh or a mesh-init hole-fill override");
        }
        auto prepared_result = project_mask_source.mesh_override
                                   ? prepare_mesh_geometry(*project_mask_source.mesh_override)
                                   : prepare_mesh_geometry(*scene_);
        if (!prepared_result) {
            return std::unexpected(std::format(
                "project_mesh mask mode: {}", prepared_result.error()));
        }
        const PreparedMesh& prepared_mesh = *prepared_result;
        if (project_mask_source.mesh_override) {
            LOG_INFO("project_mesh mask mode: using dedicated mesh override ({} vertices, {} faces)",
                     project_mask_source.mesh_override->vertex_count(),
                     project_mask_source.mesh_override->face_count());
        }

        const std::filesystem::path mask_dir = params_.dataset.data_path / "masks";
        std::error_code mkdir_ec;
        std::filesystem::create_directories(mask_dir, mkdir_ec);
        if (mkdir_ec) {
            return std::unexpected(std::format(
                "project_mesh mask mode: failed to create '{}': {}",
                lfs::core::path_to_utf8(mask_dir), mkdir_ec.message()));
        }

        // external_pointcloud mode only. The mesh still supplies coverage; this cloud is
        // what authorizes filling the silhouette-interior holes the mesh is missing. Without
        // it those holes stay masked out, the Gaussians seeded there never receive
        // photometric gradient, and their white initialization is never corrected.
        std::vector<glm::vec3> hole_fill_points_world;
        ProjectMaskHoleFillOptions hole_fill_options;
        if (project_mask_source.point_cloud_override) {
            const auto& cloud = *project_mask_source.point_cloud_override;
            auto means = cloud.means.to(lfs::core::Device::CPU)
                             .to(lfs::core::DataType::Float32)
                             .contiguous();
            if (means.ndim() != 2 || means.shape()[1] != 3) {
                return std::unexpected(
                    "project_mesh mask mode: external point-cloud override must hold [N,3] positions");
            }
            const auto point_count = static_cast<size_t>(means.shape()[0]);
            const float* means_ptr = means.ptr<float>();
            hole_fill_points_world.reserve(point_count);
            for (size_t i = 0; i < point_count; ++i) {
                hole_fill_points_world.emplace_back(
                    means_ptr[i * 3 + 0], means_ptr[i * 3 + 1], means_ptr[i * 3 + 2]);
            }

            const auto& opt = params_.optimization;
            hole_fill_options.min_points_per_hole = opt.mesh2splat_pointcloud_mask_min_points_per_hole;
            hole_fill_options.rim_close_pixels = opt.mesh2splat_pointcloud_mask_rim_close_pixels;
            hole_fill_options.splat_radius_scale = opt.mesh2splat_pointcloud_mask_splat_radius_scale;
            hole_fill_options.close_pixels = opt.mesh2splat_pointcloud_mask_close_pixels;
            hole_fill_options.point_spacing = project_mask_source.point_cloud_spacing;

            LOG_INFO("project_mesh mask mode: external point-cloud hole fill active "
                     "({} points, spacing={:.6g}, min_points_per_hole={}, rim_close={}, "
                     "splat_radius_scale={:.4g}, close_pixels={})",
                     hole_fill_points_world.size(),
                     hole_fill_options.point_spacing,
                     hole_fill_options.min_points_per_hole,
                     hole_fill_options.rim_close_pixels,
                     hole_fill_options.splat_radius_scale,
                     hole_fill_options.close_pixels);

            if (!(hole_fill_options.point_spacing > 0.0f) &&
                hole_fill_options.splat_radius_scale > 0.0f) {
                LOG_WARN("project_mesh mask mode: point spacing is zero, so the boundary-open "
                         "hole fallback is inactive; interior holes are still filled");
            }
        }

        // Collect all cameras (train + val) so masks exist for both splits.
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras = train_dataset_->get_cameras();
        if (val_dataset_) {
            for (const auto& cam : val_dataset_->get_cameras()) {
                if (!cam) continue;
                auto exists = std::find_if(cameras.begin(), cameras.end(),
                    [&](const auto& c) { return c && c->uid() == cam->uid(); }) != cameras.end();
                if (!exists) cameras.push_back(cam);
            }
        }

        // Render at the original full image resolution so the saved PNG matches
        // the source image dimensions, which the existing mask validation enforces.
        // load_image_size(1, 0) reads the underlying image dimensions when no
        // undistortion override is active (true at this point in initialize()).
        const int project_mesh_mask_erode_pixels =
            std::max(0, params_.optimization.project_mesh_mask_erode_pixels);
        size_t saved = 0;
        ProjectMaskHoleFillStats total_hole_fill_stats;
        size_t cameras_with_fill = 0;
        bool fallback_radius_capped = false;
        for (auto& cam_ptr : cameras) {
            if (!cam_ptr) continue;
            auto& cam = *cam_ptr;

            cam.load_image_size(1, 0);
            const int width = cam.image_width();
            const int height = cam.image_height();
            if (width <= 0 || height <= 0) {
                return std::unexpected(std::format(
                    "project_mesh mask mode: invalid image size for '{}'", cam.image_name()));
            }

            // Use a temporary params snapshot with resize_factor=1/max_width=0 so
            // the renderer's internal load_image_size keeps full-res dimensions.
            lfs::core::param::TrainingParameters fullres_params = params_;
            fullres_params.dataset.resize_factor = 1;
            fullres_params.dataset.max_width = 0;

            auto render_result = render_mesh_supervision_targets_for_camera(
                prepared_mesh, cam, fullres_params, false);
            if (!render_result) {
                return std::unexpected(std::format(
                    "project_mesh mask mode: render failed for '{}': {}",
                    cam.image_name(), render_result.error()));
            }

            lfs::core::Tensor mask_image;
            if (!hole_fill_points_world.empty()) {
                // external_pointcloud mode. Coverage is assembled on the CPU because the
                // interior-hole decision needs connected-component labeling; this runs once
                // per camera during initialize(), so the cost is irrelevant.
                auto depth_cpu = render_result->depth.to(lfs::core::Device::CPU)
                                     .to(lfs::core::DataType::Float32)
                                     .contiguous();
                if (depth_cpu.numel() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
                    return std::unexpected(std::format(
                        "project_mesh mask mode: depth size mismatch for '{}'", cam.image_name()));
                }

                auto w2c = cam.world_view_transform()
                               .squeeze(0)
                               .to(lfs::core::Device::CPU)
                               .to(lfs::core::DataType::Float32)
                               .contiguous();
                if (w2c.numel() < size_t{16}) {
                    return std::unexpected(std::format(
                        "project_mesh mask mode: missing camera transform for '{}'", cam.image_name()));
                }
                const float* m = w2c.ptr<float>();

                // Same indexing as mesh_transform_vertices_normals_kernel, so projected
                // points land in exactly the pixels the depth rasterizer wrote.
                std::vector<glm::vec3> camera_space_points;
                camera_space_points.reserve(hole_fill_points_world.size());
                for (const auto& p : hole_fill_points_world) {
                    camera_space_points.emplace_back(
                        m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                        m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                        m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
                }

                const auto [fx, fy, cx, cy] = cam.get_intrinsics();
                ProjectMaskHoleFillStats stats;
                auto coverage = build_project_mesh_coverage(
                    depth_cpu.ptr<float>(),
                    width,
                    height,
                    camera_space_points,
                    PinholeIntrinsics{.fx = fx, .fy = fy, .cx = cx, .cy = cy},
                    hole_fill_options,
                    stats);

                // Same square structuring element and same out-of-bounds-is-unset rule as
                // launch_project_mesh_depth_mask_erode, so the erosion semantics are
                // unchanged relative to the other hole-fill modes.
                coverage = erode_binary_mask(coverage, project_mesh_mask_erode_pixels);

                auto mask_cpu = lfs::core::Tensor::empty(
                    {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CPU,
                    lfs::core::DataType::Float32);
                float* mask_ptr = mask_cpu.ptr<float>();
                for (size_t i = 0; i < coverage.values.size(); ++i) {
                    mask_ptr[i] = coverage.values[i] != 0 ? 1.0f : 0.0f;
                }
                mask_image = std::move(mask_cpu);

                total_hole_fill_stats.points_in_frustum += stats.points_in_frustum;
                total_hole_fill_stats.points_accepted += stats.points_accepted;
                total_hole_fill_stats.enclosed_holes += stats.enclosed_holes;
                total_hole_fill_stats.filled_holes += stats.filled_holes;
                total_hole_fill_stats.filled_pixels += stats.filled_pixels;
                total_hole_fill_stats.fallback_pixels += stats.fallback_pixels;
                fallback_radius_capped = fallback_radius_capped || stats.fallback_radius_capped;
                if (stats.filled_holes > 0 || stats.fallback_pixels > 0) {
                    ++cameras_with_fill;
                }
                LOG_DEBUG("project_mesh mask '{}': {}/{} points accepted, {} enclosed holes, "
                          "{} filled ({} px), fallback {} px (close radius {})",
                          cam.image_name(),
                          stats.points_accepted, stats.points_in_frustum,
                          stats.enclosed_holes, stats.filled_holes, stats.filled_pixels,
                          stats.fallback_pixels, stats.fallback_close_radius);
            } else {
                // Build coverage mask from depth on CUDA: pixels with depth > 0 are covered by mesh.
                // depth shape is [1, H, W]; produce a [1, H, W] float mask in {0, 1}.
                auto depth_cuda = render_result->depth.to(lfs::core::Device::CUDA)
                                      .to(lfs::core::DataType::Float32)
                                      .contiguous();
                mask_image = lfs::core::Tensor::empty(
                    {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                mask_image.set_stream(depth_cuda.stream());

                lfs::core::Tensor temp_mask_cuda;
                float* temp_mask_ptr = nullptr;
                if (project_mesh_mask_erode_pixels > 0) {
                    temp_mask_cuda = lfs::core::Tensor::empty(
                        {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                        lfs::core::Device::CUDA,
                        lfs::core::DataType::Float32);
                    temp_mask_cuda.set_stream(depth_cuda.stream());
                    temp_mask_ptr = temp_mask_cuda.ptr<float>();
                }
                kernels::launch_project_mesh_depth_mask_erode(
                    depth_cuda.ptr<float>(),
                    temp_mask_ptr,
                    mask_image.ptr<float>(),
                    width,
                    height,
                    project_mesh_mask_erode_pixels,
                    depth_cuda.stream());
            }

            // Mirror the original image filename under masks/, replacing the extension
            // with .png so MaskDirCache resolves it via its image-stem lookup keys.
            const std::filesystem::path image_rel = lfs::core::utf8_to_path(cam.image_name());
            std::filesystem::path mask_rel = image_rel;
            mask_rel.replace_extension(".png");
            const std::filesystem::path out_path = mask_dir / mask_rel;
            std::error_code parent_ec;
            std::filesystem::create_directories(out_path.parent_path(), parent_ec);
            if (parent_ec) {
                return std::unexpected(std::format(
                    "project_mesh mask mode: failed to create '{}': {}",
                    lfs::core::path_to_utf8(out_path.parent_path()), parent_ec.message()));
            }

            try {
                lfs::core::save_image(out_path, mask_image);
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "project_mesh mask mode: failed to save '{}': {}",
                    lfs::core::path_to_utf8(out_path), e.what()));
            }

            cam.set_mask_path(out_path);
            ++saved;
        }

        LOG_INFO("project_mesh mask mode: rendered and saved {} masks under '{}'",
                 saved, lfs::core::path_to_utf8(mask_dir));

        if (!hole_fill_points_world.empty()) {
            LOG_INFO("project_mesh mask mode: hole fill touched {}/{} cameras — "
                     "{} points accepted of {} in frustum, {} enclosed holes seen, {} filled "
                     "({} px), boundary-open fallback added {} px",
                     cameras_with_fill, saved,
                     total_hole_fill_stats.points_accepted,
                     total_hole_fill_stats.points_in_frustum,
                     total_hole_fill_stats.enclosed_holes,
                     total_hole_fill_stats.filled_holes,
                     total_hole_fill_stats.filled_pixels,
                     total_hole_fill_stats.fallback_pixels);
            if (cameras_with_fill == 0) {
                LOG_WARN("project_mesh mask mode: no hole was filled in any view. The point "
                         "Gaussians will stay white. Check that the cloud covers regions the "
                         "mesh is missing, and that those regions read as enclosed holes "
                         "(raise --mesh2splat-pointcloud-mask-rim-close-pixels if the mesh "
                         "coverage is ragged at the hole rims)");
            }
            if (fallback_radius_capped) {
                LOG_WARN("project_mesh mask mode: the auto-derived fallback closing radius hit "
                         "its cap in at least one view; set --mesh2splat-pointcloud-mask-close-pixels "
                         "explicitly, or 0 out --mesh2splat-pointcloud-mask-splat-radius-scale, if "
                         "the saved masks look over-sealed");
            }
        }
        return {};
    }

    std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        using namespace lfs::core;
        constexpr float EPSILON = 1e-8f;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        Tensor loss, grad, grad_alpha;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore || mode == param::MaskMode::ProjectMesh) {
            if (opt_params.lambda_dssim > 0.0f) {
                // Use FUSED masked L1+SSIM kernel
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, mask_2d, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                // Squeeze gradient to match input dimensions (loss is scalar, no adjustment needed)
                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with mask (no SSIM)
                const Tensor mask_3d = mask_2d.unsqueeze(0);
                const Tensor mask_sum = mask_2d.sum() * static_cast<float>(rendered.shape()[0]) + EPSILON;
                const Tensor diff = rendered - gt_image;
                const Tensor masked_l1 = (diff.abs() * mask_3d).sum() / mask_sum;
                const Tensor sign_diff = diff.sign();
                grad = sign_diff * mask_3d / mask_sum;
                loss = masked_l1;
            }

            // Segment / ProjectMesh: opacity penalty for background.
            if ((mode == param::MaskMode::Segment || mode == param::MaskMode::ProjectMesh) && alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            const lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto result = photometric_loss_.forward(rendered, gt_image, params);
            if (!result) {
                return std::unexpected(result.error());
            }
            auto [photo_loss, ctx] = *result;
            loss = photo_loss;
            grad = ctx.grad_image;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor alpha_loss = (alpha_2d - mask_2d).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_2d).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }
        } else {
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        return MaskLossResult{.loss = loss, .grad_image = grad, .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<lfs::core::Tensor, std::string> Trainer::compute_mesh_surface_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        MeshSurfaceState& mesh_state,
        const lfs::core::Tensor* visible_mask,
        const lfs::core::Tensor* visible_count,
        const lfs::core::param::OptimizationParameters& opt_params) {
        if (!mesh_state.is_valid()) {
            return std::unexpected("Mesh surface loss requires constraint mesh vertices, indices, edge neighbors, and current face cache");
        }

        float rho = 0.0f;
        if (opt_params.mesh_scale_rho_ratio > 0.0f) {
            rho = std::max(mesh_state.scene_radius, 1e-5f) * opt_params.mesh_scale_rho_ratio;
        } else if (opt_params.mesh_scale_rho_avg_max_scale_multiplier > 0.0f) {
            const float mean_max_scale = splatData.get_mesh2splat_mean_max_scale();
            if (!(mean_max_scale > 0.0f) || !std::isfinite(mean_max_scale)) {
                return std::unexpected("mesh_scale_rho_avg_max_scale_multiplier requires mesh2splat mean max scale metadata; use mesh_scale_rho_ratio > 0 or initialize from mesh2splat");
            }
            rho = mean_max_scale * opt_params.mesh_scale_rho_avg_max_scale_multiplier;
        }
        if (opt_params.lambda_mesh_scale_max > 0.0f &&
            (!(rho > 0.0f) || !std::isfinite(rho))) {
            return std::unexpected("lambda_mesh_scale_max requires a positive finite rho");
        }

        float outside_distance_threshold = 0.0f;
        if (opt_params.lambda_mesh_outside_barrier > 0.0f) {
            const float mean_max_scale = splatData.get_mesh2splat_mean_max_scale();
            if (!(mean_max_scale > 0.0f) || !std::isfinite(mean_max_scale)) {
                return std::unexpected("lambda_mesh_outside_barrier requires positive finite mesh2splat mean max scale metadata");
            }
            outside_distance_threshold =
                mean_max_scale * opt_params.mesh_outside_distance_avg_max_scale_multiplier;
            if (!(outside_distance_threshold > 0.0f) || !std::isfinite(outside_distance_threshold)) {
                return std::unexpected("lambda_mesh_outside_barrier requires a positive finite outside distance threshold");
            }
        }
        auto inside_distance_result = resolve_mesh_inside_constraint_distance(opt_params, splatData);
        if (!inside_distance_result) {
            return std::unexpected(inside_distance_result.error());
        }
        losses::MeshSurfaceRegularization::Params params;
        params.lambda_project = opt_params.lambda_mesh_project;
        params.lambda_outside_barrier = opt_params.lambda_mesh_outside_barrier;
        params.outside_distance_threshold = outside_distance_threshold;
        params.inside_constraint_distance = *inside_distance_result;
        params.inside_constraint_fade_ratio = opt_params.mesh_inside_constraint_fade_ratio;
        params.lambda_scale_min = opt_params.lambda_mesh_scale_min;
        params.lambda_scale_max = opt_params.lambda_mesh_scale_max;
        params.rho = rho;
        params.lambda_normal = opt_params.lambda_mesh_normal;
        params.walk_steps = std::max(1, mesh_state.walk_steps);

        return losses::MeshSurfaceRegularization::forward(
            splatData.means(),
            splatData.scaling_raw(),
            splatData.rotation_raw(),
            optimizer.get_grad(ParamType::Means),
            optimizer.get_grad(ParamType::Scaling),
            optimizer.get_grad(ParamType::Rotation),
            visible_mask,
            visible_count,
            *mesh_state.current_faces,
            *mesh_state.tri_edge_neighbors,
            *mesh_state.constraint_mesh_verts,
            *mesh_state.constraint_mesh_indices,
            mesh_state.constraint_face_normals,
            params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    Trainer::Trainer(lfs::core::Scene& scene)
        : scene_(&scene) {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        if (!scene.hasTrainingData()) {
            throw std::runtime_error("Scene has no cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;
            if (params_.optimization.precompute_pseudo_view) {
                params_.optimization.precompute_mesh_depth_normal = true;
            }
            if (mesh_surface_soft_constraint_enabled(params_.optimization) ||
                params_.optimization.mesh_depth_visibility_cull) {
                params_.optimization.precompute_mesh_depth_normal = true;
            }
            mesh_supervision_cache_dir_.clear();
            mesh_supervision_cache_dirty_.store(false);

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            total_cameras_count_ = source_cameras.size();

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> explicit_splits;
                if (has_explicit_eval_split(params.optimization)) {
                    explicit_splits = std::make_tuple(
                        params.optimization.train_images,
                        params.optimization.test_images);
                    LOG_INFO("Using explicit eval split from config: {} train, {} val entries",
                             params.optimization.train_images.size(),
                             params.optimization.test_images.size());
                }

                const auto* active_splits = explicit_splits ? &*explicit_splits :
                                             (provided_splits_ ? &*provided_splits_ : nullptr);

                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                    active_splits ? std::make_optional(std::get<0>(*active_splits)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::VAL,
                    active_splits ? std::make_optional(std::get<1>(*active_splits)) : std::nullopt);

                if (explicit_splits && (train_dataset_->size() == 0 || val_dataset_->size() == 0)) {
                    return std::unexpected(std::format(
                        "Explicit eval split matched {} train and {} val cameras; check train_images/test_images names",
                        train_dataset_->size(),
                        val_dataset_->size()));
                }

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size(),
                         val_dataset_->size());
            } else {
                // Use all images for training
                // 所有图像训练跑这里
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                auto result = StrategyFactory::instance().create(params_.optimization.strategy, *model);
                if (!result) {
                    return std::unexpected(result.error());
                }
                strategy_ = std::move(*result);
                LOG_DEBUG("Created {} strategy from Scene model", params_.optimization.strategy);
            }

            auto& splat = strategy_->get_model();
            const bool inside_band_enabled =
                params_.optimization.mesh_inside_constraint_distance_avg_max_scale_multiplier > 0.0f;

            if (inside_band_enabled) {
                auto distance_result = resolve_mesh_inside_constraint_distance(params_.optimization, splat);
                if (!distance_result) {
                    return std::unexpected(distance_result.error());
                }
                if (!splat.has_constraint_mesh()) {
                    return std::unexpected("Mesh inside constraint band requires a mesh2splat constraint mesh");
                }
                const auto diagnostics = log_mesh_constraint_topology_diagnostics(
                    splat.constraint_mesh_verts(), splat.constraint_mesh_indices());
                const float fade_width = *distance_result * params_.optimization.mesh_inside_constraint_fade_ratio;
                LOG_INFO(
                    "Mesh inside constraint band enabled: distance={:.6g}, full_width={:.6g}, fade_width={:.6g}, "
                    "mean_max_scale={:.6g}, multiplier={:.6g}, components={}",
                    *distance_result,
                    *distance_result - fade_width,
                    fade_width,
                    splat.get_mesh2splat_mean_max_scale(),
                    params_.optimization.mesh_inside_constraint_distance_avg_max_scale_multiplier,
                    diagnostics.connected_components);
            }

            if (params_.optimization.lambda_mesh_outside_barrier > 0.0f) {
                const float mean_max_scale = splat.get_mesh2splat_mean_max_scale();
                if (!(mean_max_scale > 0.0f) || !std::isfinite(mean_max_scale)) {
                    return std::unexpected("lambda_mesh_outside_barrier requires positive finite mesh2splat mean max scale metadata");
                }
                const float threshold = mean_max_scale *
                                        params_.optimization.mesh_outside_distance_avg_max_scale_multiplier;
                if (!(threshold > 0.0f) || !std::isfinite(threshold)) {
                    return std::unexpected("lambda_mesh_outside_barrier requires a positive finite outside distance threshold");
                }
                if (!splat.has_constraint_mesh()) {
                    return std::unexpected("lambda_mesh_outside_barrier requires a mesh2splat constraint mesh");
                }
                if (!inside_band_enabled) {
                    (void)log_mesh_constraint_topology_diagnostics(
                        splat.constraint_mesh_verts(),
                        splat.constraint_mesh_indices());
                }
                LOG_INFO("Mesh outside barrier enabled: lambda={:.6g}, threshold={:.6g} (mean_max_scale={:.6g} x multiplier={:.6g})",
                         params_.optimization.lambda_mesh_outside_barrier,
                         threshold,
                         mean_max_scale,
                         params_.optimization.mesh_outside_distance_avg_max_scale_multiplier);
            }

            auto max_cap_result = resolve_mesh2splat_max_cap(params_.optimization, splat);
            if (!max_cap_result) {
                return std::unexpected(max_cap_result.error());
            }
            params_.optimization.max_cap = *max_cap_result;

            int max_cap = params_.optimization.max_cap;
            if (max_cap > 0 && static_cast<unsigned long>(max_cap) < splat.size()) {
                LOG_WARN("Max cap ({}) is less than initial splats ({}), randomly selecting {} splats", max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
            }

            // Re-initialize strategy with new parameters
            strategy_->set_training_dataset(train_dataset_);
            strategy_->initialize(params_.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP if enabled
            if (auto result = initialize_ppisp(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP controller if enabled
            if (auto result = initialize_ppisp_controller(); !result) {
                return std::unexpected(result.error());
            }

            // Generate per-camera masks by projecting the visible mesh when
            // the project_mesh mode is selected. Must run before validate_masks
            // and before undistortion preparation so cameras render at full
            // image resolution and the saved PNGs match the source images.
            if (auto result = generate_project_mesh_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }



            // Apply undistortion to camera intrinsics (params already precomputed at load time)
            if (params.optimization.undistort) {
                int prepared = 0;
                for (auto& cam : train_dataset_->get_cameras()) {
                    if (cam && cam->has_distortion()) {
                        cam->prepare_undistortion();
                        ++prepared;
                    }
                }
                if (val_dataset_) {
                    for (auto& cam : val_dataset_->get_cameras()) {
                        if (cam && cam->has_distortion()) {
                            cam->prepare_undistortion();
                        }
                    }
                }
                if (prepared > 0) {
                    LOG_INFO("Prepared undistortion for {} cameras", prepared);
                }
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int sparsify_steps = params.optimization.sparsify_steps;
                const int stored_iters = static_cast<int>(params.optimization.iterations);

                // Checkpoint already has total iterations; fresh start needs sparsify_steps added
                const bool is_resume = params.resume_checkpoint.has_value();
                const int base_iters = is_resume ? (stored_iters - sparsify_steps) : stored_iters;

                if (!is_resume) {
                    params_.optimization.iterations = static_cast<size_t>(base_iters + sparsify_steps);
                }

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = base_iters};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: base={}, steps={}, prune={:.0f}%",
                             base_iters, sparsify_steps, params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Initialize image cache loader before any code path that calls getInstance()
            auto& cache_loader = lfs::io::CacheLoader::getInstance(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache);
            cache_loader.update_cache_params(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache,
                train_dataset_size_,
                params_.dataset.loading_params.min_cpu_free_GB,
                params_.dataset.loading_params.min_cpu_free_memory_ratio,
                params_.dataset.loading_params.print_cache_status,
                params_.dataset.loading_params.print_status_freq_num);

            if (params_.optimization.precompute_pseudo_view) {
                if (!scene_) {
                    return std::unexpected("Pseudo-view precompute requires Scene-backed training data");
                }
                std::vector<std::shared_ptr<lfs::core::Camera>> pseudo_train_cameras;
                pseudo_train_cameras.reserve(train_dataset_->size());
                for (size_t i = 0; i < train_dataset_->size(); ++i) {
                    pseudo_train_cameras.push_back(train_dataset_->get_camera_shared(i));
                }

                auto pseudo_result = [&]() {
                    ScopedPseudoViewCpuCacheBypass cache_bypass(
                        cache_loader,
                        params_.dataset,
                        static_cast<int>(train_dataset_size_));
                    return run_pseudo_view_precompute(
                        *scene_,
                        pseudo_train_cameras,
                        params_);
                }();
                if (!pseudo_result) {
                    return std::unexpected(pseudo_result.error());
                }

                // Establish a clean CUDA/memory-pool boundary before pseudo cameras are
                // loaded and the training rasterizer starts its first frame.  All
                // precompute-local tensors have been destroyed when the call above
                // returns, so trimming here can release their cached allocations.
                if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    return std::unexpected(
                        "Pseudo-view precompute CUDA synchronization failed: " +
                        std::string(cudaGetErrorString(err)));
                }
                lfs::core::Tensor::trim_memory_pool();
                if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
                    return std::unexpected(
                        "Pseudo-view CUDA pool cleanup failed: " +
                        std::string(cudaGetErrorString(err)));
                }

                const bool log_pseudo_diagnostics =
                    !params_.optimization.pseudo_view_minimal_metadata;
                if (log_pseudo_diagnostics) {
                    if (pseudo_result->pseudo_views_written == 0) {
                        LOG_WARN("Pseudo-view precompute completed with 0 usable pseudo views (manifest: {})",
                                 lfs::core::path_to_utf8(pseudo_result->manifest_path));
                    } else {
                        LOG_INFO("Pseudo-view precompute cache ready: {} views at {}",
                                 pseudo_result->pseudo_views_written,
                                 lfs::core::path_to_utf8(pseudo_result->output_dir));
                    }
                }

                // Inject pseudo views into the training set with the configured weight.
                if (params_.optimization.use_pseudo_views_in_training &&
                    pseudo_result->pseudo_views_written > 0) {
                    if (pseudo_train_cameras.empty()) {
                        if (log_pseudo_diagnostics) {
                            LOG_WARN("Cannot inject pseudo cameras: no real training cameras available");
                        }
                    } else {
                        const float weight = params_.optimization.pseudo_view_loss_weight;
                        auto pseudo_cams = load_pseudo_view_cameras(
                            pseudo_result->manifest_path,
                            *pseudo_train_cameras.front(),
                            weight,
                            -100000,
                            static_cast<size_t>(params_.optimization.pseudo_view_min_valid_pixels),
                            log_pseudo_diagnostics);
                        if (!pseudo_cams) {
                            if (log_pseudo_diagnostics) {
                                LOG_WARN("Failed to load pseudo-view cameras into training set: {}",
                                         pseudo_cams.error());
                            }
                        } else if (pseudo_cams->empty()) {
                            if (log_pseudo_diagnostics) {
                                LOG_WARN("Pseudo-view manifest yielded 0 loadable cameras; skipping injection");
                            }
                        } else {
                            const size_t added = pseudo_cams->size();
                            train_dataset_->append_cameras(std::move(*pseudo_cams));
                            train_dataset_size_ = train_dataset_->size();
                            if (log_pseudo_diagnostics) {
                                LOG_INFO("Pseudo-view training: injected {} pseudo cameras (loss weight {:.4f}); train split now {} cameras",
                                         added, weight, train_dataset_size_);
                            }
                        }
                    }
                }
            }

            // Resume initialization is deferred until the checkpoint's own
            // flags/parameters are inspected in Trainer::load_checkpoint().
            // This avoids allocating twice and keeps an old checkpoint's
            // disabled appearance mode authoritative over current CLI defaults.
            if (!params_.resume_checkpoint.has_value()) {
                if (auto result = initialize_per_frame_affine_color(); !result) {
                    return std::unexpected(result.error());
                }
                if (auto result = initialize_per_frame_observation_blur(); !result) {
                    return std::unexpected(result.error());
                }
            } else {
                per_frame_affine_color_.reset();
                per_frame_observation_blur_.reset();
            }

            pose_refiner_.reset();
            if (params_.optimization.refine_camera_pose && !params_.resume_checkpoint.has_value()) {
                if (params_.optimization.gut) {
                    return std::unexpected("Camera pose refinement currently supports only FastGS; disable --gut");
                }
                pose_refiner_ = std::make_unique<PoseRefiner>(params_.optimization);
                if (auto pose_result = pose_refiner_->initialize(train_dataset_->get_cameras()); !pose_result) {
                    return std::unexpected(pose_result.error());
                }
            }

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
#ifdef DEBUG_BUILD
            if (true/*params.optimization.headless*/) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }
#endif

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_, scene_);
            LOG_DEBUG("Metrics evaluator initialized");

            if (params_.optimization.precompute_mesh_depth_normal) {
                const std::string init_cache_tag =
                    params_.optimization.pyramid_training ? "level_fullres" : std::string{};
                if (auto result = refresh_mesh_supervision_cache(init_cache_tag);
                    !result) {
                    return std::unexpected(result.error());
                }
            }

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }

            // Print configuration
            // LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent", "project_mesh"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            // Expose initial snapshot for Python control (iteration 0)
            // {
            //     lfs::training::HookContext ctx{
            //         .iteration = current_iteration_.load(),
            //         .loss = current_loss_.load(),
            //         .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
            //         .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
            //         .trainer = this};
            //     lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
            //     lfs::training::CommandCenter::instance().update_snapshot(
            //         ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
            //         lfs::training::TrainingPhase::SafeControl);
            // }

            // // Execute configured Python scripts to register iteration callbacks
            // if (!python_scripts_.empty()) {
            //     auto py_result = lfs::python::run_scripts(python_scripts_);
            //     if (!py_result) {
            //         return std::unexpected(std::format("Failed to run Python scripts: {}", py_result.error()));
            //     }
            // }

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    std::shared_ptr<lfs::io::PipelinedImageLoader> Trainer::getActiveImageLoader() const {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        return active_image_loader_;
    }

    Trainer::GTLoadConfigSnapshot Trainer::getGTLoadConfigSnapshot() const {
        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        return gt_load_config_snapshot_;
    }

    void Trainer::enqueue_mesh_supervision_prefetch_locked(uint32_t camera_uid) {
        if (mesh_supervision_disk_targets_.find(camera_uid) == mesh_supervision_disk_targets_.end()) {
            return;
        }
        if (mesh_supervision_hot_targets_.find(camera_uid) != mesh_supervision_hot_targets_.end()) {
            return;
        }
        if (mesh_supervision_prefetch_pending_.insert(camera_uid).second) {
            mesh_supervision_prefetch_queue_.push_back(camera_uid);
            mesh_supervision_prefetch_cv_.notify_one();
        }
    }

    void Trainer::enqueue_mesh_supervision_prefetch_window_locked(uint32_t camera_uid) {
        enqueue_mesh_supervision_prefetch_locked(camera_uid);

        if (mesh_supervision_uid_order_.empty()) {
            return;
        }

        const auto order_it = mesh_supervision_uid_order_index_.find(camera_uid);
        if (order_it == mesh_supervision_uid_order_index_.end()) {
            return;
        }

        const size_t count = std::max<size_t>(1, mesh_supervision_hot_window_size_);
        const size_t base_idx = order_it->second;
        for (size_t offset = 1; offset < count; ++offset) {
            const size_t idx = (base_idx + offset) % mesh_supervision_uid_order_.size();
            enqueue_mesh_supervision_prefetch_locked(mesh_supervision_uid_order_[idx]);
        }
    }

    void Trainer::start_mesh_supervision_prefetch_thread() {
        std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
        if (mesh_supervision_prefetch_running_) {
            return;
        }
        mesh_supervision_prefetch_running_ = true;
        mesh_supervision_prefetch_thread_ = std::thread([this]() {
            mesh_supervision_prefetch_worker();
        });
    }

    void Trainer::stop_mesh_supervision_prefetch_thread() {
        std::thread worker;
        {
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            mesh_supervision_prefetch_running_ = false;
            mesh_supervision_prefetch_queue_.clear();
            mesh_supervision_prefetch_pending_.clear();
            mesh_supervision_prefetch_cv_.notify_all();
            if (mesh_supervision_prefetch_thread_.joinable()) {
                worker = std::move(mesh_supervision_prefetch_thread_);
            }
        }
        if (worker.joinable()) {
            worker.join();
        }
    }

    void Trainer::mesh_supervision_prefetch_worker() {
        while (true) {
            uint32_t camera_uid = 0;
            MeshSupervisionDiskCacheEntry disk_entry;

            {
                std::unique_lock<std::mutex> lock(mesh_supervision_mutex_);
                mesh_supervision_prefetch_cv_.wait(lock, [this]() {
                    return !mesh_supervision_prefetch_running_ || !mesh_supervision_prefetch_queue_.empty();
                });

                if (!mesh_supervision_prefetch_running_ && mesh_supervision_prefetch_queue_.empty()) {
                    break;
                }

                camera_uid = mesh_supervision_prefetch_queue_.front();
                mesh_supervision_prefetch_queue_.pop_front();
                mesh_supervision_prefetch_pending_.erase(camera_uid);

                if (mesh_supervision_hot_targets_.find(camera_uid) != mesh_supervision_hot_targets_.end()) {
                    auto hot_it = std::find(
                        mesh_supervision_hot_lru_.begin(),
                        mesh_supervision_hot_lru_.end(),
                        camera_uid);
                    if (hot_it != mesh_supervision_hot_lru_.end()) {
                        mesh_supervision_hot_lru_.erase(hot_it);
                    }
                    mesh_supervision_hot_lru_.push_back(camera_uid);
                    continue;
                }

                const auto disk_it = mesh_supervision_disk_targets_.find(camera_uid);
                if (disk_it == mesh_supervision_disk_targets_.end()) {
                    continue;
                }
                disk_entry = disk_it->second;
            }

            auto depth_bytes_result = read_binary_file(disk_entry.depth_path);
            if (!depth_bytes_result) {
                LOG_WARN("Mesh GT prefetch failed (depth): {}", depth_bytes_result.error());
                continue;
            }
            auto normal_bytes_result = read_binary_file(disk_entry.normal_path);
            if (!normal_bytes_result) {
                LOG_WARN("Mesh GT prefetch failed (normal): {}", normal_bytes_result.error());
                continue;
            }

            std::shared_ptr<std::vector<uint8_t>> mask_bytes;
            if (disk_entry.mask_path.has_value()) {
                auto mask_bytes_result = read_binary_file(*disk_entry.mask_path);
                if (!mask_bytes_result) {
                    LOG_WARN("Mesh GT prefetch failed (mask): {}", mask_bytes_result.error());
                    continue;
                }
                mask_bytes = std::make_shared<std::vector<uint8_t>>(std::move(*mask_bytes_result));
            }

            MeshSupervisionByteCacheEntry byte_entry;
            byte_entry.depth_bytes = std::make_shared<std::vector<uint8_t>>(std::move(*depth_bytes_result));
            byte_entry.normal_bytes = std::make_shared<std::vector<uint8_t>>(std::move(*normal_bytes_result));
            byte_entry.mask_bytes = std::move(mask_bytes);

            {
                std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
                if (!mesh_supervision_prefetch_running_) {
                    continue;
                }

                if (mesh_supervision_hot_targets_.find(camera_uid) == mesh_supervision_hot_targets_.end()) {
                    mesh_supervision_hot_targets_[camera_uid] = std::move(byte_entry);
                }

                auto hot_it = std::find(
                    mesh_supervision_hot_lru_.begin(),
                    mesh_supervision_hot_lru_.end(),
                    camera_uid);
                if (hot_it != mesh_supervision_hot_lru_.end()) {
                    mesh_supervision_hot_lru_.erase(hot_it);
                }
                mesh_supervision_hot_lru_.push_back(camera_uid);

                const size_t hot_window = std::max<size_t>(1, mesh_supervision_hot_window_size_);
                while (mesh_supervision_hot_targets_.size() > hot_window && !mesh_supervision_hot_lru_.empty()) {
                    const uint32_t evict_uid = mesh_supervision_hot_lru_.front();
                    mesh_supervision_hot_lru_.pop_front();
                    if (evict_uid == camera_uid) {
                        continue;
                    }
                    mesh_supervision_hot_targets_.erase(evict_uid);
                }
            }
        }
    }

    std::optional<Trainer::MeshSupervisionTargets> Trainer::get_mesh_supervision_targets(lfs::core::Camera& camera) {
        if (!params_.optimization.precompute_mesh_depth_normal || !scene_) {
            return std::nullopt;
        }

        if (mesh_supervision_cache_dirty_.exchange(false)) {
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            mesh_supervision_gpu_targets_.clear();
            mesh_supervision_gpu_lru_.clear();
            mesh_supervision_prepared_mesh_.reset();
        }

        const uint32_t camera_uid = static_cast<uint32_t>(camera.uid());

        auto touch_gpu_lru = [this](const uint32_t uid) {
            auto it = std::find(mesh_supervision_gpu_lru_.begin(), mesh_supervision_gpu_lru_.end(), uid);
            if (it != mesh_supervision_gpu_lru_.end()) {
                mesh_supervision_gpu_lru_.erase(it);
            }
            mesh_supervision_gpu_lru_.push_back(uid);
        };

        {
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            if (auto cached_it = mesh_supervision_gpu_targets_.find(camera_uid);
                cached_it != mesh_supervision_gpu_targets_.end()) {
                const auto& cached_depth = cached_it->second.depth;
                const bool shape_matches =
                    cached_depth.is_valid() &&
                    cached_depth.ndim() == 3 &&
                    static_cast<int>(cached_depth.shape()[1]) == camera.image_height() &&
                    static_cast<int>(cached_depth.shape()[2]) == camera.image_width();

                if (shape_matches) {
                    touch_gpu_lru(camera_uid);
                    return cached_it->second;
                }

                mesh_supervision_gpu_targets_.erase(cached_it);
                auto lru_it = std::find(mesh_supervision_gpu_lru_.begin(), mesh_supervision_gpu_lru_.end(), camera_uid);
                if (lru_it != mesh_supervision_gpu_lru_.end()) {
                    mesh_supervision_gpu_lru_.erase(lru_it);
                }
            }
        }

        // Prepare mesh geometry once and reuse across camera renders.
        if (!mesh_supervision_prepared_mesh_) {
            auto prepared_result = prepare_mesh_geometry(*scene_);
            if (!prepared_result) {
                LOG_WARN("Mesh geometry preparation failed: {}", prepared_result.error());
                return std::nullopt;
            }
            mesh_supervision_prepared_mesh_ = std::make_unique<PreparedMesh>(std::move(*prepared_result));
        }

        auto render_result = render_mesh_supervision_targets_for_camera(
            *mesh_supervision_prepared_mesh_,
            camera,
            params_);
        if (!render_result) {
            LOG_WARN("Mesh GT render failed for camera {}: {}", camera_uid, render_result.error());
            return std::nullopt;
        }

        MeshSupervisionTargets targets;
        targets.depth = std::move(render_result->depth);
        targets.normal = std::move(render_result->normal);
        targets.mask = std::move(render_result->mask);

        {
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            mesh_supervision_gpu_targets_[camera_uid] = targets;
            touch_gpu_lru(camera_uid);

            const size_t gpu_window = std::max<size_t>(1, mesh_supervision_gpu_window_size_);
            while (mesh_supervision_gpu_targets_.size() > gpu_window && !mesh_supervision_gpu_lru_.empty()) {
                const uint32_t evict_uid = mesh_supervision_gpu_lru_.front();
                mesh_supervision_gpu_lru_.pop_front();
                mesh_supervision_gpu_targets_.erase(evict_uid);
            }
        }

        return targets;
    }

    void Trainer::updateGTLoadConfigSnapshot() {
        GTLoadConfigSnapshot snapshot;
        if (train_dataset_) {
            snapshot.resize_factor = std::max(1, train_dataset_->get_resize_factor());
            snapshot.max_width = train_dataset_->get_max_width();

            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam && cam->is_undistort_prepared()) {
                    snapshot.undistort = true;
                    break;
                }
            }
        }

        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        gt_load_config_snapshot_ = snapshot;
    }

    void Trainer::setActiveImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        active_image_loader_ = std::move(loader);
    }

    void Trainer::clearActiveImageLoader() {
        if (strategy_) {
            strategy_->set_image_loader(nullptr);
        }
        setActiveImageLoader(nullptr);
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        stop_requested_ = true;

        lfs::core::image_io::BatchImageSaver::instance().wait_all();

        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
            cudaStreamDestroy(callback_stream_);
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        cudaDeviceSynchronize();

        clearActiveImageLoader();
        strategy_.reset();
        bilateral_grid_.reset();
        per_frame_affine_color_.reset();
        per_frame_observation_blur_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        train_dataset_.reset();
        val_dataset_.reset();
        mesh_supervision_cache_dir_.clear();
        mesh_supervision_cache_dirty_.store(false);
        stop_mesh_supervision_prefetch_thread();
        mesh_supervision_disk_targets_.clear();
        mesh_supervision_hot_targets_.clear();
        mesh_supervision_hot_lru_.clear();
        mesh_supervision_uid_order_.clear();
        mesh_supervision_uid_order_index_.clear();
        mesh_supervision_gpu_targets_.clear();
        mesh_supervision_gpu_lru_.clear();
        mesh_supervision_prepared_mesh_.reset();

        // Release GPU memory pools back to system
        lfs::core::Tensor::trim_memory_pool();
        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
    }

    std::expected<void, std::string> Trainer::refresh_mesh_supervision_cache(
        const std::string& cache_tag) {
        if (!params_.optimization.precompute_mesh_depth_normal) {
            return {};
        }

        if (!scene_) {
            LOG_WARN("Mesh supervision refresh requested, but trainer has no Scene context; skipping.");
            return {};
        }

        // Mesh GT targets are rendered on demand and staged in a bounded GPU cache.
        // Refresh now invalidates staged targets and clears the dirty marker.
        if (!cache_tag.empty()) {
            LOG_INFO("Mesh supervision refresh acknowledged for tag '{}' (on-demand render path)", cache_tag);
        }
        {
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            mesh_supervision_cache_dir_.clear();
            mesh_supervision_gpu_targets_.clear();
            mesh_supervision_gpu_lru_.clear();
            mesh_supervision_prepared_mesh_.reset();
        }
        mesh_supervision_cache_dirty_.store(false);
        return {};
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        // Check if background image path changed and needs to be (re)loaded
        const bool bg_image_path_changed =
            params.optimization.bg_image_path != params_.optimization.bg_image_path;
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        const bool mesh_cache_inputs_changed =
            params.dataset.output_path != params_.dataset.output_path ||
            params.dataset.resize_factor != params_.dataset.resize_factor ||
            params.dataset.max_width != params_.dataset.max_width ||
            params.optimization.undistort != params_.optimization.undistort ||
            params.optimization.use_alpha_as_mask != params_.optimization.use_alpha_as_mask ||
            params.optimization.invert_masks != params_.optimization.invert_masks ||
            params.optimization.mask_threshold != params_.optimization.mask_threshold;

        // Update params first
        params_ = params;
        if (mesh_surface_soft_constraint_enabled(params_.optimization) ||
            params_.optimization.mesh_depth_visibility_cull) {
            params_.optimization.precompute_mesh_depth_normal = true;
        }
        mesh_supervision_cache_dir_.clear();
        if (mesh_cache_inputs_changed) {
            mesh_supervision_cache_dirty_.store(true);
            stop_mesh_supervision_prefetch_thread();
            std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
            mesh_supervision_disk_targets_.clear();
            mesh_supervision_hot_targets_.clear();
            mesh_supervision_hot_lru_.clear();
            mesh_supervision_uid_order_.clear();
            mesh_supervision_uid_order_index_.clear();
            mesh_supervision_prefetch_queue_.clear();
            mesh_supervision_prefetch_pending_.clear();
            mesh_supervision_gpu_targets_.clear();
            mesh_supervision_gpu_lru_.clear();
            mesh_supervision_prepared_mesh_.reset();
        }

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                bg_image_cache_.clear();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            bg_image_cache_.clear();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, iter, /*join=*/false);
            auto result = save_checkpoint(iter);
            if (result) {
                const auto checkpoint_path = lfs::training::checkpoint_output_path(params_.dataset.output_path);
                LOG_INFO("Checkpoint and PLY saved to {} (checkpoint: {})",
                         lfs::core::path_to_utf8(params_.dataset.output_path),
                         lfs::core::path_to_utf8(checkpoint_path));
            } else {
                LOG_ERROR("Failed to save checkpoint: {}", result.error());
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Phases by fraction of training
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));

        const float limit_hi = 1.0f / 4.0f;  // start limit
        const float limit_mid = 2.0f / 4.0f; // middle limit
        const float limit_lo = 3.0f / 4.0f;  // final limit

        const float weight_hi = 1.0f;  // start weight
        const float weight_mid = 0.5f; // middle weight
        const float weight_lo = 0.0f;  // final weight

        if (phase < limit_hi) {
            return weight_hi; // hold until bypasses the start limit
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t; // decay to mid value
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t; // decay to final value
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, params_.optimization.iterations);
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result), cudaMemcpyHostToDevice, bg_mix_buffer_.stream());
        return bg_mix_buffer_;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            return it->second;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            bg_image_cache_[cache_key] = bg_image_base_;
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            resized.stream());

        // Cache the resized image
        bg_image_cache_[cache_key] = resized;
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            random_bg_buffer_.stream());

        return random_bg_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else if (!params_.optimization.undistort || !cam->is_undistort_prepared()) {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected. Use --gut or --undistort to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("Use --gut or --undistort to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;
            const bool mesh_surface_loss_enabled = mesh_surface_soft_constraint_enabled(params_.optimization);
            if ((mesh_surface_loss_enabled || params_.optimization.mesh_depth_visibility_cull) &&
                params_.optimization.gut) {
                return std::unexpected("Mesh surface soft-constraint loss and mesh depth visibility cull currently support only the fast rasterizer; disable --gut, set mesh surface loss lambdas to 0, or set mesh_depth_visibility_cull=false");
            }

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            if (on_iteration_start_)
                on_iteration_start_();

            // // Python hook: iteration start (safe, pre-forward)
            // {
            //     lfs::training::HookContext ctx{
            //         .iteration = iter,
            //         .loss = current_loss_.load(),
            //         .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
            //         .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
            //         .trainer = this};
            //     lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::IterationStart);
            //     lfs::training::CommandCenter::instance().update_snapshot(
            //         ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
            //         lfs::training::TrainingPhase::IterationStart);
            //     lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::IterationStart, ctx);
            //     auto view = lfs::training::CommandCenter::instance().snapshot();
            //     lfs::training::CommandCenter::instance().drain_enqueued(view);
            // }

            // Training step entering forward/backward/optimizer region (commands blocked)
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Forward);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            nvtxRangePush("background_for_step");
            lfs::core::Tensor& bg = background_for_step(iter);
            nvtxRangePop();

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            // Configurable tile-based training to reduce peak memory
            const int full_width = cam->image_width();
            const int full_height = cam->image_height();

            // Read tile mode from parameters (1=1 tile, 2=2 tiles, 4=4 tiles)
            const TileMode tile_mode = static_cast<TileMode>(params_.optimization.tile_mode);

            // Determine tile configuration
            int tile_rows = 1, tile_cols = 1;
            switch (tile_mode) {
            case TileMode::One:
                tile_rows = 1;
                tile_cols = 1;
                break;
            case TileMode::Two:
                tile_rows = 2;
                tile_cols = 1;
                break;
            case TileMode::Four:
                tile_rows = 2;
                tile_cols = 2;
                break;
            }

            const int tile_width = full_width / tile_cols;
            const int tile_height = full_height / tile_rows;
            const int num_tiles = tile_rows * tile_cols;

            if (!loss_accumulator_.is_valid()) {
                loss_accumulator_ = core::Tensor::zeros({1}, core::Device::CUDA);
            } else {
                loss_accumulator_.zero_();
            }
            auto& loss_tensor_gpu = loss_accumulator_;
            RenderOutput r_output;
            int tiles_processed = 0;

            // Determine controller phase before tile loop (does not depend on tile results)
            const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
            const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
            const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            const bool ppisp_frozen = is_ppisp_frozen();
            const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                             params_.optimization.ppisp_use_controller &&
                                             !ppisp_frozen &&
                                             params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                             iter >= ppisp_activation_step &&
                                             ppisp_cam_idx >= 0 &&
                                             ppisp_cam_idx < ppisp_controller_pool_->num_cameras();
            const bool pose_refine_active =
                pose_refiner_ && pose_refiner_->is_active(iter) && !in_controller_phase;
            const auto break_progress_line_for_pose_log = [&]() {
                if (progress_) {
                    std::cout << "\n" << std::flush;
                }
            };
            const auto redraw_progress_after_pose_log = [&]() {
                if (progress_ && strategy_) {
                    progress_->resume(
                        iter,
                        current_loss_.load(),
                        static_cast<int>(strategy_->get_model().size()));
                }
            };
            const auto should_log_pose_refine = [&]() {
                if (!pose_refiner_ || !pose_refiner_->enabled() ||
                    params_.optimization.pose_refine_log_every == 0) {
                    return false;
                }
                if (iter == 1) {
                    return true;
                }
                if (params_.optimization.pose_refine_log_every >
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                    return false;
                }
                return iter % static_cast<int>(params_.optimization.pose_refine_log_every) == 0;
            };
            const bool pose_refine_should_log = should_log_pose_refine();
            if (pose_refiner_ && pose_refiner_->enabled() && !pose_refine_active && pose_refine_should_log) {
                break_progress_line_for_pose_log();
                LOG_INFO("[PoseRefine] iter={} inactive reason={}",
                         iter,
                         pose_refiner_->inactive_reason(iter, in_controller_phase));
                redraw_progress_after_pose_log();
            }
            PoseRefiner::Diagnostics pose_refine_diagnostics;
            bool pose_refine_has_diagnostics = false;
            if (pose_refine_active) {
                pose_refiner_->zero_grad();
            }
            fast_lfs::rasterization::ObservationBlurSettings observation_blur;
            const bool observation_blur_active =
                !params_.optimization.gut &&
                render_mode == RenderMode::RGB &&
                !cam->is_pseudo() &&
                !in_controller_phase &&
                per_frame_observation_blur_ &&
                per_frame_observation_blur_->any_active(iter);
            if (observation_blur_active) {
                if (!per_frame_observation_blur_->is_known_frame(cam->uid())) {
                    return std::unexpected(std::format(
                        "Per-frame observation blur has no registered row for real training frame '{}' (uid {})",
                        cam->image_name(), cam->uid()));
                }
                per_frame_observation_blur_->zero_grad();
                observation_blur.parameters_ptr =
                    per_frame_observation_blur_->parameters_for_uid(cam->uid());
                observation_blur.gradients_ptr =
                    per_frame_observation_blur_->gradients_for_uid(cam->uid());
                observation_blur.motion_enabled =
                    per_frame_observation_blur_->motion_active(iter);
                observation_blur.defocus_enabled =
                    per_frame_observation_blur_->defocus_active(iter);
                observation_blur.max_defocus_radius_sq = static_cast<float>(
                    per_frame_observation_blur_->config().max_defocus_radius_sq);
                const float max_observation_radius =
                    params_.optimization.per_frame_observation_blur_max_radius_px;
                observation_blur.max_observation_radius_sq =
                    max_observation_radius * max_observation_radius;
            }
            const bool use_pixel_error_densification =
                (params_.optimization.strategy == "mcmc" ||
                 params_.optimization.strategy == "igs+");
            const bool use_ssim_error = use_pixel_error_densification;
            const bool mesh_surface_schedule_active =
                iter >= params_.optimization.mesh_surface_loss_from_iter;
            const bool mesh_surface_visibility_cull_active =
                mesh_surface_schedule_active &&
                params_.optimization.mesh_depth_visibility_cull;
            const bool mesh_surface_loss_active =
                mesh_surface_loss_enabled &&
                mesh_surface_schedule_active &&
                !in_controller_phase;
            const bool use_mesh_surface_visible_mask =
                mesh_surface_loss_active &&
                mesh_surface_visibility_cull_active;
            MeshSurfaceState mesh_surface_state;
            if (mesh_surface_loss_active) {
                mesh_surface_state = strategy_->mesh_surface_state();
                if (!mesh_surface_state.is_valid()) {
                    return std::unexpected("Mesh surface soft-constraint loss is enabled, but the active strategy has no valid mesh surface state");
                }
            }

            if (use_mesh_surface_visible_mask) {
                const size_t n_gaussians = static_cast<size_t>(strategy_->get_model().size());
                if (!mesh_surface_visible_mask_.is_valid() ||
                    mesh_surface_visible_mask_.numel() < n_gaussians) {
                    mesh_surface_visible_mask_ = core::Tensor::zeros(
                        {n_gaussians}, core::Device::CUDA, core::DataType::Int32);
                } else {
                    mesh_surface_visible_mask_.zero_();
                }
                if (!mesh_surface_visible_count_.is_valid() ||
                    mesh_surface_visible_count_.numel() < 1) {
                    mesh_surface_visible_count_ = core::Tensor::zeros(
                        {1}, core::Device::CUDA, core::DataType::Int32);
                } else {
                    mesh_surface_visible_count_.zero_();
                }
            }

            lfs::core::Tensor mesh_depth_visibility_cull;
            if (mesh_surface_visibility_cull_active) {
                auto mesh_gt = get_mesh_supervision_targets(*cam);
                if (!mesh_gt || !mesh_gt->depth.is_valid() || mesh_gt->depth.is_empty()) {
                    return std::unexpected("Mesh depth visibility cull is active, but mesh GT depth is unavailable");
                }
                mesh_depth_visibility_cull = mesh_gt->depth;
            }

            // Loop over tiles (row-major order)
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                const int tile_row = tile_idx / tile_cols;
                const int tile_col = tile_idx % tile_cols;
                const int tile_x_offset = tile_col * tile_width;
                const int tile_y_offset = tile_row * tile_height;

                nvtxRangePush(std::format("tile_{}x{}", tile_row, tile_col).c_str());

                // Extract GT image tile
                lfs::core::Tensor gt_tile;
                if (num_tiles == 1) {
                    // No tiling - use full image
                    gt_tile = gt_image;
                } else if (gt_image.shape()[0] == 3) {
                    // CHW layout: gt_image is [3, H, W]
                    // Slice both height and width dimensions
                    auto tile_h = gt_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                } else {
                    // HWC layout: gt_image is [H, W, 3]
                    auto tile_h = gt_image.slice(0, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                }

                // Extract background image tile (if using background image)
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    if (num_tiles == 1) {
                        // No tiling - use full image
                        bg_tile = bg_image;
                    } else {
                        // CHW layout: bg_image is [3, H, W]
                        // Slice both height and width dimensions
                        auto tile_h = bg_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                        bg_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                    }
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // CUDA event timing for performance analysis
                cudaEvent_t t_fwd_start, t_fwd_end, t_gggs_loss_start, t_gggs_loss_end, t_mesh_depth_start, t_mesh_depth_end, t_bwd_start, t_bwd_end;
                const bool do_timing = (iter % 100 == 0);
                if (do_timing) {
                    cudaEventCreate(&t_fwd_start);
                    cudaEventCreate(&t_fwd_end);
                    cudaEventCreate(&t_gggs_loss_start);
                    cudaEventCreate(&t_gggs_loss_end);
                    cudaEventCreate(&t_mesh_depth_start);
                    cudaEventCreate(&t_mesh_depth_end);
                    cudaEventCreate(&t_bwd_start);
                    cudaEventCreate(&t_bwd_end);
                    cudaEventRecord(t_fwd_start);
                }

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                if (params_.optimization.gut) {
                    const int tw = (num_tiles > 1) ? tile_width : 0;
                    const int th = (num_tiles > 1) ? tile_height : 0;
                    auto rasterize_result = gsplat_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset, tw, th,
                        1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                    if (!rasterize_result) {
                        nvtxRangePop(); // rasterize_forward
                        nvtxRangePop(); // tile
                        return std::unexpected(rasterize_result.error());
                    }

                    output = std::move(rasterize_result->first);
                    gsplat_ctx.emplace(std::move(rasterize_result->second));
                } else {
                    // Standard mode: use fast rasterizer with tiling support
                    const bool gggs_eq25_active = params_.optimization.enable_gggs_loss &&
                                                  iter >= static_cast<size_t>(params_.optimization.regularization_from_iter);
                    const bool gggs_gtnorm_active = params_.optimization.gggs_gtnorm &&
                                                    iter >= static_cast<size_t>(params_.optimization.gggs_gtnorm_from_iter);
                    const bool mesh_depth_active_pre = params_.optimization.enable_mesh_depth_loss &&
                                                       params_.optimization.lambda_mesh_depth > 0.0f &&
                                                       params_.optimization.precompute_mesh_depth_normal &&
                                                       iter >= static_cast<size_t>(params_.optimization.mesh_depth_loss_from_iter);
                    const bool gggs_active = gggs_eq25_active || gggs_gtnorm_active || mesh_depth_active_pre;
                    auto rasterize_result = fast_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset,
                        (num_tiles > 1) ? tile_width : 0, // 0 means full image
                        (num_tiles > 1) ? tile_height : 0,
                        params_.optimization.mip_filter, bg_tile,
                        gggs_active, // require_depth_normal
                        gggs_active, // require_normal_backward
                        mesh_depth_visibility_cull,
                        mesh_surface_visibility_cull_active,
                        observation_blur);

                    // Check for OOM error
                    if (!rasterize_result) {
                        const std::string& error = rasterize_result.error();
                        if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile

                            LOG_WARN(
                                "FastGS OUT OF MEMORY: tile_idx={}, num_tiles={}, tiles_processed={}, tile_mode={}",
                                tile_idx, num_tiles, tiles_processed, static_cast<int>(tile_mode));
                            LOG_WARN("Arena error: {}", error);

                            // Earlier tiles have already accumulated gradients. Retrying the
                            // whole step would accumulate them a second time.
                            if (tiles_processed > 0) {
                                LOG_ERROR(
                                    "Cannot retry OOM after {} tile(s) completed backward; propagating the original error",
                                    tiles_processed);
                                return std::unexpected(error);
                            }

                            // Handle first-tile OOM by switching tile mode.
                            if (tile_mode == TileMode::Four) {
                                // Already at maximum tiling - can't tile further, return error
                                LOG_ERROR("OUT OF MEMORY at maximum tile mode (2x2). Cannot continue training.");
                                return std::unexpected(error);
                            } else {
                                // Upgrade to next tile mode
                                TileMode new_mode = (tile_mode == TileMode::One) ? TileMode::Two : TileMode::Four;
                                LOG_WARN("OUT OF MEMORY detected. Switching tile mode from {} to {}",
                                         static_cast<int>(tile_mode), static_cast<int>(new_mode));
                                params_.optimization.tile_mode = static_cast<int>(new_mode);

                                // Retry this step with new tile mode
                                return std::unexpected("OOM_RETRY"); // Signal to retry the step
                            }
                        } else {
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }
                    }

                    output = std::move(rasterize_result->first);
                    fast_ctx.emplace(std::move(rasterize_result->second));

                    if (fast_ctx->forward_ctx.n_visible_primitives == 0) {
                        fast_ctx.reset();
                        nvtxRangePop();
                        nvtxRangePop();
                        continue;
                    }

                    if (use_mesh_surface_visible_mask) {
                        const auto* visible_indices = static_cast<const uint32_t*>(
                            fast_ctx->forward_ctx.visible_primitive_indices);
                        auto mark_result = losses::MeshSurfaceRegularization::mark_visible_primitives(
                            visible_indices,
                            static_cast<size_t>(fast_ctx->forward_ctx.n_visible_primitives),
                            static_cast<size_t>(strategy_->get_model().size()),
                            *mesh_surface_state.current_faces,
                            mesh_surface_visible_mask_,
                            mesh_surface_visible_count_);
                        if (!mark_result) {
                            fast_ctx.reset();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(mark_result.error());
                        }
                    }
                }

                r_output = output; // Save last tile for densification
                if (do_timing)
                    cudaEventRecord(t_fwd_end);
                nvtxRangePop();

                if (in_controller_phase) {
                    // Controller phase: forward through ISP with controller params, photometric loss,
                    // backward only through controller (base params frozen)
                    nvtxRangePush("controller_phase");
                    auto cleanup_controller_tile_context = [&]() {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        if (fast_ctx) {
                            fast_ctx.reset();
                        } else if (gsplat_ctx) {
                            if (gsplat_ctx->isect_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->isect_ids_ptr);
                                gsplat_ctx->isect_ids_ptr = nullptr;
                            }
                            if (gsplat_ctx->flatten_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->flatten_ids_ptr);
                                gsplat_ctx->flatten_ids_ptr = nullptr;
                            }
                            arena.end_frame(gsplat_ctx->frame_id);
                        }
                    };

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    }
                    auto ppisp_input = corrected_image;

                    auto pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                    corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);

                    // Photometric loss
                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;

                    const bool cam_is_pseudo = cam->is_pseudo();
                    const float pseudo_weight = cam_is_pseudo ? cam->supervision_weight() : 1.0f;
                    // Pseudo views always come with a per-view valid mask; force masked
                    // photometric loss for them and use the same semantics as MaskMode::Ignore
                    // so loss/grad are only counted inside the valid region.
                    auto effective_opt_params = params_.optimization;
                    if (cam_is_pseudo) {
                        effective_opt_params.mask_mode = lfs::core::param::MaskMode::Ignore;
                    }
                    const bool use_mask = (effective_opt_params.mask_mode != lfs::core::param::MaskMode::None) &&
                                          (cam->has_mask() || (effective_opt_params.use_alpha_as_mask && cam->has_alpha()));
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                            mask = resize_mask_to_resolution(mask, cam->image_height(), cam->image_width());

                        lfs::core::Tensor mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, effective_opt_params);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, effective_opt_params);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    if (pseudo_weight != 1.0f) {
                        tile_loss = tile_loss * pseudo_weight;
                        tile_grad = tile_grad * pseudo_weight;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop(); // compute_photometric_loss

                    // ISP backward for controller params
                    auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                    ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);

                    // End arena frame explicitly (normally done inside rasterize_backward which we skip)
                    cleanup_controller_tile_context();

                    nvtxRangePop(); // controller_phase
                } else {
                    // Normal phase: full forward + backward through all components
                    const bool cam_is_pseudo = cam->is_pseudo();
                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_forward");
                        corrected_image = ppisp_->apply(corrected_image, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }

                    const bool affine_applied =
                        per_frame_affine_color_ && params_.optimization.use_per_frame_affine_color;
                    lfs::core::Tensor affine_input;
                    if (affine_applied) {
                        nvtxRangePush("per_frame_affine_color_forward");
                        affine_input = corrected_image;
                        if (cam_is_pseudo) {
                            corrected_image = per_frame_affine_color_->apply_by_key(
                                affine_input,
                                cam->pseudo_rgb_source_image_name(),
                                cam->pseudo_rgb_source_camera_id());
                        } else {
                            corrected_image = per_frame_affine_color_->apply(affine_input, cam->uid());
                        }
                        nvtxRangePop();
                    }

                    // Final tonemapping: clamp to [0, 1] for loss computation.
                    // This is redundant when PPISP is active (CRF already clamps), but ensures
                    // valid output range for bilateral grids and raw rasterizer output.
                    corrected_image = corrected_image.clamp(0.0f, 1.0f);

                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;
                    lfs::core::Tensor tile_grad_alpha;
                    lfs::core::Tensor tile_error_map;
                    lfs::core::Tensor mask_tile;

                    // 1) Compute photometric loss (populates ssim_map in workspace)
                    const float pseudo_weight = cam_is_pseudo ? cam->supervision_weight() : 1.0f;
                    // Pseudo views always come with a per-view valid mask; force masked
                    // photometric loss for them and use MaskMode::Ignore semantics so loss/grad
                    // are only counted inside the valid region (no opacity penalty bg term).
                    auto effective_opt_params = params_.optimization;
                    if (cam_is_pseudo) {
                        effective_opt_params.mask_mode = lfs::core::param::MaskMode::Ignore;
                    }
                    const bool use_mask = (effective_opt_params.mask_mode != lfs::core::param::MaskMode::None) &&
                                          (cam->has_mask() || (effective_opt_params.use_alpha_as_mask && cam->has_alpha()));
                    const bool used_masked_fused =
                        use_mask &&
                        (effective_opt_params.mask_mode == lfs::core::param::MaskMode::Segment ||
                         effective_opt_params.mask_mode == lfs::core::param::MaskMode::Ignore ||
                         effective_opt_params.mask_mode == lfs::core::param::MaskMode::ProjectMesh) &&
                        effective_opt_params.lambda_dssim > 0.0f;
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        mask = resize_mask_to_resolution(mask, cam->image_height(), cam->image_width());

                        mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, effective_opt_params);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                        tile_grad_alpha = result->grad_alpha;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, effective_opt_params);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    // 2) Extract error map from workspace's ssim_map
                    if (use_pixel_error_densification) {
                        if (use_ssim_error && params_.optimization.lambda_dssim > 0.0f) {
                            lfs::core::Tensor ssim_map;
                            if (used_masked_fused) {
                                ssim_map = masked_fused_workspace_.ssim_map;
                            } else if (params_.optimization.lambda_dssim < 1.0f) {
                                ssim_map = photometric_loss_.fused_workspace().ssim_map;
                            } else {
                                ssim_map = photometric_loss_.ssim_workspace().ssim_map;
                            }
                            {
                                const size_t H = ssim_map.shape()[2];
                                const size_t W = ssim_map.shape()[3];
                                if (!densification_error_map_.is_valid() ||
                                    densification_error_map_.shape()[0] != H ||
                                    densification_error_map_.shape()[1] != W) {
                                    densification_error_map_ = core::Tensor::empty({H, W}, core::Device::CUDA);
                                }
                                lfs::training::kernels::launch_ssim_to_error_map(ssim_map, densification_error_map_);
                                tile_error_map = densification_error_map_;
                            }
                        } else if (use_ssim_error) {
                            // lambda_dssim == 0 but error-priority densification still needs SSIM error
                            lfs::core::Tensor pred_chw = corrected_image;
                            lfs::core::Tensor gt_chw = gt_tile;
                            if (pred_chw.ndim() == 3 && pred_chw.shape()[2] == 3 &&
                                gt_chw.ndim() == 3 && gt_chw.shape()[2] == 3) {
                                pred_chw = pred_chw.permute({2, 0, 1}).contiguous();
                                gt_chw = gt_chw.permute({2, 0, 1}).contiguous();
                            }
                            lfs::training::kernels::ssim_error_map_forward(
                                pred_chw, gt_chw, densification_ssim_workspace_, densification_error_map_);
                            tile_error_map = densification_error_map_;
                        } else {
                            const lfs::core::Tensor abs_diff = (corrected_image - gt_tile).abs();
                            if (abs_diff.ndim() == 3 && abs_diff.shape()[0] == 3) {
                                tile_error_map = abs_diff.mean({0}, false);
                            } else if (abs_diff.ndim() == 3 && abs_diff.shape()[2] == 3) {
                                tile_error_map = abs_diff.mean({2}, false);
                            } else {
                                tile_error_map = abs_diff;
                            }
                            tile_error_map = tile_error_map.contiguous();
                        }

                        if (use_mask &&
                            (effective_opt_params.mask_mode == lfs::core::param::MaskMode::Segment ||
                             effective_opt_params.mask_mode == lfs::core::param::MaskMode::Ignore ||
                             effective_opt_params.mask_mode == lfs::core::param::MaskMode::ProjectMesh)) {
                            tile_error_map = (tile_error_map * mask_tile).contiguous();
                        }
                    }

                    if (pseudo_weight != 1.0f) {
                        tile_loss = tile_loss * pseudo_weight;
                        tile_grad = tile_grad * pseudo_weight;
                        if (tile_grad_alpha.is_valid()) {
                            tile_grad_alpha = tile_grad_alpha * pseudo_weight;
                        }
                        if (tile_error_map.is_valid()) {
                            tile_error_map = tile_error_map * pseudo_weight;
                        }
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop();

                    lfs::core::Tensor raster_grad = tile_grad;
                    if (affine_applied) {
                        nvtxRangePush("per_frame_affine_color_backward");
                        if (cam_is_pseudo) {
                            // Pseudo RGB is a derived observation of its real source frame.
                            // Reuse the source transform in the GS gradient chain, but do not
                            // let the number of derived pseudo views reweight color calibration.
                            raster_grad = per_frame_affine_color_->backward_by_key(
                                affine_input,
                                raster_grad,
                                cam->pseudo_rgb_source_image_name(),
                                cam->pseudo_rgb_source_camera_id(),
                                false);
                        } else {
                            raster_grad = per_frame_affine_color_->backward(
                                affine_input, raster_grad, cam->uid(), true);
                        }
                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_backward");
                        lfs::core::Tensor ppisp_input = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            ppisp_input = bilateral_grid_->apply(output.image, cam->uid());
                        }
                        raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                        if (ppisp_frozen) {
                            ppisp_->zero_grad();
                        }
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_backward");
                        raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                        nvtxRangePop();
                    }

                    // GGGS: compute normal consistency and multi-view losses
                    lfs::core::Tensor gggs_grad_render_normal;
                    lfs::core::Tensor gggs_grad_depth_normal;
                    lfs::core::Tensor gggs_render_normal;
                    lfs::core::Tensor gggs_depth;
                    lfs::core::Tensor gggs_grad_depth_direct; // direct depth grad from inverse depth loss
                    const bool gggs_eq25_active = params_.optimization.enable_gggs_loss &&
                                                  iter >= static_cast<size_t>(params_.optimization.regularization_from_iter) &&
                                                  fast_ctx.has_value() &&
                                                  output.depth.is_valid() && !output.depth.is_empty();
                    const bool gggs_gtnorm_active = params_.optimization.gggs_gtnorm &&
                                                    iter >= static_cast<size_t>(params_.optimization.gggs_gtnorm_from_iter) &&
                                                    fast_ctx.has_value() &&
                                                    output.depth.is_valid() && !output.depth.is_empty();
                    const bool mesh_depth_active = params_.optimization.enable_mesh_depth_loss &&
                                                   params_.optimization.lambda_mesh_depth > 0.0f &&
                                                   params_.optimization.precompute_mesh_depth_normal &&
                                                   iter >= static_cast<size_t>(params_.optimization.mesh_depth_loss_from_iter) &&
                                                   fast_ctx.has_value() &&
                                                   output.depth.is_valid() && !output.depth.is_empty();
                    const bool gggs_active = gggs_eq25_active || gggs_gtnorm_active || mesh_depth_active;
                    if (gggs_active) {
                        if (gggs_eq25_active && iter == static_cast<size_t>(params_.optimization.regularization_from_iter)) {
                            LOG_INFO("[GGGS] Geometry regularization activated at iter {} (lambda_depth_normal={:.4f})",
                                     iter, params_.optimization.lambda_depth_normal);
                        }
                        if (gggs_gtnorm_active && iter == static_cast<size_t>(params_.optimization.gggs_gtnorm_from_iter)) {
                            LOG_INFO("[GGGS] GT normal loss activated at iter {} (lambda_gggs_gtnorm={:.4f})",
                                     iter, params_.optimization.lambda_gggs_gtnorm);
                        }
                        if (mesh_depth_active && iter == static_cast<size_t>(params_.optimization.mesh_depth_loss_from_iter)) {
                            LOG_INFO("[GGGS] Mesh GT inverse depth loss activated at iter {} (lambda_mesh_depth={:.4f})",
                                     iter, params_.optimization.lambda_mesh_depth);
                        }
                        nvtxRangePush("gggs_losses");
                        if (do_timing)
                            cudaEventRecord(t_gggs_loss_start);

                        // Compute depth-derived normal for Eq 25
                        const auto& depth_normal = output.normal; // depth_to_normal_map result

                        // Normal consistency loss (Eq 25): render_normal vs depth_normal
                        if (params_.optimization.enable_gggs_loss &&
                            params_.optimization.lambda_depth_normal > 0.0f &&
                            depth_normal.is_valid() && !depth_normal.is_empty() &&
                            output.render_normal.is_valid() && !output.render_normal.is_empty()) {
                            losses::NormalConsistencyLoss::Params nc_params;
                            nc_params.lambda = params_.optimization.lambda_depth_normal;
                            auto nc_result = losses::NormalConsistencyLoss::forward(
                                output.render_normal, depth_normal,
                                output.depth, nc_params);
                            if (nc_result && nc_result->first.is_valid()) {
                                loss_tensor_gpu = loss_tensor_gpu + nc_result->first * params_.optimization.lambda_depth_normal;
                                gggs_grad_render_normal = nc_result->second.grad_render_normal * params_.optimization.lambda_depth_normal;
                                gggs_grad_depth_normal = nc_result->second.grad_depth_normal * params_.optimization.lambda_depth_normal;
                            }
                        }

                        // GT normal loss: depth_normal vs mesh GT normal
                        if (params_.optimization.gggs_gtnorm &&
                            params_.optimization.lambda_gggs_gtnorm > 0.0f &&
                            depth_normal.is_valid() && !depth_normal.is_empty()) {
                            auto mesh_gt = get_mesh_supervision_targets(*cam);
                            if (mesh_gt && mesh_gt->normal.is_valid() && !mesh_gt->normal.is_empty()) {
                                losses::NormalConsistencyLoss::Params gtnorm_params;
                                gtnorm_params.lambda = params_.optimization.lambda_gggs_gtnorm;
                                auto gtnorm_result = losses::NormalConsistencyLoss::forward(
                                    depth_normal, mesh_gt->normal, output.depth, gtnorm_params);
                                if (gtnorm_result && gtnorm_result->first.is_valid()) {
                                    const float gtnorm_lambda = params_.optimization.lambda_gggs_gtnorm;
                                    loss_tensor_gpu = loss_tensor_gpu + gtnorm_result->first * gtnorm_lambda;
                                    // grad_render_normal from loss corresponds to dL/d(depth_normal)
                                    // Accumulate into depth_normal backward path (depth_to_normal_backward → IFT)
                                    if (gggs_grad_depth_normal.is_valid()) {
                                        gggs_grad_depth_normal = gggs_grad_depth_normal + gtnorm_result->second.grad_render_normal * gtnorm_lambda;
                                    } else {
                                        gggs_grad_depth_normal = gtnorm_result->second.grad_render_normal * gtnorm_lambda;
                                    }
                                    // grad_depth_normal from loss = dL/d(gt_normal) → discard (GT is constant)
                                }
                            }
                        }

                        // Mesh GT inverse depth loss: mean(|1/rendered - 1/gt|) over valid pixels
                        // Gradient flows only through rendered depth → IFT → Gaussian params
                        if (do_timing)
                            cudaEventRecord(t_mesh_depth_start);
                        if (mesh_depth_active) {
                            auto mesh_gt = get_mesh_supervision_targets(*cam);
                            if (mesh_gt && mesh_gt->depth.is_valid() && !mesh_gt->depth.is_empty()) {
                                constexpr float DEPTH_EPS = 1e-4f;

                                // Squeeze to [H, W]
                                auto rendered_d = output.depth;
                                if (rendered_d.ndim() == 3 && rendered_d.shape()[0] == 1) {
                                    rendered_d = rendered_d.squeeze(0);
                                }
                                auto gt_d = mesh_gt->depth;
                                if (gt_d.ndim() == 3 && gt_d.shape()[0] == 1) {
                                    gt_d = gt_d.squeeze(0);
                                }

                                if (rendered_d.shape() == gt_d.shape()) {
                                    // Valid mask: 1.0 where both depths > DEPTH_EPS, 0.0 elsewhere
                                    auto valid_float = rendered_d.gt(DEPTH_EPS)
                                                                  .logical_and(gt_d.gt(DEPTH_EPS))
                                                                  .to(lfs::core::DataType::Float32);

                                    // Clamp for numerical stability (avoids division by zero)
                                    auto rendered_clamped = rendered_d.clamp_min(DEPTH_EPS);
                                    auto gt_clamped = gt_d.clamp_min(DEPTH_EPS);

                                    // Inverse-depth L1 diff (masked): 1/rendered - 1/gt
                                    auto diff = rendered_clamped.reciprocal() * valid_float
                                                - gt_clamped.reciprocal() * valid_float;
                                    auto depth_loss_val = diff.abs().mean(); // mean over all pixels (0 at invalid)

                                    loss_tensor_gpu = loss_tensor_gpu + depth_loss_val * params_.optimization.lambda_mesh_depth;

                                    // Gradient: d(mean(|diff|)) / d(rendered_d)
                                    //   = sign(diff) * valid_float * d(1/rendered_clamped)/d(rendered_d) / numel
                                    //   = sign(diff) * valid_float * (-1/rendered_clamped^2) / numel
                                    const float inv_numel = 1.0f / static_cast<float>(rendered_d.numel());
                                    gggs_grad_depth_direct = diff.sign()
                                                             * valid_float
                                                             * rendered_clamped.square().reciprocal().neg()
                                                             * (inv_numel * params_.optimization.lambda_mesh_depth);
                                }
                            }
                        }
                        if (do_timing)
                            cudaEventRecord(t_mesh_depth_end);

                        gggs_render_normal = output.render_normal;
                        gggs_depth = output.depth;
                        if (do_timing)
                            cudaEventRecord(t_gggs_loss_end);
                        nvtxRangePop();
                    }

                    nvtxRangePush("rasterize_backward");
                    if (do_timing)
                        cudaEventRecord(t_bwd_start);
                    if (gsplat_ctx) {
                        auto grad_alpha = tile_grad_alpha.is_valid()
                                              ? tile_grad_alpha
                                              : lfs::core::Tensor::zeros_like(output.alpha);
                        gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                  strategy_->get_model(), strategy_->get_optimizer(),
                                                  use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                    } else {
                        lfs::core::Tensor* pose_grad_w2c = nullptr;
                        if (pose_refine_active && fast_ctx) {
                            pose_grad_w2c = &pose_refiner_->grad_w2c_scratch(raster_grad.stream());
                        }
                        fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                strategy_->get_optimizer(), tile_grad_alpha,
                                                use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{},
                                                gggs_grad_render_normal, gggs_render_normal,
                                                gggs_grad_depth_normal, gggs_depth,
                                                gggs_grad_depth_direct,
                                                pose_grad_w2c);
                        if (pose_grad_w2c) {
                            pose_refiner_->accumulate_w2c_gradient(*cam, *pose_grad_w2c);
                        }
                    }
                    if (do_timing)
                        cudaEventRecord(t_bwd_end);
                    nvtxRangePop();

                    // Log timing breakdown every 100 iterations
                    if (do_timing) {
                        cudaEventSynchronize(t_bwd_end);
                        float fwd_ms = 0, gggs_loss_ms = 0, mesh_depth_ms = 0, bwd_ms = 0;
                        cudaEventElapsedTime(&fwd_ms, t_fwd_start, t_fwd_end);
                        if (gggs_active) {
                            cudaEventElapsedTime(&gggs_loss_ms, t_gggs_loss_start, t_gggs_loss_end);
                        }
                        if (mesh_depth_active) {
                            cudaEventElapsedTime(&mesh_depth_ms, t_mesh_depth_start, t_mesh_depth_end);
                        }
                        cudaEventElapsedTime(&bwd_ms, t_bwd_start, t_bwd_end);
                        LOG_INFO("[Timing] iter={} gggs={} fwd={:.3f}ms gggs_loss={:.3f}ms mesh_depth={:.3f}ms bwd={:.3f}ms total={:.3f}ms",
                                 iter, gggs_active, fwd_ms, gggs_loss_ms, mesh_depth_ms, bwd_ms, fwd_ms + gggs_loss_ms + mesh_depth_ms + bwd_ms);
                    }
                }

                if (do_timing) {
                    cudaEventDestroy(t_fwd_start);
                    cudaEventDestroy(t_fwd_end);
                    cudaEventDestroy(t_gggs_loss_start);
                    cudaEventDestroy(t_gggs_loss_end);
                    cudaEventDestroy(t_mesh_depth_start);
                    cudaEventDestroy(t_mesh_depth_end);
                    cudaEventDestroy(t_bwd_start);
                    cudaEventDestroy(t_bwd_end);
                }

                nvtxRangePop(); // End tile
            }

            if (tiles_processed > 1)
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(tiles_processed);

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            if (in_controller_phase) {
                // Controller phase: only update controller weights
                nvtxRangePush("controller_optimizer_step");
                ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                ppisp_controller_pool_->zero_grad();
                ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                nvtxRangePop();
            } else {
                // Normal phase: regularization losses + optimizer steps for all components

                if (mesh_surface_loss_active) {
                    nvtxRangePush("compute_mesh_surface_loss");
                    auto mesh_loss_result = compute_mesh_surface_loss(
                        strategy_->get_model(),
                        strategy_->get_optimizer(),
                        mesh_surface_state,
                        use_mesh_surface_visible_mask ? &mesh_surface_visible_mask_ : nullptr,
                        use_mesh_surface_visible_mask ? &mesh_surface_visible_count_ : nullptr,
                        params_.optimization);
                    if (!mesh_loss_result) {
                        nvtxRangePop();
                        return std::unexpected(mesh_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *mesh_loss_result;
                    nvtxRangePop();
                }

                if (params_.optimization.scale_reg > 0.0f) {
                    nvtxRangePush("compute_scale_reg_loss");
                    auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!scale_loss_result) {
                        return std::unexpected(scale_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                    nvtxRangePop();
                }

                if (params_.optimization.opacity_reg > 0.0f) {
                    nvtxRangePush("compute_opacity_reg_loss");
                    auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!opacity_loss_result) {
                        return std::unexpected(opacity_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                    nvtxRangePop();
                }

                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_tv_and_step");
                    const float tv_weight = params_.optimization.tv_loss_weight;

                    loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                    bilateral_grid_->tv_backward(tv_weight);
                    bilateral_grid_->optimizer_step();
                    bilateral_grid_->zero_grad();
                    bilateral_grid_->scheduler_step();

                    nvtxRangePop();
                }

                if (ppisp_ && params_.optimization.use_ppisp && !ppisp_frozen) {
                    nvtxRangePush("ppisp_reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                    ppisp_->reg_backward();
                    ppisp_->optimizer_step();
                    ppisp_->zero_grad();
                    ppisp_->scheduler_step();

                    nvtxRangePop();
                }

                if (per_frame_affine_color_ &&
                    params_.optimization.use_per_frame_affine_color &&
                    !cam->is_pseudo()) {
                    nvtxRangePush("per_frame_affine_color_reg_and_step");

                    const float identity_weight =
                        params_.optimization.per_frame_affine_color_identity_reg_weight;
                    const float gauge_weight =
                        params_.optimization.per_frame_affine_color_gauge_reg_weight;
                    loss_tensor_gpu = loss_tensor_gpu +
                                      per_frame_affine_color_->regularization_loss_gpu(
                                          identity_weight, gauge_weight);
                    per_frame_affine_color_->regularization_backward(
                        identity_weight, gauge_weight);
                    per_frame_affine_color_->optimizer_step();
                    per_frame_affine_color_->zero_grad();
                    per_frame_affine_color_->scheduler_step();

                    nvtxRangePop();
                }

                if (observation_blur_active) {
                    nvtxRangePush("per_frame_observation_blur_reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu +
                                      per_frame_observation_blur_->regularization_loss_gpu(iter);
                    per_frame_observation_blur_->regularization_backward(iter);
                    per_frame_observation_blur_->optimizer_step(iter);
                    per_frame_observation_blur_->zero_grad();

                    nvtxRangePop();
                }

                if (pose_refine_active) {
                    nvtxRangePush("pose_refine_regularization");
                    auto pose_reg_loss = pose_refiner_->regularization_loss();
                    if (pose_reg_loss.is_valid() && pose_reg_loss.numel() > 0) {
                        loss_tensor_gpu = loss_tensor_gpu + pose_reg_loss;
                        if (pose_refine_should_log) {
                            const float pose_reg_value = pose_reg_loss.item<float>();
                            pose_refine_diagnostics =
                                pose_refiner_->capture_diagnostics_pre_step(*cam, pose_reg_value);
                            pose_refine_has_diagnostics = true;
                        }
                    }
                    nvtxRangePop();
                }
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            lfs::core::Tensor sparsity_loss_gpu;
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                nvtxRangePush("sparsity_loss");
                auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                if (!sparsity_result) {
                    nvtxRangePop();
                    return std::unexpected(sparsity_result.error());
                }
                auto& [loss_tensor, ctx] = *sparsity_result;
                sparsity_loss_gpu = std::move(loss_tensor);

                if (ctx.n > 0) {
                    if (auto result = sparsity_optimizer_->compute_loss_backward(
                            ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                        !result) {
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                if (iter == base_iterations + 1) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Sync loss to CPU only at intervals - single sync point
            constexpr int LOSS_SYNC_INTERVAL = 10;
            float loss_value = 0.0f;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                // Accumulate on GPU then sync once
                auto total_loss = sparsity_loss_gpu.numel() > 0
                                      ? (loss_tensor_gpu + sparsity_loss_gpu)
                                      : loss_tensor_gpu;
                loss_value = total_loss.item<float>();

                if (std::isnan(loss_value) || std::isinf(loss_value)) {
                    return std::unexpected(std::format("NaN/Inf loss at iteration {}", iter));
                }

                current_loss_ = loss_value;
                if (progress_) {
                    progress_->update(iter, loss_value,
                                      static_cast<int>(strategy_->get_model().size()),
                                      strategy_->is_refining(iter));
                }
                lfs::core::events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            const bool in_sparsification = params_.optimization.enable_sparsity &&
                                           iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);

            if (!in_sparsification) {
                strategy_->pre_step(iter, r_output);
            }

            {
                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // // Python hook: pre-optimizer-step (post-backward, pre-step)
                    // {
                    //     lfs::training::HookContext ctx{
                    //         .iteration = iter,
                    //         .loss = current_loss_.load(),
                    //         .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    //         .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    //         .trainer = this};
                    //     lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::OptimizerStep);
                    //     lfs::training::CommandCenter::instance().update_snapshot(
                    //         ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    //         lfs::training::TrainingPhase::OptimizerStep);
                    //     lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                    // }

                    if (!in_sparsification) {
                        strategy_->post_backward(iter, r_output);
                    }

                    // Skip strategy step if we're in controller distillation phase and freeze is enabled
                    const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
                    const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                  params_.optimization.ppisp_use_controller &&
                                                  params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                  iter >= ppisp_activation_step;
                    if (!freeze_gaussians) {
                        strategy_->step(iter);
                    }
                    if (pose_refine_active) {
                        pose_refiner_->step(iter);
                        if (pose_refine_has_diagnostics) {
                            pose_refiner_->capture_diagnostics_post_step(pose_refine_diagnostics);
                        }
                        pose_refiner_->apply_to_camera(*cam);
                        if (pose_refine_has_diagnostics) {
                            const auto summary = pose_refiner_->diagnostics_summary(pose_refine_diagnostics);
                            break_progress_line_for_pose_log();
                            if (pose_refine_diagnostics.finite) {
                                LOG_INFO("[PoseRefine] iter={} {}", iter, summary);
                            } else {
                                LOG_WARN("[PoseRefine] iter={} non-finite diagnostics {}", iter, summary);
                            }
                            redraw_progress_after_pose_log();
                        }
                    }
                }

                if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity update: {}", result.error());
                }
                if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity pruning: {}", result.error());
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // // Save checkpoint (not PLY) at specified steps
                // for (size_t save_step : params_.optimization.save_steps) {
                //     if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                //         auto result = save_checkpoint(iter);
                //         if (!result) {
                //             LOG_WARN("Failed to save checkpoint at iteration {}: {}", iter, result.error());
                //         }
                //     }
                // }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    const bool save_depth_outputs = params_.optimization.save_depth;
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                const auto render_mode = save_depth_outputs ? GsplatRenderMode::RGB_ED : GsplatRenderMode::RGB;
                                rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                             1.0f, false, render_mode, true);
                            } else {
                                if (save_depth_outputs) {
                                    auto rasterize_result = fast_rasterize_forward(
                                        *cam_to_use,
                                        strategy_->get_model(),
                                        background_,
                                        0, 0, 0, 0,
                                        false,
                                        {},
                                        true);
                                    if (!rasterize_result) {
                                        throw std::runtime_error("Timelapse rasterization failed: " + rasterize_result.error());
                                    }
                                    rendered_timelapse_output = std::move(rasterize_result->first);
                                } else {
                                    rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                                }
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = lfs::io::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                  rendered_timelapse_output.image);

                            const bool depth_valid = rendered_timelapse_output.depth.is_valid();
                            const bool depth_empty = depth_valid ? rendered_timelapse_output.depth.is_empty() : true;
                            if (save_depth_outputs) {
                                LOG_INFO("[DepthDebug][timelapse] iter={} image='{}' depth_valid={} depth_empty={}",
                                         iter,
                                         img_name,
                                         depth_valid,
                                         depth_empty);
                            }

                            if (save_depth_outputs && depth_valid && !depth_empty) {
                                const auto [fx, fy, cx, cy] = cam_to_use->get_intrinsics();

                                auto depth_normal = depth_to_normal_map_gpu(rendered_timelapse_output.depth, fx, fy, cx, cy);

                                const auto depth_vis = colorize_depth_map(rendered_timelapse_output.depth);
                                const auto depth_normal_vis = colorize_normal_map(depth_normal);

                                std::vector<lfs::core::Tensor> depth_normal_panels = {depth_vis, depth_normal_vis};

                                // Append render_normal when --gggs is active
                                const bool render_normal_valid = rendered_timelapse_output.render_normal.is_valid() && !rendered_timelapse_output.render_normal.is_empty();
                                if (params_.optimization.enable_gggs_loss && render_normal_valid) {
                                    const auto render_normal_vis = colorize_normal_map(rendered_timelapse_output.render_normal);
                                    depth_normal_panels.push_back(render_normal_vis);
                                }

                                // Render mesh GT targets once (reused for concat and separate save)
                                lfs::core::Tensor mesh_gt_depth, mesh_gt_normal;
                                if (params_.optimization.precompute_mesh_depth_normal && scene_) {
                                    auto mesh_gt_result = render_mesh_supervision_targets_for_camera(
                                        *scene_,
                                        *cam_to_use,
                                        params_);
                                    if (!mesh_gt_result) {
                                        LOG_WARN("[DepthDebug][timelapse] mesh GT render failed for '{}' : {}",
                                                 img_name,
                                                 mesh_gt_result.error());
                                    } else {
                                        mesh_gt_depth = std::move(mesh_gt_result->depth);
                                        mesh_gt_normal = std::move(mesh_gt_result->normal);
                                    }
                                }

                                // Append GT normal when --gggs-gtnorm is active
                                if (params_.optimization.gggs_gtnorm &&
                                    mesh_gt_normal.is_valid() && !mesh_gt_normal.is_empty()) {
                                    const auto gt_normal_vis = colorize_normal_map(mesh_gt_normal);
                                    depth_normal_panels.push_back(gt_normal_vis);
                                }

                                const auto depth_normal_path = output_path / std::format("{:06d}_depth_normal.png", iter);
                                LOG_INFO("[DepthDebug][timelapse] queue save depth+normal concat='{}'",
                                         lfs::core::path_to_utf8(depth_normal_path));

                                lfs::core::image_io::save_images_async(
                                    depth_normal_path,
                                    depth_normal_panels,
                                    true, // horizontal
                                    4);   // separator width

                                // Save separate mesh GT depth+normal concat
                                if (mesh_gt_depth.is_valid() && mesh_gt_normal.is_valid()) {
                                    const auto mesh_depth_vis = colorize_depth_map(mesh_gt_depth);
                                    const auto mesh_normal_vis = colorize_normal_map(mesh_gt_normal);
                                    const auto mesh_depth_normal_path = output_path / std::format("{:06d}_mesh_gt_depth_normal.png", iter);
                                    LOG_INFO("[DepthDebug][timelapse] queue save mesh GT depth+normal concat='{}'",
                                             lfs::core::path_to_utf8(mesh_depth_normal_path));

                                    lfs::core::image_io::save_images_async(
                                        mesh_depth_normal_path,
                                        {mesh_depth_vis, mesh_normal_vis},
                                        true,
                                        4);
                                }
                            } else if (save_depth_outputs) {
                                LOG_INFO("[DepthDebug][timelapse] skip depth save: depth tensor invalid or empty");
                            }
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // // Python hook: post-step (after optimizer and side-effects)
            // {
            //     lfs::training::HookContext ctx{
            //         .iteration = iter,
            //         .loss = current_loss_.load(),
            //         .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
            //         .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
            //         .trainer = this};
            //     lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
            //     lfs::training::CommandCenter::instance().update_snapshot(
            //         ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
            //         lfs::training::TrainingPhase::SafeControl);
            //     lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
            // }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag
        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);

        ready_to_start_ = true; // Skip GUI wait for now

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");
        auto& cache_loader = lfs::io::CacheLoader::getInstance();
        cache_loader.reset_cache();
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache,
                                         train_dataset_size_,
                                         params_.dataset.loading_params.min_cpu_free_GB,
                                         params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                         params_.dataset.loading_params.print_cache_status,
                                         params_.dataset.loading_params.print_status_freq_num);

        // // Notify Python control layer that training is starting
        // {
        //     lfs::training::HookContext ctx{
        //         .iteration = 0,
        //         .loss = current_loss_.load(),
        //         .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
        //         .is_refining = strategy_ ? strategy_->is_refining(0) : false,
        //         .trainer = this};
        //     lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
        //     lfs::training::CommandCenter::instance().update_snapshot(
        //         ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
        //         lfs::training::TrainingPhase::SafeControl);
        //     lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
        // }

        try {
            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            // gggs_gtnorm implies mesh GT rendering
            if (params_.optimization.gggs_gtnorm && !params_.optimization.precompute_mesh_depth_normal) {
                params_.optimization.precompute_mesh_depth_normal = true;
            }
            if ((mesh_surface_soft_constraint_enabled(params_.optimization) ||
                 params_.optimization.mesh_depth_visibility_cull) &&
                !params_.optimization.precompute_mesh_depth_normal) {
                params_.optimization.precompute_mesh_depth_normal = true;
            }

            if (params_.optimization.precompute_mesh_depth_normal) {
                {
                    std::lock_guard<std::mutex> lock(mesh_supervision_mutex_);
                    mesh_supervision_gpu_window_size_ = std::max<size_t>(1, pipelined_config.prefetch_count);
                    mesh_supervision_gpu_targets_.clear();
                    mesh_supervision_gpu_lru_.clear();
                    mesh_supervision_prepared_mesh_.reset();
                }
                LOG_INFO("Mesh GT supervision enabled: GPU prefetch window={} (aligned with RGB prefetch_count)",
                         std::max<size_t>(1, pipelined_config.prefetch_count));
            }

            const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    mask_pipeline_config.use_alpha_as_mask = true;
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                } else {
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                }
            }
            
            // --- Pyramid training configuration ---
            PyramidConfig pyramid_config;
            if (params_.optimization.pyramid_training) {
                pyramid_config.enabled = true;
                pyramid_config.levels = params_.optimization.pyramid_levels;
                pyramid_config.compute_factors();
                LOG_INFO("Pyramid training enabled: {} levels, {} iters per level",
                         pyramid_config.levels, params_.optimization.pyramid_step_interval);
                for (size_t i = 0; i < pyramid_config.downsample_factors.size(); ++i) {
                    LOG_INFO("  Pyramid level {} (coarse→fine): {}x downsample",
                             i, pyramid_config.downsample_factors[i]);
                }
                LOG_INFO("  Pyramid level {} (full resolution): 1x", pyramid_config.levels - 1);
            }

            auto refresh_mesh_cache_for_pyramid_level =
                [this, &pyramid_config](const int pyramid_level, const int num_coarse_levels)
                -> std::expected<void, std::string> {
                if (!params_.optimization.precompute_mesh_depth_normal) {
                    return {};
                }

                if (!scene_) {
                    LOG_WARN("Pyramid mesh-GT refresh skipped: trainer has no Scene context.");
                    return {};
                }
                int downsample_factor = 1;

                if (pyramid_level < num_coarse_levels) {
                    if (pyramid_level < 0 ||
                        pyramid_level >= static_cast<int>(pyramid_config.downsample_factors.size())) {
                        return std::unexpected(std::format(
                            "Pyramid mesh-GT refresh failed: level {} out of range for {} coarse levels",
                            pyramid_level,
                            num_coarse_levels));
                    }

                    downsample_factor = pyramid_config.downsample_factors[static_cast<size_t>(pyramid_level)];
                }

                std::string cache_tag;
                if (pyramid_level < num_coarse_levels) {
                    cache_tag = std::format("level_{:02d}_{}x", pyramid_level, downsample_factor);
                } else {
                    cache_tag = "level_fullres";
                }

                auto refresh_result = refresh_mesh_supervision_cache(cache_tag);
                if (!refresh_result) {
                    return refresh_result;
                }

                if (params_.optimization.precompute_mesh_depth_normal) {
                    LOG_INFO("Mesh supervision state refreshed for pyramid level {} ({}x scale)",
                             pyramid_level,
                             downsample_factor);
                }
                return {};
            };

            // 在 initialize 中初始化的类 CameraDataset 的实例 train_dataset_ 中读图，随机持续输出场景中的图像
            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, mask_pipeline_config, pyramid_config);



            int pyramid_prev_level = -1; // Track pyramid level for transition logging
            bool pyramid_intrinsics_once_logged = false;
            constexpr int PYRAMID_INTRINSICS_DEBUG_ITERS = 6;

            struct PyramidIntrinsicsGuard {
                lfs::core::Camera* cam = nullptr;
                bool active = false;
                ~PyramidIntrinsicsGuard() {
                    if (active && cam) {
                        cam->restore_intrinsics();
                    }
                }
            };
            auto active_image_loader_guard = makeScopeGuard([this]() {
                clearActiveImageLoader();
            });
            updateGTLoadConfigSnapshot();
            setActiveImageLoader(train_dataloader->get_loader_shared());
            strategy_->set_image_loader(train_dataloader->get_loader());

            LOG_DEBUG("Starting training iterations");
            while (iter <= params_.optimization.iterations) {
                lfs::core::Tensor::set_memory_pool_iteration(iter);

                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load(std::memory_order_acquire)) {
                    const cudaError_t callback_status = cudaStreamQuery(callback_stream_);
                    if (callback_status == cudaSuccess) {
                        callback_busy_.store(false, std::memory_order_release);
                    } else if (callback_status != cudaErrorNotReady) {
                        LOG_WARN("Callback stream query failed: {}", cudaGetErrorString(callback_status));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                // 确认一下这里的数据结构，图像金字塔从这里入手
                auto& example = *example_opt;
                cam = example.data.camera;
                PyramidIntrinsicsGuard intrinsics_guard{cam, false};
                if (pose_refiner_ && pose_refiner_->enabled()) {
                    pose_refiner_->apply_to_camera(*cam);
                }



                // --- Pyramid training: select appropriate resolution for current iteration ---
                // Schedule: iterations are split evenly across pyramid levels.
                //   iter [1, step_interval]                              → level 0 (coarsest)
                //   iter [step_interval+1, 2*step_interval]             → level 1
                //   ...
                //   iter > (pyramid_levels-1)*step_interval             → full resolution
                // Camera intrinsics are temporarily overridden for downsampled levels.
                bool pyramid_intrinsics_overridden = false;
                if (pyramid_config.enabled && !example.data.pyramid_images.empty()) {
                    const int step_interval = params_.optimization.pyramid_step_interval;
                    const int num_coarse_levels = static_cast<int>(example.data.pyramid_images.size());

                    if (!pyramid_intrinsics_once_logged) {
                        LOG_INFO("Pyramid intrinsics table (coarse->fine):");
                        for (int li = 0; li < num_coarse_levels; ++li) {
                            const auto& intr = example.data.pyramid_intrinsics[li];
                            LOG_INFO("  level {}: fx={:.6f}, fy={:.6f}, cx={:.6f}, cy={:.6f}, W={}, H={}",
                                     li, intr.focal_x, intr.focal_y, intr.center_x, intr.center_y,
                                     intr.width, intr.height);
                        }
                        const auto [ffx, ffy, fcx, fcy] = cam->get_intrinsics();
                        LOG_INFO("  level {}: fx={:.6f}, fy={:.6f}, cx={:.6f}, cy={:.6f}, W={}, H={} (full resolution)",
                                 num_coarse_levels, ffx, ffy, fcx, fcy,
                                 cam->image_width(), cam->image_height());
                        pyramid_intrinsics_once_logged = true;
                    }

                    // Determine which pyramid level to use (0 = coarsest, num_coarse_levels = full res)
                    const int pyramid_level = std::min(
                        (iter - 1) / step_interval,
                        num_coarse_levels); // clamp to full-res at max

                    if (pyramid_level != pyramid_prev_level) {
                        // Flush current progress bar line so LOG_INFO doesn't overwrite it
                        if (progress_ && pyramid_prev_level >= 0) {
                            std::cout << "\n" << std::flush;
                        }
                        const int total_levels = num_coarse_levels + 1; // coarse levels + full res
                        if (pyramid_level < num_coarse_levels) {
                            const auto& intr = example.data.pyramid_intrinsics[pyramid_level];
                            LOG_INFO("Pyramid training: level {}/{} @ iter {} ({}x downsample, {}x{})",
                                     pyramid_level + 1, total_levels, iter,
                                     pyramid_config.downsample_factors[pyramid_level],
                                     intr.width, intr.height);
                        } else {
                            const auto& img_shape = example.data.image.shape();
                            LOG_INFO("Pyramid training: level {}/{} @ iter {} (full resolution, {}x{})",
                                     total_levels, total_levels, iter,
                                     img_shape[2], img_shape[1]);
                        }

                        if (auto refresh_result = refresh_mesh_cache_for_pyramid_level(pyramid_level, num_coarse_levels); !refresh_result) {
                            return std::unexpected(refresh_result.error());
                        }

                        pyramid_prev_level = pyramid_level;
                    }

                    if (pyramid_level < num_coarse_levels) {
                        // Use downsampled image and temporarily override camera intrinsics
                        gt_image = std::move(example.data.pyramid_images[pyramid_level]);
                        const auto& intr = example.data.pyramid_intrinsics[pyramid_level];
                        cam->set_intrinsics_for_pyramid(
                            intr.focal_x, intr.focal_y,
                            intr.center_x, intr.center_y,
                            intr.width, intr.height);
                        pyramid_intrinsics_overridden = true;
                        intrinsics_guard.active = true;
                    } else {
                        // Full resolution: use the original image as-is
                        gt_image = std::move(example.data.image);
                    }
                } else {
                    gt_image = std::move(example.data.image);
                }

                // Store pipelined mask for use in train_step.
                // When pyramid training uses a downsampled level, use the matching pyramid mask.
                if (pyramid_intrinsics_overridden && !example.data.pyramid_masks.empty()) {
                    const int step_interval = params_.optimization.pyramid_step_interval;
                    const int pl = std::min(
                        (iter - 1) / step_interval,
                        static_cast<int>(example.data.pyramid_masks.size()) - 1);
                    pipelined_mask_ = std::move(example.data.pyramid_masks[pl]);
                } else {
                    pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();
                }

                if (pyramid_config.enabled && iter <= PYRAMID_INTRINSICS_DEBUG_ITERS) {
                    const auto [fx_dbg, fy_dbg, cx_dbg, cy_dbg] = cam->get_intrinsics();
                    const auto& gshape = gt_image.shape();
                    const int gH = (gshape.rank() >= 2) ? static_cast<int>(gshape[1]) : cam->image_height();
                    const int gW = (gshape.rank() >= 3) ? static_cast<int>(gshape[2]) : cam->image_width();

                    std::string mask_info = "none";
                    if (pipelined_mask_.is_valid() && !pipelined_mask_.is_empty() && pipelined_mask_.shape().rank() >= 2) {
                        mask_info = std::format("{}x{}", pipelined_mask_.shape()[1], pipelined_mask_.shape()[0]);
                    }

                    LOG_INFO("Pyramid debug iter {}: fx={:.6f}, fy={:.6f}, cx={:.6f}, cy={:.6f}, GT={}x{}, mask={}",
                             iter, fx_dbg, fy_dbg, cx_dbg, cy_dbg, gW, gH, mask_info);
                }

                constexpr int MAX_OOM_TILE_RETRIES = 2; // Tile modes can advance at most 1 -> 2 -> 4.
                int oom_tile_retry_count = 0;
                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                while (!step_result && step_result.error() == "OOM_RETRY") {
                    if (oom_tile_retry_count >= MAX_OOM_TILE_RETRIES) {
                        return std::unexpected(
                            "OOM recovery exceeded the bounded tile-mode retry limit");
                    }

                    cudaDeviceSynchronize();
                    cudaGetLastError();

                    lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
                    lfs::core::Tensor::trim_memory_pool();

                    cudaDeviceSynchronize();
                    cudaGetLastError();

                    ++oom_tile_retry_count;
                    LOG_INFO(
                        "OOM recovery: retrying iteration {} ({}/{}) with tile mode {}",
                        iter,
                        oom_tile_retry_count,
                        MAX_OOM_TILE_RETRIES,
                        params_.optimization.tile_mode);
                    step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                }
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                // Restore temporary pyramid intrinsics immediately after the step.
                // This narrows the window where other threads (e.g. preview callbacks)
                // may observe per-level intrinsics.
                if (intrinsics_guard.active && intrinsics_guard.cam) {
                    intrinsics_guard.cam->restore_intrinsics();
                    intrinsics_guard.active = false;
                }

                // Transition to safe control phase and execute deferred Python callbacks
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().drain_callbacks();

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_ && !callback_busy_.load(std::memory_order_acquire)) {
                    callback_busy_.store(true, std::memory_order_release);
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_.store(false, std::memory_order_release);
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                ++iter;
            }

            clearActiveImageLoader();
            active_image_loader_guard.release();

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                cudaStreamSynchronize(callback_stream_);
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            }

            if (auto result = export_per_frame_affine_color_parameters(current_iteration_.load()); !result) {
                throw std::runtime_error(result.error());
            }
            if (auto result = export_per_frame_observation_blur_parameters(current_iteration_.load()); !result) {
                throw std::runtime_error(result.error());
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            // Notify training end
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingEnd, ctx);
            }

            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, const int iter_num, const bool join_threads) {
        flush_pose_refine_outputs(save_path, iter_num);

        const lfs::io::PlySaveOptions ply_options{
            // .output_path = save_path / ("splat_" + std::to_string(iter_num) + ".ply"),
            .output_path = save_path / ("Gaussian.ply"),
            .binary = true,
            .async = !join_threads};

        const auto ply_result = lfs::io::save_ply(strategy_->get_model(), ply_options);
        if (!ply_result) {
            if (ply_result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iter_num,
                    .path = ply_options.output_path,
                    .error = ply_result.error().message,
                    .required_bytes = ply_result.error().required_bytes,
                    .available_bytes = ply_result.error().available_bytes,
                    .is_disk_space_error = true,
                    .is_checkpoint = false}
                    .emit();
            }
            LOG_WARN("Failed to save PLY: {}", ply_result.error().message);
            return; // Don't save checkpoint if PLY failed
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iter_num);

        // // Save checkpoint alongside PLY for training resumption
        // auto ckpt_result = lfs::training::save_checkpoint(save_path, iter_num, *strategy_, params_,
        //                                                   bilateral_grid_.get(), ppisp_.get(), controller_to_save);
        // if (!ckpt_result) {
        //     LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
        // }

        if (ppisp_) {
            const auto ppisp_path = get_ppisp_companion_path(ply_options.output_path);
            std::optional<PPISPFileMetadata> metadata;
            if (auto metadata_result = build_ppisp_sidecar_metadata(); metadata_result) {
                metadata = std::move(*metadata_result);
            } else {
                LOG_WARN("Failed to build PPISP sidecar metadata for '{}': {}. Saving sidecar without metadata.",
                         lfs::core::path_to_utf8(ppisp_path), metadata_result.error());
            }
            const auto ppisp_result = save_ppisp_file(ppisp_path, *ppisp_, controller_to_save,
                                                      metadata ? &*metadata : nullptr);
            if (!ppisp_result) {
                LOG_WARN("Failed to save PPISP file: {}", ppisp_result.error());
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::export_per_frame_affine_color_parameters(
        const int iteration) const {
        if (!params_.optimization.save_per_frame_affine_color) {
            return {};
        }
        if (!params_.optimization.use_per_frame_affine_color || !per_frame_affine_color_) {
            return std::unexpected(
                "Cannot save per-frame affine color parameters: affine color is not initialized");
        }

        const auto path = params_.dataset.output_path / "per_frame_affine_color.json";
        if (auto result = per_frame_affine_color_->export_parameters_json(path, iteration); !result) {
            return result;
        }

        LOG_INFO("Saved {} per-frame affine color transforms to '{}'",
                 per_frame_affine_color_->num_frames(),
                 lfs::core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> Trainer::export_per_frame_observation_blur_parameters(
        const int iteration) const {
        if (!params_.optimization.save_per_frame_blur_parameters) {
            return {};
        }
        if ((!params_.optimization.use_per_frame_motion_blur &&
             !params_.optimization.use_per_frame_defocus_blur) ||
            !per_frame_observation_blur_) {
            return std::unexpected(
                "Cannot save per-frame observation blur parameters: blur is not initialized");
        }

        const auto path =
            params_.dataset.output_path / "per_frame_observation_blur.json";
        if (auto result =
                per_frame_observation_blur_->export_parameters_json(path, iteration);
            !result) {
            return result;
        }

        LOG_INFO("Saved {} per-frame observation blur parameter rows to '{}'",
                 per_frame_observation_blur_->num_frames(),
                 lfs::core::path_to_utf8(path));
        return {};
    }

    void Trainer::flush_pose_refine_outputs(const std::filesystem::path& output_path, const int iteration) {
        if (!params_.optimization.save_pose_refine_outputs ||
            !pose_refiner_ || !pose_refiner_->enabled() || !train_dataset_) {
            return;
        }

        const auto& cameras = train_dataset_->get_cameras();
        pose_refiner_->apply_to_cameras(cameras);
        if (auto result = pose_refiner_->export_refined_poses(output_path, cameras, iteration); !result) {
            LOG_WARN("Failed to export refined camera poses: {}", result.error());
        }
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        flush_pose_refine_outputs(params_.dataset.output_path, iteration);

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);
        const PerFrameAffineColor* affine_to_save =
            params_.optimization.use_per_frame_affine_color
                ? per_frame_affine_color_.get()
                : nullptr;
        const PerFrameObservationBlur* blur_to_save =
            (params_.optimization.use_per_frame_motion_blur ||
             params_.optimization.use_per_frame_defocus_blur)
                ? per_frame_observation_blur_.get()
                : nullptr;

        return lfs::training::save_checkpoint(params_.dataset.output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save,
                                              pose_refiner_.get(), affine_to_save, blur_to_save);
    }

    std::expected<void, std::string> Trainer::save_checkpoint_to(const std::filesystem::path& output_path,
                                                                 int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        flush_pose_refine_outputs(output_path, iteration);

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);
        const PerFrameAffineColor* affine_to_save =
            params_.optimization.use_per_frame_affine_color
                ? per_frame_affine_color_.get()
                : nullptr;
        const PerFrameObservationBlur* blur_to_save =
            (params_.optimization.use_per_frame_motion_blur ||
             params_.optimization.use_per_frame_defocus_blur)
                ? per_frame_observation_blur_.get()
                : nullptr;

        return lfs::training::save_checkpoint(output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save,
                                              pose_refiner_.get(), affine_to_save, blur_to_save);
    }

    lfs::core::Tensor Trainer::applyPPISPForViewport(const lfs::core::Tensor& rgb, const int camera_uid,
                                                     const PPISPViewportOverrides& overrides,
                                                     const bool use_controller) const {
        if (!ppisp_ || !params_.optimization.use_ppisp || rgb.shape().rank() != 3) {
            return rgb;
        }

        const bool is_chw = (rgb.shape()[0] == 3);
        const auto rgb_chw = is_chw ? rgb : rgb.permute({2, 0, 1}).contiguous();
        const bool is_training_camera = ppisp_->is_known_frame(camera_uid);
        const bool has_controller = ppisp_controller_pool_ && params_.optimization.ppisp_use_controller;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            constexpr int CONTROLLER_IDX = 0;
            const auto controller_params = ppisp_controller_pool_->predict(CONTROLLER_IDX, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, CONTROLLER_IDX)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, CONTROLLER_IDX,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
                                            : ppisp_->apply_with_overrides(rgb_chw, fallback_camera, fallback_frame,
                                                                           toRenderOverrides(overrides));
        }

        return is_chw ? result : result.permute({1, 2, 0}).contiguous();
    }

    PPISPControllerPool* Trainer::controller_pool_for_save(const int iteration) const {
        if (!ppisp_controller_pool_) {
            return nullptr;
        }
        if (is_ppisp_frozen()) {
            return ppisp_controller_pool_.get();
        }
        return iteration >= params_.optimization.resolved_ppisp_controller_activation_step()
                   ? ppisp_controller_pool_.get()
                   : nullptr;
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, iteration, /*join=*/true);
    }

    std::expected<int, std::string> Trainer::load_checkpoint(const std::filesystem::path& checkpoint_path) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        // Create PPISP before loading if needed
        if (params_.optimization.use_ppisp && !ppisp_) {
            if (auto init_result = initialize_ppisp(); !init_result) {
                LOG_WARN("Failed to init PPISP for resume: {}", init_result.error());
            }
        }

        // Create PPISP controller pool before loading if needed
        if (params_.optimization.ppisp_use_controller && !ppisp_controller_pool_) {
            bool should_initialize_controller = true;
            if (is_ppisp_frozen()) {
                const auto checkpoint_header = lfs::core::load_checkpoint_header(checkpoint_path);
                if (!checkpoint_header) {
                    LOG_WARN("Failed to inspect checkpoint header for PPISP controller state: {}",
                             checkpoint_header.error());
                    should_initialize_controller = false;
                } else {
                    should_initialize_controller =
                        lfs::core::has_flag(checkpoint_header->flags, lfs::core::CheckpointFlags::HAS_PPISP_CONTROLLER);
                    if (!should_initialize_controller) {
                        LOG_INFO("Checkpoint has no PPISP controller pool; frozen controller state remains disabled");
                    }
                }
            }
            if (should_initialize_controller) {
                if (auto init_result = initialize_ppisp_controller(); !init_result) {
                    LOG_WARN("Failed to init PPISP controller pool for resume: {}", init_result.error());
                }
            }
        }

        const auto checkpoint_header = lfs::core::load_checkpoint_header(checkpoint_path);
        if (!checkpoint_header) {
            return std::unexpected(checkpoint_header.error());
        }
        const auto checkpoint_params = lfs::core::load_checkpoint_params(checkpoint_path);
        if (!checkpoint_params) {
            return std::unexpected(checkpoint_params.error());
        }
        const bool checkpoint_has_pose_refiner =
            lfs::core::has_flag(checkpoint_header->flags, lfs::core::CheckpointFlags::HAS_POSE_REFINER);
        const bool checkpoint_has_per_frame_affine_color =
            lfs::core::has_flag(
                checkpoint_header->flags,
                lfs::core::CheckpointFlags::HAS_PER_FRAME_AFFINE_COLOR);
        const bool checkpoint_has_per_frame_observation_blur =
            lfs::core::has_flag(
                checkpoint_header->flags,
                lfs::core::CheckpointFlags::HAS_PER_FRAME_OBSERVATION_BLUR);
        auto affine_opt = checkpoint_params->optimization;
        if (checkpoint_has_per_frame_affine_color) {
            affine_opt.use_per_frame_affine_color = true;
        }
        if (auto affine_result = initialize_per_frame_affine_color(&affine_opt); !affine_result) {
            return std::unexpected(affine_result.error());
        }

        auto blur_opt = checkpoint_params->optimization;
        if (checkpoint_has_per_frame_observation_blur &&
            !blur_opt.use_per_frame_motion_blur &&
            !blur_opt.use_per_frame_defocus_blur) {
            // The serialized block carries the authoritative branch flags. A
            // temporary defocus-only state is sufficient to allocate and map
            // rows before deserialize(), without requiring a motion scene
            // radius for an old/hand-edited JSON that omitted both switches.
            blur_opt.use_per_frame_defocus_blur = true;
        }
        if (auto blur_result = initialize_per_frame_observation_blur(&blur_opt); !blur_result) {
            return std::unexpected(blur_result.error());
        }

        const bool checkpoint_wants_pose_refine =
            checkpoint_has_pose_refiner || checkpoint_params->optimization.refine_camera_pose;
        auto pose_refine_opt = checkpoint_params->optimization;
        if (checkpoint_has_pose_refiner) {
            pose_refine_opt.refine_camera_pose = true;
        }

        pose_refiner_.reset();
        if (checkpoint_wants_pose_refine) {
            if (pose_refine_opt.gut) {
                return std::unexpected("Camera pose refinement currently supports only FastGS; disable --gut");
            }
            pose_refiner_ = std::make_unique<PoseRefiner>(pose_refine_opt);
            if (!train_dataset_) {
                return std::unexpected("Cannot initialize pose refine state for checkpoint without a training dataset");
            }
            if (auto pose_result = pose_refiner_->initialize(train_dataset_->get_cameras()); !pose_result) {
                return std::unexpected(pose_result.error());
            }
        }

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get(), ppisp_.get(),
            ppisp_controller_pool_.get(), pose_refiner_.get(), per_frame_affine_color_.get(),
            per_frame_observation_blur_.get());
        if (!result) {
            return result;
        }
        if (ppisp_)
            params_.optimization.ppisp_exposure_only = ppisp_->get_config().exposure_only;
        if (checkpoint_has_per_frame_affine_color) {
            // The serialized component is authoritative even if an older or
            // hand-edited parameter JSON omitted the corresponding switch.
            params_.optimization.use_per_frame_affine_color = true;
        }
        if (checkpoint_has_per_frame_observation_blur &&
            per_frame_observation_blur_) {
            // The serialized component is authoritative even if an older or
            // hand-edited parameter JSON omitted one or both switches.
            params_.optimization.use_per_frame_motion_blur =
                per_frame_observation_blur_->config().motion_enabled;
            params_.optimization.use_per_frame_defocus_blur =
                per_frame_observation_blur_->config().defocus_enabled;
        }
        const bool restored_observation_blur_enabled =
            params_.optimization.use_per_frame_motion_blur ||
            params_.optimization.use_per_frame_defocus_blur;
        if (params_.optimization.save_per_frame_blur_parameters &&
            !restored_observation_blur_enabled) {
            return std::unexpected(
                "Saving per-frame blur parameters requires motion blur or defocus blur enabled");
        }
        if (restored_observation_blur_enabled) {
            if (params_.optimization.gut) {
                return std::unexpected(
                    "Per-frame motion/defocus blur currently supports only FastGS; disable gut");
            }
            if (params_.optimization.mip_filter) {
                return std::unexpected(
                    "Per-frame motion/defocus blur currently requires mip_filter=false until the mip covariance VJP is available");
            }
            if (params_.optimization.use_per_frame_defocus_blur &&
                params_.optimization.pyramid_training) {
                return std::unexpected(
                    "Per-frame defocus blur currently requires pyramid_training=false until beta is scaled with render resolution");
            }
            if (params_.optimization.enable_gggs_loss ||
                params_.optimization.gggs_gtnorm ||
                params_.optimization.enable_mesh_depth_loss ||
                params_.optimization.mesh_depth_visibility_cull) {
                return std::unexpected(
                    "Per-frame motion/defocus blur currently supports RGB supervision only; disable GGGS, mesh depth loss, and mesh depth visibility cull");
            }
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
