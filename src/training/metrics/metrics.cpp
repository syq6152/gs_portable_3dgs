/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metrics.hpp"
#include "../rasterization/fast_rasterizer.hpp"
#include "../rasterization/gsplat_rasterizer.hpp"
#include "../mesh_supervision_renderer.hpp"
#include "depth_normal_utils.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "lfs/kernels/ssim.cuh"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace lfs::training {

    namespace {
        struct TensorLayoutInfo {
            int n;
            int c;
            int h;
            int w;
        };

        struct EvalPyramidRenderState {
            bool enabled = false;
            int pyramid_level = 0;
            int coarse_levels = 0;
            int downsample_factor = 1;
            std::string level_token = "pyrLfull_1x";
        };

        class CameraIntrinsicsRestoreGuard {
        public:
            explicit CameraIntrinsicsRestoreGuard(lfs::core::Camera* camera)
                : camera_(camera) {
            }

            void arm() noexcept {
                restore_needed_ = true;
            }

            ~CameraIntrinsicsRestoreGuard() {
                if (restore_needed_ && camera_) {
                    camera_->restore_intrinsics();
                }
            }

        private:
            lfs::core::Camera* camera_ = nullptr;
            bool restore_needed_ = false;
        };

        EvalPyramidRenderState resolve_eval_pyramid_render_state(
            const lfs::core::param::TrainingParameters& params,
            const int iteration) {

            EvalPyramidRenderState state;
            if (!params.optimization.pyramid_training) {
                return state;
            }

            const int configured_levels = std::max(1, params.optimization.pyramid_levels);
            const int coarse_levels = std::max(0, configured_levels - 1);
            if (coarse_levels == 0) {
                state.enabled = true;
                state.level_token = "pyrLfull_1x";
                return state;
            }

            const int step_interval = std::max(1, params.optimization.pyramid_step_interval);
            const int iter_zero_based = std::max(0, iteration - 1);
            const int pyramid_level = std::min(iter_zero_based / step_interval, coarse_levels);

            state.enabled = true;
            state.pyramid_level = pyramid_level;
            state.coarse_levels = coarse_levels;

            if (pyramid_level < coarse_levels) {
                const int shift = std::max(0, coarse_levels - pyramid_level);
                if (shift >= static_cast<int>(sizeof(int) * 8 - 1)) {
                    state.downsample_factor = std::numeric_limits<int>::max();
                } else {
                    state.downsample_factor = 1 << shift;
                }
                state.level_token = "pyrL" + std::to_string(pyramid_level) + "_" +
                                    std::to_string(state.downsample_factor) + "x";
            } else {
                state.downsample_factor = 1;
                state.level_token = "pyrLfull_1x";
            }

            return state;
        }

        TensorLayoutInfo get_layout_info(const lfs::core::Tensor& t, const char* name) {
            if (t.ndim() == 3) {
                return {
                    .n = 1,
                    .c = static_cast<int>(t.shape()[0]),
                    .h = static_cast<int>(t.shape()[1]),
                    .w = static_cast<int>(t.shape()[2])};
            }
            if (t.ndim() == 4) {
                return {
                    .n = static_cast<int>(t.shape()[0]),
                    .c = static_cast<int>(t.shape()[1]),
                    .h = static_cast<int>(t.shape()[2]),
                    .w = static_cast<int>(t.shape()[3])};
            }

            throw std::runtime_error(std::string(name) + ": expected tensor rank 3 or 4");
        }

        float get_non_empty_mask_sum_or_throw(const lfs::core::Tensor& mask, const char* name) {
            const float mask_sum = mask.sum().item<float>();
            if (!std::isfinite(mask_sum) || mask_sum <= 0.0f) {
                throw std::runtime_error(std::string(name) + ": mask is empty or invalid");
            }
            return mask_sum;
        }

        void validate_mask_shape_or_throw(const lfs::core::Tensor& mask,
                                          const TensorLayoutInfo& layout,
                                          const char* name) {
            if (mask.ndim() != 2) {
                throw std::runtime_error(std::string(name) + ": expected 2D mask [H, W]");
            }

            const int mask_h = static_cast<int>(mask.shape()[0]);
            const int mask_w = static_cast<int>(mask.shape()[1]);
            if (mask_h != layout.h || mask_w != layout.w) {
                throw std::runtime_error(
                    std::string(name) + ": mask shape does not match image shape");
            }
        }
        lfs::core::Tensor expand_mask(const lfs::core::Tensor& mask,
                                      const TensorLayoutInfo& layout,
                                      int target_ndim) {
            assert(mask.ndim() == 2);
            assert(target_ndim == 3 || target_ndim == 4);
            if (target_ndim == 3) {
                return mask.unsqueeze(0).expand({layout.c, layout.h, layout.w});
            }
            return mask.unsqueeze(0).unsqueeze(0).expand({layout.n, layout.c, layout.h, layout.w});
        }
    } // namespace

    float PSNR::compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                        const lfs::core::Tensor& mask) const {
        if (pred.shape() != target.shape()) {
            throw std::runtime_error("PSNR: prediction and target must have the same shape");
        }

        const auto layout = get_layout_info(pred, "PSNR");

        auto squared_diff = (pred - target).square();

        float mse;
        if (mask.is_valid()) {
            validate_mask_shape_or_throw(mask, layout, "PSNR");
            const float mask_sum = get_non_empty_mask_sum_or_throw(mask, "PSNR");

            const auto expanded = expand_mask(mask, layout, pred.ndim());
            const auto weighted_sum = (squared_diff * expanded).sum();

            const float denom = mask_sum * static_cast<float>(layout.c * layout.n);
            mse = weighted_sum.item<float>() / denom;
        } else {
            mse = squared_diff.mean().item<float>();
        }

        if (!std::isfinite(mse)) {
            throw std::runtime_error("PSNR: produced non-finite MSE");
        }

        if (mse < 1e-10f)
            mse = 1e-10f;

        return 20.0f * std::log10(data_range_ / std::sqrt(mse));
    }

    // SSIM Implementation using LibTorch-free kernels
    SSIM::SSIM(bool apply_valid_padding)
        : apply_valid_padding_(apply_valid_padding) {
    }

    float SSIM::compute(const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
                        const lfs::core::Tensor& mask) {
        if (pred.shape() != target.shape()) {
            throw std::runtime_error("SSIM: prediction and target must have the same shape");
        }

        if (mask.is_valid()) {
            // Match masked training semantics: no valid-padding crop, masked mean over all pixels.
            const auto layout = get_layout_info(pred, "SSIM");
            validate_mask_shape_or_throw(mask, layout, "SSIM");
            const float mask_sum = get_non_empty_mask_sum_or_throw(mask, "SSIM");

            auto map_result = kernels::ssim_forward_map(pred, target, false);
            auto ssim_map = map_result.ssim_map;
            assert(ssim_map.ndim() == 4);
            assert(static_cast<int>(ssim_map.shape()[2]) == layout.h);
            assert(static_cast<int>(ssim_map.shape()[3]) == layout.w);

            const auto expanded = expand_mask(mask, layout, 4);
            const auto weighted_sum = (ssim_map * expanded).sum();

            const float denom = mask_sum * static_cast<float>(layout.c * layout.n);
            const float masked_ssim = weighted_sum.item<float>() / denom;
            if (!std::isfinite(masked_ssim)) {
                throw std::runtime_error("SSIM: produced non-finite masked SSIM");
            }
            return masked_ssim;
        }

        auto [ssim_value, ctx] = kernels::ssim_forward(pred, target, apply_valid_padding_);
        const float value = ssim_value.mean().item<float>();
        if (!std::isfinite(value)) {
            throw std::runtime_error("SSIM: produced non-finite SSIM");
        }
        return value;
    }

    // MetricsReporter Implementation
    MetricsReporter::MetricsReporter(const std::filesystem::path& output_dir)
        : output_dir_(output_dir),
          csv_path_(output_dir_ / "metrics.csv"),
          txt_path_(output_dir_ / "metrics_report.txt") {
        // Create CSV header if file doesn't exist
        if (!std::filesystem::exists(csv_path_)) {
            std::ofstream csv_file;
            if (lfs::core::open_file_for_write(csv_path_, csv_file)) {
                csv_file << EvalMetrics{}.to_csv_header() << std::endl;
                csv_file.close();
            }
        }
    }

    void MetricsReporter::add_metrics(const EvalMetrics& metrics) {
        all_metrics_.push_back(metrics);

        // Append to CSV immediately
        std::ofstream csv_file;
        if (lfs::core::open_file_for_write(csv_path_, std::ios::app, csv_file)) {
            csv_file << metrics.to_csv_row() << std::endl;
            csv_file.close();
        }
    }

    void MetricsReporter::save_report() const {
        std::ofstream report_file;
        if (!lfs::core::open_file_for_write(txt_path_, report_file)) {
            std::cerr << "Failed to open report file: " << lfs::core::path_to_utf8(txt_path_) << std::endl;
            return;
        }

        // Write header
        report_file << "==============================================\n";
        report_file << "3D Gaussian Splatting Evaluation Report\n";
        report_file << "==============================================\n";
        report_file << "Output Directory: " << lfs::core::path_to_utf8(output_dir_) << "\n";

        // Get current time
        const auto now = std::chrono::system_clock::now();
        const auto time_t = std::chrono::system_clock::to_time_t(now);
        report_file << "Generated: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n\n";

        // Summary statistics
        if (!all_metrics_.empty()) {
            report_file << "Summary Statistics:\n";
            report_file << "------------------\n";

            // Find best metrics
            const auto best_psnr = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                    [](const EvalMetrics& a, const EvalMetrics& b) {
                                                        return a.psnr < b.psnr;
                                                    });
            const auto best_ssim = std::max_element(all_metrics_.begin(), all_metrics_.end(),
                                                    [](const EvalMetrics& a, const EvalMetrics& b) {
                                                        return a.ssim < b.ssim;
                                                    });

            report_file << std::fixed << std::setprecision(4);
            report_file << "Best PSNR:  " << best_psnr->psnr << " (at iteration " << best_psnr->iteration << ")\n";
            report_file << "Best SSIM:  " << best_ssim->ssim << " (at iteration " << best_ssim->iteration << ")\n";

            // Final metrics
            const auto& final = all_metrics_.back();
            report_file << "\nFinal Metrics (iteration " << final.iteration << "):\n";
            report_file << "PSNR:  " << final.psnr << "\n";
            report_file << "SSIM:  " << final.ssim << "\n";
            report_file << "Time per image: " << final.elapsed_time << " seconds\n";
            report_file << "Number of Gaussians: " << final.num_gaussians << "\n";
        }

        // Detailed results
        report_file << "\nDetailed Results:\n";
        report_file << "-----------------\n";
        report_file << std::setw(10) << "Iteration"
                    << std::setw(10) << "PSNR"
                    << std::setw(10) << "SSIM"
                    << std::setw(15) << "Time(s/img)"
                    << std::setw(15) << "#Gaussians"
                    << "\n";
        report_file << std::string(60, '-') << "\n";

        for (const auto& m : all_metrics_) {
            report_file << std::setw(10) << m.iteration
                        << std::setw(10) << std::fixed << std::setprecision(4) << m.psnr
                        << std::setw(10) << m.ssim
                        << std::setw(15) << m.elapsed_time
                        << std::setw(15) << m.num_gaussians << "\n";
        }

        report_file.close();
        std::cout << "Evaluation report saved to: " << lfs::core::path_to_utf8(txt_path_) << std::endl;
        std::cout << "Metrics CSV saved to: " << lfs::core::path_to_utf8(csv_path_) << std::endl;
    }

    // MetricsEvaluator Implementation
    MetricsEvaluator::MetricsEvaluator(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene* scene)
        : _params(params),
          _scene(scene) {
        if (!params.optimization.enable_eval) {
            return;
        }

        // Initialize metrics
        _psnr_metric = std::make_unique<PSNR>(1.0f);
        _ssim_metric = std::make_unique<SSIM>(true); // apply_valid_padding = true

        // Initialize reporter
        _reporter = std::make_unique<MetricsReporter>(params.dataset.output_path);
    }

    bool MetricsEvaluator::should_evaluate(const int iteration) const {
        if (!_params.optimization.enable_eval)
            return false;

        return std::find(_params.optimization.eval_steps.cbegin(), _params.optimization.eval_steps.cend(), iteration) !=
               _params.optimization.eval_steps.cend();
    }

    lfs::core::Tensor MetricsEvaluator::apply_depth_colormap(const lfs::core::Tensor& depth_normalized) const {
        return lfs::training::colorize_depth_map(depth_normalized);
    }

    lfs::core::Tensor MetricsEvaluator::load_eval_mask(lfs::core::Camera* cam,
                                                       lfs::core::Tensor& gt_image,
                                                       const bool alpha_as_mask) const {
        if (cam->has_mask()) {
            return cam->load_and_get_mask(
                _params.dataset.resize_factor,
                _params.dataset.max_width,
                _params.optimization.invert_masks,
                _params.optimization.mask_threshold);
        }

        if (!alpha_as_mask)
            return {};

        // Re-load from disk because the dataloader strips alpha to produce RGB gt_image.
        // We need the original alpha channel as the mask, with undistortion applied consistently.
        auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
            cam->image_path(), _params.dataset.resize_factor, _params.dataset.max_width);

        if (!img_data || channels != 4) {
            if (img_data)
                lfs::core::free_image(img_data);
            return {};
        }

        const auto H = static_cast<size_t>(height);
        const auto W = static_cast<size_t>(width);

        auto cpu_tensor = lfs::core::Tensor::from_blob(
            img_data, lfs::core::TensorShape({H, W, 4}),
            lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
        lfs::core::free_image(img_data);

        auto rgb = lfs::core::Tensor::zeros(
            lfs::core::TensorShape({3, H, W}),
            lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto mask = lfs::core::Tensor::zeros(
            lfs::core::TensorShape({H, W}),
            lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        lfs::io::cuda::launch_uint8_rgba_split_to_float32_rgb_and_alpha(
            gpu_uint8.ptr<uint8_t>(), rgb.ptr<float>(), mask.ptr<float>(),
            H, W, nullptr);
        gpu_uint8 = lfs::core::Tensor();

        if (_params.optimization.invert_masks)
            lfs::io::cuda::launch_mask_invert(mask.ptr<float>(), H, W, nullptr);
        if (_params.optimization.mask_threshold > 0)
            lfs::io::cuda::launch_mask_threshold(
                mask.ptr<float>(), H, W, _params.optimization.mask_threshold, nullptr);

        if (cam->is_undistort_prepared()) {
            const auto scaled = lfs::core::scale_undistort_params(
                cam->undistort_params(),
                static_cast<int>(W), static_cast<int>(H));
            rgb = lfs::core::undistort_image(rgb, scaled, nullptr);
            mask = lfs::core::undistort_mask(mask, scaled, nullptr);
        }

        gt_image = std::move(rgb);
        return mask;
    }

    auto MetricsEvaluator::make_dataloader(std::shared_ptr<CameraDataset> dataset, const int workers) const {
        return create_dataloader_from_dataset(dataset, workers);
    }

    EvalMetrics MetricsEvaluator::evaluate(const int iteration,
                                           const lfs::core::SplatData& splatData,
                                           std::shared_ptr<CameraDataset> val_dataset,
                                           lfs::core::Tensor& background) {
        if (!_params.optimization.enable_eval) {
            throw std::runtime_error("Evaluation is not enabled");
        }

        EvalMetrics result;
        result.num_gaussians = static_cast<int>(splatData.size());
        result.iteration = iteration;

        const auto val_dataloader = make_dataloader(val_dataset);

        std::vector<float> psnr_values, ssim_values;
        const auto start_time = std::chrono::steady_clock::now();

        // Create directory for evaluation images
        const std::filesystem::path eval_dir = _params.dataset.output_path /
                                               ("eval_step_" + std::to_string(iteration));
        if (_params.optimization.enable_save_eval_images) {
            std::filesystem::create_directories(eval_dir);
        }

        int image_idx = 0;
        const size_t val_dataset_size = val_dataset->size();
        size_t skipped_images = 0;
        size_t evaluated_images = 0;

        const auto mask_mode = _params.optimization.mask_mode;
        const bool use_masking = mask_mode == lfs::core::param::MaskMode::Segment ||
                                 mask_mode == lfs::core::param::MaskMode::Ignore ||
                                 mask_mode == lfs::core::param::MaskMode::ProjectMesh;
        const bool save_depth_outputs = _params.optimization.save_depth;
        const auto eval_pyramid_state = resolve_eval_pyramid_render_state(_params, iteration);

        if (_params.optimization.enable_save_eval_images && eval_pyramid_state.enabled) {
            LOG_INFO("[DepthDebug][eval] Train-intrinsics render enabled: level_token='{}' (iteration={})",
                     eval_pyramid_state.level_token,
                     iteration);
        }

        while (auto batch_opt = val_dataloader->next()) {
            auto& batch = *batch_opt;
            auto camera_with_image = batch[0].data;
            lfs::core::Camera* cam = camera_with_image.camera;
            lfs::core::Tensor gt_image = std::move(camera_with_image.image);

            if (gt_image.device() != lfs::core::Device::CUDA) {
                gt_image = gt_image.to(lfs::core::Device::CUDA);
            }

            lfs::core::Tensor mask;
            if (use_masking) {
                const bool cam_alpha = _params.optimization.use_alpha_as_mask && cam->has_alpha();
                try {
                    mask = load_eval_mask(cam, gt_image, cam_alpha);
                } catch (const std::exception& e) {
                    LOG_WARN("Eval: skipping camera '{}' (failed to load mask: {})", cam->image_name(), e.what());
                    skipped_images++;
                    continue;
                }

                if (!mask.is_valid()) {
                    LOG_DEBUG("Eval: camera '{}' has no mask, proceeding unmasked", cam->image_name());
                    mask = lfs::core::Tensor();
                }
            }

            auto& splatData_mutable = const_cast<lfs::core::SplatData&>(splatData);
            RenderOutput r_output;
            if (_params.optimization.gut) {
                const auto render_mode = save_depth_outputs ? GsplatRenderMode::RGB_ED : GsplatRenderMode::RGB;
                r_output = gsplat_rasterize(*cam, splatData_mutable, background,
                                            1.0f, false, render_mode, true);
            } else {
                auto rasterize_result = fast_rasterize_forward(*cam, splatData_mutable, background,
                                                               0, 0, 0, 0,
                                                               _params.optimization.mip_filter,
                                                               {},
                                                               save_depth_outputs);
                if (!rasterize_result) {
                    throw std::runtime_error("Evaluation rasterization failed: " + rasterize_result.error());
                }
                r_output = std::move(rasterize_result->first);
            }
            r_output.image = r_output.image.clamp(0.0f, 1.0f);

            float psnr = 0.0f;
            float ssim = 0.0f;
            try {
                psnr = _psnr_metric->compute(r_output.image, gt_image, mask);
                ssim = _ssim_metric->compute(r_output.image, gt_image, mask);
            } catch (const std::exception& e) {
                LOG_WARN("Eval: skipping camera '{}' (metric computation failed: {})", cam->image_name(), e.what());
                skipped_images++;
                continue;
            }

            if (!std::isfinite(psnr) || !std::isfinite(ssim)) {
                LOG_WARN("Eval: skipping camera '{}' (non-finite metric values: PSNR={}, SSIM={})",
                         cam->image_name(), psnr, ssim);
                skipped_images++;
                continue;
            }

            psnr_values.push_back(psnr);
            ssim_values.push_back(ssim);
            evaluated_images++;

            if (_params.optimization.enable_save_eval_images) {
                auto gt_vis = gt_image;
                auto render_vis = r_output.image;
                if (mask.is_valid()) {
                    const int C = static_cast<int>(gt_image.shape()[0]);
                    const int H = static_cast<int>(mask.shape()[0]);
                    const int W = static_cast<int>(mask.shape()[1]);
                    auto mask_3d = mask.unsqueeze(0).expand({C, H, W});
                    gt_vis = gt_image * mask_3d;
                    render_vis = r_output.image * mask_3d;
                }
                const std::vector<lfs::core::Tensor> rgb_images = {gt_vis, render_vis};
                lfs::core::image_io::save_images_async(
                    eval_dir / (std::to_string(image_idx) + ".png"),
                    rgb_images,
                    true, // horizontal
                    4);   // separator width

                const bool depth_valid = r_output.depth.is_valid();
                const bool depth_empty = depth_valid ? r_output.depth.is_empty() : true;
                if (save_depth_outputs) {
                    LOG_INFO("[DepthDebug][eval] image_idx={} camera='{}' depth_valid={} depth_empty={}",
                             image_idx,
                             cam->image_name(),
                             depth_valid,
                             depth_empty);
                }

                if (save_depth_outputs && depth_valid && !depth_empty) {
                    const auto [fx, fy, cx, cy] = cam->get_intrinsics();

                    auto depth_normal_map = depth_to_normal_map_gpu(r_output.depth, fx, fy, cx, cy);

                    const auto depth_vis = colorize_depth_map(r_output.depth);
                    const auto depth_normal_vis = colorize_normal_map(depth_normal_map);

                    std::vector<lfs::core::Tensor> depth_normal_panels = {depth_vis, depth_normal_vis};

                    // Append render_normal (alpha-blended per-Gaussian normal) when --gggs is active
                    const bool render_normal_valid = r_output.render_normal.is_valid() && !r_output.render_normal.is_empty();
                    if (_params.optimization.enable_gggs_loss && render_normal_valid) {
                        const auto render_normal_vis = colorize_normal_map(r_output.render_normal);
                        depth_normal_panels.push_back(render_normal_vis);
                    }

                    // Render mesh GT targets once (reused for concat and separate save)
                    std::optional<MeshSupervisionTargets> mesh_gt_targets;
                    if (_params.optimization.precompute_mesh_depth_normal && _scene) {
                        auto mesh_gt_result = render_mesh_supervision_targets_for_camera(
                            *_scene,
                            *cam,
                            _params);
                        if (!mesh_gt_result) {
                            LOG_WARN("[DepthDebug][eval] mesh GT render failed for camera '{}' : {}",
                                     cam->image_name(),
                                     mesh_gt_result.error());
                        } else {
                            mesh_gt_targets = std::move(*mesh_gt_result);
                        }
                    }

                    // Append GT normal when --gggs-gtnorm is active
                    if (_params.optimization.gggs_gtnorm && mesh_gt_targets &&
                        mesh_gt_targets->normal.is_valid() && !mesh_gt_targets->normal.is_empty()) {
                        const auto gt_normal_vis = colorize_normal_map(mesh_gt_targets->normal);
                        depth_normal_panels.push_back(gt_normal_vis);
                    }

                    const auto depth_normal_path = eval_dir / (std::to_string(image_idx) + "_depth_normal.png");
                    LOG_INFO("[DepthDebug][eval] queue save depth+normal concat='{}'",
                             lfs::core::path_to_utf8(depth_normal_path));

                    lfs::core::image_io::save_images_async(
                        depth_normal_path,
                        depth_normal_panels,
                        true, // horizontal
                        4);   // separator width

                    // Save separate mesh GT depth+normal concat
                    if (mesh_gt_targets) {
                        const auto mesh_depth_vis = colorize_depth_map(mesh_gt_targets->depth);
                        const auto mesh_normal_vis = colorize_normal_map(mesh_gt_targets->normal);
                        const auto mesh_depth_normal_path = eval_dir / (std::to_string(image_idx) + "_mesh_gt_depth_normal.png");
                        LOG_INFO("[DepthDebug][eval] queue save mesh GT depth+normal concat='{}'",
                                 lfs::core::path_to_utf8(mesh_depth_normal_path));

                        lfs::core::image_io::save_images_async(
                            mesh_depth_normal_path,
                            {mesh_depth_vis, mesh_normal_vis},
                            true,
                            4);
                    }
                } else if (save_depth_outputs) {
                    LOG_INFO("[DepthDebug][eval] skip depth save: depth tensor invalid or empty");
                }

                if (eval_pyramid_state.enabled) {
                    try {
                        CameraIntrinsicsRestoreGuard intrinsics_guard(cam);
                        const auto [base_fx, base_fy, base_cx, base_cy] = cam->get_intrinsics();
                        const int base_w = cam->image_width();
                        const int base_h = cam->image_height();

                        if (eval_pyramid_state.pyramid_level < eval_pyramid_state.coarse_levels) {
                            const int factor = std::max(1, eval_pyramid_state.downsample_factor);
                            const float scale = 1.0f / static_cast<float>(factor);
                            const int pyr_w = std::max(1, base_w / factor);
                            const int pyr_h = std::max(1, base_h / factor);
                            cam->set_intrinsics_for_pyramid(
                                base_fx * scale,
                                base_fy * scale,
                                base_cx * scale,
                                base_cy * scale,
                                pyr_w,
                                pyr_h);
                            intrinsics_guard.arm();
                        }

                        RenderOutput train_intr_output;
                        if (_params.optimization.gut) {
                            train_intr_output = gsplat_rasterize(
                                *cam,
                                splatData_mutable,
                                background,
                                1.0f,
                                false,
                                GsplatRenderMode::RGB_ED,
                                true);
                        } else {
                            auto train_intr_rasterize_result = fast_rasterize_forward(
                                *cam,
                                splatData_mutable,
                                background,
                                0,
                                0,
                                0,
                                0,
                                _params.optimization.mip_filter,
                                {},
                                true);
                            if (!train_intr_rasterize_result) {
                                LOG_WARN("[DepthDebug][eval] train-intrinsics rasterization failed for camera '{}': {}",
                                         cam->image_name(),
                                         train_intr_rasterize_result.error());
                            } else {
                                train_intr_output = std::move(train_intr_rasterize_result->first);
                            }
                        }

                        if (train_intr_output.image.is_valid() && !train_intr_output.image.is_empty()) {
                            const auto train_intr_rgb = train_intr_output.image.clamp(0.0f, 1.0f);
                            const auto train_intr_rgb_path = eval_dir /
                                                             (std::to_string(image_idx) +
                                                              "_train_intrinsics_" +
                                                              eval_pyramid_state.level_token +
                                                              "_rgb.png");
                            lfs::core::image_io::save_image_async(train_intr_rgb_path, train_intr_rgb);
                        } else {
                            LOG_INFO("[DepthDebug][eval] train-intrinsics RGB skip: invalid image tensor for camera '{}'",
                                     cam->image_name());
                        }

                        const bool train_depth_valid = train_intr_output.depth.is_valid();
                        const bool train_depth_empty = train_depth_valid ? train_intr_output.depth.is_empty() : true;
                        if (train_depth_valid && !train_depth_empty) {
                            const auto [pyr_fx, pyr_fy, pyr_cx, pyr_cy] = cam->get_intrinsics();
                            auto train_depth_normal = depth_to_normal_map_gpu(
                                train_intr_output.depth,
                                pyr_fx,
                                pyr_fy,
                                pyr_cx,
                                pyr_cy);

                            const auto train_depth_vis = colorize_depth_map(train_intr_output.depth);
                            const auto train_depth_normal_vis = colorize_normal_map(train_depth_normal);

                            std::vector<lfs::core::Tensor> train_panels = {train_depth_vis, train_depth_normal_vis};

                            // Append render_normal when --gggs is active
                            const bool train_render_normal_valid = train_intr_output.render_normal.is_valid() && !train_intr_output.render_normal.is_empty();
                            if (_params.optimization.enable_gggs_loss && train_render_normal_valid) {
                                const auto train_render_normal_vis = colorize_normal_map(train_intr_output.render_normal);
                                train_panels.push_back(train_render_normal_vis);
                            }

                            // Append GT normal when --gggs-gtnorm is active
                            if (_params.optimization.gggs_gtnorm && _params.optimization.precompute_mesh_depth_normal && _scene) {
                                auto train_mesh_gt_result = render_mesh_supervision_targets_for_camera(
                                    *_scene,
                                    *cam,
                                    _params);
                                if (train_mesh_gt_result && train_mesh_gt_result->normal.is_valid() && !train_mesh_gt_result->normal.is_empty()) {
                                    const auto train_gt_normal_vis = colorize_normal_map(train_mesh_gt_result->normal);
                                    train_panels.push_back(train_gt_normal_vis);
                                }
                            }

                            const auto train_intr_depth_normal_path = eval_dir /
                                                                       (std::to_string(image_idx) +
                                                                        "_train_intrinsics_" +
                                                                        eval_pyramid_state.level_token +
                                                                        "_depth_normal.png");

                            lfs::core::image_io::save_images_async(
                                train_intr_depth_normal_path,
                                train_panels,
                                true,
                                4);
                        } else {
                            LOG_INFO("[DepthDebug][eval] train-intrinsics depth/normal skip: invalid or empty depth for camera '{}'",
                                     cam->image_name());
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("[DepthDebug][eval] train-intrinsics render/save failed for camera '{}': {}",
                                 cam->image_name(),
                                 e.what());
                    }
                }
            }

            image_idx++;
        }

        // Wait for all images to be saved before computing final timing
        if (_params.optimization.enable_save_eval_images) {
            const auto pending = lfs::core::image_io::BatchImageSaver::instance().pending_count();
            if (pending > 0) {
                lfs::core::image_io::wait_for_pending_saves();
            }
        }

        const auto end_time = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<float>(end_time - start_time).count();

        // Compute averages
        if (!psnr_values.empty()) {
            result.psnr = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
            result.ssim = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
        }
        const size_t elapsed_denom = evaluated_images > 0 ? evaluated_images : std::max<size_t>(val_dataset_size, 1);
        result.elapsed_time = elapsed / static_cast<float>(elapsed_denom);

        if (skipped_images > 0) {
            LOG_WARN("Eval: skipped {} / {} images due to mask/metric failures", skipped_images, val_dataset_size);
        }
        if (evaluated_images == 0) {
            LOG_WARN("Eval: no images were successfully evaluated at iteration {}", iteration);
        }

        // Add metrics to reporter
        _reporter->add_metrics(result);

        if (_params.optimization.enable_save_eval_images) {
            std::cout << "Saved " << image_idx << " evaluation images to: " << lfs::core::path_to_utf8(eval_dir) << std::endl;
        }

        return result;
    }
} // namespace lfs::training
