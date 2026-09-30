/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/logger.hpp"
#include "core/path_crypto.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>

#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace lfs::core {
    namespace param {
        std::string_view mesh2splat_hole_fill_mode_name(
            const Mesh2SplatHoleFillMode mode) {
            switch (mode) {
            case Mesh2SplatHoleFillMode::None:
                return "none";
            case Mesh2SplatHoleFillMode::Cgal:
                return "cgal";
            case Mesh2SplatHoleFillMode::External:
                return "external";
            case Mesh2SplatHoleFillMode::ExternalPointCloud:
                return "external_pointcloud";
            }
            return "none";
        }

        std::expected<Mesh2SplatHoleFillMode, std::string>
        parse_mesh2splat_hole_fill_mode(const std::string_view value) {
            if (value == "none")
                return Mesh2SplatHoleFillMode::None;
            if (value == "cgal")
                return Mesh2SplatHoleFillMode::Cgal;
            if (value == "external")
                return Mesh2SplatHoleFillMode::External;
            if (value == "external_pointcloud")
                return Mesh2SplatHoleFillMode::ExternalPointCloud;
            return std::unexpected(std::format(
                "Invalid mesh2splat hole-fill mode '{}'; expected none, cgal, external, or external_pointcloud",
                value));
        }

        std::string_view mesh_init_color_source_name(
            const MeshInitColorSource source) {
            switch (source) {
            case MeshInitColorSource::Auto:
                return "auto";
            case MeshInitColorSource::Texture:
                return "texture";
            case MeshInitColorSource::RgbViews:
                return "rgb_views";
            case MeshInitColorSource::White:
                return "white";
            }
            return "auto";
        }

        std::expected<MeshInitColorSource, std::string>
        parse_mesh_init_color_source(const std::string_view value) {
            if (value == "auto")
                return MeshInitColorSource::Auto;
            if (value == "texture")
                return MeshInitColorSource::Texture;
            if (value == "rgb_views")
                return MeshInitColorSource::RgbViews;
            if (value == "white")
                return MeshInitColorSource::White;
            return std::unexpected(std::format(
                "Invalid mesh init color source '{}'; expected auto, texture, rgb_views, or white",
                value));
        }

        Mesh2SplatHoleFillMode OptimizationParameters::resolved_mesh2splat_hole_fill_mode() const {
            if (mesh2splat_hole_fill_mode != Mesh2SplatHoleFillMode::None) {
                return mesh2splat_hole_fill_mode;
            }
            return mesh2splat_hole_fill_enabled
                       ? Mesh2SplatHoleFillMode::Cgal
                       : Mesh2SplatHoleFillMode::None;
        }

        namespace {
            std::expected<nlohmann::json, std::string> parse_json_text(
                const std::string& text,
                const std::filesystem::path& path) {
                try {
                    return nlohmann::json::parse(text);
                } catch (const nlohmann::json::parse_error& e) {
                    return std::unexpected(std::format("JSON parse error in {}: {}", path_to_utf8(path), e.what()));
                }
            }

            std::expected<nlohmann::json, std::string> read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Config file not found: {}", path_to_utf8(path)));
                }

                std::ifstream file;
                if (!open_file_for_read(path, file)) {
                    return std::unexpected(std::format("Cannot open config: {}", path_to_utf8(path)));
                }

                std::stringstream buffer;
                buffer << file.rdbuf();
                return parse_json_text(buffer.str(), path);
            }

            std::expected<nlohmann::json, std::string> read_encrypted_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Config file not found: {}", path_to_utf8(path)));
                }

                std::ifstream file;
                if (!open_file_for_read(path, std::ios::in | std::ios::binary, file)) {
                    return std::unexpected(std::format("Cannot open encrypted config: {}", path_to_utf8(path)));
                }

                const std::string bytes(
                    (std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
                const auto encrypted = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(bytes.data()),
                    bytes.size());

                const auto json_text = lfs::core::path_crypto::decrypt_text_from_binary(encrypted);
                if (!json_text) {
                    return std::unexpected(std::format(
                        "Encrypted config decrypt failed in {}: {}",
                        path_to_utf8(path),
                        json_text.error()));
                }

                return parse_json_text(*json_text, path);
            }

            std::string to_lower_ascii(std::string value) {
                std::transform(value.begin(), value.end(), value.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return value;
            }

            bool is_valid_log_level(std::string value) {
                value = to_lower_ascii(value);
                return value == "trace" ||
                       value == "debug" ||
                       value == "info" ||
                       value == "perf" ||
                       value == "performance" ||
                       value == "warn" ||
                       value == "warning" ||
                       value == "error" ||
                       value == "critical" ||
                       value == "off";
            }

            bool has_extension(const std::filesystem::path& path, const std::string_view extension) {
                return to_lower_ascii(path.extension().string()) == extension;
            }

            size_t count_files_with_extension(const std::filesystem::path& directory, const std::string_view extension) {
                std::error_code ec;
                if (!std::filesystem::exists(directory, ec) || !std::filesystem::is_directory(directory, ec)) {
                    return 0;
                }

                size_t count = 0;
                for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
                    if (ec) {
                        break;
                    }
                    if (!entry.is_regular_file()) {
                        continue;
                    }
                    if (has_extension(entry.path(), extension)) {
                        ++count;
                    }
                }
                return count;
            }

            std::string trim_ascii_whitespace(std::string value) {
                const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

                const auto begin = std::find_if_not(value.begin(), value.end(), is_space);
                const auto end = std::find_if_not(value.rbegin(), value.rend(), is_space).base();
                if (begin >= end) {
                    return {};
                }
                return std::string(begin, end);
            }

            std::string normalize_explicit_split_image_name(std::string value) {
                value = trim_ascii_whitespace(std::move(value));
                if (value.empty()) {
                    return {};
                }

                const auto normalized_path = lfs::core::utf8_to_path(value).filename();
                if (normalized_path.empty()) {
                    return value;
                }

                const auto stem = normalized_path.stem();
                if (stem.empty()) {
                    return lfs::core::path_to_utf8(normalized_path);
                }
                return lfs::core::path_to_utf8(stem);
            }

            std::vector<std::string> parse_explicit_split_images(
                const nlohmann::json& json,
                const char* field_name) {

                std::vector<std::string> images;
                if (!json.contains(field_name)) {
                    return images;
                }

                for (const auto& entry : json.at(field_name)) {
                    auto image_name = normalize_explicit_split_image_name(entry.get<std::string>());
                    if (!image_name.empty()) {
                        images.push_back(std::move(image_name));
                    }
                }

                return images;
            }
        } // namespace

        void OptimizationParameters::scale_steps(const float ratio) {
            const auto apply = [ratio](const size_t v) {
                return static_cast<size_t>(std::lround(static_cast<float>(v) * ratio));
            };
            iterations = apply(iterations);
            start_refine = apply(start_refine);
            stop_refine = apply(stop_refine);
            reset_every = apply(reset_every);
            refine_every = apply(refine_every);
            sh_degree_interval = apply(sh_degree_interval);
            pose_refine_lr_warmup_steps = apply(pose_refine_lr_warmup_steps);
            if (pose_refine_lr_max_steps > 0) {
                pose_refine_lr_max_steps = apply(pose_refine_lr_max_steps);
            }
            if (pose_refine_stop_iter > 0) {
                pose_refine_stop_iter = apply(pose_refine_stop_iter);
            }
            if (pose_refine_log_every > 0) {
                pose_refine_log_every = std::max<size_t>(1, apply(pose_refine_log_every));
            }

            for (auto* steps : {&eval_steps, &save_steps}) {
                std::set<size_t> unique;
                for (const auto s : *steps) {
                    if (const size_t scaled = apply(s); scaled > 0)
                        unique.insert(scaled);
                }
                steps->assign(unique.begin(), unique.end());
            }
        }

        void OptimizationParameters::apply_step_scaling() {
            if (steps_scaler <= 0.f || steps_scaler == 1.f)
                return;
            LOG_INFO("Scaling training steps by factor: {}", steps_scaler);
            scale_steps(steps_scaler);
        }

        void OptimizationParameters::remove_step_scaling() {
            if (steps_scaler <= 0.f || steps_scaler == 1.f)
                return;
            scale_steps(1.0f / steps_scaler);
        }

        int OptimizationParameters::resolved_ppisp_controller_activation_step() const {
            if (ppisp_controller_activation_step >= 0)
                return ppisp_controller_activation_step;

            const float clamped_scaler = std::max(steps_scaler, 1.0f);
            const int tail_iters = static_cast<int>(std::lround(5000.0f * clamped_scaler));
            return std::max(0, static_cast<int>(iterations) - tail_iters);
        }

        nlohmann::json OptimizationParameters::to_json() const {

            nlohmann::json opt_json;
            opt_json["iterations"] = iterations;
            opt_json["means_lr"] = means_lr;
            opt_json["shs_lr"] = shs_lr;
            opt_json["opacity_lr"] = opacity_lr;
            opt_json["scaling_lr"] = scaling_lr;
            opt_json["rotation_lr"] = rotation_lr;
            opt_json["lambda_dssim"] = lambda_dssim;
            opt_json["min_opacity"] = min_opacity;
            opt_json["refine_every"] = refine_every;
            opt_json["start_refine"] = start_refine;
            opt_json["stop_refine"] = stop_refine;
            opt_json["grad_threshold"] = grad_threshold;
            opt_json["sh_degree"] = sh_degree;
            opt_json["opacity_reg"] = opacity_reg;
            opt_json["scale_reg"] = scale_reg;
            opt_json["init_opacity"] = init_opacity;
            opt_json["init_scaling"] = init_scaling;
            opt_json["max_cap"] = max_cap;
            opt_json["refine_camera_pose"] = refine_camera_pose;
            opt_json["save_pose_refine_outputs"] = save_pose_refine_outputs;
            opt_json["pose_refine_mode"] = pose_refine_mode;
            opt_json["pose_refine_stop_iter"] = pose_refine_stop_iter;
            opt_json["pose_refine_lr_rot"] = pose_refine_lr_rot;
            opt_json["pose_refine_lr_trans"] = pose_refine_lr_trans;
            opt_json["pose_refine_lr_final_rot"] = pose_refine_lr_final_rot;
            opt_json["pose_refine_lr_final_trans"] = pose_refine_lr_final_trans;
            opt_json["pose_refine_lr_pre_warmup"] = pose_refine_lr_pre_warmup;
            opt_json["pose_refine_lr_max_steps"] = pose_refine_lr_max_steps;
            opt_json["pose_refine_lr_warmup_steps"] = pose_refine_lr_warmup_steps;
            opt_json["pose_refine_l2_rot"] = pose_refine_l2_rot;
            opt_json["pose_refine_l2_trans"] = pose_refine_l2_trans;
            opt_json["pose_refine_log_every"] = pose_refine_log_every;
            opt_json["pose_refine_max_rot_deg"] = pose_refine_max_rot_deg;
            opt_json["pose_refine_max_trans"] = pose_refine_max_trans;
            opt_json["mesh2splat_max_cap_extra_ratio"] = mesh2splat_max_cap_extra_ratio;
            opt_json["mesh2splat_opacity_no_grad"] = mesh2splat_opacity_no_grad;
            opt_json["mesh_init_color_source"] = mesh_init_color_source_name(mesh_init_color_source);
            const auto hole_fill_mode = resolved_mesh2splat_hole_fill_mode();
            opt_json["mesh2splat_hole_fill_mode"] = mesh2splat_hole_fill_mode_name(hole_fill_mode);
            opt_json["mesh2splat_hole_fill_enabled"] = hole_fill_mode == Mesh2SplatHoleFillMode::Cgal;
            opt_json["mesh2splat_hole_fill_max_boundary_edges"] = mesh2splat_hole_fill_max_boundary_edges;
            opt_json["mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio"] = mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio;
            opt_json["mesh2splat_hole_fill_max_boundary_area_bbox_ratio"] = mesh2splat_hole_fill_max_boundary_area_bbox_ratio;
            opt_json["mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio"] = mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio;
            opt_json["mesh2splat_hole_fill_weld_epsilon_bbox_ratio"] = mesh2splat_hole_fill_weld_epsilon_bbox_ratio;
            opt_json["mesh2splat_hole_fill_density_control_factor"] = mesh2splat_hole_fill_density_control_factor;
            opt_json["mesh2splat_hole_fill_min_density_ratio"] = mesh2splat_hole_fill_min_density_ratio;
            opt_json["mesh2splat_hole_fill_max_density_ratio"] = mesh2splat_hole_fill_max_density_ratio;
            opt_json["mesh2splat_external_band_ratio"] = mesh2splat_external_band_ratio;
            opt_json["mesh2splat_external_band_tolerance_ratio"] = mesh2splat_external_band_tolerance_ratio;
            opt_json["mesh2splat_pointcloud_density_ratio"] = mesh2splat_pointcloud_density_ratio;
            opt_json["mesh2splat_pointcloud_surface_reject_ratio"] = mesh2splat_pointcloud_surface_reject_ratio;
            opt_json["mesh2splat_pointcloud_mask_min_points_per_hole"] = mesh2splat_pointcloud_mask_min_points_per_hole;
            opt_json["mesh2splat_pointcloud_mask_rim_close_pixels"] = mesh2splat_pointcloud_mask_rim_close_pixels;
            opt_json["mesh2splat_pointcloud_mask_splat_radius_scale"] = mesh2splat_pointcloud_mask_splat_radius_scale;
            opt_json["mesh2splat_pointcloud_mask_close_pixels"] = mesh2splat_pointcloud_mask_close_pixels;
            opt_json["mesh_surface_walk_steps"] = mesh_surface_walk_steps;
            opt_json["mesh_surface_loss_from_iter"] = mesh_surface_loss_from_iter;
            opt_json["mesh_surface_hard_projection_enabled"] = mesh_surface_hard_projection_enabled;
            opt_json["mesh_depth_visibility_cull"] = mesh_depth_visibility_cull;
            opt_json["lambda_mesh_project"] = lambda_mesh_project;
            opt_json["lambda_mesh_outside_barrier"] = lambda_mesh_outside_barrier;
            opt_json["mesh_outside_distance_avg_max_scale_multiplier"] = mesh_outside_distance_avg_max_scale_multiplier;
            opt_json["mesh_inside_constraint_distance_avg_max_scale_multiplier"] = mesh_inside_constraint_distance_avg_max_scale_multiplier;
            opt_json["mesh_inside_constraint_fade_ratio"] = mesh_inside_constraint_fade_ratio;
            opt_json["lambda_mesh_scale_min"] = lambda_mesh_scale_min;
            opt_json["lambda_mesh_scale_max"] = lambda_mesh_scale_max;
            opt_json["mesh_scale_rho_ratio"] = mesh_scale_rho_ratio;
            opt_json["mesh_scale_rho_avg_max_scale_multiplier"] = mesh_scale_rho_avg_max_scale_multiplier;
            opt_json["lambda_mesh_normal"] = lambda_mesh_normal;
            opt_json["eval_steps"] = eval_steps;
            opt_json["save_steps"] = save_steps;
            opt_json["enable_eval"] = enable_eval;
            if (!train_images.empty()) {
                opt_json["train_images"] = train_images;
            }
            if (!test_images.empty()) {
                opt_json["test_images"] = test_images;
            }
            opt_json["enable_save_eval_images"] = enable_save_eval_images;
            opt_json["save_depth"] = save_depth;
            opt_json["precompute_mesh_depth_normal"] = precompute_mesh_depth_normal;
            opt_json["mesh_depth_loss_weight"] = mesh_depth_loss_weight;
            opt_json["mesh_normal_loss_weight"] = mesh_normal_loss_weight;
            opt_json["precompute_pseudo_view"] = precompute_pseudo_view;
            opt_json["pseudo_view_sphere_cell_angle_deg"] = pseudo_view_sphere_cell_angle_deg;
            opt_json["pseudo_view_sphere_support_angle_deg"] = pseudo_view_sphere_support_angle_deg;
            opt_json["pseudo_view_mesh_front_angle_deg"] = pseudo_view_mesh_front_angle_deg;
            opt_json["pseudo_view_face_angle_max_deg"] = pseudo_view_face_angle_max_deg;
            opt_json["pseudo_view_top_k_source"] = pseudo_view_top_k_source;
            opt_json["pseudo_view_min_valid_ratio"] = pseudo_view_min_valid_ratio;
            opt_json["pseudo_view_min_valid_pixels"] = pseudo_view_min_valid_pixels;
            opt_json["pseudo_view_rgb_face_angle_max_deg"] = pseudo_view_rgb_face_angle_max_deg;
            opt_json["pseudo_view_source_rgb_face_angle_max_deg"] = pseudo_view_source_rgb_face_angle_max_deg;
            opt_json["pseudo_view_mask_valid_open_pixels"] = pseudo_view_mask_valid_open_pixels;
            opt_json["pseudo_view_mask_erode_pixels"] = pseudo_view_mask_erode_pixels;
            opt_json["pseudo_view_mask_invalid_dilate_pixels"] = pseudo_view_mask_invalid_dilate_pixels;
            opt_json["pseudo_view_rgb_parallel_jobs"] = pseudo_view_rgb_parallel_jobs;
            opt_json["pseudo_view_minimal_metadata"] = pseudo_view_minimal_metadata;
            opt_json["use_pseudo_views_in_training"] = use_pseudo_views_in_training;
            opt_json["pseudo_view_loss_weight"] = pseudo_view_loss_weight;
            opt_json["enable_gggs_loss"] = enable_gggs_loss;
            opt_json["lambda_depth_normal"] = lambda_depth_normal;
            opt_json["regularization_from_iter"] = regularization_from_iter;
            opt_json["gggs_gtnorm"] = gggs_gtnorm;
            opt_json["lambda_gggs_gtnorm"] = lambda_gggs_gtnorm;
            opt_json["gggs_gtnorm_from_iter"] = gggs_gtnorm_from_iter;
            opt_json["enable_mesh_depth_loss"] = enable_mesh_depth_loss;
            opt_json["lambda_mesh_depth"] = lambda_mesh_depth;
            opt_json["mesh_depth_loss_from_iter"] = mesh_depth_loss_from_iter;
            opt_json["headless"] = headless;
            if (!log_level.empty()) {
                opt_json["log_level"] = log_level;
            }
            opt_json["strategy"] = strategy;
            opt_json["mip_filter"] = mip_filter;
            opt_json["use_bilateral_grid"] = use_bilateral_grid;
            opt_json["bilateral_grid_X"] = bilateral_grid_X;
            opt_json["bilateral_grid_Y"] = bilateral_grid_Y;
            opt_json["bilateral_grid_W"] = bilateral_grid_W;
            opt_json["bilateral_grid_lr"] = bilateral_grid_lr;
            opt_json["tv_loss_weight"] = tv_loss_weight;
            opt_json["use_per_frame_affine_color"] = use_per_frame_affine_color;
            opt_json["save_per_frame_affine_color"] = save_per_frame_affine_color;
            opt_json["per_frame_affine_color_lr"] = per_frame_affine_color_lr;
            opt_json["per_frame_affine_color_identity_reg_weight"] = per_frame_affine_color_identity_reg_weight;
            opt_json["per_frame_affine_color_gauge_reg_weight"] = per_frame_affine_color_gauge_reg_weight;
            opt_json["use_per_frame_motion_blur"] = use_per_frame_motion_blur;
            opt_json["use_per_frame_defocus_blur"] = use_per_frame_defocus_blur;
            opt_json["save_per_frame_blur_parameters"] = save_per_frame_blur_parameters;
            opt_json["per_frame_motion_blur_rot_lr"] = per_frame_motion_blur_rot_lr;
            opt_json["per_frame_motion_blur_trans_lr"] = per_frame_motion_blur_trans_lr;
            opt_json["per_frame_motion_blur_rot_reg_weight"] = per_frame_motion_blur_rot_reg_weight;
            opt_json["per_frame_motion_blur_trans_reg_weight"] = per_frame_motion_blur_trans_reg_weight;
            opt_json["per_frame_motion_blur_max_rot_std_deg"] = per_frame_motion_blur_max_rot_std_deg;
            opt_json["per_frame_motion_blur_max_trans_std_scene_ratio"] = per_frame_motion_blur_max_trans_std_scene_ratio;
            opt_json["per_frame_observation_blur_max_radius_px"] = per_frame_observation_blur_max_radius_px;
            opt_json["per_frame_motion_blur_start_iter"] = per_frame_motion_blur_start_iter;
            opt_json["per_frame_defocus_blur_scale_lr"] = per_frame_defocus_blur_scale_lr;
            opt_json["per_frame_defocus_blur_focus_lr"] = per_frame_defocus_blur_focus_lr;
            opt_json["per_frame_defocus_blur_reg_weight"] = per_frame_defocus_blur_reg_weight;
            opt_json["per_frame_defocus_blur_max_radius_px"] = per_frame_defocus_blur_max_radius_px;
            opt_json["per_frame_defocus_blur_start_iter"] = per_frame_defocus_blur_start_iter;
            opt_json["use_ppisp"] = use_ppisp;
            opt_json["ppisp_exposure_only"] = ppisp_exposure_only;
            opt_json["ppisp_lr"] = ppisp_lr;
            opt_json["ppisp_reg_weight"] = ppisp_reg_weight;
            opt_json["ppisp_warmup_steps"] = ppisp_warmup_steps;
            opt_json["ppisp_freeze_from_sidecar"] = ppisp_freeze_from_sidecar;
            opt_json["ppisp_sidecar_path"] = lfs::core::path_to_utf8(ppisp_sidecar_path);
            opt_json["ppisp_use_controller"] = ppisp_use_controller;
            opt_json["ppisp_freeze_gaussians_on_distill"] = ppisp_freeze_gaussians_on_distill;
            opt_json["ppisp_controller_activation_step"] = ppisp_controller_activation_step;
            opt_json["ppisp_controller_lr"] = ppisp_controller_lr;
            opt_json["prune_opacity"] = prune_opacity;
            opt_json["grow_scale3d"] = grow_scale3d;
            opt_json["grow_scale2d"] = grow_scale2d;
            opt_json["prune_scale3d"] = prune_scale3d;
            opt_json["prune_scale2d"] = prune_scale2d;
            opt_json["reset_every"] = reset_every;
            opt_json["pause_refine_after_reset"] = pause_refine_after_reset;
            opt_json["revised_opacity"] = revised_opacity;
            opt_json["gut"] = gut;
            opt_json["undistort"] = undistort;
            opt_json["steps_scaler"] = steps_scaler;
            opt_json["sh_degree_interval"] = sh_degree_interval;
            opt_json["random"] = random;
            opt_json["init_num_pts"] = init_num_pts;
            opt_json["init_extent"] = init_extent;
            opt_json["tile_mode"] = tile_mode;
            opt_json["enable_sparsity"] = enable_sparsity;
            opt_json["sparsify_steps"] = sparsify_steps;
            opt_json["init_rho"] = init_rho;
            opt_json["prune_ratio"] = prune_ratio;
            opt_json["bg_modulation"] = bg_modulation;

            // Pyramid training parameters
            opt_json["pyramid_training"] = pyramid_training;
            opt_json["pyramid_levels"] = pyramid_levels;
            opt_json["pyramid_step_interval"] = pyramid_step_interval;

            static constexpr const char* BG_MODE_NAMES[] = {"solid_color", "modulation", "image", "random"};
            opt_json["bg_mode"] = BG_MODE_NAMES[static_cast<int>(bg_mode)];
            opt_json["bg_color"] = {bg_color[0], bg_color[1], bg_color[2]};
            if (!bg_image_path.empty()) {
                opt_json["bg_image_path"] = path_to_utf8(bg_image_path);
            }

            // Mask parameters
            static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent", "project_mesh"};
            opt_json["mask_mode"] = MASK_MODE_NAMES[static_cast<int>(mask_mode)];
            opt_json["invert_masks"] = invert_masks;
            opt_json["mask_opacity_penalty_weight"] = mask_opacity_penalty_weight;
            opt_json["mask_opacity_penalty_power"] = mask_opacity_penalty_power;
            opt_json["mask_threshold"] = mask_threshold;
            opt_json["project_mesh_mask_erode_pixels"] = project_mesh_mask_erode_pixels;
            opt_json["use_alpha_as_mask"] = use_alpha_as_mask;

            return opt_json;
        }

        std::string OptimizationParameters::validate() const {
            const auto is_finite_nonnegative = [](float value) {
                return std::isfinite(value) && value >= 0.0f;
            };
            const auto pose_mode = to_lower_ascii(pose_refine_mode);
            if (refine_camera_pose) {
                if (pose_mode != "so3xr3") {
                    return "pose_refine_mode currently supports only SO3xR3";
                }
                if (gut) {
                    return "Camera pose refinement currently supports only the FastGS rasterizer; disable gut";
                }
            }
            if (!is_finite_nonnegative(pose_refine_lr_rot) ||
                !is_finite_nonnegative(pose_refine_lr_trans) ||
                !is_finite_nonnegative(pose_refine_lr_final_rot) ||
                !is_finite_nonnegative(pose_refine_lr_final_trans) ||
                !is_finite_nonnegative(pose_refine_lr_pre_warmup) ||
                !is_finite_nonnegative(pose_refine_l2_rot) ||
                !is_finite_nonnegative(pose_refine_l2_trans) ||
                !is_finite_nonnegative(pose_refine_max_rot_deg) ||
                !is_finite_nonnegative(pose_refine_max_trans)) {
                return "pose_refine learning rates, regularization weights, and caps must be finite and non-negative";
            }
            if (mesh_surface_walk_steps < 1) {
                return "mesh_surface_walk_steps must be at least 1";
            }
            if (!log_level.empty() && !is_valid_log_level(log_level)) {
                return "log_level must be one of: trace, debug, info, perf, warn, error, critical, off";
            }
            if (mesh_surface_loss_from_iter < 0) {
                return "mesh_surface_loss_from_iter must be non-negative";
            }
            if (lambda_mesh_project < 0.0f || lambda_mesh_outside_barrier < 0.0f ||
                lambda_mesh_scale_min < 0.0f ||
                lambda_mesh_scale_max < 0.0f || lambda_mesh_normal < 0.0f ||
                !std::isfinite(lambda_mesh_project) ||
                !std::isfinite(lambda_mesh_outside_barrier) ||
                !std::isfinite(lambda_mesh_scale_min) ||
                !std::isfinite(lambda_mesh_scale_max) ||
                !std::isfinite(lambda_mesh_normal)) {
                return "Mesh surface loss weights must be finite and non-negative";
            }
            if (mesh_outside_distance_avg_max_scale_multiplier < 0.0f ||
                !std::isfinite(mesh_outside_distance_avg_max_scale_multiplier)) {
                return "mesh_outside_distance_avg_max_scale_multiplier must be finite and non-negative";
            }
            if (lambda_mesh_outside_barrier > 0.0f &&
                mesh_outside_distance_avg_max_scale_multiplier <= 0.0f) {
                return "lambda_mesh_outside_barrier requires mesh_outside_distance_avg_max_scale_multiplier > 0";
            }
            if (mesh_inside_constraint_distance_avg_max_scale_multiplier < 0.0f ||
                !std::isfinite(mesh_inside_constraint_distance_avg_max_scale_multiplier)) {
                return "mesh_inside_constraint_distance_avg_max_scale_multiplier must be finite and non-negative";
            }
            if (mesh_inside_constraint_fade_ratio < 0.0f ||
                mesh_inside_constraint_fade_ratio >= 1.0f ||
                !std::isfinite(mesh_inside_constraint_fade_ratio)) {
                return "mesh_inside_constraint_fade_ratio must be finite and in [0, 1)";
            }
            if (mesh_scale_rho_ratio < 0.0f || !std::isfinite(mesh_scale_rho_ratio)) {
                return "mesh_scale_rho_ratio must be finite and non-negative";
            }
            if (mesh_scale_rho_avg_max_scale_multiplier < 0.0f ||
                !std::isfinite(mesh_scale_rho_avg_max_scale_multiplier)) {
                return "mesh_scale_rho_avg_max_scale_multiplier must be finite and non-negative";
            }
            if (mesh2splat_max_cap_extra_ratio < 0.0f ||
                !std::isfinite(mesh2splat_max_cap_extra_ratio)) {
                return "mesh2splat_max_cap_extra_ratio must be finite and non-negative";
            }
            if (mesh2splat_hole_fill_max_boundary_edges < 3) {
                return "mesh2splat_hole_fill_max_boundary_edges must be at least 3";
            }
            if (!std::isfinite(mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio) ||
                mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio <= 0.0f ||
                !std::isfinite(mesh2splat_hole_fill_max_boundary_area_bbox_ratio) ||
                mesh2splat_hole_fill_max_boundary_area_bbox_ratio <= 0.0f ||
                !std::isfinite(mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio) ||
                mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio <= 0.0f ||
                !std::isfinite(mesh2splat_hole_fill_weld_epsilon_bbox_ratio) ||
                mesh2splat_hole_fill_weld_epsilon_bbox_ratio <= 0.0f) {
                return "mesh2splat hole-fill boundary ratios and weld epsilon must be finite and positive";
            }
            if (!std::isfinite(mesh2splat_hole_fill_density_control_factor) ||
                mesh2splat_hole_fill_density_control_factor <= 0.0f) {
                return "mesh2splat_hole_fill_density_control_factor must be finite and positive";
            }
            if (!std::isfinite(mesh2splat_hole_fill_min_density_ratio) ||
                mesh2splat_hole_fill_min_density_ratio <= 0.0f ||
                !std::isfinite(mesh2splat_hole_fill_max_density_ratio) ||
                mesh2splat_hole_fill_max_density_ratio <= 0.0f) {
                return "mesh2splat hole-fill density ratios must be finite and positive";
            }
            if (mesh2splat_hole_fill_min_density_ratio > mesh2splat_hole_fill_max_density_ratio) {
                return "mesh2splat_hole_fill_min_density_ratio must not exceed mesh2splat_hole_fill_max_density_ratio";
            }
            if (!std::isfinite(mesh2splat_external_band_ratio) ||
                mesh2splat_external_band_ratio < 0.0f ||
                mesh2splat_external_band_ratio > 0.5f) {
                return "mesh2splat_external_band_ratio must be finite and in [0.0, 0.5]";
            }
            if (!std::isfinite(mesh2splat_external_band_tolerance_ratio) ||
                mesh2splat_external_band_tolerance_ratio < 0.0f ||
                mesh2splat_external_band_tolerance_ratio > 0.5f) {
                return "mesh2splat_external_band_tolerance_ratio must be finite and in [0.0, 0.5]";
            }
            if (mesh2splat_external_band_tolerance_ratio > mesh2splat_external_band_ratio) {
                return "mesh2splat_external_band_tolerance_ratio must not exceed mesh2splat_external_band_ratio";
            }
            if (!std::isfinite(mesh2splat_pointcloud_density_ratio) ||
                mesh2splat_pointcloud_density_ratio < 0.0f ||
                mesh2splat_pointcloud_density_ratio > 4.0f) {
                return "mesh2splat_pointcloud_density_ratio must be finite and in [0.0, 4.0]";
            }
            if (!std::isfinite(mesh2splat_pointcloud_surface_reject_ratio) ||
                mesh2splat_pointcloud_surface_reject_ratio < 0.0f ||
                mesh2splat_pointcloud_surface_reject_ratio > 4.0f) {
                return "mesh2splat_pointcloud_surface_reject_ratio must be finite and in [0.0, 4.0]";
            }
            if (mesh2splat_pointcloud_mask_min_points_per_hole < 1) {
                return "mesh2splat_pointcloud_mask_min_points_per_hole must be at least 1";
            }
            if (mesh2splat_pointcloud_mask_rim_close_pixels < 0) {
                return "mesh2splat_pointcloud_mask_rim_close_pixels must be non-negative";
            }
            if (!std::isfinite(mesh2splat_pointcloud_mask_splat_radius_scale) ||
                mesh2splat_pointcloud_mask_splat_radius_scale < 0.0f ||
                mesh2splat_pointcloud_mask_splat_radius_scale > 8.0f) {
                return "mesh2splat_pointcloud_mask_splat_radius_scale must be finite and in [0.0, 8.0]";
            }
            if (mesh2splat_pointcloud_mask_close_pixels < 0) {
                return "mesh2splat_pointcloud_mask_close_pixels must be non-negative";
            }
            if (lambda_mesh_scale_max > 0.0f &&
                mesh_scale_rho_ratio <= 0.0f &&
                mesh_scale_rho_avg_max_scale_multiplier <= 0.0f) {
                return "lambda_mesh_scale_max requires mesh_scale_rho_ratio > 0 or mesh_scale_rho_avg_max_scale_multiplier > 0";
            }

            const bool has_explicit_train_images = !train_images.empty();
            const bool has_explicit_test_images = !test_images.empty();
            if (has_explicit_train_images != has_explicit_test_images) {
                return "Explicit eval split requires both train_images and test_images";
            }

            if (has_explicit_train_images) {
                std::unordered_set<std::string> train_set(train_images.begin(), train_images.end());
                for (const auto& image_name : test_images) {
                    if (train_set.find(image_name) != train_set.end()) {
                        return std::format("Image '{}' appears in both train_images and test_images", image_name);
                    }
                }
            }

            if (gut && (strategy == "adc" || strategy == "igs+"))
                return "GUT and " + strategy + " strategy cannot be used together";
            if (use_ppisp && use_pseudo_views_in_training) {
                return "PPISP cannot be combined with pseudo-view training supervision; disable use_ppisp or "
                       "use_pseudo_views_in_training";
            }
            if (ppisp_exposure_only && ppisp_freeze_from_sidecar) {
                return "PPISP exposure-only mode cannot be combined with frozen PPISP sidecar";
            }
            if (ppisp_exposure_only && !use_ppisp) {
                return "PPISP exposure-only mode requires PPISP enabled";
            }
            if (ppisp_exposure_only && ppisp_use_controller) {
                return "PPISP exposure-only mode cannot be combined with the PPISP controller";
            }
            if (save_per_frame_affine_color && !use_per_frame_affine_color) {
                return "Saving per-frame affine color parameters requires per-frame affine color enabled";
            }
            if (use_per_frame_affine_color) {
                if (use_ppisp || use_bilateral_grid) {
                    return "Per-frame affine color cannot be combined with PPISP or bilateral grid";
                }
                if (!is_finite_nonnegative(per_frame_affine_color_lr) ||
                    !is_finite_nonnegative(per_frame_affine_color_identity_reg_weight) ||
                    !is_finite_nonnegative(per_frame_affine_color_gauge_reg_weight)) {
                    return "Per-frame affine color learning rate and regularization weights must be finite and non-negative";
                }
            }
            const bool use_observation_blur =
                use_per_frame_motion_blur || use_per_frame_defocus_blur;
            if (save_per_frame_blur_parameters && !use_observation_blur) {
                return "Saving per-frame blur parameters requires motion blur or defocus blur enabled";
            }
            if (use_observation_blur) {
                if (gut) {
                    return "Per-frame motion/defocus blur currently supports only FastGS; disable gut";
                }
                if (mip_filter) {
                    return "Per-frame motion/defocus blur currently requires mip_filter=false until the mip covariance VJP is available";
                }
                if (use_per_frame_defocus_blur && pyramid_training) {
                    return "Per-frame defocus blur currently requires pyramid_training=false until beta is scaled with render resolution";
                }
                if (enable_gggs_loss || gggs_gtnorm || enable_mesh_depth_loss ||
                    mesh_depth_visibility_cull) {
                    return "Per-frame motion/defocus blur currently supports RGB supervision only; disable GGGS, mesh depth loss, and mesh depth visibility cull";
                }
                if (!is_finite_nonnegative(per_frame_motion_blur_rot_lr) ||
                    !is_finite_nonnegative(per_frame_motion_blur_trans_lr) ||
                    !is_finite_nonnegative(per_frame_motion_blur_rot_reg_weight) ||
                    !is_finite_nonnegative(per_frame_motion_blur_trans_reg_weight) ||
                    !is_finite_nonnegative(per_frame_motion_blur_max_rot_std_deg) ||
                    !is_finite_nonnegative(per_frame_motion_blur_max_trans_std_scene_ratio) ||
                    !is_finite_nonnegative(per_frame_observation_blur_max_radius_px) ||
                    !is_finite_nonnegative(per_frame_defocus_blur_scale_lr) ||
                    !is_finite_nonnegative(per_frame_defocus_blur_focus_lr) ||
                    !is_finite_nonnegative(per_frame_defocus_blur_reg_weight) ||
                    !is_finite_nonnegative(per_frame_defocus_blur_max_radius_px)) {
                    return "Per-frame blur learning rates, regularization weights, and caps must be finite and non-negative";
                }
                if (per_frame_observation_blur_max_radius_px <= 0.0f) {
                    return "Per-frame observation blur requires a positive screen-space maximum radius";
                }
                if (use_per_frame_motion_blur &&
                    (per_frame_motion_blur_max_rot_std_deg <= 0.0f ||
                     per_frame_motion_blur_max_trans_std_scene_ratio <= 0.0f)) {
                    return "Per-frame motion blur requires positive rotation and translation standard-deviation caps";
                }
                if (use_per_frame_defocus_blur &&
                    per_frame_defocus_blur_max_radius_px <= 0.0f) {
                    return "Per-frame defocus blur requires a positive maximum radius";
                }
                if (per_frame_motion_blur_start_iter >
                        static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    per_frame_defocus_blur_start_iter >
                        static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    iterations > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    return "Per-frame blur iteration counts must fit in a signed int";
                }
            }
            if (ppisp_freeze_from_sidecar && !use_ppisp)
                return "PPISP sidecar freeze requires PPISP enabled";
            if (mesh_depth_loss_weight < 0.0f || mesh_normal_loss_weight < 0.0f)
                return "Mesh depth/normal loss weights must be non-negative";
            if (!(pseudo_view_sphere_cell_angle_deg > 0.0f) ||
                pseudo_view_sphere_cell_angle_deg > 90.0f ||
                !std::isfinite(pseudo_view_sphere_cell_angle_deg))
                return "pseudo_view_sphere_cell_angle_deg must be in (0, 90]";
            if (pseudo_view_sphere_support_angle_deg < 0.0f ||
                pseudo_view_sphere_support_angle_deg > 90.0f ||
                !std::isfinite(pseudo_view_sphere_support_angle_deg))
                return "pseudo_view_sphere_support_angle_deg must be in [0, 90]";
            if (pseudo_view_mesh_front_angle_deg < 0.0f ||
                pseudo_view_mesh_front_angle_deg > 90.0f ||
                !std::isfinite(pseudo_view_mesh_front_angle_deg))
                return "pseudo_view_mesh_front_angle_deg must be in [0, 90]";
            if (pseudo_view_face_angle_max_deg < 0.0f ||
                pseudo_view_face_angle_max_deg > 90.0f ||
                !std::isfinite(pseudo_view_face_angle_max_deg))
                return "pseudo_view_face_angle_max_deg must be in [0, 90]";
            if (pseudo_view_top_k_source < 1)
                return "pseudo_view_top_k_source must be at least 1";
            if (pseudo_view_min_valid_ratio < 0.0f ||
                pseudo_view_min_valid_ratio > 1.0f ||
                !std::isfinite(pseudo_view_min_valid_ratio))
                return "pseudo_view_min_valid_ratio must be in [0, 1]";
            if (pseudo_view_min_valid_pixels < 0)
                return "pseudo_view_min_valid_pixels must be non-negative";
            if (pseudo_view_rgb_face_angle_max_deg < 0.0f ||
                pseudo_view_rgb_face_angle_max_deg > 180.0f ||
                !std::isfinite(pseudo_view_rgb_face_angle_max_deg))
                return "pseudo_view_rgb_face_angle_max_deg must be in [0, 180]";
            if (pseudo_view_source_rgb_face_angle_max_deg < 0.0f ||
                pseudo_view_source_rgb_face_angle_max_deg > 180.0f ||
                !std::isfinite(pseudo_view_source_rgb_face_angle_max_deg))
                return "pseudo_view_source_rgb_face_angle_max_deg must be in [0, 180]";
            if (pseudo_view_mask_valid_open_pixels < 0)
                return "pseudo_view_mask_valid_open_pixels must be non-negative";
            if (pseudo_view_mask_erode_pixels < 0)
                return "pseudo_view_mask_erode_pixels must be non-negative";
            if (pseudo_view_mask_invalid_dilate_pixels < 0)
                return "pseudo_view_mask_invalid_dilate_pixels must be non-negative";
            if (project_mesh_mask_erode_pixels < 0)
                return "project_mesh_mask_erode_pixels must be non-negative";
            if (pseudo_view_rgb_parallel_jobs < 1 || pseudo_view_rgb_parallel_jobs > 4)
                return "pseudo_view_rgb_parallel_jobs must be in [1, 4]";
            if (pseudo_view_loss_weight < 0.0f || !std::isfinite(pseudo_view_loss_weight))
                return "pseudo_view_loss_weight must be finite and non-negative";
            return {};
        }

        std::string TrainingParameters::validate() const {
            if (auto error = optimization.validate(); !error.empty()) {
                return error;
            }

            if (!use_mesh_init && optimization.mesh_init_color_source != MeshInitColorSource::Auto) {
                return std::format(
                    "mesh_init_color_source={} requires --mesh-init-gs-scene",
                    mesh_init_color_source_name(optimization.mesh_init_color_source));
            }

            const auto hole_fill_mode = optimization.resolved_mesh2splat_hole_fill_mode();
            if ((optimization.mesh2splat_hole_fill_mode == Mesh2SplatHoleFillMode::External ||
                 optimization.mesh2splat_hole_fill_mode == Mesh2SplatHoleFillMode::ExternalPointCloud) &&
                optimization.mesh2splat_hole_fill_enabled) {
                return std::format(
                    "mesh2splat_hole_fill_mode={} conflicts with deprecated mesh2splat_hole_fill_enabled=true",
                    mesh2splat_hole_fill_mode_name(optimization.mesh2splat_hole_fill_mode));
            }
            if (hole_fill_mode != Mesh2SplatHoleFillMode::None && !use_mesh_init) {
                return "mesh2splat hole filling requires --mesh-init-gs-scene (use_mesh_init=true)";
            }
            if (hole_fill_mode != Mesh2SplatHoleFillMode::None &&
                optimization.strategy == "mesh2splat") {
                return "mesh2splat hole filling is not supported by the standalone mesh2splat strategy";
            }

            const bool has_external_mesh = mesh_init_external_filled_mesh_path.has_value() &&
                                           !mesh_init_external_filled_mesh_path->empty();
            if (hole_fill_mode != Mesh2SplatHoleFillMode::External && has_external_mesh) {
                return "External mesh-init paths require mesh2splat_hole_fill_mode=external";
            }
            if (hole_fill_mode == Mesh2SplatHoleFillMode::External) {
                if (!has_external_mesh) {
                    return "External mesh2splat hole filling requires --mesh-init-external-filled-mesh";
                }
                const auto external_path = lfs::core::utf8_to_path(*mesh_init_external_filled_mesh_path);
                if (!std::filesystem::exists(external_path) ||
                    !std::filesystem::is_regular_file(external_path)) {
                    return std::format("External filled mesh does not exist or is not a regular file: '{}'",
                                       lfs::core::path_to_utf8(external_path));
                }
                if (!has_extension(external_path, ".ply") && !has_extension(external_path, ".obj")) {
                    return "External filled mesh must be a .ply or .obj file";
                }
                if (mesh_init_mesh_path.has_value() && !mesh_init_mesh_path->empty()) {
                    std::error_code ec_original;
                    std::error_code ec_external;
                    const auto original_canonical = std::filesystem::weakly_canonical(
                        lfs::core::utf8_to_path(*mesh_init_mesh_path), ec_original);
                    const auto external_canonical = std::filesystem::weakly_canonical(
                        external_path, ec_external);
                    if (!ec_original && !ec_external && original_canonical == external_canonical) {
                        return "External filled mesh must not be the same file as the original mesh";
                    }
                }
            }

            const bool has_external_pointcloud = mesh_init_external_pointcloud_path.has_value() &&
                                                 !mesh_init_external_pointcloud_path->empty();
            if (hole_fill_mode != Mesh2SplatHoleFillMode::ExternalPointCloud && has_external_pointcloud) {
                return "External mesh-init point cloud requires mesh2splat_hole_fill_mode=external_pointcloud";
            }
            if (hole_fill_mode == Mesh2SplatHoleFillMode::ExternalPointCloud) {
                if (has_external_mesh) {
                    return "mesh2splat_hole_fill_mode=external_pointcloud conflicts with "
                           "--mesh-init-external-filled-mesh; the two hole-fill strategies do not stack";
                }
                if (!has_external_pointcloud) {
                    return "External point-cloud mesh2splat hole filling requires --mesh-init-external-pointcloud";
                }
                const auto pointcloud_path = lfs::core::utf8_to_path(*mesh_init_external_pointcloud_path);
                if (!std::filesystem::exists(pointcloud_path) ||
                    !std::filesystem::is_regular_file(pointcloud_path)) {
                    return std::format("External point cloud does not exist or is not a regular file: '{}'",
                                       lfs::core::path_to_utf8(pointcloud_path));
                }
                if (!has_extension(pointcloud_path, ".ply")) {
                    return "External point cloud must be a .ply file";
                }
                if (mesh_init_mesh_path.has_value() && !mesh_init_mesh_path->empty()) {
                    std::error_code ec_original;
                    std::error_code ec_cloud;
                    const auto original_canonical = std::filesystem::weakly_canonical(
                        lfs::core::utf8_to_path(*mesh_init_mesh_path), ec_original);
                    const auto cloud_canonical = std::filesystem::weakly_canonical(
                        pointcloud_path, ec_cloud);
                    if (!ec_original && !ec_cloud && original_canonical == cloud_canonical) {
                        return "External point cloud must not be the same file as the original mesh";
                    }
                }
            }

            if ((mesh2splat_mesh_path.has_value() && !mesh2splat_mesh_path->empty()) ||
                (mesh2splat_texture_path.has_value() && !mesh2splat_texture_path->empty())) {
                if (optimization.strategy != "mesh2splat") {
                    return "mesh2splat inputs require strategy=mesh2splat";
                }
            }

            if (optimization.strategy == "mesh2splat") {
                if (!mesh2splat_mesh_path.has_value() || mesh2splat_mesh_path->empty()) {
                    return "mesh2splat strategy requires --mesh2splat-dir with obj+mtl+png";
                }

                if (!mesh2splat_texture_path.has_value() || mesh2splat_texture_path->empty()) {
                    return "mesh2splat strategy requires --mesh2splat-dir with obj+mtl+png";
                }

                const auto mesh_path = lfs::core::utf8_to_path(*mesh2splat_mesh_path);
                if (!std::filesystem::exists(mesh_path)) {
                    return std::format("mesh2splat mesh does not exist: '{}'",
                                       lfs::core::path_to_utf8(mesh_path));
                }

                if (!has_extension(mesh_path, ".obj")) {
                    return "mesh2splat requires an .obj mesh (resolved from --mesh2splat-dir)";
                }

                const auto tex_path = lfs::core::utf8_to_path(*mesh2splat_texture_path);
                if (!std::filesystem::exists(tex_path)) {
                    return std::format("mesh2splat texture does not exist: '{}'",
                                       lfs::core::path_to_utf8(tex_path));
                }

                if (!has_extension(tex_path, ".png")) {
                    return "mesh2splat requires a .png texture (resolved from --mesh2splat-dir)";
                }

                const auto mesh_dir = mesh_path.parent_path().lexically_normal();
                const auto tex_dir = tex_path.parent_path().lexically_normal();
                if (mesh_dir != tex_dir) {
                    return "mesh2splat requires obj+mtl+png to be in the same directory";
                }

                if (count_files_with_extension(mesh_dir, ".mtl") == 0) {
                    return "mesh2splat requires at least one .mtl file in the obj/png directory";
                }
            }

            if (mesh_init_sampling_rate <= 0.0f || mesh_init_sampling_rate > 1.0f) {
                return "Mesh init raster density ratio must be in (0, 1]";
            }

            if (use_mesh_init) {
                if (!mesh_init_mesh_path.has_value() || mesh_init_mesh_path->empty()) {
                    return "Mesh init requires --mesh-init-gs-scene with a .ply/.obj mesh file or legacy obj+mtl+png/ply+png directory";
                }

                const auto mesh_path = lfs::core::utf8_to_path(*mesh_init_mesh_path);
                if (!std::filesystem::exists(mesh_path)) {
                    return std::format("Mesh init mesh does not exist: '{}'",
                                       lfs::core::path_to_utf8(mesh_path));
                }

                if (!std::filesystem::is_regular_file(mesh_path)) {
                    return std::format("Mesh init mesh is not a regular file: '{}'",
                                       lfs::core::path_to_utf8(mesh_path));
                }

                const bool is_ply_mesh = has_extension(mesh_path, ".ply");
                const bool is_obj_mesh = has_extension(mesh_path, ".obj");
                if (!is_ply_mesh && !is_obj_mesh) {
                    return "Mesh init requires a .ply or .obj mesh (resolved from --mesh-init-gs-scene)";
                }

                if (mesh_init_texture_path.has_value() && !mesh_init_texture_path->empty()) {
                    const auto tex_path = lfs::core::utf8_to_path(*mesh_init_texture_path);
                    if (!std::filesystem::exists(tex_path)) {
                        return std::format("Mesh init texture does not exist: '{}'",
                                           lfs::core::path_to_utf8(tex_path));
                    }

                    if (!has_extension(tex_path, ".png")) {
                        return "Mesh init requires a .png texture when resolved from a legacy --mesh-init-gs-scene directory";
                    }

                    const auto mesh_dir = mesh_path.parent_path().lexically_normal();
                    const auto tex_dir = tex_path.parent_path().lexically_normal();
                    if (mesh_dir != tex_dir) {
                        return "Mesh init requires mesh and png texture to be in the same directory";
                    }

                    if (is_obj_mesh && count_files_with_extension(mesh_dir, ".mtl") == 0) {
                        return "Mesh init requires obj+mtl+png when a legacy directory resolves to .obj";
                    }
                }
            }

            if (optimization.ppisp_freeze_from_sidecar && !resume_checkpoint.has_value()) {
                if (optimization.ppisp_sidecar_path.empty()) {
                    return "PPISP sidecar freeze requires a sidecar path";
                }
                if (!std::filesystem::exists(optimization.ppisp_sidecar_path)) {
                    return std::format("PPISP sidecar does not exist: '{}'",
                                       lfs::core::path_to_utf8(optimization.ppisp_sidecar_path));
                }
            }
            return {};
        }

        OptimizationParameters OptimizationParameters::mcmc_defaults() {
            return {};
        }

        OptimizationParameters OptimizationParameters::adc_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = "adc";
            p.opacity_lr = 0.025f;
            p.stop_refine = 15'000;
            p.opacity_reg = 0.0f;
            p.scale_reg = 0.0f;
            p.init_opacity = 0.1f;
            p.max_cap = 6'000'000;
            p.tv_loss_weight = 5.0f;
            return p;
        }

        OptimizationParameters OptimizationParameters::igs_plus_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = "igs+";
            p.means_lr = 0.000016f;
            p.shs_lr = 0.005f;
            p.scaling_lr = 0.02f;
            p.rotation_lr = 0.0015f;
            p.stop_refine = 15'000;
            p.refine_every = 500;
            p.opacity_reg = 0.0f;
            p.scale_reg = 0.0f;
            p.init_opacity = 0.1f;
            p.init_scaling = 0.1f;
            p.revised_opacity = true;
            p.max_cap = 4'000'000;
            p.tv_loss_weight = 5.0f;
            return p;
        }

        OptimizationParameters OptimizationParameters::from_json(const nlohmann::json& json) {

            OptimizationParameters params;
            if (json.contains("strategy")) {
                std::string strategy = json["strategy"];
                if (strategy == "mcmc" || strategy == "adc" || strategy == "igs+" || strategy == "mesh2splat") {
                    params.strategy = strategy;
                } else {
                    LOG_WARN("Invalid strategy '{}' in JSON, using default", strategy);
                }
            }

            params.iterations = json["iterations"];
            params.means_lr = json["means_lr"];
            params.shs_lr = json["shs_lr"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            if (params.strategy != "igs+" || json.contains("min_opacity")) {
                params.min_opacity = json.at("min_opacity");
            }
            params.refine_every = json["refine_every"];
            params.start_refine = json["start_refine"];
            params.stop_refine = json["stop_refine"];
            if (params.strategy != "igs+" || json.contains("grad_threshold")) {
                params.grad_threshold = json.at("grad_threshold");
            }
            params.sh_degree = json["sh_degree"];

            if (json.contains("opacity_reg")) {
                params.opacity_reg = json["opacity_reg"];
            }
            if (json.contains("scale_reg")) {
                params.scale_reg = json["scale_reg"];
            }
            if (json.contains("init_opacity")) {
                params.init_opacity = json["init_opacity"];
            }
            if (json.contains("init_scaling")) {
                params.init_scaling = json["init_scaling"];
            }
            if (json.contains("max_cap")) {
                params.max_cap = json["max_cap"];
            }
            if (json.contains("refine_camera_pose")) {
                params.refine_camera_pose = json["refine_camera_pose"];
            }
            if (json.contains("save_pose_refine_outputs")) {
                params.save_pose_refine_outputs = json["save_pose_refine_outputs"];
            }
            if (json.contains("pose_refine_mode")) {
                params.pose_refine_mode = json["pose_refine_mode"].get<std::string>();
            }
            if (json.contains("pose_refine_stop_iter")) {
                params.pose_refine_stop_iter = json["pose_refine_stop_iter"];
            }
            if (json.contains("pose_refine_lr_rot")) {
                params.pose_refine_lr_rot = json["pose_refine_lr_rot"];
            }
            if (json.contains("pose_refine_lr_trans")) {
                params.pose_refine_lr_trans = json["pose_refine_lr_trans"];
            }
            if (json.contains("pose_refine_lr_final_rot")) {
                params.pose_refine_lr_final_rot = json["pose_refine_lr_final_rot"];
            }
            if (json.contains("pose_refine_lr_final_trans")) {
                params.pose_refine_lr_final_trans = json["pose_refine_lr_final_trans"];
            }
            if (json.contains("pose_refine_lr_pre_warmup")) {
                params.pose_refine_lr_pre_warmup = json["pose_refine_lr_pre_warmup"];
            }
            if (json.contains("pose_refine_lr_max_steps")) {
                params.pose_refine_lr_max_steps = json["pose_refine_lr_max_steps"];
            }
            if (json.contains("pose_refine_lr_warmup_steps")) {
                params.pose_refine_lr_warmup_steps = json["pose_refine_lr_warmup_steps"];
            }
            if (json.contains("pose_refine_l2_rot")) {
                params.pose_refine_l2_rot = json["pose_refine_l2_rot"];
            }
            if (json.contains("pose_refine_l2_trans")) {
                params.pose_refine_l2_trans = json["pose_refine_l2_trans"];
            }
            if (json.contains("pose_refine_log_every")) {
                params.pose_refine_log_every = json["pose_refine_log_every"];
            }
            if (json.contains("pose_refine_max_rot_deg")) {
                params.pose_refine_max_rot_deg = json["pose_refine_max_rot_deg"];
            }
            if (json.contains("pose_refine_max_trans")) {
                params.pose_refine_max_trans = json["pose_refine_max_trans"];
            }
            if (json.contains("mesh2splat_max_cap_extra_ratio")) {
                params.mesh2splat_max_cap_extra_ratio = json["mesh2splat_max_cap_extra_ratio"];
            }
            if (json.contains("mesh2splat_opacity_no_grad")) {
                params.mesh2splat_opacity_no_grad = json["mesh2splat_opacity_no_grad"];
            }
            if (json.contains("mesh_init_color_source")) {
                const auto parsed_source = parse_mesh_init_color_source(
                    json["mesh_init_color_source"].get<std::string>());
                if (!parsed_source) {
                    throw std::invalid_argument(parsed_source.error());
                }
                params.mesh_init_color_source = *parsed_source;
            }
            const bool has_hole_fill_mode = json.contains("mesh2splat_hole_fill_mode");
            const bool legacy_hole_fill_enabled =
                json.contains("mesh2splat_hole_fill_enabled") &&
                json["mesh2splat_hole_fill_enabled"].get<bool>();
            if (has_hole_fill_mode) {
                const auto parsed_mode = parse_mesh2splat_hole_fill_mode(
                    json["mesh2splat_hole_fill_mode"].get<std::string>());
                if (!parsed_mode) {
                    throw std::invalid_argument(parsed_mode.error());
                }
                if (legacy_hole_fill_enabled &&
                    *parsed_mode != Mesh2SplatHoleFillMode::Cgal) {
                    throw std::invalid_argument(
                        "mesh2splat_hole_fill_mode conflicts with deprecated mesh2splat_hole_fill_enabled=true");
                }
                params.mesh2splat_hole_fill_mode = *parsed_mode;
                params.mesh2splat_hole_fill_enabled =
                    *parsed_mode == Mesh2SplatHoleFillMode::Cgal;
            } else {
                params.mesh2splat_hole_fill_enabled = legacy_hole_fill_enabled;
                params.mesh2splat_hole_fill_mode = legacy_hole_fill_enabled
                                                       ? Mesh2SplatHoleFillMode::Cgal
                                                       : Mesh2SplatHoleFillMode::None;
            }
            if (json.contains("mesh2splat_hole_fill_max_boundary_edges")) {
                params.mesh2splat_hole_fill_max_boundary_edges = json["mesh2splat_hole_fill_max_boundary_edges"];
            }
            if (json.contains("mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio")) {
                params.mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio = json["mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio"];
            }
            if (json.contains("mesh2splat_hole_fill_max_boundary_area_bbox_ratio")) {
                params.mesh2splat_hole_fill_max_boundary_area_bbox_ratio = json["mesh2splat_hole_fill_max_boundary_area_bbox_ratio"];
            }
            if (json.contains("mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio")) {
                params.mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio = json["mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio"];
            }
            if (json.contains("mesh2splat_hole_fill_weld_epsilon_bbox_ratio")) {
                params.mesh2splat_hole_fill_weld_epsilon_bbox_ratio = json["mesh2splat_hole_fill_weld_epsilon_bbox_ratio"];
            }
            if (json.contains("mesh2splat_hole_fill_density_control_factor")) {
                params.mesh2splat_hole_fill_density_control_factor = json["mesh2splat_hole_fill_density_control_factor"];
            }
            if (json.contains("mesh2splat_hole_fill_min_density_ratio")) {
                params.mesh2splat_hole_fill_min_density_ratio = json["mesh2splat_hole_fill_min_density_ratio"];
            }
            if (json.contains("mesh2splat_hole_fill_max_density_ratio")) {
                params.mesh2splat_hole_fill_max_density_ratio = json["mesh2splat_hole_fill_max_density_ratio"];
            }
            if (json.contains("mesh2splat_external_band_ratio")) {
                params.mesh2splat_external_band_ratio = json["mesh2splat_external_band_ratio"];
            }
            if (json.contains("mesh2splat_external_band_tolerance_ratio")) {
                params.mesh2splat_external_band_tolerance_ratio = json["mesh2splat_external_band_tolerance_ratio"];
            }
            if (json.contains("mesh2splat_pointcloud_density_ratio")) {
                params.mesh2splat_pointcloud_density_ratio = json["mesh2splat_pointcloud_density_ratio"];
            }
            if (json.contains("mesh2splat_pointcloud_surface_reject_ratio")) {
                params.mesh2splat_pointcloud_surface_reject_ratio = json["mesh2splat_pointcloud_surface_reject_ratio"];
            }
            if (json.contains("mesh2splat_pointcloud_mask_min_points_per_hole")) {
                params.mesh2splat_pointcloud_mask_min_points_per_hole = json["mesh2splat_pointcloud_mask_min_points_per_hole"];
            }
            if (json.contains("mesh2splat_pointcloud_mask_rim_close_pixels")) {
                params.mesh2splat_pointcloud_mask_rim_close_pixels = json["mesh2splat_pointcloud_mask_rim_close_pixels"];
            }
            if (json.contains("mesh2splat_pointcloud_mask_splat_radius_scale")) {
                params.mesh2splat_pointcloud_mask_splat_radius_scale = json["mesh2splat_pointcloud_mask_splat_radius_scale"];
            }
            if (json.contains("mesh2splat_pointcloud_mask_close_pixels")) {
                params.mesh2splat_pointcloud_mask_close_pixels = json["mesh2splat_pointcloud_mask_close_pixels"];
            }
            if (json.contains("mesh_surface_walk_steps")) {
                params.mesh_surface_walk_steps = json["mesh_surface_walk_steps"];
            }
            if (json.contains("mesh_surface_loss_from_iter")) {
                params.mesh_surface_loss_from_iter = json["mesh_surface_loss_from_iter"];
            }
            if (json.contains("mesh_surface_hard_projection_enabled")) {
                params.mesh_surface_hard_projection_enabled = json["mesh_surface_hard_projection_enabled"];
            }
            if (json.contains("mesh_depth_visibility_cull")) {
                params.mesh_depth_visibility_cull = json["mesh_depth_visibility_cull"];
            }
            if (json.contains("lambda_mesh_project")) {
                params.lambda_mesh_project = json["lambda_mesh_project"];
            }
            if (json.contains("lambda_mesh_outside_barrier")) {
                params.lambda_mesh_outside_barrier = json["lambda_mesh_outside_barrier"];
            }
            if (json.contains("mesh_outside_distance_avg_max_scale_multiplier")) {
                params.mesh_outside_distance_avg_max_scale_multiplier = json["mesh_outside_distance_avg_max_scale_multiplier"];
            }
            if (json.contains("mesh_inside_constraint_distance_avg_max_scale_multiplier")) {
                params.mesh_inside_constraint_distance_avg_max_scale_multiplier = json["mesh_inside_constraint_distance_avg_max_scale_multiplier"];
            }
            if (json.contains("mesh_inside_constraint_fade_ratio")) {
                params.mesh_inside_constraint_fade_ratio = json["mesh_inside_constraint_fade_ratio"];
            }
            if (json.contains("lambda_mesh_scale_min")) {
                params.lambda_mesh_scale_min = json["lambda_mesh_scale_min"];
            }
            if (json.contains("lambda_mesh_scale_max")) {
                params.lambda_mesh_scale_max = json["lambda_mesh_scale_max"];
            }
            if (json.contains("mesh_scale_rho_ratio")) {
                params.mesh_scale_rho_ratio = json["mesh_scale_rho_ratio"];
            }
            if (json.contains("mesh_scale_rho_avg_max_scale_multiplier")) {
                params.mesh_scale_rho_avg_max_scale_multiplier = json["mesh_scale_rho_avg_max_scale_multiplier"];
            }
            if (json.contains("lambda_mesh_normal")) {
                params.lambda_mesh_normal = json["lambda_mesh_normal"];
            }

            if (json.contains("eval_steps")) {
                params.eval_steps.clear();
                for (const auto& step : json["eval_steps"]) {
                    params.eval_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("save_steps")) {
                params.save_steps.clear();
                for (const auto& step : json["save_steps"]) {
                    params.save_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("enable_eval")) {
                params.enable_eval = json["enable_eval"];
            }
            params.train_images = parse_explicit_split_images(json, "train_images");
            params.test_images = parse_explicit_split_images(json, "test_images");
            if (json.contains("enable_save_eval_images")) {
                params.enable_save_eval_images = json["enable_save_eval_images"];
            }
            if (json.contains("save_depth")) {
                params.save_depth = json["save_depth"];
            }
            if (json.contains("precompute_mesh_depth_normal")) {
                params.precompute_mesh_depth_normal = json["precompute_mesh_depth_normal"];
            }
            if (json.contains("mesh_depth_loss_weight")) {
                params.mesh_depth_loss_weight = json["mesh_depth_loss_weight"];
            }
            if (json.contains("mesh_normal_loss_weight")) {
                params.mesh_normal_loss_weight = json["mesh_normal_loss_weight"];
            }
            if (json.contains("precompute_pseudo_view")) {
                params.precompute_pseudo_view = json["precompute_pseudo_view"];
            }
            if (json.contains("pseudo_view_sphere_cell_angle_deg")) {
                params.pseudo_view_sphere_cell_angle_deg = json["pseudo_view_sphere_cell_angle_deg"];
            }
            if (json.contains("pseudo_view_sphere_support_angle_deg")) {
                params.pseudo_view_sphere_support_angle_deg = json["pseudo_view_sphere_support_angle_deg"];
            }
            if (json.contains("pseudo_view_mesh_front_angle_deg")) {
                params.pseudo_view_mesh_front_angle_deg = json["pseudo_view_mesh_front_angle_deg"];
            }
            if (json.contains("pseudo_view_face_angle_max_deg")) {
                params.pseudo_view_face_angle_max_deg = json["pseudo_view_face_angle_max_deg"];
            }
            if (json.contains("pseudo_view_top_k_source")) {
                params.pseudo_view_top_k_source = json["pseudo_view_top_k_source"];
            }
            if (json.contains("pseudo_view_min_valid_ratio")) {
                params.pseudo_view_min_valid_ratio = json["pseudo_view_min_valid_ratio"];
            }
            if (json.contains("pseudo_view_min_valid_pixels")) {
                params.pseudo_view_min_valid_pixels = json["pseudo_view_min_valid_pixels"];
            }
            if (json.contains("pseudo_view_rgb_face_angle_max_deg")) {
                params.pseudo_view_rgb_face_angle_max_deg = json["pseudo_view_rgb_face_angle_max_deg"];
            }
            if (json.contains("pseudo_view_source_rgb_face_angle_max_deg")) {
                params.pseudo_view_source_rgb_face_angle_max_deg = json["pseudo_view_source_rgb_face_angle_max_deg"];
            }
            if (json.contains("pseudo_view_mask_valid_open_pixels")) {
                params.pseudo_view_mask_valid_open_pixels = json["pseudo_view_mask_valid_open_pixels"];
            }
            if (json.contains("pseudo_view_mask_erode_pixels")) {
                params.pseudo_view_mask_erode_pixels = json["pseudo_view_mask_erode_pixels"];
            }
            if (json.contains("pseudo_view_mask_invalid_dilate_pixels")) {
                params.pseudo_view_mask_invalid_dilate_pixels = json["pseudo_view_mask_invalid_dilate_pixels"];
            }
            if (json.contains("pseudo_view_rgb_parallel_jobs")) {
                params.pseudo_view_rgb_parallel_jobs = json["pseudo_view_rgb_parallel_jobs"];
            }
            if (json.contains("pseudo_view_minimal_metadata")) {
                params.pseudo_view_minimal_metadata = json["pseudo_view_minimal_metadata"];
            }
            if (json.contains("use_pseudo_views_in_training")) {
                params.use_pseudo_views_in_training = json["use_pseudo_views_in_training"];
            }
            if (json.contains("pseudo_view_loss_weight")) {
                params.pseudo_view_loss_weight = json["pseudo_view_loss_weight"];
            }
            if (json.contains("enable_gggs_loss")) {
                params.enable_gggs_loss = json["enable_gggs_loss"];
            }
            if (json.contains("lambda_depth_normal")) {
                params.lambda_depth_normal = json["lambda_depth_normal"];
            }
            if (json.contains("regularization_from_iter")) {
                params.regularization_from_iter = json["regularization_from_iter"];
            }
            if (json.contains("gggs_gtnorm")) {
                params.gggs_gtnorm = json["gggs_gtnorm"];
            }
            if (json.contains("lambda_gggs_gtnorm")) {
                params.lambda_gggs_gtnorm = json["lambda_gggs_gtnorm"];
            }
            if (json.contains("gggs_gtnorm_from_iter")) {
                params.gggs_gtnorm_from_iter = json["gggs_gtnorm_from_iter"];
            }
            if (json.contains("enable_mesh_depth_loss")) {
                params.enable_mesh_depth_loss = json["enable_mesh_depth_loss"];
            }
            if (json.contains("lambda_mesh_depth")) {
                params.lambda_mesh_depth = json["lambda_mesh_depth"];
            }
            if (json.contains("mesh_depth_loss_from_iter")) {
                params.mesh_depth_loss_from_iter = json["mesh_depth_loss_from_iter"];
            }
            if (json.contains("log_level")) {
                params.log_level = to_lower_ascii(json["log_level"].get<std::string>());
            } else if (json.contains("log-level")) {
                params.log_level = to_lower_ascii(json["log-level"].get<std::string>());
            }
            if (json.contains("headless")) {
                params.headless = json["headless"];
            }
            if (json.contains("mip_filter")) {
                params.mip_filter = json["mip_filter"];
            }
            if (json.contains("use_bilateral_grid")) {
                params.use_bilateral_grid = json["use_bilateral_grid"];
            }
            if (json.contains("bilateral_grid_X")) {
                params.bilateral_grid_X = json["bilateral_grid_X"];
            }
            if (json.contains("bilateral_grid_Y")) {
                params.bilateral_grid_Y = json["bilateral_grid_Y"];
            }
            if (json.contains("bilateral_grid_W")) {
                params.bilateral_grid_W = json["bilateral_grid_W"];
            }
            if (json.contains("bilateral_grid_lr")) {
                params.bilateral_grid_lr = json["bilateral_grid_lr"];
            }
            if (json.contains("tv_loss_weight")) {
                params.tv_loss_weight = json["tv_loss_weight"];
            }
            if (json.contains("use_per_frame_affine_color")) {
                params.use_per_frame_affine_color = json["use_per_frame_affine_color"];
            }
            if (json.contains("save_per_frame_affine_color")) {
                params.save_per_frame_affine_color = json["save_per_frame_affine_color"];
            }
            if (json.contains("per_frame_affine_color_lr")) {
                params.per_frame_affine_color_lr = json["per_frame_affine_color_lr"];
            }
            if (json.contains("per_frame_affine_color_identity_reg_weight")) {
                params.per_frame_affine_color_identity_reg_weight = json["per_frame_affine_color_identity_reg_weight"];
            }
            if (json.contains("per_frame_affine_color_gauge_reg_weight")) {
                params.per_frame_affine_color_gauge_reg_weight = json["per_frame_affine_color_gauge_reg_weight"];
            }
            if (json.contains("use_per_frame_motion_blur")) {
                params.use_per_frame_motion_blur = json["use_per_frame_motion_blur"];
            }
            if (json.contains("use_per_frame_defocus_blur")) {
                params.use_per_frame_defocus_blur = json["use_per_frame_defocus_blur"];
            }
            if (json.contains("save_per_frame_blur_parameters")) {
                params.save_per_frame_blur_parameters = json["save_per_frame_blur_parameters"];
            }
            if (json.contains("per_frame_motion_blur_rot_lr")) {
                params.per_frame_motion_blur_rot_lr = json["per_frame_motion_blur_rot_lr"];
            }
            if (json.contains("per_frame_motion_blur_trans_lr")) {
                params.per_frame_motion_blur_trans_lr = json["per_frame_motion_blur_trans_lr"];
            }
            if (json.contains("per_frame_motion_blur_rot_reg_weight")) {
                params.per_frame_motion_blur_rot_reg_weight = json["per_frame_motion_blur_rot_reg_weight"];
            }
            if (json.contains("per_frame_motion_blur_trans_reg_weight")) {
                params.per_frame_motion_blur_trans_reg_weight = json["per_frame_motion_blur_trans_reg_weight"];
            }
            if (json.contains("per_frame_motion_blur_max_rot_std_deg")) {
                params.per_frame_motion_blur_max_rot_std_deg = json["per_frame_motion_blur_max_rot_std_deg"];
            }
            if (json.contains("per_frame_motion_blur_max_trans_std_scene_ratio")) {
                params.per_frame_motion_blur_max_trans_std_scene_ratio = json["per_frame_motion_blur_max_trans_std_scene_ratio"];
            }
            if (json.contains("per_frame_observation_blur_max_radius_px")) {
                params.per_frame_observation_blur_max_radius_px = json["per_frame_observation_blur_max_radius_px"];
            }
            if (json.contains("per_frame_motion_blur_start_iter")) {
                params.per_frame_motion_blur_start_iter = json["per_frame_motion_blur_start_iter"];
            }
            if (json.contains("per_frame_defocus_blur_scale_lr")) {
                params.per_frame_defocus_blur_scale_lr = json["per_frame_defocus_blur_scale_lr"];
            }
            if (json.contains("per_frame_defocus_blur_focus_lr")) {
                params.per_frame_defocus_blur_focus_lr = json["per_frame_defocus_blur_focus_lr"];
            }
            if (json.contains("per_frame_defocus_blur_reg_weight")) {
                params.per_frame_defocus_blur_reg_weight = json["per_frame_defocus_blur_reg_weight"];
            }
            if (json.contains("per_frame_defocus_blur_max_radius_px")) {
                params.per_frame_defocus_blur_max_radius_px = json["per_frame_defocus_blur_max_radius_px"];
            }
            if (json.contains("per_frame_defocus_blur_start_iter")) {
                params.per_frame_defocus_blur_start_iter = json["per_frame_defocus_blur_start_iter"];
            }
            if (json.contains("use_ppisp")) {
                params.use_ppisp = json["use_ppisp"];
            }
            if (json.contains("ppisp_exposure_only")) {
                params.ppisp_exposure_only = json["ppisp_exposure_only"];
            }
            if (json.contains("ppisp_lr")) {
                params.ppisp_lr = json["ppisp_lr"];
            }
            if (json.contains("ppisp_reg_weight")) {
                params.ppisp_reg_weight = json["ppisp_reg_weight"];
            }
            if (json.contains("ppisp_warmup_steps")) {
                params.ppisp_warmup_steps = json["ppisp_warmup_steps"];
            }
            if (json.contains("ppisp_freeze_from_sidecar")) {
                params.ppisp_freeze_from_sidecar = json["ppisp_freeze_from_sidecar"];
            }
            if (json.contains("ppisp_sidecar_path")) {
                params.ppisp_sidecar_path = utf8_to_path(json["ppisp_sidecar_path"].get<std::string>());
            }
            if (json.contains("ppisp_use_controller")) {
                params.ppisp_use_controller = json["ppisp_use_controller"];
            }
            if (json.contains("ppisp_freeze_gaussians_on_distill")) {
                params.ppisp_freeze_gaussians_on_distill = json["ppisp_freeze_gaussians_on_distill"];
            }
            if (json.contains("ppisp_controller_activation_step")) {
                params.ppisp_controller_activation_step = json["ppisp_controller_activation_step"];
            }
            if (json.contains("ppisp_controller_lr")) {
                params.ppisp_controller_lr = json["ppisp_controller_lr"];
            }
            if (json.contains("prune_opacity")) {
                params.prune_opacity = json["prune_opacity"];
            }
            if (json.contains("grow_scale3d")) {
                params.grow_scale3d = json["grow_scale3d"];
            }
            if (json.contains("grow_scale2d")) {
                params.grow_scale2d = json["grow_scale2d"];
            }
            if (json.contains("prune_scale3d")) {
                params.prune_scale3d = json["prune_scale3d"];
            }
            if (json.contains("prune_scale2d")) {
                params.prune_scale2d = json["prune_scale2d"];
            }
            if (json.contains("reset_every")) {
                params.reset_every = json["reset_every"];
            }
            if (json.contains("pause_refine_after_reset")) {
                params.pause_refine_after_reset = json["pause_refine_after_reset"];
            }
            if (json.contains("revised_opacity")) {
                params.revised_opacity = json["revised_opacity"];
            }
            if (json.contains("steps_scaler")) {
                params.steps_scaler = json["steps_scaler"];
            }
            if (json.contains("sh_degree_interval")) {
                params.sh_degree_interval = json["sh_degree_interval"];
            }
            if (json.contains("random")) {
                params.random = json["random"];
            }
            if (json.contains("init_num_pts")) {
                params.init_num_pts = json["init_num_pts"];
            }
            if (json.contains("init_extent")) {
                params.init_extent = json["init_extent"];
            }
            if (json.contains("tile_mode")) {
                params.tile_mode = json["tile_mode"];
            }
            if (json.contains("enable_sparsity")) {
                params.enable_sparsity = json["enable_sparsity"];
            }
            if (json.contains("sparsify_steps")) {
                params.sparsify_steps = json["sparsify_steps"];
            }
            if (json.contains("init_rho")) {
                params.init_rho = json["init_rho"];
            }
            if (json.contains("prune_ratio")) {
                params.prune_ratio = json["prune_ratio"];
            }
            if (json.contains("bg_modulation")) {
                params.bg_modulation = json["bg_modulation"];
            }
            if (json.contains("gut")) {
                params.gut = json["gut"];
            }
            if (json.contains("undistort")) {
                params.undistort = json["undistort"];
            }

            if (json.contains("bg_mode")) {
                const std::string mode = json["bg_mode"];
                if (mode == "solid_color") {
                    params.bg_mode = BackgroundMode::SolidColor;
                } else if (mode == "modulation") {
                    params.bg_mode = BackgroundMode::Modulation;
                } else if (mode == "image") {
                    params.bg_mode = BackgroundMode::Image;
                } else if (mode == "random") {
                    params.bg_mode = BackgroundMode::Random;
                }
            }
            if (json.contains("bg_color") && json["bg_color"].is_array() && json["bg_color"].size() == 3) {
                params.bg_color = {json["bg_color"][0], json["bg_color"][1], json["bg_color"][2]};
            }
            if (json.contains("bg_image_path")) {
                params.bg_image_path = utf8_to_path(json["bg_image_path"].get<std::string>());
            }

            // Mask parameters
            if (json.contains("mask_mode")) {
                std::string mode = json["mask_mode"];
                if (mode == "none") {
                    params.mask_mode = MaskMode::None;
                } else if (mode == "segment") {
                    params.mask_mode = MaskMode::Segment;
                } else if (mode == "ignore") {
                    params.mask_mode = MaskMode::Ignore;
                } else if (mode == "alpha_consistent") {
                    params.mask_mode = MaskMode::AlphaConsistent;
                } else if (mode == "project_mesh") {
                    params.mask_mode = MaskMode::ProjectMesh;
                }
            }
            if (json.contains("invert_masks")) {
                params.invert_masks = json["invert_masks"];
            }
            if (json.contains("mask_opacity_penalty_weight")) {
                params.mask_opacity_penalty_weight = json["mask_opacity_penalty_weight"];
            }
            if (json.contains("mask_opacity_penalty_power")) {
                params.mask_opacity_penalty_power = json["mask_opacity_penalty_power"];
            }
            if (json.contains("mask_threshold")) {
                params.mask_threshold = json["mask_threshold"];
            }
            if (json.contains("project_mesh_mask_erode_pixels")) {
                params.project_mesh_mask_erode_pixels = json["project_mesh_mask_erode_pixels"];
            }
            if (json.contains("use_alpha_as_mask")) {
                params.use_alpha_as_mask = json["use_alpha_as_mask"];
            }

            // Pyramid training parameters
            if (json.contains("pyramid_training")) {
                params.pyramid_training = json["pyramid_training"];
            }
            if (json.contains("pyramid_levels")) {
                params.pyramid_levels = json["pyramid_levels"];
            }
            if (json.contains("pyramid_step_interval")) {
                params.pyramid_step_interval = json["pyramid_step_interval"];
            }

            return params;
        }

        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path) {
            auto json_result = has_extension(path, ".bin")
                                   ? read_encrypted_json_file(path)
                                   : read_json_file(path);
            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            const auto& json = *json_result;
            // Support both flat and nested {"optimization": {...}} formats
            const auto& opt_json = json.contains("optimization") ? json["optimization"] : json;

            try {
                return OptimizationParameters::from_json(opt_json);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error parsing optimization parameters: {}", e.what()));
            }
        }

        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {
            try {
                auto opt_copy = params.optimization;
                opt_copy.remove_step_scaling();

                nlohmann::json json;
                json["dataset"] = params.dataset.to_json();
                json["optimization"] = opt_copy.to_json();

                const auto now = std::chrono::system_clock::now();
                const auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                json["timestamp"] = ss.str();

                const std::filesystem::path filepath = (output_path.extension() == ".json")
                                                           ? output_path
                                                           : output_path / "training_config.json";
                std::ofstream file;
                if (!open_file_for_write(filepath, file)) {
                    return std::unexpected(std::format("Cannot write: {}", path_to_utf8(filepath)));
                }

                file << json.dump(4);
                LOG_INFO("Saved config: {}", path_to_utf8(filepath));
                return {};
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error saving training parameters: {}", e.what()));
            }
        }

        LoadingParams LoadingParams::from_json(const nlohmann::json& j) {

            LoadingParams params;
            if (j.contains("use_cpu_memory")) {
                params.use_cpu_memory = j["use_cpu_memory"];
            }
            if (j.contains("min_cpu_free_memory_ratio")) {
                params.min_cpu_free_memory_ratio = j["min_cpu_free_memory_ratio"];
            }
            if (j.contains("min_cpu_free_GB")) {
                params.min_cpu_free_GB = j["min_cpu_free_GB"];
            }
            if (j.contains("use_fs_cache")) {
                params.use_fs_cache = j["use_fs_cache"];
            }
            if (j.contains("print_cache_status")) {
                params.print_cache_status = j["print_cache_status"];
            }
            if (j.contains("print_status_freq_num")) {
                params.print_status_freq_num = j["print_status_freq_num"];
            }

            return params;
        }

        nlohmann::json LoadingParams::to_json() const {
            nlohmann::json loading_json;
            loading_json["use_cpu_memory"] = use_cpu_memory;
            loading_json["min_cpu_free_memory_ratio"] = min_cpu_free_memory_ratio;
            loading_json["min_cpu_free_GB"] = min_cpu_free_GB;
            loading_json["use_fs_cache"] = use_fs_cache;
            loading_json["print_cache_status"] = print_cache_status;
            loading_json["print_status_freq_num"] = print_status_freq_num;

            return loading_json;
        }

        nlohmann::json DatasetConfig::to_json() const {
            nlohmann::json json;

            json["data_path"] = path_to_utf8(data_path);
            json["output_folder"] = path_to_utf8(output_path);
            json["images"] = images;
            json["resize_factor"] = resize_factor;
            json["test_every"] = test_every;
            json["max_width"] = max_width;
            json["loading_params"] = loading_params.to_json();
            json["invert_masks"] = invert_masks;
            json["mask_threshold"] = mask_threshold;

            return json;
        }

        DatasetConfig DatasetConfig::from_json(const nlohmann::json& j) {
            DatasetConfig dataset;

            // Use utf8_to_path for proper Unicode handling since JSON is UTF-8 encoded
            dataset.data_path = utf8_to_path(j["data_path"].get<std::string>());
            dataset.images = j["images"].get<std::string>();
            dataset.resize_factor = j["resize_factor"].get<int>();
            dataset.max_width = j["max_width"].get<int>();
            dataset.test_every = j["test_every"].get<int>();
            dataset.output_path = utf8_to_path(j["output_folder"].get<std::string>());

            if (j.contains("loading_params")) {
                dataset.loading_params = LoadingParams::from_json(j["loading_params"]);
            }
            if (j.contains("invert_masks")) {
                dataset.invert_masks = j["invert_masks"].get<bool>();
            }
            if (j.contains("mask_threshold")) {
                dataset.mask_threshold = j["mask_threshold"].get<float>();
            }

            return dataset;
        }

    } // namespace param
} // namespace lfs::core
