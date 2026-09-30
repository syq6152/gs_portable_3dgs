/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace lfs::core {
    namespace param {
        // Mask mode for attention mask behavior during training
        enum class MaskMode {
            None,            // No masking applied
            Segment,         // Soft penalty to enforce alpha→0 in masked areas
            Ignore,          // Completely ignore masked regions in loss
            AlphaConsistent, // Enforce exact alpha values from mask
            ProjectMesh      // Generate masks by projecting mesh into each camera (treated as Segment afterwards)
        };

        // Background mode for training - only one can be active at a time
        enum class BackgroundMode {
            SolidColor, // Use bg_color RGB values
            Modulation, // Sinusoidal background modulation
            Image,      // Use custom background image
            Random      // Random per-pixel colors each iteration
        };

        enum class Mesh2SplatHoleFillMode {
            None,
            Cgal,
            External,
            ExternalPointCloud,
        };

        enum class MeshInitColorSource {
            Auto,
            Texture,
            RgbViews,
            White,
        };

        [[nodiscard]] LFS_CORE_API std::string_view mesh2splat_hole_fill_mode_name(
            Mesh2SplatHoleFillMode mode);
        [[nodiscard]] LFS_CORE_API std::expected<Mesh2SplatHoleFillMode, std::string>
        parse_mesh2splat_hole_fill_mode(std::string_view value);

        [[nodiscard]] LFS_CORE_API std::string_view mesh_init_color_source_name(
            MeshInitColorSource source);
        [[nodiscard]] LFS_CORE_API std::expected<MeshInitColorSource, std::string>
        parse_mesh_init_color_source(std::string_view value);

        struct LFS_CORE_API OptimizationParameters {
            size_t iterations = 30'000;
            size_t sh_degree_interval = 1'000;
            float means_lr = 0.000016f;
            float shs_lr = 0.0025f;
            float opacity_lr = 0.025f;
            float scaling_lr = 0.005f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            size_t refine_every = 100;
            size_t start_refine = 500;
            size_t stop_refine = 25'000;
            float grad_threshold = 0.0002f;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
            float init_opacity = 0.5f;
            float init_scaling = 0.1f;
            int max_cap = 1000000;

            // Camera pose refinement during GS training. Disabled by default.
            // V1 implements nerfstudio-style SO3xR3 only and supports the FastGS
            // rasterizer path.
            bool refine_camera_pose = false;
            // Export refined camera poses to <output>/pose_refine when saving
            // PLY/checkpoints/final results. This does not affect pose
            // refinement itself or the PoseRefiner state stored in checkpoints.
            bool save_pose_refine_outputs = true;
            std::string pose_refine_mode = "SO3xR3";

            // --- Pose LR schedule: nerfstudio ExponentialDecayScheduler ---
            // Mirrors nerfstudio's camera_opt scheduler. The effective LR follows an
            // optional sin() warmup ramp (lr_pre_warmup -> lr_init over lr_warmup_steps),
            // then a log-linear decay from lr_init to lr_final over [lr_warmup_steps,
            // lr_max_steps]. The schedule is measured on the global training iteration
            // (independent of the activation window below). Set a final >= its init to
            // disable decay for that component.
            float pose_refine_lr_rot = 1.0e-4f;          // nerfstudio: lr (init)
            float pose_refine_lr_trans = 1.0e-4f;        // nerfstudio: lr (init)
            float pose_refine_lr_final_rot = 5.0e-7f;    // nerfstudio: lr_final
            float pose_refine_lr_final_trans = 5.0e-7f;  // nerfstudio: lr_final
            float pose_refine_lr_pre_warmup = 0.0f;      // nerfstudio: lr_pre_warmup
            size_t pose_refine_lr_warmup_steps = 0;      // nerfstudio: warmup_steps (0 = no ramp)
            size_t pose_refine_lr_max_steps = 0;         // nerfstudio: max_steps (0 = total iterations)

            // --- Pose regularization, clamps, and diagnostics ---
            // Pose refinement is active until pose_refine_stop_iter when enabled;
            // the LR schedule above controls how aggressively the deltas update over time.
            size_t pose_refine_stop_iter = 0; // 0 disables early stop; otherwise inactive from iter >= this value
            float pose_refine_l2_rot = 1.0e-3f;
            float pose_refine_l2_trans = 1.0e-2f;
            size_t pose_refine_log_every = 100; // 0 disables periodic pose-refine diagnostics
            float pose_refine_max_rot_deg = 10.0f;
            float pose_refine_max_trans = 0.5f;

            // When > 0 and the training model was initialized by mesh2splat, override
            // max_cap to ceil(initial_mesh2splat_gaussians * (1 + this ratio)).
            float mesh2splat_max_cap_extra_ratio = 0.0f;
            bool mesh2splat_opacity_no_grad = false; // Zero opacity gradients on mesh-init GS before MCMC/ADC optimizer step

            // Initial Gaussian color source for training mesh initialization. Auto keeps
            // textured meshes on the existing texture path and colorizes texture-less
            // meshes from registered training RGB views.
            MeshInitColorSource mesh_init_color_source = MeshInitColorSource::Auto;

            // Optional CGAL hole filling for the training-only mesh initialization path.
            // Ratios are resolved against the normalized source mesh bounding-box diagonal.
            Mesh2SplatHoleFillMode mesh2splat_hole_fill_mode = Mesh2SplatHoleFillMode::None;
            // Deprecated compatibility alias. When no explicit mode is present, true means Cgal.
            bool mesh2splat_hole_fill_enabled = false;
            int mesh2splat_hole_fill_max_boundary_edges = 128;
            float mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio = 0.10f;
            float mesh2splat_hole_fill_max_boundary_area_bbox_ratio = 0.0025f;
            float mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio = 0.05f;
            float mesh2splat_hole_fill_weld_epsilon_bbox_ratio = 1e-6f;
            float mesh2splat_hole_fill_density_control_factor = 1.41421356f;
            float mesh2splat_hole_fill_min_density_ratio = 0.5f;
            float mesh2splat_hole_fill_max_density_ratio = 2.0f;

            // External mode: tolerance band around the original mesh hole rims.
            // External tools often move or retriangulate the rim while filling a hole, so
            // the render-face to constraint-face derivation relaxes its match test inside
            // this band and stays strict outside it. Both ratios resolve against the
            // normalized original mesh bounding-box diagonal.
            // Band radius: an original face is "near a hole" when any of its vertices lies
            // within this distance of a boundary vertex.
            float mesh2splat_external_band_ratio = 0.05f;
            // Maximum centroid displacement accepted for a band face to still map onto an
            // original face. Band faces beyond it are treated as new hole-fill geometry.
            float mesh2splat_external_band_tolerance_ratio = 0.01f;

            // ExternalPointCloud mode: the original mesh stays the only render/constraint/
            // color/project-mask source; an external point cloud seeds Gaussians in the
            // regions the mesh is missing.
            // Target point-cloud Gaussian density as a fraction of the achieved mesh
            // Gaussian density. 0 disables density matching and uses the cloud as-is.
            float mesh2splat_pointcloud_density_ratio = 1.0f;
            // Points closer to the original mesh surface than this multiple of the mesh
            // Gaussian spacing are dropped as redundant. 0 disables the rejection.
            float mesh2splat_pointcloud_surface_reject_ratio = 0.5f;
            // project-mask: minimum projected point count inside an enclosed silhouette
            // hole before that hole is filled. This is the only knob on the main path; it
            // separates "scan gap" from "genuine aperture through the object".
            int mesh2splat_pointcloud_mask_min_points_per_hole = 4;
            // project-mask: closing radius applied to the mesh coverage before component
            // labeling, sealing single-pixel rim speckles that would leak an enclosed hole
            // to the image border.
            int mesh2splat_pointcloud_mask_rim_close_pixels = 1;
            // project-mask fallback for holes that open onto the silhouette boundary: disc
            // radius as a multiple of the projected point spacing. 0 disables the fallback.
            float mesh2splat_pointcloud_mask_splat_radius_scale = 1.5f;
            // project-mask fallback closing radius. 0 derives it from the projected spacing.
            int mesh2splat_pointcloud_mask_close_pixels = 0;

            [[nodiscard]] Mesh2SplatHoleFillMode resolved_mesh2splat_hole_fill_mode() const;

            // [mesh-surface-constraint] Maximum edge crossings per constrained Gaussian
            // projection. Larger values handle bigger optimizer/noise jumps at proportional
            // GPU cost. The projection path is always surface-walk based.
            int mesh_surface_walk_steps = 10;
            int mesh_surface_loss_from_iter = 0;
            bool mesh_surface_hard_projection_enabled = true; // legacy post-step mean projection when lambda_mesh_project is 0
            bool mesh_depth_visibility_cull = false;          // Cull GS behind mesh depth before alpha blending when enabled
            float lambda_mesh_project = 0.0f;
            float lambda_mesh_outside_barrier = 0.0f;
            float mesh_outside_distance_avg_max_scale_multiplier = 1.0f;
            // When > 0, retain mesh constraints outside the mesh and within an
            // interior surface band of this many mesh2splat mean max scales.
            // The final fraction of the band fades smoothly to zero.
            float mesh_inside_constraint_distance_avg_max_scale_multiplier = 0.0f;
            float mesh_inside_constraint_fade_ratio = 0.2f;
            float lambda_mesh_scale_min = 0.0f;
            float lambda_mesh_scale_max = 0.0f;
            float mesh_scale_rho_ratio = 0.01f;
            float mesh_scale_rho_avg_max_scale_multiplier = 0.0f;
            float lambda_mesh_normal = 0.0f;
            std::vector<size_t> eval_steps = {7'000, 30'000}; // Steps to evaluate the model
            std::vector<size_t> save_steps = {7'000, 30'000}; // Steps to save the model
            bool bg_modulation = false;                       // Enable sinusoidal background modulation
            bool enable_eval = false;                         // Only evaluate when explicitly enabled
            std::vector<std::string> train_images = {};       // Explicit train split image names from JSON config
            std::vector<std::string> test_images = {};        // Explicit validation split image names from JSON config
            bool enable_save_eval_images = true;              // Save during evaluation images
            bool save_depth = false;                          // Save rendered depth/normal maps for eval/timelapse
            bool precompute_mesh_depth_normal = false;        // Enable per-step mesh GT depth/normal render during forward passes
            float mesh_depth_loss_weight = 0.0f;              // Reserved interface for future mesh depth supervision loss
            float mesh_normal_loss_weight = 0.0f;             // Reserved interface for future mesh normal supervision loss

            // Pseudo-view RGB supervision precompute (--pseudo-view).
            // When enabled, before training the trainer marks real-camera direction cells,
            // generates sphere-grid pseudo cameras for empty cells, picks the best valid source
            // camera from top-K candidates, and writes the result to <output>/pseudo_views/.
            bool precompute_pseudo_view = false;

            // Step 1: sphere-grid direction assignment. Real cameras occupy cells around the
            // mesh center; empty cells with surface-normal support emit pseudo poses.
            float pseudo_view_sphere_cell_angle_deg = 30.0f;    // longitude/latitude cell size in degrees
            float pseudo_view_sphere_support_angle_deg = 30.0f; // max angle between face normal and sphere cell ray for target selection

            // Step 2: pseudo camera pose filtering.
            float pseudo_view_mesh_front_angle_deg = 90.0f;     // max angle between pseudo camera forward and mesh center ray
            float pseudo_view_face_angle_max_deg = 20.0f;        // max angle between surface normal and pseudo camera ray

            // Step 3: source RGB selection by mesh-depth reprojection.
            int pseudo_view_top_k_source = 8;                   // top-K source cameras to evaluate for best-source RGB
            float pseudo_view_min_valid_ratio = 0.40f;           // reject pseudo views with best-source valid ratio below this
            int pseudo_view_min_valid_pixels = 0;                // optional absolute supervision-pixel floor; 0 disables it
            float pseudo_view_rgb_face_angle_max_deg = 85.0f;    // max target-side per-pixel face angle to pseudo camera
            float pseudo_view_source_rgb_face_angle_max_deg = 80.0f; // max source-side face angle to source camera; 180 disables
            int pseudo_view_mask_valid_open_pixels = 2;           // remove narrow valid slivers before pseudo supervision-mask erosion
            int pseudo_view_mask_erode_pixels = 6;                // erode pseudo supervision mask inward for SSIM-window safety
            int pseudo_view_mask_invalid_dilate_pixels = 1;       // dilate invalid pixels before pseudo supervision-mask erosion
            int pseudo_view_rgb_parallel_jobs = 1;                // final pseudo RGB/mask output jobs; 1 serial, valid range [1,4]
            bool pseudo_view_minimal_metadata = false;            // Save only training-required pseudo-view JSON metadata

            // Pseudo-view training integration. When the pseudo-view cache is available
            // (precompute_pseudo_view ran or a manifest exists in the output directory),
            // these knobs decide whether to inject pseudo cameras into the training set
            // and how strongly their photometric loss contributes relative to real views.
            bool use_pseudo_views_in_training = false;            // Append pseudo cameras to the train split
            float pseudo_view_loss_weight = 0.1f;                // Multiplier applied to per-pixel photometric loss/grad for pseudo cameras

            // GGGS regularization losses (Equations 25 & 26)
            bool enable_gggs_loss = false;                    // Enable GGGS depth-normal and multi-view losses
            float lambda_depth_normal = 0.05f;                // Weight for normal consistency loss (Eq 25)
            int regularization_from_iter = 7000;              // Start GGGS regularization losses after this iteration
            bool gggs_gtnorm = false;                         // Enable GT normal loss (depth_normal vs mesh GT normal)
            float lambda_gggs_gtnorm = 0.05f;                 // Weight for GT normal consistency loss
            int gggs_gtnorm_from_iter = 7000;                 // Start GT normal loss after this iteration

            bool enable_mesh_depth_loss = false;              // Enable mesh GT inverse depth loss (|1/rendered - 1/gt|)
            float lambda_mesh_depth = 0.05f;                  // Weight for mesh GT inverse depth loss
            int mesh_depth_loss_from_iter = 7000;             // Start mesh GT depth loss after this iteration

            bool headless = false;                            // Disable visualization during training
            bool auto_train = false;                          // Start training immediately on startup
            bool no_splash = false;                           // Skip splash screen on startup
            bool no_interop = false;                          // Disable CUDA-GL interop (use CPU fallback)
            bool debug_python = false;                        // Start debugpy listener for plugin debugging
            int debug_python_port = 5678;                     // Port for debugpy listener
            std::string log_level = "";                       // Optional JSON/CLI log level override
            std::string strategy = "mcmc";                    // Optimization strategy: mcmc, adc.

            // Mask parameters
            MaskMode mask_mode = MaskMode::None;      // Attention mask mode
            bool invert_masks = false;                // Invert mask values (swap object/background)
            float mask_threshold = 0.5f;              // Threshold: >= threshold → 1.0, < threshold → keep original
            float mask_opacity_penalty_weight = 1.0f; // Opacity penalty weight for segment mode
            float mask_opacity_penalty_power = 2.0f;  // Penalty falloff (1=linear, 2=quadratic)
            int project_mesh_mask_erode_pixels = 0;   // Erode project_mesh-generated masks inward before saving
            bool use_alpha_as_mask = true;            // Auto-use alpha channel from RGBA images as mask

            // Mip filter (anti-aliasing)
            bool mip_filter = false;

            // Background settings for training
            BackgroundMode bg_mode = BackgroundMode::SolidColor; // Which background mode to use
            std::array<float, 3> bg_color = {0.0f, 0.0f, 0.0f};  // RGB background color [0-1]
            std::filesystem::path bg_image_path = {};            // Custom background image path

            // Bilateral grid parameters
            bool use_bilateral_grid = false;
            int bilateral_grid_X = 16;
            int bilateral_grid_Y = 16;
            int bilateral_grid_W = 8;
            float bilateral_grid_lr = 2e-3f;
            float tv_loss_weight = 10.f;

            // Per-frame affine color parameters
            bool use_per_frame_affine_color = false;
            bool save_per_frame_affine_color = false;
            float per_frame_affine_color_lr = 2e-3f;
            float per_frame_affine_color_identity_reg_weight = 1e-3f;
            float per_frame_affine_color_gauge_reg_weight = 1e-2f;

            // Per-frame observation blur nuisance model from Robust Gaussian
            // Splatting. Both branches are disabled by default so legacy
            // training keeps the exact canonical rasterization path.
            bool use_per_frame_motion_blur = false;
            bool use_per_frame_defocus_blur = false;
            bool save_per_frame_blur_parameters = false;

            float per_frame_motion_blur_rot_lr = 1e-4f;
            float per_frame_motion_blur_trans_lr = 1e-4f;
            float per_frame_motion_blur_rot_reg_weight = 1e-3f;
            float per_frame_motion_blur_trans_reg_weight = 1e-3f;
            float per_frame_motion_blur_max_rot_std_deg = 3.0f;
            float per_frame_motion_blur_max_trans_std_scene_ratio = 0.05f;
            // Caps the major-axis standard deviation of the combined
            // motion+defocus covariance added in screen space.
            float per_frame_observation_blur_max_radius_px = 16.0f;
            size_t per_frame_motion_blur_start_iter = 1000;

            float per_frame_defocus_blur_scale_lr = 1e-3f;
            float per_frame_defocus_blur_focus_lr = 1e-4f;
            float per_frame_defocus_blur_reg_weight = 1e-3f;
            float per_frame_defocus_blur_max_radius_px = 32.0f;
            size_t per_frame_defocus_blur_start_iter = 3000;

            // PPISP (Physically-Plausible ISP) parameters
            bool use_ppisp = false;
            // Restrict PPISP optimization to the per-frame log2 exposure scalar.
            bool ppisp_exposure_only = false;
            float ppisp_lr = 2e-3f;
            float ppisp_reg_weight = 0.001f;
            int ppisp_warmup_steps = 500;
            bool ppisp_freeze_from_sidecar = false;
            std::filesystem::path ppisp_sidecar_path = {};
            bool ppisp_use_controller = false;
            bool ppisp_freeze_gaussians_on_distill = true;
            int ppisp_controller_activation_step = -1; // Negative values use the default tail schedule
            float ppisp_controller_lr = 2e-3f;

            // adc strategy specific parameters
            float prune_opacity = 0.005f;
            float grow_scale3d = 0.01f;
            float grow_scale2d = 0.05f;
            float prune_scale3d = 0.1f;
            float prune_scale2d = 0.15f;
            size_t reset_every = 3'000;
            size_t pause_refine_after_reset = 0;
            bool revised_opacity = false;
            bool gut = false;
            bool undistort = false;
            float steps_scaler = 1.f; // Scales training step counts; values <= 0 disable scaling

            // Random initialization parameters
            bool random = false;        // Use random initialization instead of SfM
            int init_num_pts = 100'000; // Number of random points to initialize
            float init_extent = 3.0f;   // Extent of random point cloud

            // Tile mode for memory-efficient training (1=1 tile, 2=2 tiles, 4=4 tiles)
            int tile_mode = 1;

            // Sparsity optimization parameters
            bool enable_sparsity = false;
            int sparsify_steps = 15000;
            float init_rho = 0.0005f;
            float prune_ratio = 0.6f;

            // Pyramid training parameters
            // When enabled, training starts with low-resolution images and progressively
            // increases to full resolution over the course of training.
            // Level 0 = full resolution (base resize_factor), level (pyramid_levels-1) = coarsest.
            // Each coarser level halves the resolution (doubles the effective resize_factor).
            // Transition schedule: iterations are split evenly across pyramid levels.
            //   e.g. pyramid_levels=3, pyramid_step_interval=1000:
            //     iter [1, 1000]   → level 2 (1/4 resolution)
            //     iter [1001,2000] → level 1 (1/2 resolution)
            //     iter [2001, end] → level 0 (full resolution)
            bool pyramid_training = false;
            int pyramid_levels = 3;            // Number of pyramid levels (including full resolution)
            int pyramid_step_interval = 1000;  // Iterations per pyramid level (from coarsest to finest)

            std::string config_file = "";

            void scale_steps(float ratio);
            void apply_step_scaling();
            void remove_step_scaling();
            [[nodiscard]] int resolved_ppisp_controller_activation_step() const;

            nlohmann::json to_json() const;
            static OptimizationParameters from_json(const nlohmann::json& j);

            [[nodiscard]] std::string validate() const;

            // Factory methods for strategy presets
            static OptimizationParameters mcmc_defaults();
            static OptimizationParameters adc_defaults();
            static OptimizationParameters igs_plus_defaults();
        };

        struct LFS_CORE_API LoadingParams {
            bool use_cpu_memory = true;
            float min_cpu_free_memory_ratio = 0.1f; // make sure at least 10% RAM is free
            float min_cpu_free_GB = 1.0f;           // min GB we want to be free
            bool use_fs_cache = true;
            bool print_cache_status = true;
            int print_status_freq_num = 500; // every print_status_freq_num calls for load print cache status

            nlohmann::json to_json() const;
            static LoadingParams from_json(const nlohmann::json& j);
        };

        struct LFS_CORE_API DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "";
            std::string images = "images";
            int resize_factor = -1;
            int test_every = 8;
            std::vector<std::string> timelapse_images = {};
            int timelapse_every = 50;
            int max_width = 3840;
            LoadingParams loading_params;

            // Mask loading parameters (copied from optimization params)
            bool invert_masks = false;
            float mask_threshold = 0.5f;

            nlohmann::json to_json() const;
            static DatasetConfig from_json(const nlohmann::json& j);
        };

        struct LFS_CORE_API TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;

            // Viewer mode: splat files to load (.ply, .sog, .resume)
            std::vector<std::filesystem::path> view_paths;

            // COLMAP sparse folder for camera-only import (no images required)
            std::optional<std::filesystem::path> import_cameras_path = std::nullopt;

            // Optional splat file for initialization (.ply, .sog, .spz, .resume)
            std::optional<std::string> init_path = std::nullopt;

            // Resolved .obj mesh file path from --mesh2splat-dir.
            std::optional<std::string> mesh2splat_mesh_path = std::nullopt;

            // Resolved .png texture file path from --mesh2splat-dir.
            std::optional<std::string> mesh2splat_texture_path = std::nullopt;

            // Enable mesh-based initialization via --mesh-init-gs-scene.
            bool use_mesh_init = false;

            // Resolved .ply or .obj mesh file path from --mesh-init-gs-scene.
            std::optional<std::string> mesh_init_mesh_path = std::nullopt;

            // Optional resolved .png texture file path from legacy --mesh-init-gs-scene directory input.
            // File input resolves textures from the mesh itself: OBJ/MTL map_Kd or PLY TextureFile comment.
            std::optional<std::string> mesh_init_texture_path = std::nullopt;

            // External mode: already completed/final render mesh. Its faces must keep the
            // original mesh as a prefix; hole-fill faces are appended after them so the
            // render-face to constraint-face mapping can be derived without a sidecar.
            std::optional<std::string> mesh_init_external_filled_mesh_path = std::nullopt;

            // ExternalPointCloud mode: point cloud covering the regions the original mesh
            // is missing. Its coordinates must match the original mesh before scene
            // normalization; colors in the file are ignored (Gaussians start white).
            std::optional<std::string> mesh_init_external_pointcloud_path = std::nullopt;

            // Target mesh-derived raster density ratio for initialization, in (0, 1].
            float mesh_init_sampling_rate = 1.0f;

            // Override initial Gaussian colors to pure white (skip texture-based color init).
            bool init_white_gs_color = false;

            // Checkpoint to resume training from
            std::optional<std::filesystem::path> resume_checkpoint = std::nullopt;

            // Python scripts to execute for custom training callbacks
            std::vector<std::filesystem::path> python_scripts;

            [[nodiscard]] std::string validate() const;
        };

        // Output format for conversion tool
        enum class OutputFormat { PLY,
                                  SOG,
                                  SPZ,
                                  SPLAT,
                                  HTML };

        // Parameters for the convert command
        struct LFS_CORE_API ConvertParameters {
            std::filesystem::path input_path;
            std::filesystem::path output_path; // Empty = derive from input
            OutputFormat format = OutputFormat::PLY;
            int sh_degree = 3; // 0-3, -1 = keep original
            int sog_iterations = 10;
            bool overwrite = false; // Skip overwrite prompts
        };

        // Parameters for the filter-error-scan-pose command
        struct LFS_CORE_API ScanPoseFilterParameters {
            std::filesystem::path input_bin;
            std::filesystem::path output_bin;
            std::filesystem::path report_path;
            float zncc_threshold = 0.30f;
        };

        // Modern C++23 functions returning expected values
        LFS_CORE_API std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path);

        // Save training parameters to JSON
        LFS_CORE_API std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);

    } // namespace param
} // namespace lfs::core
