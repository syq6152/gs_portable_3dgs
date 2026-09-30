/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_crypto.hpp"
#include "core/path_utils.hpp"
#include "config.h" // IWYU pragma: keep
#include <algorithm>
#include <args.hxx>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>
#if LFS_ENABLE_CLI_HELP
#include <print>
#endif
#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

    enum class ParseResult : std::uint8_t {
        Success,
        Help
    };

    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "adc", "igs+", "mesh2splat"};

    bool is_encrypted_path_option(const std::string_view option) {
        return option == "-d" || option == "--data-path" ||
               option == "-o" || option == "--output-path" ||
               option == "--config" ||
               option == "--mesh-init-gs-scene" ||
               option == "--mesh-init-external-filled-mesh" ||
               option == "--mesh-init-external-pointcloud";
    }

    bool is_inline_short_path_option(const std::string_view arg) {
        return arg.size() > 2 && !arg.starts_with("--") &&
               (arg.starts_with("-d") || arg.starts_with("-o"));
    }

    bool is_help_argument(const std::string_view arg) {
        return arg == "-h" || arg == "--help";
    }

    std::string append_cli_help(std::string message, const ::args::ArgumentParser& parser) {
#if LFS_ENABLE_CLI_HELP
        return std::format("{}\n\n{}", std::move(message), parser.Help());
#else
        (void) parser;
        return message;
#endif
    }

    void decrypt_path_arguments(std::vector<std::string>& args) {
        for (size_t i = 1; i < args.size(); ++i) {
            std::string& arg = args[i];
            if (arg == "--") {
                break;
            }

            const size_t eq_pos = arg.find('=');
            if (eq_pos != std::string::npos) {
                const std::string_view option(arg.data(), eq_pos);
                if (is_encrypted_path_option(option)) {
                    const std::string_view value(arg.data() + eq_pos + 1, arg.size() - eq_pos - 1);
                    arg = std::string(option) + "=" + lfs::core::path_crypto::decrypt_path_or_original(value);
                }
                continue;
            }

            if (is_encrypted_path_option(arg)) {
                if (i + 1 < args.size()) {
                    args[i + 1] = lfs::core::path_crypto::decrypt_path_or_original(args[i + 1]);
                    ++i;
                }
                continue;
            }

            if (is_inline_short_path_option(arg)) {
                const std::string option = arg.substr(0, 2);
                const std::string_view value(arg.data() + 2, arg.size() - 2);
                arg = option + lfs::core::path_crypto::decrypt_path_or_original(value);
            }
        }
    }

    // Parse log level from string
    lfs::core::LogLevel parse_log_level(const std::string& level_str) {
        std::string normalized = level_str;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (normalized == "trace")
            return lfs::core::LogLevel::Trace;
        if (normalized == "debug")
            return lfs::core::LogLevel::Debug;
        if (normalized == "info")
            return lfs::core::LogLevel::Info;
        if (normalized == "perf" || normalized == "performance")
            return lfs::core::LogLevel::Performance;
        if (normalized == "warn" || normalized == "warning")
            return lfs::core::LogLevel::Warn;
        if (normalized == "error")
            return lfs::core::LogLevel::Error;
        if (normalized == "critical")
            return lfs::core::LogLevel::Critical;
        if (normalized == "off")
            return lfs::core::LogLevel::Off;
        return lfs::core::LogLevel::Info; // Default
    }

    std::string to_lower_ascii(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::expected<std::string, std::string> normalize_pose_refine_mode(std::string value) {
        value = to_lower_ascii(std::move(value));
        if (value == "off" || value == "none" || value == "false") {
            return std::string("off");
        }
        if (value == "so3xr3" || value == "direct") {
            return std::string("SO3xR3");
        }
        if (value == "se3") {
            return std::unexpected("ERROR: --pose-opt SE3 is reserved but not implemented yet; use SO3xR3");
        }
        if (value == "mlp") {
            return std::unexpected("ERROR: --pose-opt mlp is not implemented in this LibTorch-free pose refine path; use SO3xR3");
        }
        return std::unexpected("ERROR: --pose-opt must be off, SO3xR3, or direct");
    }

    std::vector<std::filesystem::path> collect_files_with_extension(
        const std::filesystem::path& directory,
        const std::string_view extension) {

        std::vector<std::filesystem::path> matches;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            const auto ext = to_lower_ascii(entry.path().extension().string());
            if (ext == extension) {
                matches.push_back(entry.path());
            }
        }

        std::ranges::sort(matches);
        return matches;
    }

    std::expected<std::filesystem::path, std::string> require_single_asset(
        const std::filesystem::path& directory,
        const std::string_view extension,
        const std::string_view asset_name) {

        if (!std::filesystem::exists(directory)) {
            return std::unexpected(std::format(
                "Path does not exist: {}",
                lfs::core::path_to_utf8(directory)));
        }

        if (!std::filesystem::is_directory(directory)) {
            return std::unexpected(std::format(
                "Expected directory path but got file: {}",
                lfs::core::path_to_utf8(directory)));
        }

        const auto matches = collect_files_with_extension(directory, extension);
        if (matches.empty()) {
            return std::unexpected(std::format(
                "Missing required .{} file for {} in directory '{}'",
                extension.substr(1),
                asset_name,
                lfs::core::path_to_utf8(directory)));
        }
        if (matches.size() > 1) {
            return std::unexpected(std::format(
                "Expected exactly one .{} file for {} in directory '{}', found {}",
                extension.substr(1),
                asset_name,
                lfs::core::path_to_utf8(directory),
                matches.size()));
        }

        return matches.front();
    }

    struct MeshDirectoryAssets {
        std::filesystem::path mesh_path;
        std::filesystem::path texture_path;
    };

    struct MeshInitAssets {
        std::filesystem::path mesh_path;
        std::optional<std::filesystem::path> texture_path;
    };

    std::expected<MeshDirectoryAssets, std::string> resolve_mesh2splat_assets_from_directory(
        const std::filesystem::path& directory) {

        auto obj_file = require_single_asset(directory, ".obj", "mesh2splat mesh");
        if (!obj_file) {
            return std::unexpected(obj_file.error());
        }

        auto mtl_file = require_single_asset(directory, ".mtl", "mesh2splat material");
        if (!mtl_file) {
            return std::unexpected(mtl_file.error());
        }

        auto png_file = require_single_asset(directory, ".png", "mesh2splat texture");
        if (!png_file) {
            return std::unexpected(png_file.error());
        }

        return MeshDirectoryAssets{.mesh_path = *obj_file, .texture_path = *png_file};
    }

    std::expected<MeshInitAssets, std::string> resolve_mesh_init_assets_from_directory(
        const std::filesystem::path& directory) {

        auto png_file = require_single_asset(directory, ".png", "mesh init texture");
        if (!png_file) {
            return std::unexpected(png_file.error());
        }

        const auto ply_files = collect_files_with_extension(directory, ".ply");
        const auto obj_files = collect_files_with_extension(directory, ".obj");
        const auto mtl_files = collect_files_with_extension(directory, ".mtl");

        const bool has_ply_png = ply_files.size() == 1;
        const bool has_obj_mtl_png = obj_files.size() == 1 && mtl_files.size() == 1;

        if (has_ply_png && has_obj_mtl_png) {
            return std::unexpected(std::format(
                "Ambiguous mesh init directory '{}': found both ply+png and obj+mtl+png asset sets",
                lfs::core::path_to_utf8(directory)));
        }

        if (has_obj_mtl_png) {
            return MeshInitAssets{.mesh_path = obj_files.front(), .texture_path = *png_file};
        }

        if (has_ply_png) {
            return MeshInitAssets{.mesh_path = ply_files.front(), .texture_path = *png_file};
        }

        if (obj_files.size() > 1) {
            return std::unexpected(std::format(
                "Expected exactly one .obj file for mesh init mesh in directory '{}', found {}",
                lfs::core::path_to_utf8(directory),
                obj_files.size()));
        }
        if (obj_files.size() == 1 && mtl_files.empty()) {
            return std::unexpected(std::format(
                "Missing required .mtl file for mesh init material in directory '{}'",
                lfs::core::path_to_utf8(directory)));
        }
        if (obj_files.size() == 1 && mtl_files.size() > 1) {
            return std::unexpected(std::format(
                "Expected exactly one .mtl file for mesh init material in directory '{}', found {}",
                lfs::core::path_to_utf8(directory),
                mtl_files.size()));
        }
        if (ply_files.size() > 1) {
            return std::unexpected(std::format(
                "Expected exactly one .ply file for mesh init mesh in directory '{}', found {}",
                lfs::core::path_to_utf8(directory),
                ply_files.size()));
        }

        return std::unexpected(std::format(
            "Missing mesh init asset set in directory '{}': expected either obj+mtl+png or ply+png",
            lfs::core::path_to_utf8(directory)));
    }

    std::expected<MeshInitAssets, std::string> resolve_mesh_init_assets(
        const std::filesystem::path& path) {

        if (!std::filesystem::exists(path)) {
            return std::unexpected(std::format(
                "Path does not exist: {}",
                lfs::core::path_to_utf8(path)));
        }

        if (std::filesystem::is_directory(path)) {
            return resolve_mesh_init_assets_from_directory(path);
        }

        if (!std::filesystem::is_regular_file(path)) {
            return std::unexpected(std::format(
                "Mesh init path is not a regular file or directory: {}",
                lfs::core::path_to_utf8(path)));
        }

        const auto ext = to_lower_ascii(path.extension().string());
        if (ext == ".ply" || ext == ".obj") {
            return MeshInitAssets{.mesh_path = path, .texture_path = std::nullopt};
        }

        return std::unexpected(std::format(
            "Mesh init requires a .ply/.obj mesh file or a directory with obj+mtl+png or ply+png: {}",
            lfs::core::path_to_utf8(path)));
    }

    std::expected<std::tuple<ParseResult, std::function<void()>>, std::string> parse_arguments(
        const std::vector<std::string>& args,
        lfs::core::param::TrainingParameters& params) {

        try {
            ::args::ArgumentParser parser(
                "LichtFeld Studio: High-performance CUDA implementation of 3D Gaussian Splatting algorithm.\n",
                "\nSUBCOMMANDS:\n"
                "convert -- Convert between .ply, .sog, .spz, .html\n"
                "filter-error-scan-pose -- Filter scan frames with inconsistent poses\n"
                "plugin -- Manage plugins (create, check, list)\n"
                "\n"
                "Run '<subcommand> --help' for details.\n"
                "\n"
                "EXAMPLES:\n"
                "lichtfeld-studio -d ./data -o ./output\n"
                "lichtfeld-studio --resume checkpoint.resume\n"
                "lichtfeld-studio -v model.ply\n"
                "lichtfeld-studio convert in.ply out.spz\n"
                 "Run-GS.exe filter-error-scan-pose --input-bin scan.bin --output-bin filtered.bin --report report.json\n"
                 "lichtfeld-studio plugin create my_plugin\n"
                 "\n"
                 "SWAPTEXTURE DIRECT RECONSTRUCTION:\n"
                 "Run-GS.exe --bin_path <scanner bin> [--images_inc_path <directory>]\n"
                 "  [--enable_gs_train] [--gs-input-source registered|scan]\n"
                 "  [--gs-training-mode fast|medium|quality] [--mesh-init-gs-scene <mesh>]\n"
                 "  [--result_path <directory or .ply>] [--log_path <directory>]\n"
                 "  [--enable_inc_sim3_registration] [-- <LichtFeld training overrides>]\n"
                 "Defaults: registered input, medium quality, GS training disabled.\n"
                 "Other fusion settings are loaded from encrypted bin/Swaptexture_params.bin in the installed package.\n"
                 "The six GS presets are selected by input source and training mode; strategy is chosen at build time.\n"
                 "Sim3 registration is reserved but currently reports UnsupportedFeature.\n"
                 "Use Run-GS.exe reconstruct --help for the native reconstruction entry point.\n"
                 "\n"
                 "ENVIRONMENT:\n"
                "LOG_LEVEL -- Set log level (trace/debug/info/perf/warn/error)\n");
            parser.helpParams.width = 240;

            // =============================================================================
            // MODE SELECTION
            // =============================================================================
            ::args::Group mode_group(parser, "MODE SELECTION:");
#if LFS_ENABLE_CLI_HELP
            ::args::HelpFlag help(mode_group, "help", "Display help menu", {'h', "help"});
#endif
            ::args::Flag version(mode_group, "version", "Display version information", {'V', "version"});
            ::args::ValueFlag<std::string> view_ply(mode_group, "path", "View file(s). Supports splat (.ply, .sog, .spz) and mesh (.obj, .fbx, .gltf, .glb, .stl) formats. If directory, loads all.", {'v', "view"});
            ::args::ValueFlag<std::string> resume_checkpoint(mode_group, "checkpoint", "Resume training from checkpoint file", {"resume"});
#if LFS_ENABLE_CLI_HELP
            ::args::CompletionFlag completion(parser, {"complete"});
#endif

            // =============================================================================
            // TRAINING PATHS
            // =============================================================================
            ::args::Group paths_sep(parser, " ");
            ::args::Group paths_group(parser, "TRAINING PATHS:");
            ::args::ValueFlag<std::string> data_path(paths_group, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(paths_group, "output_path", "Path to output", {'o', "output-path"});
            ::args::ValueFlag<std::string> config_file(paths_group, "config_file", "LichtFeldStudio config file (.json or encrypted .bin)", {"config"});
            ::args::ValueFlag<std::string> init_path(paths_group, "path", "Initialize from splat file (.ply, .sog, .spz, .resume)", {"init"});

            ::args::ValueFlag<std::string> import_cameras(paths_group, "path", "Import COLMAP cameras from sparse folder (no images required)", {"import-cameras"});

            // =============================================================================
            // TRAINING PARAMETERS
            // =============================================================================
            ::args::Group training_sep(parser, " ");
            ::args::Group training_group(parser, "TRAINING PARAMETERS:");
            ::args::ValueFlag<uint32_t> iterations(training_group, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<std::string> strategy(training_group, "strategy", "Optimization strategy: mcmc, adc, igs+, mesh2splat", {"strategy"});
            ::args::ValueFlag<int> sh_degree(training_group, "sh_degree", "Max SH degree [0-3]", {"sh-degree"});
            ::args::ValueFlag<int> sh_degree_interval(training_group, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> max_cap(training_group, "max_cap", "Max Gaussians for MCMC or igs+", {"max-cap"});
            ::args::ValueFlag<float> mesh2splat_max_cap_extra_ratio(
                training_group,
                "mesh2splat_max_cap_extra_ratio",
                "When >0, set max_cap from mesh2splat initial Gaussian count: ceil(N * (1 + ratio)). Example 0.10 gives 10% headroom.",
                {"mesh2splat-max-cap-extra-ratio"});
            ::args::ValueFlag<int> mesh_surface_walk_steps(training_group, "mesh_surface_walk_steps", "Maximum edge crossings per mesh-surface projection. Default 10.", {"mesh-surface-walk-steps"});
            ::args::ValueFlag<int> mesh_surface_loss_from_iter(training_group, "mesh_surface_loss_from_iter", "Start mesh surface soft-constraint losses at iteration N. Default 0.", {"mesh-surface-loss-from-iter"});
            ::args::Flag disable_mesh_surface_hard_projection(training_group, "disable_mesh_surface_hard_projection", "Disable legacy hard post-step projection of mesh-initialized Gaussians back to the mesh surface.", {"no-mesh-surface-hard-projection"});
            ::args::ValueFlag<float> lambda_mesh_project(training_group, "lambda_mesh_project", "Weight for Gaussian mean distance-to-mesh-surface loss.", {"lambda-mesh-project"});
            ::args::ValueFlag<float> lambda_mesh_outside_barrier(training_group, "lambda_mesh_outside_barrier", "Weight for the one-sided mesh outside distance barrier.", {"lambda-mesh-outside-barrier"});
            ::args::ValueFlag<float> mesh_outside_distance_avg_max_scale_multiplier(training_group, "mesh_outside_distance_avg_max_scale_multiplier", "Outside distance threshold multiplier for mesh2splat mean initialized max scale.", {"mesh-outside-distance-avg-max-scale-multiplier"});
            ::args::ValueFlag<float> lambda_mesh_scale_min(training_group, "lambda_mesh_scale_min", "Weight for shortest Gaussian scale axis loss.", {"lambda-mesh-scale-min"});
            ::args::ValueFlag<float> lambda_mesh_scale_max(training_group, "lambda_mesh_scale_max", "Weight for largest Gaussian scale axis cap loss.", {"lambda-mesh-scale-max"});
            ::args::ValueFlag<float> mesh_scale_rho_ratio(training_group, "mesh_scale_rho_ratio", "Max scale cap as ratio of mesh scene radius. Default 0.01.", {"mesh-scale-rho-ratio"});
            ::args::ValueFlag<float> mesh_scale_rho_avg_max_scale_multiplier(training_group, "mesh_scale_rho_avg_max_scale_multiplier", "Fallback max scale cap multiplier for mesh2splat mean max initialized scale. Used only when mesh_scale_rho_ratio is 0.", {"mesh-scale-rho-avg-max-scale-multiplier"});
            ::args::ValueFlag<float> lambda_mesh_normal(training_group, "lambda_mesh_normal", "Weight for Gaussian shortest-axis normal alignment loss.", {"lambda-mesh-normal"});
            ::args::ValueFlag<float> min_opacity(training_group, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<float> steps_scaler(training_group, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> tile_mode(training_group, "tile_mode", "Tile mode for memory-efficient training: 1=1 tile, 2=2 tiles, 4=4 tiles (default: 1)", {"tile-mode"});
            ::args::Flag refine_camera_pose(training_group, "refine_camera_pose", "Enable nerfstudio-style SO3xR3 camera pose refinement during FastGS training", {"refine-camera-pose"});
            ::args::ValueFlag<std::string> pose_opt(training_group, "mode", "Camera pose refinement mode: off, SO3xR3 (direct is accepted as an alias)", {"pose-opt"});
            ::args::ValueFlag<int> pose_refine_stop_iter(training_group, "iterations", "Stop camera pose refinement at iteration N; 0 keeps it active for the whole run", {"pose-refine-stop-iter"});
            ::args::ValueFlag<float> pose_refine_lr_rot(training_group, "lr", "Pose refine rotation learning rate", {"pose-refine-lr-rot"});
            ::args::ValueFlag<float> pose_refine_lr_trans(training_group, "lr", "Pose refine translation learning rate", {"pose-refine-lr-trans"});
            ::args::ValueFlag<float> pose_refine_l2_rot(training_group, "weight", "Pose refine rotation L2 regularization weight", {"pose-refine-l2-rot"});
            ::args::ValueFlag<float> pose_refine_l2_trans(training_group, "weight", "Pose refine translation L2 regularization weight", {"pose-refine-l2-trans"});
            ::args::ValueFlag<int> pose_refine_log_every(training_group, "iterations", "Log pose refine diagnostics every N iterations; 0 disables periodic diagnostics", {"pose-refine-log-every"});
            ::args::ValueFlag<float> pose_refine_max_rot_deg(training_group, "degrees", "Clamp pose refine rotation delta norm in degrees; 0 disables clamp", {"pose-refine-max-rot-deg"});
            ::args::ValueFlag<float> pose_refine_max_trans(training_group, "distance", "Clamp pose refine translation delta norm; 0 disables clamp", {"pose-refine-max-trans"});
            ::args::ImplicitValueFlag<int> gggs(training_group, "gggs_from_iter", "Enable GGGS geometry regularization, optionally starting from iteration N (default: 7000)", {"gggs"}, 7000, 0);
            ::args::ImplicitValueFlag<int> gggs_gtnorm(training_group, "gggs_gtnorm_from_iter", "Enable GT normal loss (depth-derived normal vs mesh GT normal), optionally starting from iteration N (default: 7000)", {"gggs-gtnorm"}, 7000, 0);
            ::args::ImplicitValueFlag<int> mesh_depth(training_group, "mesh_depth_from_iter", "Enable mesh GT inverse depth loss (|1/rendered - 1/gt|), optionally starting from iteration N (default: 7000)", {"mesh-depth"}, 7000, 0);

            // =============================================================================
            // INITIALIZATION
            // =============================================================================
            ::args::Group init_sep(parser, " ");
            ::args::Group init_group(parser, "INITIALIZATION:");
            ::args::Flag random(init_group, "random", "Use random initialization instead of SfM", {"random"});
            ::args::ValueFlag<int> init_num_pts(init_group, "init_num_pts", "Number of random initialization points", {"init-num-pts"});
            ::args::ValueFlag<float> init_extent(init_group, "init_extent", "Extent of random initialization", {"init-extent"});
            ::args::ValueFlag<std::string> mesh2splat_dir(init_group, "dir", "Directory containing exactly one .obj, one .mtl, and one .png for mesh2splat", {"mesh2splat-dir"});
            ::args::ValueFlag<std::string> mesh_init_gs_scene(init_group, "path", ".ply/.obj mesh file (optional texture from OBJ/MTL map_Kd or PLY TextureFile), or legacy directory containing either obj+mtl+png or ply+png, for mesh to GS scene initialization", {"mesh-init-gs-scene"});
            ::args::ValueFlag<std::string> mesh_init_external_filled_mesh(init_group, "path", "Externally completed mesh for training mesh initialization", {"mesh-init-external-filled-mesh"});
            ::args::ValueFlag<std::string> mesh_init_external_pointcloud(init_group, "path", "External .ply point cloud covering the regions the original mesh is missing (external_pointcloud mode)", {"mesh-init-external-pointcloud"});
            ::args::ValueFlag<std::string> mesh2splat_hole_fill_mode(init_group, "mode", "Training mesh hole-fill mode: none, cgal, external, or external_pointcloud", {"mesh2splat-hole-fill-mode"});
            ::args::ValueFlag<float> mesh_sample_rate(init_group, "mesh_sample_rate", "Mesh initialization raster density ratio in (0,1], default: 1.0", {"mesh-sample-rate"});
            ::args::ValueFlag<std::string> mesh_init_color_source(init_group, "source", "Initial mesh Gaussian color source: auto, texture, rgb_views, or white", {"mesh-init-color-source"});
            ::args::Flag init_white_gs_color(init_group, "init_white_gs_color", "Initialize all Gaussian colors to white instead of texture-sampled colors", {"init-white-gs-color"});
            ::args::Flag mesh2splat_hole_fill(
                init_group,
                "mesh2splat_hole_fill",
                "Fill eligible local mesh holes with CGAL before training mesh2splat initialization",
                {"mesh2splat-hole-fill"});
            ::args::ValueFlag<int> mesh2splat_hole_fill_max_boundary_edges(
                init_group, "edges", "Maximum edge count for a fillable boundary cycle (default: 128)",
                {"mesh2splat-hole-fill-max-boundary-edges"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio(
                init_group, "ratio", "Maximum boundary perimeter / mesh bbox diagonal (default: 0.10)",
                {"mesh2splat-hole-fill-max-boundary-perimeter-bbox-ratio"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_max_boundary_area_bbox_ratio(
                init_group, "ratio", "Maximum projected boundary area / mesh bbox diagonal squared (default: 0.0025)",
                {"mesh2splat-hole-fill-max-boundary-area-bbox-ratio"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio(
                init_group, "ratio", "Maximum boundary bbox diagonal / mesh bbox diagonal (default: 0.05)",
                {"mesh2splat-hole-fill-max-boundary-bbox-diagonal-ratio"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_weld_epsilon_bbox_ratio(
                init_group, "ratio", "UV-seam weld epsilon / mesh bbox diagonal (default: 1e-6)",
                {"mesh2splat-hole-fill-weld-epsilon-bbox-ratio"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_density_control_factor(
                init_group, "factor", "CGAL patch refine density control factor (default: sqrt(2))",
                {"mesh2splat-hole-fill-density-control-factor"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_min_density_ratio(
                init_group, "ratio", "Minimum patch-to-neighbor initial Gaussian density ratio (default: 0.5)",
                {"mesh2splat-hole-fill-min-density-ratio"});
            ::args::ValueFlag<float> mesh2splat_hole_fill_max_density_ratio(
                init_group, "ratio", "Maximum patch-to-neighbor initial Gaussian density ratio (default: 2.0)",
                {"mesh2splat-hole-fill-max-density-ratio"});
            ::args::ValueFlag<float> mesh2splat_external_band_ratio(
                init_group, "ratio",
                "External mode: hole-rim tolerance band radius \\ original mesh bbox diagonal (default: 0.05)",
                {"mesh2splat-external-band-ratio"});
            ::args::ValueFlag<float> mesh2splat_external_band_tolerance_ratio(
                init_group, "ratio",
                "External mode: max band-face displacement still mapped to an original face \\ bbox diagonal (default: 0.01)",
                {"mesh2splat-external-band-tolerance-ratio"});
            ::args::ValueFlag<float> mesh2splat_pointcloud_density_ratio(
                init_group, "ratio",
                "external_pointcloud: target point Gaussian density \\ mesh Gaussian density (default: 1.0; 0 disables matching)",
                {"mesh2splat-pointcloud-density-ratio"});
            ::args::ValueFlag<float> mesh2splat_pointcloud_surface_reject_ratio(
                init_group, "ratio",
                "external_pointcloud: drop points closer to the mesh surface than this \\ mesh Gaussian spacing (default: 0.5)",
                {"mesh2splat-pointcloud-surface-reject-ratio"});
            ::args::ValueFlag<int> mesh2splat_pointcloud_mask_min_points_per_hole(
                init_group, "count",
                "external_pointcloud: min projected points inside an enclosed silhouette hole before filling it (default: 4)",
                {"mesh2splat-pointcloud-mask-min-points-per-hole"});
            ::args::ValueFlag<int> mesh2splat_pointcloud_mask_rim_close_pixels(
                init_group, "pixels",
                "external_pointcloud: closing radius applied to mesh coverage before hole labeling (default: 1)",
                {"mesh2splat-pointcloud-mask-rim-close-pixels"});
            ::args::ValueFlag<float> mesh2splat_pointcloud_mask_splat_radius_scale(
                init_group, "scale",
                "external_pointcloud: boundary-open hole fallback disc radius \\ projected point spacing (default: 1.5; 0 disables)",
                {"mesh2splat-pointcloud-mask-splat-radius-scale"});
            ::args::ValueFlag<int> mesh2splat_pointcloud_mask_close_pixels(
                init_group, "pixels",
                "external_pointcloud: fallback closing radius (default: 0 = derive from projected spacing)",
                {"mesh2splat-pointcloud-mask-close-pixels"});

            // =============================================================================
            // DATASET OPTIONS
            // =============================================================================
            ::args::Group dataset_sep(parser, " ");
            ::args::Group dataset_group(parser, "DATASET OPTIONS:");
            ::args::ValueFlag<std::string> images_folder(dataset_group, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(dataset_group, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::MapFlag<std::string, int> resize_factor(dataset_group, "resize_factor",
                                                            "Resize resolution by factor: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});
            ::args::ValueFlag<int> max_width(dataset_group, "max_width", "Max width of images in px (default: 3840)", {"max-width"});
            ::args::Flag no_cpu_cache(dataset_group, "no_cpu_cache", "Disable CPU memory caching (default: enabled)", {"no-cpu-cache"});
            ::args::Flag no_fs_cache(dataset_group, "no_fs_cache", "Disable filesystem caching (default: enabled)", {"no-fs-cache"});
            ::args::Flag undistort(dataset_group, "undistort", "Undistort images on-the-fly before training", {"undistort"});

            // =============================================================================
            // MASK OPTIONS
            // =============================================================================
            ::args::Group mask_sep(parser, " ");
            ::args::Group mask_group(parser, "MASK OPTIONS:");
            ::args::MapFlag<std::string, lfs::core::param::MaskMode> mask_mode(mask_group, "mask_mode",
                                                                               "Mask mode: none, segment, ignore, alpha_consistent, project_mesh (default: none)",
                                                                               {"mask-mode"},
                                                                               std::unordered_map<std::string, lfs::core::param::MaskMode>{
                                                                                   {"none", lfs::core::param::MaskMode::None},
                                                                                   {"segment", lfs::core::param::MaskMode::Segment},
                                                                                   {"ignore", lfs::core::param::MaskMode::Ignore},
                                                                                   {"alpha_consistent", lfs::core::param::MaskMode::AlphaConsistent},
                                                                                   {"project_mesh", lfs::core::param::MaskMode::ProjectMesh}});
            ::args::Flag invert_masks(mask_group, "invert_masks", "Invert mask values (swap object/background)", {"invert-masks"});
            ::args::Flag no_alpha_as_mask(mask_group, "no_alpha_as_mask", "Disable automatic alpha-as-mask for RGBA images", {"no-alpha-as-mask"});

            // =============================================================================
            // SPARSITY OPTIMIZATION
            // =============================================================================
            ::args::Group sparsity_sep(parser, " ");
            ::args::Group sparsity_group(parser, "SPARSITY OPTIMIZATION:");
            ::args::Flag enable_sparsity(sparsity_group, "enable_sparsity", "Enable sparsity optimization", {"enable-sparsity"});
            ::args::ValueFlag<int> sparsify_steps(sparsity_group, "sparsify_steps", "Number of steps for sparsification (default: 15000)", {"sparsify-steps"});
            ::args::ValueFlag<float> init_rho(sparsity_group, "init_rho", "Initial ADMM penalty parameter (default: 0.0005)", {"init-rho"});
            ::args::ValueFlag<float> prune_ratio(sparsity_group, "prune_ratio", "Final pruning ratio for sparsity (default: 0.6)", {"prune-ratio"});

            // =============================================================================
            // PYRAMID TRAINING
            // =============================================================================
            ::args::Group pyramid_sep(parser, " ");
            ::args::Group pyramid_group(parser, "PYRAMID TRAINING:");
            ::args::Flag pyramid_training(pyramid_group, "pyramid", "Enable pyramid (coarse-to-fine) training", {"pyramid"});
            ::args::ValueFlag<int> pyramid_levels(pyramid_group, "levels", "Number of pyramid levels including full resolution (default: 3)", {"pyramid-levels"});
            ::args::ValueFlag<int> pyramid_step_interval(pyramid_group, "steps", "Iterations per pyramid level from coarsest to finest (default: 1000)", {"pyramid-step-interval"});

            // =============================================================================
            // RENDERING OPTIONS
            // =============================================================================
            ::args::Group rendering_sep(parser, " ");
            ::args::Group rendering_group(parser, "RENDERING OPTIONS:");
            ::args::Flag enable_mip(rendering_group, "enable_mip", "Enable mip filter (anti-aliasing)", {"enable-mip"});
            ::args::Flag use_bilateral_grid(rendering_group, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag use_per_frame_affine_color(rendering_group, "per_frame_affine_color", "Enable per-frame affine color correction", {"per-frame-affine-color"});
            ::args::Flag use_ppisp(rendering_group, "ppisp", "Enable PPISP for per-camera appearance modeling", {"ppisp"});
            ::args::Flag ppisp_exposure_only(rendering_group, "ppisp_exposure_only", "Optimize only per-frame PPISP exposure", {"ppisp-exposure-only"});
            ::args::Flag ppisp_controller(rendering_group, "ppisp_controller", "Enable PPISP controller for novel views", {"ppisp-controller"});
            ::args::Flag ppisp_freeze_from_sidecar(rendering_group, "ppisp_freeze", "Freeze PPISP learning and load PPISP weights from a sidecar file", {"ppisp-freeze"});
            ::args::ValueFlag<std::string> ppisp_sidecar_path(rendering_group, "path", "Path to PPISP sidecar (.ppisp) used for frozen PPISP training", {"ppisp-sidecar"});
            ::args::Flag bg_modulation(rendering_group, "bg_modulation", "Enable sinusoidal background modulation", {"bg-modulation"});
            ::args::Flag gut(rendering_group, "gut", "Enable GUT mode", {"gut"});

            // =============================================================================
            // OUTPUT OPTIONS
            // =============================================================================
            ::args::Group output_sep(parser, " ");
            ::args::Group output_group(parser, "OUTPUT OPTIONS:");
            ::args::Flag enable_eval(output_group, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag enable_save_eval_images(output_group, "save_eval_images", "Save evaluation comparison images (GT vs rendered)", {"save-eval-images"});
            ::args::Flag save_per_frame_affine_color(output_group, "save_per_frame_affine_color", "Save final per-frame affine color parameters as JSON", {"save-per-frame-affine-color"});
            ::args::Flag disable_save_pose_refine_outputs(output_group, "no_save_pose_refine_outputs", "Do not save PoseRefiner camera pose exports under <output>/pose_refine", {"no-save-pose-refine-outputs"});
            ::args::Flag save_depth(output_group, "save_depth", "Save rendered depth and normal maps during evaluation/timelapse", {"save-depth"});
            ::args::Flag precompute_mesh_depth_normal(
                output_group,
                "precompute_mesh_depth_normal",
                "Enable per-step mesh GT depth/normal rendering during training forward passes",
                {"mesh-gt"});
            ::args::Flag precompute_pseudo_view(
                output_group,
                "precompute_pseudo_view",
                "Precompute pseudo-view RGB supervision (sphere-grid low-coverage mesh views reprojected from real images) before training. Implies --mesh-gt.",
                {"pseudo-view"});
            ::args::ValueFlag<float> pseudo_view_loss_weight(
                output_group,
                "pseudo_view_loss_weight",
                "Per-pixel photometric loss/grad multiplier for pseudo-view cameras (default: 0.1).",
                {"pseudo-view-weight"});
            ::args::ValueFlag<int> pseudo_view_min_valid_pixels(
                output_group,
                "pseudo_view_min_valid_pixels",
                "Reject pseudo-view supervision masks with fewer than N valid pixels; 0 disables the absolute threshold (default: 0).",
                {"pseudo-view-min-valid-pixels"});
            ::args::ValueFlag<int> pseudo_view_mask_erode_pixels(
                output_group,
                "pseudo_view_mask_erode_pixels",
                "Erode pseudo-view supervision masks inward by N pixels (default: 6).",
                {"pseudo-view-mask-erode-pixels"});
            ::args::ValueFlag<int> pseudo_view_mask_valid_open_pixels(
                output_group,
                "pseudo_view_mask_valid_open_pixels",
                "Remove narrow valid slivers from pseudo-view supervision masks by opening with radius N (default: 2).",
                {"pseudo-view-mask-valid-open-pixels"});
            ::args::ValueFlag<int> pseudo_view_mask_invalid_dilate_pixels(
                output_group,
                "pseudo_view_mask_invalid_dilate_pixels",
                "Dilate invalid pixels before pseudo-view supervision-mask erosion (default: 1).",
                {"pseudo-view-mask-invalid-dilate-pixels"});
            ::args::Flag disable_pseudo_view_training(
                output_group,
                "disable_pseudo_view_training",
                "Generate the pseudo-view cache but do NOT inject pseudo cameras into the training set.",
                {"no-pseudo-view-training"});
            ::args::ValueFlagList<std::string> timelapse_images(output_group, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});
            ::args::ValueFlag<int> timelapse_every(output_group, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});

            // =============================================================================
            // UI OPTIONS
            // =============================================================================
            ::args::Group ui_sep(parser, " ");
            ::args::Group ui_group(parser, "UI OPTIONS:");
            ::args::Flag headless(ui_group, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag auto_train(ui_group, "train", "Start training immediately on startup", {"train"});
#ifndef LFS_BUILD_PORTABLE
            ::args::Flag no_splash(ui_group, "no_splash", "Skip splash screen on startup", {"no-splash"});
#endif
            ::args::Flag no_interop(ui_group, "no_interop", "Disable CUDA-GL interop (use CPU fallback for display)", {"no-interop"});
            ::args::Flag debug_python(ui_group, "debug_python", "Start debugpy listener on port 5678 for plugin debugging", {"debug-python"});
            ::args::ValueFlag<int> debug_python_port(ui_group, "port", "Port for debugpy listener (default: 5678)", {"debug-python-port"});

            // =============================================================================
            // LOGGING
            // =============================================================================
            ::args::Group logging_sep(parser, " ");
            ::args::Group logging_group(parser, "LOGGING:");
            ::args::ValueFlag<std::string> log_level(logging_group, "level", "Log level: trace, debug, info, perf, warn, error, critical, off (default: info)", {"log-level"});
            ::args::Flag verbose(logging_group, "verbose", "Verbose output (equivalent to --log-level debug)", {"verbose"});
            ::args::Flag quiet(logging_group, "quiet", "Suppress non-error output (equivalent to --log-level error)", {'q', "quiet"});
            ::args::ValueFlag<std::string> log_file(logging_group, "file", "Optional log file path", {"log-file"});
            ::args::ValueFlag<std::string> log_filter(logging_group, "pattern", "Filter log messages (glob: *foo*, regex: \\\\d+)", {"log-filter"});

            // =============================================================================
            // EXTENSIONS
            // =============================================================================
            ::args::Group extensions_sep(parser, " ");
            ::args::Group extensions_group(parser, "EXTENSIONS:");
            ::args::ValueFlagList<std::string> python_scripts(extensions_group, "path", "Python script(s) for custom training callbacks", {"python-script"});

            // Parse arguments
            try {
                parser.Prog(args.front());
#if !LFS_ENABLE_CLI_HELP
                if (args.size() == 2 && is_help_argument(args[1])) {
                    return std::make_tuple(ParseResult::Help, std::function<void()>{});
                }
#endif
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Help&) {
#if LFS_ENABLE_CLI_HELP
                std::print("{}", parser.Help());
#endif
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::Completion& e) {
#if LFS_ENABLE_CLI_HELP
                std::print("{}", e.what());
#else
                (void) e;
#endif
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::ParseError& e) {
                return std::unexpected(append_cli_help(std::format("Parse error: {}", e.what()), parser));
            }

            // Initialize logger (CLI args override environment variable)
            {
                auto level = lfs::core::LogLevel::Error;
#ifdef DEBUG_BUILD
                level = lfs::core::LogLevel::Info;
#endif
                std::string log_file_path;
                std::string filter_pattern;

                // Check environment variable first
#ifdef DEBUG_BUILD
                if (const char* env_level = std::getenv("LOG_LEVEL")) {
                    level = parse_log_level(env_level);
                }
                // Verbose/quiet flags override environment variable
                if (verbose) {
                    level = lfs::core::LogLevel::Debug;
                }
                if (quiet) {
                    level = lfs::core::LogLevel::Error;
                }
                // CLI --log-level takes final precedence
                if (log_level) {
                    level = parse_log_level(::args::get(log_level));
                }
#else
                (void)verbose;
                (void)log_level;
                (void)quiet;
#endif
                if (log_file) {
                    log_file_path = ::args::get(log_file);
                }
                if (log_filter) {
                    filter_pattern = ::args::get(log_filter);
                }

                lfs::core::Logger::get().init(level, log_file_path, filter_pattern);

                LOG_DEBUG("Logger initialized with level: {}", static_cast<int>(level));
                if (!filter_pattern.empty()) {
                    LOG_DEBUG("Log filter: {}", filter_pattern);
                }
                if (!log_file_path.empty()) {
                    LOG_DEBUG("Logging to file: {}", log_file_path);
                }
            }

            // Check if explicitly displaying help
#if LFS_ENABLE_CLI_HELP
            if (help) {
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            }
#endif

            // NO ARGUMENTS = VIEWER MODE (empty)
            if (args.size() == 1) {
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Viewer mode: file or directory
            if (view_ply) {
                const auto& view_path_str = ::args::get(view_ply);
                if (!view_path_str.empty()) {
                    const std::filesystem::path view_path = lfs::core::utf8_to_path(view_path_str);

                    if (!std::filesystem::exists(view_path)) {
                        return std::unexpected(std::format("Path does not exist: {}", lfs::core::path_to_utf8(view_path)));
                    }

                    constexpr std::array<std::string_view, 12> SUPPORTED_EXTENSIONS = {
                        ".ply", ".sog", ".spz", ".resume",
                        ".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".blend"};
                    const auto is_supported = [&](const std::filesystem::path& p) {
                        auto ext = p.extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        return std::ranges::find(SUPPORTED_EXTENSIONS, ext) != SUPPORTED_EXTENSIONS.end();
                    };

                    if (std::filesystem::is_directory(view_path)) {
                        for (const auto& entry : std::filesystem::directory_iterator(view_path)) {
                            if (entry.is_regular_file() && is_supported(entry.path())) {
                                params.view_paths.push_back(entry.path());
                            }
                        }
                        std::ranges::sort(params.view_paths);

                        if (params.view_paths.empty()) {
                            return std::unexpected(std::format(
                                "No supported files found in: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        LOG_DEBUG("Found {} view files in directory", params.view_paths.size());
                    } else {
                        if (!is_supported(view_path)) {
                            return std::unexpected(std::format(
                                "Unsupported file format: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        params.view_paths.push_back(view_path);
                    }
                }

                if (gut) {
                    params.optimization.gut = true;
                }
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Import COLMAP cameras only (no images required)
            if (import_cameras) {
                const auto& import_path_str = ::args::get(import_cameras);
                if (!import_path_str.empty()) {
                    const std::filesystem::path import_path = lfs::core::utf8_to_path(import_path_str);
                    if (!std::filesystem::exists(import_path)) {
                        return std::unexpected(std::format("Path does not exist: {}", lfs::core::path_to_utf8(import_path)));
                    }
                    if (!std::filesystem::is_directory(import_path)) {
                        return std::unexpected(std::format("Expected directory for --import-cameras: {}", lfs::core::path_to_utf8(import_path)));
                    }
                    params.import_cameras_path = import_path;
                }
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Check for resume mode
            if (resume_checkpoint) {
                const auto ckpt_path_str = ::args::get(resume_checkpoint);
                if (!ckpt_path_str.empty()) {
                    const auto ckpt_path = lfs::core::utf8_to_path(ckpt_path_str);
                    if (!std::filesystem::exists(ckpt_path)) {
                        return std::unexpected(std::format("Checkpoint file does not exist: {}", ckpt_path_str));
                    }
                    params.resume_checkpoint = ckpt_path;
                }
            }

            if (init_path) {
                const auto path_str = ::args::get(init_path);
                params.init_path = path_str;

                if (!std::filesystem::exists(lfs::core::utf8_to_path(path_str))) {
                    return std::unexpected(std::format("Initialization file does not exist: {}", path_str));
                }
            }

            if (mesh2splat_dir) {
                const auto dir_path = lfs::core::utf8_to_path(::args::get(mesh2splat_dir));
                const auto assets_result = resolve_mesh2splat_assets_from_directory(dir_path);
                if (!assets_result) {
                    return std::unexpected(assets_result.error());
                }

                params.mesh2splat_mesh_path = lfs::core::path_to_utf8(assets_result->mesh_path);
                params.mesh2splat_texture_path = lfs::core::path_to_utf8(assets_result->texture_path);
            }

            if (mesh_init_gs_scene) {
                const auto mesh_init_path = lfs::core::utf8_to_path(::args::get(mesh_init_gs_scene));
                const auto assets_result = resolve_mesh_init_assets(mesh_init_path);
                if (!assets_result) {
                    return std::unexpected(assets_result.error());
                }

                params.mesh_init_mesh_path = lfs::core::path_to_utf8(assets_result->mesh_path);
                if (assets_result->texture_path.has_value()) {
                    params.mesh_init_texture_path = lfs::core::path_to_utf8(*assets_result->texture_path);
                }
                params.use_mesh_init = true;
            }

            if (mesh_init_color_source) {
                const auto parsed_source = lfs::core::param::parse_mesh_init_color_source(
                    ::args::get(mesh_init_color_source));
                if (!parsed_source) {
                    return std::unexpected(std::format("ERROR: {}", parsed_source.error()));
                }
                params.optimization.mesh_init_color_source = *parsed_source;
            }

            if (mesh_init_external_filled_mesh) {
                const auto external_path = lfs::core::utf8_to_path(::args::get(mesh_init_external_filled_mesh));
                if (!std::filesystem::exists(external_path)) {
                    return std::unexpected(std::format("External filled mesh does not exist: {}", ::args::get(mesh_init_external_filled_mesh)));
                }
                params.mesh_init_external_filled_mesh_path = lfs::core::path_to_utf8(external_path);
            }
            if (mesh_init_external_pointcloud) {
                const auto pointcloud_path = lfs::core::utf8_to_path(::args::get(mesh_init_external_pointcloud));
                if (!std::filesystem::exists(pointcloud_path)) {
                    return std::unexpected(std::format("External point cloud does not exist: {}", ::args::get(mesh_init_external_pointcloud)));
                }
                params.mesh_init_external_pointcloud_path = lfs::core::path_to_utf8(pointcloud_path);
            }

            if (init_white_gs_color) {
                params.init_white_gs_color = true;
            }

            // Training mode
            const bool has_data_path = data_path && !::args::get(data_path).empty();
            const bool has_output_path = output_path && !::args::get(output_path).empty();
            const bool has_resume = params.resume_checkpoint.has_value();
            const std::string requested_strategy = strategy ? ::args::get(strategy) : "";
            const bool use_direct_mesh2splat = requested_strategy == "mesh2splat";
            const bool has_mesh2splat_inputs = params.mesh2splat_mesh_path.has_value() || params.mesh2splat_texture_path.has_value();
            const bool has_mesh_init_inputs = params.mesh_init_mesh_path.has_value() || params.mesh_init_texture_path.has_value();

            if (!requested_strategy.empty() && VALID_STRATEGIES.find(requested_strategy) == VALID_STRATEGIES.end()) {
                return std::unexpected(std::format(
                    "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, adc, igs+, mesh2splat",
                    requested_strategy));
            }

            if (params.use_mesh_init && params.init_path.has_value()) {
                return std::unexpected("ERROR: Mesh init now uses --mesh-init-gs-scene. Do not combine with --init");
            }

            if (has_mesh2splat_inputs && !use_direct_mesh2splat && !config_file) {
                return std::unexpected("ERROR: --mesh2splat-dir requires --strategy mesh2splat");
            }

            if (use_direct_mesh2splat && !has_mesh2splat_inputs) {
                return std::unexpected("ERROR: mesh2splat strategy requires --mesh2splat-dir with obj+mtl+png");
            }

            if (use_direct_mesh2splat && params.init_path.has_value()) {
                return std::unexpected("ERROR: mesh2splat strategy uses --mesh2splat-dir, not --init");
            }

            if (params.use_mesh_init && !has_mesh_init_inputs) {
                return std::unexpected("ERROR: --mesh-init-gs-scene requires a .ply/.obj mesh file or a directory with obj+mtl+png or ply+png");
            }

            // If headless mode, require data path or resume checkpoint
            if (headless && !use_direct_mesh2splat && !has_data_path && !has_resume) {
                return std::unexpected(append_cli_help("ERROR: Headless mode requires --data-path or --resume", parser));
            }

            // Training/resume mode requires both data-path and output-path
            // Exception: resume mode can work without explicit paths (extracted from checkpoint)
            if (use_direct_mesh2splat) {
                if (has_data_path) {
                    params.dataset.data_path = lfs::core::utf8_to_path(::args::get(data_path));
                }
                if (has_output_path) {
                    params.dataset.output_path = lfs::core::utf8_to_path(::args::get(output_path));
                }
            } else if (has_data_path && has_output_path) {
                params.dataset.data_path = lfs::core::utf8_to_path(::args::get(data_path));
                params.dataset.output_path = lfs::core::utf8_to_path(::args::get(output_path));

                // Create output directory
                std::error_code ec;
                std::filesystem::create_directories(params.dataset.output_path, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        lfs::core::path_to_utf8(params.dataset.output_path), ec.message()));
                }
            } else if (has_data_path != has_output_path && !has_resume) {
                // Only require both if not in resume mode
                return std::unexpected(append_cli_help("ERROR: Training mode requires both --data-path and --output-path", parser));
            } else if (has_resume) {
                // Resume mode: paths are optional (will be read from checkpoint)
                if (has_data_path) {
                    params.dataset.data_path = lfs::core::utf8_to_path(::args::get(data_path));
                }
                if (has_output_path) {
                    params.dataset.output_path = lfs::core::utf8_to_path(::args::get(output_path));

                    // Create output directory if provided
                    std::error_code ec;
                    std::filesystem::create_directories(params.dataset.output_path, ec);
                    if (ec) {
                        return std::unexpected(std::format(
                            "Failed to create output directory '{}': {}",
                            lfs::core::path_to_utf8(params.dataset.output_path), ec.message()));
                    }
                }
            }

            if (strategy) {
                const auto strat = ::args::get(strategy);
                // Unlike other parameters that will be set later as overrides,
                // strategy must be set immediately to ensure correct JSON loading
                // in `read_optim_params_from_json()`
                params.optimization.strategy = strat;
            }

            if (config_file) {
                params.optimization.config_file = ::args::get(config_file);
                if (!strategy) {
                    params.optimization.strategy = ""; // Clear strategy to avoid using default strategy for evaluation of conflict
                }
            }

            if (max_width) {
                int width = ::args::get(max_width);
                if (width <= 0) {
                    return std::unexpected("ERROR: --max-width must be greather than 0");
                }
                if (width > 4096) {
                    return std::unexpected("ERROR: --max-width cannot be higher than 4096");
                }
            }

            if (tile_mode) {
                int mode = ::args::get(tile_mode);
                if (mode != 1 && mode != 2 && mode != 4) {
                    return std::unexpected("ERROR: --tile-mode must be 1 (1 tile), 2 (2 tiles), or 4 (4 tiles)");
                }
            }
            if (mesh_surface_loss_from_iter && ::args::get(mesh_surface_loss_from_iter) < 0) {
                return std::unexpected("ERROR: --mesh-surface-loss-from-iter must be non-negative");
            }
            auto validate_nonnegative_finite = [](float v) {
                return v >= 0.0f && std::isfinite(v);
            };
            if (lambda_mesh_project && !validate_nonnegative_finite(::args::get(lambda_mesh_project))) {
                return std::unexpected("ERROR: --lambda-mesh-project must be finite and non-negative");
            }
            if (lambda_mesh_outside_barrier && !validate_nonnegative_finite(::args::get(lambda_mesh_outside_barrier))) {
                return std::unexpected("ERROR: --lambda-mesh-outside-barrier must be finite and non-negative");
            }
            if (mesh_outside_distance_avg_max_scale_multiplier &&
                !validate_nonnegative_finite(::args::get(mesh_outside_distance_avg_max_scale_multiplier))) {
                return std::unexpected("ERROR: --mesh-outside-distance-avg-max-scale-multiplier must be finite and non-negative");
            }
            if (lambda_mesh_scale_min && !validate_nonnegative_finite(::args::get(lambda_mesh_scale_min))) {
                return std::unexpected("ERROR: --lambda-mesh-scale-min must be finite and non-negative");
            }
            if (lambda_mesh_scale_max && !validate_nonnegative_finite(::args::get(lambda_mesh_scale_max))) {
                return std::unexpected("ERROR: --lambda-mesh-scale-max must be finite and non-negative");
            }
            if (lambda_mesh_normal && !validate_nonnegative_finite(::args::get(lambda_mesh_normal))) {
                return std::unexpected("ERROR: --lambda-mesh-normal must be finite and non-negative");
            }
            if (mesh_scale_rho_ratio) {
                const float ratio = ::args::get(mesh_scale_rho_ratio);
                if (ratio < 0.0f || !std::isfinite(ratio)) {
                    return std::unexpected("ERROR: --mesh-scale-rho-ratio must be finite and non-negative");
                }
            }
            if (mesh_scale_rho_avg_max_scale_multiplier && !validate_nonnegative_finite(::args::get(mesh_scale_rho_avg_max_scale_multiplier))) {
                return std::unexpected("ERROR: --mesh-scale-rho-avg-max-scale-multiplier must be finite and non-negative");
            }
            if (mesh2splat_max_cap_extra_ratio &&
                !validate_nonnegative_finite(::args::get(mesh2splat_max_cap_extra_ratio))) {
                return std::unexpected("ERROR: --mesh2splat-max-cap-extra-ratio must be finite and non-negative");
            }
            if (pseudo_view_mask_erode_pixels && ::args::get(pseudo_view_mask_erode_pixels) < 0) {
                return std::unexpected("ERROR: --pseudo-view-mask-erode-pixels must be non-negative");
            }
            if (pseudo_view_min_valid_pixels && ::args::get(pseudo_view_min_valid_pixels) < 0) {
                return std::unexpected("ERROR: --pseudo-view-min-valid-pixels must be non-negative");
            }
            if (pseudo_view_mask_valid_open_pixels && ::args::get(pseudo_view_mask_valid_open_pixels) < 0) {
                return std::unexpected("ERROR: --pseudo-view-mask-valid-open-pixels must be non-negative");
            }
            if (pseudo_view_mask_invalid_dilate_pixels && ::args::get(pseudo_view_mask_invalid_dilate_pixels) < 0) {
                return std::unexpected("ERROR: --pseudo-view-mask-invalid-dilate-pixels must be non-negative");
            }
            std::optional<std::string> pose_refine_mode_cli;
            std::optional<bool> refine_camera_pose_cli;
            if (refine_camera_pose) {
                refine_camera_pose_cli = true;
                pose_refine_mode_cli = "SO3xR3";
            }
            if (pose_opt) {
                auto normalized_pose_opt = normalize_pose_refine_mode(::args::get(pose_opt));
                if (!normalized_pose_opt) {
                    return std::unexpected(normalized_pose_opt.error());
                }
                if (*normalized_pose_opt == "off") {
                    refine_camera_pose_cli = false;
                } else {
                    refine_camera_pose_cli = true;
                    pose_refine_mode_cli = *normalized_pose_opt;
                }
            }
            if (pose_refine_log_every && ::args::get(pose_refine_log_every) < 0) {
                return std::unexpected("ERROR: --pose-refine-log-every must be non-negative");
            }
            if (pose_refine_stop_iter && ::args::get(pose_refine_stop_iter) < 0) {
                return std::unexpected("ERROR: --pose-refine-stop-iter must be non-negative");
            }
            if (pose_refine_lr_rot && !validate_nonnegative_finite(::args::get(pose_refine_lr_rot))) {
                return std::unexpected("ERROR: --pose-refine-lr-rot must be finite and non-negative");
            }
            if (pose_refine_lr_trans && !validate_nonnegative_finite(::args::get(pose_refine_lr_trans))) {
                return std::unexpected("ERROR: --pose-refine-lr-trans must be finite and non-negative");
            }
            if (pose_refine_l2_rot && !validate_nonnegative_finite(::args::get(pose_refine_l2_rot))) {
                return std::unexpected("ERROR: --pose-refine-l2-rot must be finite and non-negative");
            }
            if (pose_refine_l2_trans && !validate_nonnegative_finite(::args::get(pose_refine_l2_trans))) {
                return std::unexpected("ERROR: --pose-refine-l2-trans must be finite and non-negative");
            }
            if (pose_refine_max_rot_deg && !validate_nonnegative_finite(::args::get(pose_refine_max_rot_deg))) {
                return std::unexpected("ERROR: --pose-refine-max-rot-deg must be finite and non-negative");
            }
            if (pose_refine_max_trans && !validate_nonnegative_finite(::args::get(pose_refine_max_trans))) {
                return std::unexpected("ERROR: --pose-refine-max-trans must be finite and non-negative");
            }

            // Validate sh_degree (0-3)
            if (sh_degree) {
                int degree = ::args::get(sh_degree);
                if (degree < 0 || degree > 3) {
                    return std::unexpected("ERROR: --sh-degree must be 0, 1, 2, or 3");
                }
            }

            // Validate min_opacity (0.0-1.0)
            if (min_opacity) {
                float opacity = ::args::get(min_opacity);
                if (opacity < 0.0f || opacity > 1.0f) {
                    return std::unexpected("ERROR: --min-opacity must be between 0.0 and 1.0");
                }
            }

            // Validate init_num_pts (> 0)
            if (init_num_pts) {
                int pts = ::args::get(init_num_pts);
                if (pts <= 0) {
                    return std::unexpected("ERROR: --init-num-pts must be greater than 0");
                }
            }

            // Validate prune_ratio (0.0-1.0)
            if (prune_ratio) {
                float ratio = ::args::get(prune_ratio);
                if (ratio < 0.0f || ratio > 1.0f) {
                    return std::unexpected("ERROR: --prune-ratio must be between 0.0 and 1.0");
                }
            }

            if (mesh_sample_rate) {
                float ratio = ::args::get(mesh_sample_rate);
                if (ratio <= 0.0f || ratio > 1.0f) {
                    return std::unexpected("ERROR: --mesh-sample-rate must be in (0.0, 1.0]");
                }
            }

            if (mesh2splat_hole_fill_mode) {
                const auto parsed_mode = lfs::core::param::parse_mesh2splat_hole_fill_mode(
                    ::args::get(mesh2splat_hole_fill_mode));
                if (!parsed_mode) {
                    return std::unexpected("ERROR: " + parsed_mode.error());
                }
                if (mesh2splat_hole_fill &&
                    *parsed_mode != lfs::core::param::Mesh2SplatHoleFillMode::Cgal) {
                    return std::unexpected(
                        "ERROR: --mesh2splat-hole-fill conflicts with --mesh2splat-hole-fill-mode unless mode=cgal");
                }
            }

            // Validate pyramid parameters
            if (pyramid_levels) {
                int levels = ::args::get(pyramid_levels);
                if (levels < 2 || levels > 6) {
                    return std::unexpected("ERROR: --pyramid-levels must be between 2 and 6");
                }
            }
            if (pyramid_step_interval) {
                int interval = ::args::get(pyramid_step_interval);
                if (interval <= 0) {
                    return std::unexpected("ERROR: --pyramid-step-interval must be greater than 0");
                }
            }

            // Create lambda to apply command line overrides after JSON loading
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // default 1
                                        max_width_val = max_width ? std::optional<int>(::args::get(max_width)) : std::optional<int>(3840),          // default 3840
                                        no_cpu_cache_flag = static_cast<bool>(no_cpu_cache),
                                        no_fs_cache_flag = static_cast<bool>(no_fs_cache),
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        mesh_surface_walk_steps_val = mesh_surface_walk_steps ? std::optional<int>(::args::get(mesh_surface_walk_steps)) : std::optional<int>(),
                                        mesh_surface_loss_from_iter_val = mesh_surface_loss_from_iter ? std::optional<int>(::args::get(mesh_surface_loss_from_iter)) : std::optional<int>(),
                                        disable_mesh_surface_hard_projection_flag = bool(disable_mesh_surface_hard_projection),
                                        lambda_mesh_project_val = lambda_mesh_project ? std::optional<float>(::args::get(lambda_mesh_project)) : std::optional<float>(),
                                        lambda_mesh_outside_barrier_val = lambda_mesh_outside_barrier ? std::optional<float>(::args::get(lambda_mesh_outside_barrier)) : std::optional<float>(),
                                        mesh_outside_distance_avg_max_scale_multiplier_val = mesh_outside_distance_avg_max_scale_multiplier ? std::optional<float>(::args::get(mesh_outside_distance_avg_max_scale_multiplier)) : std::optional<float>(),
                                        lambda_mesh_scale_min_val = lambda_mesh_scale_min ? std::optional<float>(::args::get(lambda_mesh_scale_min)) : std::optional<float>(),
                                        lambda_mesh_scale_max_val = lambda_mesh_scale_max ? std::optional<float>(::args::get(lambda_mesh_scale_max)) : std::optional<float>(),
                                        mesh2splat_max_cap_extra_ratio_val = mesh2splat_max_cap_extra_ratio
                                                                                  ? std::optional<float>(::args::get(mesh2splat_max_cap_extra_ratio))
                                                                                  : std::optional<float>(),
                                        mesh2splat_hole_fill_flag = bool(mesh2splat_hole_fill),
                                        mesh2splat_hole_fill_mode_val = mesh2splat_hole_fill_mode
                                                                            ? std::optional<std::string>(::args::get(mesh2splat_hole_fill_mode))
                                                                            : std::optional<std::string>(),
                                        mesh2splat_hole_fill_max_boundary_edges_val = mesh2splat_hole_fill_max_boundary_edges
                                                                                          ? std::optional<int>(::args::get(mesh2splat_hole_fill_max_boundary_edges))
                                                                                          : std::optional<int>(),
                                        mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio_val = mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio
                                                                                                        ? std::optional<float>(::args::get(mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio))
                                                                                                        : std::optional<float>(),
                                        mesh2splat_hole_fill_max_boundary_area_bbox_ratio_val = mesh2splat_hole_fill_max_boundary_area_bbox_ratio
                                                                                                   ? std::optional<float>(::args::get(mesh2splat_hole_fill_max_boundary_area_bbox_ratio))
                                                                                                   : std::optional<float>(),
                                        mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio_val = mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio
                                                                                                      ? std::optional<float>(::args::get(mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio))
                                                                                                      : std::optional<float>(),
                                        mesh2splat_hole_fill_weld_epsilon_bbox_ratio_val = mesh2splat_hole_fill_weld_epsilon_bbox_ratio
                                                                                              ? std::optional<float>(::args::get(mesh2splat_hole_fill_weld_epsilon_bbox_ratio))
                                                                                              : std::optional<float>(),
                                        mesh2splat_hole_fill_density_control_factor_val = mesh2splat_hole_fill_density_control_factor
                                                                                             ? std::optional<float>(::args::get(mesh2splat_hole_fill_density_control_factor))
                                                                                             : std::optional<float>(),
                                        mesh2splat_hole_fill_min_density_ratio_val = mesh2splat_hole_fill_min_density_ratio
                                                                                        ? std::optional<float>(::args::get(mesh2splat_hole_fill_min_density_ratio))
                                                                                        : std::optional<float>(),
                                        mesh2splat_hole_fill_max_density_ratio_val = mesh2splat_hole_fill_max_density_ratio
                                                                                        ? std::optional<float>(::args::get(mesh2splat_hole_fill_max_density_ratio))
                                                                                        : std::optional<float>(),
                                        mesh2splat_external_band_ratio_val = mesh2splat_external_band_ratio
                                                                                 ? std::optional<float>(::args::get(mesh2splat_external_band_ratio))
                                                                                 : std::optional<float>(),
                                        mesh2splat_external_band_tolerance_ratio_val = mesh2splat_external_band_tolerance_ratio
                                                                                           ? std::optional<float>(::args::get(mesh2splat_external_band_tolerance_ratio))
                                                                                           : std::optional<float>(),
                                        mesh2splat_pointcloud_density_ratio_val = mesh2splat_pointcloud_density_ratio
                                                                                     ? std::optional<float>(::args::get(mesh2splat_pointcloud_density_ratio))
                                                                                     : std::optional<float>(),
                                        mesh2splat_pointcloud_surface_reject_ratio_val = mesh2splat_pointcloud_surface_reject_ratio
                                                                                            ? std::optional<float>(::args::get(mesh2splat_pointcloud_surface_reject_ratio))
                                                                                            : std::optional<float>(),
                                        mesh2splat_pointcloud_mask_min_points_per_hole_val = mesh2splat_pointcloud_mask_min_points_per_hole
                                                                                                ? std::optional<int>(::args::get(mesh2splat_pointcloud_mask_min_points_per_hole))
                                                                                                : std::optional<int>(),
                                        mesh2splat_pointcloud_mask_rim_close_pixels_val = mesh2splat_pointcloud_mask_rim_close_pixels
                                                                                             ? std::optional<int>(::args::get(mesh2splat_pointcloud_mask_rim_close_pixels))
                                                                                             : std::optional<int>(),
                                        mesh2splat_pointcloud_mask_splat_radius_scale_val = mesh2splat_pointcloud_mask_splat_radius_scale
                                                                                               ? std::optional<float>(::args::get(mesh2splat_pointcloud_mask_splat_radius_scale))
                                                                                               : std::optional<float>(),
                                        mesh2splat_pointcloud_mask_close_pixels_val = mesh2splat_pointcloud_mask_close_pixels
                                                                                         ? std::optional<int>(::args::get(mesh2splat_pointcloud_mask_close_pixels))
                                                                                         : std::optional<int>(),
                                        mesh_scale_rho_ratio_val = mesh_scale_rho_ratio ? std::optional<float>(::args::get(mesh_scale_rho_ratio)) : std::optional<float>(),
                                        mesh_scale_rho_avg_max_scale_multiplier_val = mesh_scale_rho_avg_max_scale_multiplier ? std::optional<float>(::args::get(mesh_scale_rho_avg_max_scale_multiplier)) : std::optional<float>(),
                                        lambda_mesh_normal_val = lambda_mesh_normal ? std::optional<float>(::args::get(lambda_mesh_normal)) : std::optional<float>(),
                                        config_file_val = config_file ? std::optional<std::string>(::args::get(config_file)) : std::optional<std::string>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        init_num_pts_val = init_num_pts ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = init_extent ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        mesh_sample_rate_val = mesh_sample_rate ? std::optional<float>(::args::get(mesh_sample_rate)) : std::optional<float>(),
                                        mesh_init_color_source_val = mesh_init_color_source
                                                                         ? std::optional<lfs::core::param::MeshInitColorSource>(params.optimization.mesh_init_color_source)
                                                                         : std::optional<lfs::core::param::MeshInitColorSource>(),
                                        mesh_init_external_filled_mesh_val = mesh_init_external_filled_mesh ? std::optional<std::string>(::args::get(mesh_init_external_filled_mesh)) : std::optional<std::string>(),
                                        mesh_init_external_pointcloud_val = mesh_init_external_pointcloud ? std::optional<std::string>(::args::get(mesh_init_external_pointcloud)) : std::optional<std::string>(),
                                        strategy_val = strategy ? std::optional<std::string>(::args::get(strategy)) : std::optional<std::string>(),
                                        timelapse_images_val = timelapse_images ? std::optional<std::vector<std::string>>(::args::get(timelapse_images)) : std::optional<std::vector<std::string>>(),
                                        timelapse_every_val = timelapse_every ? std::optional<int>(::args::get(timelapse_every)) : std::optional<int>(),
                                        tile_mode_val = tile_mode ? std::optional<int>(::args::get(tile_mode)) : std::optional<int>(),
                                        refine_camera_pose_val = refine_camera_pose_cli,
                                        pose_refine_mode_val = pose_refine_mode_cli,
                                        pose_refine_stop_iter_val = pose_refine_stop_iter ? std::optional<int>(::args::get(pose_refine_stop_iter)) : std::optional<int>(),
                                        pose_refine_lr_rot_val = pose_refine_lr_rot ? std::optional<float>(::args::get(pose_refine_lr_rot)) : std::optional<float>(),
                                        pose_refine_lr_trans_val = pose_refine_lr_trans ? std::optional<float>(::args::get(pose_refine_lr_trans)) : std::optional<float>(),
                                        pose_refine_l2_rot_val = pose_refine_l2_rot ? std::optional<float>(::args::get(pose_refine_l2_rot)) : std::optional<float>(),
                                        pose_refine_l2_trans_val = pose_refine_l2_trans ? std::optional<float>(::args::get(pose_refine_l2_trans)) : std::optional<float>(),
                                        pose_refine_log_every_val = pose_refine_log_every ? std::optional<int>(::args::get(pose_refine_log_every)) : std::optional<int>(),
                                        pose_refine_max_rot_deg_val = pose_refine_max_rot_deg ? std::optional<float>(::args::get(pose_refine_max_rot_deg)) : std::optional<float>(),
                                        pose_refine_max_trans_val = pose_refine_max_trans ? std::optional<float>(::args::get(pose_refine_max_trans)) : std::optional<float>(),
                                        // Sparsity parameters
                                        sparsify_steps_val = sparsify_steps ? std::optional<int>(::args::get(sparsify_steps)) : std::optional<int>(),
                                        init_rho_val = init_rho ? std::optional<float>(::args::get(init_rho)) : std::optional<float>(),
                                        prune_ratio_val = prune_ratio ? std::optional<float>(::args::get(prune_ratio)) : std::optional<float>(),
                                        // Mask parameters
                                        mask_mode_val = mask_mode ? std::optional<lfs::core::param::MaskMode>(::args::get(mask_mode)) : std::optional<lfs::core::param::MaskMode>(),
                                        // Python scripts
                                        python_scripts_val = python_scripts ? std::optional<std::vector<std::string>>(::args::get(python_scripts)) : std::optional<std::vector<std::string>>(),
                                        // Pyramid training parameters
                                        pyramid_levels_val = pyramid_levels ? std::optional<int>(::args::get(pyramid_levels)) : std::optional<int>(),
                                        pyramid_step_interval_val = pyramid_step_interval ? std::optional<int>(::args::get(pyramid_step_interval)) : std::optional<int>(),
                                        pyramid_training_flag = bool(pyramid_training),
                                        // Capture flag states
                                        enable_mip_flag = bool(enable_mip),
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        use_per_frame_affine_color_flag = bool(use_per_frame_affine_color),
                                        use_ppisp_flag = bool(use_ppisp),
                                        ppisp_exposure_only_flag = bool(ppisp_exposure_only),
                                        ppisp_controller_flag = bool(ppisp_controller),
                                        ppisp_freeze_from_sidecar_flag = bool(ppisp_freeze_from_sidecar),
                                        ppisp_sidecar_path_val = ppisp_sidecar_path ? std::optional<std::string>(::args::get(ppisp_sidecar_path)) : std::optional<std::string>(),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        auto_train_flag = bool(auto_train),
#ifdef LFS_BUILD_PORTABLE
                                        no_splash_flag = false,
#else
                                        no_splash_flag = bool(no_splash),
#endif
                                        no_interop_flag = bool(no_interop),
                                        debug_python_flag = bool(debug_python),
                                        debug_python_port_val = debug_python_port ? std::optional<int>(::args::get(debug_python_port)) : std::optional<int>(),
                                        verbose_flag = bool(verbose),
                                        quiet_flag = bool(quiet),
                                        log_level_val = log_level ? std::optional<std::string>(::args::get(log_level)) : std::optional<std::string>(),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        save_per_frame_affine_color_flag = bool(save_per_frame_affine_color),
                                        disable_save_pose_refine_outputs_flag = bool(disable_save_pose_refine_outputs),
                                        save_depth_flag = bool(save_depth),
                                        precompute_mesh_depth_normal_flag = bool(precompute_mesh_depth_normal),
                                        precompute_pseudo_view_flag = bool(precompute_pseudo_view),
                                        pseudo_view_loss_weight_val = pseudo_view_loss_weight ? std::optional<float>(::args::get(pseudo_view_loss_weight)) : std::optional<float>(),
                                        pseudo_view_min_valid_pixels_val = pseudo_view_min_valid_pixels ? std::optional<int>(::args::get(pseudo_view_min_valid_pixels)) : std::optional<int>(),
                                        pseudo_view_mask_erode_pixels_val = pseudo_view_mask_erode_pixels ? std::optional<int>(::args::get(pseudo_view_mask_erode_pixels)) : std::optional<int>(),
                                        pseudo_view_mask_valid_open_pixels_val = pseudo_view_mask_valid_open_pixels ? std::optional<int>(::args::get(pseudo_view_mask_valid_open_pixels)) : std::optional<int>(),
                                        pseudo_view_mask_invalid_dilate_pixels_val = pseudo_view_mask_invalid_dilate_pixels ? std::optional<int>(::args::get(pseudo_view_mask_invalid_dilate_pixels)) : std::optional<int>(),
                                        disable_pseudo_view_training_flag = bool(disable_pseudo_view_training),
                                        gggs_val = gggs ? std::optional<int>(::args::get(gggs)) : std::optional<int>(),
                                        gggs_gtnorm_val = gggs_gtnorm ? std::optional<int>(::args::get(gggs_gtnorm)) : std::optional<int>(),
                                        mesh_depth_val = mesh_depth ? std::optional<int>(::args::get(mesh_depth)) : std::optional<int>(),
                                        bg_modulation_flag = bool(bg_modulation),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut),
                                        undistort_flag = bool(undistort),
                                        enable_sparsity_flag = bool(enable_sparsity),
                                        invert_masks_flag = bool(invert_masks),
                                        no_alpha_as_mask_flag = bool(no_alpha_as_mask)]() {
                auto& opt = params.optimization;
                auto& ds = params.dataset;

                // Simple lambdas to apply if flag/value exists
                auto setVal = [](const auto& flag, auto& target) {
                    if (flag)
                        target = *flag;
                };

                auto setFlag = [](bool flag, auto& target) {
                    if (flag)
                        target = true;
                };

                // Apply all overrides
                setVal(iterations_val, opt.iterations);
                setVal(resize_factor_val, ds.resize_factor);
                setVal(max_width_val, ds.max_width);
                if (no_cpu_cache_flag)
                    ds.loading_params.use_cpu_memory = false;
                if (no_fs_cache_flag)
                    ds.loading_params.use_fs_cache = false;
                setVal(max_cap_val, opt.max_cap);
                setVal(mesh_surface_walk_steps_val, opt.mesh_surface_walk_steps);
                setVal(mesh_surface_loss_from_iter_val, opt.mesh_surface_loss_from_iter);
                if (disable_mesh_surface_hard_projection_flag) {
                    opt.mesh_surface_hard_projection_enabled = false;
                }
                setVal(lambda_mesh_project_val, opt.lambda_mesh_project);
                setVal(lambda_mesh_outside_barrier_val, opt.lambda_mesh_outside_barrier);
                setVal(mesh_outside_distance_avg_max_scale_multiplier_val, opt.mesh_outside_distance_avg_max_scale_multiplier);
                setVal(lambda_mesh_scale_min_val, opt.lambda_mesh_scale_min);
                setVal(lambda_mesh_scale_max_val, opt.lambda_mesh_scale_max);
                setVal(mesh2splat_max_cap_extra_ratio_val, opt.mesh2splat_max_cap_extra_ratio);
                setFlag(mesh2splat_hole_fill_flag, opt.mesh2splat_hole_fill_enabled);
                if (mesh2splat_hole_fill_mode_val) {
                    const auto parsed_mode = lfs::core::param::parse_mesh2splat_hole_fill_mode(
                        *mesh2splat_hole_fill_mode_val);
                    if (parsed_mode) {
                        opt.mesh2splat_hole_fill_mode = *parsed_mode;
                        opt.mesh2splat_hole_fill_enabled = *parsed_mode == lfs::core::param::Mesh2SplatHoleFillMode::Cgal;
                    }
                }
                setVal(mesh2splat_hole_fill_max_boundary_edges_val, opt.mesh2splat_hole_fill_max_boundary_edges);
                setVal(mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio_val, opt.mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio);
                setVal(mesh2splat_hole_fill_max_boundary_area_bbox_ratio_val, opt.mesh2splat_hole_fill_max_boundary_area_bbox_ratio);
                setVal(mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio_val, opt.mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio);
                setVal(mesh2splat_hole_fill_weld_epsilon_bbox_ratio_val, opt.mesh2splat_hole_fill_weld_epsilon_bbox_ratio);
                setVal(mesh2splat_hole_fill_density_control_factor_val, opt.mesh2splat_hole_fill_density_control_factor);
                setVal(mesh2splat_hole_fill_min_density_ratio_val, opt.mesh2splat_hole_fill_min_density_ratio);
                setVal(mesh2splat_hole_fill_max_density_ratio_val, opt.mesh2splat_hole_fill_max_density_ratio);
                setVal(mesh2splat_external_band_ratio_val, opt.mesh2splat_external_band_ratio);
                setVal(mesh2splat_external_band_tolerance_ratio_val, opt.mesh2splat_external_band_tolerance_ratio);
                setVal(mesh2splat_pointcloud_density_ratio_val, opt.mesh2splat_pointcloud_density_ratio);
                setVal(mesh2splat_pointcloud_surface_reject_ratio_val, opt.mesh2splat_pointcloud_surface_reject_ratio);
                setVal(mesh2splat_pointcloud_mask_min_points_per_hole_val, opt.mesh2splat_pointcloud_mask_min_points_per_hole);
                setVal(mesh2splat_pointcloud_mask_rim_close_pixels_val, opt.mesh2splat_pointcloud_mask_rim_close_pixels);
                setVal(mesh2splat_pointcloud_mask_splat_radius_scale_val, opt.mesh2splat_pointcloud_mask_splat_radius_scale);
                setVal(mesh2splat_pointcloud_mask_close_pixels_val, opt.mesh2splat_pointcloud_mask_close_pixels);
                setVal(mesh_scale_rho_ratio_val, opt.mesh_scale_rho_ratio);
                setVal(mesh_scale_rho_avg_max_scale_multiplier_val, opt.mesh_scale_rho_avg_max_scale_multiplier);
                setVal(lambda_mesh_normal_val, opt.lambda_mesh_normal);
                setVal(images_folder_val, ds.images);
                setVal(test_every_val, ds.test_every);
                setVal(steps_scaler_val, opt.steps_scaler);
                setVal(sh_degree_interval_val, opt.sh_degree_interval);
                setVal(sh_degree_val, opt.sh_degree);
                setVal(min_opacity_val, opt.min_opacity);
                setVal(init_num_pts_val, opt.init_num_pts);
                setVal(init_extent_val, opt.init_extent);
                setVal(mesh_sample_rate_val, params.mesh_init_sampling_rate);
                setVal(mesh_init_color_source_val, opt.mesh_init_color_source);
                if (mesh_init_external_filled_mesh_val)
                    params.mesh_init_external_filled_mesh_path = lfs::core::path_to_utf8(lfs::core::utf8_to_path(*mesh_init_external_filled_mesh_val));
                if (mesh_init_external_pointcloud_val)
                    params.mesh_init_external_pointcloud_path = lfs::core::path_to_utf8(lfs::core::utf8_to_path(*mesh_init_external_pointcloud_val));
                setVal(strategy_val, opt.strategy);
                setVal(timelapse_images_val, ds.timelapse_images);
                setVal(timelapse_every_val, ds.timelapse_every);
                setVal(tile_mode_val, opt.tile_mode);
                if (refine_camera_pose_val) {
                    opt.refine_camera_pose = *refine_camera_pose_val;
                }
                if (pose_refine_mode_val) {
                    opt.pose_refine_mode = *pose_refine_mode_val;
                }
                setVal(pose_refine_stop_iter_val, opt.pose_refine_stop_iter);
                setVal(pose_refine_lr_rot_val, opt.pose_refine_lr_rot);
                setVal(pose_refine_lr_trans_val, opt.pose_refine_lr_trans);
                setVal(pose_refine_l2_rot_val, opt.pose_refine_l2_rot);
                setVal(pose_refine_l2_trans_val, opt.pose_refine_l2_trans);
                setVal(pose_refine_log_every_val, opt.pose_refine_log_every);
                setVal(pose_refine_max_rot_deg_val, opt.pose_refine_max_rot_deg);
                setVal(pose_refine_max_trans_val, opt.pose_refine_max_trans);

                // Sparsity parameters
                setVal(sparsify_steps_val, opt.sparsify_steps);
                setVal(init_rho_val, opt.init_rho);
                setVal(prune_ratio_val, opt.prune_ratio);

                setFlag(enable_mip_flag, opt.mip_filter);
                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);
                setFlag(use_per_frame_affine_color_flag, opt.use_per_frame_affine_color);
                setFlag(use_ppisp_flag, opt.use_ppisp);
                setFlag(ppisp_exposure_only_flag, opt.ppisp_exposure_only);
                setFlag(ppisp_controller_flag, opt.ppisp_use_controller);
                setFlag(ppisp_freeze_from_sidecar_flag, opt.ppisp_freeze_from_sidecar);
                if (ppisp_sidecar_path_val) {
                    opt.ppisp_sidecar_path = lfs::core::utf8_to_path(*ppisp_sidecar_path_val);
                }
                if (opt.ppisp_use_controller)
                    opt.use_ppisp = true;
                if (opt.ppisp_freeze_from_sidecar)
                    opt.use_ppisp = true;
                if (opt.ppisp_exposure_only)
                    opt.use_ppisp = true;
                setFlag(enable_eval_flag, opt.enable_eval);
                setFlag(headless_flag, opt.headless);
                setFlag(auto_train_flag, opt.auto_train);
                setFlag(no_splash_flag, opt.no_splash);
                setFlag(no_interop_flag, opt.no_interop);
                setFlag(debug_python_flag, opt.debug_python);
                setVal(debug_python_port_val, opt.debug_python_port);
                if (verbose_flag) {
                    opt.log_level = "debug";
                }
                if (quiet_flag) {
                    opt.log_level = "error";
                }
                setVal(log_level_val, opt.log_level);
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images);
                setFlag(save_per_frame_affine_color_flag, opt.save_per_frame_affine_color);
                if (disable_save_pose_refine_outputs_flag) {
                    opt.save_pose_refine_outputs = false;
                }
                setFlag(save_depth_flag, opt.save_depth);
                setFlag(precompute_mesh_depth_normal_flag, opt.precompute_mesh_depth_normal);
                if (precompute_pseudo_view_flag) {
                    opt.precompute_pseudo_view = true;
                    opt.precompute_mesh_depth_normal = true;
                }
                if (pseudo_view_loss_weight_val) {
                    opt.pseudo_view_loss_weight = *pseudo_view_loss_weight_val;
                }
                setVal(pseudo_view_min_valid_pixels_val, opt.pseudo_view_min_valid_pixels);
                setVal(pseudo_view_mask_erode_pixels_val, opt.pseudo_view_mask_erode_pixels);
                setVal(pseudo_view_mask_valid_open_pixels_val, opt.pseudo_view_mask_valid_open_pixels);
                setVal(pseudo_view_mask_invalid_dilate_pixels_val, opt.pseudo_view_mask_invalid_dilate_pixels);
                if (disable_pseudo_view_training_flag) {
                    opt.use_pseudo_views_in_training = false;
                }
                if (gggs_val) {
                    opt.enable_gggs_loss = true;
                    opt.regularization_from_iter = *gggs_val;
                }
                if (gggs_gtnorm_val) {
                    opt.gggs_gtnorm = true;
                    opt.gggs_gtnorm_from_iter = *gggs_gtnorm_val;
                    opt.precompute_mesh_depth_normal = true;
                }
                if (mesh_depth_val) {
                    opt.enable_mesh_depth_loss = true;
                    opt.mesh_depth_loss_from_iter = *mesh_depth_val;
                    opt.precompute_mesh_depth_normal = true;
                }
                setFlag(bg_modulation_flag, opt.bg_modulation);
                setFlag(random_flag, opt.random);
                setFlag(gut_flag, opt.gut);
                setFlag(undistort_flag, opt.undistort);
                setFlag(enable_sparsity_flag, opt.enable_sparsity);

                // Pyramid training parameters
                setFlag(pyramid_training_flag, opt.pyramid_training);
                setVal(pyramid_levels_val, opt.pyramid_levels);
                setVal(pyramid_step_interval_val, opt.pyramid_step_interval);

                // Mask parameters
                setVal(mask_mode_val, opt.mask_mode);
                setFlag(invert_masks_flag, opt.invert_masks);
                if (no_alpha_as_mask_flag)
                    opt.use_alpha_as_mask = false;
                if (opt.mesh_depth_loss_weight > 0.0f ||
                    opt.mesh_normal_loss_weight > 0.0f) {
                    opt.precompute_mesh_depth_normal = true;
                }
                // Also propagate to dataset config for loading
                ds.invert_masks = opt.invert_masks;
                ds.mask_threshold = opt.mask_threshold;

                // Python scripts
                if (python_scripts_val) {
                    for (const auto& script : *python_scripts_val) {
                        params.python_scripts.emplace_back(script);
                    }
                }
            };

            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    void apply_step_scaling(lfs::core::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        opt.apply_step_scaling();
    }

    void apply_ppisp_defaults(lfs::core::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        if (!opt.ppisp_use_controller)
            return;

        if (opt.ppisp_controller_activation_step < 0) {
            opt.ppisp_controller_activation_step = opt.resolved_ppisp_controller_activation_step();
        }
    }


    void apply_configured_log_level(const lfs::core::param::TrainingParameters& params) {
#ifndef DEBUG_BUILD
        lfs::core::Logger::get().set_level(lfs::core::LogLevel::Error);
        return;
#else
        if (!params.optimization.log_level.empty()) {
            lfs::core::Logger::get().set_level(parse_log_level(params.optimization.log_level));
            LOG_DEBUG("Logger level applied from config/CLI: {}", params.optimization.log_level);
        }
#endif
    }

    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        auto args = std::vector<std::string>(argv, argv + argc);
        decrypt_path_arguments(args);
        return args;
    }
} // anonymous namespace

// Public interface
std::expected<std::unique_ptr<lfs::core::param::TrainingParameters>, std::string>
lfs::core::args::parse_args_and_params(int argc, const char* const argv[]) {

    auto params = std::make_unique<lfs::core::param::TrainingParameters>();
    auto args = convert_args(argc, argv);

    if (args.size() >= 2 && !args[1].starts_with('-') && args[1] != "convert" &&
        args[1] != "filter-error-scan-pose" && args[1] != "plugin") {
        const std::filesystem::path p = lfs::core::utf8_to_path(args[1]);
        std::error_code ec;
        if (std::filesystem::exists(p, ec))
            args.insert(args.begin() + 1, "-v");
    }

    auto parse_result = parse_arguments(args, *params);
    const std::string& strategy = params->optimization.strategy;
    const std::string& config_file = params->optimization.config_file;

    if (!parse_result) {
        return std::unexpected(parse_result.error());
    }

    const auto [result, apply_overrides] = *parse_result;
    if (result == ParseResult::Help) {
        std::exit(0);
    }

    // Load from --config or use hardcoded defaults
    if (!config_file.empty()) {
        const auto opt_result = lfs::core::param::read_optim_params_from_json(lfs::core::utf8_to_path(config_file));
        if (!opt_result) {
            return std::unexpected(std::format("Config load failed: {}", opt_result.error()));
        }
        params->optimization = *opt_result;

        if (!strategy.empty() && strategy != params->optimization.strategy) {
            return std::unexpected("--strategy conflicts with config file");
        }
    } else {
        if (strategy == "adc")
            params->optimization = lfs::core::param::OptimizationParameters::adc_defaults();
        else if (strategy == "igs+")
            params->optimization = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        else if (strategy == "mesh2splat") {
            params->optimization = lfs::core::param::OptimizationParameters::mcmc_defaults();
            params->optimization.strategy = "mesh2splat";
        }
        else
            params->optimization = lfs::core::param::OptimizationParameters::mcmc_defaults();
    }

    params->dataset.loading_params = lfs::core::param::LoadingParams{};

    if (apply_overrides) {
        apply_overrides();
    }
    apply_step_scaling(*params);
    apply_ppisp_defaults(*params);

    if (auto error = params->validate(); !error.empty())
        return std::unexpected("ERROR: " + error);

    apply_configured_log_level(*params);

    if (params->optimization.resolved_mesh2splat_hole_fill_mode() != lfs::core::param::Mesh2SplatHoleFillMode::None) {
        const auto& opt = params->optimization;
        if (opt.resolved_mesh2splat_hole_fill_mode() == lfs::core::param::Mesh2SplatHoleFillMode::External) {
            LOG_INFO("Mesh2Splat external completed-mesh mode enabled: filled_mesh='{}'",
                     params->mesh_init_external_filled_mesh_path.value_or(""));
        } else if (opt.resolved_mesh2splat_hole_fill_mode() ==
                   lfs::core::param::Mesh2SplatHoleFillMode::ExternalPointCloud) {
            LOG_INFO("Mesh2Splat external point-cloud mode enabled: pointcloud='{}', density_ratio={:.4g}, "
                     "surface_reject_ratio={:.4g}, mask_min_points_per_hole={}",
                     params->mesh_init_external_pointcloud_path.value_or(""),
                     opt.mesh2splat_pointcloud_density_ratio,
                     opt.mesh2splat_pointcloud_surface_reject_ratio,
                     opt.mesh2splat_pointcloud_mask_min_points_per_hole);
        } else {
        LOG_INFO(
            "Mesh2Splat hole fill enabled for training mesh initialization: max_edges={}, "
            "perimeter_ratio={}, area_ratio={}, bbox_diagonal_ratio={}, weld_ratio={}, "
            "density_factor={}, density_ratio=[{}, {}]",
            opt.mesh2splat_hole_fill_max_boundary_edges,
            opt.mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio,
            opt.mesh2splat_hole_fill_max_boundary_area_bbox_ratio,
            opt.mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio,
            opt.mesh2splat_hole_fill_weld_epsilon_bbox_ratio,
            opt.mesh2splat_hole_fill_density_control_factor,
            opt.mesh2splat_hole_fill_min_density_ratio,
            opt.mesh2splat_hole_fill_max_density_ratio);
        }
    }

    return params;
}

namespace {
    constexpr const char* CONVERT_HELP_HEADER = "LichtFeld Studio - Convert splat files between formats\n";
    constexpr const char* CONVERT_HELP_FOOTER =
        "\n"
        "EXAMPLES:\n"
        "  LichtFeld-Studio convert input.ply output.spz --sh-degree 0\n"
        "  LichtFeld-Studio convert input.ply -f html\n"
        "  LichtFeld-Studio convert input.ply -f splat\n"
        "  LichtFeld-Studio convert ./splats/ -f sog --sh-degree 2\n"
        "\n"
        "SUPPORTED FORMATS:\n"
        "  Input:  .ply, .sog, .spz, .resume (checkpoint)\n"
        "  Output: .ply, .sog, .spz, .splat, .html\n"
        "\n";

    constexpr const char* SCAN_POSE_FILTER_HELP_HEADER =
        "LichtFeld Studio - Filter scan frames with inconsistent poses\n";
    constexpr const char* SCAN_POSE_FILTER_HELP_FOOTER =
        "\n"
        "EXAMPLE:\n"
        "  Run-GS.exe filter-error-scan-pose --input-bin scan.bin --output-bin filtered.bin "
        "--report report.json --zncc-threshold 0.30\n"
        "\n";

    std::optional<lfs::core::param::OutputFormat> parseFormat(const std::string& str) {
        using lfs::core::param::OutputFormat;
        if (str == "ply" || str == ".ply")
            return OutputFormat::PLY;
        if (str == "sog" || str == ".sog")
            return OutputFormat::SOG;
        if (str == "spz" || str == ".spz")
            return OutputFormat::SPZ;
        if (str == "splat" || str == ".splat")
            return OutputFormat::SPLAT;
        if (str == "html" || str == ".html")
            return OutputFormat::HTML;
        return std::nullopt;
    }

    std::expected<lfs::core::args::ParsedArgs, std::string>
    parse_scan_pose_filter_args(const int argc, const char* const argv[]) {
        ::args::ArgumentParser parser(SCAN_POSE_FILTER_HELP_HEADER, SCAN_POSE_FILTER_HELP_FOOTER);
#if LFS_ENABLE_CLI_HELP
        ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
#endif
        ::args::ValueFlag<std::string> input_bin(
            parser, "path", "Input scanner bin file", {"input-bin"});
        ::args::ValueFlag<std::string> output_bin(
            parser, "path", "Output filtered scanner bin file", {"output-bin"});
        ::args::ValueFlag<std::string> report(
            parser, "path", "Output filter report path", {"report"});
        ::args::ValueFlag<float> zncc_threshold(
            parser, "value", "ZNCC rejection threshold in [-1, 1] (default: 0.30)", {"zncc-threshold"});

        std::vector<std::string> args_vec(argv + 1, argv + argc);
        args_vec[0] = std::string(argv[0]) + " filter-error-scan-pose";
        parser.Prog(args_vec[0]);

#if !LFS_ENABLE_CLI_HELP
        if (args_vec.size() == 2 && is_help_argument(args_vec[1])) {
            return lfs::core::args::HelpMode{};
        }
#endif

        try {
            parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
        } catch (const ::args::Help&) {
#if LFS_ENABLE_CLI_HELP
            std::print("{}", parser.Help());
#endif
            return lfs::core::args::HelpMode{};
        } catch (const ::args::ParseError& e) {
            return std::unexpected(append_cli_help(e.what(), parser));
        }

        if (!input_bin) {
            return std::unexpected(append_cli_help("Missing --input-bin", parser));
        }
        if (!output_bin) {
            return std::unexpected(append_cli_help("Missing --output-bin", parser));
        }
        if (!report) {
            return std::unexpected(append_cli_help("Missing --report", parser));
        }

        lfs::core::param::ScanPoseFilterParameters params;
        params.input_bin = lfs::core::utf8_to_path(::args::get(input_bin));
        params.output_bin = lfs::core::utf8_to_path(::args::get(output_bin));
        params.report_path = lfs::core::utf8_to_path(::args::get(report));
        if (zncc_threshold) {
            params.zncc_threshold = ::args::get(zncc_threshold);
        }

        if (!std::isfinite(params.zncc_threshold) ||
            params.zncc_threshold < -1.0f || params.zncc_threshold > 1.0f) {
            return std::unexpected("ZNCC threshold must be finite and within [-1, 1]");
        }

        return lfs::core::args::ScanPoseFilterMode{std::move(params)};
    }
} // namespace

std::expected<lfs::core::args::ParsedArgs, std::string>
lfs::core::args::parse_args(const int argc, const char* const argv[]) {
    if (argc >= 2) {
        const std::string_view arg1 = argv[1];

        if (arg1 == "reconstruct") {
            return ReconstructionMode{std::vector<std::string>(argv + 2, argv + argc), false};
        }

        // Swaptexture keeps its compact public CLI as a direct invocation while
        // the historical `reconstruct` subcommand remains available for native
        // LichtFeld tooling and fixtures.
        if (arg1 == "--bin_path" || arg1 == "--images_inc_path" ||
            arg1 == "--enable_gs_train" || arg1 == "--gs-input-source" ||
            arg1 == "--gs-training-mode" || arg1 == "--mesh-init-gs-scene" ||
            arg1 == "--result_path" || arg1 == "--log_path" ||
            arg1 == "--enable_inc_sim3_registration") {
            return ReconstructionMode{std::vector<std::string>(argv + 1, argv + argc), true};
        }

        if (arg1 == "-V" || arg1 == "--version") {
            return VersionMode{};
        }

        if (arg1 == "--warmup") {
            return WarmupMode{};
        }

#if !LFS_ENABLE_CLI_HELP
        if (is_help_argument(arg1)) {
            return HelpMode{};
        }
#endif

        if (arg1 == "filter-error-scan-pose") {
            return parse_scan_pose_filter_args(argc, argv);
        } else if (arg1 == "convert") {
            // Handle convert subcommand below
        } else if (arg1 == "plugin") {
            if (argc < 3) {
                return std::unexpected(
#if LFS_ENABLE_CLI_HELP
                    "Usage: LichtFeld-Studio plugin <create|check|list> [name]"
#else
                    "Missing plugin command"
#endif
                );
            }

            const std::string_view subcmd = argv[2];
            PluginMode mode;

            if (subcmd == "create") {
                if (argc < 4) {
                    return std::unexpected(
#if LFS_ENABLE_CLI_HELP
                        "Usage: LichtFeld-Studio plugin create <name>"
#else
                        "Missing plugin name for command 'create'"
#endif
                    );
                }
                mode.command = PluginMode::Command::CREATE;
                mode.name = argv[3];
            } else if (subcmd == "check") {
                if (argc < 4) {
                    return std::unexpected(
#if LFS_ENABLE_CLI_HELP
                        "Usage: LichtFeld-Studio plugin check <name>"
#else
                        "Missing plugin name for command 'check'"
#endif
                    );
                }
                mode.command = PluginMode::Command::CHECK;
                mode.name = argv[3];
            } else if (subcmd == "list") {
                mode.command = PluginMode::Command::LIST;
            } else if (subcmd == "-h" || subcmd == "--help") {
#if LFS_ENABLE_CLI_HELP
                std::print(R"(Usage: LichtFeld-Studio plugin <command> [name]

Commands:
  create <name>   Create plugin with venv and VS Code config
  check <name>    Validate plugin structure
  list            List installed plugins
)");
#endif
                return HelpMode{};
            } else {
                return std::unexpected(std::format("Unknown plugin command: {}", subcmd));
            }

            return mode;
        } else {
            auto result = parse_args_and_params(argc, argv);
            if (!result)
                return std::unexpected(result.error());
            return TrainingMode{std::move(*result)};
        }
    } else {
        auto result = parse_args_and_params(argc, argv);
        if (!result)
            return std::unexpected(result.error());
        return TrainingMode{std::move(*result)};
    }

    // Convert subcommand
    ::args::ArgumentParser parser(CONVERT_HELP_HEADER, CONVERT_HELP_FOOTER);
#if LFS_ENABLE_CLI_HELP
    ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
#endif
    ::args::Positional<std::string> input(parser, "input", "Input file or directory");
    ::args::Positional<std::string> output(parser, "output", "Output file (optional)");
    ::args::ValueFlag<int> sh_degree(parser, "degree", "SH degree [0-3], -1 to keep original (default: -1)", {"sh-degree"});
    ::args::ValueFlag<std::string> format(parser, "format", "Output format: ply, sog, spz, splat, html", {'f', "format"});
    ::args::ValueFlag<int> sog_iter(parser, "iterations", "K-means iterations for SOG (default: 10)", {"sog-iterations"});
    ::args::Flag overwrite(parser, "overwrite", "Overwrite existing files without prompting", {'y', "overwrite"});

    std::vector<std::string> args_vec(argv + 1, argv + argc);
    args_vec[0] = std::string(argv[0]) + " convert";
    parser.Prog(args_vec[0]);

#if !LFS_ENABLE_CLI_HELP
    if (args_vec.size() == 2 && is_help_argument(args_vec[1])) {
        return HelpMode{};
    }
#endif

    try {
        parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
    } catch (const ::args::Help&) {
#if LFS_ENABLE_CLI_HELP
        std::print("{}", parser.Help());
#endif
        return HelpMode{};
    } catch (const ::args::ParseError& e) {
        return std::unexpected(append_cli_help(e.what(), parser));
    }

    if (!input) {
        return std::unexpected(append_cli_help("Missing input path", parser));
    }

    param::ConvertParameters params;
    params.input_path = lfs::core::utf8_to_path(::args::get(input));
    params.sh_degree = sh_degree ? ::args::get(sh_degree) : -1;

    if (!std::filesystem::exists(params.input_path)) {
        return std::unexpected(std::format("Input not found: {}", lfs::core::path_to_utf8(params.input_path)));
    }

    if (params.sh_degree < -1 || params.sh_degree > 3) {
        return std::unexpected("SH degree must be -1 (keep) or 0-3");
    }

    if (output)
        params.output_path = lfs::core::utf8_to_path(::args::get(output));
    if (sog_iter)
        params.sog_iterations = ::args::get(sog_iter);
    params.overwrite = overwrite;

    if (format) {
        if (const auto fmt = parseFormat(::args::get(format))) {
            params.format = *fmt;
        } else {
            return std::unexpected(std::format("Invalid format '{}'. Use: ply, sog, spz, splat, html", ::args::get(format)));
        }
    } else if (!params.output_path.empty()) {
        if (const auto fmt = parseFormat(params.output_path.extension().string())) {
            params.format = *fmt;
        } else {
            return std::unexpected(std::format("Unknown extension '{}'. Use --format", params.output_path.extension().string()));
        }
    }

    return ConvertMode{params};
}
