/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/providers.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace lfs::preprocess {
    namespace {
        std::string system_path() {
#ifdef _WIN32
            std::array<wchar_t, 32768> buffer{};
            auto size = GetSystemDirectoryW(buffer.data(), static_cast<UINT>(buffer.size()));
            if (!size || size >= buffer.size())
                throw std::runtime_error("Cannot find Windows system directory");
            return path_utf8(std::filesystem::path(buffer.data()));
#else
            return {};
#endif
        }
        std::expected<void, Error> preflight(const RuntimePaths& paths, bool cuda, const ExecutionContext& context) {
            if (context.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Cancelled before provider launch"});
            if (auto result = verify_runtime_manifest(paths); !result)
                return result;
            return check_runtime_compatibility(cuda);
        }
    } // namespace

    ColmapExeProvider::ColmapExeProvider(RuntimePaths paths, ProcessRunner runner)
        : paths_(std::move(paths)), runner_(std::move(runner)) {}

    std::filesystem::path colmap_working_directory(
        const ColmapRequest& request, const std::filesystem::path& fallback) {
        std::filesystem::path common;
        for (const auto& value : {request.database, request.images, request.input_model,
                                  request.output_model, request.reference_images, request.image_list,
                                  request.image_masks}) {
            if (value.empty())
                continue;
            const auto parent = std::filesystem::absolute(value).lexically_normal().parent_path();
            if (common.empty()) {
                common = parent;
                continue;
            }
            while (!common.empty()) {
                const auto relative = parent.lexically_relative(common);
                if (!relative.empty() && *relative.begin() != "..")
                    break;
                const auto next = common.parent_path();
                if (next == common)
                    return fallback; // Different volumes: retain absolute-path behavior.
                common = next;
            }
        }
        while (!common.empty() && !std::filesystem::is_directory(common)) {
            const auto next = common.parent_path();
            if (next == common)
                return fallback;
            common = next;
        }
        return common.empty() ? fallback : common;
    }

    std::expected<ColmapResult, Error> ColmapExeProvider::run(
        const ColmapRequest& request, const ExecutionContext& context) try {
        const std::array<std::string, 10> commands{"feature_extractor", "exhaustive_matcher", "sequential_matcher",
                                                   "mapper", "image_registrator", "bundle_adjuster", "model_aligner", "point_triangulator", "model_converter", "image_undistorter"};
        const auto operation = static_cast<std::size_t>(request.operation);
        if (operation >= commands.size()) {
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "invalid COLMAP subcommand"});
        }
        if (auto result = preflight(paths_, true, context); !result)
            return std::unexpected(result.error());
        ProcessRequest process{
            .executable = paths_.colmap_executable,
            .arguments = {commands[operation]},
            .working_directory = colmap_working_directory(request, paths_.colmap_root),
        };
        auto path_option = [&](const char* key, const std::filesystem::path& value) {
            if (!value.empty()) {
                process.arguments.push_back(key);
                const auto absolute = std::filesystem::absolute(value).lexically_normal();
                const auto relative = absolute.lexically_relative(process.working_directory);
                process.arguments.push_back(path_utf8(relative.empty() ? absolute : relative));
            }
        };
        auto option = [&](const char* key, const auto& value) {
            if (value) {
                process.arguments.push_back(key);
                if constexpr (std::is_same_v<typename std::decay_t<decltype(value)>::value_type, std::string>)
                    process.arguments.push_back(*value);
                else
                    process.arguments.push_back(std::to_string(*value));
            }
        };
        path_option("--database_path", request.database);
        path_option("--image_path", request.images);
        path_option("--input_path", request.input_model);
        path_option("--output_path", request.output_model);
        path_option("--ref_images_path", request.reference_images);
        path_option("--image_list_path", request.image_list);
        path_option("--ImageReader.mask_path", request.image_masks);
        if (request.operation == ColmapOperation::ExtractFeatures) {
            option("--FeatureExtraction.max_image_size", request.extraction_max_image_size);
            option("--SiftExtraction.max_num_features", request.extraction_max_features);
            option("--FeatureExtraction.num_threads", request.num_threads);
            option("--FeatureExtraction.use_gpu", request.use_gpu);
            process.arguments.insert(process.arguments.end(), {"--FeatureExtraction.type", "SIFT"});
        } else if (request.operation == ColmapOperation::MatchExhaustive || request.operation == ColmapOperation::MatchSequential) {
            option("--FeatureMatching.guided_matching", request.guided_matching);
            option("--FeatureMatching.num_threads", request.num_threads);
            option("--FeatureMatching.use_gpu", request.use_gpu);
            process.arguments.insert(process.arguments.end(), {"--FeatureMatching.type", "SIFT_BRUTEFORCE"});
        } else if (request.operation == ColmapOperation::Map || request.operation == ColmapOperation::RegisterImages || request.operation == ColmapOperation::TriangulatePoints) {
            option("--Mapper.num_threads", request.num_threads);
            option("--Mapper.fix_existing_frames", request.fix_existing_frames);
            option("--Mapper.ba_use_gpu", request.use_gpu);
        } else if (request.operation == ColmapOperation::BundleAdjust) {
            option("--BundleAdjustmentCeres.use_gpu", request.use_gpu);
        }
        option("--ImageReader.camera_model", request.camera_model);
        option("--ImageReader.camera_params", request.camera_parameters);
        option("--ImageReader.single_camera", request.single_camera);
        option("--Mapper.min_num_matches", request.minimum_matches);
        option("--Mapper.init_min_num_inliers", request.initial_minimum_inliers);
        option("--Mapper.abs_pose_min_num_inliers", request.absolute_pose_minimum_inliers);
        option("--Mapper.min_model_size", request.minimum_model_size);
        option("--output_type", request.output_format);
        option("--ref_is_gps", request.references_are_gps);
        option("--alignment_type", request.alignment_type);
        option("--alignment_max_error", request.alignment_maximum_error);
        if (request.probe_help)
            process.arguments.push_back("-h");
        process.environment = {{"PATH", path_utf8(paths_.colmap_executable.parent_path()) + ";" + system_path()},
                               {"QT_QPA_PLATFORM_PLUGIN_PATH", path_utf8(paths_.colmap_executable.parent_path() / "platforms")},
                               {"QT_PLUGIN_PATH", ""},
                               {"QT_QPA_PLATFORM", "offscreen"}};
        auto execution = context;
        execution.stage = Stage::Colmap;
        auto result = runner_.run(process, execution);
        if (!result)
            return std::unexpected(result.error());
        return ColmapResult{.operation = request.operation,
                            .diagnostics = ToolDiagnostics{result->exit_code, std::move(result->stdout_tail), std::move(result->stderr_tail)}};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }

    SuperResolutionExeProvider::SuperResolutionExeProvider(RuntimePaths paths, ProcessRunner runner)
        : paths_(std::move(paths)), runner_(std::move(runner)) {}

    std::expected<SuperResolutionResult, Error> SuperResolutionExeProvider::run(
        const SuperResolutionRequest& request, const ExecutionContext& context) try {
        if (request.input_directory.empty() || request.output_directory.empty() || request.model_name.empty() ||
            request.model_scale <= 0 || request.final_scale <= 0 ||
            request.model_name.find_first_of("/\\:") != std::string::npos || request.model_name.find("..") != std::string::npos) {
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "invalid super-resolution request"});
        }
        if (auto result = preflight(paths_, false, context); !result)
            return std::unexpected(result.error());
        const auto param = paths_.super_resolution_models / (request.model_name + ".param");
        const auto bin = paths_.super_resolution_models / (request.model_name + ".bin");
        if (!std::filesystem::is_regular_file(param) || !std::filesystem::is_regular_file(bin)) {
            return std::unexpected(Error{.code = ErrorCode::RuntimeMissing,
                                         .message = std::format("model pair is missing for {}", request.model_name)});
        }
        std::set<std::filesystem::path> expected;
        std::error_code ec;
        if (!std::filesystem::is_directory(request.input_directory, ec))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "SR input directory missing"});
        for (const auto& item : std::filesystem::directory_iterator(request.input_directory)) {
            if (!item.is_regular_file())
                continue;
            auto ext = item.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png")
                continue;
            auto name = item.path().filename();
            name.replace_extension(".png");
            if (!expected.insert(name).second)
                return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Ambiguous SR output stems"});
        }
        if (expected.empty() || std::filesystem::exists(request.output_directory, ec))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "SR requires images and a fresh output directory"});
        if (!std::filesystem::create_directories(request.output_directory, ec) || ec)
            return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot create SR output directory"});
        ProcessRequest process{
            .executable = paths_.super_resolution_executable,
            .arguments = {"-i", path_utf8(std::filesystem::absolute(request.input_directory)), "-o", path_utf8(std::filesystem::absolute(request.output_directory)),
                          "-s", std::to_string(request.model_scale), "-r", std::to_string(request.final_scale),
                          "-n", request.model_name, "-m", path_utf8(paths_.super_resolution_models)},
            .working_directory = paths_.super_resolution_root,
        };
        process.environment = {{"PATH", path_utf8(paths_.super_resolution_root) + ";" + system_path()}};
        auto execution = context;
        execution.stage = Stage::SuperResolution;
        auto result = runner_.run(process, execution);
        if (!result)
            return std::unexpected(result.error());
        std::set<std::filesystem::path> actual;
        for (const auto& item : std::filesystem::directory_iterator(request.output_directory))
            if (item.is_regular_file() && item.file_size() > 0)
                actual.insert(item.path().filename());
        if (actual != expected)
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "SR output image set differs from expected"});
        return SuperResolutionResult{.output_directory = std::filesystem::absolute(request.output_directory), .image_count = actual.size(), .diagnostics = ToolDiagnostics{result->exit_code, std::move(result->stdout_tail), std::move(result->stderr_tail)}};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }

} // namespace lfs::preprocess
