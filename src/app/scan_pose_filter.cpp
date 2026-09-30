/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/scan_pose_filter.hpp"

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "io/loader.hpp"
#include "io/scanner_bin.hpp"
#include "training/mesh_supervision_renderer.hpp"
#include "training/scan_pose_filter.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <print>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace lfs::app {
    namespace {

        namespace fs = std::filesystem;
        using json = nlohmann::json;

        // Scanner bin poses are stored in millimetres, while mesh.ply and the
        // mesh supervision renderer use metres.  This conversion is local to
        // the filter cameras; the bin data is written back unchanged.
        constexpr float kScannerPoseTranslationToMeshUnits = 0.001f;

        struct ResolvedFrame {
            std::size_t flat_index = 0;
            std::size_t group_index = 0;
            std::size_t frame_index = 0;
            fs::path image_path;
        };

        fs::path normalized_path_for_comparison(const fs::path& path) {
            std::error_code ec;
            auto normalized = fs::weakly_canonical(path, ec);
            return ec ? path.lexically_normal() : normalized.lexically_normal();
        }

        bool paths_refer_to_same_target(const fs::path& lhs, const fs::path& rhs) {
            std::error_code ec;
            if (fs::exists(lhs, ec) && !ec && fs::exists(rhs, ec) && !ec &&
                fs::equivalent(lhs, rhs, ec) && !ec) {
                return true;
            }

            auto lhs_normalized = normalized_path_for_comparison(lhs).native();
            auto rhs_normalized = normalized_path_for_comparison(rhs).native();
#ifdef _WIN32
            std::transform(lhs_normalized.begin(), lhs_normalized.end(), lhs_normalized.begin(),
                           [](const wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); });
            std::transform(rhs_normalized.begin(), rhs_normalized.end(), rhs_normalized.begin(),
                           [](const wchar_t ch) { return static_cast<wchar_t>(std::towlower(ch)); });
#endif
            return lhs_normalized == rhs_normalized;
        }

        void append_mesh_candidates_from_ancestors(
            std::vector<fs::path>& candidates,
            fs::path parent,
            const bool include_direct_mesh) {
            while (!parent.empty()) {
                candidates.push_back(parent / "tex_dump" / "mesh.ply");
                candidates.push_back(parent / "tex_dump" / "Mesh.ply");
                if (include_direct_mesh) {
                    candidates.push_back(parent / "mesh.ply");
                    candidates.push_back(parent / "Mesh.ply");
                }
                const auto next = parent.parent_path();
                if (next == parent) {
                    break;
                }
                parent = next;
            }
        }

        std::expected<fs::path, std::string> absolute_existing_file(
            const fs::path& candidate,
            const char* description) {
            std::error_code ec;
            if (candidate.empty() || !fs::is_regular_file(candidate, ec)) {
                return std::unexpected(std::string(description) + " does not exist: " + lfs::core::path_to_utf8(candidate));
            }
            auto absolute = fs::absolute(candidate, ec);
            if (ec) {
                return std::unexpected(std::string("Failed to resolve ") + description + ": " + ec.message());
            }
            return absolute.lexically_normal();
        }

        fs::path path_from_record(const std::string& value, const fs::path& input_dir) {
            if (value.empty()) {
                return {};
            }
            auto path = lfs::core::utf8_to_path(value);
            if (path.is_absolute()) {
                return path;
            }

            std::error_code ec;
            const auto input_relative = input_dir / path;
            if (fs::exists(input_relative, ec)) {
                return input_relative;
            }
            return path;
        }

        std::expected<fs::path, std::string> resolve_mesh_path(
            const lfs::io::ScannerBinData& data,
            const fs::path& input_dir) {
            std::vector<fs::path> candidates;
            const auto recorded = path_from_record(data.mesh_path_utf8, input_dir);
            if (!recorded.empty()) {
                candidates.push_back(recorded);
                candidates.push_back(recorded.parent_path() / "Mesh.ply");
                append_mesh_candidates_from_ancestors(
                    candidates,
                    recorded.parent_path(),
                    true);
            }
            candidates.push_back(input_dir / "mesh.ply");
            candidates.push_back(input_dir / "Mesh.ply");
            candidates.push_back(input_dir / "tex_dump" / "mesh.ply");
            candidates.push_back(input_dir / "tex_dump" / "Mesh.ply");
            append_mesh_candidates_from_ancestors(
                candidates,
                input_dir.parent_path(),
                false);

            std::error_code ec;
            for (const auto& candidate : candidates) {
                if (candidate.empty() || !fs::is_regular_file(candidate, ec)) {
                    ec.clear();
                    continue;
                }
                return absolute_existing_file(candidate, "Scanner mesh");
            }
            return std::unexpected(
                "Scanner mesh does not exist in the bin record or the swaptexture fallback locations");
        }

        std::expected<std::vector<ResolvedFrame>, std::string> resolve_frame_paths(
            lfs::io::ScannerBinData& data,
            const fs::path& input_dir) {
            std::vector<ResolvedFrame> resolved;
            resolved.reserve(data.frame_count());
            std::size_t flat_index = 0;
            for (std::size_t group_index = 0; group_index < data.groups.size(); ++group_index) {
                auto& group = data.groups[group_index];
                for (std::size_t frame_index = 0; frame_index < group.frames.size(); ++frame_index, ++flat_index) {
                    auto& frame = group.frames[frame_index];
                    auto candidate = path_from_record(frame.image_path_utf8, input_dir);
                    std::error_code ec;
                    if (candidate.empty() || !fs::is_regular_file(candidate, ec)) {
                        candidate = input_dir / (std::to_string(flat_index) + ".png");
                    }
                    auto absolute = absolute_existing_file(candidate, "Scanner RGB image");
                    if (!absolute) {
                        return std::unexpected(std::format(
                            "{} (frame {})", absolute.error(), flat_index));
                    }
                    frame.image_path_utf8 = lfs::core::path_to_utf8(*absolute);
                    resolved.push_back({flat_index, group_index, frame_index, *absolute});
                }
            }
            return resolved;
        }

        std::expected<std::shared_ptr<lfs::core::Camera>, std::string> make_camera(
            const lfs::io::ScannerBinIntrinsicGroup& group,
            const lfs::io::ScannerBinFrame& frame,
            const fs::path& image_path,
            const int uid,
            const int camera_id) {
            std::array<float, 9> rotation_w2c{};
            std::array<float, 3> translation_w2c{};
            for (const float value : frame.pose_c2w) {
                if (!std::isfinite(value)) {
                    return std::unexpected("Scanner pose contains a non-finite value");
                }
            }
            auto pose_c2w = frame.pose_c2w;
            pose_c2w[3] *= kScannerPoseTranslationToMeshUnits;
            pose_c2w[7] *= kScannerPoseTranslationToMeshUnits;
            pose_c2w[11] *= kScannerPoseTranslationToMeshUnits;
            if (!std::isfinite(group.fx) || !std::isfinite(group.fy) ||
                !std::isfinite(group.cx) || !std::isfinite(group.cy) ||
                group.fx <= 0.0f || group.fy <= 0.0f) {
                return std::unexpected("Scanner intrinsics are not a valid finite pinhole calibration");
            }

            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    rotation_w2c[row * 3 + col] = pose_c2w[col * 4 + row];
                }
                translation_w2c[row] = -(
                    rotation_w2c[row * 3 + 0] * pose_c2w[3] + rotation_w2c[row * 3 + 1] * pose_c2w[7] + rotation_w2c[row * 3 + 2] * pose_c2w[11]);
            }

            try {
                const auto [width, height, channels] = lfs::core::get_image_info(image_path);
                if (width <= 0 || height <= 0 || channels < 3) {
                    return std::unexpected("Scanner RGB image must contain at least three channels");
                }
                auto rotation = lfs::core::Tensor::from_vector(
                    std::vector<float>(rotation_w2c.begin(), rotation_w2c.end()),
                    {size_t{3}, size_t{3}},
                    lfs::core::Device::CPU);
                auto translation = lfs::core::Tensor::from_vector(
                    std::vector<float>(translation_w2c.begin(), translation_w2c.end()),
                    {size_t{3}},
                    lfs::core::Device::CPU);

                return std::make_shared<lfs::core::Camera>(
                    rotation,
                    translation,
                    group.fx,
                    group.fy,
                    group.cx,
                    group.cy,
                    lfs::core::Tensor{},
                    lfs::core::Tensor{},
                    lfs::core::CameraModelType::PINHOLE,
                    lfs::core::path_to_utf8(image_path.filename()),
                    image_path,
                    fs::path{},
                    width,
                    height,
                    uid,
                    camera_id);
            } catch (const std::exception& error) {
                return std::unexpected(std::string("Failed to create scanner camera: ") + error.what());
            }
        }

        std::expected<void, std::string> publish_text_file(
            const fs::path& output_path,
            const std::string& contents) {
            std::error_code ec;
            if (!output_path.parent_path().empty()) {
                fs::create_directories(output_path.parent_path(), ec);
                if (ec) {
                    return std::unexpected("Failed to create report directory: " + ec.message());
                }
            }

            auto temp_path = output_path;
            temp_path += ".tmp." + std::to_string(
                                       std::chrono::steady_clock::now().time_since_epoch().count());
            std::ofstream stream;
            if (!lfs::core::open_file_for_write(temp_path, std::ios::out | std::ios::trunc, stream)) {
                return std::unexpected("Failed to open temporary filter report: " + lfs::core::path_to_utf8(temp_path));
            }
            stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
            stream.flush();
            if (!stream) {
                stream.close();
                fs::remove(temp_path, ec);
                return std::unexpected("Failed to write temporary filter report");
            }
            stream.close();

#ifdef _WIN32
            if (!MoveFileExW(temp_path.c_str(), output_path.c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const auto error = GetLastError();
                fs::remove(temp_path, ec);
                return std::unexpected("Failed to publish filter report atomically (Win32 error " + std::to_string(error) + ")");
            }
#else
            fs::rename(temp_path, output_path, ec);
            if (ec) {
                fs::remove(temp_path, ec);
                return std::unexpected("Failed to publish filter report atomically: " + ec.message());
            }
#endif
            return {};
        }

        int fail(const std::string& message) {
            std::println(stderr, "filter-error-scan-pose: {}", message);
            return 1;
        }

    } // namespace

    int run_scan_pose_filter(const lfs::core::param::ScanPoseFilterParameters& params) {
        try {
            std::error_code ec;
            const auto input_path = fs::absolute(params.input_bin, ec).lexically_normal();
            if (ec) {
                return fail("Failed to resolve input bin path: " + ec.message());
            }
            const auto output_path = fs::absolute(params.output_bin, ec).lexically_normal();
            if (ec) {
                return fail("Failed to resolve output bin path: " + ec.message());
            }
            const auto report_path = fs::absolute(params.report_path, ec).lexically_normal();
            if (ec) {
                return fail("Failed to resolve report path: " + ec.message());
            }
            if (paths_refer_to_same_target(input_path, output_path) ||
                paths_refer_to_same_target(input_path, report_path) ||
                paths_refer_to_same_target(output_path, report_path)) {
                return fail("Input bin, output bin, and report paths must be distinct");
            }
            for (const auto& [path, description] : std::array{
                     std::pair{output_path, "output bin"},
                     std::pair{report_path, "report"}}) {
                ec.clear();
                if (fs::exists(path, ec)) {
                    return fail(std::format(
                        "Refusing to overwrite existing {}: {}",
                        description,
                        lfs::core::path_to_utf8(path)));
                }
                if (ec) {
                    return fail(std::format(
                        "Failed to inspect {} path '{}': {}",
                        description,
                        lfs::core::path_to_utf8(path),
                        ec.message()));
                }
            }

            auto data_result = lfs::io::read_scanner_bin(input_path);
            if (!data_result) {
                return fail(data_result.error());
            }
            auto source_data = std::move(*data_result);
            if (source_data.frame_count() == 0) {
                return fail("Input scanner bin contains no frames");
            }

            auto mesh_path = resolve_mesh_path(source_data, input_path.parent_path());
            if (!mesh_path) {
                return fail(mesh_path.error());
            }
            source_data.mesh_path_utf8 = lfs::core::path_to_utf8(*mesh_path);
            auto resolved_frames = resolve_frame_paths(source_data, input_path.parent_path());
            if (!resolved_frames) {
                return fail(resolved_frames.error());
            }

            const auto loader = lfs::io::Loader::create();
            auto loaded_mesh = loader->load(*mesh_path);
            if (!loaded_mesh) {
                return fail("Failed to load scanner mesh: " + loaded_mesh.error().format());
            }
            auto* mesh_data = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&loaded_mesh->data);
            if (!mesh_data || !*mesh_data) {
                return fail("Scanner mesh path did not load as a triangle mesh");
            }
            lfs::core::Scene mesh_scene;
            if (mesh_scene.addMesh("scan_pose_filter_mesh", *mesh_data) == lfs::core::NULL_NODE) {
                return fail("Failed to add scanner mesh to the filter scene");
            }
            auto prepared_mesh = lfs::training::prepare_mesh_geometry(mesh_scene);
            if (!prepared_mesh) {
                return fail(prepared_mesh.error());
            }

            lfs::io::CacheLoader::getInstance(false, false);
            lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& request) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(
                    request.path,
                    {.resize_factor = request.resize_factor,
                     .max_width = request.max_width,
                     .cuda_stream = request.stream});
            });

            lfs::io::ScannerBinData filtered_data = source_data;
            for (auto& group : filtered_data.groups) {
                group.frames.clear();
            }

            json report = {
                {"command", "filter-error-scan-pose"},
                {"input_bin", lfs::core::path_to_utf8(input_path)},
                {"output_bin", lfs::core::path_to_utf8(output_path)},
                {"mesh_path", lfs::core::path_to_utf8(*mesh_path)},
                {"zncc_threshold", params.zncc_threshold},
                {"input_frame_count", source_data.frame_count()},
                {"frames", json::array()},
            };

            std::size_t flat_offset = 0;
            std::size_t kept_count = 0;
            std::size_t dropped_count = 0;
            std::size_t uncertain_count = 0;
            for (std::size_t group_index = 0; group_index < source_data.groups.size(); ++group_index) {
                const auto& source_group = source_data.groups[group_index];
                auto& output_group = filtered_data.groups[group_index];
                if (source_group.frames.empty()) {
                    continue;
                }
                std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
                cameras.reserve(source_group.frames.size());
                for (std::size_t frame_index = 0; frame_index < source_group.frames.size(); ++frame_index) {
                    const auto& record = source_group.frames[frame_index];
                    const auto& resolved = (*resolved_frames)[flat_offset + frame_index];
                    auto camera = make_camera(
                        source_group,
                        record,
                        resolved.image_path,
                        static_cast<int>(flat_offset + frame_index),
                        static_cast<int>(group_index));
                    if (!camera) {
                        return fail(std::format(
                            "Frame {}: {}", flat_offset + frame_index, camera.error()));
                    }
                    cameras.push_back(std::move(*camera));
                }

                auto scores = lfs::training::score_scan_pose_frames_zncc(
                    *prepared_mesh,
                    cameras,
                    params.zncc_threshold);
                if (!scores) {
                    return fail(std::format("Intrinsic group {}: {}", group_index, scores.error()));
                }
                if (scores->frames.size() != source_group.frames.size()) {
                    return fail("ZNCC filter returned a frame-count mismatch");
                }

                for (std::size_t frame_index = 0; frame_index < source_group.frames.size(); ++frame_index) {
                    const auto& source_frame = source_group.frames[frame_index];
                    const auto& frame_score = scores->frames[frame_index];
                    const std::size_t flat_index = flat_offset + frame_index;
                    const char* decision = nullptr;
                    if (!frame_score.zncc) {
                        decision = "uncertain";
                        ++uncertain_count;
                    } else if (frame_score.keep) {
                        decision = "keep";
                    } else {
                        decision = "drop";
                    }

                    if (frame_score.keep) {
                        output_group.frames.push_back(source_frame);
                        ++kept_count;
                    } else {
                        ++dropped_count;
                    }

                    json frame_json = {
                        {"flat_index", flat_index},
                        {"group_index", group_index},
                        {"frame_index", frame_index},
                        {"image_path", source_frame.image_path_utf8},
                        {"decision", decision},
                        {"pair_scores", json::array()},
                    };
                    frame_json["zncc"] = frame_score.zncc
                                             ? json(*frame_score.zncc)
                                             : json(nullptr);
                    frame_json["photometric_zncc"] = frame_score.photometric_zncc
                                                         ? json(*frame_score.photometric_zncc)
                                                         : json(nullptr);
                    for (const auto& pair : frame_score.pair_scores) {
                        json pair_json = {
                            {"peer_frame_index", pair.peer_index},
                            {"peer_flat_index", flat_offset + pair.peer_index},
                            {"zncc", pair.zncc},
                            {"informative_regions", pair.informative_regions},
                            {"valid_samples", pair.valid_samples},
                            {"target_samples", pair.target_samples},
                            {"projected_samples", pair.projected_samples},
                            {"depth_consistent_samples", pair.depth_consistent_samples},
                            {"geometry_consistent_regions", pair.geometry_consistent_regions},
                        };
                        pair_json["photometric_zncc"] = pair.photometric_zncc
                                                            ? json(*pair.photometric_zncc)
                                                            : json(nullptr);
                        frame_json["pair_scores"].push_back(std::move(pair_json));
                    }
                    report["frames"].push_back(std::move(frame_json));
                }
                flat_offset += source_group.frames.size();
            }

            if (kept_count == 0) {
                // This is an output-validity guard, not a second frame classifier:
                // no rejected frame is reclassified or retained here.  Publishing
                // a zero-frame bin would make the downstream swaptexture loader
                // fail later without a useful filter-level diagnostic.
                return fail("The configured ZNCC threshold rejected every frame; no output was published");
            }
            // Empty leading intrinsic groups make swaptexture's legacy 0.cfg
            // fallback describe a camera that has no retained frames.  Remove
            // only empty groups; non-empty groups and their frame order remain
            // unchanged.
            std::erase_if(filtered_data.groups, [](const auto& group) {
                return group.frames.empty();
            });
            report["kept_count"] = kept_count;
            report["dropped_count"] = dropped_count;
            report["uncertain_count"] = uncertain_count;
            report["output_frame_count"] = kept_count;
            report["output_intrinsic_group_count"] = filtered_data.groups.size();

            auto write_bin = lfs::io::write_scanner_bin_atomic(output_path, filtered_data);
            if (!write_bin) {
                return fail(write_bin.error());
            }
            auto write_report = publish_text_file(report_path, report.dump(2) + "\n");
            if (!write_report) {
                fs::remove(output_path, ec);
                return fail(write_report.error());
            }

            std::println(
                "filter-error-scan-pose: kept {}, dropped {}, uncertain {}; output: {}",
                kept_count,
                dropped_count,
                uncertain_count,
                lfs::core::path_to_utf8(output_path));
            return 0;
        } catch (const std::exception& error) {
            return fail(error.what());
        }
    }

} // namespace lfs::app
