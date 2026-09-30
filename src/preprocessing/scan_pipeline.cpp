/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/scan_pipeline.hpp"
#include "blur_filter_stage.hpp"
#include "io/scanner_bin.hpp"
#include "preprocessing/colmap_model.hpp"
#include "preprocessing/dataset_validation.hpp"
#include "preprocessing/debug_exports.hpp"
#include "preprocessing/image_pipeline.hpp"
#include "preprocessing/workspace.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <set>
#include <sstream>
#include <tuple>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;
        struct Failure : std::runtime_error {
            Error error;
            explicit Failure(Error e) : std::runtime_error(e.message), error(std::move(e)) {}
        };
        void require(bool condition, const std::string& message, ErrorCode code = ErrorCode::InvalidDataset) {
            if (!condition)
                throw Failure({.code = code, .message = message});
        }
        template <class T>
        T take(std::expected<T, Error> result) {
            if (!result)
                throw Failure(result.error());
            if constexpr (!std::is_void_v<T>)
                return std::move(*result);
        }
        template <class F>
        std::expected<void, Error> guarded(F&& action) {
            try {
                action();
                return {};
            } catch (const Failure& e) {
                return std::unexpected(e.error);
            } catch (const std::exception& e) {
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
            }
        }
        void progress(const ExecutionContext& ctx, float value, const std::string& message) {
            require(!ctx.stop_token.stop_requested(), "Scan preprocessing cancelled", ErrorCode::ProcessCancelled);
            if (ctx.on_progress)
                require(ctx.on_progress(ctx.stage, value, message), "Scan progress callback cancelled", ErrorCode::ProcessCancelled);
        }
        void write_text(const fs::path& path, const std::string& text) {
            std::ofstream output(path, std::ios::binary | std::ios::noreplace);
            output << text;
            output.close();
            require(bool(output), "Cannot write scan artifact: " + path_utf8(path), ErrorCode::IoFailure);
        }
        // Match the frozen Python disk round-trip: float32 translation / 1000,
        // all 16 values formatted to six decimals, then inverse in double precision.
        std::pair<ColmapImage, std::string> scan_pose(const std::array<float, 16>& raw) {
            take(scanner_pose_to_colmap(raw)); // reject non-rigid or non-finite scanner poses
            auto meters = raw;
            for (const auto index : {3, 7, 11})
                meters[index] /= 1000.F;
            std::ostringstream text;
            text.imbue(std::locale::classic());
            text << std::fixed << std::setprecision(6);
            for (float value : meters)
                text << value << ' ';
            cv::Matx44d c2w, w2c;
            std::istringstream input(text.str());
            input.imbue(std::locale::classic());
            for (double& value : c2w.val)
                input >> value;
            require(cv::invert(c2w, w2c, cv::DECOMP_LU) != 0, "Singular scanner pose");
            const double xx = w2c(0, 0), yx = w2c(0, 1), zx = w2c(0, 2);
            const double xy = w2c(1, 0), yy = w2c(1, 1), zy = w2c(1, 2);
            const double xz = w2c(2, 0), yz = w2c(2, 1), zz = w2c(2, 2);
            // numpy.linalg.eigh consumes the lower triangle. OpenCV consumes the
            // complete symmetric matrix and returns eigenvectors in descending order.
            const cv::Matx44d k(xx - yy - zz, yx + xy, zx + xz, yz - zy,
                                yx + xy, yy - xx - zz, zy + yz, zx - xz,
                                zx + xz, zy + yz, zz - xx - yy, xy - yx,
                                yz - zy, zx - xz, xy - yx, xx + yy + zz);
            cv::Mat values, vectors;
            require(cv::eigen(cv::Mat(k / 3.), values, vectors), "Cannot convert scan rotation");
            ColmapImage image;
            image.rotation = {vectors.at<double>(0, 3), vectors.at<double>(0, 0),
                              vectors.at<double>(0, 1), vectors.at<double>(0, 2)};
            if (image.rotation[0] < 0)
                for (auto& value : image.rotation)
                    value = -value;
            image.translation = {w2c(0, 3), w2c(1, 3), w2c(2, 3)};
            return {image, text.str()};
        }
        using CameraKey = std::tuple<std::array<double, 4>, int, int>;
        CameraKey camera_key(const io::ScannerBinIntrinsicGroup& group, const CpuImage& image) {
            std::ostringstream text;
            text.imbue(std::locale::classic());
            text << std::fixed << std::setprecision(6) << group.fx << ' ' << group.fy << ' ' << group.cx << ' ' << group.cy;
            std::array<double, 4> rounded{};
            std::istringstream parsed(text.str());
            parsed.imbue(std::locale::classic());
            for (double& value : rounded)
                parsed >> value;
            // Python numeric tuple keys consider rounded -0.0 and +0.0 equal.
            return {rounded, image.width, image.height};
        }
        void validate_images(const fs::path& images, const ColmapModel& model, const ExecutionContext& ctx) {
            std::set<std::string> expected, actual;
            for (const auto& [id, image] : model.images)
                expected.insert(image.name);
            for (const auto& file : fs::directory_iterator(images)) {
                require(file.is_regular_file() && !file.is_symlink(), "Unexpected non-image entry in scan images");
                actual.insert(path_utf8(file.path().filename()));
            }
            require(actual == expected, "Scan image names/count do not exactly match cameras/images model");
            for (const auto& [id, image] : model.images) {
                require(!ctx.stop_token.stop_requested(), "Cancelled while validating scan images", ErrorCode::ProcessCancelled);
                auto dimensions = take(image_dimensions(images / fs::u8path(image.name)));
                const auto& camera = model.cameras.at(image.camera_id);
                require(camera.width == uint64_t(dimensions[0]) && camera.height == uint64_t(dimensions[1]),
                        "Scan/SR dimensions do not match scaled intrinsics: " + image.name);
            }
        }
    } // namespace

    std::expected<ColmapModel, Error> prepare_scan_source(const PreprocessRequest& request,
                                                          const fs::path& staging, std::optional<size_t> sample_limit,
                                                          const ExecutionContext& stage_ctx) try {
        auto loaded = io::read_scanner_bin(request.scanner_bin);
        require(bool(loaded), loaded ? "" : loaded.error());
        const auto& data = *loaded;
        struct Frame {
            size_t index;
            size_t group;
            const io::ScannerBinFrame* frame;
            fs::path source;
        };
        std::vector<Frame> available;
        std::vector<fs::path> inputs{fs::absolute(request.scanner_bin).parent_path()};
        size_t index = 0;
        for (size_t group = 0; group < data.groups.size(); ++group) {
            for (const auto& frame : data.groups[group].frames) {
                // Deliberately no decryption of bin-internal paths. Relative paths
                // retain Python's cwd semantics before the bin-relative fallback.
                auto source = fs::u8path(frame.image_path_utf8);
                std::error_code ec;
                if (!fs::is_regular_file(source, ec))
                    source = request.scanner_bin.parent_path() / (std::to_string(index) + ".png");
                ec.clear();
                if (fs::is_regular_file(source, ec)) {
                    source = fs::absolute(source);
                    available.push_back({index, group, &frame, source});
                    inputs.push_back(source);
                }
                ++index;
            }
        }
        require(!available.empty(), "No scanner images exist (including indexed PNG fallback)");
        std::sort(available.begin(), available.end(), [](const Frame& a, const Frame& b) {
            return std::to_string(a.index) + ".png" < std::to_string(b.index) + ".png";
        });
        const auto selection = uniform_sample(available.size(), sample_limit.value_or(available.size()));
        require(selection.size() < std::numeric_limits<uint32_t>::max(), "Too many scan frames");
        ColmapModel model;
        fs::create_directory(staging / "sampled_images");
        fs::create_directory(staging / "sampled_poses");
        fs::create_directory(staging / "temp_cameras");
        std::map<CameraKey, uint32_t> camera_ids;
        Json mapping{{"schema_version", 1}, {"scanner_bin", path_utf8(fs::absolute(request.scanner_bin))}, {"source_frame_count", data.frame_count()}, {"available_frame_count", available.size()}, {"sampled_indices", Json::array()}, {"frames", Json::array()}};
        for (size_t i = 0; i < selection.size(); ++i) {
            progress(stage_ctx, float(i) / selection.size(), "Importing scan frame " + std::to_string(i));
            const auto& selected = available[selection[i]];
            const auto& group = data.groups[selected.group];
            const auto pixels = take(read_image(selected.source));
            const auto name = std::to_string(i) + ".png";
            take(write_png(staging / "sampled_images" / name, pixels));
            const auto key = camera_key(group, pixels);
            auto [camera, inserted] = camera_ids.emplace(key, uint32_t(camera_ids.size() + 1));
            if (inserted)
                model.cameras.emplace(camera->second, ColmapCamera{camera->second, 1, uint64_t(pixels.width), uint64_t(pixels.height), {group.fx, group.fy, group.cx, group.cy}});
            auto [image, pose] = scan_pose(selected.frame->pose_c2w);
            image.id = uint32_t(i + 1);
            image.camera_id = camera->second;
            image.name = name;
            model.images.emplace(image.id, image);
            write_text(staging / "sampled_poses" / (std::to_string(i) + ".pose"), pose);
            mapping["sampled_indices"].push_back(selected.index);
            mapping["frames"].push_back({{"source_index", selected.index}, {"intrinsic_group", selected.group}, {"source", path_utf8(selected.source)}, {"name", name}, {"camera_id", image.camera_id}});
        }
        const auto* scan_input = std::get_if<ScanInput>(&request.input);
        const auto* registered_input = std::get_if<RegisteredInput>(&request.input);
        const bool blur_enabled = scan_input         ? scan_input->options.enable_blur_filter
                                  : registered_input ? registered_input->options.enable_scan_blur_filter
                                                     : std::get<MeshExportInput>(request.input).options.enable_blur_filter;
        if (blur_enabled) {
            auto filter_ctx = stage_ctx;
            // Imported images have already advanced this shared stage's progress.
            const float import_progress = float(selection.size() - 1) / selection.size();
            filter_ctx.on_progress = [stage_ctx, import_progress](Stage, float, std::string message) {
                return !stage_ctx.stop_token.stop_requested() &&
                       (!stage_ctx.on_progress || stage_ctx.on_progress(stage_ctx.stage, import_progress, std::move(message)));
            };
            const auto filtered = take(detail::filter_stage_images(staging, staging / "sampled_images", "scan",
                                                                   filter_ctx, staging / "sampled_poses", request.write_diagnostic_json));
            const std::set<std::string> keep(filtered.selected_names.begin(), filtered.selected_names.end());
            std::erase_if(model.images, [&](const auto& item) { return !keep.contains(item.second.name); });
            std::set<uint32_t> used_cameras;
            for (const auto& [id, image] : model.images)
                used_cameras.insert(image.camera_id);
            std::erase_if(model.cameras, [&](const auto& item) { return !used_cameras.contains(item.first); });
            for (auto& frame : mapping["frames"])
                frame["blur_selected"] = keep.contains(frame.at("name").get<std::string>());
            mapping["blur_filter_report"] = "blur_filter_scan.json";
        }
        take(write_colmap_model(staging / "created" / "sparse", model, ModelFormat::Text));
        const auto& first = data.groups.front();
        std::ostringstream cfg;
        cfg.imbue(std::locale::classic());
        cfg << std::fixed << std::setprecision(6) << "fx " << first.fx << "\nfy " << first.fy
            << "\ncx " << first.cx << "\ncy " << first.cy << '\n';
        write_text(staging / "temp_cameras" / "0.cfg", cfg.str());
        if (request.write_diagnostic_json)
            write_text(staging / "sampled_indices.json", mapping.dump(2) + "\n");
        return model;
    } catch (const Failure& e) {
        return std::unexpected(e.error);
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }

    std::expected<PreprocessResult, Error> preprocess_scan(const PreprocessRequest& request,
                                                           ISuperResolutionProvider* provider,
                                                           const ExecutionContext& ctx) try {
        require(std::holds_alternative<ScanInput>(request.input), "Expected scan input", ErrorCode::InvalidRequest);
        const auto& options = std::get<ScanInput>(request.input).options;
        require(!options.sample_limit || *options.sample_limit > 0, "Scan sample limit must be positive", ErrorCode::InvalidRequest);
        const auto& sr = options.super_resolution;
        require(!sr.enabled || (provider && sr.model_scale > 0 && sr.final_scale > 0 && sr.final_scale <= sr.model_scale),
                "SR requires a provider and 0 < final_scale <= model_scale", ErrorCode::InvalidRequest);
        require(!request.scanner_bin.empty() && fs::is_regular_file(request.scanner_bin), "Scanner bin does not exist", ErrorCode::InvalidRequest);
        auto loaded = io::read_scanner_bin(request.scanner_bin);
        require(bool(loaded), loaded ? "" : loaded.error());
        const auto& data = *loaded;
        fs::path overlay_mesh_path;
        if (options.enable_mesh_overlay) {
            const auto resolved_mesh = io::resolve_scanner_mesh_path(request.scanner_bin, data);
            require(bool(resolved_mesh), resolved_mesh ? "" : resolved_mesh.error(), ErrorCode::InvalidRequest);
            overlay_mesh_path = *resolved_mesh;
        }
        struct Frame {
            size_t index;
            size_t group;
            const io::ScannerBinFrame* frame;
            fs::path source;
        };
        std::vector<Frame> available;
        std::vector<fs::path> inputs{fs::absolute(request.scanner_bin).parent_path()};
        size_t index = 0;
        for (size_t group = 0; group < data.groups.size(); ++group) {
            for (const auto& frame : data.groups[group].frames) {
                // Deliberately no decryption of bin-internal paths. Relative paths
                // retain Python's cwd semantics before the bin-relative fallback.
                auto source = fs::u8path(frame.image_path_utf8);
                std::error_code ec;
                if (!fs::is_regular_file(source, ec))
                    source = request.scanner_bin.parent_path() / (std::to_string(index) + ".png");
                ec.clear();
                if (fs::is_regular_file(source, ec)) {
                    source = fs::absolute(source);
                    available.push_back({index, group, &frame, source});
                    inputs.push_back(source);
                }
                ++index;
            }
        }
        require(!available.empty(), "No scanner images exist (including indexed PNG fallback)");
        std::sort(available.begin(), available.end(), [](const Frame& a, const Frame& b) {
            return std::to_string(a.index) + ".png" < std::to_string(b.index) + ".png";
        });
        const auto selection = uniform_sample(available.size(), options.sample_limit.value_or(available.size()));
        require(selection.size() < std::numeric_limits<uint32_t>::max(), "Too many scan frames");
        if (options.enable_mesh_overlay)
            inputs.push_back(fs::absolute(overlay_mesh_path));
        auto workspace = take(Workspace::create(request.workspace_root, inputs, request.retain_failed_stages,
                                                request.write_diagnostic_json));
        workspace->set_scan_parameters(options);
        ColmapModel model;
        std::vector<std::string> names;
        const bool process_images = options.enable_enhancement || options.enable_denoise;
        auto import_ctx = ctx, processing_ctx = ctx;
        if (process_images && ctx.on_progress) {
            import_ctx.on_progress = [&](Stage stage, float value, std::string message) {
                return ctx.on_progress(stage, value * 0.5F, std::move(message));
            };
            processing_ctx.on_progress = [&](Stage stage, float value, std::string message) {
                return ctx.on_progress(stage, 0.5F + value * 0.5F, std::move(message));
            };
        }
        const StageAction import_scan = [&](const fs::path& staging, const ExecutionContext& stage_ctx) {
            return guarded([&] {
                model = take(prepare_scan_source(request, staging, options.sample_limit, stage_ctx));
                for (const auto& [id, image] : model.images)
                    names.push_back(image.name);
            });
        };
        const StageValidator validate_scan = [&](const fs::path& staging) {
            return guarded([&] {
                const auto written = take(read_colmap_model(staging / "created" / "sparse"));
                validate_images(staging / "sampled_images", written, ctx);
            });
        };
        const auto scan = take(workspace->run_stage("scan", Stage::ImageProcessing, import_scan, validate_scan, import_ctx));
        if (options.enable_mesh_overlay) {
            const StageAction overlays = [&](const fs::path& staging, const ExecutionContext& stage_ctx) {
                return guarded([&] {
                    auto inner = stage_ctx;
                    inner.on_progress = [stage_ctx](Stage, float, std::string message) {
                        return !stage_ctx.stop_token.stop_requested() &&
                               (!stage_ctx.on_progress || stage_ctx.on_progress(stage_ctx.stage, 0, std::move(message)));
                    };
                    const auto mesh = take(read_mesh_geometry(overlay_mesh_path, 1.0, inner));
                    take(write_mesh_projection_overlays(mesh, model, scan / "sampled_images", staging, "scan_", inner,
                                                        request.write_diagnostic_json));
                });
            };
            const StageValidator valid_overlays = [](const fs::path&) -> std::expected<void, Error> { return {}; };
            take(workspace->run_stage("debug_mesh_projection", Stage::DebugExport, overlays, valid_overlays, ctx));
        }
        auto prepared_images = scan / "sampled_images";
        if (process_images) {
            const StageAction process_scan_images = [&](const fs::path& staging, const ExecutionContext& stage_ctx) {
                EnhancementOptions enhancement;
                if (!options.enable_enhancement)
                    enhancement.white_balance = enhancement.luma_tone = enhancement.contrast =
                        enhancement.saturation = enhancement.sharpen = false;
                return enhance_images(prepared_images, staging / "images", names, options.enable_denoise, enhancement, stage_ctx);
            };
            const StageValidator validate_processed = [&](const fs::path& staging) {
                return guarded([&] {
                    // Optional .denoised debug subdirectory belongs to this stage,
                    // but never gets copied to the final training image directory.
                    for (const auto& [id, image] : model.images) {
                        const auto dim = take(image_dimensions(staging / "images" / image.name));
                        const auto& cam = model.cameras.at(image.camera_id);
                        require(cam.width == uint64_t(dim[0]) && cam.height == uint64_t(dim[1]), "Enhanced dimensions changed");
                    }
                });
            };
            const auto processed = take(workspace->run_stage("scan_processed", Stage::ImageProcessing, process_scan_images, validate_processed, processing_ctx));
            prepared_images = processed / "images";
        }
        auto final_model = model;
        const StageAction assemble_scan = [&](const fs::path& staging, const ExecutionContext& stage_ctx) {
            return guarded([&] {
                if (sr.enabled) {
                    for (auto& [id, camera] : final_model.cameras) {
                        require(camera.width <= uint64_t(std::numeric_limits<int>::max() / sr.final_scale) &&
                                    camera.height <= uint64_t(std::numeric_limits<int>::max() / sr.final_scale),
                                "SR dimensions overflow");
                        camera = take(scale_camera(camera, camera.width * sr.final_scale, camera.height * sr.final_scale));
                    }
                    const auto result = take(provider->run({prepared_images, staging / "images", sr.model_name, sr.model_scale, sr.final_scale}, stage_ctx));
                    require(result.image_count == model.images.size(), "SR returned incorrect image count");
                    if (request.write_diagnostic_json && result.diagnostics)
                        write_text(workspace->run_root() / "sr_diagnostics.json", Json{{"exit_code", result.diagnostics->exit_code},
                                                                                       {"stdout_tail", result.diagnostics->stdout_tail},
                                                                                       {"stderr_tail", result.diagnostics->stderr_tail}}
                                                                                      .dump(2, ' ', false, Json::error_handler_t::replace));
                } else {
                    fs::create_directory(staging / "images");
                    for (size_t i = 0; i < names.size(); ++i) {
                        progress(stage_ctx, float(i) / names.size(), "Assembling scan dataset");
                        auto copy_ctx = stage_ctx;
                        copy_ctx.on_progress = {}; // outer loop owns monotonic progress
                        take(copy_image(prepared_images / names[i], staging / "images" / names[i], copy_ctx));
                    }
                }
                take(write_colmap_model(staging / "sparse", final_model, ModelFormat::Text));
            });
        };
        const StageValidator validate_scan_dataset = [&](const fs::path& staging) {
            return guarded([&] {
                validate_images(staging / "images", final_model, ctx);
                take(validate_dataset(staging, ctx));
            });
        };
        const auto dataset = take(workspace->run_stage("gs_input_scan", sr.enabled ? Stage::SuperResolution : Stage::Assemble,
                                                       assemble_scan, validate_scan_dataset, ctx));
        take(workspace->finish());
        return PreprocessResult{workspace->run_root(), dataset, workspace->run_root() / "preprocess_report.json",
                                model.images.size(), InputMode::Scan};
    } catch (const Failure& e) {
        return std::unexpected(e.error);
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
} // namespace lfs::preprocess
