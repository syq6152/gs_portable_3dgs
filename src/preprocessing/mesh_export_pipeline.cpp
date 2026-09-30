/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/mesh_export_pipeline.hpp"

#include "io/scanner_bin.hpp"
#include "preprocessing/colmap_database.hpp"
#include "preprocessing/image_pipeline.hpp"
#include "preprocessing/mesh_export.hpp"
#include "preprocessing/mesh_geometry.hpp"
#include "preprocessing/scan_pipeline.hpp"
#include "preprocessing/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <thread>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;

        struct Failure final : std::runtime_error {
            Error error;
            explicit Failure(Error value) : std::runtime_error(value.message), error(std::move(value)) {}
        };

        void require(bool condition, std::string message, ErrorCode code = ErrorCode::InvalidDataset) {
            if (!condition)
                throw Failure({.code = code, .message = std::move(message)});
        }

        template <class T>
        T take(std::expected<T, Error> value) {
            if (!value)
                throw Failure(value.error());
            if constexpr (!std::is_void_v<T>)
                return std::move(*value);
        }

        template <class Action>
        std::expected<void, Error> guarded(Action&& action) {
            try {
                action();
                return {};
            } catch (const Failure& error) {
                return std::unexpected(error.error);
            } catch (const std::exception& error) {
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = error.what()});
            }
        }

        void check_stop(const ExecutionContext& context) {
            require(!context.stop_token.stop_requested(), "Mesh export preprocessing cancelled", ErrorCode::ProcessCancelled);
        }

        ExecutionContext nested(const ExecutionContext& context) {
            auto result = context;
            result.on_progress = [context](Stage, float, std::string message) {
                return !context.stop_token.stop_requested() &&
                       (!context.on_progress || context.on_progress(context.stage, 0, std::move(message)));
            };
            return result;
        }

        void write_text(const fs::path& path, const std::string& contents) {
            std::ofstream output(path, std::ios::binary | std::ios::noreplace);
            output << contents;
            output.close();
            require(bool(output), "Cannot write mesh export artifact: " + path_utf8(path), ErrorCode::IoFailure);
        }

        fs::path scanner_mesh(const PreprocessRequest& request, const io::ScannerBinData& data) {
            const auto path = io::resolve_scanner_mesh_path(request.scanner_bin, data);
            require(bool(path), path ? "" : path.error(), ErrorCode::InvalidRequest);
            return *path;
        }

        void validate_images(const fs::path& root, const ColmapModel& model) {
            std::set<std::string> actual, expected;
            for (const auto& item : fs::directory_iterator(root))
                if (item.is_regular_file() && !item.is_symlink()) {
                    const auto name = item.path().filename().generic_u8string();
                    actual.emplace(name.begin(), name.end());
                }
            for (const auto& [id, image] : model.images) {
                expected.insert(image.name);
                const auto dimensions = take(image_dimensions(root / fs::u8path(image.name), false));
                const auto& camera = model.cameras.at(image.camera_id);
                require(camera.width == static_cast<uint64_t>(dimensions[0]) &&
                            camera.height == static_cast<uint64_t>(dimensions[1]),
                        "Mesh export image dimensions differ from camera: " + image.name);
            }
            require(actual == expected, "Mesh export image directory differs from model");
        }
    } // namespace

    std::expected<PreprocessResult, Error> preprocess_mesh_export(
        const PreprocessRequest& request, IColmapProvider* colmap,
        ISuperResolutionProvider* super_resolution, const ExecutionContext& context) try {
        require(std::holds_alternative<MeshExportInput>(request.input), "Expected mesh export input", ErrorCode::InvalidRequest);
        const auto& options = std::get<MeshExportInput>(request.input).options;
        require(colmap, "Mesh export requires a COLMAP provider", ErrorCode::InvalidRequest);
        require(!options.super_resolution.enabled || super_resolution,
                "Mesh super-resolution requires a provider", ErrorCode::InvalidRequest);
        require(std::isfinite(options.match_3d_threshold) && options.match_3d_threshold > 0,
                "Mesh export match threshold must be finite and positive", ErrorCode::InvalidRequest);
        require(options.fill_black_component_max_pixels >= 0,
                "Mesh export black-component threshold must be non-negative", ErrorCode::InvalidRequest);
        require(!options.super_resolution.enabled ||
                    (options.super_resolution.model_scale > 0 && options.super_resolution.final_scale > 0 &&
                     options.super_resolution.final_scale <= options.super_resolution.model_scale),
                "Mesh export SR requires 0 < final_scale <= model_scale", ErrorCode::InvalidRequest);
        require(fs::is_regular_file(request.scanner_bin), "Scanner bin does not exist", ErrorCode::InvalidRequest);
        const auto scanner = io::read_scanner_bin(request.scanner_bin);
        require(bool(scanner), scanner ? "" : scanner.error());
        const auto mesh_path = scanner_mesh(request, *scanner);

        std::vector<fs::path> inputs{fs::absolute(request.scanner_bin).parent_path(), mesh_path};
        size_t source_index = 0;
        for (const auto& group : scanner->groups)
            for (const auto& frame : group.frames) {
                auto path = fs::u8path(frame.image_path_utf8);
                if (!fs::is_regular_file(path))
                    path = request.scanner_bin.parent_path() / (std::to_string(source_index) + ".png");
                if (fs::is_regular_file(path))
                    inputs.push_back(fs::absolute(path));
                ++source_index;
            }
        check_stop(context);
        auto workspace = take(Workspace::create(request.workspace_root, inputs, request.retain_failed_stages,
                                                request.write_diagnostic_json));
        workspace->set_mesh_export_parameters(options);

        ColmapModel scan_model;
        CpuMesh mesh;
        fs::path image_relative = "sampled_images";
        const StageAction prepare = [&](const fs::path& stage, const ExecutionContext& stage_context) {
            return guarded([&] {
                auto inner = nested(stage_context);
                scan_model = take(prepare_scan_source(request, stage, std::nullopt, inner));
                mesh = take(read_mesh_geometry(mesh_path, 1.0, inner));
                if (options.super_resolution.enabled) {
                    const auto root = stage / "mesh_points3d_superresolution";
                    fs::create_directory(root);
                    take(prepare_mesh_superresolution_images(
                        stage / "sampled_images", root, options.super_resolution, super_resolution,
                        options.enhance_before_super_resolution, inner));
                    for (auto& [id, camera] : scan_model.cameras) {
                        require(camera.width <= uint64_t((std::numeric_limits<int>::max)() / options.super_resolution.final_scale) &&
                                    camera.height <= uint64_t((std::numeric_limits<int>::max)() / options.super_resolution.final_scale),
                                "Mesh super-resolution dimensions overflow");
                        camera = take(scale_camera(camera, camera.width * options.super_resolution.final_scale,
                                                   camera.height * options.super_resolution.final_scale));
                    }
                    image_relative = "mesh_points3d_superresolution/images";
                }
                fs::create_directory(stage / "scan_masks");
                std::string image_list;
                for (const auto& [id, image] : scan_model.images) {
                    check_stop(stage_context);
                    const auto mask = take(project_mesh_mask(mesh, scan_model.cameras.at(image.camera_id), image, 3, inner));
                    take(write_png(stage / "scan_masks" / fs::u8path(image.name + ".png"), mask));
                    image_list += image.name + "\n";
                }
                write_text(stage / "scan_images.txt", image_list);
            });
        };
        const StageValidator validate_prepare = [&](const fs::path& stage) {
            return guarded([&] { validate_images(stage / image_relative, scan_model); });
        };
        const auto prepared = take(workspace->run_stage("prepared", Stage::ImageProcessing,
                                                        prepare, validate_prepare, context));

        const int threads = int(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
        const StageAction features = [&](const fs::path& stage, const ExecutionContext& stage_context) {
            ColmapRequest request;
            request.operation = ColmapOperation::ExtractFeatures;
            request.database = stage / "database.db";
            request.images = prepared / image_relative;
            request.image_list = prepared / "scan_images.txt";
            request.image_masks = prepared / "scan_masks";
            request.extraction_max_image_size = 1200;
            request.extraction_max_features = 5000;
            request.num_threads = threads;
            request.use_gpu = true;
            if (scan_model.cameras.size() == 1)
                request.single_camera = true;
            return colmap->run(request, stage_context).transform([](const auto&) {});
        };
        const StageValidator validate_features = [](const fs::path& stage) -> std::expected<void, Error> {
            if (!fs::is_regular_file(stage / "database.db"))
                return std::unexpected(Error{.code = ErrorCode::InvalidDataset,
                                             .message = "COLMAP did not create mesh export feature database"});
            return {};
        };
        const auto feature_stage = take(workspace->run_stage("scan_features", Stage::Colmap,
                                                             features, validate_features, context));

        ColmapModel triangulated_model;
        const StageAction triangulate = [&](const fs::path& stage, const ExecutionContext& stage_context) {
            return guarded([&] {
                auto database = take(ColmapDatabase::clone(feature_stage / "database.db", stage / "database.db"));
                const auto extracted = take(database->read_metadata());
                std::map<std::string, uint32_t> database_ids;
                for (const auto& [id, image] : extracted.images)
                    database_ids.emplace(image.name, id);
                require(database_ids.size() == scan_model.images.size(), "Mesh export feature database image count mismatch");
                ColmapModel authoritative;
                std::map<uint32_t, uint32_t> camera_ids;
                std::map<std::string, ColmapImage> by_name;
                for (const auto& [id, image] : scan_model.images)
                    by_name.emplace(image.name, image);
                for (auto& [name, image] : by_name) {
                    require(database_ids.contains(name), "Mesh export feature database omitted image " + name);
                    auto [camera_id, inserted] = camera_ids.emplace(image.camera_id, uint32_t(camera_ids.size() + 1));
                    if (inserted) {
                        auto camera = scan_model.cameras.at(image.camera_id);
                        camera.id = camera_id->second;
                        authoritative.cameras.emplace(camera.id, camera);
                    }
                    image.id = database_ids.at(name);
                    image.camera_id = camera_id->second;
                    authoritative.images.emplace(image.id, image);
                }
                scan_model = std::move(authoritative);
                take(database->synchronize_scan_metadata(scan_model));
                std::map<uint32_t, Keypoints> keypoints;
                for (const auto& [id, image] : scan_model.images)
                    keypoints.emplace(id, take(database->read_keypoints(id)));
                auto result = take(triangulate_mesh(
                    mesh, scan_model, keypoints, prepared / image_relative,
                    {.match_3d_threshold = options.match_3d_threshold}, nested(stage_context)));
                for (const auto& pair : result.pairs) {
                    check_stop(stage_context);
                    take(database->write_matches(pair.first, pair.second, pair.matches));
                    take(database->write_geometry(pair.first, pair.second,
                                                  {.configuration = 2, .matches = pair.matches}));
                }
                triangulated_model = std::move(result.model);
                require(triangulated_model.points && !triangulated_model.points->empty(),
                        "Mesh export triangulation produced no points");
                take(write_colmap_model(stage / "sparse", triangulated_model, ModelFormat::Text));
            });
        };
        const StageValidator validate_triangulated = [](const fs::path& stage) -> std::expected<void, Error> {
            auto model = read_colmap_model(stage / "sparse");
            if (!model)
                return std::unexpected(model.error());
            if (!model->points || model->points->empty())
                return std::unexpected(Error{.code = ErrorCode::InvalidDataset,
                                             .message = "Mesh export triangulation has no points"});
            return {};
        };
        take(workspace->run_stage("triangulated", Stage::Colmap,
                                  triangulate, validate_triangulated, context));

        MeshExportResult export_result;
        const StageAction export_action = [&](const fs::path& stage, const ExecutionContext& stage_context) {
            return guarded([&] {
                export_result = take(write_mesh_triangulation_export(
                    prepared / image_relative, scan_model, triangulated_model, mesh, stage,
                    options.fill_black_component_max_pixels, stage_context));
            });
        };
        const StageValidator validate_export = [&](const fs::path& stage) {
            return guarded([&] {
                const auto model = take(read_colmap_model(stage / "sparse"));
                require(model == triangulated_model, "Published mesh export sparse model differs");
                require(export_result.images == scan_model.images.size() &&
                            export_result.masks == export_result.images && export_result.sparse_files == 3,
                        "Published mesh export counts differ from contract");
                validate_images(stage / "images", scan_model);
            });
        };
        const auto exported = take(workspace->run_stage("mesh_export", Stage::DebugExport,
                                                        export_action, validate_export, context));
        take(workspace->finish());
        return PreprocessResult{workspace->run_root(), exported,
                                workspace->run_root() / "preprocess_report.json",
                                scan_model.images.size(), InputMode::MeshExport, exported};
    } catch (const Failure& error) {
        return std::unexpected(error.error);
    } catch (const std::exception& error) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = error.what()});
    }

} // namespace lfs::preprocess
