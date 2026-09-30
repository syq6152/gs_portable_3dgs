/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/registered_pipeline.hpp"
#include "blur_filter_stage.hpp"
#include "io/scanner_bin.hpp"
#include "preprocessing/dataset_validation.hpp"
#include "preprocessing/debug_exports.hpp"
#include "preprocessing/foreground_mask.hpp"
#include "preprocessing/mesh_geometry.hpp"
#include "preprocessing/registered_twice.hpp"
#include "preprocessing/registration_triplets.hpp"
#include "preprocessing/scan_pipeline.hpp"
#include "preprocessing/video_frames.hpp"
#include "preprocessing/workspace.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;
        struct Failure : std::runtime_error {
            Error error;
            explicit Failure(Error e) : std::runtime_error(e.message), error(std::move(e)) {}
        };
        void require(bool value, const std::string& message, ErrorCode code = ErrorCode::InvalidDataset) {
            if (!value)
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
            } catch (const Failure& e) { return std::unexpected(e.error); } catch (const std::exception& e) {
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
            }
        }
        void text_file(const fs::path& path, const std::string& contents) {
            std::ofstream out(path, std::ios::binary | std::ios::noreplace);
            out << contents;
            out.close();
            require(bool(out), "Cannot write registered artifact: " + path_utf8(path), ErrorCode::IoFailure);
        }
        void cancelled(const ExecutionContext& ctx) {
            require(!ctx.stop_token.stop_requested(), "Registered preprocessing cancelled", ErrorCode::ProcessCancelled);
        }
        // Several operations share one owned stage. Keep its progress monotonic
        // without suppressing cancellation or callback failures from nested work.
        ExecutionContext nested(const ExecutionContext& ctx) {
            auto result = ctx;
            result.on_progress = [ctx](Stage, float, std::string message) {
                return !ctx.stop_token.stop_requested() && (!ctx.on_progress || ctx.on_progress(ctx.stage, 0, std::move(message)));
            };
            return result;
        }
        std::array<double, 3> center(const ColmapImage& image) {
            const auto r = quaternion_rotation(image.rotation);
            std::array<double, 3> c{};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    c[i] -= r[3 * j + i] * image.translation[j];
            return c;
        }
        double point_error_mean(const ColmapModel& model) {
            double sum = 0;
            size_t count = 0;
            if (model.points)
                for (const auto& [id, point] : *model.points)
                    if (std::isfinite(point.error)) {
                        sum += point.error;
                        ++count;
                    }
            return count ? sum / count : 0;
        }
        void validate_camera_split(const ColmapModel& model, const std::set<std::string>& incremental_names,
                                   bool shared_incremental, const std::string& stage) {
            std::set<uint32_t> scanner_cameras, incremental_cameras;
            for (const auto& [id, image] : model.images)
                (incremental_names.contains(image.name) ? incremental_cameras : scanner_cameras).insert(image.camera_id);
            for (uint32_t id : incremental_cameras)
                require(!scanner_cameras.contains(id), stage + " incremental camera overlaps scanner camera");
            require(!shared_incremental || incremental_cameras.size() <= 1,
                    stage + " changed shared incremental camera grouping");
        }
        std::pair<double, size_t> reprojection(const ColmapImage& image, const ColmapModel& model) {
            const auto& camera = model.cameras.at(image.camera_id);
            require(camera.model == 1, "Undistorted registered camera must be PINHOLE");
            auto r = quaternion_rotation(image.rotation);
            double sum = 0;
            size_t count = 0;
            if (model.points)
                for (const auto& obs : image.observations) {
                    if (obs.point_id < 0 || !model.points->contains(obs.point_id))
                        continue;
                    const auto& p = model.points->at(obs.point_id).position;
                    auto cam = image.translation;
                    for (size_t i = 0; i < 3; ++i)
                        for (size_t j = 0; j < 3; ++j)
                            cam[i] += r[3 * i + j] * p[j];
                    const double x = camera.parameters[0] * cam[0] / cam[2] + camera.parameters[2];
                    const double y = camera.parameters[1] * cam[1] / cam[2] + camera.parameters[3];
                    sum += std::hypot(x - obs.x, y - obs.y);
                    ++count;
                }
            return {count ? sum / count : std::numeric_limits<double>::infinity(), count};
        }
        void write_texture_metadata(const fs::path& destination, const fs::path& final_root, const ColmapModel& model) {
            io::ScannerBinData data;
            std::memcpy(data.magic.data(), "TEX_BIN_INC_V1.0", 15);
            data.header_version = 1;
            data.endian = io::ScannerBinEndian::Little;
            std::map<std::array<double, 4>, std::vector<io::ScannerBinFrame>> groups;
            for (const auto& [id, image] : model.images) {
                const auto& camera = model.cameras.at(image.camera_id);
                std::array<double, 4> key{};
                for (size_t i = 0; i < 4; ++i) {
                    std::ostringstream formatted;
                    formatted.imbue(std::locale::classic());
                    formatted << std::fixed << std::setprecision(6) << camera.parameters[i];
                    std::istringstream parsed(formatted.str());
                    parsed.imbue(std::locale::classic());
                    parsed >> key[i];
                }
                io::ScannerBinFrame frame;
                frame.image_path_utf8 = path_utf8(final_root / "images" / image.name);
                const auto r = quaternion_rotation(image.rotation);
                const auto c = center(image);
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col)
                        frame.pose_c2w[row * 4 + col] = float(r[col * 3 + row]);
                    frame.pose_c2w[row * 4 + 3] = float(c[row]);
                }
                frame.pose_c2w[15] = 1;
                groups[key].push_back(std::move(frame));
            }
            for (auto& [key, frames] : groups)
                data.groups.push_back({float(key[0]), float(key[1]), float(key[2]), float(key[3]), std::move(frames)});
            auto result = io::write_scanner_bin_atomic(destination, data);
            require(bool(result), result ? "" : result.error(), ErrorCode::IoFailure);
        }
    } // namespace

    std::expected<PreprocessResult, Error> preprocess_registered(const PreprocessRequest& request,
                                                                 IColmapProvider* provider, const ExecutionContext& ctx,
                                                                 IForegroundMaskProvider* foreground_override) try {
        require(std::holds_alternative<RegisteredInput>(request.input), "Expected registered input", ErrorCode::InvalidRequest);
        const auto& input = std::get<RegisteredInput>(request.input);
        const auto& options = input.options;
        require(bool(input.incremental_images) != bool(input.incremental_video),
                "Registered input requires exactly one images or video source", ErrorCode::InvalidRequest);
        require(!input.video_frame_count || (input.incremental_video && *input.video_frame_count > 0),
                "video_frame_count requires a video source and positive count", ErrorCode::InvalidRequest);
        require(options.matcher == "exhaustive" && options.registration_strategy == "colmap",
                "Only default exhaustive COLMAP registration is enabled", ErrorCode::UnsupportedFeature);
        require(provider && (input.incremental_video ? fs::is_regular_file(*input.incremental_video) : fs::is_directory(*input.incremental_images)),
                "Registered input requires an existing local images/video source and COLMAP provider", ErrorCode::InvalidRequest);
        require(!options.enable_incremental_mask || (options.foreground_model && fs::is_regular_file(*options.foreground_model)),
                "Incremental mask requires an existing explicit foreground model", ErrorCode::InvalidRequest);
        require(options.enable_incremental_mask || !options.foreground_model,
                "Foreground model requires incremental mask enabled", ErrorCode::InvalidRequest);
        require(std::isfinite(options.match_3d_threshold) && options.match_3d_threshold > 0 &&
                    (!options.sample_limit || *options.sample_limit > 0),
                "Invalid registered sampling/mesh threshold", ErrorCode::InvalidRequest);
        require(fs::is_regular_file(request.scanner_bin), "Scanner bin does not exist", ErrorCode::InvalidRequest);
        auto loaded = io::read_scanner_bin(request.scanner_bin);
        require(bool(loaded), loaded ? "" : loaded.error());
        const auto resolved_mesh = io::resolve_scanner_mesh_path(request.scanner_bin, *loaded);
        require(bool(resolved_mesh), resolved_mesh ? "" : resolved_mesh.error(), ErrorCode::InvalidRequest);
        const auto mesh_path = *resolved_mesh;
        std::vector<fs::path> inputs{fs::absolute(request.scanner_bin).parent_path(),
                                     fs::absolute(input.incremental_video ? *input.incremental_video : *input.incremental_images), fs::absolute(mesh_path)};
        if (options.foreground_model)
            inputs.push_back(fs::absolute(*options.foreground_model));
        size_t source_index = 0;
        for (const auto& group : loaded->groups)
            for (const auto& frame : group.frames) {
                auto path = fs::u8path(frame.image_path_utf8);
                if (!fs::is_regular_file(path))
                    path = request.scanner_bin.parent_path() / (std::to_string(source_index) + ".png");
                if (fs::is_regular_file(path))
                    inputs.push_back(fs::absolute(path));
                ++source_index;
            }
        cancelled(ctx);
        auto workspace = take(Workspace::create(request.workspace_root, inputs, request.retain_failed_stages,
                                                request.write_diagnostic_json));
        workspace->set_registered_parameters(input);
        Json diagnostics{{"schema_version", 1}, {"triangulation", options.use_mesh_triangulation ? "mesh" : "colmap"}, {"mesh_path", path_utf8(fs::absolute(mesh_path))}, {"mesh_coordinate_scale", 1.0}, {"register_twice", {{"enabled", options.register_twice}}}, {"calls", Json::array()}};
        auto run = [&](const ColmapRequest& command, const ExecutionContext& stage_ctx) {
            auto result = provider->run(command, nested(stage_ctx));
            Json record{{"operation", static_cast<int>(command.operation)}, {"database", path_utf8(command.database)}, {"input_model", path_utf8(command.input_model)}, {"output_model", path_utf8(command.output_model)}};
            record["images"] = path_utf8(command.images);
            record["image_list"] = path_utf8(command.image_list);
            record["image_masks"] = path_utf8(command.image_masks);
            record["num_threads"] = command.num_threads ? Json(*command.num_threads) : Json(nullptr);
            record["use_gpu"] = command.use_gpu ? Json(*command.use_gpu) : Json(nullptr);
            record["fix_existing_frames"] = command.fix_existing_frames ? Json(*command.fix_existing_frames) : Json(nullptr);
            record["single_camera"] = command.single_camera ? Json(*command.single_camera) : Json(nullptr);
            record["extraction_max_image_size"] = command.extraction_max_image_size ? Json(*command.extraction_max_image_size) : Json(nullptr);
            record["extraction_max_features"] = command.extraction_max_features ? Json(*command.extraction_max_features) : Json(nullptr);
            if (result && result->diagnostics) {
                record["exit_code"] = result->diagnostics->exit_code;
                record["stdout_tail"] = result->diagnostics->stdout_tail;
                record["stderr_tail"] = result->diagnostics->stderr_tail;
            } else if (!result) {
                record["exit_code"] = result.error().exit_code;
                record["error"] = result.error().message;
                record["stdout_tail"] = result.error().stdout_tail;
                record["stderr_tail"] = result.error().stderr_tail;
            }
            diagnostics["calls"].push_back(std::move(record));
            return result;
        };
        auto validate_model_at = [](const fs::path& stage) -> std::expected<void, Error> {
            auto model = read_colmap_model(stage / "sparse");
            if (!model)
                return std::unexpected(model.error());
            return {};
        };
        const int threads = int(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
        ColmapModel scan_model, seed_model, aligned_model;
        CpuMesh mesh;
        std::vector<ImageImportRecord> imported;
        std::set<std::string> incremental_names;
        bool single_incremental = false;
        fs::path incremental_source = input.incremental_images.value_or(fs::path{});
        VideoExtractionResult extracted_video;
        if (input.incremental_video) {
            const StageAction extract = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
                return guarded([&] {
                    extracted_video = take(extract_video_frames(*input.incremental_video, stage / "images",
                                                                {input.video_frame_count.value_or(80), 2048}, nested(stage_ctx)));
                    Json video{{"schema_version", 1}, {"source", path_utf8(fs::absolute(*input.incremental_video))}, {"backend", extracted_video.backend}, {"requested_frames", input.video_frame_count.value_or(80)}, {"total_frames", extracted_video.total_frames}, {"counted_frames", extracted_video.counted_frames}, {"sample_indices", extracted_video.sample_indices}, {"incomplete", extracted_video.incomplete}, {"frames", Json::array()}};
                    for (const auto& frame : extracted_video.frames)
                        video["frames"].push_back({{"name", frame.name}, {"source_index", frame.source_index}, {"width", frame.width}, {"height", frame.height}});
                    diagnostics["video"] = video;
                    if (request.write_diagnostic_json)
                        text_file(stage / "video_frames.json", video.dump(2));
                });
            };
            const auto video_stage = take(workspace->run_stage("video_frames", Stage::ImageProcessing, extract, [](const fs::path&) -> std::expected<void, Error> { return {}; }, ctx));
            incremental_source = video_stage / "images";
        }
        const StageAction prepared_action = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
            const auto work = [&] {
                auto inner = nested(stage_ctx);
                scan_model = take(prepare_scan_source(request, stage, options.sample_limit, inner));
                mesh = take(read_mesh_geometry(mesh_path, 1.0, inner));
                if (input.incremental_video) {
                    // Frozen video extraction already normalizes size. Preserve its
                    // video_*.png names and encoded bytes; image import would rename
                    // and potentially reorient/re-encode these extracted frames.
                    fs::create_directory(stage / "images_inc");
                    for (const auto& frame : extracted_video.frames) {
                        cancelled(stage_ctx);
                        const auto source = incremental_source / fs::u8path(frame.name);
                        take(copy_image(source, stage / "images_inc" / fs::u8path(frame.name), inner));
                        imported.push_back({source, frame.name, frame.width, frame.height, false});
                    }
                } else {
                    imported = take(import_images(incremental_source, stage / "images_inc", {}, inner));
                }
                require(!imported.empty(), "No supported incremental images");
                if (options.enable_blur_filter) {
                    const auto filtered = take(detail::filter_stage_images(stage, stage / "images_inc", "inc", inner,
                                                                           {}, request.write_diagnostic_json));
                    const std::set<std::string> keep(filtered.selected_names.begin(), filtered.selected_names.end());
                    std::erase_if(imported, [&](const auto& item) { return !keep.contains(item.name); });
                    diagnostics["incremental_blur_filter"] = {{"selected", filtered.selected_names}, {"removed", filtered.removed_names}};
                }
                std::set<std::array<int, 2>> sizes;
                Json mapping = Json::array();
                std::string incremental_list;
                for (const auto& image : imported) {
                    sizes.insert({image.width, image.height});
                    incremental_names.insert(image.name);
                    incremental_list += image.name + "\n";
                    mapping.push_back({{"source", path_utf8(image.source)}, {"name", image.name}, {"width", image.width}, {"height", image.height}, {"rotated", image.rotated}});
                    if (input.incremental_video) {
                        const auto frame = std::find_if(extracted_video.frames.begin(), extracted_video.frames.end(),
                                                        [&](const auto& item) { return fs::u8path(item.name) == image.source.filename(); });
                        require(frame != extracted_video.frames.end(), "Missing video source frame mapping");
                        mapping.back()["video_source"] = path_utf8(fs::absolute(*input.incremental_video));
                        mapping.back()["source_frame_index"] = frame->source_index;
                    }
                }
                single_incremental = sizes.size() == 1;
                if (request.write_diagnostic_json)
                    text_file(stage / "incremental_mapping.json", mapping.dump(2));
                text_file(stage / "new_images.txt", incremental_list);
                if (options.enable_incremental_mask) {
                    std::unique_ptr<IForegroundMaskProvider> owned_foreground;
                    auto* foreground = foreground_override;
                    if (!foreground) {
                        owned_foreground = take(create_u2netp_cpu_provider(*options.foreground_model, inner));
                        foreground = owned_foreground.get();
                    }
                    fs::create_directory(stage / "inc_masks");
                    Json masks{{"schema_version", 1}, {"model_sha256", u2netp_model_sha256}, {"provider", foreground_override ? "injected" : "CPUExecutionProvider"}, {"erosion_pixels", 3}, {"frames", Json::array()}};
                    for (const auto& image : imported) {
                        cancelled(stage_ctx);
                        const auto rgb = take(read_image(stage / "images_inc" / fs::u8path(image.name)));
                        const auto mask = take(foreground->mask(rgb, {}, inner));
                        require(mask.width == rgb.width && mask.height == rgb.height && mask.channels == 1 &&
                                    mask.pixels.size() == size_t(rgb.width) * size_t(rgb.height),
                                "Foreground mask dimensions/channels differ from imported image");
                        take(write_png(stage / "inc_masks" / fs::u8path(image.name + ".png"), mask));
                        masks["frames"].push_back({{"image", image.name}, {"mask", image.name + ".png"}, {"width", mask.width}, {"height", mask.height}});
                    }
                    diagnostics["foreground_masks"] = masks;
                    if (request.write_diagnostic_json)
                        text_file(stage / "foreground_masks.json", masks.dump(2));
                }
                fs::create_directory(stage / "scan_masks");
                std::set<std::string> scan_names;
                for (const auto& [id, image] : scan_model.images) {
                    cancelled(stage_ctx);
                    auto mask = take(project_mesh_mask(mesh, scan_model.cameras.at(image.camera_id), image, 3, inner));
                    take(write_png(stage / "scan_masks" / (image.name + ".png"), mask));
                    scan_names.insert(image.name);
                }
                std::string scan_list;
                for (const auto& name : scan_names)
                    scan_list += name + "\n";
                text_file(stage / "scan_images.txt", scan_list);
                diagnostics["input_image_count"] = imported.size();
                diagnostics["scan_image_count"] = scan_model.images.size();
                diagnostics["incremental_shared_camera"] = single_incremental;
            };
            return guarded(work);
        };
        const StageValidator prepared_validator = [](const fs::path&) -> std::expected<void, Error> { return {}; };
        const auto prepared = take(workspace->run_stage("prepared", Stage::ImageProcessing, prepared_action, prepared_validator, ctx));
        const StageAction features_action = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
            const auto work = [&] {
                ColmapRequest command;
                command.database = stage / "database.db";
                command.images = prepared / "sampled_images";
                command.image_list = prepared / "scan_images.txt";
                command.image_masks = prepared / "scan_masks";
                command.extraction_max_image_size = 1200;
                command.extraction_max_features = 5000;
                command.num_threads = threads;
                command.use_gpu = true;
                if (scan_model.cameras.size() == 1)
                    command.single_camera = true;
                take(run(command, stage_ctx));
            };
            return guarded(work);
        };
        const StageValidator features_validator = [](const fs::path& stage) -> std::expected<void, Error> {
if (!fs::is_regular_file(stage / "database.db")) return std::unexpected(Error{.code=ErrorCode::InvalidDataset,.message="COLMAP did not create scan database"});
return {}; };
        const auto features = take(workspace->run_stage("scan_features", Stage::Colmap, features_action, features_validator, ctx));
        const StageAction seed_action = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
            const auto work = [&] {
                auto db = take(ColmapDatabase::clone(features / "database.db", stage / "database.db"));
                const auto extracted = take(db->read_metadata());
                diagnostics["scan_database_tables"] = take(db->statistics());
                std::map<std::string, uint32_t> ids;
                for (const auto& [id, image] : extracted.images)
                    ids.emplace(image.name, id);
                require(ids.size() == scan_model.images.size(), "Scan feature database image count mismatch");
                ColmapModel authoritative;
                std::map<uint32_t, uint32_t> camera_ids;
                std::map<std::string, ColmapImage> by_name;
                for (const auto& [id, image] : scan_model.images)
                    by_name.emplace(image.name, image);
                for (auto& [name, image] : by_name) {
                    require(ids.contains(name), "Scan feature database omitted image " + name);
                    auto [camera, added] = camera_ids.emplace(image.camera_id, uint32_t(camera_ids.size() + 1));
                    if (added) {
                        auto c = scan_model.cameras.at(image.camera_id);
                        c.id = camera->second;
                        authoritative.cameras.emplace(c.id, c);
                    }
                    image.id = ids.at(name);
                    image.camera_id = camera->second;
                    authoritative.images.emplace(image.id, image);
                }
                scan_model = std::move(authoritative);
                take(db->synchronize_scan_metadata(scan_model));
                if (options.use_mesh_triangulation) {
                    std::map<uint32_t, Keypoints> keypoints;
                    size_t keypoint_count = 0;
                    for (const auto& [id, image] : scan_model.images) {
                        auto points = take(db->read_keypoints(id));
                        keypoint_count += points.data.size() / points.columns;
                        keypoints.emplace(id, std::move(points));
                    }
                    auto triangulated = take(triangulate_mesh(mesh, scan_model, keypoints, prepared / "sampled_images",
                                                              {.match_3d_threshold = options.match_3d_threshold}, nested(stage_ctx)));
                    for (const auto& pair : triangulated.pairs) {
                        cancelled(stage_ctx);
                        take(db->write_matches(pair.first, pair.second, pair.matches));
                        take(db->write_geometry(pair.first, pair.second, {.configuration = 2, .matches = pair.matches}));
                    }
                    diagnostics["scan_keypoint_count"] = keypoint_count;
                    diagnostics["mesh_hit_count"] = triangulated.hit_count;
                    diagnostics["seed_verified_pair_count"] = triangulated.pairs.size();
                    diagnostics["seed_dropped_images"] = triangulated.dropped_names;
                    seed_model = std::move(triangulated.model);
                    take(write_colmap_model(stage / "sparse", seed_model, ModelFormat::Text));
                } else {
                    db.reset();
                    take(write_colmap_model(stage / "input_sparse", scan_model, ModelFormat::Text));
                    text_file(stage / "input_sparse" / "points3D.txt", "");
                    ColmapRequest match;
                    match.operation = ColmapOperation::MatchExhaustive;
                    match.database = stage / "database.db";
                    match.guided_matching = true;
                    match.use_gpu = true;
                    match.num_threads = threads;
                    // This retry belongs only to the explicitly selected COLMAP
                    // seed branch. Cancellation/timeout/launch errors never retry.
                    for (int attempt = 0; attempt < 3; ++attempt) {
                        auto matched = run(match, stage_ctx);
                        if (matched)
                            break;
                        if (matched.error().code != ErrorCode::ProcessFailed || attempt == 2)
                            throw Failure(matched.error());
                    }
                    fs::create_directory(stage / "sparse");
                    ColmapRequest triangulate;
                    triangulate.operation = ColmapOperation::TriangulatePoints;
                    triangulate.database = stage / "database.db";
                    triangulate.images = prepared / "sampled_images";
                    triangulate.input_model = stage / "input_sparse";
                    triangulate.output_model = stage / "sparse";
                    take(run(triangulate, stage_ctx));
                    seed_model = take(read_colmap_model(stage / "sparse"));
                }
                require(seed_model.points && !seed_model.points->empty(), "Triangulation produced no seed points");
                diagnostics["seed_point_count"] = seed_model.points->size();
                diagnostics["seed_mean_reprojection_error"] = point_error_mean(seed_model);
                diagnostics["seed_image_count"] = seed_model.images.size();
                if (db)
                    diagnostics["seed_database_tables"] = take(db->statistics());
            };
            return guarded(work);
        };
        const StageValidator seed_validator = validate_model_at;
        const auto seed = take(workspace->run_stage("seed", Stage::Colmap, seed_action, seed_validator, ctx));
        // Free the potentially large acceleration input before COLMAP/GS allocations.
        mesh = {};
        const StageAction registered_action = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
            const auto work = [&] {
                { auto db = take(ColmapDatabase::clone(seed / "database.db", stage / "database.db")); }
                ColmapRequest extract;
                extract.database = stage / "database.db";
                extract.images = prepared / "images_inc";
                extract.image_list = prepared / "new_images.txt";
                if (options.enable_incremental_mask)
                    extract.image_masks = prepared / "inc_masks";
                extract.camera_model = "SIMPLE_RADIAL";
                if (single_incremental)
                    extract.single_camera = true;
                extract.extraction_max_image_size = 1200;
                extract.extraction_max_features = 5000;
                extract.num_threads = threads;
                extract.use_gpu = true;
                take(run(extract, stage_ctx));
                // Clone the now-updated private DB for read-only inspection; never open a user's DB writable.
                {
                    auto inspection = take(ColmapDatabase::clone(stage / "database.db", stage / "extraction_snapshot.db"));
                    const auto metadata = take(inspection->read_metadata());
                    std::set<uint32_t> inc_cameras;
                    size_t count = 0;
                    for (const auto& [id, image] : metadata.images)
                        if (incremental_names.contains(image.name)) {
                            ++count;
                            inc_cameras.insert(image.camera_id);
                            require(!scan_model.cameras.contains(image.camera_id), "Incremental camera overlaps authoritative scan camera");
                        }
                    require(count == imported.size() && (!single_incremental || inc_cameras.size() == 1), "Incremental database camera grouping mismatch");
                    diagnostics["incremental_camera_count"] = inc_cameras.size();
                }
                ColmapRequest match;
                match.operation = ColmapOperation::MatchExhaustive;
                match.database = stage / "database.db";
                match.guided_matching = true;
                match.num_threads = threads;
                match.use_gpu = true;
                take(run(match, stage_ctx));
                fs::create_directory(stage / "mapped");
                ColmapRequest mapper;
                mapper.operation = ColmapOperation::Map;
                mapper.database = stage / "database.db";
                mapper.images = prepared / "images_inc";
                mapper.input_model = seed / "sparse";
                mapper.output_model = stage / "mapped";
                mapper.fix_existing_frames = true;
                mapper.num_threads = threads;
                mapper.use_gpu = true;
                auto mapped = run(mapper, stage_ctx);
                fs::path mapped_path = mapper.output_model;
                if (!mapped) {
                    if (mapped.error().code != ErrorCode::ProcessFailed)
                        throw Failure(mapped.error());
                    diagnostics["registration_fallback"] = "mapper_failed_then_image_registrator";
                    // Keep failed mapper outputs for diagnosis; fallback gets its own empty directory.
                    fs::create_directory(stage / "fallback");
                    ColmapRequest fallback;
                    fallback.operation = ColmapOperation::RegisterImages;
                    fallback.database = mapper.database;
                    fallback.input_model = mapper.input_model;
                    fallback.output_model = stage / "fallback";
                    fallback.num_threads = threads;
                    take(run(fallback, stage_ctx));
                    mapped_path = fallback.output_model;
                    ColmapRequest ba;
                    ba.operation = ColmapOperation::BundleAdjust;
                    ba.input_model = mapped_path;
                    ba.output_model = mapped_path;
                    ba.use_gpu = true;
                    auto adjusted = run(ba, stage_ctx);
                    if (!adjusted && adjusted.error().code != ErrorCode::ProcessFailed)
                        throw Failure(adjusted.error());
                    if (!adjusted)
                        diagnostics["bundle_adjustment_warning"] = adjusted.error().message;
                }
                const auto mapped_model = take(read_colmap_model(mapped_path));
                validate_camera_split(mapped_model, incremental_names, single_incremental, "Mapper output");
                {
                    auto snapshot = take(ColmapDatabase::clone(stage / "database.db", stage / "registration_snapshot.db"));
                    diagnostics["database_tables"] = take(snapshot->statistics());
                }
                size_t count = 0;
                for (const auto& [id, image] : mapped_model.images)
                    if (incremental_names.contains(image.name))
                        ++count;
                require(count > 0, "COLMAP registered zero incremental images");
                diagnostics["registered_image_count"] = count;
                diagnostics["registered_rate"] = double(count) / imported.size();
                fs::path undistort_input = mapped_path;
                if (options.register_twice) {
                    const auto second_root = stage / "mapped_incremental_only";
                    fs::create_directory(second_root);
                    ColmapRequest second_mapper;
                    second_mapper.operation = ColmapOperation::Map;
                    second_mapper.database = stage / "database.db";
                    second_mapper.images = prepared / "images_inc";
                    second_mapper.image_list = prepared / "new_images.txt";
                    second_mapper.output_model = second_root;
                    second_mapper.num_threads = threads;
                    second_mapper.use_gpu = true;
                    take(run(second_mapper, stage_ctx));
                    cancelled(stage_ctx);

                    std::vector<fs::path> model_paths;
                    for (const auto& entry : fs::directory_iterator(second_root))
                        if (entry.is_directory())
                            model_paths.push_back(entry.path());
                    std::sort(model_paths.begin(), model_paths.end());
                    require(!model_paths.empty(), "Register-twice mapper produced no models");
                    std::vector<ColmapModel> second_models;
                    second_models.reserve(model_paths.size());
                    for (const auto& path : model_paths) {
                        auto model = take(read_colmap_model(path));
                        for (const auto& [id, image] : model.images) {
                            (void)id;
                            require(incremental_names.contains(image.name),
                                    "Register-twice mapper produced an image outside its allowlist");
                        }
                        second_models.push_back(std::move(model));
                        cancelled(stage_ctx);
                    }
                    auto merged = take(merge_register_twice_models(mapped_model, second_models));
                    const auto merged_path = stage / "mapped_merged";
                    take(write_colmap_model(merged_path, merged.model, ModelFormat::Text));
                    validate_camera_split(merged.model, incremental_names, single_incremental,
                                          "Register-twice merged output");
                    Json model_reports = Json::array();
                    for (std::size_t index = 0; index < merged.models.size(); ++index) {
                        const auto& report = merged.models[index];
                        model_reports.push_back({
                            {"model", path_utf8(model_paths[index].filename())},
                            {"common_image_count", report.common_image_count},
                            {"added_image_count", report.added_image_count},
                            {"scale", report.scale ? Json(*report.scale) : Json(nullptr)},
                            {"primary_first_image_id", report.primary_first_image_id ? Json(*report.primary_first_image_id) : Json(nullptr)},
                            {"primary_last_image_id", report.primary_last_image_id ? Json(*report.primary_last_image_id) : Json(nullptr)},
                            {"secondary_first_image_id", report.secondary_first_image_id ? Json(*report.secondary_first_image_id) : Json(nullptr)},
                            {"secondary_last_image_id", report.secondary_last_image_id ? Json(*report.secondary_last_image_id) : Json(nullptr)},
                        });
                    }
                    diagnostics["register_twice"]["model_count"] = second_models.size();
                    diagnostics["register_twice"]["added_image_count"] = merged.added_image_count;
                    diagnostics["register_twice"]["models"] = std::move(model_reports);
                    undistort_input = merged_path;
                }
                ColmapRequest undistort;
                undistort.operation = ColmapOperation::UndistortImages;
                undistort.images = prepared / "images_inc";
                undistort.input_model = undistort_input;
                undistort.output_model = stage / "undistorted";
                undistort.output_format = "COLMAP";
                take(run(undistort, stage_ctx));
                const auto undistorted = take(read_colmap_model(stage / "undistorted" / "sparse"));
                validate_camera_split(undistorted, incremental_names, single_incremental, "Undistorted output");
                std::ostringstream refs;
                refs.imbue(std::locale::classic());
                refs << std::setprecision(17);
                size_t references = 0;
                for (const auto& [id, image] : undistorted.images)
                    if (seed_model.images.contains(id)) {
                        const auto c = center(seed_model.images.at(id));
                        refs << image.name << ' ' << c[0] << ' ' << c[1] << ' ' << c[2] << '\n';
                        ++references;
                    }
                require(references >= 3, "Alignment requires at least three scanner reference centers");
                text_file(stage / "reference_images.txt", refs.str());
                fs::create_directory(stage / "sparse");
                ColmapRequest align;
                align.operation = ColmapOperation::AlignModel;
                align.input_model = stage / "undistorted" / "sparse";
                align.output_model = stage / "sparse";
                align.reference_images = stage / "reference_images.txt";
                align.references_are_gps = false;
                align.alignment_maximum_error = .1;
                take(run(align, stage_ctx));
                aligned_model = take(read_colmap_model(stage / "sparse"));
                diagnostics["mean_reprojection_error"] = point_error_mean(aligned_model);
                diagnostics["registered_point_count"] = aligned_model.points ? aligned_model.points->size() : 0;
                if (request.write_diagnostic_json)
                    text_file(stage / "calls.json", diagnostics["calls"].dump(2, ' ', false, Json::error_handler_t::replace));
            };
            return guarded(work);
        };
        const StageValidator registered_validator = validate_model_at;
        const auto registered = take(workspace->run_stage("registration", Stage::Colmap, registered_action, registered_validator, ctx));
        if (options.enable_registration_triplets) {
            // COLMAP may publish BIN only. Create a separate text representation
            // for diagnostics, never rewrite the authoritative registration stage.
            const auto text_model = take(workspace->run_stage("triplet_model", Stage::DebugExport, [&](const fs::path& stage, const ExecutionContext&) { return write_colmap_model(stage / "sparse", aligned_model, ModelFormat::Text); }, validate_model_at, ctx));
            const StageAction triplets = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
                return guarded([&] {
                    const auto result = take(write_registration_triplets(
                        {text_model / "sparse", registered / "undistorted" / "sparse", registered / "undistorted" / "images", registered / "undistorted" / "mask",
                         // Legacy scans only masked_images/*.png, not its mask/
                         // subdirectory. Active preprocessing writes no root PNGs.
                         prepared / "sampled_images",
                         {}},
                        stage, stage_ctx, request.write_diagnostic_json));
                    diagnostics["registration_triplets"] = {{"directory", path_utf8(workspace->run_root() / "registration_triplets")},
                                                            {"images", result.images},
                                                            {"masks", result.masks},
                                                            {"sparse_files", result.sparse_files}};
                });
            };
            take(workspace->run_stage("registration_triplets", Stage::DebugExport, triplets, prepared_validator, ctx));
        }
        if (options.enable_mesh_overlay) {
            const StageAction overlays = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
                return guarded([&] {
                    auto inner = nested(stage_ctx);
                    // The seed stage frees its mesh before COLMAP allocations.
                    const auto debug_mesh = take(read_mesh_geometry(mesh_path, 1.0, inner));
                    const auto scan_count = take(write_mesh_projection_overlays(debug_mesh, scan_model, prepared / "sampled_images",
                                                                                stage, "scan_", inner, request.write_diagnostic_json));
                    auto incremental = aligned_model;
                    std::erase_if(incremental.images, [&](const auto& item) { return !incremental_names.contains(item.second.name); });
                    const auto inc_count = take(write_mesh_projection_overlays(debug_mesh, incremental, registered / "undistorted" / "images",
                                                                               stage, "inc_", inner, request.write_diagnostic_json));
                    diagnostics["debug_mesh_projection"] = {{"directory", path_utf8(workspace->run_root() / "debug_mesh_projection")},
                                                            {"scan_images", scan_count},
                                                            {"incremental_images", inc_count}};
                });
            };
            take(workspace->run_stage("debug_mesh_projection", Stage::DebugExport, overlays, prepared_validator, ctx));
        }
        ColmapModel final_model;
        const StageAction assembled_action = [&](const fs::path& stage, const ExecutionContext& stage_ctx) {
            const auto work = [&] {
                std::map<uint32_t, std::pair<double, size_t>> errors;
                std::vector<double> finite;
                for (const auto& [id, image] : aligned_model.images)
                    if (incremental_names.contains(image.name)) {
                        auto error = reprojection(image, aligned_model);
                        errors.emplace(id, error);
                        if (std::isfinite(error.first))
                            finite.push_back(error.first);
                    }
                require(!errors.empty(), "Aligned model contains no incremental images");
                double threshold = 10;
                if (!finite.empty()) {
                    double mean = std::accumulate(finite.begin(), finite.end(), 0.) / finite.size(), variance = 0;
                    for (double value : finite)
                        variance += (value - mean) * (value - mean);
                    threshold = mean + 2 * std::sqrt(variance / finite.size());
                }
                std::vector<uint32_t> selected;
                for (const auto& [id, error] : errors)
                    if (error.first < threshold && error.second >= 50)
                        selected.push_back(id);
                if (selected.empty())
                    for (const auto& [id, error] : errors)
                        selected.push_back(id);
                fs::create_directory(stage / "images");
                Json frames = Json::array();
                for (uint32_t original : selected) {
                    cancelled(stage_ctx);
                    auto image = aligned_model.images.at(original);
                    const auto imported_image = std::find_if(imported.begin(), imported.end(), [&](const auto& item) { return item.name == image.name; });
                    require(imported_image != imported.end(), "Missing incremental source mapping");
                    const auto source_name = image.name;
                    image.id = uint32_t(final_model.images.size() + 1);
                    image.name = std::to_string(image.id) + ".png";
                    image.observations.clear();
                    take(copy_image(registered / "undistorted" / "images" / fs::u8path(source_name), stage / "images" / image.name, nested(stage_ctx)));
                    final_model.images.emplace(image.id, image);
                    final_model.cameras.emplace(image.camera_id, aligned_model.cameras.at(image.camera_id));
                    frames.push_back({{"source", path_utf8(imported_image->source)}, {"imported_name", source_name}, {"name", image.name}, {"original_image_id", original}, {"observation_count", errors.at(original).second}, {"mean_reprojection_error", std::isfinite(errors.at(original).first) ? Json(errors.at(original).first) : Json(nullptr)}});
                }
                take(write_colmap_model(stage / "sparse", final_model, ModelFormat::Text));
                write_texture_metadata(stage / "tex_data_inc.bin", workspace->run_root(), final_model);
                diagnostics["final_image_count"] = final_model.images.size();
                diagnostics["final_registered_rate"] = double(final_model.images.size()) / imported.size();
                diagnostics["image_error_threshold"] = threshold;
                diagnostics["frames"] = std::move(frames);
                if (request.write_diagnostic_json)
                    text_file(stage / "registration_diagnostics.json", diagnostics.dump(2, ' ', false, Json::error_handler_t::replace));
            };
            return guarded(work);
        };
        const StageValidator assembled_validator = [&](const fs::path& stage) { return guarded([&] { take(validate_dataset(stage, ctx)); }); };
        const auto assembled = take(workspace->run_stage("assembled", Stage::Assemble, assembled_action, assembled_validator, ctx));
        take(workspace->publish_dataset_root(assembled));
        return PreprocessResult{workspace->run_root(), workspace->run_root(), workspace->run_root() / "preprocess_report.json", final_model.images.size(), InputMode::Registered};
    } catch (const Failure& e) { return std::unexpected(e.error); } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
} // namespace lfs::preprocess
