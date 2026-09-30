/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/mesh_export.hpp"

#include "preprocessing/image_pipeline.hpp"
#include "preprocessing/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <set>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;

        struct Failure final : std::runtime_error {
            Error error;
            explicit Failure(Error value) : std::runtime_error(value.message), error(std::move(value)) {}
        };

        [[noreturn]] void fail(ErrorCode code, std::string message) {
            throw Failure({.code = code, .message = std::move(message)});
        }

        template <class T>
        T take(std::expected<T, Error> value) {
            if (!value)
                throw Failure(value.error());
            if constexpr (!std::is_void_v<T>)
                return std::move(*value);
        }

        void check_stop(const ExecutionContext& context) {
            if (context.stop_token.stop_requested())
                fail(ErrorCode::ProcessCancelled, "Mesh export cancelled");
        }

        void emit(const ExecutionContext& context, float value, std::string message) {
            check_stop(context);
            if (context.on_progress) {
                try {
                    if (!context.on_progress(context.stage, value, std::move(message)))
                        fail(ErrorCode::ProcessCancelled, "Mesh export progress callback cancelled");
                } catch (const Failure&) {
                    throw;
                } catch (...) {
                    fail(ErrorCode::CallbackFailure, "Mesh export progress callback threw");
                }
            }
            check_stop(context);
        }

        void require_owned_empty(const fs::path& output, const std::vector<fs::path>& inputs) {
            if (output.empty() || !fs::is_directory(output) || fs::is_symlink(fs::symlink_status(output)) || !fs::is_empty(output))
                fail(ErrorCode::InvalidRequest, "Mesh export requires an existing empty owned output directory");
            take(check_output_isolation(output, inputs));
        }

        fs::path contained_name(const std::string& name) {
            const auto path = fs::u8path(name);
            if (name.empty() || path.is_absolute() || path.has_parent_path() || path.filename() != path || name == "." || name == "..")
                fail(ErrorCode::InvalidDataset, "Unsafe mesh export image name: " + name);
            return path;
        }

        std::set<std::string> png_names(const fs::path& root) {
            if (!fs::is_directory(root))
                fail(ErrorCode::InvalidRequest, "Mesh export image directory is missing");
            std::set<std::string> names;
            for (const auto& item : fs::directory_iterator(root)) {
                if (!item.is_regular_file() || item.is_symlink())
                    continue;
                auto extension = item.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(),
                               [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                if (extension != ".png")
                    continue;
                const auto native = item.path().filename().generic_u8string();
                const std::string name(native.begin(), native.end());
                if (!names.insert(name).second)
                    fail(ErrorCode::InvalidDataset, "Duplicate mesh export image name: " + name);
            }
            if (names.empty())
                fail(ErrorCode::InvalidDataset, "No PNG scan images found for mesh export");
            return names;
        }

        int round_even(double value) {
            const auto lower = static_cast<int64_t>(std::floor(value));
            const double fraction = value - static_cast<double>(lower);
            return static_cast<int>(lower + (fraction > .5 || (fraction == .5 && (lower & 1)) ? 1 : 0));
        }

        CpuImage export_mask(const CpuMesh& mesh, const ColmapCamera& camera, const ColmapImage& pose,
                             int fill_black_component_max_pixels, const ExecutionContext& context) {
            if (camera.model != 1 || camera.parameters.size() != 4 || camera.width == 0 || camera.height == 0 ||
                camera.width > static_cast<uint64_t>((std::numeric_limits<int>::max)()) ||
                camera.height > static_cast<uint64_t>((std::numeric_limits<int>::max)()))
                fail(ErrorCode::InvalidDataset, "Mesh export requires finite PINHOLE cameras");
            if (fill_black_component_max_pixels < 0)
                fail(ErrorCode::InvalidRequest, "Mesh export black-component threshold must be non-negative");

            const int width = static_cast<int>(camera.width);
            const int height = static_cast<int>(camera.height);
            CpuImage result{width, height, 1, std::vector<uint8_t>(static_cast<size_t>(width) * height, 0)};
            cv::Mat mask(height, width, CV_8UC1, result.pixels.data());
            const auto rotation = quaternion_rotation(pose.rotation);
            struct Projected {
                double x = 0;
                double y = 0;
                bool valid = false;
            };
            std::vector<Projected> projected(mesh.vertices.size());
            for (size_t index = 0; index < mesh.vertices.size(); ++index) {
                if ((index & 4095) == 0)
                    check_stop(context);
                const auto& point = mesh.vertices[index];
                std::array<double, 3> camera_point = pose.translation;
                for (size_t row = 0; row < 3; ++row)
                    for (size_t column = 0; column < 3; ++column)
                        camera_point[row] += rotation[row * 3 + column] * point[column];
                if (!(camera_point[2] > 0))
                    continue;
                const double x = camera.parameters[0] * (camera_point[0] / camera_point[2]) + camera.parameters[2];
                const double y = camera.parameters[1] * (camera_point[1] / camera_point[2]) + camera.parameters[3];
                if (std::isfinite(x) && std::isfinite(y))
                    projected[index] = {x, y, true};
            }

            const double coordinate_limit = 4.0 * std::max({width, height, 1});
            std::vector<std::vector<cv::Point>> polygons;
            polygons.reserve(std::min<size_t>(mesh.triangles.size(), 4096));
            bool filled_triangles = false;
            auto flush = [&] {
                if (polygons.empty())
                    return;
                cv::fillPoly(mask, polygons, cv::Scalar(255));
                polygons.clear();
                filled_triangles = true;
            };
            for (size_t index = 0; index < mesh.triangles.size(); ++index) {
                if ((index & 4095) == 0)
                    check_stop(context);
                const auto& triangle = mesh.triangles[index];
                if (triangle[0] >= projected.size() || triangle[1] >= projected.size() || triangle[2] >= projected.size())
                    fail(ErrorCode::InvalidDataset, "Mesh export triangle index is out of range");
                const auto& a = projected[triangle[0]];
                const auto& b = projected[triangle[1]];
                const auto& c = projected[triangle[2]];
                if (!a.valid || !b.valid || !c.valid)
                    continue;
                const double minimum_x = std::min({a.x, b.x, c.x});
                const double maximum_x = std::max({a.x, b.x, c.x});
                const double minimum_y = std::min({a.y, b.y, c.y});
                const double maximum_y = std::max({a.y, b.y, c.y});
                if (maximum_x < 0 || maximum_y < 0 || minimum_x >= width || minimum_y >= height)
                    continue;
                const double doubled_area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                if (std::abs(doubled_area) < .5)
                    continue;
                std::vector<cv::Point> polygon;
                polygon.reserve(3);
                for (const auto* point : {&a, &b, &c})
                    polygon.emplace_back(round_even(std::clamp(point->x, -coordinate_limit, coordinate_limit)),
                                         round_even(std::clamp(point->y, -coordinate_limit, coordinate_limit)));
                polygons.push_back(std::move(polygon));
                if (polygons.size() == 4096)
                    flush();
            }
            flush();

            if (!filled_triangles) {
                for (const auto& point : projected)
                    if (point.valid && point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
                        const int x = std::clamp(round_even(point.x), 0, width - 1);
                        const int y = std::clamp(round_even(point.y), 0, height - 1);
                        result.pixels[static_cast<size_t>(y) * width + x] = 255;
                    }
            }

            if (fill_black_component_max_pixels > 0) {
                cv::Mat black;
                cv::compare(mask, 0, black, cv::CMP_EQ);
                cv::Mat labels, stats, centers;
                const int count = cv::connectedComponentsWithStats(black, labels, stats, centers, 8, CV_32S);
                for (int label = 1; label < count; ++label) {
                    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                    const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
                    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
                    const int component_width = stats.at<int>(label, cv::CC_STAT_WIDTH);
                    const int component_height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
                    if (area < fill_black_component_max_pixels && left != 0 && top != 0 &&
                        left + component_width < width && top + component_height < height)
                        mask.setTo(255, labels == label);
                }
            }
            check_stop(context);
            return result;
        }

        Error exception_error(const std::exception& error) {
            return {.code = ErrorCode::IoFailure, .message = error.what()};
        }
    } // namespace

    std::expected<MeshExportResult, Error> write_mesh_triangulation_export(
        const fs::path& images, const ColmapModel& scan_model, const ColmapModel& triangulated,
        const CpuMesh& mesh, const fs::path& output, int fill_black_component_max_pixels,
        const ExecutionContext& context) try {
        require_owned_empty(output, {images});
        take(validate_model(scan_model));
        if (!triangulated.points || triangulated.points->empty())
            fail(ErrorCode::InvalidDataset, "Mesh triangulation export requires nonempty points3D");
        take(validate_model(triangulated));
        if (mesh.vertices.empty() || mesh.triangles.empty())
            fail(ErrorCode::InvalidDataset, "Mesh triangulation export requires triangle geometry");
        if (fill_black_component_max_pixels < 0)
            fail(ErrorCode::InvalidRequest, "Mesh export black-component threshold must be non-negative");
        const auto names = png_names(images);
        std::set<std::string> scan_names;
        for (const auto& [id, image] : scan_model.images) {
            contained_name(image.name);
            scan_names.insert(image.name);
        }
        if (names != scan_names)
            fail(ErrorCode::InvalidDataset, "Mesh export image set differs from prepared scan model");
        for (const auto& [id, image] : triangulated.images)
            if (!scan_names.contains(image.name))
                fail(ErrorCode::InvalidDataset, "Triangulated sparse model contains a non-scan image: " + image.name);
        emit(context, 0, "Starting mesh triangulation export");
        fs::create_directory(output / "images");
        fs::create_directory(output / "mask");
        for (size_t index = 0; const auto& [id, image] : scan_model.images) {
            check_stop(context);
            const auto name = contained_name(image.name);
            const auto source = images / name;
            const auto dimensions = take(image_dimensions(source, false));
            const auto& camera = scan_model.cameras.at(image.camera_id);
            if (camera.width != static_cast<uint64_t>(dimensions[0]) || camera.height != static_cast<uint64_t>(dimensions[1]))
                fail(ErrorCode::InvalidDataset, "Mesh export image dimensions differ from camera: " + image.name);
            fs::copy_file(source, output / "images" / name, fs::copy_options::none);
            auto mask = export_mask(mesh, camera, image, fill_black_component_max_pixels, context);
            take(write_png(output / "mask" / name, mask));
            ++index;
            emit(context, .9F * static_cast<float>(index) / scan_model.images.size(), "Exporting mesh image " + image.name);
        }
        take(write_colmap_model(output / "sparse", triangulated, ModelFormat::Text));
        emit(context, 1, "Mesh triangulation export complete");
        return MeshExportResult{fs::absolute(output), scan_names.size(), scan_names.size(), 3};
    } catch (const Failure& error) {
        return std::unexpected(error.error);
    } catch (const std::exception& error) {
        return std::unexpected(exception_error(error));
    }

    std::expected<MeshSuperResolutionResult, Error> prepare_mesh_superresolution_images(
        const fs::path& input, const fs::path& output, const SuperResolutionOptions& options,
        ISuperResolutionProvider* provider, bool enhance_before_super_resolution,
        const ExecutionContext& context) try {
        require_owned_empty(output, {input});
        if (!options.enabled || !provider || options.model_name.empty() || options.model_scale <= 0 ||
            options.final_scale <= 0 || options.final_scale > options.model_scale)
            fail(ErrorCode::InvalidRequest, "Invalid mesh super-resolution request");
        const auto names = png_names(input);
        std::vector<std::string> ordered(names.begin(), names.end());
        emit(context, 0, "Starting mesh super-resolution export");
        fs::path provider_input = input;
        if (enhance_before_super_resolution) {
            auto enhance_context = context;
            enhance_context.on_progress = [&](Stage, float fraction, std::string message) {
                emit(context, .45F * fraction, std::move(message));
                return true;
            };
            take(enhance_images(input, output / "_enhanced_scan", ordered, false, {}, enhance_context));
            provider_input = output / "_enhanced_scan";
        }
        auto provider_context = context;
        const float provider_start = enhance_before_super_resolution ? .45F : 0.F;
        const float provider_span = .45F + (enhance_before_super_resolution ? 0.F : .45F);
        provider_context.on_progress = [&](Stage, float fraction, std::string message) {
            emit(context, provider_start + provider_span * fraction, std::move(message));
            return true;
        };
        auto provider_result = take(provider->run(
            {provider_input, output / "images", options.model_name, options.model_scale, options.final_scale}, provider_context));
        if (provider_result.image_count != names.size())
            fail(ErrorCode::InvalidDataset, "Mesh super-resolution returned incorrect image count");
        const auto actual = png_names(output / "images");
        if (actual != names)
            fail(ErrorCode::InvalidDataset, "Mesh super-resolution images do not match scan input");
        size_t index = 0;
        for (const auto& name : names) {
            check_stop(context);
            const auto before = take(image_dimensions(input / fs::u8path(name), false));
            const auto after = take(image_dimensions(output / "images" / fs::u8path(name), false));
            if (after[0] != before[0] * options.final_scale || after[1] != before[1] * options.final_scale)
                fail(ErrorCode::InvalidDataset, "Mesh super-resolution dimensions differ from final scale: " + name);
            emit(context, .9F + .1F * static_cast<float>(++index) / names.size(), "Validating mesh super-resolution image " + name);
        }
        return MeshSuperResolutionResult{fs::absolute(output), fs::absolute(output / "images"), names.size(),
                                         enhance_before_super_resolution, std::move(provider_result.diagnostics)};
    } catch (const Failure& error) {
        return std::unexpected(error.error);
    } catch (const std::exception& error) {
        return std::unexpected(exception_error(error));
    }

} // namespace lfs::preprocess
