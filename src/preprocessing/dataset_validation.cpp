/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/dataset_validation.hpp"
#include "preprocessing/image_pipeline.hpp"
#include <algorithm>
#include <cctype>
#include <set>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        void check(bool condition, const char* message) {
            if (!condition)
                throw std::runtime_error(message);
        }
        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return s;
        }
        bool inside(const fs::path& path, const fs::path& parent) {
            auto p = path.begin();
            for (auto q = parent.begin(); q != parent.end(); ++p, ++q)
                if (p == path.end() || *p != *q)
                    return false;
            return true;
        }
    } // namespace
    std::expected<DatasetValidation, Error> validate_dataset(const fs::path& root, const ExecutionContext& ctx) try {
        check(fs::is_directory(root / "images"), "Dataset images directory missing");
        const auto image_root = fs::canonical(root / "images");
        auto sparse = root / "sparse";
        if (fs::is_directory(sparse / "0"))
            sparse /= "0";
        auto model = read_colmap_model(sparse);
        if (!model)
            return std::unexpected(model.error());
        // A stray malformed points file must not be hidden by the loader's binary preference.
        if (fs::exists(sparse / "points3D.bin") && fs::exists(sparse / "points3D.txt"))
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset,
                                         .message = "Ambiguous mixed points3D formats; publish a single model format"});
        const bool binary = fs::exists(sparse / "cameras.bin");
        check(!fs::exists(sparse / (binary ? "points3D.txt" : "points3D.bin")),
              "points3D format does not match cameras/images");
        std::set<fs::path> resolved;
        std::set<std::string> names;
        size_t count = 0;
        for (const auto& [id, image] : model->images) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(
                    Error{.code = ErrorCode::ProcessCancelled, .message = "Dataset validation cancelled"});
            try {
                if (ctx.on_progress &&
                    !ctx.on_progress(Stage::Validate, float(count) / model->images.size(), "Validating dataset"))
                    return std::unexpected(
                        Error{.code = ErrorCode::ProcessCancelled, .message = "Dataset validation callback cancelled"});
            } catch (...) {
                return std::unexpected(
                    Error{.code = ErrorCode::CallbackFailure, .message = "Dataset validation callback threw"});
            }
            auto relative = fs::u8path(image.name);
            const auto extension = lower(relative.extension().string());
            check(std::set<std::string>{".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}.contains(extension),
                  "Unsupported training image extension");
            check(!relative.empty() && !relative.is_absolute() && !relative.has_root_name() &&
                      image.name.find(':') == std::string::npos && image.name.find('\0') == std::string::npos,
                  "Invalid model image path");
            for (const auto& component : relative)
                check(component != "..", "Image path escapes dataset");
            check(names.insert(lower(relative.lexically_normal().generic_string())).second,
                  "Case-ambiguous model image names");
            const auto path = fs::canonical(image_root / relative);
            check(inside(path, image_root) && resolved.insert(path).second,
                  "Image alias escapes dataset or maps twice");
            auto decoded = read_image(path, false);
            if (!decoded)
                return std::unexpected(decoded.error());
            const auto& camera = model->cameras.at(image.camera_id);
            check(uint64_t(decoded->width) == camera.width && uint64_t(decoded->height) == camera.height,
                  "Camera dimensions do not match decoded image");
            ++count;
        }
        try {
            if (ctx.on_progress && !ctx.on_progress(Stage::Validate, 1, "Dataset valid"))
                return std::unexpected(
                    Error{.code = ErrorCode::ProcessCancelled, .message = "Dataset validation callback cancelled"});
        } catch (...) {
            return std::unexpected(
                Error{.code = ErrorCode::CallbackFailure, .message = "Dataset validation callback threw"});
        }
        return DatasetValidation{count, model->cameras.size(), model->points.has_value()};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = e.what()});
    }
    std::expected<fs::path, Error> validate_mesh_input(const fs::path& path) try {
        if (fs::is_regular_file(path)) {
            auto ext = lower(path.extension().string());
            check(ext == ".obj" || ext == ".ply", "Mesh file must be OBJ or PLY");
            check(fs::file_size(path) > 0, "Mesh file is empty");
            return path;
        }
        check(fs::is_directory(path), "Mesh path missing or not a file/directory");
        std::vector<fs::path> obj, ply, mtl, png;
        for (const auto& item : fs::directory_iterator(path))
            if (item.is_regular_file()) {
                const auto ext = lower(item.path().extension().string());
                if (ext == ".obj")
                    obj.push_back(item.path());
                else if (ext == ".ply")
                    ply.push_back(item.path());
                else if (ext == ".mtl")
                    mtl.push_back(item.path());
                else if (ext == ".png")
                    png.push_back(item.path());
            }
        check(png.size() == 1, "Mesh directory requires exactly one PNG texture");
        const bool have_obj = obj.size() == 1 && mtl.size() == 1, have_ply = ply.size() == 1;
        check(!(have_obj && have_ply), "Ambiguous mesh directory: both OBJ/MTL and PLY sets");
        check(have_obj || have_ply, "Mesh directory requires OBJ+MTL+PNG or PLY+PNG");
        const auto mesh = have_obj ? obj.front() : ply.front();
        check(fs::file_size(mesh) > 0 && fs::file_size(png.front()) > 0, "Empty mesh/texture asset");
        if (have_obj)
            check(fs::file_size(mtl.front()) > 0, "Empty material asset");
        return mesh;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = e.what()});
    }
} // namespace lfs::preprocess
