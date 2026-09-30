/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/debug_exports.hpp"
#include "preprocessing/runtime.hpp"
#include "preprocessing/workspace.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::preprocess {
    std::expected<std::size_t, Error> write_mesh_projection_overlays(
        const CpuMesh& mesh, const ColmapModel& model, const std::filesystem::path& images,
        const std::filesystem::path& output, const std::string& prefix,
        const ExecutionContext& ctx, bool write_diagnostic_json) try {
        namespace fs = std::filesystem;
        if (prefix != "scan_" && prefix != "inc_")
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Invalid mesh overlay prefix"});
        if (auto isolated = check_output_isolation(output, {images}); !isolated)
            return std::unexpected(isolated.error());
        if (!fs::is_directory(output))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Mesh overlay output must be an owned staging directory"});
        nlohmann::json manifest{{"schema_version", 1}, {"kind", "mesh_vertex_overlay"}, {"prefix", prefix}, {"color", {0, 255, 0}}, {"alpha", 160}, {"frames", nlohmann::json::array()}};
        for (const auto& [id, image] : model.images) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Mesh overlay cancelled"});
            const auto name = fs::u8path(image.name);
            if (name.empty() || name.has_parent_path() || name == "." || name == ".." ||
                image.name.find_first_of("/\\:") != std::string::npos || !model.cameras.contains(image.camera_id))
                return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Invalid mesh overlay image/camera mapping"});
            const auto destination = output / fs::u8path(prefix + image.name);
            if (fs::exists(destination))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Mesh overlay destination already exists"});
            auto pixels = read_image(images / name, false);
            if (!pixels)
                return std::unexpected(pixels.error());
            // Pillow converts grayscale sources to RGBA for alpha compositing.
            // Expand only the diagnostic copy, never the sampled/training image.
            if (pixels->channels == 1) {
                CpuImage rgb{pixels->width, pixels->height, 3, std::vector<uint8_t>(pixels->pixels.size() * 3)};
                for (std::size_t i = 0; i < pixels->pixels.size(); ++i)
                    rgb.pixels[i * 3] = rgb.pixels[i * 3 + 1] = rgb.pixels[i * 3 + 2] = pixels->pixels[i];
                *pixels = std::move(rgb);
            }
            auto overlay = project_mesh_overlay(mesh, *pixels, model.cameras.at(image.camera_id), image, ctx);
            if (!overlay)
                return std::unexpected(overlay.error());
            if (auto saved = write_png(destination, *overlay); !saved)
                return std::unexpected(saved.error());
            if (write_diagnostic_json)
                manifest["frames"].push_back({{"source", path_utf8(images / name)}, {"output", prefix + image.name}, {"image_id", id}, {"camera_id", image.camera_id}});
        }
        if (ctx.stop_token.stop_requested())
            return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Mesh overlay cancelled"});
        if (write_diagnostic_json) {
            std::ofstream file(output / (prefix + "manifest.json"), std::ios::binary | std::ios::noreplace);
            file << manifest.dump(2) << '\n';
            file.close();
            if (!file)
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot write mesh overlay manifest"});
        }
        return model.images.size();
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
} // namespace lfs::preprocess
