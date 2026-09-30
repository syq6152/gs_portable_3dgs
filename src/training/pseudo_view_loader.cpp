/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pseudo_view_loader.hpp"

#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lfs::training {

    namespace {

        std::expected<std::array<float, 9>, std::string> read_mat3(const nlohmann::json& j) {
            if (!j.is_array() || j.size() != 3) {
                return std::unexpected("R must be a 3x3 array");
            }
            std::array<float, 9> out{};
            for (size_t r = 0; r < 3; ++r) {
                if (!j[r].is_array() || j[r].size() != 3) {
                    return std::unexpected("R must be a 3x3 array");
                }
                for (size_t c = 0; c < 3; ++c) {
                    try {
                        out[r * 3 + c] = j[r][c].get<float>();
                    } catch (const std::exception& e) {
                        return std::unexpected(std::string("R contains a non-numeric value: ") + e.what());
                    }
                    if (!std::isfinite(out[r * 3 + c])) {
                        return std::unexpected("R contains a non-finite value");
                    }
                }
            }
            return out;
        }

        std::expected<std::array<float, 3>, std::string> read_vec3(const nlohmann::json& j) {
            if (!j.is_array() || j.size() != 3) {
                return std::unexpected("T must be a length-3 array");
            }
            std::array<float, 3> out{};
            try {
                out = {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
            } catch (const std::exception& e) {
                return std::unexpected(std::string("T contains a non-numeric value: ") + e.what());
            }
            if (!std::isfinite(out[0]) || !std::isfinite(out[1]) || !std::isfinite(out[2])) {
                return std::unexpected("T contains a non-finite value");
            }
            return out;
        }

        std::filesystem::path resolve_relative(
            const std::filesystem::path& base,
            const std::string& rel_or_abs) {
            std::filesystem::path p = lfs::core::utf8_to_path(rel_or_abs);
            if (p.is_absolute()) {
                return p;
            }
            return base / p;
        }

        std::expected<size_t, std::string> count_valid_mask_pixels(
            const std::filesystem::path& mask_path) {
            try {
                auto [pixels, width, height, channels] =
                    lfs::core::load_image(mask_path, 1, 0);
                const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> owned_pixels(
                    pixels,
                    &lfs::core::free_image);
                if (!pixels || width <= 0 || height <= 0 || channels <= 0) {
                    return std::unexpected("decoded mask has no pixels");
                }

                const size_t pixel_count =
                    static_cast<size_t>(width) * static_cast<size_t>(height);
                size_t valid_pixels = 0;
                for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
                    const size_t offset = pixel * static_cast<size_t>(channels);
                    bool valid = false;
                    for (int channel = 0; channel < channels; ++channel) {
                        valid = valid || pixels[offset + static_cast<size_t>(channel)] != 0;
                    }
                    valid_pixels += valid ? 1u : 0u;
                }
                return valid_pixels;
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

    } // namespace

    std::expected<std::vector<std::shared_ptr<lfs::core::Camera>>, std::string>
    load_pseudo_view_cameras(
        const std::filesystem::path& manifest_path,
        const lfs::core::Camera& base_camera,
        float loss_weight,
        int starting_uid,
        std::size_t min_valid_pixels,
        bool log_diagnostics) {
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
        if (!std::filesystem::exists(manifest_path)) {
            return std::unexpected(std::format(
                "Pseudo-view manifest not found: {}",
                lfs::core::path_to_utf8(manifest_path)));
        }

        nlohmann::json manifest;
        try {
            std::ifstream in(manifest_path);
            if (!in) {
                return std::unexpected(std::format(
                    "Cannot open pseudo-view manifest: {}",
                    lfs::core::path_to_utf8(manifest_path)));
            }
            in >> manifest;
        } catch (const std::exception& e) {
            return std::unexpected(std::format(
                "Failed to parse pseudo-view manifest {}: {}",
                lfs::core::path_to_utf8(manifest_path),
                e.what()));
        }

        int manifest_version = 1;
        if (manifest.is_object()) {
            if (const auto version_it = manifest.find("version"); version_it != manifest.end()) {
                if (!version_it->is_number_integer()) {
                    return std::unexpected("Pseudo-view manifest: 'version' must be an integer");
                }
                manifest_version = version_it->get<int>();
            }
        }
        if (manifest_version != 1 && manifest_version != 2) {
            return std::unexpected(std::format(
                "Unsupported pseudo-view manifest version: {}",
                manifest_version));
        }
        if (manifest_version == 1 && log_diagnostics) {
            LOG_WARN("Pseudo-view cache uses manifest v1; re-run pseudo-view precompute to apply post-morphology coverage filtering");
        }

        const auto entries_it = manifest.find("views");
        const nlohmann::json& entries = (entries_it != manifest.end() && entries_it->is_array())
                                            ? *entries_it
                                            : (manifest.is_array() ? manifest : nlohmann::json::array());
        if (!entries.is_array()) {
            return std::unexpected("Pseudo-view manifest: 'views' field is not an array");
        }

        const auto base_dir = manifest_path.parent_path();
        int next_uid = starting_uid;
        cameras.reserve(entries.size());

        for (const auto& entry : entries) {
            // Skip entries that didn't produce a usable pseudo view.
            if (!entry.is_object()) {
                continue;
            }
            if (entry.contains("status") &&
                (!entry["status"].is_string() || entry["status"].get<std::string>() != "ok")) {
                continue;
            }
            if (!entry.contains("rgb_path") || !entry["rgb_path"].is_string() ||
                !entry.contains("pose_path") || !entry["pose_path"].is_string()) {
                continue;
            }

            const auto rgb_path = resolve_relative(base_dir, entry["rgb_path"].get<std::string>());
            const auto pose_path = resolve_relative(base_dir, entry["pose_path"].get<std::string>());
            std::filesystem::path mask_path;
            if (entry.contains("supervision_mask_path") && entry["supervision_mask_path"].is_string()) {
                mask_path = resolve_relative(base_dir, entry["supervision_mask_path"].get<std::string>());
            } else if (entry.contains("valid_mask_path") && entry["valid_mask_path"].is_string()) {
                mask_path = resolve_relative(base_dir, entry["valid_mask_path"].get<std::string>());
            }

            if (mask_path.empty()) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view supervision mask path missing, skipping entry");
                }
                continue;
            }

            if (!std::filesystem::is_regular_file(rgb_path)) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view RGB missing, skipping entry: {}",
                             lfs::core::path_to_utf8(rgb_path));
                }
                continue;
            }
            if (!std::filesystem::is_regular_file(pose_path)) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON missing, skipping entry: {}",
                             lfs::core::path_to_utf8(pose_path));
                }
                continue;
            }
            if (!std::filesystem::is_regular_file(mask_path)) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view supervision mask missing, skipping entry: {}",
                             lfs::core::path_to_utf8(mask_path));
                }
                continue;
            }

            auto valid_mask_pixels = count_valid_mask_pixels(mask_path);
            if (!valid_mask_pixels) {
                if (log_diagnostics) {
                    LOG_WARN("Failed to read pseudo-view supervision mask {}, skipping entry: {}",
                             lfs::core::path_to_utf8(mask_path), valid_mask_pixels.error());
                }
                continue;
            }
            if (*valid_mask_pixels == 0) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view supervision mask is empty, skipping entry: {}",
                             lfs::core::path_to_utf8(mask_path));
                }
                continue;
            }
            if (min_valid_pixels > 0 && *valid_mask_pixels < min_valid_pixels) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view supervision mask has {} valid pixels, below configured minimum {}; skipping entry: {}",
                             *valid_mask_pixels,
                             min_valid_pixels,
                             lfs::core::path_to_utf8(mask_path));
                }
                continue;
            }

            nlohmann::json pose_json;
            try {
                std::ifstream in(pose_path);
                in >> pose_json;
            } catch (const std::exception& e) {
                if (log_diagnostics) {
                    LOG_WARN("Failed to parse pseudo-view pose JSON {}: {}",
                             lfs::core::path_to_utf8(pose_path), e.what());
                }
                continue;
            }

            if (!pose_json.contains("R") || !pose_json.contains("T") ||
                !pose_json.contains("intrinsics") ||
                !pose_json.contains("image_width") || !pose_json.contains("image_height")) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON missing required fields: {}",
                             lfs::core::path_to_utf8(pose_path));
                }
                continue;
            }

            auto R_mat = read_mat3(pose_json["R"]);
            if (!R_mat) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON {} bad R: {}",
                             lfs::core::path_to_utf8(pose_path), R_mat.error());
                }
                continue;
            }
            auto T_vec = read_vec3(pose_json["T"]);
            if (!T_vec) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON {} bad T: {}",
                             lfs::core::path_to_utf8(pose_path), T_vec.error());
                }
                continue;
            }

            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
            int width = 0;
            int height = 0;
            try {
                const auto& intr = pose_json["intrinsics"];
                if (!intr.is_object() ||
                    !intr.contains("fx") || !intr.contains("fy") ||
                    !intr.contains("cx") || !intr.contains("cy")) {
                    throw std::invalid_argument("intrinsics must contain fx/fy/cx/cy");
                }
                fx = intr["fx"].get<float>();
                fy = intr["fy"].get<float>();
                cx = intr["cx"].get<float>();
                cy = intr["cy"].get<float>();
                width = pose_json["image_width"].get<int>();
                height = pose_json["image_height"].get<int>();
            } catch (const std::exception& e) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON {} has invalid intrinsics or dimensions: {}",
                             lfs::core::path_to_utf8(pose_path), e.what());
                }
                continue;
            }
            if (!(fx > 0.0f) || !(fy > 0.0f) ||
                !std::isfinite(fx) || !std::isfinite(fy) ||
                !std::isfinite(cx) || !std::isfinite(cy) ||
                width <= 0 || height <= 0) {
                if (log_diagnostics) {
                    LOG_WARN("Pseudo-view pose JSON has non-finite intrinsics or invalid dimensions: {}",
                             lfs::core::path_to_utf8(pose_path));
                }
                continue;
            }
            const int base_camera_uid =
                entry.contains("base_camera_uid") && entry["base_camera_uid"].is_number_integer()
                    ? entry["base_camera_uid"].get<int>()
                    : -1;
            const std::string source_image_name =
                entry.contains("source_image_name") && entry["source_image_name"].is_string()
                    ? entry["source_image_name"].get<std::string>()
                    : std::string{};
            const int source_camera_id =
                entry.contains("source_camera_id") && entry["source_camera_id"].is_number_integer()
                    ? entry["source_camera_id"].get<int>()
                    : -1;
            // source_uid is only useful within the dataset instance that produced
            // the cache. Accept the old detailed-manifest spelling for diagnostics,
            // but never use either UID field as the persistent source identity.
            const int source_uid =
                entry.contains("source_uid") && entry["source_uid"].is_number_integer()
                    ? entry["source_uid"].get<int>()
                    : (entry.contains("source_id") && entry["source_id"].is_number_integer()
                           ? entry["source_id"].get<int>()
                           : -1);

            auto R_tensor = lfs::core::Tensor::from_vector(
                std::vector<float>(R_mat->begin(), R_mat->end()),
                {std::size_t{3}, std::size_t{3}},
                lfs::core::Device::CPU);
            auto T_tensor = lfs::core::Tensor::from_vector(
                std::vector<float>(T_vec->begin(), T_vec->end()),
                {std::size_t{3}},
                lfs::core::Device::CPU);

            const std::string image_name = lfs::core::path_to_utf8(rgb_path.filename());

            try {
                auto cam = std::make_shared<lfs::core::Camera>(
                    R_tensor,
                    T_tensor,
                    fx, fy, cx, cy,
                    base_camera.radial_distortion(),
                    base_camera.tangential_distortion(),
                    base_camera.camera_model_type(),
                    image_name,
                    rgb_path,
                    mask_path,
                    width, height,
                    next_uid--,
                    base_camera.camera_id());
                cam->set_image_dimensions(width, height);
                cam->set_pseudo(true, loss_weight);
                cam->set_pseudo_base_camera_uid(base_camera_uid);
                cam->set_pseudo_rgb_source(source_image_name, source_camera_id, source_uid);
                cameras.push_back(std::move(cam));
            } catch (const std::exception& e) {
                if (log_diagnostics) {
                    LOG_WARN("Failed to construct pseudo Camera for {}: {}",
                             lfs::core::path_to_utf8(rgb_path), e.what());
                }
                continue;
            }
        }

        return cameras;
    }

} // namespace lfs::training
