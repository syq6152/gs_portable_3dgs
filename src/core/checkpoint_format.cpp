/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/checkpoint_format.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace lfs::core {

    namespace {

        void restore_mesh_init_training_parameters(
            const nlohmann::json& json,
            param::TrainingParameters& params) {
            if (json.contains("use_mesh_init")) {
                params.use_mesh_init = json.at("use_mesh_init").get<bool>();
            }
            if (json.contains("mesh_init_mesh_path")) {
                if (json.at("mesh_init_mesh_path").is_null()) {
                    params.mesh_init_mesh_path = std::nullopt;
                } else {
                    const auto path = json.at("mesh_init_mesh_path").get<std::string>();
                    params.mesh_init_mesh_path = path.empty()
                                                     ? std::nullopt
                                                     : std::optional<std::string>{path};
                }
            }
            if (json.contains("mesh_init_texture_path")) {
                if (json.at("mesh_init_texture_path").is_null()) {
                    params.mesh_init_texture_path = std::nullopt;
                } else {
                    const auto path = json.at("mesh_init_texture_path").get<std::string>();
                    params.mesh_init_texture_path = path.empty()
                                                        ? std::nullopt
                                                        : std::optional<std::string>{path};
                }
            }
            if (json.contains("mesh_init_external_filled_mesh_path")) {
                if (json.at("mesh_init_external_filled_mesh_path").is_null()) {
                    params.mesh_init_external_filled_mesh_path = std::nullopt;
                } else {
                    const auto path = json.at("mesh_init_external_filled_mesh_path").get<std::string>();
                    params.mesh_init_external_filled_mesh_path = path.empty()
                                                                     ? std::nullopt
                                                                     : std::optional<std::string>{path};
                }
            }
            if (json.contains("mesh_init_external_pointcloud_path")) {
                if (json.at("mesh_init_external_pointcloud_path").is_null()) {
                    params.mesh_init_external_pointcloud_path = std::nullopt;
                } else {
                    const auto path = json.at("mesh_init_external_pointcloud_path").get<std::string>();
                    params.mesh_init_external_pointcloud_path = path.empty()
                                                                   ? std::nullopt
                                                                   : std::optional<std::string>{path};
                }
            }
            if (json.contains("mesh_init_sampling_rate")) {
                params.mesh_init_sampling_rate = json.at("mesh_init_sampling_rate").get<float>();
            }
            // Older training checkpoints may store mesh-init-only fields at the
            // top level. Keep this compatibility path in sync with the training
            // checkpoint loader; current checkpoints persist the source inside
            // the optimization object.
            if (json.contains("mesh_init_color_source")) {
                const auto parsed_source = param::parse_mesh_init_color_source(
                    json.at("mesh_init_color_source").get<std::string>());
                if (!parsed_source) {
                    throw std::invalid_argument(parsed_source.error());
                }
                params.optimization.mesh_init_color_source = *parsed_source;
            }
            if (json.contains("init_white_gs_color")) {
                params.init_white_gs_color = json.at("init_white_gs_color").get<bool>();
            }
        }

        std::expected<CheckpointHeader, std::string> open_and_validate(
            const std::filesystem::path& path, std::ifstream& file) {

            if (!open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + path_to_utf8(path));
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }
            return header;
        }

    } // namespace

    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            return open_and_validate(path, file);
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Read header failed: ") + e.what());
        }
    }

    std::expected<SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            auto header = open_and_validate(path, file);
            if (!header) {
                return std::unexpected(header.error());
            }

            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            file.seekg(type_len, std::ios::cur);

            SplatData splat;
            splat.deserialize(file);

            LOG_DEBUG("SplatData loaded: {} Gaussians, iter {}", header->num_gaussians, header->iteration);
            return splat;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load SplatData failed: ") + e.what());
        }
    }

    std::expected<param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path) {

        try {
            std::ifstream file;
            auto header = open_and_validate(path, file);
            if (!header) {
                return std::unexpected(header.error());
            }

            param::TrainingParameters params;
            if (header->params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header->params_json_offset));
                std::string params_str(header->params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header->params_json_size));

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    params.optimization = param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        params.dataset = param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                } else {
                    params.optimization = param::OptimizationParameters::from_json(params_json);
                }
                restore_mesh_init_training_parameters(params_json, params);
            }

            LOG_DEBUG("Params loaded from checkpoint: {}", path_to_utf8(params.dataset.data_path));
            return params;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load params failed: ") + e.what());
        }
    }

} // namespace lfs::core
