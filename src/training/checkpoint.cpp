/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "checkpoint.hpp"
#include "components/bilateral_grid.hpp"
#include "components/per_frame_affine_color.hpp"
#include "components/per_frame_observation_blur.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "pose_refiner.hpp"
#include "strategies/istrategy.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr char kCheckpointTempSuffix[] = ".tmp";

        std::filesystem::path checkpoint_temp_path(const std::filesystem::path& checkpoint_path) {
            auto temp_path = checkpoint_path;
            temp_path += kCheckpointTempSuffix;
            return temp_path;
        }

        std::expected<void, std::string> replace_checkpoint_file(
            const std::filesystem::path& checkpoint_path,
            const std::filesystem::path& temp_checkpoint_path) {

            std::error_code ec;
            std::filesystem::remove(checkpoint_path, ec);
            if (ec) {
                return std::unexpected("Failed to remove existing checkpoint file '" +
                                       lfs::core::path_to_utf8(checkpoint_path) + "': " + ec.message());
            }

            std::filesystem::rename(temp_checkpoint_path, checkpoint_path, ec);
            if (ec) {
                return std::unexpected("Failed to replace checkpoint file '" +
                                       lfs::core::path_to_utf8(checkpoint_path) + "': " + ec.message());
            }

            return {};
        }

        void restore_mesh_init_training_parameters(
            const nlohmann::json& json,
            lfs::core::param::TrainingParameters& params) {
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
            if (json.contains("mesh_init_color_source")) {
                const auto parsed_source = lfs::core::param::parse_mesh_init_color_source(
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

        void write_mesh_init_training_parameters(
            nlohmann::json& json,
            const lfs::core::param::TrainingParameters& params) {
            json["use_mesh_init"] = params.use_mesh_init;
            json["mesh_init_sampling_rate"] = params.mesh_init_sampling_rate;
            json["init_white_gs_color"] = params.init_white_gs_color;
            if (params.mesh_init_mesh_path && !params.mesh_init_mesh_path->empty()) {
                json["mesh_init_mesh_path"] = *params.mesh_init_mesh_path;
            }
            if (params.mesh_init_texture_path && !params.mesh_init_texture_path->empty()) {
                json["mesh_init_texture_path"] = *params.mesh_init_texture_path;
            }
            if (params.mesh_init_external_filled_mesh_path && !params.mesh_init_external_filled_mesh_path->empty()) {
                json["mesh_init_external_filled_mesh_path"] = *params.mesh_init_external_filled_mesh_path;
            }
            if (params.mesh_init_external_pointcloud_path && !params.mesh_init_external_pointcloud_path->empty()) {
                json["mesh_init_external_pointcloud_path"] = *params.mesh_init_external_pointcloud_path;
            }
        }
    } // namespace

    using lfs::core::CHECKPOINT_MAGIC;
    using lfs::core::CHECKPOINT_VERSION;
    using lfs::core::CheckpointFlags;
    using lfs::core::CheckpointHeader;
    using lfs::core::has_flag;

    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        const int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params,
        const BilateralGrid* bilateral_grid,
        const PPISP* ppisp,
        const PPISPControllerPool* ppisp_controller_pool,
        const PoseRefiner* pose_refiner,
        const PerFrameAffineColor* per_frame_affine_color,
        const PerFrameObservationBlur* per_frame_observation_blur) {

        try {
            // Validate input path
            if (path.empty()) {
                return std::unexpected("Cannot save checkpoint: output path is empty");
            }

            const auto checkpoint_dir = checkpoint_directory(path);
            const auto checkpoint_path = checkpoint_output_path(path);
            const auto temp_checkpoint_path = checkpoint_temp_path(checkpoint_path);

            // Create checkpoint directory with error checking
            std::error_code ec;
            std::filesystem::create_directories(checkpoint_dir, ec);
            if (ec) {
                return std::unexpected("Failed to create checkpoint directory '" +
                                       lfs::core::path_to_utf8(checkpoint_dir) + "': " + ec.message());
            }

            const auto& model = strategy.get_model();

            // Model tensors
            size_t model_bytes = 0;
            model_bytes += model.means().bytes();
            model_bytes += model.sh0().bytes();
            model_bytes += model.scaling_raw().bytes();
            model_bytes += model.rotation_raw().bytes();
            model_bytes += model.opacity_raw().bytes();
            if (model.shN().is_valid()) {
                model_bytes += model.shN().bytes();
            }
            if (model.deleted().is_valid()) {
                model_bytes += model.deleted().bytes();
            }
            if (model._densification_info.is_valid()) {
                model_bytes += model._densification_info.bytes();
            }

            // Optimizer: 2x learnable model tensors (Adam m & v). Static mesh
            // context is part of the model checkpoint but has no optimizer state.
            const size_t optimizer_bytes = model_bytes * 2;

            if (model.constraint_mesh_verts().is_valid()) {
                model_bytes += model.constraint_mesh_verts().bytes();
            }
            if (model.constraint_mesh_indices().is_valid()) {
                model_bytes += model.constraint_mesh_indices().bytes();
            }
            if (model.tri_edge_neighbors().is_valid()) {
                model_bytes += model.tri_edge_neighbors().bytes();
            }

            // Bilateral grid: 3x (grids + Adam state)
            size_t bilateral_grid_bytes = 0;
            if (bilateral_grid) {
                bilateral_grid_bytes = bilateral_grid->grids().bytes() * 3;
            }

            // PPISP: estimate based on num_cameras and num_frames
            size_t ppisp_bytes = 0;
            if (ppisp) {
                // exposure + vignetting + color + crf, each with params + 2x Adam state
                const size_t exp_size = ppisp->num_frames() * sizeof(float) * 3;
                const size_t vig_size = ppisp->num_cameras() * 3 * 5 * sizeof(float) * 3;
                const size_t color_size = ppisp->num_frames() * 8 * sizeof(float) * 3;
                const size_t crf_size = ppisp->num_cameras() * 3 * 4 * sizeof(float) * 3;
                ppisp_bytes = exp_size + vig_size + color_size + crf_size;
            }

            const size_t pose_refiner_bytes =
                (pose_refiner && pose_refiner->enabled()) ? pose_refiner->serialized_size_bytes() : 0;
            const size_t per_frame_affine_color_bytes =
                per_frame_affine_color ? per_frame_affine_color->serialized_size_bytes() : 0;
            const size_t per_frame_observation_blur_bytes =
                per_frame_observation_blur
                    ? per_frame_observation_blur->serialized_size_bytes()
                    : 0;

            // Strategy-private row state: Int32 birth_tri plus Bool mesh-init and
            // immutable hole-fill provenance masks.
            const size_t strategy_mesh_state_bytes = static_cast<size_t>(model.size()) * 6;

            constexpr size_t OVERHEAD_BYTES = 64 * 1024;

            const size_t estimated_size = sizeof(CheckpointHeader) +
                                          model_bytes +
                                          optimizer_bytes +
                                          bilateral_grid_bytes +
                                          ppisp_bytes +
                                          pose_refiner_bytes +
                                          per_frame_affine_color_bytes +
                                          per_frame_observation_blur_bytes +
                                          strategy_mesh_state_bytes +
                                          OVERHEAD_BYTES;

            if (auto space_check = lfs::io::check_disk_space(checkpoint_path, estimated_size, 1.1f);
                !space_check) {
                const auto& error = space_check.error();
                const bool is_disk_space = error.is(lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE);

                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iteration,
                    .path = checkpoint_path,
                    .error = error.format(),
                    .required_bytes = estimated_size,
                    .available_bytes = error.available_bytes,
                    .is_disk_space_error = is_disk_space}
                    .emit();

                return std::unexpected(error.format());
            }

            std::ofstream file;
            if (!lfs::core::open_file_for_write(temp_checkpoint_path, std::ios::binary, file)) {
                return std::unexpected("Failed to open checkpoint file: " +
                                       lfs::core::path_to_utf8(temp_checkpoint_path));
            }

            CheckpointHeader header{};
            header.iteration = iteration;
            header.num_gaussians = static_cast<uint32_t>(model.size());
            header.sh_degree = model.get_max_sh_degree();
            header.flags = CheckpointFlags::NONE;
            if (bilateral_grid)
                header.flags = header.flags | CheckpointFlags::HAS_BILATERAL_GRID;
            if (ppisp)
                header.flags = header.flags | CheckpointFlags::HAS_PPISP;
            if (ppisp_controller_pool)
                header.flags = header.flags | CheckpointFlags::HAS_PPISP_CONTROLLER;
            if (pose_refiner && pose_refiner->enabled())
                header.flags = header.flags | CheckpointFlags::HAS_POSE_REFINER;
            if (per_frame_affine_color)
                header.flags = header.flags | CheckpointFlags::HAS_PER_FRAME_AFFINE_COLOR;
            if (per_frame_observation_blur)
                header.flags = header.flags | CheckpointFlags::HAS_PER_FRAME_OBSERVATION_BLUR;

            const auto header_pos = file.tellp();
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            // Strategy type
            const char* const strategy_type = strategy.strategy_type();
            const uint32_t type_len = static_cast<uint32_t>(std::strlen(strategy_type));
            file.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
            file.write(strategy_type, type_len);

            // Model and strategy state
            model.serialize(file);
            strategy.serialize(file);

            // Bilateral grid (if present)
            if (bilateral_grid) {
                bilateral_grid->serialize(file);
                LOG_DEBUG("Bilateral grid state saved (step={}, lr={:.2e})",
                          bilateral_grid->get_step(), bilateral_grid->get_lr());
            }

            // PPISP (if present)
            if (ppisp) {
                ppisp->serialize(file);
                LOG_DEBUG("PPISP state saved (step={}, lr={:.2e})",
                          ppisp->get_step(), ppisp->get_lr());
            }

            // PPISP controller pool (if present)
            if (ppisp_controller_pool) {
                ppisp_controller_pool->serialize(file);
                LOG_DEBUG("PPISP controller pool saved: {} cameras", ppisp_controller_pool->num_cameras());
            }

            if (pose_refiner && pose_refiner->enabled()) {
                pose_refiner->serialize(file);
                LOG_DEBUG("Pose refine state saved: {} real cameras", pose_refiner->real_camera_count());
            }

            // Keep new optional blocks appended after all version-1 blocks so
            // the legacy block order remains stable.
            if (per_frame_affine_color) {
                per_frame_affine_color->serialize(file);
                LOG_DEBUG("Per-frame affine color state saved");
            }
            if (per_frame_observation_blur) {
                per_frame_observation_blur->serialize(file);
                LOG_DEBUG("Per-frame observation blur state saved");
            }

            // Training parameters as JSON
            const auto params_pos = file.tellp();
            nlohmann::json params_json;
            params_json["optimization"] = params.optimization.to_json();
            params_json["dataset"] = params.dataset.to_json();
            write_mesh_init_training_parameters(params_json, params);
            const std::string params_str = params_json.dump();
            file.write(params_str.data(), static_cast<std::streamsize>(params_str.size()));
            const auto params_end = file.tellp();

            // Update header with JSON offset
            header.params_json_offset = static_cast<uint64_t>(params_pos);
            header.params_json_size = static_cast<uint64_t>(params_end - params_pos);
            file.seekp(header_pos);
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            file.close();
            if (!file) {
                return std::unexpected("Failed to finalize checkpoint file: " +
                                       lfs::core::path_to_utf8(temp_checkpoint_path));
            }

            if (auto replace_result = replace_checkpoint_file(checkpoint_path, temp_checkpoint_path); !replace_result)
                return std::unexpected(replace_result.error());

            std::string extras;
            if (bilateral_grid)
                extras += ", +bilateral";
            if (ppisp)
                extras += ", +ppisp";
            if (ppisp_controller_pool)
                extras += ", +ppisp_ctrl(" + std::to_string(ppisp_controller_pool->num_cameras()) + ")";
            if (pose_refiner && pose_refiner->enabled())
                extras += ", +pose_refine(" + std::to_string(pose_refiner->real_camera_count()) + ")";
            if (per_frame_affine_color)
                extras += ", +affine_color";
            if (per_frame_observation_blur)
                extras += ", +observation_blur";
            LOG_INFO("Checkpoint saved: {} ({} Gaussians, iter {}{})",
                     lfs::core::path_to_utf8(checkpoint_path), header.num_gaussians, iteration,
                     extras);
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Save checkpoint failed: ") + e.what());
        }
    }

    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params,
        BilateralGrid* bilateral_grid,
        PPISP* ppisp,
        PPISPControllerPool* ppisp_controller_pool,
        PoseRefiner* pose_refiner,
        PerFrameAffineColor* per_frame_affine_color,
        PerFrameObservationBlur* per_frame_observation_blur) {

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open: " + lfs::core::path_to_utf8(path));
            }

            CheckpointHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != CHECKPOINT_MAGIC) {
                return std::unexpected("Invalid checkpoint: wrong magic");
            }
            if (header.version > CHECKPOINT_VERSION) {
                return std::unexpected("Unsupported version: " + std::to_string(header.version));
            }

            // Verify strategy compatibility
            uint32_t type_len = 0;
            file.read(reinterpret_cast<char*>(&type_len), sizeof(type_len));
            std::string saved_type(type_len, '\0');
            file.read(saved_type.data(), type_len);

            if (saved_type != strategy.strategy_type()) {
                return std::unexpected("Strategy mismatch: '" + saved_type +
                                       "' vs '" + strategy.strategy_type() + "'");
            }

            // Load params from checkpoint up front so strategy internals can be synced before deserialization.
            const auto strategy_state_pos = file.tellg();
            if (header.params_json_size > 0) {
                file.seekg(static_cast<std::streamoff>(header.params_json_offset));
                std::string params_str(header.params_json_size, '\0');
                file.read(params_str.data(), static_cast<std::streamsize>(header.params_json_size));

                const auto cli_data_path = params.dataset.data_path;
                const auto cli_output_path = params.dataset.output_path;
                const bool runtime_save_per_frame_affine_color =
                    params.optimization.save_per_frame_affine_color;
                const bool runtime_save_per_frame_blur_parameters =
                    params.optimization.save_per_frame_blur_parameters;
                const bool runtime_save_pose_refine_outputs =
                    params.optimization.save_pose_refine_outputs;
                const float runtime_observation_blur_max_radius_px =
                    params.optimization.per_frame_observation_blur_max_radius_px;
                const bool runtime_use_mesh_init = params.use_mesh_init;
                const auto runtime_mesh2splat_hole_fill_mode = params.optimization.mesh2splat_hole_fill_mode;
                const bool runtime_mesh2splat_hole_fill_enabled = params.optimization.mesh2splat_hole_fill_enabled;
                const auto runtime_mesh_init_mesh_path = params.mesh_init_mesh_path;
                const auto runtime_mesh_init_texture_path = params.mesh_init_texture_path;
                const auto runtime_mesh_init_external_filled_mesh_path = params.mesh_init_external_filled_mesh_path;
                const auto runtime_mesh_init_external_pointcloud_path = params.mesh_init_external_pointcloud_path;
                const float runtime_mesh_init_sampling_rate = params.mesh_init_sampling_rate;
                const auto runtime_mesh_init_color_source = params.optimization.mesh_init_color_source;
                const bool runtime_init_white_gs_color = params.init_white_gs_color;

                const auto params_json = nlohmann::json::parse(params_str);
                if (params_json.contains("optimization")) {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json["optimization"]);
                    if (params_json.contains("dataset")) {
                        params.dataset = lfs::core::param::DatasetConfig::from_json(params_json["dataset"]);
                    }
                } else {
                    params.optimization = lfs::core::param::OptimizationParameters::from_json(params_json);
                }
                restore_mesh_init_training_parameters(params_json, params);

                if (!cli_data_path.empty())
                    params.dataset.data_path = cli_data_path;
                if (!cli_output_path.empty())
                    params.dataset.output_path = cli_output_path;

                // This controls an output side effect of the current invocation, not
                // the training state represented by the checkpoint.
                params.optimization.save_per_frame_affine_color =
                    runtime_save_per_frame_affine_color;
                params.optimization.save_per_frame_blur_parameters =
                    runtime_save_per_frame_blur_parameters;
                // This controls an output side effect of the current
                // invocation, not the training state represented by the
                // checkpoint.
                params.optimization.save_pose_refine_outputs =
                    runtime_save_pose_refine_outputs;
                // This is a runtime memory-safety policy, not learned state.
                // Keep the current invocation's cap when resuming instead of
                // allowing an older checkpoint JSON to override it.
                params.optimization.per_frame_observation_blur_max_radius_px =
                    runtime_observation_blur_max_radius_px;

                // A mesh-init source explicitly supplied for this invocation is
                // a runtime input override. Keep its resolved paths and sampling
                // choices instead of silently rebuilding from the saved source.
                if (runtime_use_mesh_init) {
                    params.use_mesh_init = true;
                    params.optimization.mesh2splat_hole_fill_mode = runtime_mesh2splat_hole_fill_mode;
                    params.optimization.mesh2splat_hole_fill_enabled = runtime_mesh2splat_hole_fill_enabled;
                    params.mesh_init_mesh_path = runtime_mesh_init_mesh_path;
                    params.mesh_init_texture_path = runtime_mesh_init_texture_path;
                    params.mesh_init_external_filled_mesh_path = runtime_mesh_init_external_filled_mesh_path;
                    params.mesh_init_external_pointcloud_path = runtime_mesh_init_external_pointcloud_path;
                    params.mesh_init_sampling_rate = runtime_mesh_init_sampling_rate;
                    params.optimization.mesh_init_color_source = runtime_mesh_init_color_source;
                    params.init_white_gs_color = runtime_init_white_gs_color;
                }
            }
            strategy.set_optimization_params(params.optimization);
            file.clear();
            file.seekg(strategy_state_pos);

            // Model and strategy state
            strategy.get_model().deserialize(file);
            strategy.deserialize(file);

            // Bilateral grid (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_BILATERAL_GRID)) {
                if (bilateral_grid) {
                    bilateral_grid->deserialize(file);
                    LOG_INFO("Bilateral grid restored (step={}, lr={:.2e})",
                             bilateral_grid->get_step(), bilateral_grid->get_lr());
                } else {
                    LOG_WARN("Checkpoint has bilateral grid but none provided - skipping data");
                    BilateralGrid temp(1, 1, 1, 1, 1);
                    temp.deserialize(file);
                }
            } else if (bilateral_grid) {
                LOG_WARN("Bilateral grid requested but not in checkpoint - using fresh state");
            }

            // PPISP (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP)) {
                if (ppisp) {
                    ppisp->deserialize(file);
                    LOG_INFO("PPISP restored (step={}, lr={:.2e})", ppisp->get_step(), ppisp->get_lr());
                } else {
                    LOG_WARN("Checkpoint has PPISP but none provided - skipping data");
                    PPISP temp(1);
                    temp.deserialize(file);
                }
            } else if (ppisp) {
                LOG_WARN("PPISP requested but not in checkpoint - using fresh state");
            }

            // PPISP controller pool (if present in checkpoint)
            if (has_flag(header.flags, CheckpointFlags::HAS_PPISP_CONTROLLER)) {
                if (ppisp_controller_pool) {
                    ppisp_controller_pool->deserialize(file);
                    LOG_INFO("PPISP controller pool restored: {} cameras (step={}, lr={:.2e})",
                             ppisp_controller_pool->num_cameras(),
                             0, // step not easily accessible from pool
                             ppisp_controller_pool->get_learning_rate());
                } else {
                    LOG_WARN("Checkpoint has PPISP controller pool but none provided - skipping");
                    // Skip the pool data by reading into a temporary
                    // Pool format: magic + version + num_cameras + ... (variable size)
                    // We need to read it to advance the file position
                    uint32_t magic, version;
                    int num_cameras, total_iter;
                    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                    file.read(reinterpret_cast<char*>(&version), sizeof(version));
                    file.read(reinterpret_cast<char*>(&num_cameras), sizeof(num_cameras));
                    // Create a temporary pool to skip the data
                    PPISPControllerPool temp(num_cameras, 1);
                    // Rewind and deserialize properly to skip
                    file.seekg(-static_cast<std::streamoff>(sizeof(magic) + sizeof(version) + sizeof(num_cameras)),
                               std::ios::cur);
                    temp.deserialize(file);
                }
            } else if (ppisp_controller_pool) {
                LOG_WARN("PPISP controller pool requested but not in checkpoint - using fresh state");
            }

            if (has_flag(header.flags, CheckpointFlags::HAS_POSE_REFINER)) {
                if (pose_refiner) {
                    if (auto pose_result = pose_refiner->deserialize(file); !pose_result) {
                        return std::unexpected(pose_result.error());
                    }
                } else {
                    LOG_WARN("Checkpoint has pose refine state but none provided - skipping");
                    if (auto skip_result = PoseRefiner::skip_serialized(file); !skip_result) {
                        return std::unexpected(skip_result.error());
                    }
                }
            } else if (pose_refiner) {
                LOG_WARN("Pose refine requested but not in checkpoint - using fresh state");
            }

            if (has_flag(header.flags, CheckpointFlags::HAS_PER_FRAME_AFFINE_COLOR)) {
                if (per_frame_affine_color) {
                    per_frame_affine_color->deserialize(file);
                    LOG_INFO("Per-frame affine color state restored");
                } else {
                    LOG_WARN("Checkpoint has per-frame affine color state but none provided - skipping");
                    if (auto skip_result = PerFrameAffineColor::skip_serialized(file); !skip_result) {
                        return std::unexpected(skip_result.error());
                    }
                }
            } else if (per_frame_affine_color) {
                LOG_WARN("Per-frame affine color requested but not in checkpoint - using fresh identity state");
            }

            if (has_flag(header.flags, CheckpointFlags::HAS_PER_FRAME_OBSERVATION_BLUR)) {
                if (per_frame_observation_blur) {
                    per_frame_observation_blur->deserialize(file);
                    LOG_INFO("Per-frame observation blur state restored");
                } else {
                    LOG_WARN("Checkpoint has per-frame observation blur state but none provided - skipping");
                    if (auto skip_result = PerFrameObservationBlur::skip_serialized(file); !skip_result) {
                        return std::unexpected(skip_result.error());
                    }
                }
            } else if (per_frame_observation_blur) {
                LOG_WARN("Per-frame observation blur requested but not in checkpoint - using fresh zero-blur state");
            }

            // Reserve capacity for densification after the checkpoint params are resolved.
            const size_t max_cap = static_cast<size_t>(params.optimization.max_cap);
            if (max_cap > strategy.get_model().size()) {
                LOG_DEBUG("Reserving capacity: {} (current: {})", max_cap, strategy.get_model().size());
                strategy.get_model().reserve_capacity(max_cap);
                strategy.reserve_optimizer_capacity(max_cap);
            }

            LOG_INFO("Checkpoint loaded: {} ({} Gaussians, iter {})",
                     lfs::core::path_to_utf8(path), header.num_gaussians, header.iteration);
            return header.iteration;

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Load checkpoint failed: ") + e.what());
        }
    }

} // namespace lfs::training
