/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/reconstruction_command.hpp"
#include "app/reconstruction_log.hpp"
#include "app/training_runner.hpp"
#include "app/reconstruction_progress.hpp"
#include "app/reconstruction_output.hpp"
#include "app/shared_progress.hpp"
#include "core/argument_parser.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_crypto.hpp"
#include "core/path_utils.hpp"
#include "config.h"
#include "io/scanner_bin.hpp"
#include "preprocessing/dataset_validation.hpp"
#include "preprocessing/legacy_compat.hpp"
#include "preprocessing/pipeline.hpp"
#include "preprocessing/runtime.hpp"
#include "preprocessing/workspace.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <span>
#include <thread>
namespace lfs::app {
    namespace {
        using Json = nlohmann::json;
        std::atomic<bool> reconstruction_interrupted{false};

        void reconstruction_signal_handler(int) {
            reconstruction_interrupted.store(true, std::memory_order_relaxed);
        }
#ifdef _WIN32
        BOOL WINAPI reconstruction_console_handler(DWORD event) {
            if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT || event == CTRL_SHUTDOWN_EVENT) {
                reconstruction_signal_handler(0);
                return TRUE;
            }
            return FALSE;
        }
#endif
        // Signal handlers only set a flag; a normal thread propagates cancellation
        // even while a provider is waiting for a child. The runner installs its own
        // nested training handlers and restores these before returning.
        class ReconstructionInterrupts {
        public:
            ReconstructionInterrupts() {
                reconstruction_interrupted.store(false, std::memory_order_relaxed);
                previous_int_ = std::signal(SIGINT, reconstruction_signal_handler);
                previous_term_ = std::signal(SIGTERM, reconstruction_signal_handler);
#ifdef _WIN32
                SetConsoleCtrlHandler(nullptr, FALSE);
                console_installed_ = SetConsoleCtrlHandler(reconstruction_console_handler, TRUE) != FALSE;
#endif
                monitor_ = std::jthread([this](std::stop_token stop) {
                    std::unique_lock lock(mutex_);
                    while (!stop.stop_requested()) {
                        if (reconstruction_interrupted.load(std::memory_order_relaxed)) {
                            source_.request_stop();
                            return;
                        }
                        wake_.wait_for(lock, stop, std::chrono::milliseconds(20), [] { return false; });
                    }
                });
            }
            ~ReconstructionInterrupts() {
                monitor_.request_stop();
                wake_.notify_all();
                monitor_.join();
#ifdef _WIN32
                if (console_installed_)
                    SetConsoleCtrlHandler(reconstruction_console_handler, FALSE);
#endif
                if (previous_int_ != SIG_ERR)
                    std::signal(SIGINT, previous_int_);
                if (previous_term_ != SIG_ERR)
                    std::signal(SIGTERM, previous_term_);
            }
            std::stop_token token() const { return source_.get_token(); }

        private:
            using Handler = void (*)(int);
            Handler previous_int_ = SIG_ERR, previous_term_ = SIG_ERR;
            std::stop_source source_;
            std::mutex mutex_;
            std::condition_variable_any wake_;
            std::jthread monitor_;
#ifdef _WIN32
            bool console_installed_ = false;
#endif
        };

        size_t positive_integer(const std::string& value, const std::string& option) {
            size_t result = 0;
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), result);
            if (ec != std::errc{} || end != value.data() + value.size() || result == 0)
                throw std::invalid_argument(option + " requires a positive integer");
            return result;
        }

        bool binary_option(const std::map<std::string, std::string>& options, const std::string& key, bool fallback) {
            if (!options.contains(key))
                return fallback;
            const auto& value = options.at(key);
            if (value != "0" && value != "1")
                throw std::invalid_argument(key + " must be 0 or 1");
            return value == "1";
        }

        Json read_reconstruction_json(const std::filesystem::path& path) {
            std::ifstream stream(path, std::ios::binary);
            if (!stream)
                throw std::invalid_argument("Cannot open reconstruction config: " + core::path_to_utf8(path));
            std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
            if (path.extension() == ".bin") {
                const auto decrypted = core::path_crypto::decrypt_text_from_binary(
                    std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
                if (!decrypted)
                    throw std::invalid_argument("Cannot decrypt reconstruction config: " + core::path_to_utf8(path));
                text = *decrypted;
            } else if (path.extension() != ".json") {
                throw std::invalid_argument("Reconstruction config must be .bin or .json");
            }
            std::vector<std::set<std::string>> keys;
            return Json::parse(text, [&](int, Json::parse_event_t event, Json& value) {
                if (event == Json::parse_event_t::object_start)
                    keys.emplace_back();
                else if (event == Json::parse_event_t::object_end)
                    keys.pop_back();
                else if (event == Json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
                    throw std::invalid_argument("Duplicate reconstruction config key: " + value.get<std::string>());
                return true;
            });
        }

        enum class ConfigType { Boolean,
                                PositiveInteger,
                                PositiveNumber,
                                String };

        struct ConfigField {
            const char* option;
            ConfigType type;
        };

        using ConfigFields = std::map<std::string, ConfigField>;

        const std::map<std::string, ConfigFields> reconstruction_config_fields{
            {"reconstruction", {{"workspace_root", {"--workspace", ConfigType::String}}, {"input_mode", {"--input-mode", ConfigType::String}}, {"preprocess_only", {"--preprocess-only", ConfigType::Boolean}}, {"retain_failed_stages", {"--keep-failed-stages", ConfigType::Boolean}}, {"write_diagnostic_json", {"", ConfigType::Boolean}}}},
            {"common", {{"debug_mesh_overlay", {"--debug-mesh-overlay", ConfigType::Boolean}}, {"scan_blur_filter", {"--scan-blur-filter", ConfigType::Boolean}}, {"verbose_output", {"", ConfigType::Boolean}}}},
            {"scan", {{"sample_limit", {"--sample-limit", ConfigType::PositiveInteger}}, {"enhance", {"--enhance", ConfigType::Boolean}}, {"denoise", {"--denoise", ConfigType::Boolean}}, {"super_resolution", {"--super-resolution", ConfigType::Boolean}}, {"sr_final_scale", {"--sr-final-scale", ConfigType::PositiveInteger}}}},
            {"registered", {{"triangulation", {"--triangulation", ConfigType::String}}, {"sample_limit", {"--sample-limit", ConfigType::PositiveInteger}}, {"match_3d_threshold", {"--match-3d-threshold", ConfigType::PositiveNumber}}, {"inc_blur_filter", {"--inc-blur-filter", ConfigType::Boolean}}, {"export_registration_triplets", {"--export-registration-triplets", ConfigType::Boolean}}, {"register_twice", {"--register-twice", ConfigType::Boolean}}, {"face_mode", {"--face_mode", ConfigType::Boolean}}, {"incremental_mask", {"--incremental-mask", ConfigType::Boolean}}, {"foreground_model", {"--foreground-model", ConfigType::String}}, {"incremental_video", {"--video-inc-path", ConfigType::String}}, {"video_frame_count", {"--video-frame-count", ConfigType::PositiveInteger}}}},
            {"mesh_export", {{"enhance", {"--enhance", ConfigType::Boolean}}, {"super_resolution", {"--super-resolution", ConfigType::Boolean}}, {"sr_final_scale", {"--sr-final-scale", ConfigType::PositiveInteger}}, {"match_3d_threshold", {"--match-3d-threshold", ConfigType::PositiveNumber}}}}};

        Json read_swaptexture_config() {
            auto path = core::getRuntimeDataDir() / "Swaptexture_params.bin";
#ifdef _WIN32
            if (const auto* override_path = _wgetenv(L"LFS_SWAPTEXTURE_CONFIG"); override_path && *override_path)
                path = override_path;
#else
            if (const auto* override_path = std::getenv("LFS_SWAPTEXTURE_CONFIG"); override_path && *override_path)
                path = core::utf8_to_path(override_path);
#endif
            else if (!std::filesystem::exists(path))
                path.replace_extension(".json");
            auto config = read_reconstruction_json(path);
            if (!config.is_object() || !config.contains("schema_version") ||
                !config["schema_version"].is_number_integer() || config["schema_version"] != 1)
                throw std::invalid_argument("Swaptexture config requires schema_version 1");
            for (const auto& [section, values] : config.items()) {
                if (section == "schema_version")
                    continue;
                const auto fields = reconstruction_config_fields.find(section);
                if (fields == reconstruction_config_fields.end() || !values.is_object())
                    throw std::invalid_argument("Invalid Swaptexture config section: " + section);
                for (const auto& [name, value] : values.items()) {
                    const auto field = fields->second.find(name);
                    if (field == fields->second.end())
                        throw std::invalid_argument("Unknown Swaptexture config field: " + section + "." + name);
                    bool valid = false;
                    switch (field->second.type) {
                    case ConfigType::Boolean: valid = value.is_boolean(); break;
                    case ConfigType::String: valid = value.is_string(); break;
                    case ConfigType::PositiveInteger:
                        valid = value.is_number_integer() && value > 0 && value <= (std::numeric_limits<int>::max)();
                        break;
                    case ConfigType::PositiveNumber:
                        valid = value.is_number() && std::isfinite(value.get<double>()) && value.get<double>() > 0;
                        break;
                    }
                    if (!valid)
                        throw std::invalid_argument("Invalid Swaptexture config value: " + section + "." + name);
                    if (name == "sr_final_scale" && value > 4)
                        throw std::invalid_argument(section + ".sr_final_scale must be 1..4");
                    if (name == "triangulation" && value != "mesh" && value != "colmap")
                        throw std::invalid_argument("registered.triangulation must be mesh or colmap");
                    if (name == "input_mode" && value != "auto" && value != "scan" && value != "registered" && value != "mesh-export")
                        throw std::invalid_argument("reconstruction.input_mode must be auto, scan, registered or mesh-export");
                }
            }
            return config;
        }

        Json training_parameters(const core::param::TrainingParameters& params) {
            return {{"dataset", params.dataset.to_json()}, {"optimization", params.optimization.to_json()}, {"use_mesh_init", params.use_mesh_init}, {"mesh_init_mesh_path", params.mesh_init_mesh_path.value_or("")}, {"mesh_init_texture_path", params.mesh_init_texture_path.value_or("")}, {"mesh_init_sampling_rate", params.mesh_init_sampling_rate}, {"init_white_gs_color", params.init_white_gs_color}};
        }

        void write_report(const std::filesystem::path& path, const Json& report) {
            const auto temporary = path.parent_path() / ("." + path.stem().string() + ".tmp");
            {
                std::ofstream stream(temporary, std::ios::binary | std::ios::noreplace);
                if (!stream)
                    throw std::runtime_error("Cannot open reconstruction report: " + core::path_to_utf8(temporary));
                stream << report.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
                stream.close();
                if (!stream) {
                    std::error_code ignored;
                    std::filesystem::remove(temporary, ignored);
                    throw std::runtime_error("Cannot write reconstruction report: " + core::path_to_utf8(temporary));
                }
            }
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const auto code = GetLastError();
                std::error_code ignored;
                std::filesystem::remove(temporary, ignored);
                throw std::runtime_error("Cannot publish reconstruction report: " + std::to_string(code));
            }
#else
            std::filesystem::rename(temporary, path);
#endif
        }

        Json ply_artifact(const std::filesystem::path& path) {
            std::ifstream stream(path, std::ios::binary);
            std::string line;
            std::getline(stream, line);
            if (line != "ply" && line != "ply\r")
                throw std::runtime_error("Training artifact is not PLY");
            size_t count = 0;
            bool end_header = false;
            for (int i = 0; i < 1024 && std::getline(stream, line); ++i) {
                if (line.starts_with("element vertex ")) {
                    if (line.ends_with('\r'))
                        line.pop_back();
                    count = positive_integer(line.substr(15), "PLY vertex count");
                }
                if (line == "end_header" || line == "end_header\r") {
                    end_header = true;
                    break;
                }
            }
            if (!end_header || count == 0 || stream.peek() == std::char_traits<char>::eof())
                throw std::runtime_error("Training PLY has no vertices/payload");
            return {{"path", core::path_to_utf8(path)}, {"size_bytes", std::filesystem::file_size(path)}, {"vertex_count", count}, {"provenance", "in_process_run_training"}};
        }
    } // namespace

    static int run_fixture_reconstruction(const std::vector<std::string>& arguments) {
        try {
            if (arguments == std::vector<std::string>{"--help"}) {
#if LFS_ENABLE_CLI_HELP
                std::cout << "reconstruct --fixture-dataset <prepared COLMAP dataset> --workspace <root>\n"
                             "  [--input-mode scan|registered] [--quality fast|medium|quality]\n"
                             "  [--mesh-init-scene <mesh>] [--result-ply <file>] [-- <training options>]\n"
                             "Prepared fixture entry; M2 validates artifacts, raw orchestration remains M3-M4.\n";
                return 0;
#else
                return 0;
#endif
            }
            const std::set<std::string> names{"--fixture-dataset", "--workspace", "--input-mode", "--quality", "--mesh-init-scene", "--result-ply"};
            std::map<std::string, std::string> options;
            std::vector<std::string> training_args;
            for (size_t i = 0; i < arguments.size(); ++i) {
                const auto& key = arguments[i];
                if (key == "--") {
                    training_args.assign(arguments.begin() + i + 1, arguments.end());
                    break;
                }
                if (!names.contains(key)) {
                    std::cerr << "UnsupportedFeature: raw reconstruction option is not implemented: " << key << "\n";
                    return 2;
                }
                if (options.contains(key) || i + 1 >= arguments.size() || arguments[i + 1].starts_with("--"))
                    throw std::invalid_argument("Missing/duplicate reconstruction option: " + key);
                options[key] = arguments[++i];
            }
            if (!options.contains("--fixture-dataset") || !options.contains("--workspace")) {
                std::cerr << "UnsupportedFeature: requires --fixture-dataset and --workspace; raw orchestration belongs to M3-M4\n";
                return 2;
            }
            // Do not let training arguments override paths owned by orchestration.
            for (const auto& arg : training_args) {
                const auto name = arg.substr(0, arg.find('='));
                if (name == "-d" || name == "--data-path" || name == "-o" || name == "--output-path" ||
                    name == "--resume" || name == "-v" || name == "--view" || name == "--strategy" ||
                    (arg.starts_with("-d") && !arg.starts_with("--")) ||
                    (arg.starts_with("-o") && !arg.starts_with("--")))
                    throw std::invalid_argument("Training option conflicts with reconstruction ownership: " + arg);
            }
            ReconstructionRequest request;
            request.prepared_fixture = core::utf8_to_path(options.at("--fixture-dataset"));
            request.preprocessing.workspace_root = core::utf8_to_path(options.at("--workspace"));
            const auto mode = options.contains("--input-mode") ? options.at("--input-mode") : "scan";
            if (mode != "scan" && mode != "registered")
                throw std::invalid_argument("input-mode must be scan or registered");
            request.preprocessing.input = mode == "scan" ? preprocess::InputRequest{preprocess::ScanInput{}} : preprocess::InputRequest{preprocess::RegisteredInput{}};
            const auto quality = options.contains("--quality") ? options.at("--quality") : "fast";
            if (quality != "fast" && quality != "medium" && quality != "quality")
                throw std::invalid_argument("Invalid training quality");
            request.quality = quality == "fast" ? TrainingQuality::Fast : quality == "medium" ? TrainingQuality::Medium
                                                                                              : TrainingQuality::Quality;
            if (options.contains("--result-ply"))
                request.final_ply = core::utf8_to_path(options.at("--result-ply"));
            if (options.contains("--mesh-init-scene")) {
                request.mesh_init_scene = core::utf8_to_path(options.at("--mesh-init-scene"));
                if (auto mesh = preprocess::validate_mesh_input(request.mesh_init_scene); !mesh)
                    throw std::invalid_argument(mesh.error().message);
            }
            const auto valid = preprocess::validate_dataset_skeleton(*request.prepared_fixture);
            if (!valid) {
                std::cerr << "InvalidDataset: " << valid.error().message << "\n";
                return 2;
            }
            std::vector<std::filesystem::path> read_only_inputs{*request.prepared_fixture};
            if (!request.mesh_init_scene.empty())
                read_only_inputs.push_back(request.mesh_init_scene);
            const auto workspace = preprocess::Workspace::create(request.preprocessing.workspace_root, read_only_inputs);
            if (!workspace)
                throw std::runtime_error(workspace.error().message);
            preprocess::PreprocessResult prepared{.run_root = (*workspace)->run_root(), .dataset_root = std::filesystem::absolute(*request.prepared_fixture), .input_mode = mode == "scan" ? preprocess::InputMode::Scan : preprocess::InputMode::Registered};
            const int iterations = mode == "scan" ? (quality == "fast" ? 4000 : quality == "medium" ? 6000
                                                                                                    : 8000)
                                                  : (quality == "fast" ? 5000 : quality == "medium" ? 8000
                                                                                                    : 10000);
            std::vector<std::string> args{"Run-GS", "-d", core::path_to_utf8(prepared.dataset_root),
                                          "-o", core::path_to_utf8(prepared.run_root / "GaussianSplatting")};
            // Explicit fixture training options take precedence over the fixture default iteration preset.
            bool explicit_iterations = false;
            for (const auto& arg : training_args)
                if (arg == "-i" || arg == "--iter" || arg.starts_with("--iter="))
                    explicit_iterations = true;
            if (!explicit_iterations) {
                args.push_back("--iter");
                args.push_back(std::to_string(iterations));
            }
            if (!request.mesh_init_scene.empty()) {
                args.push_back("--mesh-init-gs-scene");
                args.push_back(core::path_to_utf8(request.mesh_init_scene));
            }
            args.insert(args.end(), training_args.begin(), training_args.end());
            std::vector<const char*> argv;
            for (const auto& arg : args)
                argv.push_back(arg.c_str());
            auto parsed = core::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
            if (!parsed)
                throw std::invalid_argument(parsed.error());
            request.training = std::move(**parsed);
            if (request.training.optimization.strategy == "mesh2splat")
                throw std::invalid_argument("mesh2splat conversion is not fixture training");
            std::cout << "Reconstruction run root: " << core::path_to_utf8(prepared.run_root) << "\n";
            auto result = run_training(std::move(request.training));
            if (!result) {
                std::cerr << result.error().message << "\n";
                return result.error().exit_code;
            }
            if (result->cancelled)
                return 130;
            if (request.final_ply) {
                auto publication = preprocess::publish_ply(result->gaussian_ply, *request.final_ply);
                if (!publication) {
                    std::cerr << "PublishFailure: " << publication.error().message << "\n";
                    return 1;
                }
            }
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Reconstruction failed: " << e.what() << "\n";
            return 1;
        }
    }
    static int run_configured_reconstruction(const std::vector<std::string>& arguments,
                                             const std::filesystem::path& log_directory,
                                             const bool configured_verbose_output = false,
                                             const bool write_diagnostic_json = true) {
        // Keep the already verified prepared-dataset entry independent of raw scan
        // defaults. In particular, fixtures still use fast and do not load a preset.
        std::filesystem::path report_path;
        std::unique_ptr<ReconstructionLog> log;
        Json report;
        const auto persist_report = [&] {
            if (write_diagnostic_json)
                write_report(report_path, report);
        };
        try {
            const bool detailed_output = detailed_reconstruction_output(configured_verbose_output);
            if (arguments == std::vector<std::string>{"--help"}) {
#if LFS_ENABLE_CLI_HELP
                std::cout << "reconstruct --bin-path <scanner bin> --workspace <root>\n"
                             "  [--mesh-init-gs-scene <ignored>] (compatibility only; initialization mesh comes from bin)\n"
                             "  [--input-mode scan|registered|mesh-export] [--quality fast|medium|quality] [--sample-limit N]\n"
                             "  [--super-resolution 0|1] [--sr-final-scale 1..4] [--enhance 0|1] [--denoise 0|1]\n"
                             "  [--debug-mesh-overlay 0|1] (diagnostic PNGs only; never modifies training images)\n"
                             "  [--scan-blur-filter 0|1] [--inc-blur-filter 0|1] (incremental filter is registered-only)\n"
                             "  Registered: --images-inc-path <directory> [--triangulation mesh|colmap]\n"
                             "    or --video-inc-path <local video> [--video-frame-count N] (default 80)\n"
                             "    [--export-registration-triplets 0|1]\n"
                             "    [--incremental-mask 0|1 --foreground-model <u2netp.onnx>] (CPU, SHA-256 locked)\n"
                             "    [--register-twice 0|1] [--face_mode] (frozen compatibility no-op)\n"
                             "    [--match-3d-threshold <meters>] (default mesh raycast, 0.0004 meters)\n"
                             "  Mesh export (terminal): --input-mode mesh-export --preprocess-only\n"
                             "    [--super-resolution 0|1] [--sr-final-scale 1..4] (frozen SR default 2) [--enhance 0|1]\n"
                             "    [--scan-blur-filter 0|1] [--match-3d-threshold <meters>]\n"
                             "  [--keep-failed-stages 0|1] [--preprocess-only] [--result-ply <file>] [-- <training options>]\n"
                             "Scan defaults: quality, 200 frames, enhancement on, SR on (final scale 2), denoise off.\n"
                             "Frozen native scan iterations: fast=4000, medium=6000, quality=10000.\n"
                             "Frozen native registered iterations: fast=5000, medium=8000, quality=12000.\n"
                             "Registered scan seed: 120 frames (override with --sample-limit); enhancement/SR options are scan-only.\n"
                             "Prepared fixture compatibility: --fixture-dataset <COLMAP dataset> replaces --bin-path;\n"
                             "  fixture default quality is fast, scan|registered accepted, --mesh-init-scene optional.\n";
#endif
                return 0;
            }
            const std::set<std::string> names{"--bin-path", "--workspace", "--input-mode", "--quality", "--mesh-init-scene",
                                              "--mesh-init-gs-scene", "--result-ply", "--sample-limit", "--super-resolution",
                                              "--sr-final-scale", "--enhance", "--denoise", "--keep-failed-stages",
                                              "--images-inc-path", "--incremental-images", "--triangulation", "--match-3d-threshold", "--debug-mesh-overlay",
                                              "--scan-blur-filter", "--inc-blur-filter", "--export-registration-triplets",
                                              "--video-inc-path", "--incremental-video", "--video-frame-count",
                                              "--incremental-mask", "--foreground-model", "--register-twice"};
            std::map<std::string, std::string> options;
            std::vector<std::string> training_args;
            bool preprocess_only = false;
            bool face_mode = false;
            for (size_t i = 0; i < arguments.size(); ++i) {
                const auto& key = arguments[i];
                if (key == "--") {
                    training_args.assign(arguments.begin() + i + 1, arguments.end());
                    break;
                }
                if (key == "--preprocess-only") {
                    if (preprocess_only)
                        throw std::invalid_argument("Duplicate --preprocess-only");
                    preprocess_only = true;
                    continue;
                }
                if (key == "--face_mode") {
                    if (face_mode)
                        throw std::invalid_argument("Duplicate --face_mode");
                    face_mode = true;
                    continue;
                }
                if (!names.contains(key)) {
                    std::cerr << "UnsupportedFeature: reconstruction option is not implemented: " << key << '\n';
                    return 2;
                }
                if (options.contains(key) || i + 1 >= arguments.size() || arguments[i + 1].starts_with("--"))
                    throw std::invalid_argument("Missing/duplicate reconstruction option: " + key);
                options[key] = arguments[++i];
            }
            if (!options.contains("--workspace") || !options.contains("--bin-path"))
                throw std::invalid_argument("Requires --bin-path and --workspace (or use --fixture-dataset)");
            if (preprocess_only && (!training_args.empty() || options.contains("--result-ply")))
                throw std::invalid_argument("--preprocess-only cannot take training options or --result-ply");
            const auto mode = options.contains("--input-mode") ? options.at("--input-mode") : "scan";
            if (mode != "scan" && mode != "registered" && mode != "mesh-export")
                throw std::invalid_argument("input-mode must be scan, registered, or mesh-export");
            const bool registered = mode == "registered";
            const bool mesh_export = mode == "mesh-export";
            if (face_mode && !registered)
                throw std::invalid_argument("--face_mode is registered-only");
            if (registered) {
                for (const auto* key : {"--super-resolution", "--sr-final-scale", "--enhance", "--denoise"})
                    if (options.contains(key)) {
                        std::cerr << "UnsupportedFeature: scan-only option in registered mode: " << key << '\n';
                        return 2;
                    }
                const auto source_count = int(options.contains("--images-inc-path")) + int(options.contains("--incremental-images")) +
                                          int(options.contains("--video-inc-path")) + int(options.contains("--incremental-video"));
                if (source_count != 1)
                    throw std::invalid_argument("Registered mode requires exactly one images or video source; aliases cannot be combined");
                if (options.contains("--video-frame-count") && !options.contains("--video-inc-path") && !options.contains("--incremental-video"))
                    throw std::invalid_argument("--video-frame-count requires a video input");
            } else if (mesh_export) {
                if (!preprocess_only)
                    throw std::invalid_argument("--input-mode mesh-export requires --preprocess-only");
                for (const auto* key : {"--images-inc-path", "--incremental-images", "--triangulation", "--inc-blur-filter",
                                        "--video-inc-path", "--incremental-video", "--video-frame-count", "--export-registration-triplets",
                                        "--incremental-mask", "--foreground-model", "--debug-mesh-overlay", "--denoise", "--sample-limit",
                                        "--quality", "--register-twice"})
                    if (options.contains(key))
                        throw std::invalid_argument(std::string("Option is not valid in mesh-export mode: ") + key);
            } else {
                for (const auto* key : {"--images-inc-path", "--incremental-images", "--triangulation", "--match-3d-threshold", "--inc-blur-filter",
                                        "--video-inc-path", "--incremental-video", "--video-frame-count", "--export-registration-triplets",
                                        "--incremental-mask", "--foreground-model", "--register-twice"})
                    if (options.contains(key))
                        throw std::invalid_argument(std::string("Registered-only option in scan mode: ") + key);
            }
            // Paths, preset, initialization and mode are owned by orchestration.
            // Individual training settings are explicit, reportable overrides.
            const std::set<std::string> owned_training{"-d", "--data-path", "-o", "--output-path", "--resume", "-v", "--view",
                                                       "--strategy", "--config", "--init", "--import-cameras", "--mesh-init-gs-scene",
                                                       "--mesh2splat-dir", "--python-script", "--debug-python", "--help", "-h", "--complete",
                                                       "--version", "-V", "--images", "--log-file", "--timelapse-images",
                                                       "--mesh-init-external-filled-mesh", "--mesh-init-external-pointcloud", "--ppisp-sidecar"};
            for (const auto& arg : training_args) {
                const auto name = arg.substr(0, arg.find('='));
                // Do not allow short-flag bundles to smuggle in -d/-o/-v/-h.
                // Only the parser's training aliases -i, -r and -q are useful here;
                // negative numeric values remain valid arguments to long options.
                if (arg.size() > 1 && arg[0] == '-' && arg[1] != '-' &&
                    arg[1] != 'i' && arg[1] != 'r' && arg != "-q" &&
                    !(arg[1] >= '0' && arg[1] <= '9') && arg[1] != '.')
                    throw std::invalid_argument("Unsupported short reconstruction training option: " + arg);
                if (owned_training.contains(name) ||
                    (arg.starts_with("-d") && !arg.starts_with("--")) ||
                    (arg.starts_with("-o") && !arg.starts_with("--")))
                    throw std::invalid_argument("Training option conflicts with reconstruction ownership: " + arg);
            }

            ReconstructionRequest request;
            request.preprocessing.workspace_root = core::utf8_to_path(options.at("--workspace"));
            request.preprocessing.scanner_bin = core::utf8_to_path(core::path_crypto::decrypt_path_or_original(options.at("--bin-path")));
            if (!std::filesystem::is_regular_file(request.preprocessing.scanner_bin))
                throw std::invalid_argument("Scanner bin not found: " + core::path_to_utf8(request.preprocessing.scanner_bin));
            if (registered) {
                preprocess::RegisteredInput input;
                const bool video = options.contains("--video-inc-path") || options.contains("--incremental-video");
                const auto source_key = video ? (options.contains("--video-inc-path") ? "--video-inc-path" : "--incremental-video")
                                              : (options.contains("--images-inc-path") ? "--images-inc-path" : "--incremental-images");
                const auto source_path = std::filesystem::absolute(core::utf8_to_path(core::path_crypto::decrypt_path_or_original(options.at(source_key))));
                if (video) {
                    if (!std::filesystem::is_regular_file(source_path))
                        throw std::invalid_argument("Incremental video file not found: " + core::path_to_utf8(source_path));
                    input.incremental_video = source_path;
                    if (options.contains("--video-frame-count")) {
                        const auto count = positive_integer(options.at("--video-frame-count"), "--video-frame-count");
                        if (count > static_cast<size_t>((std::numeric_limits<int>::max)()))
                            throw std::invalid_argument("--video-frame-count exceeds supported integer range");
                        input.video_frame_count = static_cast<int>(count);
                    }
                } else {
                    if (!std::filesystem::is_directory(source_path))
                        throw std::invalid_argument("Incremental images directory not found: " + core::path_to_utf8(source_path));
                    input.incremental_images = source_path;
                }
                if (const auto isolated = preprocess::check_output_isolation(request.preprocessing.workspace_root, {source_path}); !isolated)
                    throw std::invalid_argument(isolated.error().message);
                const auto triangulation = options.contains("--triangulation") ? options.at("--triangulation") : "mesh";
                if (triangulation != "mesh" && triangulation != "colmap")
                    throw std::invalid_argument("--triangulation must be mesh or colmap");
                input.options.use_mesh_triangulation = triangulation == "mesh";
                input.options.enable_mesh_overlay = binary_option(options, "--debug-mesh-overlay", false);
                input.options.enable_scan_blur_filter = binary_option(options, "--scan-blur-filter", false);
                input.options.enable_blur_filter = binary_option(options, "--inc-blur-filter", false);
                input.options.enable_registration_triplets = binary_option(options, "--export-registration-triplets", false);
                input.options.register_twice = binary_option(options, "--register-twice", false);
                input.options.face_mode = face_mode;
                input.options.enable_incremental_mask = binary_option(options, "--incremental-mask", false);
                if (options.contains("--foreground-model")) {
                    if (!input.options.enable_incremental_mask)
                        throw std::invalid_argument("--foreground-model requires --incremental-mask 1");
                    input.options.foreground_model = std::filesystem::absolute(core::utf8_to_path(options.at("--foreground-model")));
                    if (!std::filesystem::is_regular_file(*input.options.foreground_model))
                        throw std::invalid_argument("--foreground-model file not found");
                    if (const auto isolated = preprocess::check_output_isolation(request.preprocessing.workspace_root, {*input.options.foreground_model}); !isolated)
                        throw std::invalid_argument(isolated.error().message);
                } else if (input.options.enable_incremental_mask) {
                    throw std::invalid_argument("--incremental-mask requires explicit --foreground-model (locked u2netp.onnx)");
                }
                input.options.match_3d_threshold = 0.0004;
                if (options.contains("--sample-limit"))
                    input.options.sample_limit = positive_integer(options.at("--sample-limit"), "--sample-limit");
                if (options.contains("--match-3d-threshold")) {
                    const auto& value = options.at("--match-3d-threshold");
                    double threshold = 0;
                    const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), threshold);
                    if (ec != std::errc{} || end != value.data() + value.size() || !std::isfinite(threshold) || threshold <= 0)
                        throw std::invalid_argument("--match-3d-threshold requires a finite positive number");
                    input.options.match_3d_threshold = threshold;
                }
                request.preprocessing.input = std::move(input);
            } else if (mesh_export) {
                preprocess::MeshExportInput input;
                input.options.enable_blur_filter = binary_option(options, "--scan-blur-filter", false);
                input.options.enhance_before_super_resolution = binary_option(options, "--enhance", true);
                input.options.super_resolution.enabled = binary_option(options, "--super-resolution", false);
                if (options.contains("--sr-final-scale")) {
                    const auto scale = positive_integer(options.at("--sr-final-scale"), "--sr-final-scale");
                    if (scale > static_cast<size_t>(input.options.super_resolution.model_scale))
                        throw std::invalid_argument("--sr-final-scale must not exceed model scale 4");
                    input.options.super_resolution.final_scale = static_cast<int>(scale);
                }
                if (options.contains("--match-3d-threshold")) {
                    const auto& value = options.at("--match-3d-threshold");
                    double threshold = 0;
                    const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), threshold);
                    if (ec != std::errc{} || end != value.data() + value.size() || !std::isfinite(threshold) || threshold <= 0)
                        throw std::invalid_argument("--match-3d-threshold requires a finite positive number");
                    input.options.match_3d_threshold = threshold;
                }
                request.preprocessing.input = std::move(input);
            }
            const auto quality = options.contains("--quality") ? options.at("--quality") : "quality";
            if (quality != "fast" && quality != "medium" && quality != "quality")
                throw std::invalid_argument("Invalid training quality");
            request.quality = quality == "fast" ? TrainingQuality::Fast : quality == "medium" ? TrainingQuality::Medium
                                                                                              : TrainingQuality::Quality;
            if (options.contains("--result-ply"))
                request.final_ply = core::utf8_to_path(options.at("--result-ply"));
            // Both legacy mesh options are accepted but ignored. The scanner
            // mesh is authoritative for preprocessing and GS initialization.
            if (!preprocess_only) {
                const auto scanner = io::read_scanner_bin(request.preprocessing.scanner_bin);
                if (!scanner)
                    throw std::invalid_argument(scanner.error());
                const auto mesh_path = io::resolve_scanner_mesh_path(request.preprocessing.scanner_bin, *scanner);
                if (!mesh_path)
                    throw std::invalid_argument("GS initialization requires a valid mesh from scanner bin: " + mesh_path.error());
                request.mesh_init_scene = *mesh_path;
                if (auto mesh = preprocess::validate_mesh_input(request.mesh_init_scene); !mesh)
                    throw std::invalid_argument(mesh.error().message);
                if (auto isolated = preprocess::check_output_isolation(request.preprocessing.workspace_root, {request.mesh_init_scene}); !isolated)
                    throw std::invalid_argument(isolated.error().message);
            }
            if (request.final_ply) {
                // Publication may replace an earlier result, never an input. Include
                // external scanner image paths as well as the bin/mesh asset folders.
                const auto scanner = io::read_scanner_bin(request.preprocessing.scanner_bin);
                if (!scanner)
                    throw std::invalid_argument(scanner.error());
                std::vector<std::filesystem::path> inputs{std::filesystem::absolute(request.preprocessing.scanner_bin).parent_path()};
                if (registered) {
                    const auto& registered_input = std::get<preprocess::RegisteredInput>(request.preprocessing.input);
                    inputs.push_back(registered_input.incremental_video ? *registered_input.incremental_video : *registered_input.incremental_images);
                    if (registered_input.options.foreground_model)
                        inputs.push_back(*registered_input.options.foreground_model);
                }
                if (registered || binary_option(options, "--debug-mesh-overlay", false)) {
                    const auto scanner_mesh = io::resolve_scanner_mesh_path(request.preprocessing.scanner_bin, *scanner);
                    if (scanner_mesh)
                        inputs.push_back(*scanner_mesh);
                }
                if (!request.mesh_init_scene.empty())
                    inputs.push_back(std::filesystem::is_directory(request.mesh_init_scene) ? request.mesh_init_scene : request.mesh_init_scene.parent_path());
                size_t index = 0;
                for (const auto& group : scanner->groups)
                    for (const auto& frame : group.frames) {
                        auto image = core::utf8_to_path(frame.image_path_utf8);
                        std::error_code ec;
                        if (!std::filesystem::is_regular_file(image, ec))
                            image = request.preprocessing.scanner_bin.parent_path() / (std::to_string(index) + ".png");
                        inputs.push_back(image);
                        ++index;
                    }
                if (const auto isolated = preprocess::check_output_isolation(*request.final_ply, inputs); !isolated)
                    throw std::invalid_argument("Unsafe final PLY destination: " + isolated.error().message);
            }
            if (!registered && !mesh_export) {
                auto& scan = std::get<preprocess::ScanInput>(request.preprocessing.input).options;
                if (options.contains("--sample-limit"))
                    scan.sample_limit = positive_integer(options.at("--sample-limit"), "--sample-limit");
                scan.enable_enhancement = binary_option(options, "--enhance", true);
                scan.enable_mesh_overlay = binary_option(options, "--debug-mesh-overlay", false);
                scan.enable_blur_filter = binary_option(options, "--scan-blur-filter", false);
                scan.enable_denoise = binary_option(options, "--denoise", false);
                scan.super_resolution.enabled = binary_option(options, "--super-resolution", true);
                if (options.contains("--sr-final-scale")) {
                    const auto scale = positive_integer(options.at("--sr-final-scale"), "--sr-final-scale");
                    if (scale > static_cast<size_t>(scan.super_resolution.model_scale))
                        throw std::invalid_argument("--sr-final-scale must not exceed model scale 4");
                    scan.super_resolution.final_scale = static_cast<int>(scale);
                }
            }
            request.preprocessing.retain_failed_stages = binary_option(options, "--keep-failed-stages", false);
            request.preprocessing.write_diagnostic_json = write_diagnostic_json;
            std::vector<std::filesystem::path> log_inputs{std::filesystem::absolute(request.preprocessing.scanner_bin).parent_path()};
            if (!request.mesh_init_scene.empty())
                log_inputs.push_back(request.mesh_init_scene);
            if (registered) {
                const auto& input = std::get<preprocess::RegisteredInput>(request.preprocessing.input);
                log_inputs.push_back(input.incremental_video ? *input.incremental_video : *input.incremental_images);
                if (input.options.foreground_model)
                    log_inputs.push_back(*input.options.foreground_model);
            }
            if (const auto isolated = preprocess::check_output_isolation(log_directory, log_inputs); !isolated)
                throw std::invalid_argument("Unsafe log directory: " + isolated.error().message);
            std::filesystem::create_directories(log_directory);
            log = std::make_unique<ReconstructionLog>(log_directory);
            const auto progress_mode = mesh_export ? ReconstructionInputMode::NoGs : registered ? ReconstructionInputMode::Registered : ReconstructionInputMode::Scan;
            const bool has_video = registered && std::get<preprocess::RegisteredInput>(request.preprocessing.input).incremental_video.has_value();
            const bool has_enhancement = !registered && !mesh_export && std::get<preprocess::ScanInput>(request.preprocessing.input).options.enable_enhancement;
            const bool has_image_processing = !registered && !mesh_export &&
                                              (std::get<preprocess::ScanInput>(request.preprocessing.input).options.enable_enhancement ||
                                               std::get<preprocess::ScanInput>(request.preprocessing.input).options.enable_denoise);
            ReconstructionProgress total_progress(progress_mode, has_video, has_enhancement);
            ReconstructionOutput status_output(
                detailed_output, std::cout,
                reconstruction_stage_order(progress_mode, has_video, has_image_processing),
                !mesh_export && !preprocess_only);
            SharedProgressMemory shared_progress;
            const auto publish_progress = [&](float value) {
                shared_progress.write(value);
                report["progress"] = value;
            };
            publish_progress(0.0F);
            if (mesh_export) {
                ReconstructionInterrupts interrupts;
                preprocess::ExecutionContext context{.stop_token = interrupts.token()};
                context.on_log = [detailed_output](preprocess::Stage, std::string message) {
                    if (detailed_output)
                        std::cout << message << '\n';
                };
                context.on_stage_progress = [&](std::string_view name, preprocess::Stage, float value, std::string) {
                    const float overall = total_progress.update_stage(name, value);
                    publish_progress(overall);
                    status_output.stage_progress(name, value, overall);
                    return true;
                };
                auto preprocessed = preprocess::preprocess(request.preprocessing, context);
                if (!preprocessed) {
                    std::cerr << "Preprocessing failed: " << preprocessed.error().message << '\n';
                    return preprocess::native_cli_exit_code(preprocessed.error().code);
                }
                const auto& prepared = *preprocessed;
                const auto valid = preprocess::validate_dataset(prepared.dataset_root);
                if (!valid)
                    throw std::runtime_error(valid.error().message);
                if (prepared.input_mode != preprocess::InputMode::MeshExport ||
                    !prepared.mesh_export_root || *prepared.mesh_export_root != prepared.dataset_root)
                    throw std::runtime_error("Mesh export runner returned an invalid terminal artifact root");
                std::set<std::string> artifact_names;
                for (const auto& item : std::filesystem::directory_iterator(prepared.dataset_root))
                    artifact_names.insert(core::path_to_utf8(item.path().filename()));
                if (artifact_names != std::set<std::string>{"images", "mask", "sparse"})
                    throw std::runtime_error("Mesh export root must contain exactly images/mask/sparse");
                report_path = prepared.run_root / "mesh_export_e2e_report.json";
                report = {{"schema_name", "lfs.swaptexture.mesh-export-e2e-runtime"},
                          {"schema_version", 1},
                          {"producer_version", "m5.1"},
                          {"input_mode", "mesh-export"},
                          {"status", "preprocessed"},
                          {"run_root", core::path_to_utf8(prepared.run_root)},
                          {"dataset_root", core::path_to_utf8(prepared.dataset_root)},
                          {"preprocess_report", core::path_to_utf8(prepared.debug_report)},
                          {"artifacts",
                           {{"image_count", prepared.image_count},
                            {"mask_count", prepared.image_count},
                            {"sparse_image_count", valid->image_count},
                            {"sparse_file_count", 3},
                            {"camera_count", valid->camera_count},
                            {"has_points", valid->has_points}}},
                          {"super_resolution", std::get<preprocess::MeshExportInput>(request.preprocessing.input).options.super_resolution.enabled}};
                persist_report();
                const float completed_progress = total_progress.complete();
                publish_progress(completed_progress);
                status_output.completed(completed_progress);
                if (detailed_output) {
                    std::cout << "Mesh export run root: " << core::path_to_utf8(prepared.run_root) << '\n';
                    std::cout << "Mesh export dataset root: " << core::path_to_utf8(prepared.dataset_root) << std::endl;
                }
                return interrupts.token().stop_requested() ? 130 : 0;
            }
            const auto preset_source = registered ? "inc" : "scan";
            const auto preset_path = core::getRuntimeDataDir() / ("GS_params_" + std::string(preset_source) + "_" + quality + ".bin");
            const auto preset = read_reconstruction_json(preset_path);
            const int iterations = preset.at("iterations").get<int>();
            const int expected_iterations = registered ? (quality == "fast" ? 5000 : quality == "medium" ? 8000
                                                                                                         : 12000)
                                                       : (quality == "fast" ? 4000 : quality == "medium" ? 6000
                                                                                                         : 10000);
            const auto preset_strategy = preset.at("strategy").get<std::string>();
            if (iterations != expected_iterations || preset_strategy != "igs+")
                throw std::runtime_error("Bundled " + mode + " preset does not match selected quality");

            ReconstructionInterrupts interrupts;
            preprocess::ExecutionContext context{.stop_token = interrupts.token()};
            context.on_log = [detailed_output](preprocess::Stage, std::string message) {
                if (detailed_output)
                    std::cout << message << '\n';
            };
            context.on_stage_progress = [&](std::string_view name, preprocess::Stage, float value, std::string) {
                const float overall = total_progress.update_stage(name, value);
                publish_progress(overall);
                status_output.stage_progress(name, value, overall);
                return true;
            };
            auto preprocessed = preprocess::preprocess(request.preprocessing, context);
            if (!preprocessed) {
                std::cerr << "Preprocessing failed: " << preprocessed.error().message << '\n';
                return preprocess::native_cli_exit_code(preprocessed.error().code);
            }
            const auto& prepared = *preprocessed;
            publish_progress(total_progress.preprocessing_done());
            const auto valid = preprocess::validate_dataset(prepared.dataset_root);
            if (!valid)
                throw std::runtime_error(valid.error().message);
            if (registered && prepared.dataset_root != prepared.run_root)
                throw std::runtime_error("Registered dataset root must exactly equal its run root");
            report_path = prepared.run_root / (mode + "_e2e_report.json");
            report = {{"schema_version", 1}, {"input_mode", "scan"}, {"status", "preprocessed"}, {"run_root", core::path_to_utf8(prepared.run_root)}, {"dataset_root", core::path_to_utf8(prepared.dataset_root)}, {"preprocess_report", core::path_to_utf8(prepared.debug_report)}, {"training_data_path", nullptr}, {"training_output_path", nullptr}, {"quality", quality}, {"preset_path", core::path_to_utf8(preset_path)}, {"preset_parameters", preset}, {"preset_iterations", iterations}, {"requested_iterations", nullptr}, {"final_iteration", nullptr}, {"parsed_parameters", nullptr}, {"effective_parameters", nullptr}, {"training_overrides", training_args}, {"full_preset_run", false}, {"final_ply", nullptr}, {"error", nullptr}, {"artifacts", {{"image_count", valid->image_count}, {"camera_count", valid->camera_count}, {"has_points", valid->has_points}}}, {"comparisons", {{"same_dataset_path", nullptr}, {"preset_iterations_match", nullptr}, {"baseline_artifacts", nullptr}}}};
            report["input_mode"] = mode;
            report["schema_name"] = "lfs.swaptexture." + mode + "-e2e-runtime";
            report["producer_version"] = registered ? "m4.1" : "m3.1";
            report["progress"] = total_progress.value();
            if (registered && write_diagnostic_json) {
                const auto& compatibility = std::get<preprocess::RegisteredInput>(request.preprocessing.input).options;
                report["compatibility"] = {{"register_twice", compatibility.register_twice},
                                           {"face_mode", compatibility.face_mode},
                                           {"face_mode_behavior", "legacy_noop"},
                                           {"non_gs_texture_export", preprocess_only}};
                std::ifstream registration_stream(prepared.run_root / "registration_diagnostics.json", std::ios::binary);
                if (!registration_stream)
                    throw std::runtime_error("Registered preprocessing diagnostics are missing");
                report["registration"] = Json::parse(registration_stream);
                if (!report["registration"].is_object())
                    throw std::runtime_error("Registered preprocessing diagnostics must be an object");
            }
            // Frozen parity references from M0, not the revision of this executable.
            // The independent verifier records the actual executable SHA-256.
            report["source_baselines"] = {{"lichtfeld_commit", "e8ee38d4"},
                                          {"swaptexture_commit", "9180f87aabde0bdf8d0957006e53bdb42528c52d"}};
            persist_report();
            if (detailed_output)
                std::cout << "Reconstruction run root: " << core::path_to_utf8(prepared.run_root) << std::endl;
            if (preprocess_only) {
                if (interrupts.token().stop_requested()) {
                    report["status"] = "cancelled";
                    persist_report();
                    return 130;
                }
                const float completed_progress = total_progress.complete();
                publish_progress(completed_progress);
                persist_report();
                status_output.completed(completed_progress);
                return 0;
            }

            std::vector<std::string> args{"Run-GS", "-d", core::path_to_utf8(prepared.dataset_root),
                                          "-o", core::path_to_utf8(prepared.run_root / "GaussianSplatting"),
                                          "--config", core::path_to_utf8(preset_path),
                                          "--mesh-init-gs-scene", core::path_to_utf8(request.mesh_init_scene)};
            args.insert(args.end(), training_args.begin(), training_args.end());
            std::vector<const char*> argv;
            for (const auto& arg : args)
                argv.push_back(arg.c_str());
            auto parsed = core::args::parse_args_and_params(static_cast<int>(argv.size()), argv.data());
            if (!parsed)
                throw std::invalid_argument(parsed.error());
            if (detailed_output)
                core::Logger::get().set_level(core::LogLevel::Info);
            request.training = std::move(**parsed);
            if (request.training.optimization.strategy != preset_strategy || !request.training.use_mesh_init)
                throw std::invalid_argument("Native reconstruction requires the frozen preset strategy and mesh-initialized training");
            if (request.training.optimization.iterations == 0)
                throw std::invalid_argument("Native reconstruction training requires at least one iteration");
            if (request.training.dataset.data_path != prepared.dataset_root || request.training.dataset.output_path != prepared.run_root / "GaussianSplatting")
                throw std::runtime_error("Training parser changed an orchestration-owned path");
            report["status"] = "training";
            report["training_data_path"] = core::path_to_utf8(request.training.dataset.data_path);
            report["training_output_path"] = core::path_to_utf8(request.training.dataset.output_path);
            report["parsed_parameters"] = training_parameters(request.training);
            report["requested_iterations"] = request.training.optimization.iterations;
            report["comparisons"]["same_dataset_path"] = true;
            report["comparisons"]["preset_iterations_match"] = request.training.optimization.iterations == static_cast<size_t>(iterations);
            persist_report();
            TrainingCallbacks callbacks;
            callbacks.on_parameters = [&](const core::param::TrainingParameters& effective) {
                if (effective.dataset.data_path != prepared.dataset_root || effective.dataset.output_path != prepared.run_root / "GaussianSplatting")
                    throw std::runtime_error("Training runner changed an orchestration-owned path");
                report["effective_parameters"] = training_parameters(effective);
                std::filesystem::create_directories(effective.dataset.output_path);
                if (write_diagnostic_json) {
                    const auto saved = core::param::save_training_parameters_to_json(effective, effective.dataset.output_path);
                    if (!saved)
                        throw std::runtime_error(saved.error());
                }
                persist_report();
            };
            callbacks.on_progress = [&](float value) { publish_progress(total_progress.training(value)); };
            const auto training_iterations = request.training.optimization.iterations;
            callbacks.on_iteration = [&](int iteration, float loss, int gaussians) {
                status_output.training_progress(iteration, training_iterations, loss, gaussians,
                                                total_progress.value());
            };
            status_output.training_started(training_iterations, total_progress.value());
            auto result = run_training(std::move(request.training), interrupts.token(), std::move(callbacks));
            if (!result) {
                status_output.training_finished("failed", total_progress.value());
                report["status"] = "failed";
                report["error"] = result.error().message;
                persist_report();
                std::cerr << result.error().message << '\n';
                return result.error().exit_code;
            }
            status_output.training_finished(result->cancelled ? "cancelled" : "completed", total_progress.value());
            report["status"] = result->cancelled ? "cancelled" : "completed";
            report["final_iteration"] = result->final_iteration;
            if (!result->cancelled) {
                report["final_ply"] = ply_artifact(result->gaussian_ply);
                report["full_preset_run"] = training_args.empty() && report["requested_iterations"] == iterations && result->final_iteration == iterations;
            }
            persist_report();
            if (result->cancelled)
                return 130;
            if (request.final_ply) {
                auto publication = preprocess::publish_ply(result->gaussian_ply, *request.final_ply);
                if (!publication)
                    throw std::runtime_error("PublishFailure: " + publication.error().message);
            }
            const float completed_progress = total_progress.complete();
            publish_progress(completed_progress);
            status_output.completed(completed_progress);
            return 0;
        } catch (const std::exception& e) {
            if (!report_path.empty()) {
                report["status"] = "failed";
                report["full_preset_run"] = false;
                report["error"] = e.what();
                try {
                    persist_report();
                } catch (const std::exception& report_error) {
                    std::cerr << "Reconstruction report failure: " << report_error.what() << '\n';
                }
            }
            std::cerr << "Reconstruction failed: " << e.what() << '\n';
            return dynamic_cast<const std::invalid_argument*>(&e) ? 2 : 1;
        }
    }
    int run_reconstruction(const std::vector<std::string>& arguments, const bool swaptexture_cli) {
        if (!swaptexture_cli) {
            if (std::find(arguments.begin(), arguments.end(), "--fixture-dataset") != arguments.end())
                return run_fixture_reconstruction(arguments);
            return run_configured_reconstruction(arguments, core::getRuntimeDataDir() / "outputs");
        }
        try {
            if (arguments == std::vector<std::string>{"--help"}) {
#if LFS_ENABLE_CLI_HELP
                std::cout << "reconstruct --bin_path <scanner bin> [--images_inc_path <directory>]\n"
                             "  [--enable_gs_train] [--gs-input-source registered|scan]\n"
                             "  [--gs-training-mode fast|medium|quality] [--mesh-init-gs-scene <mesh>]\n"
                             "  [--result_path <directory or .ply>] [--log_path <directory>]\n"
                             "  [--enable_inc_sim3_registration] [-- <LichtFeld training options>]\n"
                             "Defaults: registered input, medium quality, GS training disabled.\n"
                             "GS initialization uses the scanner-bin mesh; --mesh-init-gs-scene is accepted but ignored.\n"
                             "Sim3 is not supported.\n"
                             "Other reconstruction settings are loaded from bin/Swaptexture_params.bin in the installed package.\n";
                return 0;
#else
                return 0;
#endif
            }
            const std::map<std::string, std::string> value_options{
                {"--bin_path", "--bin-path"},
                {"--images_inc_path", "--images-inc-path"},
                {"--gs-input-source", "--input-mode"},
                {"--gs-training-mode", "--quality"},
                {"--mesh-init-gs-scene", "--mesh-init-gs-scene"},
                {"--result_path", ""},
                {"--log_path", ""}};
            std::map<std::string, std::string> supplied;
            std::set<std::string> flags;
            std::vector<std::string> training_args;
            for (size_t i = 0; i < arguments.size(); ++i) {
                const auto& argument = arguments[i];
                if (argument == "--") {
                    training_args.assign(arguments.begin() + i + 1, arguments.end());
                    break;
                }
                const auto separator = argument.find('=');
                const auto key = argument.substr(0, separator);
                if (key == "--enable_gs_train" || key == "--enable_inc_sim3_registration") {
                    if (separator != std::string::npos || !flags.insert(key).second)
                        throw std::invalid_argument("Duplicate flag or unexpected flag value: " + key);
                    continue;
                }
                if (!value_options.contains(key))
                    throw std::invalid_argument("Unsupported reconstruction CLI option: " + key + "; use Swaptexture_params.bin for integration settings");
                if (supplied.contains(key) || (separator == std::string::npos &&
                                               (i + 1 >= arguments.size() || arguments[i + 1].starts_with("--"))))
                    throw std::invalid_argument("Missing/duplicate reconstruction option: " + key);
                auto value = separator == std::string::npos ? arguments[++i] : argument.substr(separator + 1);
                if (value.empty())
                    throw std::invalid_argument("Empty reconstruction option: " + key);
                supplied.emplace(key, std::move(value));
            }
            if (flags.contains("--enable_inc_sim3_registration"))
                throw std::invalid_argument("UnsupportedFeature: --enable_inc_sim3_registration requires the unavailable native Sim3 implementation");
            if (!supplied.contains("--bin_path"))
                throw std::invalid_argument("Requires --bin_path");
            const bool training = flags.contains("--enable_gs_train");
            if (!training && !training_args.empty())
                throw std::invalid_argument("Training options require --enable_gs_train");
            const auto source = supplied.contains("--gs-input-source") ? supplied.at("--gs-input-source") : "registered";
            if (source != "scan" && source != "registered")
                throw std::invalid_argument("--gs-input-source must be scan or registered");
            const auto quality = supplied.contains("--gs-training-mode") ? supplied.at("--gs-training-mode") : "medium";
            if (quality != "fast" && quality != "medium" && quality != "quality")
                throw std::invalid_argument("--gs-training-mode must be fast, medium or quality");

            const auto config = read_swaptexture_config();
            const auto reconstruction = config.value("reconstruction", Json::object());
            const auto configured_mode = reconstruction.value("input_mode", "auto");
            const auto mode = configured_mode == "auto" ? source : configured_mode;
            if (configured_mode != "auto" && supplied.contains("--gs-input-source") && mode != source)
                throw std::invalid_argument("--gs-input-source conflicts with reconstruction.input_mode");
            const bool configured_preprocess = reconstruction.value("preprocess_only", false);
            if (training && (configured_preprocess || mode == "mesh-export"))
                throw std::invalid_argument("--enable_gs_train conflicts with preprocessing-only JSON configuration");
            if (mode == "scan" && !training && !configured_preprocess)
                throw std::invalid_argument("Scan mode requires --enable_gs_train or reconstruction.preprocess_only in JSON");

            const auto runtime_data_dir = core::getRuntimeDataDir();
            const auto config_path = [&](const std::string& value) {
                const auto path = core::utf8_to_path(core::path_crypto::decrypt_path_or_original(value));
                return path.is_absolute() ? path : runtime_data_dir / path;
            };
            const auto cli_path = [](const std::string& value) {
                return core::utf8_to_path(core::path_crypto::decrypt_path_or_original(value));
            };
            const auto workspace_setting = reconstruction.value("workspace_root", "outputs");
            if (workspace_setting.empty())
                throw std::invalid_argument("reconstruction.workspace_root must not be empty");
            auto workspace = config_path(workspace_setting);
            std::map<std::string, std::string> options{{"--bin-path", supplied.at("--bin_path")}, {"--input-mode", mode}};
            if (mode != "mesh-export")
                options["--quality"] = quality;
            else if (supplied.contains("--gs-training-mode"))
                throw std::invalid_argument("--gs-training-mode is not valid in mesh-export mode");
            if (supplied.contains("--result_path")) {
                const auto result = cli_path(supplied.at("--result_path"));
                if (training) {
                    auto extension = result.extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                    if (extension != ".ply")
                        throw std::invalid_argument("--result_path must end in .ply when GS training is enabled");
                    options["--result-ply"] = core::path_to_utf8(result);
                } else {
                    if (result.extension() == ".ply")
                        throw std::invalid_argument("--result_path must be a directory when GS training is disabled");
                    workspace = result;
                }
            }
            options["--workspace"] = core::path_to_utf8(workspace);
            if (supplied.contains("--images_inc_path"))
                options["--images-inc-path"] = supplied.at("--images_inc_path");
            // --mesh-init-gs-scene remains parseable for existing callers, but
            // its value is neither resolved nor forwarded to reconstruction.
            options["--keep-failed-stages"] = reconstruction.value("retain_failed_stages", false) ? "1" : "0";
            bool face_mode = false;
            const auto apply_section = [&](const std::string& name) {
                if (!config.contains(name))
                    return;
                const auto& values = config.at(name);
                for (const auto& [field, value] : values.items()) {
                    if (field == "verbose_output" || field == "write_diagnostic_json")
                        continue;
                    if (field == "face_mode") {
                        face_mode = value.get<bool>();
                        continue;
                    }
                    if (field == "video_frame_count" && values.value("incremental_video", "").empty())
                        continue;
                    if (mode == "mesh-export" && field == "debug_mesh_overlay") {
                        if (value.get<bool>())
                            throw std::invalid_argument("common.debug_mesh_overlay is not supported in mesh-export mode");
                        continue;
                    }
                    const auto& definition = reconstruction_config_fields.at(name).at(field);
                    if (value.is_boolean())
                        options[definition.option] = value.get<bool>() ? "1" : "0";
                    else if (value.is_string()) {
                        auto text = value.get<std::string>();
                        if (text.empty())
                            continue;
                        if (field == "foreground_model" || field == "incremental_video")
                            text = core::path_to_utf8(config_path(text));
                        options[definition.option] = std::move(text);
                    } else
                        options[definition.option] = value.dump();
                }
            };
            apply_section("common");
            apply_section(mode == "mesh-export" ? "mesh_export" : mode);
            std::vector<std::string> configured;
            for (const auto& [key, value] : options) {
                configured.push_back(key);
                configured.push_back(value);
            }
            if (!training)
                configured.push_back("--preprocess-only");
            if (face_mode)
                configured.push_back("--face_mode");
            if (!training_args.empty()) {
                configured.push_back("--");
                configured.insert(configured.end(), training_args.begin(), training_args.end());
            }
            auto log_directory = runtime_data_dir / "outputs";
            if (supplied.contains("--log_path")) {
                log_directory = cli_path(supplied.at("--log_path"));
                auto extension = log_directory.extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                if (extension == ".log")
                    log_directory = std::filesystem::absolute(log_directory).parent_path();
            }
            const bool verbose_output = config.value("common", Json::object()).value("verbose_output", false);
            const bool write_diagnostic_json = reconstruction.value("write_diagnostic_json", false);
            return run_configured_reconstruction(configured, std::filesystem::absolute(log_directory), verbose_output,
                                                 write_diagnostic_json);
        } catch (const std::exception& error) {
            std::cerr << "Reconstruction failed: " << error.what() << '\n';
            return 2;
        }
    }
} // namespace lfs::app
