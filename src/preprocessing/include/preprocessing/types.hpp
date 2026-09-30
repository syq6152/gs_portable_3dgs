/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <variant>

namespace lfs::preprocess {

    enum class InputMode { Scan,
                           Registered,
                           MeshExport };

    enum class Stage { Validate,
                       ImageProcessing,
                       Colmap,
                       SuperResolution,
                       Assemble,
                       Training,
                       DebugExport };

    enum class ErrorCode {
        InvalidRequest,
        InvalidDataset,
        UnsupportedFeature,
        RuntimeMissing,
        RuntimeManifestMismatch,
        ProcessLaunchFailed,
        ProcessFailed,
        ProcessTimeout,
        ProcessCancelled,
        CallbackFailure,
        PlatformUnsupported,
        IncompatibleRuntime,
        IoFailure,
    };

    struct Error {
        ErrorCode code = ErrorCode::InvalidRequest;
        std::string message;
        int exit_code = 0;
        std::string stdout_tail;
        std::string stderr_tail;
    };

    using LogCallback = std::function<void(Stage, std::string)>;
    using ProgressCallback = std::function<bool(Stage, float, std::string)>;
    using NamedProgressCallback = std::function<bool(std::string_view, Stage, float, std::string)>;

    struct ExecutionContext {
        std::stop_token stop_token{};
        std::chrono::milliseconds timeout{0};
        std::chrono::milliseconds stop_grace{250};
        Stage stage = Stage::Validate;
        LogCallback on_log;
        ProgressCallback on_progress;
        NamedProgressCallback on_stage_progress;
    };

    struct SuperResolutionOptions {
        bool enabled = false;
        std::string model_name = "realesr-general-x4v3-dn000";
        int model_scale = 4;
        int final_scale = 1;
    };

    struct MeshExportOptions {
        bool enable_blur_filter = false;
        int fill_black_component_max_pixels = 500;
        bool enhance_before_super_resolution = true;
        SuperResolutionOptions super_resolution{false, "realesr-general-x4v3-dn000", 4, 2};
        double match_3d_threshold = 0.0004;
    };

    struct ScanOptions {
        std::optional<std::size_t> sample_limit = 200;
        bool enable_blur_filter = false;
        bool enable_mesh_overlay = false;
        bool enable_denoise = false;
        bool enable_enhancement = true;
        SuperResolutionOptions super_resolution{true, "realesr-general-x4v3-dn000", 4, 2};
    };

    struct RegisteredOptions {
        std::string matcher = "exhaustive";
        std::string registration_strategy = "colmap";
        bool use_mesh_triangulation = true;
        bool enable_blur_filter = false;
        bool enable_scan_blur_filter = false;
        bool enable_mesh_overlay = false;
        bool enable_registration_triplets = false;
        bool register_twice = false;
        // Frozen face_mode only reached commented-out texture-mapping calls.
        // Preserve the accepted request explicitly without changing artifacts.
        bool face_mode = false;
        bool enable_incremental_mask = false;
        std::optional<std::filesystem::path> foreground_model;
        double match_3d_threshold = 0.0004;
        std::optional<std::size_t> sample_limit = 120;
    };

    struct ScanInput {
        ScanOptions options;
    };

    struct RegisteredInput {
        std::optional<std::filesystem::path> incremental_images;
        std::optional<std::filesystem::path> incremental_video;
        std::optional<int> video_frame_count;
        RegisteredOptions options;
    };

    struct MeshExportInput {
        MeshExportOptions options;
    };

    using InputRequest = std::variant<ScanInput, RegisteredInput, MeshExportInput>;

    struct PreprocessRequest {
        std::filesystem::path scanner_bin;
        std::filesystem::path workspace_root;
        InputRequest input;
        bool retain_failed_stages = false;
        bool write_diagnostic_json = true;
    };

    struct PreprocessResult {
        std::filesystem::path run_root;
        std::filesystem::path dataset_root;
        std::filesystem::path debug_report;
        std::size_t image_count = 0;
        InputMode input_mode = InputMode::Scan;
        std::optional<std::filesystem::path> mesh_export_root;
    };

} // namespace lfs::preprocess
