/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <preprocessing/types.hpp>

#include <array>
#include <cstddef>
#include <string_view>

namespace lfs::preprocess {

    // Frozen Swaptexture exit codes from commit
    // 9180f87aabde0bdf8d0957006e53bdb42528c52d:ErrorCode.md. Keep the
    // individual legacy meaning even when several codes share one native
    // ErrorCode; operation context selects the legacy code, not the reverse.
    enum class LegacyErrorCode : int {
        InvalidInput = 10001,
        ScanInputReadFailed = 10002,
        IncrementalInputReadFailed = 10003,
        CameraPoseReadFailed = 10004,
        CameraIntrinsicsReadFailed = 10005,
        ResultWriteFailed = 10006,
        AlgorithmFailed = 10007,
        OutOfMemory = 10008,
        Unauthorized = 10009,
        ScanFeatureExtractionFailed = 10010,
        ScanFeatureMatchingFailed = 10011,
        ScanTriangulationFailed = 10012,
        IncrementalFeatureExtractionFailed = 10013,
        IncrementalFeatureMatchingFailed = 10014,
        IncrementalRegistrationFailed = 10015,
        IncrementalReconstructionFailed = 10016,
        IncrementalUndistortionFailed = 10017,
        IncrementalAlignmentFailed = 10018,
        TrainingFailed = 10019,
        ScanModelWriteFailed = 10020,
        ScanIntrinsicsDatabaseWriteFailed = 10021,
    };

    struct LegacyErrorMapping {
        LegacyErrorCode legacy_code;
        std::string_view symbol;
        std::string_view meaning;
        ErrorCode typed_code;
        int native_cli_exit_code;
    };

    inline constexpr std::array<LegacyErrorMapping, 21> LEGACY_ERROR_MAPPINGS{{
        {LegacyErrorCode::InvalidInput, "invalid_input", "Input parameters are invalid", ErrorCode::InvalidRequest, 2},
        {LegacyErrorCode::ScanInputReadFailed, "scan_input_read_failed", "Scanner input images cannot be read", ErrorCode::InvalidDataset, 2},
        {LegacyErrorCode::IncrementalInputReadFailed, "incremental_input_read_failed", "Incremental input images cannot be read", ErrorCode::InvalidDataset, 2},
        {LegacyErrorCode::CameraPoseReadFailed, "camera_pose_read_failed", "Camera poses cannot be read", ErrorCode::InvalidDataset, 2},
        {LegacyErrorCode::CameraIntrinsicsReadFailed, "camera_intrinsics_read_failed", "Camera intrinsics cannot be read", ErrorCode::InvalidDataset, 2},
        {LegacyErrorCode::ResultWriteFailed, "result_write_failed", "Result publication failed", ErrorCode::IoFailure, 1},
        {LegacyErrorCode::AlgorithmFailed, "algorithm_failed", "Algorithm processing failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::OutOfMemory, "out_of_memory", "Processing ran out of memory", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::Unauthorized, "unauthorized", "Legacy encrypted input authorization failed", ErrorCode::InvalidRequest, 2},
        {LegacyErrorCode::ScanFeatureExtractionFailed, "scan_feature_extraction_failed", "Scanner feature extraction failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::ScanFeatureMatchingFailed, "scan_feature_matching_failed", "Scanner feature matching failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::ScanTriangulationFailed, "scan_triangulation_failed", "Scanner triangulation failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalFeatureExtractionFailed, "incremental_feature_extraction_failed", "Incremental feature extraction failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalFeatureMatchingFailed, "incremental_feature_matching_failed", "Incremental feature matching failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalRegistrationFailed, "incremental_registration_failed", "Incremental image registration failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalReconstructionFailed, "incremental_reconstruction_failed", "Incremental sparse reconstruction failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalUndistortionFailed, "incremental_undistortion_failed", "Incremental image undistortion failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::IncrementalAlignmentFailed, "incremental_alignment_failed", "Incremental model alignment failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::TrainingFailed, "training_failed", "Gaussian-splatting training failed", ErrorCode::ProcessFailed, 1},
        {LegacyErrorCode::ScanModelWriteFailed, "scan_model_write_failed", "Scanner COLMAP model publication failed", ErrorCode::IoFailure, 1},
        {LegacyErrorCode::ScanIntrinsicsDatabaseWriteFailed, "scan_intrinsics_database_write_failed", "Scanner intrinsics database write failed", ErrorCode::IoFailure, 1},
    }};

    [[nodiscard]] constexpr const LegacyErrorMapping* legacy_error_mapping(int legacy_code) noexcept {
        constexpr int first = static_cast<int>(LegacyErrorCode::InvalidInput);
        constexpr int last = static_cast<int>(LegacyErrorCode::ScanIntrinsicsDatabaseWriteFailed);
        if (legacy_code < first || legacy_code > last)
            return nullptr;
        const auto& mapping = LEGACY_ERROR_MAPPINGS[static_cast<std::size_t>(legacy_code - first)];
        return static_cast<int>(mapping.legacy_code) == legacy_code ? &mapping : nullptr;
    }

    [[nodiscard]] constexpr const LegacyErrorMapping& legacy_error_mapping(LegacyErrorCode legacy_code) noexcept {
        return *legacy_error_mapping(static_cast<int>(legacy_code));
    }

    // Native CLI exit statuses are intentionally small and portable. The
    // original 10001-10021 value remains available in LegacyErrorMapping for
    // compatibility reports and callers that still expose the old protocol.
    [[nodiscard]] constexpr int native_cli_exit_code(ErrorCode code) noexcept {
        switch (code) {
        case ErrorCode::InvalidRequest:
        case ErrorCode::InvalidDataset:
        case ErrorCode::UnsupportedFeature:
            return 2;
        case ErrorCode::ProcessCancelled:
            return 130;
        case ErrorCode::RuntimeMissing:
        case ErrorCode::RuntimeManifestMismatch:
        case ErrorCode::ProcessLaunchFailed:
        case ErrorCode::ProcessFailed:
        case ErrorCode::ProcessTimeout:
        case ErrorCode::CallbackFailure:
        case ErrorCode::PlatformUnsupported:
        case ErrorCode::IncompatibleRuntime:
        case ErrorCode::IoFailure:
            return 1;
        }
        return 1;
    }

    consteval bool valid_legacy_error_mappings() {
        for (std::size_t index = 0; index < LEGACY_ERROR_MAPPINGS.size(); ++index) {
            const auto& mapping = LEGACY_ERROR_MAPPINGS[index];
            if (static_cast<int>(mapping.legacy_code) != 10001 + static_cast<int>(index) ||
                mapping.symbol.empty() || mapping.meaning.empty() ||
                mapping.native_cli_exit_code != native_cli_exit_code(mapping.typed_code))
                return false;
            for (std::size_t other = 0; other < index; ++other)
                if (LEGACY_ERROR_MAPPINGS[other].symbol == mapping.symbol)
                    return false;
        }
        return true;
    }

    static_assert(valid_legacy_error_mappings());

} // namespace lfs::preprocess
