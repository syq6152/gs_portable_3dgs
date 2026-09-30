/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/process_runner.hpp"
#include "preprocessing/runtime.hpp"

#include <expected>
#include <string>
#include <vector>

namespace lfs::preprocess {

    enum class ColmapOperation { ExtractFeatures,
                                 MatchExhaustive,
                                 MatchSequential,
                                 Map,
                                 RegisterImages,
                                 BundleAdjust,
                                 AlignModel,
                                 TriangulatePoints,
                                 ConvertModel,
                                 UndistortImages };

    // Domain request: executable discovery, argv construction and cwd belong to the provider.
    struct ColmapRequest {
        ColmapOperation operation = ColmapOperation::ExtractFeatures;
        std::filesystem::path database;
        std::filesystem::path images;
        std::filesystem::path input_model;
        std::filesystem::path output_model;
        std::filesystem::path reference_images;
        std::filesystem::path image_list;
        std::filesystem::path image_masks;
        std::optional<int> extraction_max_image_size;
        std::optional<int> extraction_max_features;
        std::optional<int> num_threads;
        std::optional<bool> use_gpu;
        std::optional<bool> guided_matching;
        std::optional<bool> fix_existing_frames;
        std::optional<std::string> camera_model;
        std::optional<std::string> camera_parameters;
        std::optional<bool> single_camera;
        std::optional<int> minimum_matches;
        std::optional<int> initial_minimum_inliers;
        std::optional<int> absolute_pose_minimum_inliers;
        std::optional<int> minimum_model_size;
        std::optional<std::string> output_format;
        std::optional<bool> references_are_gps;
        std::optional<std::string> alignment_type;
        std::optional<double> alignment_maximum_error;
        bool probe_help = false;
    };

    struct ToolDiagnostics {
        int exit_code = 0;
        std::string stdout_tail;
        std::string stderr_tail;
    };
    // Pure path planning for the frozen Windows tool's narrow image I/O.
    // Pipeline-owned ASCII leaves stay relative to a Unicode-safe Win32 cwd.
    [[nodiscard]] std::filesystem::path colmap_working_directory(
        const ColmapRequest&, const std::filesystem::path& fallback);
    struct ColmapResult {
        ColmapOperation operation;
        std::optional<ToolDiagnostics> diagnostics;
    };

    class IColmapProvider {
    public:
        virtual ~IColmapProvider() = default;
        [[nodiscard]] virtual std::expected<ColmapResult, Error> run(
            const ColmapRequest&, const ExecutionContext&) = 0;
    };

    class ColmapExeProvider final : public IColmapProvider {
    public:
        explicit ColmapExeProvider(RuntimePaths paths, ProcessRunner runner = {});
        [[nodiscard]] std::expected<ColmapResult, Error> run(
            const ColmapRequest&, const ExecutionContext&) override;

    private:
        RuntimePaths paths_;
        ProcessRunner runner_;
    };

    struct SuperResolutionRequest {
        std::filesystem::path input_directory;
        std::filesystem::path output_directory;
        std::string model_name = "realesr-general-x4v3-dn000";
        int model_scale = 4;
        int final_scale = 1;
    };

    struct SuperResolutionResult {
        std::filesystem::path output_directory;
        std::size_t image_count = 0;
        std::optional<ToolDiagnostics> diagnostics;
    };

    class ISuperResolutionProvider {
    public:
        virtual ~ISuperResolutionProvider() = default;
        [[nodiscard]] virtual std::expected<SuperResolutionResult, Error> run(
            const SuperResolutionRequest&, const ExecutionContext&) = 0;
    };

    class SuperResolutionExeProvider final : public ISuperResolutionProvider {
    public:
        explicit SuperResolutionExeProvider(RuntimePaths paths, ProcessRunner runner = {});
        [[nodiscard]] std::expected<SuperResolutionResult, Error> run(
            const SuperResolutionRequest&, const ExecutionContext&) override;

    private:
        RuntimePaths paths_;
        ProcessRunner runner_;
    };

} // namespace lfs::preprocess
