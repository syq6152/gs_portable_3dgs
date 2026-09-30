/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "preprocessing/types.hpp"
#include <expected>
#include <optional>
#include <span>
#include <vector>

namespace lfs::preprocess {
    struct BlurFilterOptions {
        bool enabled = true;
        int window_size = 10;
        int sensitivity = 50;
    };
    struct BlurScoreSelection {
        std::vector<bool> selected;
        double global_range = 0;
        double threshold_percent = 12.5;
        std::size_t effective_window = 0;
        bool skipped = false;
    };
    // Frozen sharp-frames 0.3.1 CLI selector: neighbor mean relative to the global
    // range, not a percentile filter. Scores must already be in filename order.
    std::expected<BlurScoreSelection, Error> select_blur_outliers(
        std::span<const double> scores, const BlurFilterOptions& = {});
    // Codec grayscale, INTER_AREA half-size, CV_64F Laplacian population variance.
    std::expected<double, Error> blur_sharpness_score(const std::filesystem::path&);
    struct BlurImageRecord {
        std::filesystem::path source;
        std::string name;
        std::optional<std::filesystem::path> pose;
        std::optional<double> sharpness_score;
        bool selected = true;
    };
    struct BlurFilterResult {
        std::vector<BlurImageRecord> records;
        std::vector<std::string> selected_names;
        std::vector<std::string> removed_names;
        double global_range = 0;
        double threshold_percent = 12.5;
        std::size_t effective_window = 0;
        bool skipped = false;
        bool kept_all_fallback = false;
    };
    // Read-only: never writes/deletes images or poses. Apply selection only to
    // owned staging copies. Pose paths are associated by image stem; absence is
    // permitted. Unscorable images are omitted, unless nothing is selected, when
    // the legacy wrapper keeps everything (kept_all_fallback).
    std::expected<BlurFilterResult, Error> select_blur_images(
        const std::filesystem::path& image_directory,
        const BlurFilterOptions& = {}, const ExecutionContext& = {},
        const std::optional<std::filesystem::path>& pose_directory = {});
} // namespace lfs::preprocess
