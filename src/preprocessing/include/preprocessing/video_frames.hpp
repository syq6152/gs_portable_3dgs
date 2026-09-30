/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "preprocessing/types.hpp"
#include <cstdint>
#include <expected>
#include <vector>

namespace lfs::preprocess {
    struct VideoExtractionOptions {
        int target_frame_count = 80;
        int longest_side = 2048;
    };
    struct VideoFrameRecord {
        std::string name;
        std::int64_t source_index = 0;
        int width = 0;
        int height = 0;
    };
    struct VideoExtractionResult {
        std::int64_t total_frames = 0;
        bool counted_frames = false;
        std::vector<std::int64_t> sample_indices;
        std::vector<VideoFrameRecord> frames;
        bool incomplete = false;
        std::string backend = "FFMPEG";
    };

    // Frozen NumPy linspace(..., dtype=int64), capped by total, including both ends.
    std::expected<std::vector<std::int64_t>, Error> video_sample_indices(std::int64_t total_frames, int requested);

    // Read-only local input, native FFmpeg backend only. Destination must not exist.
    // Partial output on failure remains owned by the caller's staging workspace.
    std::expected<VideoExtractionResult, Error> extract_video_frames(
        const std::filesystem::path& video, const std::filesystem::path& destination,
        const VideoExtractionOptions& = {}, const ExecutionContext& = {});
} // namespace lfs::preprocess
