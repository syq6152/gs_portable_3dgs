/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/types.hpp"
#include <array>
#include <expected>
#include <map>
#include <span>
#include <vector>

namespace lfs::preprocess {
    struct CpuImage {
        int width = 0, height = 0, channels = 3;
        std::vector<uint8_t> pixels; // RGB or grayscale, top-left origin
        bool operator==(const CpuImage&) const = default;
    };
    std::expected<CpuImage, Error> read_image(const std::filesystem::path&, bool apply_exif = true);
    std::expected<void, Error> copy_image(const std::filesystem::path& source, const std::filesystem::path& destination,
                                          const ExecutionContext& = {});
    std::expected<void, Error> write_png(const std::filesystem::path&, const CpuImage&);
    std::expected<std::array<int, 2>, Error> image_dimensions(const std::filesystem::path&, bool apply_exif = false);
    std::array<int, 2> resized_dimensions(int width, int height, int longest_side);
    std::expected<CpuImage, Error> resize_lanczos(const CpuImage&, int width, int height);
    CpuImage rotate_ccw(const CpuImage&);
    std::vector<std::size_t> uniform_sample(std::size_t total, std::size_t requested);
    std::string incremental_image_name(std::size_t index);
    struct ImageImportOptions {
        int longest_side = 2048;
        bool largest_size_group_only = false;
        bool normalize_swapped_orientation = true;
    };
    struct ImageImportRecord {
        std::filesystem::path source;
        std::string name;
        int width = 0, height = 0;
        bool rotated = false;
    };
    // Fresh stage directory only; partial files remain owned by the caller's staging workspace.
    std::expected<std::vector<ImageImportRecord>, Error> import_images(
        const std::filesystem::path& source, const std::filesystem::path& destination,
        const ImageImportOptions& = {}, const ExecutionContext& = {});
    struct EnhancementOptions {
        bool white_balance = false, luma_tone = false, contrast = true, saturation = true, sharpen = true;
        float wb_min = 0.8F, wb_max = 1.2F, luma_target = 140.F, relative_luma_keep = 0.65F, tone_blend = 0.55F;
        float contrast_gain = 1.3F, saturation_gain = 1.2F, sharpen_amount = 0.15F;
        float low_percentile = 5.F, high_percentile = 95.F;
        int stats_max_size = 512;
    };
    struct EnhancementStats {
        std::array<float, 3> channel_reference{127.5F, 127.5F, 127.5F}; // BGR, matches frozen Python
        std::array<float, 3> wb_gains{1, 1, 1}, sequence_luma{32, 127.5F, 223};
        std::map<std::string, std::array<float, 3>> image_luma;
    };
    std::expected<EnhancementStats, Error> enhancement_stats(
        const std::filesystem::path&, const std::vector<std::string>&, const EnhancementOptions& = {}, const ExecutionContext& = {});
    std::expected<CpuImage, Error> denoise_image(const CpuImage&);
    std::expected<CpuImage, Error> enhance_image(const CpuImage&, const std::string& name, const EnhancementStats&, const EnhancementOptions& = {});
    // Native denoise -> sequence statistics -> WB -> tone/contrast -> saturation -> sharpen.
    std::expected<void, Error> enhance_images(const std::filesystem::path& input, const std::filesystem::path& output,
                                              const std::vector<std::string>& names, bool denoise, const EnhancementOptions& = {}, const ExecutionContext& = {});
} // namespace lfs::preprocess
