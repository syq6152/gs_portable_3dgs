/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/colmap_model.hpp"

#include <expected>
#include <optional>
#include <vector>

namespace lfs::preprocess {

    struct RegisterTwiceModelMerge {
        std::size_t common_image_count = 0;
        std::size_t added_image_count = 0;
        std::optional<double> scale;
        std::optional<uint32_t> primary_first_image_id;
        std::optional<uint32_t> primary_last_image_id;
        std::optional<uint32_t> secondary_first_image_id;
        std::optional<uint32_t> secondary_last_image_id;
    };

    struct RegisterTwiceMergeResult {
        ColmapModel model;
        std::vector<RegisterTwiceModelMerge> models;
        std::size_t added_image_count = 0;
    };

    // Reproduces the frozen Swaptexture merge_sfm_and_ir pose merge. Each
    // secondary model uses its first and last name-shared images to recover a
    // scale and rigid c2w transform. Only images absent from the primary model
    // are appended; primary cameras, images and points remain authoritative.
    [[nodiscard]] std::expected<RegisterTwiceMergeResult, Error> merge_register_twice_models(
        const ColmapModel& primary,
        const std::vector<ColmapModel>& secondary_models);

} // namespace lfs::preprocess
