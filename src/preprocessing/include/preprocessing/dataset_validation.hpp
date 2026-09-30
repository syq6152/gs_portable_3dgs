/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/colmap_model.hpp"
namespace lfs::preprocess {
    struct DatasetValidation {
        std::size_t image_count = 0;
        std::size_t camera_count = 0;
        bool has_points = false;
    };
    std::expected<DatasetValidation, Error> validate_dataset(const std::filesystem::path&, const ExecutionContext& = {});
    // Mirrors the trainer's file or legacy directory asset-selection rules; does not rescale geometry.
    std::expected<std::filesystem::path, Error> validate_mesh_input(const std::filesystem::path&);
} // namespace lfs::preprocess
