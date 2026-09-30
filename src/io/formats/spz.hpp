/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "io/error.hpp"
#include <expected>
#include <filesystem>

namespace lfs::io {

    using lfs::core::SplatData;

    // Load SPZ (Niantic compressed gaussian splat format)
    std::expected<SplatData, std::string> load_spz(const std::filesystem::path& filepath);

    // Save SPZ format
    struct SpzSaveOptions {
        std::filesystem::path output_path;
    };

    [[nodiscard]] Result<void> save_spz(const SplatData& splat_data, const SpzSaveOptions& options);

} // namespace lfs::io
