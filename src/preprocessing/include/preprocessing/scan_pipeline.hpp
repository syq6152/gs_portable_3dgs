/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/colmap_model.hpp"
#include "preprocessing/providers.hpp"
#include "preprocessing/types.hpp"
#include <expected>

namespace lfs::preprocess {
    // Shared frozen scanner import: caller supplies an owned fresh stage directory.
    std::expected<ColmapModel, Error> prepare_scan_source(
        const PreprocessRequest&, const std::filesystem::path&, std::optional<size_t>, const ExecutionContext& = {});
    // Dependency injection is for embedders/tests. Production preprocess() resolves the
    // manifest-checked bundled provider; scan never invokes COLMAP or another trainer.
    std::expected<PreprocessResult, Error> preprocess_scan(
        const PreprocessRequest&, ISuperResolutionProvider*, const ExecutionContext& = {});
} // namespace lfs::preprocess
