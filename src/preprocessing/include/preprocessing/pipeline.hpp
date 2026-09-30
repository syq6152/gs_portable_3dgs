/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/types.hpp"
#include <expected>
namespace lfs::preprocess {
    std::expected<PreprocessResult, Error> preprocess(const PreprocessRequest&, const ExecutionContext& = {});
    std::expected<void, Error> validate_dataset_skeleton(const std::filesystem::path&);
    std::expected<std::filesystem::path, Error> create_workspace(const std::filesystem::path&);
    std::expected<void, Error> publish_ply(const std::filesystem::path& source, const std::filesystem::path& destination);
} // namespace lfs::preprocess
