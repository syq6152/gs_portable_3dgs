/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/types.hpp"

#include <expected>
#include <filesystem>

namespace lfs::preprocess {

    struct RuntimePaths {
        std::filesystem::path root;
        std::filesystem::path colmap_root;
        std::filesystem::path colmap_executable;
        std::filesystem::path super_resolution_root;
        std::filesystem::path super_resolution_executable;
        std::filesystem::path super_resolution_models;
        std::filesystem::path manifest;
    };

    [[nodiscard]] std::expected<RuntimePaths, Error> resolve_runtime_paths(
        const std::filesystem::path& executable_dir,
        std::optional<std::filesystem::path> colmap_override = {},
        std::optional<std::filesystem::path> super_resolution_override = {},
        std::optional<std::filesystem::path> manifest_override = {});

    [[nodiscard]] std::expected<void, Error> validate_runtime_paths(const RuntimePaths& paths);
    [[nodiscard]] std::expected<void, Error> verify_runtime_manifest(const RuntimePaths& paths);
    [[nodiscard]] std::expected<void, Error> check_runtime_compatibility(bool require_cuda);
    struct RuntimeHostInfo {
        bool x64 = false;
        unsigned windows_major = 0;
        unsigned windows_build = 0;
        unsigned vc_major = 0;
        unsigned vc_minor = 0;
        int cuda_driver = 0;
        int gpu_sm = 0;
    };
    [[nodiscard]] std::expected<void, Error> validate_runtime_host(const RuntimeHostInfo&, bool require_cuda);

    inline std::string path_utf8(const std::filesystem::path& path) {
        const auto text = path.u8string();
        return {reinterpret_cast<const char*>(text.data()), text.size()};
    }

} // namespace lfs::preprocess
