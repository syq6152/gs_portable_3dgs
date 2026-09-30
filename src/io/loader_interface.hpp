/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"
#include "io/loader.hpp"
#include <string>
#include <vector>

namespace lfs::io {

    /**
     * @brief Base interface for data loaders
     *
     * Each loader implementation handles a specific file format or data source.
     */
    class IDataLoader {
    public:
        virtual ~IDataLoader() = default;

        /**
         * @brief Load data from the specified path
         * @param path File or directory path to load from
         * @param options Loading options
         * @return LoadResult on success, Error on failure
         */
        [[nodiscard]] virtual Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) = 0;

        /**
         * @brief Check if this loader can handle the given path
         * @param path File or directory path to check
         * @return true if this loader can handle the path
         */
        virtual bool canLoad(const std::filesystem::path& path) const = 0;

        /**
         * @brief Get human-readable name of this loader
         * @return Loader name (e.g., "PLY", "COLMAP", "Blender")
         */
        virtual std::string name() const = 0;

        /**
         * @brief Get list of supported file extensions
         * @return Extensions including the dot (e.g., {".ply", ".PLY"})
         */
        virtual std::vector<std::string> supportedExtensions() const = 0;

        /**
         * @brief Get loader priority (higher = preferred when multiple loaders match)
         * @return Priority value, default is 0
         */
        virtual int priority() const { return 0; }
    };
} // namespace lfs::io