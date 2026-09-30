/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/error.hpp"
#include "io/loader_interface.hpp"
#include "io/loader_registry.hpp"
#include <memory>
#include <vector>

namespace lfs::io {

    /**
     * @brief Simple service for loading data files
     *
     * Provides a clean interface for loading any supported format.
     */
    class LoaderService {
    public:
        LoaderService();
        ~LoaderService() = default;

        // Delete copy operations
        LoaderService(const LoaderService&) = delete;
        LoaderService& operator=(const LoaderService&) = delete;

        /**
         * @brief Load data from any supported format
         * @param path File or directory to load
         * @param options Loading options
         * @return LoadResult on success, Error on failure (path not found, invalid format, etc.)
         */
        [[nodiscard]] Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {});

        /**
         * @brief Get information about available loaders
         */
        std::vector<std::string> getAvailableLoaders() const;

        /**
         * @brief Get supported extensions
         */
        std::vector<std::string> getSupportedExtensions() const;

    private:
        std::unique_ptr<DataLoaderRegistry> registry_;
    };

} // namespace lfs::io