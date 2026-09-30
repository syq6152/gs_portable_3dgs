/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/reconstruction_command.hpp"
#include "app/converter.hpp"
#include "app/scan_pose_filter.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "git_version.h"
#include "package_version.h"
#include "python/plugin_runner.hpp"
#include "python/runner.hpp"

#include <print>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shellapi.h>
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
    auto result = [&]() {
#ifdef _WIN32
        int wide_argc = 0;
        LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
        if (wide_argv != nullptr) {
            std::vector<std::string> utf8_args = lfs::core::wide_argv_to_utf8(wide_argc, wide_argv);
            LocalFree(wide_argv);

            std::vector<const char*> utf8_argv;
            utf8_argv.reserve(utf8_args.size());
            for (const auto& arg : utf8_args) {
                utf8_argv.push_back(arg.c_str());
            }

            return lfs::core::args::parse_args(static_cast<int>(utf8_argv.size()), utf8_argv.data());
        }
#endif
        return lfs::core::args::parse_args(argc, argv);
    }();
    if (!result) {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    return std::visit([](auto&& mode) -> int {
        using T = std::decay_t<decltype(mode)>;

        if constexpr (std::is_same_v<T, lfs::core::args::HelpMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::VersionMode>) {
            std::println("Swaptexture v{}", LFS_SWAPTEXTURE_PACKAGE_VERSION);
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::WarmupMode>) {
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::ConvertMode>) {
            return lfs::app::run_converter(mode.params);
        } else if constexpr (std::is_same_v<T, lfs::core::args::ScanPoseFilterMode>) {
            return lfs::app::run_scan_pose_filter(mode.params);
        } else if constexpr (std::is_same_v<T, lfs::core::args::PluginMode>) {
            // return lfs::python::run_plugin_command(mode);
            std::println(stderr, "PluginMode is disabled.");
            return 0;
        } else if constexpr (std::is_same_v<T, lfs::core::args::TrainingMode>) {
            LOG_INFO("LichtFeld Studio");
            LOG_INFO("version {} | tag {}", GIT_TAGGED_VERSION, GIT_COMMIT_HASH_SHORT);

            // if (mode.params->optimization.debug_python) {
            //     lfs::python::start_debugpy(mode.params->optimization.debug_python_port);
            // }

            lfs::app::Application app;
            return app.run(std::move(mode.params));
        } else if constexpr (std::is_same_v<T, lfs::core::args::ReconstructionMode>) {
            return lfs::app::run_reconstruction(mode.arguments, mode.swaptexture_cli);
        }
    },
                      std::move(*result));
}
