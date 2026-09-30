/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/types.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs::preprocess {

    struct ProcessRequest {
        std::filesystem::path executable;
        std::vector<std::string> arguments;
        std::filesystem::path working_directory;
        std::vector<std::pair<std::string, std::string>> environment;
    };

    struct ProcessResult {
        int exit_code = -1;
        bool timed_out = false;
        bool cancelled = false;
        std::string stdout_tail;
        std::string stderr_tail;
    };

    class ProcessRunner {
    public:
        [[nodiscard]] std::expected<ProcessResult, Error> run(
            const ProcessRequest& request,
            const ExecutionContext& context = {}) const;
    };

} // namespace lfs::preprocess
