/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/application.hpp"
#include "app/training_runner.hpp"
#include "core/logger.hpp"

namespace lfs::app {
    int Application::run(std::unique_ptr<core::param::TrainingParameters> params) {
        if (!params)
            return 1;
        const auto result = run_training(std::move(*params));
        if (!result) {
            LOG_ERROR("{}", result.error().message);
            return result.error().exit_code;
        }
        return result->cancelled ? 130 : 0;
    }
} // namespace lfs::app
