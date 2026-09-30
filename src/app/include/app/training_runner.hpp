/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/parameters.hpp"
#include <expected>
#include <functional>
#include <stop_token>
namespace lfs::app {
    struct TrainingRunResult {
        std::filesystem::path gaussian_ply;
        int final_iteration = 0;
        bool cancelled = false;
    };
    struct TrainingError {
        std::string message;
        int exit_code = 1;
    };
    struct TrainingCallbacks {
        std::function<void(const core::param::TrainingParameters&)> on_parameters;
        // Normalized [0, 1] progress of the in-process GS training phase.
        std::function<void(float)> on_progress;
        std::function<void(int iteration, float loss, int gaussians)> on_iteration;
    };
    std::expected<TrainingRunResult, TrainingError> run_training(
        core::param::TrainingParameters params, std::stop_token stop = {}, TrainingCallbacks callbacks = {});
} // namespace lfs::app
