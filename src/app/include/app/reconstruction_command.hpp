/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/parameters.hpp"
#include "preprocessing/types.hpp"
#include <vector>
namespace lfs::app {
    enum class TrainingQuality { Fast,
                                 Medium,
                                 Quality };
    struct ReconstructionRequest {
        preprocess::PreprocessRequest preprocessing;
        core::param::TrainingParameters training;
        TrainingQuality quality = TrainingQuality::Fast;
        std::filesystem::path mesh_init_scene;
        std::optional<std::filesystem::path> final_ply;
        std::optional<std::filesystem::path> prepared_fixture;
    };
    int run_reconstruction(const std::vector<std::string>& arguments, bool swaptexture_cli);
} // namespace lfs::app
