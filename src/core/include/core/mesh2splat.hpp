/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <string>

#include <glm/glm.hpp>

namespace lfs::core {

    struct Mesh2SplatOptions {
        static constexpr int kMinResolution = 16;

        int resolution_target = 1024;
        float sigma = 0.65f;
        glm::vec3 light_dir{0.0f, 0.0f, 1.0f};
        float light_intensity = 0.7f;
        float ambient = 0.4f;
        int sh_degree = 0;
        // Target raster density ratio in (0,1]; applied by scaling conversion resolution.
        float sampling_rate = 1.0f;
        // Optional target upper bound for generated Gaussians; 0 disables dynamic resolution probing.
        int target_max_gaussians = 0;
    };

    using Mesh2SplatProgressCallback = std::function<bool(float progress, const std::string& stage)>;

} // namespace lfs::core
