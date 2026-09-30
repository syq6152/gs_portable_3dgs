/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/mesh_geometry.hpp"

namespace lfs::preprocess {
    // Writes diagnostic copies only, into the caller's fresh owned stage.
    // scan_/inc_ prefixes and green point blending match the frozen debug outputs.
    std::expected<std::size_t, Error> write_mesh_projection_overlays(
        const CpuMesh&, const ColmapModel&, const std::filesystem::path& images,
        const std::filesystem::path& output, const std::string& prefix,
        const ExecutionContext& = {}, bool write_diagnostic_json = true);
} // namespace lfs::preprocess
