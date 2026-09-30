/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/colmap_model.hpp"
#include "preprocessing/mesh_geometry.hpp"
#include "preprocessing/providers.hpp"

namespace lfs::preprocess {

    struct MeshExportResult {
        std::filesystem::path root;
        std::size_t images = 0;
        std::size_t masks = 0;
        std::size_t sparse_files = 0;
    };

    // Writes the frozen mesh-triangulation artifact (images/mask/sparse) into an
    // existing empty directory owned by a Workspace stage. Source images are
    // copied byte-for-byte for every prepared scan frame, while the sparse
    // model may contain only the frames retained by mesh triangulation.
    std::expected<MeshExportResult, Error> write_mesh_triangulation_export(
        const std::filesystem::path& images, const ColmapModel& scan_model,
        const ColmapModel& triangulated, const CpuMesh& mesh, const std::filesystem::path& output,
        int fill_black_component_max_pixels = 500, const ExecutionContext& = {});

    struct MeshSuperResolutionResult {
        std::filesystem::path root;
        std::filesystem::path images;
        std::size_t image_count = 0;
        bool enhanced = false;
        std::optional<ToolDiagnostics> diagnostics;
    };

    // Reproduces prepare_mesh_points3d_superresolution_scan_images inside an
    // owned empty stage directory. The output is root/images and has the exact
    // same PNG filename set as the input. Partial files are intentionally left
    // to Workspace rollback when the provider or validation fails.
    std::expected<MeshSuperResolutionResult, Error> prepare_mesh_superresolution_images(
        const std::filesystem::path& input, const std::filesystem::path& output,
        const SuperResolutionOptions&, ISuperResolutionProvider*,
        bool enhance_before_super_resolution = true, const ExecutionContext& = {});

} // namespace lfs::preprocess
