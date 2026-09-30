/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "preprocessing/providers.hpp"

namespace lfs::preprocess {

    // Dedicated terminal mode for the legacy SAVE_MESH_POINTS3D paths. It uses
    // every available scan frame and never requires or imports incremental RGB.
    std::expected<PreprocessResult, Error> preprocess_mesh_export(
        const PreprocessRequest&, IColmapProvider*, ISuperResolutionProvider*,
        const ExecutionContext& = {});

} // namespace lfs::preprocess
