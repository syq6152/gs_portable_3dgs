/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis::tools {

    // How new selection combines with existing
    enum class SelectionOp {
        Replace,    // Clear active group, set new selection
        Add,        // Add to active group
        Remove,     // Remove from active group
        Toggle,     // Toggle within active group
        ClearGroup, // Clear active group
        ClearAll,   // Clear all groups
    };

} // namespace lfs::vis::tools
