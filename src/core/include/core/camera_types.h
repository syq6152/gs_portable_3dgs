/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::core {

    enum class CameraModelType {
        PINHOLE = 0,
        ORTHO = 1,
        FISHEYE = 2,
        EQUIRECTANGULAR = 3,
        THIN_PRISM_FISHEYE = 4
    };

} // namespace lfs::core
