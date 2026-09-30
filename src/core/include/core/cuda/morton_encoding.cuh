/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::core {
    /**
     * @brief Compute Morton codes for 3D positions using lfs::core::Tensor
     *
     * This function encodes 3D positions into Morton codes (Z-order curve) for
     * spatial sorting. This improves cache locality during rendering.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32)
     * @return Tensor of shape [N] containing Morton codes as Int64
     */
    Tensor morton_encode_new(const Tensor& positions);

    /**
     * @brief Sort indices by Morton codes using lfs::core::Tensor
     *
     * @param morton_codes Tensor of Morton codes (Int64)
     * @return Tensor of indices that would sort the Morton codes (Int64)
     */
    Tensor morton_sort_indices_new(const Tensor& morton_codes);
} // namespace lfs::core
