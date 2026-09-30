/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::io {

    using lfs::core::Tensor;

    /**
     * @brief Compute Morton codes for 3D positions
     *
     * Encodes 3D positions into Morton codes (Z-order curve) for spatial sorting.
     * Improves cache locality during rendering and compression.
     *
     * @param positions Tensor of shape [N, 3] containing 3D positions (Float32, CUDA)
     * @return Tensor of shape [N] containing Morton codes as Int64
     */
    Tensor morton_encode(const Tensor& positions);

    /**
     * @brief Sort indices by Morton codes
     *
     * @param morton_codes Tensor of Morton codes (Int64, CUDA)
     * @return Tensor of indices that would sort the Morton codes (Int64)
     */
    Tensor morton_sort_indices(const Tensor& morton_codes);

} // namespace lfs::io
