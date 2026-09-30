/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace lfs::training::kernels {

    // Produces a [count] Bool-compatible mask. A row is true only when the
    // squared norm of its raw quaternion is strictly less than 1e-8.
    void launch_near_zero_quaternion_mask(
        const float* rotations,
        uint8_t* mask,
        size_t count,
        cudaStream_t stream);

} // namespace lfs::training::kernels
