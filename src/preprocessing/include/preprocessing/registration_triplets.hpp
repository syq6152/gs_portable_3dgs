/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/types.hpp"
#include <cstddef>
#include <expected>

namespace lfs::preprocess {
    struct RegistrationTripletSources {
        std::filesystem::path aligned_sparse;
        std::filesystem::path undistorted_sparse;
        std::filesystem::path incremental_images;
        std::filesystem::path incremental_masks;
        std::filesystem::path scan_images;
        std::filesystem::path scan_masks;
    };

    struct RegistrationTripletResult {
        std::size_t images = 0;
        std::size_t masks = 0;
        std::size_t sparse_files = 0;
    };

    // Copies the frozen legacy registration diagnostic snapshot, without decoding
    // images or rewriting COLMAP coordinates/IDs. Sparse TXT prefers aligned data
    // per file; scan PNGs override incremental PNGs with the same destination name.
    // Missing sources are optional. Output must be a fresh, empty owned stage.
    std::expected<RegistrationTripletResult, Error> write_registration_triplets(
        const RegistrationTripletSources&, const std::filesystem::path& output,
        const ExecutionContext& = {}, bool write_diagnostic_json = true);
} // namespace lfs::preprocess
