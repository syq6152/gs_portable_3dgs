/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "preprocessing/image_pipeline.hpp"
#include <memory>
#include <string_view>

namespace lfs::preprocess {
    inline constexpr std::string_view u2netp_model_sha256 =
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8";
    struct ForegroundMaskOptions {
        int erosion_pixels = 3;
    };
    // Frozen rembg 2.0.67/u2netp single-pass input. Uses RGB (or replicated gray),
    // Pillow-compatible LANCZOS, image-maximum normalization, then ImageNet
    // mean/std in double precision before conversion to float32 NCHW.
    std::expected<std::vector<float>, Error> u2netp_input_tensor(const CpuImage&);
    // A 320x320 first-channel output plane. Min/max normalization and uint8
    // truncation precede LANCZOS back to source size, alpha>0, and edge erosion.
    // A constant prediction becomes an empty mask (legacy NaN-to-uint8 result).
    std::expected<CpuImage, Error> u2netp_output_mask(
        std::span<const float> prediction, int width, int height, const ForegroundMaskOptions& = {});

    class IForegroundMaskProvider {
    public:
        virtual ~IForegroundMaskProvider() = default;
        virtual std::expected<CpuImage, Error> mask(
            const CpuImage&, const ForegroundMaskOptions& = {}, const ExecutionContext& = {}) = 0;
    };
    // CPU-only local provider. No download, Python, CUDA, directory search or
    // output writes. Loads the verified bytes into ORT to avoid a hash/load race.
    // Model redistribution and CUDA equivalence are separate acceptance items.
    std::expected<std::unique_ptr<IForegroundMaskProvider>, Error> create_u2netp_cpu_provider(
        const std::filesystem::path& model, const ExecutionContext& = {});
} // namespace lfs::preprocess
