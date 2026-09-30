/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "splat.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <fstream>
#include <numeric>
#include <optional>
#include <vector>

namespace lfs::io {

    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        constexpr size_t SPLAT_RECORD_BYTES = 32;
        constexpr float SH_C0 = 0.28209479177387814f;

        [[nodiscard]] bool is_valid_2d_tensor(const Tensor& tensor, const int64_t rows, const int64_t cols) {
            return tensor.is_valid() && tensor.ndim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols;
        }

        [[nodiscard]] bool is_valid_sh0_tensor(const Tensor& tensor, const int64_t rows) {
            if (!tensor.is_valid() || tensor.size(0) != rows) {
                return false;
            }
            if (tensor.ndim() == 2) {
                return tensor.size(1) >= 3;
            }
            if (tensor.ndim() == 3) {
                return tensor.size(1) >= 1 && tensor.size(2) >= 3;
            }
            return false;
        }

        [[nodiscard]] bool is_valid_opacity_tensor(const Tensor& tensor, const int64_t rows) {
            if (!tensor.is_valid() || tensor.size(0) != rows) {
                return false;
            }
            return (tensor.ndim() == 1) || (tensor.ndim() == 2 && tensor.size(1) >= 1);
        }

        [[nodiscard]] float sigmoid(const float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        [[nodiscard]] uint8_t quantize_byte(const float value) {
            return static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
        }

        [[nodiscard]] float sh0_component(const Tensor& sh0, const float* ptr, const int64_t index, const int channel) {
            if (sh0.ndim() == 3) {
                return ptr[(index * sh0.size(1) * sh0.size(2)) + channel];
            }
            return ptr[index * sh0.size(1) + channel];
        }

        [[nodiscard]] float opacity_component(const Tensor& opacity, const float* ptr, const int64_t index) {
            if (opacity.ndim() == 1) {
                return ptr[index];
            }
            return ptr[index * opacity.size(1)];
        }

        [[nodiscard]] std::optional<Error> validate_splat_data(const SplatData& splat_data,
                                                               const std::filesystem::path& output_path) {
            const auto rows = static_cast<int64_t>(splat_data.size());
            if (rows == 0) {
                return Error{ErrorCode::EMPTY_DATASET, "No splats to write", output_path};
            }
            if (!is_valid_2d_tensor(splat_data.means_raw(), rows, 3)) {
                return Error{ErrorCode::INTERNAL_ERROR, "SplatData means must be [N,3]", output_path};
            }
            if (!is_valid_2d_tensor(splat_data.scaling_raw(), rows, 3)) {
                return Error{ErrorCode::INTERNAL_ERROR, "SplatData scaling must be [N,3]", output_path};
            }
            if (!is_valid_2d_tensor(splat_data.rotation_raw(), rows, 4)) {
                return Error{ErrorCode::INTERNAL_ERROR, "SplatData rotation must be [N,4]", output_path};
            }
            if (!is_valid_sh0_tensor(splat_data.sh0_raw(), rows)) {
                return Error{ErrorCode::INTERNAL_ERROR, "SplatData SH0 must be [N,1,3] or [N,3]", output_path};
            }
            if (!is_valid_opacity_tensor(splat_data.opacity_raw(), rows)) {
                return Error{ErrorCode::INTERNAL_ERROR, "SplatData opacity must be [N] or [N,1]", output_path};
            }

            return std::nullopt;
        }
    } // namespace

    Result<void> save_splat(const SplatData& splat_data, const SplatSaveOptions& options) {
        auto start = std::chrono::high_resolution_clock::now();

        LOG_INFO("SPLAT write: {}", lfs::core::path_to_utf8(options.output_path));

        const auto rows = static_cast<int64_t>(splat_data.size());

        if (auto validation_error = validate_splat_data(splat_data, options.output_path)) {
            return std::unexpected(*validation_error);
        }

        const auto estimated_size = static_cast<std::uintmax_t>(rows) * SPLAT_RECORD_BYTES;
        if (auto space_check = check_disk_space(options.output_path, estimated_size, 1.1f); !space_check) {
            return std::unexpected(space_check.error());
        }

        if (auto writable_check = verify_writable(options.output_path); !writable_check) {
            return std::unexpected(writable_check.error());
        }

        const auto means = splat_data.means_raw().cpu().contiguous();
        const auto scales = splat_data.scaling_raw().cpu().contiguous();
        const auto rotations = splat_data.get_rotation().cpu().contiguous();
        const auto sh0 = splat_data.sh0_raw().cpu().contiguous();
        const auto opacity = splat_data.opacity_raw().cpu().contiguous();

        const auto* means_ptr = means.ptr<float>();
        const auto* scales_ptr = scales.ptr<float>();
        const auto* rotations_ptr = rotations.ptr<float>();
        const auto* sh0_ptr = sh0.ptr<float>();
        const auto* opacity_ptr = opacity.ptr<float>();

        std::vector<int64_t> indices(static_cast<size_t>(rows));
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<double> sort_keys(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) {
            const float raw_opacity = opacity_component(opacity, opacity_ptr, i);
            const double scale_volume = std::exp(
                static_cast<double>(scales_ptr[i * 3 + 0]) +
                static_cast<double>(scales_ptr[i * 3 + 1]) +
                static_cast<double>(scales_ptr[i * 3 + 2]));
            sort_keys[static_cast<size_t>(i)] = scale_volume * static_cast<double>(sigmoid(raw_opacity));
        }

        std::sort(indices.begin(), indices.end(), [&sort_keys](const int64_t a, const int64_t b) {
            return sort_keys[static_cast<size_t>(a)] > sort_keys[static_cast<size_t>(b)];
        });

        std::ofstream out;
        if (!lfs::core::open_file_for_write(options.output_path, std::ios::binary | std::ios::out, out)) {
            return make_error(ErrorCode::WRITE_FAILURE, "Failed to open SPLAT file for writing", options.output_path);
        }

        for (const int64_t idx : indices) {
            const float scale_values[3] = {
                std::exp(scales_ptr[idx * 3 + 0]),
                std::exp(scales_ptr[idx * 3 + 1]),
                std::exp(scales_ptr[idx * 3 + 2])};

            const std::array<uint8_t, 4> color = {
                quantize_byte((std::clamp(sh0_component(sh0, sh0_ptr, idx, 0) * SH_C0 + 0.5f, 0.0f, 1.0f)) * 255.0f),
                quantize_byte((std::clamp(sh0_component(sh0, sh0_ptr, idx, 1) * SH_C0 + 0.5f, 0.0f, 1.0f)) * 255.0f),
                quantize_byte((std::clamp(sh0_component(sh0, sh0_ptr, idx, 2) * SH_C0 + 0.5f, 0.0f, 1.0f)) * 255.0f),
                quantize_byte(sigmoid(opacity_component(opacity, opacity_ptr, idx)) * 255.0f)};

            const std::array<uint8_t, 4> rotation = {
                quantize_byte(rotations_ptr[idx * 4 + 0] * 128.0f + 128.0f),
                quantize_byte(rotations_ptr[idx * 4 + 1] * 128.0f + 128.0f),
                quantize_byte(rotations_ptr[idx * 4 + 2] * 128.0f + 128.0f),
                quantize_byte(rotations_ptr[idx * 4 + 3] * 128.0f + 128.0f)};

            out.write(reinterpret_cast<const char*>(means_ptr + idx * 3), sizeof(float) * 3);
            out.write(reinterpret_cast<const char*>(scale_values), sizeof(float) * 3);
            out.write(reinterpret_cast<const char*>(color.data()), color.size());
            out.write(reinterpret_cast<const char*>(rotation.data()), rotation.size());
        }

        out.close();
        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE, "Failed to write SPLAT file", options.output_path);
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

        LOG_INFO("SPLAT saved: {} gaussians, {:.1f} MB in {}ms",
                 rows,
                 static_cast<double>(estimated_size) / (1024.0 * 1024.0),
                 elapsed.count());

        return {};
    }

} // namespace lfs::io
