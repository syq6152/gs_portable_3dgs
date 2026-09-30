/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace lfs::io {

    enum class ScannerBinEndian : std::uint8_t {
        Little = 0x01,
        Big = 0x02,
    };

    struct ScannerBinFrame {
        std::string image_path_utf8;
        std::array<float, 16> pose_c2w{};
    };

    struct ScannerBinIntrinsicGroup {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        std::vector<ScannerBinFrame> frames;
    };

    struct ScannerBinData {
        std::array<char, 16> magic{};
        ScannerBinEndian endian = ScannerBinEndian::Little;
        std::uint8_t reserved = 0;
        std::uint16_t header_version = 0;
        std::string mesh_path_utf8;
        std::vector<ScannerBinIntrinsicGroup> groups;
        std::vector<std::byte> trailing_bytes;

        [[nodiscard]] std::size_t frame_count() const noexcept;
    };

    [[nodiscard]] LFS_IO_API std::expected<ScannerBinData, std::string>
    read_scanner_bin(const std::filesystem::path& path);

    // Resolve the external mesh with Python load_data_from_bin fallback order.
    // Bin-internal relative paths retain cwd semantics; mesh coordinates are unchanged.
    [[nodiscard]] LFS_IO_API std::expected<std::filesystem::path, std::string>
    resolve_scanner_mesh_path(const std::filesystem::path& bin_path, const ScannerBinData& data);

    // Writes to a temporary sibling and atomically publishes the completed file.
    [[nodiscard]] LFS_IO_API std::expected<void, std::string>
    write_scanner_bin_atomic(const std::filesystem::path& path, const ScannerBinData& data);

} // namespace lfs::io
