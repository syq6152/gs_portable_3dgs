/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core::path_crypto {

    inline constexpr std::string_view kDefaultPassword = "851479obc781532";

    LFS_CORE_API std::expected<std::string, std::string> encrypt_path(
        std::string_view plaintext,
        std::string_view password = kDefaultPassword);

    LFS_CORE_API std::expected<std::string, std::string> decrypt_path(
        std::string_view ciphertext_hex,
        std::string_view password = kDefaultPassword);

    LFS_CORE_API std::expected<std::vector<std::uint8_t>, std::string> encrypt_text_to_binary(
        std::string_view plaintext,
        std::string_view password = kDefaultPassword);

    LFS_CORE_API std::expected<std::string, std::string> decrypt_text_from_binary(
        std::span<const std::uint8_t> ciphertext,
        std::string_view password = kDefaultPassword);

    LFS_CORE_API std::string decrypt_path_or_original(
        std::string_view value,
        std::string_view password = kDefaultPassword);

} // namespace lfs::core::path_crypto
