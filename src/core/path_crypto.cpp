/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/path_crypto.hpp"

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace {

    using EvpCipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;

    std::string openssl_error() {
        const unsigned long code = ERR_get_error();
        if (code == 0) {
            return "OpenSSL operation failed";
        }

        std::array<char, 256> buffer{};
        ERR_error_string_n(code, buffer.data(), buffer.size());
        return std::string(buffer.data());
    }

    int hex_value(const char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return -1;
    }

    std::expected<std::vector<std::uint8_t>, std::string> hex_decode(const std::string_view text) {
        if ((text.size() % 2) != 0) {
            return std::unexpected("Ciphertext hex length must be even");
        }

        std::vector<std::uint8_t> bytes;
        bytes.reserve(text.size() / 2);

        for (size_t i = 0; i < text.size(); i += 2) {
            const int hi = hex_value(text[i]);
            const int lo = hex_value(text[i + 1]);
            if (hi < 0 || lo < 0) {
                return std::unexpected("Ciphertext contains non-hex characters");
            }
            bytes.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
        }

        return bytes;
    }

    std::string hex_encode(const std::vector<std::uint8_t>& bytes) {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string text;
        text.resize(bytes.size() * 2);

        for (size_t i = 0; i < bytes.size(); ++i) {
            text[i * 2] = kHex[(bytes[i] >> 4) & 0x0F];
            text[i * 2 + 1] = kHex[bytes[i] & 0x0F];
        }

        return text;
    }

    std::array<std::uint8_t, 32> derive_key(const std::string_view password) {
        std::array<std::uint8_t, 32> key{};
        unsigned int key_len = 0;
        EVP_Digest(
            password.data(),
            password.size(),
            key.data(),
            &key_len,
            EVP_sha256(),
            nullptr);
        return key;
    }

    bool is_valid_utf8(const std::string_view text) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(text.data());
        const size_t len = text.size();

        for (size_t i = 0; i < len;) {
            const std::uint8_t c = bytes[i];
            if (c <= 0x7F) {
                ++i;
            } else if (c >= 0xC2 && c <= 0xDF) {
                if (i + 1 >= len || (bytes[i + 1] & 0xC0) != 0x80) {
                    return false;
                }
                i += 2;
            } else if (c == 0xE0) {
                if (i + 2 >= len || bytes[i + 1] < 0xA0 || bytes[i + 1] > 0xBF || (bytes[i + 2] & 0xC0) != 0x80) {
                    return false;
                }
                i += 3;
            } else if (c >= 0xE1 && c <= 0xEC) {
                if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) {
                    return false;
                }
                i += 3;
            } else if (c == 0xED) {
                if (i + 2 >= len || bytes[i + 1] < 0x80 || bytes[i + 1] > 0x9F || (bytes[i + 2] & 0xC0) != 0x80) {
                    return false;
                }
                i += 3;
            } else if (c >= 0xEE && c <= 0xEF) {
                if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) {
                    return false;
                }
                i += 3;
            } else if (c == 0xF0) {
                if (i + 3 >= len || bytes[i + 1] < 0x90 || bytes[i + 1] > 0xBF || (bytes[i + 2] & 0xC0) != 0x80 ||
                    (bytes[i + 3] & 0xC0) != 0x80) {
                    return false;
                }
                i += 4;
            } else if (c >= 0xF1 && c <= 0xF3) {
                if (i + 3 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80 ||
                    (bytes[i + 3] & 0xC0) != 0x80) {
                    return false;
                }
                i += 4;
            } else if (c == 0xF4) {
                if (i + 3 >= len || bytes[i + 1] < 0x80 || bytes[i + 1] > 0x8F || (bytes[i + 2] & 0xC0) != 0x80 ||
                    (bytes[i + 3] & 0xC0) != 0x80) {
                    return false;
                }
                i += 4;
            } else {
                return false;
            }
        }

        return true;
    }

} // namespace

std::expected<std::string, std::string> lfs::core::path_crypto::encrypt_path(
    const std::string_view plaintext,
    const std::string_view password) {

    const auto binary_result = encrypt_text_to_binary(plaintext, password);
    if (!binary_result) {
        return std::unexpected(binary_result.error());
    }

    return hex_encode(*binary_result);
}

std::expected<std::vector<std::uint8_t>, std::string> lfs::core::path_crypto::encrypt_text_to_binary(
    const std::string_view plaintext,
    const std::string_view password) {

    if (plaintext.size() > static_cast<size_t>(std::numeric_limits<int>::max() - EVP_MAX_BLOCK_LENGTH)) {
        return std::unexpected("Plaintext is too long to encrypt");
    }

    const auto key = derive_key(password);
    std::array<std::uint8_t, 16> iv{};
    if (RAND_bytes(iv.data(), static_cast<int>(iv.size())) != 1) {
        return std::unexpected(openssl_error());
    }

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx) {
        return std::unexpected("Failed to allocate OpenSSL cipher context");
    }

    if (EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_cbc(), nullptr, key.data(), iv.data()) != 1) {
        return std::unexpected(openssl_error());
    }

    std::vector<std::uint8_t> encrypted(plaintext.size() + EVP_CIPHER_CTX_block_size(ctx.get()));
    int update_len = 0;
    if (EVP_EncryptUpdate(
            ctx.get(),
            encrypted.data(),
            &update_len,
            reinterpret_cast<const std::uint8_t*>(plaintext.data()),
            static_cast<int>(plaintext.size())) != 1) {
        return std::unexpected(openssl_error());
    }

    int final_len = 0;
    if (EVP_EncryptFinal_ex(ctx.get(), encrypted.data() + update_len, &final_len) != 1) {
        return std::unexpected(openssl_error());
    }
    encrypted.resize(static_cast<size_t>(update_len + final_len));

    std::vector<std::uint8_t> combined;
    combined.reserve(iv.size() + encrypted.size());
    combined.insert(combined.end(), iv.begin(), iv.end());
    combined.insert(combined.end(), encrypted.begin(), encrypted.end());

    return combined;
}

std::expected<std::string, std::string> lfs::core::path_crypto::decrypt_path(
    const std::string_view ciphertext_hex,
    const std::string_view password) {

    const auto decoded_result = hex_decode(ciphertext_hex);
    if (!decoded_result) {
        return std::unexpected(decoded_result.error());
    }

    return decrypt_text_from_binary(
        std::span<const std::uint8_t>(decoded_result->data(), decoded_result->size()),
        password);
}

std::expected<std::string, std::string> lfs::core::path_crypto::decrypt_text_from_binary(
    const std::span<const std::uint8_t> decoded,
    const std::string_view password) {

    constexpr size_t iv_size = 16;
    if (decoded.size() <= iv_size || ((decoded.size() - iv_size) % iv_size) != 0) {
        return std::unexpected("Ciphertext must be iv + AES-CBC ciphertext");
    }

    const auto key = derive_key(password);
    const std::uint8_t* iv = decoded.data();
    const std::uint8_t* encrypted = decoded.data() + iv_size;
    const size_t encrypted_size = decoded.size() - iv_size;

    if (encrypted_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::unexpected("Ciphertext is too long to decrypt");
    }

    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new(), EVP_CIPHER_CTX_free);
    if (!ctx) {
        return std::unexpected("Failed to allocate OpenSSL cipher context");
    }

    if (EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_cbc(), nullptr, key.data(), iv) != 1) {
        return std::unexpected(openssl_error());
    }

    std::vector<std::uint8_t> plaintext(encrypted_size + EVP_CIPHER_CTX_block_size(ctx.get()));
    int update_len = 0;
    if (EVP_DecryptUpdate(
            ctx.get(),
            plaintext.data(),
            &update_len,
            encrypted,
            static_cast<int>(encrypted_size)) != 1) {
        return std::unexpected(openssl_error());
    }

    int final_len = 0;
    if (EVP_DecryptFinal_ex(ctx.get(), plaintext.data() + update_len, &final_len) != 1) {
        return std::unexpected(openssl_error());
    }
    plaintext.resize(static_cast<size_t>(update_len + final_len));

    std::string result(reinterpret_cast<const char*>(plaintext.data()), plaintext.size());
    if (!is_valid_utf8(result)) {
        return std::unexpected("Decrypted text is not valid UTF-8");
    }

    return result;
}

std::string lfs::core::path_crypto::decrypt_path_or_original(
    const std::string_view value,
    const std::string_view password) {

    const auto decrypted = decrypt_path(value, password);
    if (decrypted) {
        return *decrypted;
    }
    return std::string(value);
}
