/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/scanner_bin.hpp"

#include "core/path_utils.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <system_error>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace lfs::io {
    namespace {

        constexpr std::size_t HEADER_PREFIX_SIZE = 16 + 1 + 1;
        constexpr std::uint32_t MAX_INTRINSIC_GROUPS = 10'000;

        bool host_matches(const ScannerBinEndian endian) {
            return (std::endian::native == std::endian::little && endian == ScannerBinEndian::Little) || (std::endian::native == std::endian::big && endian == ScannerBinEndian::Big);
        }

        template <typename T>
            requires(std::is_integral_v<T> || std::is_enum_v<T>)
        T maybe_swap(T value, const ScannerBinEndian endian) {
            if (host_matches(endian) || sizeof(T) == 1) {
                return value;
            }
            if constexpr (std::is_enum_v<T>) {
                using U = std::underlying_type_t<T>;
                return static_cast<T>(std::byteswap(static_cast<U>(value)));
            } else {
                return std::byteswap(value);
            }
        }

        class Reader {
        public:
            explicit Reader(const std::span<const std::byte> bytes, const ScannerBinEndian endian)
                : bytes_(bytes), endian_(endian) {}

            template <typename T>
                requires std::is_integral_v<T>
            std::expected<T, std::string> read_integer(const char* field) {
                if (remaining() < sizeof(T)) {
                    return std::unexpected(std::string("Scanner bin is truncated while reading ") + field);
                }
                T value{};
                std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
                offset_ += sizeof(T);
                return maybe_swap(value, endian_);
            }

            std::expected<float, std::string> read_float(const char* field) {
                auto bits = read_integer<std::uint32_t>(field);
                if (!bits) {
                    return std::unexpected(bits.error());
                }
                return std::bit_cast<float>(*bits);
            }

            std::expected<std::string, std::string> read_string(const std::uint32_t length, const char* field) {
                if (remaining() < length) {
                    return std::unexpected(std::string("Scanner bin is truncated while reading ") + field);
                }
                const auto* first = reinterpret_cast<const char*>(bytes_.data() + offset_);
                std::string value(first, first + length);
                offset_ += length;
                return value;
            }

            [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
            [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

        private:
            std::span<const std::byte> bytes_;
            ScannerBinEndian endian_;
            std::size_t offset_ = 0;
        };

        struct HeaderFields {
            std::uint16_t version = 0;
            std::uint32_t intrinsic_count = 0;
            std::uint32_t mesh_length = 0;
        };

        std::expected<HeaderFields, std::string> parse_header_fields(
            const std::span<const std::byte> bytes,
            const ScannerBinEndian endian) {
            if (bytes.size() < 10) {
                return std::unexpected("Scanner bin is truncated while reading its header");
            }
            Reader reader(bytes, endian);
            auto version = reader.read_integer<std::uint16_t>("header version");
            auto count = reader.read_integer<std::uint32_t>("intrinsic count");
            auto mesh_length = reader.read_integer<std::uint32_t>("mesh path length");
            if (!version || !count || !mesh_length) {
                return std::unexpected(!version ? version.error() : (!count ? count.error() : mesh_length.error()));
            }
            if (*count == 0 || *count > MAX_INTRINSIC_GROUPS || *mesh_length > reader.remaining()) {
                return std::unexpected("Scanner bin header fields are outside the supported range");
            }
            return HeaderFields{*version, *count, *mesh_length};
        }

        template <typename T>
            requires std::is_integral_v<T>
        void append_integer(std::vector<std::byte>& bytes, T value, const ScannerBinEndian endian) {
            value = maybe_swap(value, endian);
            const auto* first = reinterpret_cast<const std::byte*>(&value);
            bytes.insert(bytes.end(), first, first + sizeof(T));
        }

        void append_float(std::vector<std::byte>& bytes, const float value, const ScannerBinEndian endian) {
            append_integer(bytes, std::bit_cast<std::uint32_t>(value), endian);
        }

        std::expected<std::uint32_t, std::string> checked_u32(const std::size_t value, const char* field) {
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                return std::unexpected(std::string(field) + " does not fit in the scanner bin format");
            }
            return static_cast<std::uint32_t>(value);
        }

        std::expected<void, std::string> publish_temp_file(
            const std::filesystem::path& temp_path,
            const std::filesystem::path& output_path) {
#ifdef _WIN32
            if (!MoveFileExW(temp_path.c_str(), output_path.c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                return std::unexpected("Failed to publish scanner bin atomically (Win32 error " + std::to_string(GetLastError()) + ")");
            }
#else
            std::error_code ec;
            std::filesystem::rename(temp_path, output_path, ec);
            if (ec) {
                return std::unexpected("Failed to publish scanner bin atomically: " + ec.message());
            }
#endif
            return {};
        }

    } // namespace

    std::size_t ScannerBinData::frame_count() const noexcept {
        std::size_t count = 0;
        for (const auto& group : groups) {
            count += group.frames.size();
        }
        return count;
    }

    std::expected<std::filesystem::path, std::string>
    resolve_scanner_mesh_path(const std::filesystem::path& bin_path, const ScannerBinData& data) try {
        namespace fs = std::filesystem;
        std::vector<fs::path> candidates;
        const auto primary = core::utf8_to_path(data.mesh_path_utf8);
        const auto root = fs::absolute(bin_path).parent_path();
        if (!primary.empty())
            candidates.push_back(primary);
        // load_data_from_bin checks the adjacent mesh before the broader search.
        candidates.push_back(root / "mesh.ply");
        if (!primary.empty()) {
            candidates.push_back(primary.parent_path() / "Mesh.ply");
            for (auto parent = primary.parent_path();;) {
                candidates.push_back(parent / "tex_dump" / "mesh.ply");
                candidates.push_back(parent / "tex_dump" / "Mesh.ply");
                candidates.push_back(parent / "mesh.ply");
                candidates.push_back(parent / "Mesh.ply");
                // An empty relative parent is Python's Path('.'), whose
                // tex_dump/mesh candidates must be visited before stopping.
                if (parent.empty())
                    break;
                const auto next = parent.parent_path();
                if (next == parent)
                    break;
                parent = next;
            }
        }
        candidates.push_back(root / "Mesh.ply");
        candidates.push_back(root / "tex_dump" / "mesh.ply");
        candidates.push_back(root / "tex_dump" / "Mesh.ply");
        for (auto parent = root.parent_path(); !parent.empty();) {
            candidates.push_back(parent / "tex_dump" / "mesh.ply");
            candidates.push_back(parent / "tex_dump" / "Mesh.ply");
            const auto next = parent.parent_path();
            if (next == parent)
                break;
            parent = next;
        }
        for (const auto& candidate : candidates) {
            std::error_code ec;
            if (!fs::exists(candidate, ec))
                continue;
            if (!fs::is_regular_file(candidate, ec))
                return std::unexpected("Scanner mesh is missing or not a regular file: " + core::path_to_utf8(candidate));
            return fs::absolute(candidate);
        }
        return std::unexpected("Scanner mesh is missing (bin: " + core::path_to_utf8(bin_path) + ")");
    } catch (const std::exception& error) {
        return std::unexpected(std::string("Cannot resolve scanner mesh: ") + error.what());
    }

    std::expected<ScannerBinData, std::string> read_scanner_bin(const std::filesystem::path& path) {
        std::ifstream stream;
        if (!lfs::core::open_file_for_read(path, std::ios::binary, stream)) {
            return std::unexpected("Failed to open scanner bin: " + lfs::core::path_to_utf8(path));
        }

        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        if (end < 0) {
            return std::unexpected("Failed to query scanner bin size: " + lfs::core::path_to_utf8(path));
        }
        const auto file_size = static_cast<std::uintmax_t>(end);
        if (file_size < HEADER_PREFIX_SIZE + 10 || file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
            return std::unexpected("Scanner bin size is invalid: " + lfs::core::path_to_utf8(path));
        }

        std::vector<std::byte> bytes(static_cast<std::size_t>(file_size));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream || static_cast<std::size_t>(stream.gcount()) != bytes.size()) {
            return std::unexpected("Failed to read scanner bin: " + lfs::core::path_to_utf8(path));
        }

        ScannerBinData result;
        std::memcpy(result.magic.data(), bytes.data(), result.magic.size());
        const auto endian_flag = std::to_integer<std::uint8_t>(bytes[16]);
        result.reserved = std::to_integer<std::uint8_t>(bytes[17]);
        if (endian_flag != static_cast<std::uint8_t>(ScannerBinEndian::Little) && endian_flag != static_cast<std::uint8_t>(ScannerBinEndian::Big)) {
            return std::unexpected("Scanner bin has an invalid endian flag");
        }

        result.endian = static_cast<ScannerBinEndian>(endian_flag);
        const auto header_bytes = std::span<const std::byte>(bytes).subspan(18);
        auto header = parse_header_fields(header_bytes, result.endian);
        if (!header) {
            const auto fallback = result.endian == ScannerBinEndian::Little
                                      ? ScannerBinEndian::Big
                                      : ScannerBinEndian::Little;
            header = parse_header_fields(header_bytes, fallback);
            if (!header) {
                return std::unexpected("Failed to parse scanner bin header: " + header.error());
            }
            result.endian = fallback;
        }

        result.header_version = header->version;
        Reader reader(header_bytes, result.endian);
        (void)reader.read_integer<std::uint16_t>("header version");
        (void)reader.read_integer<std::uint32_t>("intrinsic count");
        (void)reader.read_integer<std::uint32_t>("mesh path length");

        auto mesh_path = reader.read_string(header->mesh_length, "mesh path");
        if (!mesh_path) {
            return std::unexpected(mesh_path.error());
        }
        result.mesh_path_utf8 = std::move(*mesh_path);
        result.groups.reserve(header->intrinsic_count);

        for (std::uint32_t group_index = 0; group_index < header->intrinsic_count; ++group_index) {
            ScannerBinIntrinsicGroup group;
            auto fx = reader.read_float("fx");
            auto fy = reader.read_float("fy");
            auto cx = reader.read_float("cx");
            auto cy = reader.read_float("cy");
            auto frame_count = reader.read_integer<std::uint32_t>("frame count");
            if (!fx || !fy || !cx || !cy || !frame_count) {
                if (!fx)
                    return std::unexpected(fx.error());
                if (!fy)
                    return std::unexpected(fy.error());
                if (!cx)
                    return std::unexpected(cx.error());
                if (!cy)
                    return std::unexpected(cy.error());
                return std::unexpected(frame_count.error());
            }
            if (*frame_count > reader.remaining() / (sizeof(std::uint32_t) + 16 * sizeof(float))) {
                return std::unexpected("Scanner bin frame count exceeds the remaining data");
            }
            group.fx = *fx;
            group.fy = *fy;
            group.cx = *cx;
            group.cy = *cy;
            group.frames.reserve(*frame_count);

            for (std::uint32_t frame_index = 0; frame_index < *frame_count; ++frame_index) {
                auto path_length = reader.read_integer<std::uint32_t>("image path length");
                if (!path_length) {
                    return std::unexpected(path_length.error());
                }
                auto image_path = reader.read_string(*path_length, "image path");
                if (!image_path) {
                    return std::unexpected(image_path.error());
                }
                ScannerBinFrame frame;
                frame.image_path_utf8 = std::move(*image_path);
                for (float& value : frame.pose_c2w) {
                    auto component = reader.read_float("pose matrix");
                    if (!component) {
                        return std::unexpected(component.error());
                    }
                    value = *component;
                }
                group.frames.push_back(std::move(frame));
            }
            result.groups.push_back(std::move(group));
        }

        const auto trailing = std::span<const std::byte>(bytes).subspan(18 + reader.offset());
        result.trailing_bytes.assign(trailing.begin(), trailing.end());
        return result;
    }

    std::expected<void, std::string> write_scanner_bin_atomic(
        const std::filesystem::path& path,
        const ScannerBinData& data) {
        if (data.groups.empty() || data.groups.size() > MAX_INTRINSIC_GROUPS) {
            return std::unexpected("Scanner bin must contain between 1 and 10000 intrinsic groups");
        }

        auto group_count = checked_u32(data.groups.size(), "Intrinsic group count");
        auto mesh_length = checked_u32(data.mesh_path_utf8.size(), "Mesh path length");
        if (!group_count || !mesh_length) {
            return std::unexpected(!group_count ? group_count.error() : mesh_length.error());
        }

        std::vector<std::byte> bytes;
        bytes.reserve(HEADER_PREFIX_SIZE + 10 + data.mesh_path_utf8.size() + data.frame_count() * (sizeof(std::uint32_t) + 16 * sizeof(float)));
        const auto* magic = reinterpret_cast<const std::byte*>(data.magic.data());
        bytes.insert(bytes.end(), magic, magic + data.magic.size());
        bytes.push_back(static_cast<std::byte>(data.endian));
        bytes.push_back(static_cast<std::byte>(data.reserved));
        append_integer(bytes, data.header_version, data.endian);
        append_integer(bytes, *group_count, data.endian);
        append_integer(bytes, *mesh_length, data.endian);
        const auto* mesh = reinterpret_cast<const std::byte*>(data.mesh_path_utf8.data());
        bytes.insert(bytes.end(), mesh, mesh + data.mesh_path_utf8.size());

        for (const auto& group : data.groups) {
            auto frame_count = checked_u32(group.frames.size(), "Frame count");
            if (!frame_count) {
                return std::unexpected(frame_count.error());
            }
            append_float(bytes, group.fx, data.endian);
            append_float(bytes, group.fy, data.endian);
            append_float(bytes, group.cx, data.endian);
            append_float(bytes, group.cy, data.endian);
            append_integer(bytes, *frame_count, data.endian);

            for (const auto& frame : group.frames) {
                auto path_length = checked_u32(frame.image_path_utf8.size(), "Image path length");
                if (!path_length) {
                    return std::unexpected(path_length.error());
                }
                append_integer(bytes, *path_length, data.endian);
                const auto* image_path = reinterpret_cast<const std::byte*>(frame.image_path_utf8.data());
                bytes.insert(bytes.end(), image_path, image_path + frame.image_path_utf8.size());
                for (const float value : frame.pose_c2w) {
                    append_float(bytes, value, data.endian);
                }
            }
        }
        bytes.insert(bytes.end(), data.trailing_bytes.begin(), data.trailing_bytes.end());

        std::error_code ec;
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec) {
                return std::unexpected("Failed to create scanner bin output directory: " + ec.message());
            }
        }

        auto temp_path = path;
        temp_path += ".tmp." + std::to_string(
                                   std::chrono::steady_clock::now().time_since_epoch().count());
        std::ofstream stream;
        if (!lfs::core::open_file_for_write(temp_path, std::ios::binary | std::ios::trunc, stream)) {
            return std::unexpected("Failed to open temporary scanner bin: " + lfs::core::path_to_utf8(temp_path));
        }
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        if (!stream) {
            stream.close();
            std::filesystem::remove(temp_path, ec);
            return std::unexpected("Failed to write temporary scanner bin: " + lfs::core::path_to_utf8(temp_path));
        }
        stream.close();

        auto published = publish_temp_file(temp_path, path);
        if (!published) {
            std::filesystem::remove(temp_path, ec);
            return published;
        }
        return {};
    }

} // namespace lfs::io
