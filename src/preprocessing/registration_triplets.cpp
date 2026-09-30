/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/registration_triplets.hpp"
#include "preprocessing/runtime.hpp"
#include "preprocessing/workspace.hpp"
#include <algorithm>
#include <cctype>
#include <cwctype>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        struct CopyEntry {
            fs::path source;
            fs::path relative_destination;
        };
        bool reparse(const fs::path& path) {
#ifdef _WIN32
            const auto flags = GetFileAttributesW(path.c_str());
            return flags != INVALID_FILE_ATTRIBUTES && (flags & FILE_ATTRIBUTE_REPARSE_POINT);
#else
            return fs::is_symlink(fs::symlink_status(path));
#endif
        }
        void no_aliases(const fs::path& path) {
            fs::path part;
            for (const auto& component : fs::absolute(path)) {
                part /= component;
                if (reparse(part))
                    throw std::runtime_error("Registration triplet path contains a symlink/reparse point");
            }
        }
        auto filename_key(const fs::path& name) {
            auto key = name.native();
#ifdef _WIN32
            std::transform(key.begin(), key.end(), key.begin(), [](wchar_t c) { return std::towlower(c); });
#endif
            return key;
        }
        bool png_name(const fs::path& name) {
            auto extension = name.extension().string();
#ifdef _WIN32
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
#endif
            return extension == ".png";
        }
        std::vector<CopyEntry> collect_pngs(const fs::path& incremental, const fs::path& scan, const char* folder) {
            std::map<fs::path::string_type, CopyEntry> entries;
            for (const auto& directory : {incremental, scan}) {
                if (directory.empty() || !fs::is_directory(directory))
                    continue;
                no_aliases(directory);
                std::vector<fs::path> files;
                for (const auto& file : fs::directory_iterator(directory))
                    if (png_name(file.path()) && file.is_regular_file()) {
                        no_aliases(file.path());
                        files.push_back(file.path());
                    }
                std::sort(files.begin(), files.end());
                for (const auto& file : files) {
                    const auto name = file.filename();
                    auto [it, added] = entries.try_emplace(filename_key(name), CopyEntry{file, fs::path(folder) / name});
                    // Retain the first destination spelling, as overwrite-copy
                    // does on case-insensitive Windows volumes.
                    if (!added)
                        it->second.source = file;
                }
            }
            std::vector<CopyEntry> result;
            for (auto& [key, entry] : entries)
                result.push_back(std::move(entry));
            return result;
        }
        std::expected<void, Error> progress(const ExecutionContext& ctx, float fraction) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Registration triplet export cancelled"});
            try {
                if (ctx.on_progress && !ctx.on_progress(ctx.stage, fraction, "Exporting registration triplets"))
                    return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Registration triplet export cancelled"});
            } catch (const std::exception& error) {
                return std::unexpected(Error{.code = ErrorCode::CallbackFailure, .message = error.what()});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::CallbackFailure, .message = "Registration triplet callback failed"});
            }
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Registration triplet export cancelled"});
            return {};
        }
    } // namespace

    std::expected<RegistrationTripletResult, Error> write_registration_triplets(
        const RegistrationTripletSources& sources, const fs::path& output, const ExecutionContext& ctx,
        bool write_diagnostic_json) try {
        if (ctx.stop_token.stop_requested())
            return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Registration triplet export cancelled"});
        const std::vector<fs::path> inputs{sources.aligned_sparse, sources.undistorted_sparse,
                                           sources.incremental_images, sources.incremental_masks,
                                           sources.scan_images, sources.scan_masks};
        if (auto isolated = check_output_isolation(output, inputs); !isolated)
            return std::unexpected(isolated.error());
        no_aliases(output);
        if (!fs::is_directory(output) || !fs::is_empty(output))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Registration triplet output must be a fresh empty owned stage"});

        auto copies = collect_pngs(sources.incremental_images, sources.scan_images, "images");
        RegistrationTripletResult result{.images = copies.size()};
        auto masks = collect_pngs(sources.incremental_masks, sources.scan_masks, "mask");
        result.masks = masks.size();
        copies.insert(copies.end(), masks.begin(), masks.end());
        for (const auto* name : {"cameras.txt", "images.txt", "points3D.txt"})
            for (const auto& directory : {sources.aligned_sparse, sources.undistorted_sparse})
                if (!directory.empty() && fs::is_regular_file(directory / name)) {
                    no_aliases(directory / name);
                    copies.push_back({directory / name, fs::path("sparse") / name});
                    ++result.sparse_files;
                    break;
                }
        if (auto tick = progress(ctx, 0); !tick)
            return std::unexpected(tick.error());
        for (const auto* directory : {"images", "sparse", "mask"})
            if (!fs::create_directory(output / directory))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot create registration triplet output directory"});
        nlohmann::json manifest{{"schema_version", 1}, {"kind", "registration_triplets"}, {"images", result.images}, {"masks", result.masks}, {"sparse_files", result.sparse_files}, {"files", nlohmann::json::array()}};
        std::size_t completed = 0;
        for (const auto& copy : copies) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Registration triplet export cancelled"});
            if (!fs::copy_file(copy.source, output / copy.relative_destination, fs::copy_options::none))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot copy registration triplet artifact"});
            if (write_diagnostic_json)
                manifest["files"].push_back({{"source", path_utf8(copy.source)}, {"output", path_utf8(copy.relative_destination)}});
            if (auto tick = progress(ctx, float(++completed) / float(copies.size())); !tick)
                return std::unexpected(tick.error());
        }
        if (copies.empty())
            if (auto tick = progress(ctx, 1); !tick)
                return std::unexpected(tick.error());
        if (write_diagnostic_json) {
            std::ofstream file(output / "manifest.json", std::ios::binary | std::ios::noreplace);
            file << manifest.dump(2) << '\n';
            file.close();
            if (!file)
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot write registration triplet manifest"});
        }
        return result;
    } catch (const std::exception& error) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = error.what()});
    }
} // namespace lfs::preprocess
