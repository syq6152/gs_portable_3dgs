/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/pipeline.hpp"
#include "preprocessing/dataset_validation.hpp"
#include "preprocessing/mesh_export_pipeline.hpp"
#include "preprocessing/registered_pipeline.hpp"
#include "preprocessing/scan_pipeline.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <random>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
namespace lfs::preprocess {
    namespace {
        std::string run_id() {
            static std::atomic<unsigned long long> sequence{0};
            return std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                   std::to_string(std::random_device{}()) + "-" + std::to_string(sequence++);
        }
    } // namespace
    std::expected<PreprocessResult, Error> preprocess(const PreprocessRequest& request, const ExecutionContext& ctx) {
        if (std::holds_alternative<RegisteredInput>(request.input)) {
            class BundledColmap final : public IColmapProvider {
                std::expected<ColmapResult, Error> run(const ColmapRequest& request, const ExecutionContext& context) override {
                    auto paths = resolve_runtime_paths({});
                    if (!paths)
                        return std::unexpected(paths.error());
                    return ColmapExeProvider(*paths).run(request, context);
                }
            } colmap;
            return preprocess_registered(request, &colmap, ctx);
        }
        if (std::holds_alternative<MeshExportInput>(request.input)) {
            class BundledColmap final : public IColmapProvider {
                std::expected<ColmapResult, Error> run(const ColmapRequest& request,
                                                       const ExecutionContext& context) override {
                    auto paths = resolve_runtime_paths({});
                    if (!paths)
                        return std::unexpected(paths.error());
                    return ColmapExeProvider(*paths).run(request, context);
                }
            } colmap;
            class BundledSr final : public ISuperResolutionProvider {
                std::expected<SuperResolutionResult, Error> run(const SuperResolutionRequest& sr,
                                                                const ExecutionContext& context) override {
                    auto paths = resolve_runtime_paths({});
                    if (!paths)
                        return std::unexpected(paths.error());
                    return SuperResolutionExeProvider(*paths).run(sr, context);
                }
            } sr;
            return preprocess_mesh_export(request, &colmap, &sr, ctx);
        }
        // Resolve lazily inside the owned SR stage, so runtime failures are reported
        // and cleaned up by the same transaction as other preprocessing failures.
        class BundledSr final : public ISuperResolutionProvider {
            std::expected<SuperResolutionResult, Error> run(const SuperResolutionRequest& sr,
                                                            const ExecutionContext& context) override {
                auto paths = resolve_runtime_paths({});
                if (!paths)
                    return std::unexpected(paths.error());
                return SuperResolutionExeProvider(*paths).run(sr, context);
            }
        } provider;
        return preprocess_scan(request, &provider, ctx);
    }
    std::expected<void, Error> validate_dataset_skeleton(const std::filesystem::path& root) {
        // Retained as an M1 source-compatible entry; M2 performs full validation.
        const auto valid = validate_dataset(root);
        if (!valid)
            return std::unexpected(valid.error());
        return {};
    }
    std::expected<std::filesystem::path, Error> create_workspace(const std::filesystem::path& root) {
        if (root.empty())
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "workspace root required"});
        try {
            const auto runs = std::filesystem::absolute(root) / "runs";
            std::filesystem::create_directories(runs);
            for (int attempt = 0; attempt < 10; ++attempt) {
                auto path = runs / run_id();
                if (std::filesystem::create_directory(path))
                    return path;
            }
            throw std::runtime_error("Cannot allocate unique run");
        } catch (const std::exception& e) { return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()}); }
    }
    std::expected<void, Error> publish_ply(const std::filesystem::path& source, const std::filesystem::path& destination) {
        std::filesystem::path temporary;
        try {
            if (destination.empty())
                throw std::runtime_error("Empty publish target");
            if (!std::filesystem::is_regular_file(source) || std::filesystem::file_size(source) < 16)
                throw std::runtime_error("Training PLY missing/empty");
            std::ifstream input(source, std::ios::binary);
            std::string line;
            std::getline(input, line);
            if (line != "ply" && line != "ply\r")
                throw std::runtime_error("Invalid PLY header");
            bool vertices = false, header = false;
            for (int i = 0; i < 1024 && std::getline(input, line); ++i) {
                if (line.starts_with("element vertex "))
                    vertices = std::stoull(line.substr(15)) > 0;
                if (line == "end_header" || line == "end_header\r") {
                    header = true;
                    break;
                }
            }
            if (!header || !vertices || input.peek() == std::char_traits<char>::eof())
                throw std::runtime_error("PLY has no vertices/payload");
            if (std::filesystem::exists(destination) && std::filesystem::equivalent(source, destination))
                return {};
            temporary = destination.parent_path() / (".publish-" + run_id() + ".tmp");
            std::filesystem::copy_file(source, temporary, std::filesystem::copy_options::none);
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Atomic publish failed: " + std::to_string(GetLastError()));
#else
            std::filesystem::rename(temporary, destination);
#endif
            return {};
        } catch (const std::exception& e) {
            std::error_code ignored;
            if (!temporary.empty())
                std::filesystem::remove(temporary, ignored);
            return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
        }
    }
} // namespace lfs::preprocess
