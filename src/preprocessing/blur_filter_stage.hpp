/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/blur_filter.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::preprocess::detail {
    // Internal staging-only adapter. Rejected copies are archived, not discarded;
    // original user images and poses are never passed here.
    inline std::expected<BlurFilterResult, Error> filter_stage_images(
        const std::filesystem::path& stage, const std::filesystem::path& images,
        const std::string& label, const ExecutionContext& ctx,
        const std::optional<std::filesystem::path>& poses = {}, bool write_diagnostic_json = true) try {
        namespace fs = std::filesystem;
        if ((label != "scan" && label != "inc") || images.parent_path() != stage ||
            (poses && poses->parent_path() != stage))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Blur filtering requires owned staging paths"});
        auto result = select_blur_images(images, {}, ctx, poses);
        if (!result)
            return std::unexpected(result.error());
        nlohmann::json report{{"schema_version", 1}, {"algorithm", "sharp_frames_0.3.1"}, {"label", label}, {"selected_names", result->selected_names}, {"removed_names", result->removed_names}, {"global_range", result->global_range}, {"threshold_percent", result->threshold_percent}, {"effective_window", result->effective_window}, {"skipped", result->skipped}, {"kept_all_fallback", result->kept_all_fallback}, {"records", nlohmann::json::array()}};
        const auto archive = stage / ("blur_removed_" + label);
        for (const auto& record : result->records) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Blur filtering cancelled"});
            const auto name = fs::u8path(record.name);
            if (name.has_parent_path() || name.empty() || record.source.parent_path() != images ||
                record.name.find_first_of("/\\:") != std::string::npos || name == "." || name == "..")
                return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Unsafe blur image mapping"});
            report["records"].push_back({{"name", record.name}, {"selected", record.selected}, {"score", record.sharpness_score ? nlohmann::json(*record.sharpness_score) : nlohmann::json(nullptr)}});
            if (record.selected)
                continue;
            fs::create_directories(archive / "images");
            fs::rename(images / name, archive / "images" / name);
            if (poses) {
                auto pose_name = name;
                pose_name.replace_extension(".pose");
                if (fs::is_regular_file(*poses / pose_name)) {
                    fs::create_directories(archive / "poses");
                    fs::rename(*poses / pose_name, archive / "poses" / pose_name);
                }
            }
        }
        if (write_diagnostic_json) {
            std::ofstream file(stage / ("blur_filter_" + label + ".json"), std::ios::binary | std::ios::noreplace);
            file << report.dump(2) << '\n';
            file.close();
            if (!file)
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot write blur filter report"});
        }
        return std::move(*result);
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
} // namespace lfs::preprocess::detail
