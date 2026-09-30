/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/workspace.hpp"
#include "preprocessing/pipeline.hpp"
#include "preprocessing/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cwctype>
#include <fstream>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;
        Error io_error(const std::exception& e) { return {.code = ErrorCode::IoFailure, .message = e.what()}; }
        fs::path normalized(const fs::path& p) {
            auto resolved = fs::weakly_canonical(fs::absolute(p));
#ifdef _WIN32
            auto s = resolved.native();
            std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) {
                return std::towlower(c);
            });
            return fs::path(s);
#else
            return resolved;
#endif
        }
        bool within(const fs::path& path, const fs::path& parent) {
            auto i = path.begin();
            for (auto j = parent.begin(); j != parent.end(); ++j, ++i)
                if (i == path.end() || *i != *j)
                    return false;
            return true;
        }
        bool reparse(const fs::path& p) {
#ifdef _WIN32
            const auto flags = GetFileAttributesW(p.c_str());
            return flags != INVALID_FILE_ATTRIBUTES && (flags & FILE_ATTRIBUTE_REPARSE_POINT);
#else
            return fs::is_symlink(fs::symlink_status(p));
#endif
        }
        void no_aliases(const fs::path& p) {
            fs::path part;
            for (const auto& component : fs::absolute(p)) {
                part /= component;
                if (reparse(part))
                    throw std::runtime_error("Workspace path contains a symlink/reparse point");
            }
        }
        void clean_owned_stage(const fs::path& run, const fs::path& staging) {
            no_aliases(staging);
            if (staging.parent_path() != run / ".staging" ||
                normalized(staging.parent_path()) != normalized(run / ".staging"))
                throw std::runtime_error("Refusing cleanup outside owned staging root");
            if (!fs::exists(staging))
                return;
            // Fail closed on reparse points before a recursive operation (including Windows junctions).
            for (const auto& item : fs::recursive_directory_iterator(staging))
                if (reparse(item.path()))
                    throw std::runtime_error("Refusing cleanup of staging tree with reparse points");
            fs::remove_all(staging);
        }
        void atomic_json(const fs::path& path, const Json& value) {
            auto temp = path;
            temp += ".tmp";
            no_aliases(path);
            no_aliases(temp);
            std::ofstream file(temp, std::ios::binary | std::ios::noreplace);
            if (!file)
                throw std::runtime_error("Cannot create report temporary file");
            // Process tails may end in a truncated UTF-8 sequence; keep reports parseable.
            file << value.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
            file.close();
            if (!file) {
                fs::remove(temp);
                throw std::runtime_error("Cannot write report");
            }
#ifdef _WIN32
            if (!MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                fs::remove(temp);
                throw std::runtime_error("Cannot atomically publish report");
            }
#else
            fs::rename(temp, path);
#endif
        }
        std::string error_name(ErrorCode code) {
            switch (code) {
#define LFS_ERROR_NAME(x) \
    case ErrorCode::x: return #x
                LFS_ERROR_NAME(InvalidRequest);
                LFS_ERROR_NAME(InvalidDataset);
                LFS_ERROR_NAME(UnsupportedFeature);
                LFS_ERROR_NAME(RuntimeMissing);
                LFS_ERROR_NAME(RuntimeManifestMismatch);
                LFS_ERROR_NAME(ProcessLaunchFailed);
                LFS_ERROR_NAME(ProcessFailed);
                LFS_ERROR_NAME(ProcessTimeout);
                LFS_ERROR_NAME(ProcessCancelled);
                LFS_ERROR_NAME(CallbackFailure);
                LFS_ERROR_NAME(PlatformUnsupported);
                LFS_ERROR_NAME(IncompatibleRuntime);
                LFS_ERROR_NAME(IoFailure);
#undef LFS_ERROR_NAME
            }
            return "Unknown";
        }
    } // namespace
    std::expected<void, Error> check_output_isolation(const fs::path& output, const std::vector<fs::path>& inputs) try {
        if (output.empty())
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Empty output path"});
        const auto target = normalized(output);
        for (const auto& input : inputs) {
            if (input.empty())
                continue;
            const auto source = normalized(input);
            if (within(target, source) || within(source, target))
                return std::unexpected(Error{.code = ErrorCode::InvalidRequest,
                                             .message = "Output overlaps a read-only input: " + path_utf8(input)});
        }
        return {};
    } catch (const std::exception& e) { return std::unexpected(io_error(e)); }
    struct Workspace::Impl {
        fs::path root;
        bool retain = false, terminal = false, active = false, write_diagnostic_json = true;
        Json report{{"schema_name", "lfs.swaptexture.preprocess-report"},
                    {"schema_version", 1},
                    {"producer_version", "m2.1"},
                    {"artifact_contract", "docs/swaptexture_m0/contracts/artifact_contract.json"},
                    {"parameters", Json::object()},
                    {"status", "running"},
                    {"stages", Json::array()}};
        void save() {
            if (write_diagnostic_json)
                atomic_json(root / "preprocess_report.json", report);
        }
    };
    Workspace::Workspace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
    Workspace::~Workspace() = default;
    const fs::path& Workspace::run_root() const { return impl_->root; }
    void Workspace::set_registered_parameters(const RegisteredInput& input) {
        const auto& options = input.options;
        impl_->report["producer_version"] = "m4.1";
        impl_->report["parameters"] = {{"input_mode", "registered"}, {"matcher", options.matcher}, {"triangulation", options.use_mesh_triangulation ? "mesh" : "colmap"}, {"match_3d_threshold", options.match_3d_threshold}, {"sample_limit", options.sample_limit ? Json(*options.sample_limit) : Json(nullptr)}};
        impl_->report["parameters"]["debug_mesh_overlay"] = options.enable_mesh_overlay;
        impl_->report["parameters"]["scan_blur_filter"] = options.enable_scan_blur_filter;
        impl_->report["parameters"]["incremental_blur_filter"] = options.enable_blur_filter;
        impl_->report["parameters"]["registration_triplets"] = options.enable_registration_triplets;
        impl_->report["parameters"]["register_twice"] = options.register_twice;
        impl_->report["parameters"]["face_mode"] = options.face_mode;
        impl_->report["parameters"]["face_mode_behavior"] = "legacy_noop";
        impl_->report["parameters"]["incremental_mask"] = options.enable_incremental_mask;
        impl_->report["parameters"]["foreground_model"] = options.foreground_model ? Json(path_utf8(*options.foreground_model)) : Json(nullptr);
        impl_->report["parameters"]["incremental_source"] = input.incremental_video ? "video" : "images";
        impl_->report["parameters"]["video_frame_count"] = input.incremental_video ? Json(input.video_frame_count.value_or(80)) : Json(nullptr);
    }
    std::expected<void, Error> Workspace::publish_dataset_root(const fs::path& stage) try {
        if (impl_->terminal || impl_->active || stage.parent_path() != impl_->root)
            throw std::runtime_error("Invalid dataset publication state");
        no_aliases(stage);
        if (!fs::is_directory(stage / "images") || !fs::is_directory(stage / "sparse"))
            throw std::runtime_error("Root publication requires validated images and sparse directories");
        std::vector<fs::path> artifacts;
        for (const auto& entry : fs::directory_iterator(stage)) {
            const auto name = entry.path().filename();
            if (name != "images" && name != "sparse" && name != "tex_data_inc.bin" && name != "registration_diagnostics.json")
                throw std::runtime_error("Unexpected root dataset artifact");
            no_aliases(entry.path());
            if (fs::exists(impl_->root / name))
                throw std::runtime_error("Dataset root destination exists");
            artifacts.push_back(name);
        }
        size_t moved = 0;
        try {
            for (const auto& name : artifacts) {
                fs::rename(stage / name, impl_->root / name);
                ++moved;
            }
            impl_->report["dataset_root"] = path_utf8(impl_->root);
            impl_->report["status"] = "completed";
            impl_->save();
        } catch (const std::exception& e) {
            std::string message = e.what();
            for (size_t i = moved; i > 0; --i) {
                std::error_code rollback_error;
                fs::rename(impl_->root / artifacts[i - 1], stage / artifacts[i - 1], rollback_error);
                if (rollback_error)
                    message += "; rollback " + path_utf8(artifacts[i - 1]) + ": " + rollback_error.message();
            }
            throw std::runtime_error(message);
        }
        impl_->terminal = true;
        return {};
    } catch (const std::exception& e) {
        auto error = io_error(e);
        impl_->terminal = true;
        impl_->report.erase("dataset_root");
        impl_->report["status"] = "failed";
        impl_->report["stages"].push_back({{"name", "publish_dataset_root"}, {"stage", static_cast<int>(Stage::Assemble)}, {"status", "failed"}, {"progress", 0.0}, {"artifact", nullptr}, {"retained_staging", false}, {"error", {{"code", "IoFailure"}, {"message", error.message}, {"exit_code", 0}, {"stdout_tail", ""}, {"stderr_tail", ""}}}});
        try {
            impl_->save();
        } catch (const std::exception& reporting) { error.message += "; preprocessing report: " + std::string(reporting.what()); }
        try {
            if (impl_->write_diagnostic_json)
                atomic_json(impl_->root / "failure_report.json", impl_->report);
        } catch (const std::exception& reporting) { error.message += "; failure report: " + std::string(reporting.what()); }
        return std::unexpected(std::move(error));
    }
    void Workspace::set_scan_parameters(const ScanOptions& options) {
        impl_->report["producer_version"] = "m3.1";
        impl_->report["parameters"] = {
            {"input_mode", "scan"},
            {"debug_mesh_overlay", options.enable_mesh_overlay},
            {"scan_blur_filter", options.enable_blur_filter},
            {"sample_limit", options.sample_limit ? Json(*options.sample_limit) : Json(nullptr)},
            {"denoise", options.enable_denoise},
            {"enhancement", options.enable_enhancement},
            {"super_resolution", options.super_resolution.enabled},
            {"sr_model", options.super_resolution.model_name},
            {"sr_model_scale", options.super_resolution.model_scale},
            {"sr_final_scale", options.super_resolution.final_scale}};
    }
    void Workspace::set_mesh_export_parameters(const MeshExportOptions& options) {
        impl_->report["producer_version"] = "m5.1";
        impl_->report["parameters"] = {
            {"input_mode", "mesh_export"},
            {"scan_blur_filter", options.enable_blur_filter},
            {"match_3d_threshold", options.match_3d_threshold},
            {"fill_black_component_max_pixels", options.fill_black_component_max_pixels},
            {"super_resolution", options.super_resolution.enabled},
            {"enhancement", options.enhance_before_super_resolution},
            {"sr_model", options.super_resolution.model_name},
            {"sr_model_scale", options.super_resolution.model_scale},
            {"sr_final_scale", options.super_resolution.final_scale}};
    }
    std::expected<std::unique_ptr<Workspace>, Error>
    Workspace::create(const fs::path& root, const std::vector<fs::path>& inputs, bool retain, bool write_diagnostic_json) try {
        if (auto isolated = check_output_isolation(root, inputs); !isolated)
            return std::unexpected(isolated.error());
        no_aliases(root);
        auto created = create_workspace(root);
        if (!created)
            return std::unexpected(created.error());
        auto impl = std::make_unique<Impl>();
        impl->root = *created;
        impl->retain = retain;
        impl->write_diagnostic_json = write_diagnostic_json;
        impl->report["run_root"] = path_utf8(*created);
        impl->report["retain_failed_stages"] = retain;
        impl->report["inputs"] = Json::array();
        for (const auto& p : inputs)
            impl->report["inputs"].push_back(path_utf8(fs::absolute(p)));
        // The newly allocated run is empty; reports and staging begin with the first stage.
        return std::unique_ptr<Workspace>(new Workspace(std::move(impl)));
    } catch (const std::exception& e) { return std::unexpected(io_error(e)); }
    std::expected<fs::path, Error> Workspace::run_stage(const std::string& name, Stage stage, const StageAction& action,
                                                        const StageValidator& validator, const ExecutionContext& ctx) {
        if (impl_->terminal || impl_->active || name.empty() || name.size() > 64 ||
            !std::all_of(name.begin(), name.end(),
                         [](unsigned char c) {
                             return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
                         }) ||
            !action || !validator)
            return std::unexpected(
                Error{.code = ErrorCode::InvalidRequest, .message = "Invalid stage, callback, or workspace state"});
        const auto staging = impl_->root / ".staging" / name, published = impl_->root / name;
        for (const auto& s : impl_->report["stages"])
            if (s["name"] == name)
                return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Stage already recorded"});
        impl_->active = true;
        struct Reset {
            bool& active;
            ~Reset() { active = false; }
        } reset{impl_->active};
        auto& records = impl_->report["stages"];
        records.push_back({{"name", name},
                           {"stage", static_cast<int>(stage)},
                           {"status", "running"},
                           {"progress", 0.0},
                           {"artifact", nullptr}});
        auto& record = records.back();
        bool owns_staging = false;
        bool owns_published = false;
        auto failure = [&](Error error) -> std::expected<fs::path, Error> {
            impl_->terminal = true;
            const auto status = error.code == ErrorCode::ProcessCancelled ? "cancelled" : "failed";
            record["status"] = status;
            impl_->report["status"] = status;
            // Publication is not committed until its completed report is
            // durable. If that final save fails, move the owned artifact back
            // under .staging so the ordinary retain/cleanup policy applies.
            if (owns_published)
                try {
                    no_aliases(staging);
                    no_aliases(published);
                    fs::rename(published, staging);
                    owns_published = false;
                    owns_staging = true;
                    record["artifact"] = nullptr;
                } catch (const std::exception& e) { record["cleanup_error"] = e.what(); }
            if (owns_staging && !impl_->retain)
                try {
                    clean_owned_stage(impl_->root, staging);
                    owns_staging = false;
                } catch (const std::exception& e) { record["cleanup_error"] = e.what(); }
            std::error_code staging_error;
            record["retained_staging"] = owns_staging && fs::exists(staging, staging_error);
            if (staging_error)
                record["cleanup_error"] = staging_error.message();
            std::error_code published_error;
            record["retained_published"] = owns_published && fs::exists(published, published_error);
            if (published_error)
                record["cleanup_error"] = published_error.message();
            record["error"] = {
                {"code", error_name(error.code)},
                {"message", error.message},
                {"exit_code", error.exit_code},
                {"stdout_tail",
                 error.stdout_tail.substr(error.stdout_tail.size() > 8192 ? error.stdout_tail.size() - 8192 : 0)},
                {"stderr_tail",
                 error.stderr_tail.substr(error.stderr_tail.size() > 8192 ? error.stderr_tail.size() - 8192 : 0)}};
            try {
                impl_->save();
                if (impl_->write_diagnostic_json)
                    atomic_json(impl_->root / "failure_report.json", impl_->report);
            } catch (const std::exception& e) { error.message += "; failure report: " + std::string(e.what()); }
            return std::unexpected(std::move(error));
        };
        try {
            no_aliases(staging);
            no_aliases(published);
            if (fs::exists(published) || fs::exists(staging))
                return failure({.code = ErrorCode::IoFailure, .message = "Stage path already exists"});
            fs::create_directories(staging.parent_path());
            owns_staging = fs::create_directory(staging);
            if (!owns_staging)
                throw std::runtime_error("Cannot reserve stage");
            impl_->save();
            std::optional<Error> callback_error;
            float last = 0;
            auto wrapped = ctx;
            wrapped.stage = stage;
            wrapped.on_progress = [&](Stage, float value, std::string message) {
                if (callback_error)
                    return false;
                if (ctx.stop_token.stop_requested()) {
                    callback_error = Error{.code = ErrorCode::ProcessCancelled, .message = "Stage cancelled"};
                    return false;
                }
                if (!std::isfinite(value) || value < 0 || value > 1 || value < last) {
                    callback_error = Error{.code = ErrorCode::CallbackFailure,
                                           .message = "Stage progress must be finite, normalized and monotonic"};
                    return false;
                }
                last = value;
                record["progress"] = value;
                try {
                    if (ctx.on_progress && !ctx.on_progress(stage, value, message)) {
                        callback_error =
                            Error{.code = ErrorCode::ProcessCancelled, .message = "Progress callback cancelled stage"};
                        return false;
                    }
                    if (ctx.on_stage_progress && !ctx.on_stage_progress(name, stage, value, std::move(message))) {
                        callback_error =
                            Error{.code = ErrorCode::ProcessCancelled, .message = "Named progress callback cancelled stage"};
                        return false;
                    }
                } catch (...) {
                    callback_error = Error{.code = ErrorCode::CallbackFailure, .message = "Progress callback threw"};
                    return false;
                }
                // A successful callback may request the shared stop token. The
                // final callback has no later action/validation check to catch it.
                if (ctx.stop_token.stop_requested()) {
                    callback_error = Error{.code = ErrorCode::ProcessCancelled, .message = "Stage cancelled by progress callback"};
                    return false;
                }
                return true;
            };
            wrapped.on_log = [&](Stage, std::string message) {
                try {
                    if (ctx.on_log)
                        ctx.on_log(stage, std::move(message));
                } catch (...) {
                    callback_error = Error{.code = ErrorCode::CallbackFailure, .message = "Log callback threw"};
                }
            };
            if (!wrapped.on_progress(stage, 0, "Starting stage"))
                return failure(*callback_error);
            auto result = action(staging, wrapped);
            if (callback_error)
                return failure(*callback_error);
            if (!result)
                return failure(result.error());
            if (ctx.stop_token.stop_requested())
                return failure({.code = ErrorCode::ProcessCancelled, .message = "Cancelled before stage validation"});
            auto valid = validator(staging);
            if (!valid)
                return failure(valid.error());
            if (!wrapped.on_progress(stage, 1, "Stage validated"))
                return failure(*callback_error);
            no_aliases(staging);
            no_aliases(published);
            for (const auto& item : fs::recursive_directory_iterator(staging))
                if (reparse(item.path()))
                    return failure({.code = ErrorCode::InvalidDataset, .message = "Stage artifact contains a reparse point"});
            if (ctx.stop_token.stop_requested())
                return failure({.code = ErrorCode::ProcessCancelled, .message = "Cancelled before stage publication"});
            fs::rename(staging, published);
            owns_staging = false;
            owns_published = true;
            record["status"] = "completed";
            record["artifact"] = name;
            impl_->save();
            owns_published = false;
            return published;
        } catch (const std::exception& e) { return failure(io_error(e)); } catch (...) {
            return failure({.code = ErrorCode::IoFailure, .message = "Stage callback threw a nonstandard exception"});
        }
    }
    std::expected<void, Error> Workspace::finish() {
        if (impl_->terminal || impl_->active || impl_->report["stages"].empty())
            return std::unexpected(
                Error{.code = ErrorCode::InvalidRequest, .message = "Cannot finish empty, active or failed workspace"});
        auto& record = impl_->report["stages"].back();
        try {
            impl_->report["status"] = "completed";
            impl_->save();
            impl_->terminal = true;
            return {};
        } catch (const std::exception& exception) {
            auto error = io_error(exception);
            impl_->terminal = true;
            impl_->report["status"] = "failed";
            record["status"] = "failed";
            bool retained_staging = false;
            bool retained_published = false;
            try {
                if (record["artifact"].is_string()) {
                    const auto name = record["artifact"].get<std::string>();
                    const auto published = impl_->root / name;
                    const auto staging = impl_->root / ".staging" / name;
                    no_aliases(published);
                    no_aliases(staging);
                    if (published.parent_path() != impl_->root || fs::exists(staging))
                        throw std::runtime_error("Cannot roll back terminal stage publication");
                    if (fs::exists(published)) {
                        fs::rename(published, staging);
                        record["artifact"] = nullptr;
                        if (impl_->retain)
                            retained_staging = true;
                        else
                            clean_owned_stage(impl_->root, staging);
                    }
                    retained_published = fs::exists(published);
                }
            } catch (const std::exception& cleanup) {
                record["cleanup_error"] = cleanup.what();
                if (record["artifact"].is_string())
                    retained_published = fs::exists(impl_->root / record["artifact"].get<std::string>());
            }
            record["retained_staging"] = retained_staging;
            record["retained_published"] = retained_published;
            record["error"] = {{"code", "IoFailure"},
                               {"message", error.message},
                               {"exit_code", 0},
                               {"stdout_tail", ""},
                               {"stderr_tail", ""}};
            try {
                impl_->save();
            } catch (const std::exception& reporting) {
                error.message += "; preprocessing report: " + std::string(reporting.what());
            }
            try {
                if (impl_->write_diagnostic_json)
                    atomic_json(impl_->root / "failure_report.json", impl_->report);
            } catch (const std::exception& reporting) {
                error.message += "; failure report: " + std::string(reporting.what());
            }
            return std::unexpected(std::move(error));
        }
    }
} // namespace lfs::preprocess
