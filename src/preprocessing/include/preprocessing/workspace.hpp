/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/types.hpp"
#include <expected>
#include <memory>
#include <vector>

namespace lfs::preprocess {
    // Reject output equal to, inside, or containing any read-only input, including aliases.
    std::expected<void, Error> check_output_isolation(const std::filesystem::path& output,
                                                      const std::vector<std::filesystem::path>& inputs);
    using StageAction = std::function<std::expected<void, Error>(const std::filesystem::path&, const ExecutionContext&)>;
    using StageValidator = std::function<std::expected<void, Error>(const std::filesystem::path&)>;

    // One serial pipeline owns one fresh run. It never removes the run or training outputs.
    class Workspace {
    public:
        static std::expected<std::unique_ptr<Workspace>, Error> create(
            const std::filesystem::path& root, const std::vector<std::filesystem::path>& read_only_inputs,
            bool retain_failed_stages = false, bool write_diagnostic_json = true);
        ~Workspace();
        Workspace(const Workspace&) = delete;
        Workspace& operator=(const Workspace&) = delete;
        const std::filesystem::path& run_root() const;
        void set_scan_parameters(const ScanOptions&);
        void set_registered_parameters(const RegisteredInput&);
        void set_mesh_export_parameters(const MeshExportOptions&);
        // Promote a validated owned stage to root and finalize its completed report.
        // Roll back artifact renames if publication or report persistence fails;
        // a successful call is terminal and must not be followed by finish().
        std::expected<void, Error> publish_dataset_root(const std::filesystem::path& stage);
        std::expected<std::filesystem::path, Error> run_stage(
            const std::string& name, Stage, const StageAction&, const StageValidator&, const ExecutionContext& = {});
        std::expected<void, Error> finish();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        explicit Workspace(std::unique_ptr<Impl>);
    };
} // namespace lfs::preprocess
