/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/colmap_model.hpp"
#include <memory>
#include <span>

namespace lfs::preprocess {
    constexpr uint32_t colmap_max_image_id = 2147483647;
    std::expected<uint64_t, Error> image_pair_id(uint32_t first, uint32_t second);
    struct Keypoints {
        uint32_t columns = 2;
        std::vector<float> data;
        bool operator==(const Keypoints&) const = default;
    };
    using Matches = std::vector<std::array<uint32_t, 2>>;
    struct TwoViewGeometry {
        int configuration = 0;
        Matches matches;
        std::optional<std::array<double, 9>> fundamental, essential, homography;
        std::optional<std::array<double, 4>> rotation;
        std::optional<std::array<double, 3>> translation;
        bool operator==(const TwoViewGeometry&) const = default;
    };
    enum class DatabaseSchema { Legacy,
                                RigFrames };
    // Only creates a fresh private DB or clones a read-only source. No writable open of user input.
    class ColmapDatabase {
    public:
        static std::expected<std::unique_ptr<ColmapDatabase>, Error> create(const std::filesystem::path&, DatabaseSchema = DatabaseSchema::RigFrames);
        static std::expected<std::unique_ptr<ColmapDatabase>, Error> clone(const std::filesystem::path& source, const std::filesystem::path& destination);
        ~ColmapDatabase();
        ColmapDatabase(const ColmapDatabase&) = delete;
        ColmapDatabase& operator=(const ColmapDatabase&) = delete;
        std::expected<void, Error> sync_metadata(const ColmapModel&);
        // Scan-only phase: preserve extracted feature/image IDs while replacing
        // trivial extractor camera/rig/frame assignments with scanner authority.
        // Requires the complete existing image set and rejects multi-sensor rigs.
        std::expected<void, Error> synchronize_scan_metadata(const ColmapModel&);
        std::expected<ColmapModel, Error> read_metadata() const;
        std::expected<std::map<std::string, uint64_t>, Error> statistics() const;
        std::expected<void, Error> write_keypoints(uint32_t image, const Keypoints&);
        std::expected<Keypoints, Error> read_keypoints(uint32_t image) const;
        std::expected<void, Error> write_matches(uint32_t first, uint32_t second, const Matches&);
        std::expected<Matches, Error> read_matches(uint32_t first, uint32_t second) const;
        // Geometry matrices/relative pose are directional: caller must supply canonical first < second.
        std::expected<void, Error> write_geometry(uint32_t first, uint32_t second, const TwoViewGeometry&);
        std::expected<TwoViewGeometry, Error> read_geometry(uint32_t first, uint32_t second) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        explicit ColmapDatabase(std::unique_ptr<Impl>);
        std::expected<void, Error> sync_metadata_impl(const ColmapModel&, bool rebuild_scan_rigs);
    };
} // namespace lfs::preprocess
