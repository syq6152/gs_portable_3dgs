/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/types.hpp"
#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <vector>

namespace lfs::preprocess {
    struct ColmapCamera {
        uint32_t id = 0;
        int model = 1;
        uint64_t width = 0, height = 0;
        std::vector<double> parameters;
        bool operator==(const ColmapCamera&) const = default;
    };
    struct Observation {
        double x = 0, y = 0;
        int64_t point_id = -1;
        bool operator==(const Observation&) const = default;
    };
    struct ColmapImage {
        uint32_t id = 0, camera_id = 0;
        std::array<double, 4> rotation{1, 0, 0, 0}; // w2c, qw qx qy qz
        std::array<double, 3> translation{};
        std::string name;
        std::vector<Observation> observations;
        bool operator==(const ColmapImage&) const = default;
    };
    struct TrackElement {
        uint32_t image_id = 0, point_index = 0;
        bool operator==(const TrackElement&) const = default;
    };
    struct ColmapPoint {
        uint64_t id = 0;
        std::array<double, 3> position{};
        std::array<uint8_t, 3> color{};
        double error = 0;
        std::vector<TrackElement> track;
        bool operator==(const ColmapPoint&) const = default;
    };
    struct ColmapModel {
        std::map<uint32_t, ColmapCamera> cameras;
        std::map<uint32_t, ColmapImage> images;
        // Missing is allowed; present but empty/corrupt is not a valid training dataset.
        std::optional<std::map<uint64_t, ColmapPoint>> points;
        bool operator==(const ColmapModel&) const = default;
    };
    enum class ModelFormat { Text,
                             Binary };
    std::string camera_model_name(int id);
    int camera_model_id(const std::string& name);
    std::size_t camera_parameter_count(int id);
    std::expected<void, Error> validate_model(const ColmapModel& model);
    std::expected<ColmapModel, Error> read_colmap_model(const std::filesystem::path& sparse);
    // Only writes a fresh directory; callers publish their owned stage atomically.
    std::expected<void, Error> write_colmap_model(const std::filesystem::path& sparse, const ColmapModel&, ModelFormat);
    std::expected<ColmapImage, Error> scanner_pose_to_colmap(const std::array<float, 16>& c2w_mm);
    std::expected<ColmapCamera, Error> scale_camera(const ColmapCamera&, uint64_t width, uint64_t height);
    std::array<double, 9> quaternion_rotation(const std::array<double, 4>& quaternion);
} // namespace lfs::preprocess
