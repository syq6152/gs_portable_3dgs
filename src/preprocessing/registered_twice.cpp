/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/registered_twice.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>

namespace lfs::preprocess {
    namespace {
        using M3 = std::array<double, 9>;
        using V3 = std::array<double, 3>;
        M3 tr(const M3& m) {
            return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
        }
        M3 mul(const M3& a, const M3& b) {
            M3 r{};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    for (size_t k = 0; k < 3; ++k)
                        r[3 * i + j] += a[3 * i + k] * b[3 * k + j];
            return r;
        }
        V3 mul(const M3& a, const V3& b) {
            V3 r{};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    r[i] += a[3 * i + j] * b[j];
            return r;
        }
        V3 center(const ColmapImage& image) {
            const auto v = mul(tr(quaternion_rotation(image.rotation)), image.translation);
            return {-v[0], -v[1], -v[2]};
        }
        double distance(const V3& a, const V3& b) {
            return std::hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        }
        std::array<double, 4> quaternion(const M3& m) {
            std::array<double, 4> q{};
            const double trace = m[0] + m[4] + m[8];
            if (trace > 0) {
                const double s = std::sqrt(trace + 1) * 2;
                q = {s / 4, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s};
            } else if (m[0] > m[4] && m[0] > m[8]) {
                const double s = std::sqrt(1 + m[0] - m[4] - m[8]) * 2;
                q = {(m[7] - m[5]) / s, s / 4, (m[1] + m[3]) / s, (m[2] + m[6]) / s};
            } else if (m[4] > m[8]) {
                const double s = std::sqrt(1 + m[4] - m[0] - m[8]) * 2;
                q = {(m[2] - m[6]) / s, (m[1] + m[3]) / s, s / 4, (m[5] + m[7]) / s};
            } else {
                const double s = std::sqrt(1 + m[8] - m[0] - m[4]) * 2;
                q = {(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, s / 4};
            }
            double norm = 0;
            for (double v : q)
                norm += v * v;
            norm = std::sqrt(norm);
            for (double& v : q)
                v /= norm;
            if (q[0] < 0)
                for (double& v : q)
                    v = -v;
            return q;
        }
        std::expected<RegisterTwiceMergeResult, Error> bad(std::string message) {
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = std::move(message)});
        }
    } // namespace

    std::expected<RegisterTwiceMergeResult, Error> merge_register_twice_models(
        const ColmapModel& primary, const std::vector<ColmapModel>& secondary_models) {
        if (const auto valid = validate_model(primary); !valid)
            return std::unexpected(valid.error());
        for (const auto& secondary : secondary_models)
            if (const auto valid = validate_model(secondary); !valid)
                return std::unexpected(valid.error());
        RegisterTwiceMergeResult result{.model = primary};
        std::map<std::string, uint32_t> primary_ids;
        for (const auto& [id, image] : primary.images)
            primary_ids.emplace(image.name, id);
        std::set<std::string> appended_names;
        uint32_t next_id = primary.images.rbegin()->first;
        for (const auto& secondary : secondary_models) {
            RegisterTwiceModelMerge report;
            std::vector<std::pair<uint32_t, uint32_t>> common;
            for (const auto& [secondary_id, image] : secondary.images)
                if (const auto found = primary_ids.find(image.name); found != primary_ids.end())
                    common.emplace_back(found->second, secondary_id);
            report.common_image_count = common.size();
            if (common.size() < 2) {
                result.models.push_back(report);
                continue;
            }
            const auto [p1_id, s1_id] = common.front();
            const auto [p2_id, s2_id] = common.back();
            report.primary_first_image_id = p1_id;
            report.primary_last_image_id = p2_id;
            report.secondary_first_image_id = s1_id;
            report.secondary_last_image_id = s2_id;
            const auto& p1 = primary.images.at(p1_id);
            const auto& p2 = primary.images.at(p2_id);
            const auto& s1 = secondary.images.at(s1_id);
            const auto& s2 = secondary.images.at(s2_id);
            const double d1 = distance(center(p1), center(p2));
            const double d2 = distance(center(s1), center(s2));
            if (d1 == 0 || d2 == 0)
                return bad("Register-twice anchor camera centers coincide");
            const double scale = d1 / d2;
            if (!std::isfinite(scale) || scale <= 0)
                return bad("Register-twice scale is not finite and positive");
            report.scale = scale;
            const auto transform_r = mul(tr(quaternion_rotation(p1.rotation)),
                                         quaternion_rotation(s1.rotation));
            auto scaled_anchor = center(s1);
            for (double& value : scaled_anchor)
                value *= scale;
            const auto rotated_anchor = mul(transform_r, scaled_anchor);
            const auto primary_anchor = center(p1);
            const V3 transform_t{primary_anchor[0] - rotated_anchor[0],
                                 primary_anchor[1] - rotated_anchor[1],
                                 primary_anchor[2] - rotated_anchor[2]};
            for (const auto& [secondary_id, image] : secondary.images) {
                (void)secondary_id;
                if (primary_ids.contains(image.name))
                    continue;
                if (!appended_names.insert(image.name).second)
                    return bad("Register-twice secondary models duplicate missing image: " + image.name);
                if (next_id >= static_cast<uint32_t>(std::numeric_limits<int32_t>::max() - 1))
                    return bad("Register-twice image ID range exhausted");
                const auto merged_c2w_r = mul(transform_r, tr(quaternion_rotation(image.rotation)));
                const auto merged_w2c_r = tr(merged_c2w_r);
                auto merged_center = center(image);
                for (double& value : merged_center)
                    value *= scale;
                merged_center = mul(transform_r, merged_center);
                for (size_t axis = 0; axis < 3; ++axis)
                    merged_center[axis] += transform_t[axis];
                const auto rotated_center = mul(merged_w2c_r, merged_center);
                ColmapImage merged = image;
                merged.id = ++next_id;
                merged.camera_id = p1.camera_id;
                merged.rotation = quaternion(merged_w2c_r);
                merged.translation = {-rotated_center[0], -rotated_center[1], -rotated_center[2]};
                for (auto& observation : merged.observations)
                    observation.point_id = -1;
                result.model.images.emplace(merged.id, std::move(merged));
                ++report.added_image_count;
                ++result.added_image_count;
            }
            result.models.push_back(report);
        }
        if (const auto valid = validate_model(result.model); !valid)
            return std::unexpected(valid.error());
        return result;
    }
} // namespace lfs::preprocess
