/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <list>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::training::pseudo_view {

    inline constexpr size_t MIB = size_t{1024} * size_t{1024};
    inline constexpr size_t SOURCE_CACHE_MAX_BYTES = size_t{512} * MIB;
    inline constexpr size_t RGB_JOB_HEADROOM_BYTES = size_t{512} * MIB;

    [[nodiscard]] constexpr size_t source_cache_budget_bytes(const size_t free_vram_bytes) {
        return std::min(SOURCE_CACHE_MAX_BYTES, free_vram_bytes / size_t{4});
    }

    [[nodiscard]] constexpr size_t saturating_add(const size_t lhs, const size_t rhs) {
        return rhs > std::numeric_limits<size_t>::max() - lhs
                   ? std::numeric_limits<size_t>::max()
                   : lhs + rhs;
    }

    [[nodiscard]] constexpr size_t saturating_multiply(const size_t lhs, const size_t rhs) {
        return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs
                   ? std::numeric_limits<size_t>::max()
                   : lhs * rhs;
    }

    [[nodiscard]] constexpr size_t round_up_allocation(
        const size_t bytes,
        const size_t granularity) {
        if (bytes == 0 || granularity == 0) {
            return bytes;
        }
        const size_t padding = granularity - 1;
        if (bytes > std::numeric_limits<size_t>::max() - padding) {
            return std::numeric_limits<size_t>::max();
        }
        return ((bytes + padding) / granularity) * granularity;
    }

    // Mirrors SizeBucketedPool::get_bucket_size without depending on a CUDA or
    // internal allocator header. Returning the 256 KiB minimum for small live
    // tensors is conservative when the slab allocator serves them.
    [[nodiscard]] constexpr size_t pooled_allocation_bytes(const size_t bytes) {
        constexpr size_t KIB = size_t{1024};
        constexpr size_t GIB = size_t{1024} * MIB;
        if (bytes == 0) {
            return 0;
        }
        if (bytes <= size_t{256} * KIB) {
            return size_t{256} * KIB;
        }
        if (bytes <= MIB) {
            return round_up_allocation(bytes, size_t{256} * KIB);
        }
        if (bytes <= size_t{16} * MIB) {
            return round_up_allocation(bytes, MIB);
        }
        if (bytes <= size_t{256} * MIB) {
            return round_up_allocation(bytes, size_t{16} * MIB);
        }
        if (bytes <= GIB) {
            return round_up_allocation(bytes, size_t{64} * MIB);
        }
        if (bytes <= size_t{8} * GIB) {
            return round_up_allocation(bytes, size_t{256} * MIB);
        }
        return round_up_allocation(bytes, GIB);
    }

    // Conservative live CUDA allocation estimate for one final pseudo-view
    // render/reprojection job. The source RGB is intentionally excluded because
    // it is loaded once per source group before free memory is sampled.
    [[nodiscard]] constexpr size_t estimate_rgb_job_bytes(
        const int width,
        const int height,
        const int mesh_vertex_count) {
        if (width <= 0 || height <= 0) {
            return std::numeric_limits<size_t>::max();
        }
        const size_t pixels = saturating_multiply(
            static_cast<size_t>(width),
            static_cast<size_t>(height));
        const auto pixel_buffer = [pixels](const size_t bytes_per_pixel) constexpr {
            return pooled_allocation_bytes(saturating_multiply(pixels, bytes_per_pixel));
        };
        size_t bytes = 0;
        // Each tensor is a separate allocation and therefore rounds to its own
        // pool bucket. This matters for e.g. a 18.75 MiB tensor occupying a
        // 32 MiB bucket.
        for (const size_t bytes_per_pixel : {size_t{4},  // depth
                                             size_t{8},  // depth keys
                                             size_t{4},  // triangle id
                                             size_t{4},  // front mask
                                             size_t{12}, // RGB output
                                             size_t{4},  // valid mask
                                             size_t{4},  // mesh mask
                                             size_t{8}}) // intermediate slack
        {
            bytes = saturating_add(bytes, pixel_buffer(bytes_per_pixel));
        }
        const size_t vertex_bytes = mesh_vertex_count > 0
                                        ? pooled_allocation_bytes(saturating_multiply(
                                              static_cast<size_t>(mesh_vertex_count),
                                              size_t{12}))
                                        : 0;
        return saturating_add(bytes, vertex_bytes);
    }

    [[nodiscard]] constexpr int effective_rgb_parallel_jobs(
        const int requested_jobs,
        const size_t candidate_count,
        const size_t free_vram_bytes,
        const size_t single_job_bytes,
        const size_t headroom_bytes = RGB_JOB_HEADROOM_BYTES) {
        if (candidate_count == 0) {
            return 0;
        }

        const int requested = std::clamp(requested_jobs, 1, 4);
        const size_t usable = free_vram_bytes > headroom_bytes
                                  ? free_vram_bytes - headroom_bytes
                                  : 0;
        const size_t by_memory = single_job_bytes > 0 &&
                                         single_job_bytes != std::numeric_limits<size_t>::max()
                                     ? usable / single_job_bytes
                                     : 0;
        // One job is the serial fallback. It may still fail normally if even a
        // single job cannot fit, but it prevents the concurrency policy itself
        // from silently dropping otherwise valid work.
        const size_t memory_cap = std::max<size_t>(1, by_memory);
        return static_cast<int>(std::min({
            static_cast<size_t>(requested),
            candidate_count,
            memory_cap}));
    }

    struct SourceGroup {
        int source_index = -1;
        std::vector<size_t> candidate_indices;
    };

    // Stable first-source ordering keeps pseudo ids deterministic while allowing
    // each source RGB tensor to have one bounded lifetime.
    [[nodiscard]] inline std::vector<SourceGroup> group_candidate_indices_by_source(
        const std::vector<int>& source_indices) {
        std::vector<SourceGroup> groups;
        std::unordered_map<int, size_t> group_by_source;
        groups.reserve(source_indices.size());
        for (size_t candidate_index = 0; candidate_index < source_indices.size(); ++candidate_index) {
            const int source_index = source_indices[candidate_index];
            const auto [it, inserted] = group_by_source.try_emplace(source_index, groups.size());
            if (inserted) {
                groups.push_back(SourceGroup{.source_index = source_index});
            }
            groups[it->second].candidate_indices.push_back(candidate_index);
        }
        return groups;
    }

    [[nodiscard]] inline nlohmann::json make_minimal_manifest_root(const std::string& status) {
        return {
            {"version", 2},
            {"status", status},
            {"views", nlohmann::json::array()}};
    }

    [[nodiscard]] inline nlohmann::json make_minimal_view_entry(
        const int pseudo_id,
        const uint32_t base_camera_uid,
        const std::string& source_image_name,
        const int source_camera_id,
        const int source_uid,
        const std::string& rgb_path,
        const std::string& supervision_mask_path,
        const std::string& pose_path) {
        return {
            {"status", "ok"},
            {"pseudo_id", pseudo_id},
            {"base_camera_uid", base_camera_uid},
            {"source_image_name", source_image_name},
            {"source_camera_id", source_camera_id},
            // Diagnostic only. Stable lookup must use source_image_name + source_camera_id.
            {"source_uid", source_uid},
            {"rgb_path", rgb_path},
            {"supervision_mask_path", supervision_mask_path},
            {"pose_path", pose_path}};
    }

    [[nodiscard]] inline nlohmann::json make_minimal_pose(
        const int pseudo_id,
        nlohmann::json rotation,
        nlohmann::json translation,
        nlohmann::json intrinsics,
        const int image_width,
        const int image_height) {
        return {
            {"pseudo_id", pseudo_id},
            {"R", std::move(rotation)},
            {"T", std::move(translation)},
            {"intrinsics", std::move(intrinsics)},
            {"image_width", image_width},
            {"image_height", image_height}};
    }

    enum class QualityRejection {
        None,
        NoMeshHit,
        RawCoverage,
        EmptySupervision,
        PostMorphCoverage,
        MinValidPixels,
    };

    struct QualityDecision {
        QualityRejection rejection = QualityRejection::None;
        float raw_ratio = 0.0f;
        float post_morph_ratio = 0.0f;

        [[nodiscard]] constexpr bool accepted() const {
            return rejection == QualityRejection::None;
        }
    };

    [[nodiscard]] constexpr QualityDecision evaluate_supervision_quality(
        const int mesh_valid_pixels,
        const int raw_valid_pixels,
        const int supervision_valid_pixels,
        const float min_valid_ratio,
        const int min_valid_pixels) {
        QualityDecision decision;
        if (mesh_valid_pixels <= 0) {
            decision.rejection = QualityRejection::NoMeshHit;
            return decision;
        }

        decision.raw_ratio = static_cast<float>(std::max(0, raw_valid_pixels)) /
                             static_cast<float>(mesh_valid_pixels);
        decision.post_morph_ratio = static_cast<float>(std::max(0, supervision_valid_pixels)) /
                                    static_cast<float>(mesh_valid_pixels);
        if (decision.raw_ratio < min_valid_ratio) {
            decision.rejection = QualityRejection::RawCoverage;
        } else if (supervision_valid_pixels <= 0) {
            decision.rejection = QualityRejection::EmptySupervision;
        } else if (decision.post_morph_ratio < min_valid_ratio) {
            decision.rejection = QualityRejection::PostMorphCoverage;
        } else if (min_valid_pixels > 0 && supervision_valid_pixels < min_valid_pixels) {
            decision.rejection = QualityRejection::MinValidPixels;
        }
        return decision;
    }

    template <typename Key, typename Value>
    class ByteBudgetLru {
    public:
        ByteBudgetLru(const size_t budget_bytes, const size_t max_entries)
            : budget_bytes_(budget_bytes), max_entries_(max_entries) {}

        ByteBudgetLru(const ByteBudgetLru&) = delete;
        ByteBudgetLru& operator=(const ByteBudgetLru&) = delete;
        ByteBudgetLru(ByteBudgetLru&&) = default;
        ByteBudgetLru& operator=(ByteBudgetLru&&) = default;

        [[nodiscard]] Value* find(const Key& key) {
            const auto it = entries_.find(key);
            if (it == entries_.end()) {
                return nullptr;
            }
            touch(it);
            return &it->second.value;
        }

        [[nodiscard]] const Value* peek(const Key& key) const {
            const auto it = entries_.find(key);
            return it == entries_.end() ? nullptr : &it->second.value;
        }

        // Returns nullptr when the item exceeds the byte budget (or persistence
        // is disabled). Callers can then use their owning value transiently.
        [[nodiscard]] Value* insert(const Key& key, Value&& value, const size_t bytes) {
            if (max_entries_ == 0 || bytes > budget_bytes_) {
                return nullptr;
            }
            erase(key);

            while (!order_.empty() &&
                   (entries_.size() >= max_entries_ || bytes > budget_bytes_ - live_bytes_)) {
                evict_oldest();
            }
            if (entries_.size() >= max_entries_ || bytes > budget_bytes_ - live_bytes_) {
                return nullptr;
            }

            order_.push_front(key);
            auto [it, inserted] = entries_.emplace(
                key,
                Entry{std::move(value), bytes, order_.begin()});
            if (!inserted) {
                order_.pop_front();
                return nullptr;
            }
            live_bytes_ += bytes;
            peak_bytes_ = std::max(peak_bytes_, live_bytes_);
            return &it->second.value;
        }

        void erase(const Key& key) {
            const auto it = entries_.find(key);
            if (it == entries_.end()) {
                return;
            }
            live_bytes_ -= it->second.bytes;
            order_.erase(it->second.order_it);
            entries_.erase(it);
        }

        void clear() {
            entries_.clear();
            order_.clear();
            live_bytes_ = 0;
        }

        [[nodiscard]] size_t size() const { return entries_.size(); }
        [[nodiscard]] size_t live_bytes() const { return live_bytes_; }
        [[nodiscard]] size_t peak_bytes() const { return peak_bytes_; }
        [[nodiscard]] size_t eviction_count() const { return eviction_count_; }
        [[nodiscard]] size_t budget_bytes() const { return budget_bytes_; }

    private:
        struct Entry {
            Value value;
            size_t bytes = 0;
            typename std::list<Key>::iterator order_it;
        };

        using MapIterator = typename std::unordered_map<Key, Entry>::iterator;

        void touch(const MapIterator it) {
            order_.splice(order_.begin(), order_, it->second.order_it);
            it->second.order_it = order_.begin();
        }

        void evict_oldest() {
            const Key& key = order_.back();
            const auto it = entries_.find(key);
            if (it != entries_.end()) {
                live_bytes_ -= it->second.bytes;
                entries_.erase(it);
                ++eviction_count_;
            }
            order_.pop_back();
        }

        size_t budget_bytes_ = 0;
        size_t max_entries_ = 0;
        size_t live_bytes_ = 0;
        size_t peak_bytes_ = 0;
        size_t eviction_count_ = 0;
        std::list<Key> order_;
        std::unordered_map<Key, Entry> entries_;
    };

} // namespace lfs::training::pseudo_view
