/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fast_lfs::rasterization {

    inline int extract_end_bit(uint n) {
        int leading_zeros = 0;
        if ((n & 0xffff0000u) == 0) {
            leading_zeros += 16;
            n <<= 16;
        }
        if ((n & 0xff000000u) == 0) {
            leading_zeros += 8;
            n <<= 8;
        }
        if ((n & 0xf0000000u) == 0) {
            leading_zeros += 4;
            n <<= 4;
        }
        if ((n & 0xc0000000u) == 0) {
            leading_zeros += 2;
            n <<= 2;
        }
        if ((n & 0x80000000u) == 0) {
            leading_zeros += 1;
        }
        return 32 - leading_zeros;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    inline std::size_t checked_size_add(
        std::size_t lhs,
        std::size_t rhs,
        const char* stage = "buffer_size.add") {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
            throw std::overflow_error(std::string("FastGS size overflow at stage=") + stage +
                                      " (add " + std::to_string(lhs) + " + " + std::to_string(rhs) + ")");
        }
        return lhs + rhs;
    }

    inline std::size_t checked_size_multiply(
        std::size_t lhs,
        std::size_t rhs,
        const char* stage = "buffer_size.multiply") {
        if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
            throw std::overflow_error(std::string("FastGS size overflow at stage=") + stage +
                                      " (multiply " + std::to_string(lhs) + " * " + std::to_string(rhs) + ")");
        }
        return lhs * rhs;
    }

    inline int checked_cub_count(std::size_t count, const char* stage) {
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error(std::string("FastGS CUB item count exceeds INT_MAX at stage=") + stage +
                                    " (count=" + std::to_string(count) + ")");
        }
        return static_cast<int>(count);
    }

    inline std::size_t checked_nonnegative_count(int count, const char* stage) {
        if (count < 0) {
            throw std::invalid_argument(std::string("FastGS negative item count at stage=") + stage +
                                        " (count=" + std::to_string(count) + ")");
        }
        return static_cast<std::size_t>(count);
    }

    template <typename Query>
    std::size_t checked_cub_workspace_query(const char* stage, Query&& query) {
        std::size_t workspace_bytes = 0;
        check_cuda(query(workspace_bytes), stage);
        return workspace_bytes;
    }

    inline std::uintptr_t checked_align_up(std::uintptr_t address, std::size_t alignment) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("FastGS buffer alignment must be a non-zero power of two");
        }
        const auto aligned = checked_size_add(
            static_cast<std::size_t>(address), alignment - 1, "buffer_layout.align");
        return static_cast<std::uintptr_t>(aligned & ~(alignment - 1));
    }

    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        const std::uintptr_t offset = checked_align_up(reinterpret_cast<std::uintptr_t>(blob), alignment);
        const std::size_t bytes = checked_size_multiply(count, sizeof(T), "buffer_layout.element_bytes");
        const std::uintptr_t end = static_cast<std::uintptr_t>(
            checked_size_add(static_cast<std::size_t>(offset), bytes, "buffer_layout.advance"));
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(end);
    }

    template <typename T, typename... Args>
    std::size_t required(std::size_t count, Args... args) {
        const int cub_count = checked_cub_count(count, "buffer_layout.required");
        char* cursor = nullptr;
        T::from_blob(cursor, cub_count, args...);
        return checked_size_add(reinterpret_cast<std::size_t>(cursor), 128, "buffer_layout.padding");
    }

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size = 0;
        char* cub_workspace = nullptr;
        cub::DoubleBuffer<uint> depth_keys;
        cub::DoubleBuffer<uint> primitive_indices;
        uint* n_touched_tiles;
        uint* offset;
        ushort4* screen_bounds;
        float2* mean2d;
        float4* conic_opacity;
        float3* color;
        float* depth;
        float4* ray_plane;
        float3* normal;
        float* grad_compensated_opacity;
        uint* n_visible_primitives = nullptr;
        unsigned long long* n_instances = nullptr;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives) {
            const std::size_t primitive_count = checked_nonnegative_count(n_primitives, "per_primitive.count");
            PerPrimitiveBuffers buffers{};
            uint* depth_keys_current = nullptr;
            obtain(blob, depth_keys_current, primitive_count, 128);
            uint* depth_keys_alternate = nullptr;
            obtain(blob, depth_keys_alternate, primitive_count, 128);
            buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
            uint* primitive_indices_current = nullptr;
            obtain(blob, primitive_indices_current, primitive_count, 128);
            uint* primitive_indices_alternate = nullptr;
            obtain(blob, primitive_indices_alternate, primitive_count, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            obtain(blob, buffers.n_touched_tiles, primitive_count, 128);
            obtain(blob, buffers.offset, primitive_count, 128);
            obtain(blob, buffers.screen_bounds, primitive_count, 128);
            obtain(blob, buffers.mean2d, primitive_count, 128);
            obtain(blob, buffers.conic_opacity, primitive_count, 128);
            obtain(blob, buffers.color, primitive_count, 128);
            obtain(blob, buffers.depth, primitive_count, 128);
            obtain(blob, buffers.ray_plane, primitive_count, 128);
            obtain(blob, buffers.normal, primitive_count, 128);
            obtain(blob, buffers.grad_compensated_opacity, primitive_count, 128);
            buffers.cub_workspace_size = checked_cub_workspace_query(
                "cub_query.per_primitive.exclusive_sum",
                [&](std::size_t& workspace_bytes) {
                    return cub::DeviceScan::ExclusiveSum(
                        nullptr, workspace_bytes,
                        buffers.offset, buffers.offset,
                        n_primitives);
                });
            const size_t sorting_workspace_size = checked_cub_workspace_query(
                "cub_query.per_primitive.depth_sort",
                [&](std::size_t& workspace_bytes) {
                    return cub::DeviceRadixSort::SortPairs(
                        nullptr, workspace_bytes,
                        buffers.depth_keys, buffers.primitive_indices,
                        n_primitives);
                });
            buffers.cub_workspace_size = std::max(buffers.cub_workspace_size, sorting_workspace_size);
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            obtain(blob, buffers.n_visible_primitives, 1, 128);
            obtain(blob, buffers.n_instances, 1, 128);
            return buffers;
        }
    };

    struct PerInstanceBuffers {
        size_t cub_workspace_size = 0;
        char* cub_workspace = nullptr;
        cub::DoubleBuffer<ushort> keys;
        cub::DoubleBuffer<uint> primitive_indices;

        static PerInstanceBuffers from_blob(char*& blob, int n_instances, int end_bit = 16) {
            const std::size_t instance_count = checked_nonnegative_count(n_instances, "per_instance.count");
            PerInstanceBuffers buffers{};
            ushort* keys_current = nullptr;
            obtain(blob, keys_current, instance_count, 128);
            ushort* keys_alternate = nullptr;
            obtain(blob, keys_alternate, instance_count, 128);
            buffers.keys = cub::DoubleBuffer<ushort>(keys_current, keys_alternate);
            uint* primitive_indices_current = nullptr;
            obtain(blob, primitive_indices_current, instance_count, 128);
            uint* primitive_indices_alternate = nullptr;
            obtain(blob, primitive_indices_alternate, instance_count, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            buffers.cub_workspace_size = checked_cub_workspace_query(
                "cub_query.per_instance.tile_sort",
                [&](std::size_t& workspace_bytes) {
                    return cub::DeviceRadixSort::SortPairs(
                        nullptr, workspace_bytes,
                        buffers.keys, buffers.primitive_indices,
                        n_instances, 0, end_bit);
                });
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerTileBuffers {
        size_t cub_workspace_size = 0;
        char* cub_workspace = nullptr;
        uint2* instance_ranges;
        uint* n_buckets;
        uint* bucket_offsets;
        uint* max_n_contributions;
        uint* n_contributions;

        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            const std::size_t tile_count = checked_nonnegative_count(n_tiles, "per_tile.count");
            PerTileBuffers buffers{};
            obtain(blob, buffers.instance_ranges, tile_count, 128);
            obtain(blob, buffers.n_buckets, tile_count, 128);
            obtain(blob, buffers.bucket_offsets, tile_count, 128);
            obtain(blob, buffers.max_n_contributions, tile_count, 128);
            const std::size_t contribution_count = checked_size_multiply(
                tile_count, static_cast<std::size_t>(config::block_size_blend),
                "per_tile.contribution_count");
            obtain(blob, buffers.n_contributions, contribution_count, 128);
            buffers.cub_workspace_size = checked_cub_workspace_query(
                "cub_query.per_tile.bucket_scan",
                [&](std::size_t& workspace_bytes) {
                    return cub::DeviceScan::InclusiveSum(
                        nullptr, workspace_bytes,
                        buffers.n_buckets, buffers.bucket_offsets,
                        n_tiles);
                });
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerBucketBuffers {
        uint* tile_index;
        uint* checkpoint_uint8; // packed RGBA as 4x uint8 (colors [0,4], transmittance [0,1])
        uint* checkpoint_normal_uint8; // packed normal xyz as 3x uint8 + spare (for GGGS normal backward)

        static PerBucketBuffers from_blob(char*& blob, int n_buckets, bool need_normal_checkpoint = false) {
            const std::size_t bucket_count = checked_nonnegative_count(n_buckets, "per_bucket.count");
            PerBucketBuffers buffers{};
            obtain(blob, buffers.tile_index, bucket_count, 128);
            const std::size_t checkpoint_count = checked_size_multiply(
                bucket_count, static_cast<std::size_t>(config::block_size_blend),
                "per_bucket.checkpoint_count");
            obtain(blob, buffers.checkpoint_uint8, checkpoint_count, 128);
            if (need_normal_checkpoint) {
                obtain(blob, buffers.checkpoint_normal_uint8, checkpoint_count, 128);
            } else {
                buffers.checkpoint_normal_uint8 = nullptr;
            }
            return buffers;
        }
    };

} // namespace fast_lfs::rasterization
