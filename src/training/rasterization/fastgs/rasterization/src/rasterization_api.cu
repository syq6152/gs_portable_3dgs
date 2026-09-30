/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "core/cuda/memory_arena.hpp"
#include "cuda_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fast_lfs::rasterization {

    namespace {

        class ArenaFrameGuard {
        public:
            explicit ArenaFrameGuard(lfs::core::RasterizerMemoryArena& arena)
                : arena_(&arena), frame_id_(arena.begin_frame()) {}

            ArenaFrameGuard(const ArenaFrameGuard&) = delete;
            ArenaFrameGuard& operator=(const ArenaFrameGuard&) = delete;

            ~ArenaFrameGuard() {
                if (active_) {
                    arena_->end_frame(frame_id_);
                }
            }

            [[nodiscard]] uint64_t frame_id() const noexcept { return frame_id_; }
            void release() noexcept { active_ = false; }

        private:
            lfs::core::RasterizerMemoryArena* arena_ = nullptr;
            uint64_t frame_id_ = 0;
            bool active_ = true;
        };

        class ExistingArenaFrameGuard {
        public:
            ExistingArenaFrameGuard(
                lfs::core::RasterizerMemoryArena& arena,
                uint64_t frame_id,
                bool active) noexcept
                : arena_(&arena), frame_id_(frame_id), active_(active) {}

            ExistingArenaFrameGuard(const ExistingArenaFrameGuard&) = delete;
            ExistingArenaFrameGuard& operator=(const ExistingArenaFrameGuard&) = delete;

            ~ExistingArenaFrameGuard() {
                if (active_) {
                    arena_->end_frame(frame_id_);
                }
            }

            void release() noexcept { active_ = false; }

        private:
            lfs::core::RasterizerMemoryArena* arena_ = nullptr;
            uint64_t frame_id_ = 0;
            bool active_ = false;
        };

        class ScopedCudaBuffer {
        public:
            ScopedCudaBuffer() = default;
            ScopedCudaBuffer(const ScopedCudaBuffer&) = delete;
            ScopedCudaBuffer& operator=(const ScopedCudaBuffer&) = delete;

            ~ScopedCudaBuffer() {
                if (ptr_) {
                    cudaFree(ptr_);
                }
            }

            char* allocate(size_t bytes, const char* stage) {
                if (ptr_) {
                    throw std::logic_error("FastGS API CUDA buffer allocated twice");
                }
                check_cuda(cudaMalloc(&ptr_, bytes), stage);
                return static_cast<char*>(ptr_);
            }

            void free_checked(const char* stage) {
                void* const ptr = std::exchange(ptr_, nullptr);
                if (ptr) {
                    check_cuda(cudaFree(ptr), stage);
                }
            }

        private:
            void* ptr_ = nullptr;
        };

        std::string format_cuda_status(cudaError_t error) {
            if (error == cudaSuccess) {
                return "none";
            }
            const char* const name = cudaGetErrorName(error);
            const char* const description = cudaGetErrorString(error);
            return (name ? std::string(name) : std::string("unknown")) +
                   " (" + (description ? std::string(description) : std::string("unknown")) + ")";
        }

        std::string format_forward_error(
            const std::exception& error,
            const char* api_stage,
            const ForwardRuntimeInfo& runtime_info,
            int n_primitives,
            uint64_t n_tiles,
            size_t per_primitive_bytes,
            size_t per_tile_bytes,
            size_t per_instance_bytes,
            size_t per_bucket_bytes,
            size_t grad_mean2d_bytes,
            size_t grad_conic_bytes,
            size_t normal_accum_bytes) {
            const auto* const cuda_error = dynamic_cast<const ::fast_lfs::CudaRuntimeError*>(&error);
            const char* stage = api_stage;
            if (cuda_error) {
                stage = cuda_error->stage().c_str();
            } else if (runtime_info.stage && std::string_view(runtime_info.stage) != "not_started") {
                stage = runtime_info.stage;
            }

            std::ostringstream oss;
            oss << "FastGS forward failed: stage=" << (stage ? stage : "unknown")
                << "; cuda=" << format_cuda_status(cuda_error ? cuda_error->error() : cudaSuccess)
                << "; detail=" << error.what()
                << "; counts={primitives=" << n_primitives
                << ", visible=" << runtime_info.n_visible_primitives
                << ", instances=" << runtime_info.n_instances
                << ", tiles=" << n_tiles
                << ", buckets=" << runtime_info.n_buckets << "}"
                << "; buffer_bytes={primitive=" << per_primitive_bytes
                << ", tile=" << per_tile_bytes
                << ", instance=" << per_instance_bytes
                << ", bucket=" << per_bucket_bytes
                << ", grad_mean2d=" << grad_mean2d_bytes
                << ", grad_conic=" << grad_conic_bytes
                << ", normal_accum=" << normal_accum_bytes << "}";
            return oss.str();
        }

        std::string format_backward_error(
            const std::exception& error,
            const char* api_stage,
            const ForwardContext& forward_ctx) {
            const auto* const cuda_error = dynamic_cast<const ::fast_lfs::CudaRuntimeError*>(&error);
            const char* const stage = cuda_error ? cuda_error->stage().c_str() : api_stage;
            std::ostringstream oss;
            oss << "FastGS backward failed: stage=" << (stage ? stage : "unknown")
                << "; cuda=" << format_cuda_status(cuda_error ? cuda_error->error() : cudaSuccess)
                << "; detail=" << error.what()
                << "; counts={visible=" << forward_ctx.n_visible_primitives
                << ", instances=" << forward_ctx.n_instances
                << ", buckets=" << forward_ctx.n_buckets << "}"
                << "; buffer_bytes={primitive=" << forward_ctx.per_primitive_buffers_size
                << ", tile=" << forward_ctx.per_tile_buffers_size
                << ", instance=" << forward_ctx.per_instance_buffers_size
                << ", bucket=" << forward_ctx.per_bucket_buffers_size << "}";
            return oss.str();
        }

    } // namespace

    ForwardContext forward_raw(
        const float* means_ptr,
        const float* scales_raw_ptr,
        const float* rotations_raw_ptr,
        const float* opacities_raw_ptr,
        const float* sh_coefficients_0_ptr,
        const float* sh_coefficients_rest_ptr,
        const float* w2c_ptr,
        const float* cam_position_ptr,
        float* image_ptr,
        float* alpha_ptr,
        float* depth_ptr,
        float* normal_ptr,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter,
        bool require_depth,
        bool require_normal_backward,
        const float* mesh_depth_cull_ptr,
        int mesh_depth_cull_width,
        int mesh_depth_cull_height,
        int mesh_depth_cull_x_offset,
        int mesh_depth_cull_y_offset,
        const ObservationBlurSettings& observation_blur) {

        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        std::optional<ArenaFrameGuard> frame_guard;
        uint64_t frame_id = 0;
        const char* api_stage = "forward.entry";
        ForwardRuntimeInfo runtime_info{};
        uint64_t n_tiles_u64 = 0;
        int n_tiles = 0;
        size_t per_primitive_size = 0;
        size_t per_tile_size = 0;
        size_t per_instance_size = 0;
        size_t per_bucket_size = 0;
        size_t grad_mean2d_size = 0;
        size_t grad_conic_size = 0;
        size_t normal_accum_length_size = 0;
        char* per_primitive_buffers_blob = nullptr;
        char* per_tile_buffers_blob = nullptr;
        char* per_instance_buffers_blob = nullptr;
        char* per_bucket_buffers_blob = nullptr;
        char* grad_mean2d_helper = nullptr;
        char* grad_conic_helper = nullptr;
        char* normal_accum_length_map_buf = nullptr;

        try {
            api_stage = "forward.entry.pending_cuda_error";
            check_no_pending_cuda_error("forward_raw.entry.pending_cuda_error");

            // begin_frame() also synchronizes before reusing arena storage, but
            // its legacy API does not report that status. Synchronize here,
            // inside the owning error boundary, so an asynchronous error from
            // earlier work cannot be swallowed and later misattributed to CUB.
            api_stage = "forward.entry.synchronize";
            check_cuda(cudaDeviceSynchronize(), "forward_raw.entry.synchronize");

            api_stage = "forward.dimension_validation";
            if (n_primitives <= 0 || width <= 0 || height <= 0 ||
                active_sh_bases <= 0 || total_bases_sh_rest < 0) {
                throw std::invalid_argument("Invalid dimensions or SH basis counts in forward_raw");
            }

            const dim3 grid(div_round_up(width, config::tile_width),
                            div_round_up(height, config::tile_height), 1);
            n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
            if (n_tiles_u64 == 0 ||
                n_tiles_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                throw std::length_error("FastGS tile count is outside the supported CUB int range");
            }
            n_tiles = static_cast<int>(n_tiles_u64);

            api_stage = "forward.pointer_validation";
            CHECK_CUDA_PTR(means_ptr, "means_ptr");
            CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
            CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
            CHECK_CUDA_PTR(opacities_raw_ptr, "opacities_raw_ptr");
            CHECK_CUDA_PTR(sh_coefficients_0_ptr, "sh_coefficients_0_ptr");
            if (total_bases_sh_rest > 0) {
                CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
            }
            CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
            CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");
            CHECK_CUDA_PTR(image_ptr, "image_ptr");
            CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");
            if (require_depth) {
                CHECK_CUDA_PTR(depth_ptr, "depth_ptr");
                CHECK_CUDA_PTR(normal_ptr, "normal_ptr");
            }
            if (mesh_depth_cull_ptr != nullptr) {
                CHECK_CUDA_PTR(mesh_depth_cull_ptr, "mesh_depth_cull_ptr");
                if (mesh_depth_cull_width <= 0 || mesh_depth_cull_height <= 0) {
                    throw std::invalid_argument("Invalid mesh depth cull dimensions in forward_raw");
                }
            }
            if ((observation_blur.motion_enabled || observation_blur.defocus_enabled) &&
                observation_blur.parameters_ptr == nullptr) {
                throw std::invalid_argument(
                    "FastGS observation blur branch enabled without a parameter row");
            }
            if (observation_blur.enabled()) {
                CHECK_CUDA_PTR(observation_blur.parameters_ptr, "observation_blur.parameters_ptr");
                CHECK_CUDA_PTR_OPTIONAL(observation_blur.gradients_ptr, "observation_blur.gradients_ptr");
                if (!std::isfinite(observation_blur.max_defocus_radius_sq) ||
                    observation_blur.max_defocus_radius_sq < 0.0f ||
                    !std::isfinite(observation_blur.max_observation_radius_sq) ||
                    observation_blur.max_observation_radius_sq <= 0.0f) {
                    throw std::invalid_argument(
                        "FastGS observation blur radius caps must be finite; the combined cap must be positive");
                }
                if (mip_filter) {
                    throw std::invalid_argument(
                        "FastGS observation blur does not support mip filtering until its covariance VJP is available");
                }
                if (require_normal_backward || mesh_depth_cull_ptr != nullptr) {
                    throw std::invalid_argument(
                        "FastGS observation blur currently supports RGB supervision only");
                }
            }

            api_stage = "arena.begin_frame";
            frame_guard.emplace(arena);
            frame_id = frame_guard->frame_id();
            auto arena_allocator = arena.get_allocator(frame_id);

            api_stage = "buffer_size.initial";
            per_primitive_size = required<PerPrimitiveBuffers>(n_primitives);
            per_tile_size = required<PerTileBuffers>(n_tiles);

            api_stage = "arena_allocate.initial";
            per_primitive_buffers_blob = arena_allocator(per_primitive_size);
            per_tile_buffers_blob = arena_allocator(per_tile_size);
            if (!per_primitive_buffers_blob || !per_tile_buffers_blob) {
                throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate initial FastGS buffers from arena");
            }

            api_stage = "buffer_size.backward_helpers";
            grad_mean2d_size = checked_size_multiply(
                static_cast<size_t>(n_primitives), 2 * sizeof(float), "grad_mean2d_helper");
            grad_conic_size = checked_size_multiply(
                static_cast<size_t>(n_primitives), 3 * sizeof(float), "grad_conic_helper");
            api_stage = "arena_allocate.backward_helpers";
            grad_mean2d_helper = arena_allocator(grad_mean2d_size);
            grad_conic_helper = arena_allocator(grad_conic_size);
            if (!grad_mean2d_helper || !grad_conic_helper) {
                throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS backward helper buffers");
            }

            if (require_normal_backward) {
                api_stage = "buffer_size.normal_accum";
                const size_t n_pixels = checked_size_multiply(
                    static_cast<size_t>(width), static_cast<size_t>(height), "normal_accum.pixel_count");
                normal_accum_length_size = checked_size_multiply(
                    n_pixels, sizeof(float), "normal_accum.bytes");
                api_stage = "arena_allocate.normal_accum";
                normal_accum_length_map_buf = arena_allocator(normal_accum_length_size);
                if (!normal_accum_length_map_buf) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS normal accumulation buffer");
                }
            }

            std::function<char*(size_t)> per_primitive_buffers_func =
                [&](size_t size) -> char* {
                if (size > per_primitive_size) {
                    throw std::runtime_error("FastGS per-primitive layout exceeded its sized arena allocation");
                }
                return per_primitive_buffers_blob;
            };
            std::function<char*(size_t)> per_tile_buffers_func =
                [&](size_t size) -> char* {
                if (size > per_tile_size) {
                    throw std::runtime_error("FastGS per-tile layout exceeded its sized arena allocation");
                }
                return per_tile_buffers_blob;
            };
            std::function<char*(size_t)> per_instance_buffers_func =
                [&](size_t size) -> char* {
                per_instance_size = size;
                per_instance_buffers_blob = arena_allocator(size);
                if (!per_instance_buffers_blob) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS instance buffers");
                }
                return per_instance_buffers_blob;
            };
            std::function<char*(size_t)> per_bucket_buffers_func =
                [&](size_t size) -> char* {
                per_bucket_size = size;
                per_bucket_buffers_blob = arena_allocator(size);
                if (!per_bucket_buffers_blob) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS bucket buffers");
                }
                return per_bucket_buffers_blob;
            };

            api_stage = "forward.dispatch";
            auto [n_visible_primitives, n_instances, n_buckets,
                  primitive_primitive_indices_selector,
                  instance_primitive_indices_selector] = forward(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                per_bucket_buffers_func,
                reinterpret_cast<const float3*>(means_ptr),
                reinterpret_cast<const float3*>(scales_raw_ptr),
                reinterpret_cast<const float4*>(rotations_raw_ptr),
                opacities_raw_ptr,
                reinterpret_cast<const float3*>(sh_coefficients_0_ptr),
                reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
                reinterpret_cast<const float4*>(w2c_ptr),
                reinterpret_cast<const float3*>(cam_position_ptr),
                image_ptr,
                alpha_ptr,
                depth_ptr,
                normal_ptr,
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane,
                require_depth,
                mip_filter,
                require_normal_backward,
                reinterpret_cast<float*>(normal_accum_length_map_buf),
                mesh_depth_cull_ptr,
                mesh_depth_cull_width,
                mesh_depth_cull_height,
                mesh_depth_cull_x_offset,
                mesh_depth_cull_y_offset,
                &runtime_info,
                observation_blur);

            if (n_instances > 0 && !per_instance_buffers_blob) {
                throw std::runtime_error("FastGS instance buffers are missing after a non-empty forward pass");
            }
            if (n_buckets > 0 && !per_bucket_buffers_blob) {
                throw std::runtime_error("FastGS bucket buffers are missing after a non-empty forward pass");
            }

            ForwardContext ctx{};
            ctx.per_primitive_buffers = per_primitive_buffers_blob;
            ctx.per_tile_buffers = per_tile_buffers_blob;
            ctx.per_instance_buffers = per_instance_buffers_blob;
            ctx.per_bucket_buffers = per_bucket_buffers_blob;
            ctx.per_primitive_buffers_size = per_primitive_size;
            ctx.per_tile_buffers_size = per_tile_size;
            ctx.per_instance_buffers_size = per_instance_size;
            ctx.per_bucket_buffers_size = per_bucket_size;
            ctx.n_visible_primitives = n_visible_primitives;
            ctx.n_instances = n_instances;
            ctx.n_buckets = n_buckets;
            ctx.primitive_primitive_indices_selector = primitive_primitive_indices_selector;
            ctx.instance_primitive_indices_selector = instance_primitive_indices_selector;
            {
                char* per_primitive_view = per_primitive_buffers_blob;
                PerPrimitiveBuffers per_primitive_view_buffers =
                    PerPrimitiveBuffers::from_blob(per_primitive_view, n_primitives);
                ctx.visible_primitive_indices =
                    per_primitive_view_buffers.primitive_indices.d_buffers[primitive_primitive_indices_selector];
            }
            ctx.frame_id = frame_id;
            ctx.grad_mean2d_helper = grad_mean2d_helper;
            ctx.grad_conic_helper = grad_conic_helper;
            ctx.observation_blur = observation_blur;
            ctx.normal_accum_length_map = normal_accum_length_map_buf;
            ctx.has_normal_backward_buffers = require_normal_backward;
            ctx.success = true;
            ctx.error_message.clear();

            frame_guard->release();
            return ctx;

        } catch (const std::exception& e) {
            ForwardContext error_ctx{};
            error_ctx.success = false;
            error_ctx.error_message = format_forward_error(
                e,
                api_stage,
                runtime_info,
                n_primitives,
                n_tiles_u64,
                per_primitive_size,
                per_tile_size,
                per_instance_size,
                per_bucket_size,
                grad_mean2d_size,
                grad_conic_size,
                normal_accum_length_size);
            error_ctx.frame_id = frame_id;
            return error_ctx;
        }
    }

    BackwardOutputs backward_raw(
        float* densification_info_ptr,
        const float* densification_error_map_ptr,
        const float* grad_image_ptr,
        const float* grad_alpha_ptr,
        const float* image_ptr,
        const float* alpha_ptr,
        const float* means_ptr,
        const float* scales_raw_ptr,
        const float* rotations_raw_ptr,
        const float* raw_opacities_ptr,
        const float* sh_coefficients_rest_ptr,
        const float* w2c_ptr,
        const float* cam_position_ptr,
        const ForwardContext& forward_ctx,
        float* grad_means_ptr,
        float* grad_scales_raw_ptr,
        float* grad_rotations_raw_ptr,
        float* grad_opacities_raw_ptr,
        float* grad_sh_coefficients_0_ptr,
        float* grad_sh_coefficients_rest_ptr,
        float* grad_w2c_ptr,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter,
        const float* grad_render_normal_ptr,
        const float* render_normal_ptr,
        const float* grad_depth_ptr,
        const float* depth_map_ptr,
        const ObservationBlurSettings& observation_blur) {

        BackwardOutputs outputs{};
        outputs.success = false;
        outputs.error_message.clear();
        const char* api_stage = "backward.entry";
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        ExistingArenaFrameGuard frame_guard(arena, forward_ctx.frame_id, forward_ctx.success);

        try {
            check_no_pending_cuda_error("backward_raw.entry.pending_cuda_error");
            if (!forward_ctx.success) {
                throw std::invalid_argument("Cannot run FastGS backward with a failed forward context");
            }

            // A successful forward owns the authoritative settings.  Callers
            // may omit the trailing argument on backward; in that case reuse
            // the row captured in ForwardContext.  If a non-empty override is
            // supplied, it must refer to the same enabled branches so that
            // the VJP matches the forward covariance path.
            const bool observation_blur_override_supplied =
                observation_blur.parameters_ptr != nullptr ||
                observation_blur.gradients_ptr != nullptr ||
                observation_blur.motion_enabled ||
                observation_blur.defocus_enabled ||
                observation_blur.max_defocus_radius_sq != 0.0f ||
                observation_blur.max_observation_radius_sq != 0.0f;
            if ((observation_blur.motion_enabled || observation_blur.defocus_enabled) &&
                observation_blur.parameters_ptr == nullptr) {
                throw std::invalid_argument(
                    "FastGS backward observation blur branch enabled without a parameter row");
            }
            const ObservationBlurSettings effective_observation_blur =
                observation_blur_override_supplied ? observation_blur : forward_ctx.observation_blur;
            if (observation_blur_override_supplied) {
                if (!forward_ctx.observation_blur.enabled() ||
                    observation_blur.parameters_ptr !=
                        forward_ctx.observation_blur.parameters_ptr ||
                    observation_blur.motion_enabled !=
                        forward_ctx.observation_blur.motion_enabled ||
                    observation_blur.defocus_enabled !=
                        forward_ctx.observation_blur.defocus_enabled ||
                    observation_blur.max_defocus_radius_sq !=
                        forward_ctx.observation_blur.max_defocus_radius_sq ||
                    observation_blur.max_observation_radius_sq !=
                        forward_ctx.observation_blur.max_observation_radius_sq) {
                    throw std::invalid_argument(
                        "FastGS backward observation blur settings do not match forward");
                }
            }
            if (effective_observation_blur.enabled()) {
                CHECK_CUDA_PTR(effective_observation_blur.parameters_ptr,
                               "observation_blur.parameters_ptr");
                CHECK_CUDA_PTR_OPTIONAL(effective_observation_blur.gradients_ptr,
                                        "observation_blur.gradients_ptr");
                if (!std::isfinite(effective_observation_blur.max_defocus_radius_sq) ||
                    effective_observation_blur.max_defocus_radius_sq < 0.0f ||
                    !std::isfinite(effective_observation_blur.max_observation_radius_sq) ||
                    effective_observation_blur.max_observation_radius_sq <= 0.0f) {
                    throw std::invalid_argument(
                        "FastGS backward observation blur radius caps must be finite; the combined cap must be positive");
                }
                if (mip_filter) {
                    throw std::invalid_argument(
                        "FastGS observation blur does not support mip filtering until its covariance VJP is available");
                }
                if (grad_render_normal_ptr != nullptr || render_normal_ptr != nullptr ||
                    grad_depth_ptr != nullptr || depth_map_ptr != nullptr) {
                    throw std::invalid_argument(
                        "FastGS observation blur currently supports RGB supervision only");
                }
            } else if (effective_observation_blur.motion_enabled ||
                       effective_observation_blur.defocus_enabled) {
                throw std::invalid_argument(
                    "FastGS backward observation blur branch enabled without a parameter row");
            }

            api_stage = "backward.dimension_validation";
            if (n_primitives <= 0 || width <= 0 || height <= 0 ||
                active_sh_bases <= 0 || total_bases_sh_rest < 0 ||
                forward_ctx.n_visible_primitives < 0 ||
                forward_ctx.n_instances < 0 || forward_ctx.n_buckets < 0) {
                throw std::invalid_argument("Invalid dimensions, counts, or SH bases in backward_raw");
            }

            api_stage = "backward.pointer_validation";
            CHECK_CUDA_PTR(grad_image_ptr, "grad_image_ptr");
            CHECK_CUDA_PTR(grad_alpha_ptr, "grad_alpha_ptr");
            CHECK_CUDA_PTR(image_ptr, "image_ptr");
            CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");
            CHECK_CUDA_PTR(means_ptr, "means_ptr");
            CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
            CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
            CHECK_CUDA_PTR(raw_opacities_ptr, "raw_opacities_ptr");
            if (total_bases_sh_rest > 0) {
                CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
            }
            CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
            CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");

            CHECK_CUDA_PTR(grad_means_ptr, "grad_means_ptr");
            CHECK_CUDA_PTR(grad_scales_raw_ptr, "grad_scales_raw_ptr");
            CHECK_CUDA_PTR(grad_rotations_raw_ptr, "grad_rotations_raw_ptr");
            CHECK_CUDA_PTR(grad_opacities_raw_ptr, "grad_opacities_raw_ptr");
            CHECK_CUDA_PTR(grad_sh_coefficients_0_ptr, "grad_sh_coefficients_0_ptr");
            if (total_bases_sh_rest > 0) {
                CHECK_CUDA_PTR(grad_sh_coefficients_rest_ptr, "grad_sh_coefficients_rest_ptr");
            }

            CHECK_CUDA_PTR_OPTIONAL(densification_info_ptr, "densification_info_ptr");
            CHECK_CUDA_PTR_OPTIONAL(densification_error_map_ptr, "densification_error_map_ptr");
            CHECK_CUDA_PTR_OPTIONAL(grad_w2c_ptr, "grad_w2c_ptr");
            CHECK_CUDA_PTR_OPTIONAL(grad_render_normal_ptr, "grad_render_normal_ptr");
            CHECK_CUDA_PTR_OPTIONAL(render_normal_ptr, "render_normal_ptr");
            CHECK_CUDA_PTR_OPTIONAL(grad_depth_ptr, "grad_depth_ptr");
            CHECK_CUDA_PTR_OPTIONAL(depth_map_ptr, "depth_map_ptr");
            if ((grad_render_normal_ptr == nullptr) != (render_normal_ptr == nullptr)) {
                throw std::invalid_argument("FastGS normal backward requires both gradient and rendered normal pointers");
            }
            if ((grad_depth_ptr == nullptr) != (depth_map_ptr == nullptr)) {
                throw std::invalid_argument("FastGS depth backward requires both gradient and depth-map pointers");
            }

            api_stage = "backward.context_validation";
            if (!forward_ctx.per_primitive_buffers || !forward_ctx.per_tile_buffers) {
                throw std::invalid_argument("Invalid FastGS forward context buffers");
            }

            if (forward_ctx.n_instances > 0 && !forward_ctx.per_instance_buffers) {
                throw std::invalid_argument("Missing instance buffers in FastGS forward context");
            }

            if (forward_ctx.n_buckets > 0 && !forward_ctx.per_bucket_buffers) {
                throw std::invalid_argument("Missing bucket buffers in FastGS forward context");
            }

            if (!forward_ctx.grad_mean2d_helper || !forward_ctx.grad_conic_helper) {
                throw std::invalid_argument("Missing pre-allocated helper buffers in FastGS forward context");
            }

            float* grad_mean2d_helper = static_cast<float*>(forward_ctx.grad_mean2d_helper);
            float* grad_conic_helper = static_cast<float*>(forward_ctx.grad_conic_helper);
            char* primitive_blob = static_cast<char*>(forward_ctx.per_primitive_buffers);
            PerPrimitiveBuffers primitive_buffers =
                PerPrimitiveBuffers::from_blob(primitive_blob, n_primitives);

            api_stage = "backward.helper_sizing";
            const size_t grad_mean2d_size = checked_size_multiply(
                static_cast<size_t>(n_primitives), 2 * sizeof(float), "backward.grad_mean2d");
            const size_t grad_conic_size = checked_size_multiply(
                static_cast<size_t>(n_primitives), 3 * sizeof(float), "backward.grad_conic");
            api_stage = "backward.helper_memset";
            check_cuda(cudaMemset(grad_mean2d_helper, 0, grad_mean2d_size),
                       "memset.backward.grad_mean2d");
            check_cuda(cudaMemset(grad_conic_helper, 0, grad_conic_size),
                       "memset.backward.grad_conic");
            if (effective_observation_blur.enabled()) {
                const size_t grad_compensated_opacity_size = checked_size_multiply(
                    static_cast<size_t>(n_primitives), sizeof(float),
                    "backward.grad_compensated_opacity");
                check_cuda(cudaMemset(primitive_buffers.grad_compensated_opacity, 0,
                                      grad_compensated_opacity_size),
                           "memset.backward.grad_compensated_opacity");
            }

        // NOTE: Output gradients are NOT zeroed here to support tile-based training
        // where gradients accumulate across multiple backward calls (one per tile).
        // The caller (e.g., fast_rasterize_backward wrapper) is responsible for zeroing
        // gradients once before the first tile.
        //
        // cudaMemset(grad_means_ptr, 0, n_primitives * 3 * sizeof(float));
        // cudaMemset(grad_scales_raw_ptr, 0, n_primitives * 3 * sizeof(float));
        // cudaMemset(grad_rotations_raw_ptr, 0, n_primitives * 4 * sizeof(float));
        // cudaMemset(grad_opacities_raw_ptr, 0, n_primitives * sizeof(float));
        // cudaMemset(grad_sh_coefficients_0_ptr, 0, n_primitives * 3 * sizeof(float));
        // cudaMemset(grad_sh_coefficients_rest_ptr, 0, n_primitives * total_bases_sh_rest * 3 * sizeof(float));

            if (grad_w2c_ptr) {
                check_cuda(cudaMemset(grad_w2c_ptr, 0, 4 * 4 * sizeof(float)),
                           "memset.backward.grad_w2c");
            }

            api_stage = "backward.color_alpha";
            backward(
                densification_error_map_ptr,
                grad_image_ptr,
                grad_alpha_ptr,
                image_ptr,
                alpha_ptr,
                reinterpret_cast<const float3*>(means_ptr),
                reinterpret_cast<const float3*>(scales_raw_ptr),
                reinterpret_cast<const float4*>(rotations_raw_ptr),
                raw_opacities_ptr,
                reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
                reinterpret_cast<const float4*>(w2c_ptr),
                reinterpret_cast<const float3*>(cam_position_ptr),
                static_cast<char*>(forward_ctx.per_primitive_buffers),
                static_cast<char*>(forward_ctx.per_tile_buffers),
                static_cast<char*>(forward_ctx.per_instance_buffers),
                static_cast<char*>(forward_ctx.per_bucket_buffers),
                reinterpret_cast<float3*>(grad_means_ptr),
                reinterpret_cast<float3*>(grad_scales_raw_ptr),
                reinterpret_cast<float4*>(grad_rotations_raw_ptr),
                grad_opacities_raw_ptr,
                reinterpret_cast<float3*>(grad_sh_coefficients_0_ptr),
                reinterpret_cast<float3*>(grad_sh_coefficients_rest_ptr),
                reinterpret_cast<float2*>(grad_mean2d_helper),
                grad_conic_helper,
                grad_w2c_ptr ? reinterpret_cast<float4*>(grad_w2c_ptr) : nullptr,
                densification_info_ptr,
                n_primitives,
                forward_ctx.n_visible_primitives,
                forward_ctx.n_instances,
                forward_ctx.n_buckets,
                forward_ctx.primitive_primitive_indices_selector,
                forward_ctx.instance_primitive_indices_selector,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                mip_filter,
                effective_observation_blur);

            // Optional GGGS normal backward
            if (grad_render_normal_ptr && render_normal_ptr &&
                forward_ctx.has_normal_backward_buffers && forward_ctx.normal_accum_length_map) {
                api_stage = "backward.normal";
                backward_normal(
                    grad_render_normal_ptr,
                    render_normal_ptr,
                    static_cast<const float*>(forward_ctx.normal_accum_length_map),
                    reinterpret_cast<const float3*>(means_ptr),
                    reinterpret_cast<const float3*>(scales_raw_ptr),
                    reinterpret_cast<const float4*>(rotations_raw_ptr),
                    raw_opacities_ptr,
                    reinterpret_cast<const float4*>(w2c_ptr),
                    static_cast<char*>(forward_ctx.per_primitive_buffers),
                    static_cast<char*>(forward_ctx.per_tile_buffers),
                    static_cast<char*>(forward_ctx.per_instance_buffers),
                    static_cast<char*>(forward_ctx.per_bucket_buffers),
                    reinterpret_cast<float2*>(grad_mean2d_helper),
                    grad_conic_helper,
                    reinterpret_cast<float3*>(grad_means_ptr),
                    reinterpret_cast<float3*>(grad_scales_raw_ptr),
                    reinterpret_cast<float4*>(grad_rotations_raw_ptr),
                    grad_opacities_raw_ptr,
                    n_primitives,
                    forward_ctx.n_visible_primitives,
                    forward_ctx.n_instances,
                    forward_ctx.n_buckets,
                    forward_ctx.primitive_primitive_indices_selector,
                    forward_ctx.instance_primitive_indices_selector,
                    width,
                    height,
                    focal_x,
                    focal_y,
                    center_x,
                    center_y,
                    mip_filter);
            }

            // Optional GGGS depth backward (IFT-based, Equation 18)
            if (grad_depth_ptr && depth_map_ptr) {
                api_stage = "backward.depth";
                backward_depth(
                    grad_depth_ptr,
                    depth_map_ptr,
                    reinterpret_cast<const float3*>(means_ptr),
                    reinterpret_cast<const float3*>(scales_raw_ptr),
                    reinterpret_cast<const float4*>(rotations_raw_ptr),
                    raw_opacities_ptr,
                    reinterpret_cast<const float4*>(w2c_ptr),
                    static_cast<char*>(forward_ctx.per_primitive_buffers),
                    static_cast<char*>(forward_ctx.per_tile_buffers),
                    static_cast<char*>(forward_ctx.per_instance_buffers),
                    static_cast<char*>(forward_ctx.per_bucket_buffers),
                    reinterpret_cast<float2*>(grad_mean2d_helper),
                    grad_conic_helper,
                    reinterpret_cast<float3*>(grad_means_ptr),
                    reinterpret_cast<float3*>(grad_scales_raw_ptr),
                    reinterpret_cast<float4*>(grad_rotations_raw_ptr),
                    grad_opacities_raw_ptr,
                    n_primitives,
                    forward_ctx.n_visible_primitives,
                    forward_ctx.n_instances,
                    forward_ctx.n_buckets,
                    forward_ctx.primitive_primitive_indices_selector,
                    forward_ctx.instance_primitive_indices_selector,
                    width,
                    height,
                    focal_x,
                    focal_y,
                    center_x,
                    center_y,
                    mip_filter);
            }

            outputs.success = true;
            return outputs;

        } catch (const std::exception& e) {
            outputs.error_message = format_backward_error(e, api_stage, forward_ctx);
            return outputs;
        }
    }

    void warmup_kernels() {
        // Pre-compile rasterization kernels via minimal forward+backward pass.
        // All allocated memory is released before returning.

        constexpr int NUM_GAUSSIANS = 100;
        constexpr int IMG_WIDTH = 64;
        constexpr int IMG_HEIGHT = 64;
        constexpr float FOCAL = 50.0f;
        constexpr float CENTER_X = IMG_WIDTH / 2.0f;
        constexpr float CENTER_Y = IMG_HEIGHT / 2.0f;

        // Allocate all buffers in one block for efficiency
        constexpr size_t INPUT_SIZE = NUM_GAUSSIANS * (3 + 3 + 4 + 1 + 3) * sizeof(float) // means, scales, rotations, opacities, sh0
                                      + 16 * sizeof(float)                                // w2c
                                      + 3 * sizeof(float)                                 // cam_pos
                                      + IMG_WIDTH * IMG_HEIGHT * 4 * sizeof(float);       // image + alpha

        check_no_pending_cuda_error("warmup.entry.pending_cuda_error");
        ScopedCudaBuffer input_buffer_owner;
        char* const buffer = input_buffer_owner.allocate(INPUT_SIZE, "malloc.warmup.input");
        check_cuda(cudaMemset(buffer, 0, INPUT_SIZE), "memset.warmup.input");

        float* const means = reinterpret_cast<float*>(buffer);
        float* const scales = means + NUM_GAUSSIANS * 3;
        float* const rotations = scales + NUM_GAUSSIANS * 3;
        float* const opacities = rotations + NUM_GAUSSIANS * 4;
        float* const sh0 = opacities + NUM_GAUSSIANS;
        float* const w2c = sh0 + NUM_GAUSSIANS * 3;
        float* const cam_pos = w2c + 16;
        float* const image = cam_pos + 3;
        float* const alpha = image + IMG_WIDTH * IMG_HEIGHT * 3;

        // Initialize rotations to identity quaternion
        std::vector<float> rot_data(NUM_GAUSSIANS * 4);
        for (int i = 0; i < NUM_GAUSSIANS; ++i) {
            rot_data[i * 4] = 1.0f; // w=1, x=y=z=0
        }
        check_cuda(cudaMemcpy(rotations, rot_data.data(), rot_data.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "copy.warmup.rotations");

        // Initialize w2c to identity and camera position
        const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const float cam[3] = {0.0f, 0.0f, 5.0f};
        check_cuda(cudaMemcpy(w2c, identity, sizeof(identity), cudaMemcpyHostToDevice),
                   "copy.warmup.w2c");
        check_cuda(cudaMemcpy(cam_pos, cam, sizeof(cam), cudaMemcpyHostToDevice),
                   "copy.warmup.camera_position");

        // Forward pass compiles forward kernels
        const auto ctx = forward_raw(
            means, scales, rotations, opacities, sh0, nullptr,
            w2c, cam_pos, image, alpha, nullptr, nullptr,
            NUM_GAUSSIANS, 1, 0,
            IMG_WIDTH, IMG_HEIGHT,
            FOCAL, FOCAL, CENTER_X, CENTER_Y,
            0.01f, 100.0f);

        if (ctx.success) {
            auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
            ExistingArenaFrameGuard pending_forward_frame(arena, ctx.frame_id, true);

            // Allocate gradient buffers
            constexpr size_t GRAD_SIZE = IMG_WIDTH * IMG_HEIGHT * 4 * sizeof(float) + NUM_GAUSSIANS * (3 + 3 + 4 + 1 + 3) * sizeof(float);
            ScopedCudaBuffer grad_buffer_owner;
            char* const grad_buffer = grad_buffer_owner.allocate(GRAD_SIZE, "malloc.warmup.gradients");
            check_cuda(cudaMemset(grad_buffer, 0, GRAD_SIZE), "memset.warmup.gradients");

            float* const grad_image = reinterpret_cast<float*>(grad_buffer);
            float* const grad_alpha = grad_image + IMG_WIDTH * IMG_HEIGHT * 3;
            float* const grad_means = grad_alpha + IMG_WIDTH * IMG_HEIGHT;
            float* const grad_scales = grad_means + NUM_GAUSSIANS * 3;
            float* const grad_rotations = grad_scales + NUM_GAUSSIANS * 3;
            float* const grad_opacities = grad_rotations + NUM_GAUSSIANS * 4;
            float* const grad_sh0 = grad_opacities + NUM_GAUSSIANS;

            // Backward pass compiles backward kernels (also releases arena)
            // backward_raw assumes ownership of ending a successful forward frame.
            pending_forward_frame.release();
            const auto backward_result = backward_raw(
                nullptr, nullptr, grad_image, grad_alpha, image, alpha,
                means, scales, rotations, opacities, nullptr, w2c, cam_pos, ctx,
                grad_means, grad_scales, grad_rotations, grad_opacities,
                grad_sh0, nullptr, nullptr,
                NUM_GAUSSIANS, 1, 0,
                IMG_WIDTH, IMG_HEIGHT, FOCAL, FOCAL, CENTER_X, CENTER_Y, true);

            if (!backward_result.success) {
                throw std::runtime_error(backward_result.error_message);
            }
            grad_buffer_owner.free_checked("free.warmup.gradients");
        } else {
            throw std::runtime_error(ctx.error_message);
        }

        input_buffer_owner.free_checked("free.warmup.input");
    }

} // namespace fast_lfs::rasterization
