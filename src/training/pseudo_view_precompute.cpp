/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pseudo_view_precompute.hpp"

#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "mesh_supervision_renderer.hpp"
#include "pseudo_view_memory_policy.hpp"
#include "training/kernels/pseudo_view_reprojection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <glm/glm.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr float PI_F = 3.14159265358979323846f;
        constexpr float THETA_REF_RAD = 15.0f * PI_F / 180.0f;
        constexpr float T_REF = 0.05f;
        constexpr float DEPTH_REL_EPS = 0.01f;
        constexpr float VALID_RATIO_TIE_EPS = 0.03f;
        constexpr const char* PSEUDO_VIEW_POSE_GENERATION_MODE = "sphere_empty_cell_surface_lookat";

        struct Vec3 {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
        };

        struct PoseData {
            std::array<float, 9> R{};
            Vec3 T{};
            Vec3 C{};
            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
            int width = 0;
            int height = 0;
        };

        struct FaceGeom {
            Vec3 center;
            Vec3 normal;
            float area = 0.0f;
        };

        struct SourceCache {
            lfs::core::Tensor depth;
            lfs::core::Tensor mask;
            lfs::core::Tensor w2c_cuda;
            PoseData pose;
            int width = 0;
            int height = 0;
            bool has_mask = false;
        };

        struct ReprojectionChoice {
            int source_index = -1;
            int source_uid = -1;
            std::string source_image_name;
            int source_camera_id = -1;
            float valid_ratio = 0.0f;
            float pose_distance = std::numeric_limits<float>::infinity();
            float occlusion_fail_ratio = 0.0f;
            float projection_fail_ratio = 0.0f;
            float source_mask_fail_ratio = 0.0f;
            float source_face_angle_fail_ratio = 0.0f;
            int mesh_valid_pixels = 0;
            int valid_pixels = 0;
            lfs::core::Tensor rgb;
            lfs::core::Tensor valid_mask;
            lfs::core::Tensor mesh_mask;
        };

        struct PseudoViewTimingStats {
            double candidate_render_ms = 0.0;
            double candidate_reprojection_ms = 0.0;
            double source_cache_build_ms = 0.0;
            double source_rgb_load_ms = 0.0;
            double final_render_ms = 0.0;
            double final_reprojection_ms = 0.0;
            double mask_erode_ms = 0.0;
            double image_save_ms = 0.0;
            int candidate_render_count = 0;
            int candidate_reprojection_count = 0;
            int source_cache_build_count = 0;
            int source_rgb_load_count = 0;
            int final_render_count = 0;
            int final_reprojection_count = 0;
            int mask_erode_count = 0;
            int image_save_count = 0;
        };

        struct SaveCandidateOutputResult {
            bool accepted = false;
            std::string rejection_reason;
            int mesh_valid_pixels = 0;
            int raw_valid_pixels = 0;
            int supervision_valid_pixels = 0;
            float post_morph_valid_ratio = 0.0f;
            nlohmann::json entry;
            PseudoViewTimingStats timing;
        };

        struct PseudoFrontFaceMask {
            lfs::core::Tensor mask_cuda;
            int mesh_pixels = 0;
            int front_face_pixels = 0;
            int backface_pixels = 0;
        };

        struct FaceGpuTensors {
            lfs::core::Tensor centers_cuda;
            lfs::core::Tensor normals_cuda;
        };

        struct PseudoPreparedMeshWithCpuStaging {
            PreparedMesh prepared_mesh;
            lfs::core::Tensor world_vertices_xyz_cpu;
            lfs::core::Tensor triangle_indices_cpu;
        };

        struct FaceGeometryBuildResult {
            std::vector<FaceGeom> faces;
            FaceGpuTensors tensors;
            Vec3 bbox_min;
            Vec3 bbox_max;
        };

        struct BinaryMaskBytes {
            int width = 0;
            int height = 0;
            std::vector<uint8_t> values;
        };

        struct SphereCell {
            int id = -1;
            int lat_index = -1;
            int lon_index = -1;
            float lat_deg = 0.0f;
            float lon_deg = 0.0f;
            Vec3 direction;
            bool occupied_by_real = false;
            int real_camera_count = 0;
        };

        using SourceCacheLru = pseudo_view::ByteBudgetLru<int, SourceCache>;

        size_t pooled_tensor_bytes(const lfs::core::Tensor& tensor) {
            return tensor.is_valid() && !tensor.is_empty()
                       ? pseudo_view::pooled_allocation_bytes(tensor.bytes())
                       : 0;
        }

        size_t source_cache_bytes(const SourceCache& cache) {
            return pseudo_view::saturating_add(
                pooled_tensor_bytes(cache.depth),
                pseudo_view::saturating_add(
                    pooled_tensor_bytes(cache.mask),
                    pooled_tensor_bytes(cache.w2c_cuda)));
        }

        std::expected<size_t, std::string> cuda_free_memory_bytes(const std::string_view stage) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (error != cudaSuccess) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute: cudaMemGetInfo failed at {}: {}",
                    stage,
                    cudaGetErrorString(error)));
            }
            return free_bytes;
        }

        const char* quality_rejection_name(const pseudo_view::QualityRejection rejection) {
            using enum pseudo_view::QualityRejection;
            switch (rejection) {
            case None: return "none";
            case NoMeshHit: return "no_mesh_hit";
            case RawCoverage: return "raw_coverage_below_threshold";
            case EmptySupervision: return "empty_supervision_mask";
            case PostMorphCoverage: return "post_morphology_coverage_below_threshold";
            case MinValidPixels: return "valid_pixels_below_absolute_threshold";
            }
            return "unknown";
        }

        struct SurfaceTarget {
            Vec3 centroid;
            Vec3 normal;
            float area = 0.0f;
            float weight = 0.0f;
            float mean_cos = 0.0f;
            int face_count = 0;
            bool valid = false;
        };

        struct CandidateRecord {
            int sphere_cell_id = -1;
            float sphere_cell_lat_deg = 0.0f;
            float sphere_cell_lon_deg = 0.0f;
            float sphere_cell_angle_deg = 0.0f;
            int base_camera_index = -1;
            uint32_t base_camera_uid = 0;
            PoseData pose;
            Vec3 sphere_cell_direction;
            Vec3 surface_target;
            Vec3 surface_normal;
            float surface_area = 0.0f;
            int surface_face_count = 0;
            float surface_mean_cos = 0.0f;
            float radius = 0.0f;
            float radius_scale = 1.0f;
            int front_face_pixels = 0;
            int backface_pixels = 0;
            float front_face_ratio = 0.0f;
            float normal_alignment = 0.0f;
            float framing_score = 0.0f;
            ReprojectionChoice source;
            float score = -std::numeric_limits<float>::infinity();
        };

        Vec3 operator+(const Vec3& a, const Vec3& b) {
            return {a.x + b.x, a.y + b.y, a.z + b.z};
        }

        Vec3 operator-(const Vec3& a, const Vec3& b) {
            return {a.x - b.x, a.y - b.y, a.z - b.z};
        }

        Vec3 operator*(const Vec3& a, const float s) {
            return {a.x * s, a.y * s, a.z * s};
        }

        Vec3 operator/(const Vec3& a, const float s) {
            return {a.x / s, a.y / s, a.z / s};
        }

        float dot(const Vec3& a, const Vec3& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        Vec3 cross(const Vec3& a, const Vec3& b) {
            return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
        }

        float length(const Vec3& v) {
            return std::sqrt(std::max(0.0f, dot(v, v)));
        }

        Vec3 normalize(const Vec3& v, const Vec3& fallback = {0.0f, 0.0f, 1.0f}) {
            const float len = length(v);
            if (!(len > 1e-8f) || !std::isfinite(len)) {
                return fallback;
            }
            return v / len;
        }

        float distance(const Vec3& a, const Vec3& b) {
            return length(a - b);
        }

        float median_sorted(std::vector<float> values, const float fallback) {
            if (values.empty()) {
                return fallback;
            }
            std::sort(values.begin(), values.end());
            const size_t mid = values.size() / 2;
            if ((values.size() % 2) == 1) {
                return values[mid];
            }
            return 0.5f * (values[mid - 1] + values[mid]);
        }

        float clamp01(const float v) {
            return std::clamp(v, 0.0f, 1.0f);
        }

        std::array<float, 16> w2c_matrix(const PoseData& pose) {
            return {
                pose.R[0], pose.R[1], pose.R[2], pose.T.x,
                pose.R[3], pose.R[4], pose.R[5], pose.T.y,
                pose.R[6], pose.R[7], pose.R[8], pose.T.z,
                0.0f, 0.0f, 0.0f, 1.0f};
        }

        std::array<float, 16> c2w_matrix(const PoseData& pose) {
            return {
                pose.R[0], pose.R[3], pose.R[6], pose.C.x,
                pose.R[1], pose.R[4], pose.R[7], pose.C.y,
                pose.R[2], pose.R[5], pose.R[8], pose.C.z,
                0.0f, 0.0f, 0.0f, 1.0f};
        }

        lfs::core::Tensor mat4_cuda(const std::array<float, 16>& matrix) {
            return lfs::core::Tensor::from_vector(
                       std::vector<float>(matrix.begin(), matrix.end()),
                       {size_t{4}, size_t{4}},
                       lfs::core::Device::CPU)
                .to(lfs::core::Device::CUDA)
                .to(lfs::core::DataType::Float32)
                .contiguous();
        }

        using TimingClock = std::chrono::steady_clock;

        double elapsed_ms(const TimingClock::time_point start) {
            return std::chrono::duration<double, std::milli>(TimingClock::now() - start).count();
        }

        void add_timing(double& total_ms, int& count, const TimingClock::time_point start) {
            total_ms += elapsed_ms(start);
            count++;
        }

        void merge_timing(PseudoViewTimingStats& dst, const PseudoViewTimingStats& src) {
            dst.candidate_render_ms += src.candidate_render_ms;
            dst.candidate_reprojection_ms += src.candidate_reprojection_ms;
            dst.source_cache_build_ms += src.source_cache_build_ms;
            dst.source_rgb_load_ms += src.source_rgb_load_ms;
            dst.final_render_ms += src.final_render_ms;
            dst.final_reprojection_ms += src.final_reprojection_ms;
            dst.mask_erode_ms += src.mask_erode_ms;
            dst.image_save_ms += src.image_save_ms;
            dst.candidate_render_count += src.candidate_render_count;
            dst.candidate_reprojection_count += src.candidate_reprojection_count;
            dst.source_cache_build_count += src.source_cache_build_count;
            dst.source_rgb_load_count += src.source_rgb_load_count;
            dst.final_render_count += src.final_render_count;
            dst.final_reprojection_count += src.final_reprojection_count;
            dst.mask_erode_count += src.mask_erode_count;
            dst.image_save_count += src.image_save_count;
        }

        nlohmann::json timing_json(const PseudoViewTimingStats& timing) {
            return {
                {"candidate_render_ms", timing.candidate_render_ms},
                {"candidate_reprojection_ms", timing.candidate_reprojection_ms},
                {"source_cache_build_ms", timing.source_cache_build_ms},
                {"source_rgb_load_ms", timing.source_rgb_load_ms},
                {"final_render_ms", timing.final_render_ms},
                {"final_reprojection_ms", timing.final_reprojection_ms},
                {"mask_erode_ms", timing.mask_erode_ms},
                {"image_save_ms", timing.image_save_ms},
                {"candidate_render_count", timing.candidate_render_count},
                {"candidate_reprojection_count", timing.candidate_reprojection_count},
                {"source_cache_build_count", timing.source_cache_build_count},
                {"source_rgb_load_count", timing.source_rgb_load_count},
                {"final_render_count", timing.final_render_count},
                {"final_reprojection_count", timing.final_reprojection_count},
                {"mask_erode_count", timing.mask_erode_count},
                {"image_save_count", timing.image_save_count}};
        }

        Vec3 camera_right_world(const PoseData& pose) {
            return {pose.R[0], pose.R[1], pose.R[2]};
        }

        Vec3 camera_up_world(const PoseData& pose) {
            return {pose.R[3], pose.R[4], pose.R[5]};
        }

        Vec3 camera_forward_world(const PoseData& pose) {
            return {pose.R[6], pose.R[7], pose.R[8]};
        }

        Vec3 transform_point_w2c(const PoseData& pose, const Vec3& p) {
            return {
                pose.R[0] * p.x + pose.R[1] * p.y + pose.R[2] * p.z + pose.T.x,
                pose.R[3] * p.x + pose.R[4] * p.y + pose.R[5] * p.z + pose.T.y,
                pose.R[6] * p.x + pose.R[7] * p.y + pose.R[8] * p.z + pose.T.z};
        }

        bool project_inside(const PoseData& pose, const Vec3& point) {
            const Vec3 c = transform_point_w2c(pose, point);
            if (!(c.z > 1e-6f) || !std::isfinite(c.z)) {
                return false;
            }
            const float u = pose.fx * (c.x / c.z) + pose.cx;
            const float v = pose.fy * (c.y / c.z) + pose.cy;
            return std::isfinite(u) && std::isfinite(v) &&
                   u >= 0.0f && v >= 0.0f &&
                   u < static_cast<float>(pose.width) &&
                   v < static_cast<float>(pose.height);
        }

        float projected_framing_score(const PoseData& pose, const Vec3& point) {
            const Vec3 c = transform_point_w2c(pose, point);
            if (!(c.z > 1e-6f) || !std::isfinite(c.z)) {
                return 0.0f;
            }
            const float u = pose.fx * (c.x / c.z) + pose.cx;
            const float v = pose.fy * (c.y / c.z) + pose.cy;
            if (!std::isfinite(u) || !std::isfinite(v) ||
                u < 0.0f || v < 0.0f ||
                u >= static_cast<float>(pose.width) ||
                v >= static_cast<float>(pose.height)) {
                return 0.0f;
            }
            const float dx = (u - static_cast<float>(pose.width) * 0.5f) /
                             std::max(static_cast<float>(pose.width) * 0.5f, 1.0f);
            const float dy = (v - static_cast<float>(pose.height) * 0.5f) /
                             std::max(static_cast<float>(pose.height) * 0.5f, 1.0f);
            return 1.0f - 0.5f * clamp01(std::sqrt(dx * dx + dy * dy));
        }

        float rotation_distance(const std::array<float, 9>& a, const std::array<float, 9>& b) {
            float trace = 0.0f;
            trace += a[0] * b[0] + a[3] * b[3] + a[6] * b[6];
            trace += a[1] * b[1] + a[4] * b[4] + a[7] * b[7];
            trace += a[2] * b[2] + a[5] * b[5] + a[8] * b[8];
            const float c = std::clamp((trace - 1.0f) * 0.5f, -1.0f, 1.0f);
            return std::acos(c);
        }

        float pose_distance(const PoseData& a, const PoseData& b, const float scene_radius) {
            const float theta_r = rotation_distance(a.R, b.R);
            const float dist_t = distance(a.C, b.C) / std::max(scene_radius, 1e-6f);
            return 0.6f * (theta_r / THETA_REF_RAD) + 0.4f * (dist_t / T_REF);
        }

        std::expected<PseudoFrontFaceMask, std::string> build_pseudo_front_face_mask(
            const PoseData& pseudo_pose,
            const lfs::core::Tensor& triangle_id,
            const lfs::core::Tensor& face_centers_cuda,
            const lfs::core::Tensor& face_normals_cuda,
            const float face_angle_max_deg) {
            if (!triangle_id.is_valid() || triangle_id.is_empty()) {
                return std::unexpected("Pseudo-view front-face mask requires triangle ids");
            }
            if (triangle_id.ndim() != 2) {
                return std::unexpected("Pseudo-view front-face mask triangle id tensor must be [H,W]");
            }
            if (!face_centers_cuda.is_valid() || face_centers_cuda.is_empty() ||
                !face_normals_cuda.is_valid() || face_normals_cuda.is_empty() ||
                face_centers_cuda.ndim() != 2 || face_normals_cuda.ndim() != 2 ||
                face_centers_cuda.shape()[1] != 3 || face_normals_cuda.shape()[1] != 3 ||
                face_centers_cuda.shape()[0] != face_normals_cuda.shape()[0]) {
                return std::unexpected("Pseudo-view front-face mask requires face centers/normals [F,3]");
            }

            const int height = static_cast<int>(triangle_id.shape()[0]);
            const int width = static_cast<int>(triangle_id.shape()[1]);
            auto mask_cuda = lfs::core::Tensor::zeros(
                {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            auto counters = lfs::core::Tensor::zeros(
                {static_cast<size_t>(kernels::PseudoViewFrontFaceCounter::Count)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Int32);

            const float clamped_angle = std::clamp(face_angle_max_deg, 0.0f, 180.0f);
            const float front_min_cos = std::cos(clamped_angle * PI_F / 180.0f);

            kernels::launch_pseudo_view_front_face_mask(
                triangle_id.ptr<int32_t>(),
                width,
                height,
                face_centers_cuda.ptr<float>(),
                face_normals_cuda.ptr<float>(),
                static_cast<int>(face_centers_cuda.shape()[0]),
                pseudo_pose.C.x,
                pseudo_pose.C.y,
                pseudo_pose.C.z,
                front_min_cos,
                mask_cuda.ptr<float>(),
                counters.ptr<int32_t>(),
                mask_cuda.stream());

            auto counters_cpu = counters.to(lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Int32)
                                    .contiguous();
            const int32_t* c = counters_cpu.ptr<int32_t>();

            PseudoFrontFaceMask result;
            result.mask_cuda = std::move(mask_cuda);
            result.mesh_pixels = c[static_cast<int>(kernels::PseudoViewFrontFaceCounter::Mesh)];
            result.front_face_pixels = c[static_cast<int>(kernels::PseudoViewFrontFaceCounter::Front)];
            result.backface_pixels = c[static_cast<int>(kernels::PseudoViewFrontFaceCounter::Back)];
            return result;
        }

        std::expected<void, std::string> save_image_low_ram(
            const std::filesystem::path& path,
            const lfs::core::Tensor& image,
            const std::string& label) {
            if (!image.is_valid() || image.is_empty()) {
                return std::unexpected(std::format("{} image is empty", label));
            }

            lfs::core::Tensor img = image;
            if (img.ndim() == 4) {
                img = img.squeeze(0);
            }

            int width = 0;
            int height = 0;
            int channels = 0;
            lfs::core::Tensor img_u8;
            if (img.ndim() == 2) {
                height = static_cast<int>(img.shape()[0]);
                width = static_cast<int>(img.shape()[1]);
                channels = 1;
                img_u8 = (img.clamp(0.0f, 1.0f) * 255.0f)
                             .to(lfs::core::DataType::UInt8)
                             .contiguous();
            } else if (img.ndim() == 3 && img.shape()[0] >= 1 && img.shape()[0] <= 4) {
                channels = static_cast<int>(img.shape()[0]);
                height = static_cast<int>(img.shape()[1]);
                width = static_cast<int>(img.shape()[2]);
                img_u8 = (img.clamp(0.0f, 1.0f) * 255.0f)
                             .to(lfs::core::DataType::UInt8);
                if (channels == 1) {
                    img_u8 = img_u8.squeeze(0).contiguous();
                } else {
                    img_u8 = img_u8.permute({1, 2, 0}).contiguous();
                }
            } else {
                return std::unexpected(std::format("{} image must be [H,W] or [C,H,W]", label));
            }

            auto img_cpu = img_u8.to(lfs::core::Device::CPU).contiguous();
            auto* data = reinterpret_cast<unsigned char*>(img_cpu.ptr<uint8_t>());
            if (!lfs::core::save_img_data(path, std::make_tuple(data, width, height, channels))) {
                return std::unexpected(std::format(
                    "Failed to save {} image '{}'",
                    label,
                    lfs::core::path_to_utf8(path)));
            }
            return {};
        }

        std::expected<BinaryMaskBytes, std::string> binary_mask_bytes_from_tensor(
            const lfs::core::Tensor& mask,
            const std::string& label) {
            if (!mask.is_valid() || mask.is_empty()) {
                return std::unexpected(std::format("{} mask is empty", label));
            }

            auto mask_cpu = mask.to(lfs::core::DataType::UInt8)
                                .to(lfs::core::Device::CPU)
                                .contiguous();
            int height = 0;
            int width = 0;
            if (mask_cpu.ndim() == 2) {
                height = static_cast<int>(mask_cpu.shape()[0]);
                width = static_cast<int>(mask_cpu.shape()[1]);
            } else if (mask_cpu.ndim() == 3 && mask_cpu.shape()[0] == 1) {
                height = static_cast<int>(mask_cpu.shape()[1]);
                width = static_cast<int>(mask_cpu.shape()[2]);
            } else {
                return std::unexpected(std::format("{} mask must be [H,W] or [1,H,W]", label));
            }

            BinaryMaskBytes out;
            out.width = width;
            out.height = height;
            out.values.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
            const uint8_t* src = mask_cpu.ptr<uint8_t>();
            for (size_t i = 0; i < out.values.size(); ++i) {
                out.values[i] = src[i] > 0 ? uint8_t{1} : uint8_t{0};
            }
            return out;
        }

        BinaryMaskBytes erode_binary_mask_bytes(
            const BinaryMaskBytes& mask,
            const int radius_pixels) {
            if (radius_pixels <= 0) {
                return mask;
            }

            BinaryMaskBytes out;
            out.width = mask.width;
            out.height = mask.height;
            out.values.assign(mask.values.size(), uint8_t{0});

            for (int y = 0; y < mask.height; ++y) {
                for (int x = 0; x < mask.width; ++x) {
                    const int pixel = y * mask.width + x;
                    if (mask.values[static_cast<size_t>(pixel)] == 0) {
                        continue;
                    }

                    bool keep = true;
                    for (int dy = -radius_pixels; dy <= radius_pixels && keep; ++dy) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= mask.height) {
                            keep = false;
                            break;
                        }
                        for (int dx = -radius_pixels; dx <= radius_pixels; ++dx) {
                            const int xx = x + dx;
                            if (xx < 0 || xx >= mask.width ||
                                mask.values[static_cast<size_t>(yy * mask.width + xx)] == 0) {
                                keep = false;
                                break;
                            }
                        }
                    }

                    if (keep) {
                        out.values[static_cast<size_t>(pixel)] = 1;
                    }
                }
            }

            return out;
        }

        BinaryMaskBytes dilate_binary_mask_bytes(
            const BinaryMaskBytes& mask,
            const int radius_pixels) {
            if (radius_pixels <= 0) {
                return mask;
            }

            BinaryMaskBytes out;
            out.width = mask.width;
            out.height = mask.height;
            out.values.assign(mask.values.size(), uint8_t{0});

            for (int y = 0; y < mask.height; ++y) {
                for (int x = 0; x < mask.width; ++x) {
                    const int pixel = y * mask.width + x;
                    bool keep = false;
                    for (int dy = -radius_pixels; dy <= radius_pixels && !keep; ++dy) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= mask.height) {
                            continue;
                        }
                        for (int dx = -radius_pixels; dx <= radius_pixels; ++dx) {
                            const int xx = x + dx;
                            if (xx >= 0 && xx < mask.width &&
                                mask.values[static_cast<size_t>(yy * mask.width + xx)] != 0) {
                                keep = true;
                                break;
                            }
                        }
                    }

                    if (keep) {
                        out.values[static_cast<size_t>(pixel)] = 1;
                    }
                }
            }

            return out;
        }

        BinaryMaskBytes open_binary_mask_bytes(
            const BinaryMaskBytes& mask,
            const int radius_pixels) {
            if (radius_pixels <= 0) {
                return mask;
            }

            return dilate_binary_mask_bytes(erode_binary_mask_bytes(mask, radius_pixels), radius_pixels);
        }

        int count_binary_mask_pixels(const BinaryMaskBytes& mask) {
            int count = 0;
            for (const uint8_t value : mask.values) {
                if (value != 0) {
                    count++;
                }
            }
            return count;
        }

        std::expected<BinaryMaskBytes, std::string> build_supervision_mask(
            const lfs::core::Tensor& raw_valid_mask,
            const int valid_open_pixels,
            const int invalid_dilate_pixels,
            const int erode_pixels,
            const std::string& label) {
            auto mask_result = binary_mask_bytes_from_tensor(raw_valid_mask, label);
            if (!mask_result) {
                return std::unexpected(mask_result.error());
            }
            BinaryMaskBytes supervision_mask = std::move(*mask_result);
            if (valid_open_pixels > 0) {
                supervision_mask = open_binary_mask_bytes(supervision_mask, valid_open_pixels);
            }
            if (invalid_dilate_pixels > 0) {
                supervision_mask = erode_binary_mask_bytes(supervision_mask, invalid_dilate_pixels);
            }
            if (erode_pixels > 0) {
                supervision_mask = erode_binary_mask_bytes(supervision_mask, erode_pixels);
            }
            return supervision_mask;
        }

        std::expected<void, std::string> save_binary_mask_low_ram(
            const std::filesystem::path& path,
            const BinaryMaskBytes& mask,
            const std::string& label) {
            if (mask.width <= 0 || mask.height <= 0 ||
                mask.values.size() != static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height)) {
                return std::unexpected(std::format("{} mask has invalid dimensions", label));
            }

            std::vector<uint8_t> png_bytes(mask.values.size());
            for (size_t i = 0; i < mask.values.size(); ++i) {
                png_bytes[i] = mask.values[i] != 0 ? uint8_t{255} : uint8_t{0};
            }
            auto* data = reinterpret_cast<unsigned char*>(png_bytes.data());
            if (!lfs::core::save_img_data(path, std::make_tuple(data, mask.width, mask.height, 1))) {
                return std::unexpected(std::format(
                    "Failed to save {} mask '{}'",
                    label,
                    lfs::core::path_to_utf8(path)));
            }
            return {};
        }

        std::filesystem::path relative_to_output(
            const std::filesystem::path& output_dir,
            const std::filesystem::path& path) {
            std::error_code ec;
            auto rel = std::filesystem::relative(path, output_dir, ec);
            return ec ? path.filename() : rel;
        }

        std::string sanitize_token(const std::string& raw) {
            std::string out;
            out.reserve(raw.size());
            for (const unsigned char ch : raw) {
                if ((ch >= 'a' && ch <= 'z') ||
                    (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') ||
                    ch == '_' || ch == '-' || ch == '.') {
                    out.push_back(static_cast<char>(ch));
                } else {
                    out.push_back('_');
                }
            }
            return out.empty() ? "pseudo" : out;
        }

        nlohmann::json vec3_json(const Vec3& v) {
            return nlohmann::json::array({v.x, v.y, v.z});
        }

        nlohmann::json mat3_json(const std::array<float, 9>& m) {
            return nlohmann::json::array({
                nlohmann::json::array({m[0], m[1], m[2]}),
                nlohmann::json::array({m[3], m[4], m[5]}),
                nlohmann::json::array({m[6], m[7], m[8]})});
        }

        nlohmann::json mat4_json(const std::array<float, 16>& m) {
            return nlohmann::json::array({
                nlohmann::json::array({m[0], m[1], m[2], m[3]}),
                nlohmann::json::array({m[4], m[5], m[6], m[7]}),
                nlohmann::json::array({m[8], m[9], m[10], m[11]}),
                nlohmann::json::array({m[12], m[13], m[14], m[15]})});
        }

        Vec3 direction_from_lat_lon(const float lat_deg, const float lon_deg) {
            const float lat = lat_deg * PI_F / 180.0f;
            const float lon = lon_deg * PI_F / 180.0f;
            const float c = std::cos(lat);
            return normalize({c * std::cos(lon), c * std::sin(lon), std::sin(lat)});
        }

        std::vector<SphereCell> build_sphere_cells(const float cell_angle_deg) {
            const float step = std::clamp(cell_angle_deg, 1.0f, 90.0f);
            const int lat_count = std::max(1, static_cast<int>(std::round(180.0f / step)));
            const int lon_count = std::max(1, static_cast<int>(std::round(360.0f / step)));
            const float lat_step = 180.0f / static_cast<float>(lat_count);
            const float lon_step = 360.0f / static_cast<float>(lon_count);

            std::vector<SphereCell> cells;
            cells.reserve(static_cast<size_t>(lat_count * lon_count));
            for (int lat = 0; lat < lat_count; ++lat) {
                const float lat_deg = -90.0f + (static_cast<float>(lat) + 0.5f) * lat_step;
                for (int lon = 0; lon < lon_count; ++lon) {
                    const float lon_deg = -180.0f + (static_cast<float>(lon) + 0.5f) * lon_step;
                    SphereCell cell;
                    cell.id = static_cast<int>(cells.size());
                    cell.lat_index = lat;
                    cell.lon_index = lon;
                    cell.lat_deg = lat_deg;
                    cell.lon_deg = lon_deg;
                    cell.direction = direction_from_lat_lon(lat_deg, lon_deg);
                    cells.push_back(std::move(cell));
                }
            }
            return cells;
        }

        int nearest_sphere_cell(const std::vector<SphereCell>& cells, const Vec3& direction) {
            if (cells.empty()) {
                return -1;
            }
            const Vec3 d = normalize(direction);
            int best = 0;
            float best_dot = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < cells.size(); ++i) {
                const float s = dot(cells[i].direction, d);
                if (s > best_dot) {
                    best_dot = s;
                    best = static_cast<int>(i);
                }
            }
            return best;
        }

        int nearest_camera_for_direction(
            const std::vector<PoseData>& camera_poses,
            const Vec3& mesh_center,
            const Vec3& direction) {
            if (camera_poses.empty()) {
                return -1;
            }
            const Vec3 d = normalize(direction);
            int best = 0;
            float best_dot = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < camera_poses.size(); ++i) {
                const Vec3 real_dir = normalize(camera_poses[i].C - mesh_center, d);
                const float s = dot(real_dir, d);
                if (s > best_dot) {
                    best_dot = s;
                    best = static_cast<int>(i);
                }
            }
            return best;
        }

        SurfaceTarget surface_target_for_cell(
            const std::vector<FaceGeom>& faces,
            const Vec3& cell_direction,
            const float support_min_cos) {
            SurfaceTarget target;
            Vec3 centroid_sum;
            Vec3 normal_sum;
            float cos_sum = 0.0f;
            const Vec3 view_dir = normalize(cell_direction);
            for (const auto& face : faces) {
                const float face_cos = dot(face.normal, view_dir);
                if (!(face_cos >= support_min_cos) || !(face.area > 0.0f) || !std::isfinite(face_cos)) {
                    continue;
                }
                const float weight = face.area * std::max(face_cos, 0.0f);
                if (!(weight > 0.0f) || !std::isfinite(weight)) {
                    continue;
                }
                centroid_sum = centroid_sum + face.center * weight;
                normal_sum = normal_sum + face.normal * weight;
                target.area += face.area;
                target.weight += weight;
                cos_sum += face_cos;
                target.face_count++;
            }
            if (target.face_count <= 0 || !(target.weight > 0.0f)) {
                return target;
            }
            target.centroid = centroid_sum / target.weight;
            target.normal = normalize(normal_sum, view_dir);
            target.mean_cos = cos_sum / static_cast<float>(target.face_count);
            target.valid = true;
            return target;
        }

        std::expected<void, std::string> write_json_file(
            const std::filesystem::path& path,
            const nlohmann::json& json) {
            std::ofstream out;
            if (!lfs::core::open_file_for_write(path, std::ios::out | std::ios::trunc, out)) {
                return std::unexpected(std::format(
                    "Failed to open '{}' for writing",
                    lfs::core::path_to_utf8(path)));
            }
            out << json.dump(2);
            if (!out.good()) {
                return std::unexpected(std::format(
                    "Failed to write '{}'",
                    lfs::core::path_to_utf8(path)));
            }
            return {};
        }

        PoseData pose_from_camera(lfs::core::Camera& camera) {
            if (camera.image_width() <= 0 || camera.image_height() <= 0) {
                camera.load_image_size();
            }

            PoseData pose;
            const auto [fx, fy, cx, cy] = camera.get_intrinsics();
            pose.fx = fx;
            pose.fy = fy;
            pose.cx = cx;
            pose.cy = cy;
            pose.width = camera.image_width();
            pose.height = camera.image_height();

            auto r_cpu = camera.R().to(lfs::core::Device::CPU)
                             .to(lfs::core::DataType::Float32)
                             .contiguous();
            const float* r = r_cpu.ptr<float>();
            for (int i = 0; i < 9; ++i) {
                pose.R[static_cast<size_t>(i)] = r[i];
            }

            auto c_cpu = camera.cam_position().to(lfs::core::Device::CPU)
                             .to(lfs::core::DataType::Float32)
                             .contiguous();
            const float* c = c_cpu.ptr<float>();
            pose.C = {c[0], c[1], c[2]};

            pose.T = {
                -(pose.R[0] * pose.C.x + pose.R[1] * pose.C.y + pose.R[2] * pose.C.z),
                -(pose.R[3] * pose.C.x + pose.R[4] * pose.C.y + pose.R[5] * pose.C.z),
                -(pose.R[6] * pose.C.x + pose.R[7] * pose.C.y + pose.R[8] * pose.C.z)};
            return pose;
        }

        PoseData make_look_at_pose(
            const PoseData& base,
            const Vec3& center,
            const Vec3& target) {
            PoseData pose = base;
            pose.C = center;

            const Vec3 forward = normalize(target - center, camera_forward_world(base));
            Vec3 up_seed = camera_up_world(base);
            Vec3 right = cross(up_seed, forward);
            if (length(right) < 1e-5f) {
                up_seed = {0.0f, 1.0f, 0.0f};
                right = cross(up_seed, forward);
            }
            if (length(right) < 1e-5f) {
                up_seed = camera_right_world(base);
                right = cross(up_seed, forward);
            }
            right = normalize(right, camera_right_world(base));
            const Vec3 up = normalize(cross(forward, right), camera_up_world(base));

            pose.R = {
                right.x, right.y, right.z,
                up.x, up.y, up.z,
                forward.x, forward.y, forward.z};
            pose.T = {
                -dot(right, center),
                -dot(up, center),
                -dot(forward, center)};
            return pose;
        }

        lfs::core::Camera make_pseudo_camera(
            const PoseData& pose,
            const lfs::core::Camera& base,
            const std::string& image_name,
            const int uid) {
            auto r_tensor = lfs::core::Tensor::from_vector(
                std::vector<float>(pose.R.begin(), pose.R.end()),
                {size_t{3}, size_t{3}},
                lfs::core::Device::CPU);
            auto t_tensor = lfs::core::Tensor::from_vector(
                std::vector<float>{pose.T.x, pose.T.y, pose.T.z},
                {size_t{3}},
                lfs::core::Device::CPU);

            lfs::core::Camera camera(
                r_tensor,
                t_tensor,
                pose.fx,
                pose.fy,
                pose.cx,
                pose.cy,
                base.radial_distortion(),
                base.tangential_distortion(),
                base.camera_model_type(),
                image_name,
                base.image_path(),
                {},
                pose.width,
                pose.height,
                uid,
                base.camera_id());
            camera.set_image_dimensions(pose.width, pose.height);
            return camera;
        }

        std::expected<PseudoPreparedMeshWithCpuStaging, std::string> prepare_mesh_geometry_with_cpu_staging(
            lfs::core::Scene& scene) {
            const auto visible_meshes = scene.getVisibleMeshes();
            if (visible_meshes.empty()) {
                return std::unexpected("Pseudo-view precompute skipped: no visible mesh nodes in scene");
            }

            size_t total_vertices = 0;
            size_t total_face_capacity = 0;
            for (const auto& vm : visible_meshes) {
                if (!vm.mesh) {
                    continue;
                }
                const auto& mesh = *vm.mesh;
                if (!mesh.vertices.is_valid() || !mesh.indices.is_valid() ||
                    mesh.vertex_count() <= 0 || mesh.face_count() <= 0) {
                    continue;
                }
                if (mesh.vertices.ndim() != 2 || mesh.vertices.shape()[1] != 3 ||
                    mesh.indices.ndim() != 2 || mesh.indices.shape()[1] != 3) {
                    return std::unexpected("Mesh tensor layout mismatch for pseudo-view rasterization");
                }
                total_vertices += static_cast<size_t>(mesh.vertex_count());
                total_face_capacity += static_cast<size_t>(mesh.face_count());
            }

            if (total_vertices == 0 || total_face_capacity == 0) {
                return std::unexpected("Pseudo-view precompute skipped: mesh tensors are empty");
            }

            auto world_vertices_cpu = lfs::core::Tensor::empty_unpinned(
                {total_vertices, size_t{3}},
                lfs::core::DataType::Float32);
            auto triangle_indices_cpu = lfs::core::Tensor::empty_unpinned(
                {total_face_capacity, size_t{3}},
                lfs::core::DataType::Int32);

            float* world_vertices = world_vertices_cpu.ptr<float>();
            int32_t* world_indices = triangle_indices_cpu.ptr<int32_t>();
            size_t vertex_offset = 0;
            size_t face_count = 0;

            for (const auto& vm : visible_meshes) {
                if (!vm.mesh) {
                    continue;
                }
                const auto& mesh = *vm.mesh;
                if (!mesh.vertices.is_valid() || !mesh.indices.is_valid() ||
                    mesh.vertex_count() <= 0 || mesh.face_count() <= 0) {
                    continue;
                }

                auto local_vertices_cpu = mesh.vertices.to(lfs::core::Device::CPU)
                                              .to(lfs::core::DataType::Float32)
                                              .contiguous();
                auto local_indices_cpu = mesh.indices.to(lfs::core::Device::CPU)
                                             .to(lfs::core::DataType::Int32)
                                             .contiguous();
                if (local_vertices_cpu.ndim() != 2 || local_vertices_cpu.shape()[1] != 3 ||
                    local_indices_cpu.ndim() != 2 || local_indices_cpu.shape()[1] != 3) {
                    return std::unexpected("Mesh tensor layout mismatch for pseudo-view rasterization");
                }

                const int64_t num_vertices = static_cast<int64_t>(local_vertices_cpu.shape()[0]);
                const int64_t num_faces = static_cast<int64_t>(local_indices_cpu.shape()[0]);

                const float* vertices = local_vertices_cpu.ptr<float>();
                const int32_t* indices = local_indices_cpu.ptr<int32_t>();

                for (int64_t v = 0; v < num_vertices; ++v) {
                    const glm::vec4 wp = vm.transform * glm::vec4(
                        vertices[v * 3 + 0],
                        vertices[v * 3 + 1],
                        vertices[v * 3 + 2],
                        1.0f);
                    const size_t dst = (vertex_offset + static_cast<size_t>(v)) * 3;
                    world_vertices[dst + 0] = wp.x;
                    world_vertices[dst + 1] = wp.y;
                    world_vertices[dst + 2] = wp.z;
                }

                for (int64_t f = 0; f < num_faces; ++f) {
                    const int32_t i0 = indices[f * 3 + 0];
                    const int32_t i1 = indices[f * 3 + 1];
                    const int32_t i2 = indices[f * 3 + 2];
                    if (i0 < 0 || i1 < 0 || i2 < 0 ||
                        i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices) {
                        continue;
                    }

                    world_indices[face_count * 3 + 0] = static_cast<int32_t>(vertex_offset + static_cast<size_t>(i0));
                    world_indices[face_count * 3 + 1] = static_cast<int32_t>(vertex_offset + static_cast<size_t>(i1));
                    world_indices[face_count * 3 + 2] = static_cast<int32_t>(vertex_offset + static_cast<size_t>(i2));
                    face_count++;
                }

                vertex_offset += static_cast<size_t>(num_vertices);
            }

            if (vertex_offset == 0 || face_count == 0) {
                return std::unexpected("Pseudo-view precompute skipped: mesh tensors are empty");
            }
            if (face_count < total_face_capacity) {
                auto compact_indices_cpu = lfs::core::Tensor::empty_unpinned(
                    {face_count, size_t{3}},
                    lfs::core::DataType::Int32);
                std::memcpy(
                    compact_indices_cpu.ptr<int32_t>(),
                    triangle_indices_cpu.ptr<int32_t>(),
                    face_count * 3 * sizeof(int32_t));
                triangle_indices_cpu = std::move(compact_indices_cpu);
            }

            PseudoPreparedMeshWithCpuStaging result;
            result.prepared_mesh.vertex_count = static_cast<int>(vertex_offset);
            result.prepared_mesh.face_count = static_cast<int>(face_count);
            result.prepared_mesh.world_vertices_xyz = world_vertices_cpu.to(lfs::core::Device::CUDA)
                                                          .to(lfs::core::DataType::Float32)
                                                          .contiguous();
            result.prepared_mesh.triangle_indices = triangle_indices_cpu.to(lfs::core::Device::CUDA)
                                                       .to(lfs::core::DataType::Int32)
                                                       .contiguous();
            result.world_vertices_xyz_cpu = std::move(world_vertices_cpu);
            result.triangle_indices_cpu = std::move(triangle_indices_cpu);
            return result;
        }

        std::expected<FaceGeometryBuildResult, std::string> build_face_geometry_and_tensors(
            const PreparedMesh& mesh,
            const lfs::core::Tensor& verts_cpu,
            const lfs::core::Tensor& indices_cpu) {
            if (verts_cpu.ndim() != 2 || verts_cpu.shape()[1] != 3 ||
                indices_cpu.ndim() != 2 || indices_cpu.shape()[1] != 3 ||
                static_cast<int>(verts_cpu.shape()[0]) < mesh.vertex_count ||
                static_cast<int>(indices_cpu.shape()[0]) < mesh.face_count) {
                return std::unexpected("Pseudo-view precompute: prepared mesh CPU staging tensor layout mismatch");
            }

            const float* v = verts_cpu.ptr<float>();
            const int32_t* idx = indices_cpu.ptr<int32_t>();
            auto face_centers_cpu = lfs::core::Tensor::empty_unpinned(
                {static_cast<size_t>(mesh.face_count), size_t{3}},
                lfs::core::DataType::Float32);
            auto face_normals_cpu = lfs::core::Tensor::empty_unpinned(
                {static_cast<size_t>(mesh.face_count), size_t{3}},
                lfs::core::DataType::Float32);
            float* centers = face_centers_cpu.ptr<float>();
            float* normals = face_normals_cpu.ptr<float>();

            FaceGeometryBuildResult result;
            result.faces.resize(static_cast<size_t>(mesh.face_count));
            result.bbox_min = {
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};
            result.bbox_max = {
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity()};

            for (int vi = 0; vi < mesh.vertex_count; ++vi) {
                const Vec3 p{v[vi * 3 + 0], v[vi * 3 + 1], v[vi * 3 + 2]};
                result.bbox_min.x = std::min(result.bbox_min.x, p.x);
                result.bbox_min.y = std::min(result.bbox_min.y, p.y);
                result.bbox_min.z = std::min(result.bbox_min.z, p.z);
                result.bbox_max.x = std::max(result.bbox_max.x, p.x);
                result.bbox_max.y = std::max(result.bbox_max.y, p.y);
                result.bbox_max.z = std::max(result.bbox_max.z, p.z);
            }

            for (int f = 0; f < mesh.face_count; ++f) {
                const int32_t i0 = idx[f * 3 + 0];
                const int32_t i1 = idx[f * 3 + 1];
                const int32_t i2 = idx[f * 3 + 2];
                FaceGeom face;
                if (i0 >= 0 && i1 >= 0 && i2 >= 0 &&
                    i0 < mesh.vertex_count && i1 < mesh.vertex_count && i2 < mesh.vertex_count) {
                    const Vec3 p0{v[i0 * 3 + 0], v[i0 * 3 + 1], v[i0 * 3 + 2]};
                    const Vec3 p1{v[i1 * 3 + 0], v[i1 * 3 + 1], v[i1 * 3 + 2]};
                    const Vec3 p2{v[i2 * 3 + 0], v[i2 * 3 + 1], v[i2 * 3 + 2]};
                    const Vec3 n_raw = cross(p1 - p0, p2 - p0);
                    const float double_area = length(n_raw);
                    face = {
                        (p0 + p1 + p2) / 3.0f,
                        normalize(n_raw, {0.0f, 0.0f, 1.0f}),
                        0.5f * double_area};
                }

                result.faces[static_cast<size_t>(f)] = face;
                centers[f * 3 + 0] = face.center.x;
                centers[f * 3 + 1] = face.center.y;
                centers[f * 3 + 2] = face.center.z;
                normals[f * 3 + 0] = face.normal.x;
                normals[f * 3 + 1] = face.normal.y;
                normals[f * 3 + 2] = face.normal.z;
            }

            result.tensors.centers_cuda = face_centers_cpu.to(lfs::core::Device::CUDA)
                                              .to(lfs::core::DataType::Float32)
                                              .contiguous();
            result.tensors.normals_cuda = face_normals_cpu.to(lfs::core::Device::CUDA)
                                              .to(lfs::core::DataType::Float32)
                                              .contiguous();
            return result;
        }

        std::expected<SourceCache, std::string> build_source_cache(
            const PreparedMesh& prepared_mesh,
            lfs::core::Camera& camera,
            const lfs::core::param::TrainingParameters& params) {
            SourceCache cache;
            camera.load_image_size(params.dataset.resize_factor, params.dataset.max_width);
            cache.width = camera.image_width();
            cache.height = camera.image_height();
            cache.pose = pose_from_camera(camera);
            cache.w2c_cuda = mat4_cuda(w2c_matrix(cache.pose));

            auto depth_result = render_mesh_supervision_targets_for_camera(
                prepared_mesh,
                camera,
                params,
                MeshSupervisionOutputRequest::depth_only(),
                false);
            if (!depth_result) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute: source mesh depth render failed for '{}': {}",
                    camera.image_name(),
                    depth_result.error()));
            }
            cache.depth = std::move(depth_result->depth);

            auto mask = camera.load_and_get_mask(
                params.dataset.resize_factor,
                params.dataset.max_width,
                params.optimization.invert_masks,
                params.optimization.mask_threshold);
            if (mask.is_valid() && !mask.is_empty()) {
                cache.mask = mask.to(lfs::core::Device::CUDA)
                                 .to(lfs::core::DataType::Float32)
                                 .contiguous();
                if (cache.mask.ndim() == 3 && cache.mask.shape()[0] == 1) {
                    cache.mask = cache.mask.squeeze(0).contiguous();
                }
                cache.has_mask =
                    cache.mask.is_valid() &&
                    cache.mask.ndim() == 2 &&
                    static_cast<int>(cache.mask.shape()[1]) == cache.width &&
                    static_cast<int>(cache.mask.shape()[0]) == cache.height;
                if (!cache.has_mask) {
                    cache.mask = lfs::core::Tensor();
                }
            }
            camera.clear_cached_mask();
            return cache;
        }

        std::expected<lfs::core::Tensor, std::string> load_source_rgb(
            const SourceCache& cache,
            lfs::core::Camera& camera,
            const lfs::core::param::TrainingParameters& params) {
            auto rgb = camera.load_and_get_image(params.dataset.resize_factor, params.dataset.max_width)
                           .to(lfs::core::Device::CUDA)
                           .to(lfs::core::DataType::Float32)
                           .contiguous();
            if (!rgb.is_valid() || rgb.ndim() != 3 || rgb.shape()[0] < 3) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute: invalid source RGB for '{}'",
                    camera.image_name()));
            }
            if (static_cast<int>(rgb.shape()[2]) != cache.width ||
                static_cast<int>(rgb.shape()[1]) != cache.height) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute: source RGB size mismatch for '{}'",
                    camera.image_name()));
            }

            return rgb;
        }

        std::expected<ReprojectionChoice, std::string> reproject_from_source(
            const PoseData& pseudo_pose,
            const lfs::core::Tensor& pseudo_depth,
            const lfs::core::Tensor& pseudo_triangle_id,
            const lfs::core::Tensor* pseudo_mesh_mask,
            const lfs::core::Tensor& face_normals_cuda,
            const float source_face_angle_min_cos,
            const lfs::core::Tensor& pseudo_c2w,
            const SourceCache& source,
            const lfs::core::Tensor* source_rgb,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            const int source_index,
            const float scene_radius,
            const bool keep_outputs) {
            const int width = pseudo_pose.width;
            const int height = pseudo_pose.height;
            if (width <= 0 || height <= 0) {
                return std::unexpected("Pseudo-view precompute: invalid pseudo camera size");
            }
            if (keep_outputs &&
                (!source_rgb || !source_rgb->is_valid() || source_rgb->is_empty())) {
                return std::unexpected("Pseudo-view precompute: full RGB reprojection requires loaded source RGB");
            }
            if (!pseudo_triangle_id.is_valid() || pseudo_triangle_id.is_empty() || pseudo_triangle_id.ndim() != 2) {
                return std::unexpected("Pseudo-view precompute: source reprojection requires pseudo triangle ids");
            }
            if (!face_normals_cuda.is_valid() || face_normals_cuda.is_empty() ||
                face_normals_cuda.ndim() != 2 || face_normals_cuda.shape()[1] != 3) {
                return std::unexpected("Pseudo-view precompute: source reprojection requires face normals [F,3]");
            }

            auto counters = lfs::core::Tensor::zeros(
                {static_cast<size_t>(kernels::PseudoViewReprojectionCounter::Count)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Int32);

            const float* source_mask_ptr = source.has_mask ? source.mask.ptr<float>() : nullptr;
            const float* pseudo_mesh_mask_ptr =
                pseudo_mesh_mask && pseudo_mesh_mask->is_valid() && !pseudo_mesh_mask->is_empty()
                    ? pseudo_mesh_mask->ptr<float>()
                    : nullptr;
            const int face_count = static_cast<int>(face_normals_cuda.shape()[0]);
            lfs::core::Tensor rgb;
            lfs::core::Tensor valid_mask;
            lfs::core::Tensor mesh_mask;
            if (keep_outputs) {
                rgb = lfs::core::Tensor::zeros(
                    {size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                valid_mask = lfs::core::Tensor::zeros(
                    {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                mesh_mask = lfs::core::Tensor::zeros(
                    {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
                kernels::launch_pseudo_view_reproject_rgb(
                    pseudo_depth.ptr<float>(),
                    pseudo_mesh_mask_ptr,
                    pseudo_triangle_id.ptr<int32_t>(),
                    width,
                    height,
                    pseudo_pose.fx,
                    pseudo_pose.fy,
                    pseudo_pose.cx,
                    pseudo_pose.cy,
                    pseudo_c2w.ptr<float>(),
                    source.depth.ptr<float>(),
                    source_rgb->ptr<float>(),
                    source_mask_ptr,
                    source.width,
                    source.height,
                    source.pose.fx,
                    source.pose.fy,
                    source.pose.cx,
                    source.pose.cy,
                    source.w2c_cuda.ptr<float>(),
                    face_normals_cuda.ptr<float>(),
                    face_count,
                    source.pose.C.x,
                    source.pose.C.y,
                    source.pose.C.z,
                    source_face_angle_min_cos,
                    scene_radius,
                    DEPTH_REL_EPS,
                    std::clamp(source.has_mask ? 0.5f : 0.0f, 0.0f, 1.0f),
                    rgb.ptr<float>(),
                    valid_mask.ptr<float>(),
                    mesh_mask.ptr<float>(),
                    counters.ptr<int32_t>(),
                    rgb.stream());
            } else {
                kernels::launch_pseudo_view_reproject_score(
                    pseudo_depth.ptr<float>(),
                    pseudo_mesh_mask_ptr,
                    pseudo_triangle_id.ptr<int32_t>(),
                    width,
                    height,
                    pseudo_pose.fx,
                    pseudo_pose.fy,
                    pseudo_pose.cx,
                    pseudo_pose.cy,
                    pseudo_c2w.ptr<float>(),
                    source.depth.ptr<float>(),
                    source_mask_ptr,
                    source.width,
                    source.height,
                    source.pose.fx,
                    source.pose.fy,
                    source.pose.cx,
                    source.pose.cy,
                    source.w2c_cuda.ptr<float>(),
                    face_normals_cuda.ptr<float>(),
                    face_count,
                    source.pose.C.x,
                    source.pose.C.y,
                    source.pose.C.z,
                    source_face_angle_min_cos,
                    scene_radius,
                    DEPTH_REL_EPS,
                    std::clamp(source.has_mask ? 0.5f : 0.0f, 0.0f, 1.0f),
                    counters.ptr<int32_t>(),
                    counters.stream());
            }

            auto counters_cpu = counters.to(lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Int32)
                                    .contiguous();
            const int32_t* c = counters_cpu.ptr<int32_t>();
            const int mesh_valid = c[static_cast<int>(kernels::PseudoViewReprojectionCounter::MeshValid)];
            const int valid = c[static_cast<int>(kernels::PseudoViewReprojectionCounter::Valid)];

            ReprojectionChoice choice;
            choice.source_index = source_index;
            const auto& source_camera = cameras[static_cast<size_t>(source_index)];
            choice.source_uid = source_camera->uid();
            choice.source_image_name = source_camera->image_name();
            choice.source_camera_id = source_camera->camera_id();
            choice.mesh_valid_pixels = mesh_valid;
            choice.valid_pixels = valid;
            choice.valid_ratio = mesh_valid > 0 ? static_cast<float>(valid) / static_cast<float>(mesh_valid) : 0.0f;
            choice.occlusion_fail_ratio = mesh_valid > 0
                                               ? static_cast<float>(c[static_cast<int>(kernels::PseudoViewReprojectionCounter::OcclusionFail)]) / static_cast<float>(mesh_valid)
                                               : 0.0f;
            choice.projection_fail_ratio = mesh_valid > 0
                                               ? static_cast<float>(c[static_cast<int>(kernels::PseudoViewReprojectionCounter::ProjectionFail)]) / static_cast<float>(mesh_valid)
                                               : 0.0f;
            choice.source_mask_fail_ratio = mesh_valid > 0
                                                ? static_cast<float>(c[static_cast<int>(kernels::PseudoViewReprojectionCounter::SourceMaskFail)]) / static_cast<float>(mesh_valid)
                                                : 0.0f;
            choice.source_face_angle_fail_ratio = mesh_valid > 0
                                                      ? static_cast<float>(c[static_cast<int>(kernels::PseudoViewReprojectionCounter::SourceFaceAngleFail)]) / static_cast<float>(mesh_valid)
                                                      : 0.0f;
            choice.pose_distance = pose_distance(source.pose, pseudo_pose, scene_radius);
            if (keep_outputs) {
                choice.rgb = std::move(rgb);
                choice.valid_mask = std::move(valid_mask);
                choice.mesh_mask = std::move(mesh_mask);
            }
            return choice;
        }

        std::expected<ReprojectionChoice, std::string> choose_best_source(
            const PoseData& pseudo_pose,
            const lfs::core::Tensor& pseudo_depth,
            const lfs::core::Tensor& pseudo_triangle_id,
            const lfs::core::Tensor* pseudo_mesh_mask,
            const PreparedMesh& prepared_mesh,
            const lfs::core::Tensor& face_normals_cuda,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            const lfs::core::param::TrainingParameters& params,
            SourceCacheLru& source_cache,
            const float scene_radius,
            const float source_face_angle_min_cos,
            const bool keep_outputs,
            const std::optional<int> forced_source = std::nullopt,
            PseudoViewTimingStats* timing = nullptr,
            const bool detailed_logs = true) {
            std::vector<int> source_indices;
            if (forced_source.has_value()) {
                source_indices.push_back(*forced_source);
            } else {
                source_indices.resize(cameras.size());
                for (size_t i = 0; i < cameras.size(); ++i) {
                    source_indices[i] = static_cast<int>(i);
                }
                std::sort(source_indices.begin(), source_indices.end(), [&](const int a, const int b) {
                    const PoseData pa = pose_from_camera(*cameras[static_cast<size_t>(a)]);
                    const PoseData pb = pose_from_camera(*cameras[static_cast<size_t>(b)]);
                    const float da = distance(pa.C, pseudo_pose.C);
                    const float db = distance(pb.C, pseudo_pose.C);
                    return da < db;
                });
                const int top_k = std::max(1, params.optimization.pseudo_view_top_k_source);
                if (static_cast<int>(source_indices.size()) > top_k) {
                    source_indices.resize(static_cast<size_t>(top_k));
                }
                std::sort(source_indices.begin(), source_indices.end(), [&](const int a, const int b) {
                    const PoseData pa = pose_from_camera(*cameras[static_cast<size_t>(a)]);
                    const PoseData pb = pose_from_camera(*cameras[static_cast<size_t>(b)]);
                    return pose_distance(pa, pseudo_pose, scene_radius) <
                           pose_distance(pb, pseudo_pose, scene_radius);
                });
            }

            const auto pseudo_c2w = mat4_cuda(c2w_matrix(pseudo_pose));
            std::optional<ReprojectionChoice> best;
            for (const int source_index : source_indices) {
                if (source_index < 0 || source_index >= static_cast<int>(cameras.size())) {
                    continue;
                }
                SourceCache transient_cache;
                SourceCache* cached_source = source_cache.find(source_index);
                if (!cached_source) {
                    const auto cache_start = TimingClock::now();
                    auto cache_result = build_source_cache(prepared_mesh, *cameras[static_cast<size_t>(source_index)], params);
                    if (timing) {
                        add_timing(timing->source_cache_build_ms, timing->source_cache_build_count, cache_start);
                    }
                    if (!cache_result) {
                        if (detailed_logs) {
                            LOG_WARN("Pseudo-view source '{}' skipped: {}",
                                     cameras[static_cast<size_t>(source_index)]->image_name(),
                                     cache_result.error());
                        }
                        continue;
                    }
                    const size_t bytes = source_cache_bytes(*cache_result);
                    if (bytes <= source_cache.budget_bytes()) {
                        cached_source = source_cache.insert(
                            source_index,
                            std::move(*cache_result),
                            bytes);
                        if (!cached_source) {
                            return std::unexpected(
                                "Pseudo-view source cache failed to retain an in-budget entry");
                        }
                    } else {
                        transient_cache = std::move(*cache_result);
                        cached_source = &transient_cache;
                    }
                }

                const auto reproject_start = TimingClock::now();
                auto choice_result = reproject_from_source(
                    pseudo_pose,
                    pseudo_depth,
                    pseudo_triangle_id,
                    pseudo_mesh_mask,
                    face_normals_cuda,
                    source_face_angle_min_cos,
                    pseudo_c2w,
                    *cached_source,
                    nullptr,
                    cameras,
                    source_index,
                    scene_radius,
                    keep_outputs);
                if (timing && !keep_outputs) {
                    add_timing(
                        timing->candidate_reprojection_ms,
                        timing->candidate_reprojection_count,
                        reproject_start);
                }
                if (!choice_result) {
                    return std::unexpected(choice_result.error());
                }
                auto choice = std::move(*choice_result);
                if (!best.has_value() ||
                    choice.valid_ratio > best->valid_ratio + VALID_RATIO_TIE_EPS ||
                    (std::abs(choice.valid_ratio - best->valid_ratio) <= VALID_RATIO_TIE_EPS &&
                     choice.pose_distance < best->pose_distance)) {
                    best = std::move(choice);
                }
            }

            if (!best.has_value()) {
                return std::unexpected("Pseudo-view precompute: no usable source camera for reprojection");
            }
            return std::move(*best);
        }

        std::expected<SaveCandidateOutputResult, std::string> save_candidate_outputs(
            const CandidateRecord& candidate,
            const int pseudo_id,
            const PreparedMesh& prepared_mesh,
            const lfs::core::Tensor& face_centers_cuda,
            const lfs::core::Tensor& face_normals_cuda,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            const lfs::core::param::TrainingParameters& params,
            const SourceCache& selected_source,
            const lfs::core::Tensor& source_rgb,
            const float scene_radius,
            const float source_rgb_face_angle_max_deg,
            const float source_rgb_face_angle_min_cos,
            const std::filesystem::path& output_dir) {
            SaveCandidateOutputResult output;
            auto& timing = output.timing;
            auto& base = *cameras[static_cast<size_t>(candidate.base_camera_index)];
            auto pseudo_camera = make_pseudo_camera(
                candidate.pose,
                base,
                std::format("pseudo_{:05d}.png", pseudo_id),
                -1 - pseudo_id);
            const auto final_render_start = TimingClock::now();
            auto render_result = render_mesh_supervision_targets_for_camera(
                prepared_mesh,
                pseudo_camera,
                params,
                MeshSupervisionOutputRequest::depth_and_triangle_id(),
                false);
            add_timing(timing.final_render_ms, timing.final_render_count, final_render_start);
            if (!render_result) {
                return std::unexpected(std::format(
                    "Pseudo-view render failed while saving pseudo_{:05d}: {}",
                    pseudo_id,
                    render_result.error()));
            }
            auto front_mask_result = build_pseudo_front_face_mask(
                candidate.pose,
                render_result->triangle_id,
                face_centers_cuda,
                face_normals_cuda,
                params.optimization.pseudo_view_rgb_face_angle_max_deg);
            if (!front_mask_result) {
                return std::unexpected(front_mask_result.error());
            }
            auto front_mask = std::move(*front_mask_result);
            if (front_mask.mesh_pixels <= 0 || front_mask.front_face_pixels <= 0) {
                output.rejection_reason = "no_mesh_hit";
                return output;
            }
            const float saved_front_face_ratio = front_mask.mesh_pixels > 0
                                                     ? static_cast<float>(front_mask.front_face_pixels) /
                                                           static_cast<float>(front_mask.mesh_pixels)
                                                     : 0.0f;

            const int source_index = candidate.source.source_index;
            if (!source_rgb.is_valid() || source_rgb.is_empty()) {
                return std::unexpected(std::format(
                    "Pseudo-view save failed for pseudo_{:05d}: source RGB is not loaded",
                    pseudo_id));
            }

            const auto final_reproject_start = TimingClock::now();
            const auto pseudo_c2w = mat4_cuda(c2w_matrix(candidate.pose));
            auto source_result = reproject_from_source(
                candidate.pose,
                render_result->depth,
                render_result->triangle_id,
                &front_mask.mask_cuda,
                face_normals_cuda,
                source_rgb_face_angle_min_cos,
                pseudo_c2w,
                selected_source,
                &source_rgb,
                cameras,
                source_index,
                scene_radius,
                true);
            add_timing(timing.final_reprojection_ms, timing.final_reprojection_count, final_reproject_start);
            if (!source_result) {
                return std::unexpected(source_result.error());
            }
            auto source = std::move(*source_result);
            output.mesh_valid_pixels = source.mesh_valid_pixels;
            output.raw_valid_pixels = source.valid_pixels;
            if (source.mesh_valid_pixels <= 0) {
                output.rejection_reason = "no_mesh_hit";
                return output;
            }
            if (source.valid_ratio < params.optimization.pseudo_view_min_valid_ratio) {
                output.rejection_reason = "raw_coverage_below_threshold";
                return output;
            }
            const int supervision_mask_valid_open_pixels =
                std::max(0, params.optimization.pseudo_view_mask_valid_open_pixels);
            const int supervision_mask_erode_pixels = std::max(0, params.optimization.pseudo_view_mask_erode_pixels);
            const int supervision_mask_invalid_dilate_pixels =
                std::max(0, params.optimization.pseudo_view_mask_invalid_dilate_pixels);
            BinaryMaskBytes supervision_mask;
            int supervision_valid_pixels = source.valid_pixels;
            {
                const auto mask_erode_start = TimingClock::now();
                auto supervision_mask_result = build_supervision_mask(
                    source.valid_mask,
                    supervision_mask_valid_open_pixels,
                    supervision_mask_invalid_dilate_pixels,
                    supervision_mask_erode_pixels,
                    std::format("pseudo_{:05d}", pseudo_id));
                if (!supervision_mask_result) {
                    return std::unexpected(supervision_mask_result.error());
                }
                supervision_mask = std::move(*supervision_mask_result);
                supervision_valid_pixels = count_binary_mask_pixels(supervision_mask);
                if (supervision_mask_valid_open_pixels > 0 ||
                    supervision_mask_erode_pixels > 0 ||
                    supervision_mask_invalid_dilate_pixels > 0) {
                    add_timing(timing.mask_erode_ms, timing.mask_erode_count, mask_erode_start);
                }
            }

            const auto quality = pseudo_view::evaluate_supervision_quality(
                source.mesh_valid_pixels,
                source.valid_pixels,
                supervision_valid_pixels,
                params.optimization.pseudo_view_min_valid_ratio,
                params.optimization.pseudo_view_min_valid_pixels);
            output.mesh_valid_pixels = source.mesh_valid_pixels;
            output.raw_valid_pixels = source.valid_pixels;
            output.supervision_valid_pixels = supervision_valid_pixels;
            output.post_morph_valid_ratio = quality.post_morph_ratio;
            if (!quality.accepted()) {
                output.rejection_reason = quality_rejection_name(quality.rejection);
                return output;
            }

            try {
                kernels::launch_pseudo_view_apply_valid_mask_rgb(
                    source.rgb.ptr<float>(),
                    source.valid_mask.ptr<float>(),
                    candidate.pose.width,
                    candidate.pose.height,
                    source.rgb.stream());
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "Pseudo-view RGB final mask application failed for pseudo_{:05d}: {}",
                    pseudo_id,
                    e.what()));
            }

            const auto rgb_path = output_dir / "rgb" / std::format("pseudo_{:05d}.png", pseudo_id);
            const auto valid_mask_path = output_dir / "mask" / std::format("pseudo_{:05d}_valid.png", pseudo_id);
            const auto mesh_mask_path = output_dir / "mask" / std::format("pseudo_{:05d}_mesh.png", pseudo_id);
            const auto supervision_mask_path = output_dir / "mask" / std::format("pseudo_{:05d}_supervision.png", pseudo_id);
            const auto pose_path = output_dir / "pose" / std::format("pseudo_{:05d}.json", pseudo_id);

            const bool minimal_metadata = params.optimization.pseudo_view_minimal_metadata;
            try {
                const auto image_save_start = TimingClock::now();
                if (auto save_rgb = save_image_low_ram(rgb_path, source.rgb, "pseudo RGB"); !save_rgb) {
                    return std::unexpected(save_rgb.error());
                }
                if (!minimal_metadata) {
                    if (auto save_valid = save_image_low_ram(valid_mask_path, source.valid_mask, "pseudo valid"); !save_valid) {
                        return std::unexpected(save_valid.error());
                    }
                    if (auto save_mesh = save_image_low_ram(mesh_mask_path, source.mesh_mask, "pseudo mesh"); !save_mesh) {
                        return std::unexpected(save_mesh.error());
                    }
                }
                if (auto save_supervision = save_binary_mask_low_ram(
                        supervision_mask_path,
                        supervision_mask,
                        "pseudo supervision");
                    !save_supervision) {
                    return std::unexpected(save_supervision.error());
                }
                add_timing(timing.image_save_ms, timing.image_save_count, image_save_start);
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "Pseudo-view output image save failed for pseudo_{:05d}: {}",
                    pseudo_id,
                    e.what()));
            }

            const nlohmann::json pose_intrinsics = {
                {"fx", candidate.pose.fx},
                {"fy", candidate.pose.fy},
                {"cx", candidate.pose.cx},
                {"cy", candidate.pose.cy}};
            nlohmann::json pose_json = pseudo_view::make_minimal_pose(
                pseudo_id,
                mat3_json(candidate.pose.R),
                vec3_json(candidate.pose.T),
                pose_intrinsics,
                candidate.pose.width,
                candidate.pose.height);
            if (!minimal_metadata) {
                const auto w2c = w2c_matrix(candidate.pose);
                const auto c2w = c2w_matrix(candidate.pose);
                pose_json["world_to_camera"] = mat4_json(w2c);
                pose_json["camera_to_world"] = mat4_json(c2w);
            }
            if (!minimal_metadata) {
                pose_json["camera_center"] = vec3_json(candidate.pose.C);
                pose_json["sphere_cell_id"] = candidate.sphere_cell_id;
                pose_json["sphere_cell_lat_deg"] = candidate.sphere_cell_lat_deg;
                pose_json["sphere_cell_lon_deg"] = candidate.sphere_cell_lon_deg;
                pose_json["sphere_cell_angle_deg"] = candidate.sphere_cell_angle_deg;
                pose_json["sphere_cell_direction"] = vec3_json(candidate.sphere_cell_direction);
                pose_json["radius"] = candidate.radius;
                pose_json["radius_scale"] = candidate.radius_scale;
                pose_json["surface_target"] = vec3_json(candidate.surface_target);
                pose_json["surface_normal"] = vec3_json(candidate.surface_normal);
                pose_json["surface_area"] = candidate.surface_area;
                pose_json["surface_face_count"] = candidate.surface_face_count;
                pose_json["surface_mean_cos"] = candidate.surface_mean_cos;
                pose_json["support_centroid"] = vec3_json(candidate.surface_target);
                pose_json["support_normal"] = vec3_json(candidate.surface_normal);
                pose_json["support_area"] = candidate.surface_area;
                pose_json["support_face_count"] = candidate.surface_face_count;
                pose_json["front_face_pixels"] = source.mesh_valid_pixels;
                pose_json["backface_pixels"] = front_mask.backface_pixels;
                pose_json["front_face_ratio"] = saved_front_face_ratio;
                pose_json["valid_ratio"] = source.valid_ratio;
                pose_json["source_rgb_face_angle_max_deg"] = source_rgb_face_angle_max_deg;
                pose_json["source_face_angle_fail_ratio"] = source.source_face_angle_fail_ratio;
                pose_json["supervision_valid_pixels"] = supervision_valid_pixels;
                pose_json["supervision_mask_valid_open_pixels"] = supervision_mask_valid_open_pixels;
                pose_json["supervision_mask_erode_pixels"] = supervision_mask_erode_pixels;
                pose_json["supervision_mask_invalid_dilate_pixels"] = supervision_mask_invalid_dilate_pixels;
                pose_json["pose_generation_mode"] = PSEUDO_VIEW_POSE_GENERATION_MODE;
                pose_json["source_id"] = source.source_uid;
                pose_json["source_uid"] = source.source_uid;
                pose_json["source_image_name"] = source.source_image_name;
                pose_json["source_camera_id"] = source.source_camera_id;
            }
            if (auto write_pose = write_json_file(pose_path, pose_json); !write_pose) {
                return std::unexpected(write_pose.error());
            }

            nlohmann::json entry = pseudo_view::make_minimal_view_entry(
                pseudo_id,
                candidate.base_camera_uid,
                source.source_image_name,
                source.source_camera_id,
                source.source_uid,
                lfs::core::path_to_utf8(relative_to_output(output_dir, rgb_path)),
                lfs::core::path_to_utf8(relative_to_output(output_dir, supervision_mask_path)),
                lfs::core::path_to_utf8(relative_to_output(output_dir, pose_path)));
            if (!minimal_metadata) {
                entry["source_id"] = source.source_uid;
                entry["sphere_cell_id"] = candidate.sphere_cell_id;
                entry["sphere_cell_lat_deg"] = candidate.sphere_cell_lat_deg;
                entry["sphere_cell_lon_deg"] = candidate.sphere_cell_lon_deg;
                entry["sphere_cell_angle_deg"] = candidate.sphere_cell_angle_deg;
                entry["sphere_cell_direction"] = vec3_json(candidate.sphere_cell_direction);
                entry["radius"] = candidate.radius;
                entry["radius_scale"] = candidate.radius_scale;
                entry["surface_target"] = vec3_json(candidate.surface_target);
                entry["surface_normal"] = vec3_json(candidate.surface_normal);
                entry["surface_area"] = candidate.surface_area;
                entry["surface_face_count"] = candidate.surface_face_count;
                entry["surface_mean_cos"] = candidate.surface_mean_cos;
                entry["support_centroid"] = vec3_json(candidate.surface_target);
                entry["support_normal"] = vec3_json(candidate.surface_normal);
                entry["support_area"] = candidate.surface_area;
                entry["support_face_count"] = candidate.surface_face_count;
                entry["pose_generation_mode"] = PSEUDO_VIEW_POSE_GENERATION_MODE;
                entry["valid_mask_path"] = lfs::core::path_to_utf8(relative_to_output(output_dir, valid_mask_path));
                entry["mesh_mask_path"] = lfs::core::path_to_utf8(relative_to_output(output_dir, mesh_mask_path));
                entry["pose_distance"] = source.pose_distance;
                entry["valid_ratio"] = source.valid_ratio;
                entry["valid_pixels"] = source.valid_pixels;
                entry["mesh_valid_pixels"] = source.mesh_valid_pixels;
                entry["supervision_valid_pixels"] = supervision_valid_pixels;
                entry["supervision_mask_valid_open_pixels"] = supervision_mask_valid_open_pixels;
                entry["supervision_mask_erode_pixels"] = supervision_mask_erode_pixels;
                entry["supervision_mask_invalid_dilate_pixels"] = supervision_mask_invalid_dilate_pixels;
                entry["front_face_pixels"] = source.mesh_valid_pixels;
                entry["backface_pixels"] = front_mask.backface_pixels;
                entry["front_face_ratio"] = saved_front_face_ratio;
                entry["normal_alignment"] = candidate.normal_alignment;
                entry["framing_score"] = candidate.framing_score;
                entry["occlusion_fail_ratio"] = source.occlusion_fail_ratio;
                entry["projection_fail_ratio"] = source.projection_fail_ratio;
                entry["source_mask_fail_ratio"] = source.source_mask_fail_ratio;
                entry["source_rgb_face_angle_max_deg"] = source_rgb_face_angle_max_deg;
                entry["source_face_angle_fail_ratio"] = source.source_face_angle_fail_ratio;
                entry["candidate_score"] = candidate.score;
            }
            output.accepted = true;
            output.entry = std::move(entry);
            return output;
        }

    } // namespace

    std::expected<PseudoViewPrecomputeStats, std::string> run_pseudo_view_precompute(
        lfs::core::Scene& scene,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& train_cameras,
        const lfs::core::param::TrainingParameters& params) {
        PseudoViewPrecomputeStats stats;
        PseudoViewTimingStats timing;
        stats.total_train_cameras = train_cameras.size();
        stats.output_dir = params.dataset.output_path / "pseudo_views";
        stats.manifest_path = stats.output_dir / "manifest.json";

        if (train_cameras.empty()) {
            return std::unexpected("Pseudo-view precompute requires at least one training camera");
        }
        if (!scene.hasVisibleMeshes()) {
            return std::unexpected("Pseudo-view precompute requires at least one visible GT mesh");
        }

        try {
            // This directory is a generated cache. Remove prior generated
            // artifacts so a minimal-metadata refresh cannot leave stale full
            // diagnostic pose/mask files reachable on disk.
            std::filesystem::remove_all(stats.output_dir / "rgb");
            std::filesystem::remove_all(stats.output_dir / "mask");
            std::filesystem::remove_all(stats.output_dir / "pose");
            std::filesystem::remove(stats.manifest_path);
            std::filesystem::create_directories(stats.output_dir / "rgb");
            std::filesystem::create_directories(stats.output_dir / "mask");
            std::filesystem::create_directories(stats.output_dir / "pose");

            auto prepared_result = prepare_mesh_geometry_with_cpu_staging(scene);
            if (!prepared_result) {
                return std::unexpected(prepared_result.error());
            }
            auto prepared_with_staging = std::move(*prepared_result);
            PreparedMesh prepared_mesh = std::move(prepared_with_staging.prepared_mesh);
            if (!prepared_mesh.is_valid()) {
                return std::unexpected("Pseudo-view precompute: prepared mesh is empty");
            }

            auto face_result = build_face_geometry_and_tensors(
                prepared_mesh,
                prepared_with_staging.world_vertices_xyz_cpu,
                prepared_with_staging.triangle_indices_cpu);
            if (!face_result) {
                return std::unexpected(face_result.error());
            }
            auto face_geometry = std::move(*face_result);
            std::vector<FaceGeom> faces = std::move(face_geometry.faces);
            FaceGpuTensors face_tensors = std::move(face_geometry.tensors);
            const Vec3 bbox_min = face_geometry.bbox_min;
            const Vec3 bbox_max = face_geometry.bbox_max;
            prepared_with_staging.world_vertices_xyz_cpu = lfs::core::Tensor();
            prepared_with_staging.triangle_indices_cpu = lfs::core::Tensor();
            auto free_after_mesh_result = cuda_free_memory_bytes("source-cache budgeting");
            if (!free_after_mesh_result) {
                return std::unexpected(free_after_mesh_result.error());
            }
            const size_t source_cache_budget =
                pseudo_view::source_cache_budget_bytes(*free_after_mesh_result);
            SourceCacheLru source_cache(
                source_cache_budget,
                static_cast<size_t>(std::max(1, params.optimization.pseudo_view_top_k_source)));
            size_t permanent_mesh_bytes = 0;
            for (const auto* tensor : {
                     &prepared_mesh.world_vertices_xyz,
                     &prepared_mesh.triangle_indices,
                     &face_tensors.centers_cuda,
                     &face_tensors.normals_cuda}) {
                permanent_mesh_bytes = pseudo_view::saturating_add(
                    permanent_mesh_bytes,
                    pooled_tensor_bytes(*tensor));
            }
            size_t managed_gpu_peak_estimate_bytes = permanent_mesh_bytes;
            const auto update_managed_gpu_peak = [&](const size_t transient_bytes) {
                const size_t live_estimate = pseudo_view::saturating_add(
                    permanent_mesh_bytes,
                    pseudo_view::saturating_add(source_cache.live_bytes(), transient_bytes));
                managed_gpu_peak_estimate_bytes = std::max(
                    managed_gpu_peak_estimate_bytes,
                    live_estimate);
            };
            const int face_count = prepared_mesh.face_count;
            const Vec3 mesh_center = (bbox_min + bbox_max) * 0.5f;
            float scene_radius = length(bbox_max - bbox_min) * 0.5f;
            if (!(scene_radius > 1e-5f) || !std::isfinite(scene_radius)) {
                scene_radius = 1.0f;
            }

            const float sphere_cell_angle_deg = std::clamp(
                params.optimization.pseudo_view_sphere_cell_angle_deg,
                1.0f,
                90.0f);
            const float sphere_support_angle_deg = std::clamp(
                params.optimization.pseudo_view_sphere_support_angle_deg,
                0.0f,
                90.0f);
            const float sphere_support_cos = std::cos(sphere_support_angle_deg * PI_F / 180.0f);
            const float mesh_front_angle_deg = std::clamp(
                params.optimization.pseudo_view_mesh_front_angle_deg,
                0.0f,
                90.0f);
            const float mesh_front_cos = std::cos(mesh_front_angle_deg * PI_F / 180.0f);
            const float surface_front_angle_deg = std::clamp(
                params.optimization.pseudo_view_face_angle_max_deg,
                0.0f,
                90.0f);
            const float surface_front_cos = std::cos(surface_front_angle_deg * PI_F / 180.0f);
            const float source_rgb_face_angle_max_deg = std::clamp(
                params.optimization.pseudo_view_source_rgb_face_angle_max_deg,
                0.0f,
                180.0f);
            const float source_rgb_face_angle_min_cos =
                std::cos(source_rgb_face_angle_max_deg * PI_F / 180.0f);

            if (!params.optimization.pseudo_view_minimal_metadata) {
                LOG_INFO("Pseudo-view precompute: {} train cameras, {} mesh faces, sphere cell angle {:.1f} deg",
                         train_cameras.size(), face_count, sphere_cell_angle_deg);
            }

            std::vector<SphereCell> sphere_cells = build_sphere_cells(sphere_cell_angle_deg);
            stats.sphere_cells = sphere_cells.size();
            std::vector<PoseData> camera_poses(train_cameras.size());
            std::vector<float> camera_radii;
            camera_radii.reserve(camera_poses.size());

            for (size_t camera_index = 0; camera_index < train_cameras.size(); ++camera_index) {
                auto& camera = *train_cameras[camera_index];
                camera.load_image_size(params.dataset.resize_factor, params.dataset.max_width);
                camera_poses[camera_index] = pose_from_camera(camera);
                const float radius = distance(camera_poses[camera_index].C, mesh_center);
                if (radius > 1e-5f && std::isfinite(radius)) {
                    camera_radii.push_back(radius);
                }
                const int cell_id = nearest_sphere_cell(sphere_cells, camera_poses[camera_index].C - mesh_center);
                if (cell_id >= 0) {
                    auto& cell = sphere_cells[static_cast<size_t>(cell_id)];
                    if (!cell.occupied_by_real) {
                        stats.occupied_sphere_cells++;
                    }
                    cell.occupied_by_real = true;
                    cell.real_camera_count++;
                }
            }

            for (const auto& cell : sphere_cells) {
                if (!cell.occupied_by_real) {
                    stats.empty_sphere_cells++;
                }
            }

            const float radius_fallback = std::max(scene_radius, 1e-5f);
            float radius_q50 = median_sorted(camera_radii, radius_fallback);
            if (!(radius_q50 > 1e-5f) || !std::isfinite(radius_q50)) {
                radius_q50 = radius_fallback;
            }

            std::vector<CandidateRecord> candidates;
            for (const auto& cell : sphere_cells) {
                if (cell.occupied_by_real) {
                    continue;
                }
                const SurfaceTarget surface = surface_target_for_cell(faces, cell.direction, sphere_support_cos);
                if (!surface.valid) {
                    continue;
                }
                stats.supported_sphere_cells++;

                const int base_index = nearest_camera_for_direction(camera_poses, mesh_center, cell.direction);
                if (base_index < 0) {
                    continue;
                }
                const auto& base_pose = camera_poses[static_cast<size_t>(base_index)];
                const float radius = std::max(radius_q50, 1e-5f);
                const Vec3 candidate_center = mesh_center + cell.direction * radius;
                PoseData pseudo_pose = make_look_at_pose(base_pose, candidate_center, surface.centroid);
                const Vec3 forward = camera_forward_world(pseudo_pose);
                const float mesh_alignment = dot(forward, normalize(mesh_center - candidate_center, forward));
                if (mesh_alignment < mesh_front_cos) {
                    continue;
                }

                const float normal_alignment = dot(
                    surface.normal,
                    normalize(candidate_center - surface.centroid, cell.direction));
                if (normal_alignment < surface_front_cos) {
                    continue;
                }

                const float framing_score = projected_framing_score(pseudo_pose, surface.centroid);
                if (!(framing_score > 0.0f)) {
                    continue;
                }

                auto pseudo_camera = make_pseudo_camera(
                    pseudo_pose,
                    *train_cameras[static_cast<size_t>(base_index)],
                    std::format(
                        "pseudo_candidate_cell{}_r{:.3f}_{}.png",
                        cell.id,
                        radius,
                        sanitize_token(train_cameras[static_cast<size_t>(base_index)]->image_name())),
                    -1000000 - cell.id);
                const auto candidate_render_start = TimingClock::now();
                auto pseudo_render = render_mesh_supervision_targets_for_camera(
                    prepared_mesh,
                    pseudo_camera,
                    params,
                    MeshSupervisionOutputRequest::depth_and_triangle_id(),
                    false);
                add_timing(timing.candidate_render_ms, timing.candidate_render_count, candidate_render_start);
                if (!pseudo_render) {
                    if (!params.optimization.pseudo_view_minimal_metadata) {
                        LOG_WARN("Pseudo candidate render skipped: {}", pseudo_render.error());
                    }
                    continue;
                }
                const size_t candidate_job_estimate = pseudo_view::estimate_rgb_job_bytes(
                    pseudo_pose.width,
                    pseudo_pose.height,
                    prepared_mesh.vertex_count);
                update_managed_gpu_peak(candidate_job_estimate);
                auto front_mask_result = build_pseudo_front_face_mask(
                    pseudo_pose,
                    pseudo_render->triangle_id,
                    face_tensors.centers_cuda,
                    face_tensors.normals_cuda,
                    params.optimization.pseudo_view_rgb_face_angle_max_deg);
                if (!front_mask_result) {
                    if (!params.optimization.pseudo_view_minimal_metadata) {
                        LOG_WARN("Pseudo candidate front-face mask skipped: {}", front_mask_result.error());
                    }
                    continue;
                }
                auto front_mask = std::move(*front_mask_result);
                // Reject all-miss and all-backface candidates before selecting
                // sources so they never populate the source depth/mask cache.
                if (front_mask.mesh_pixels <= 0 || front_mask.front_face_pixels <= 0) {
                    stats.pseudo_candidates++;
                    stats.pseudo_views_unobservable++;
                    continue;
                }

                stats.pseudo_candidates++;
                auto source_result = choose_best_source(
                    pseudo_pose,
                    pseudo_render->depth,
                    pseudo_render->triangle_id,
                    &front_mask.mask_cuda,
                    prepared_mesh,
                    face_tensors.normals_cuda,
                    train_cameras,
                    params,
                    source_cache,
                    scene_radius,
                    source_rgb_face_angle_min_cos,
                    false,
                    std::nullopt,
                    &timing,
                    !params.optimization.pseudo_view_minimal_metadata);
                update_managed_gpu_peak(candidate_job_estimate);
                if (!source_result) {
                    stats.pseudo_views_unobservable++;
                    if (!params.optimization.pseudo_view_minimal_metadata) {
                        LOG_WARN("Pseudo candidate marked unobservable: {}", source_result.error());
                    }
                    continue;
                }
                auto source = std::move(*source_result);
                if (source.valid_ratio < params.optimization.pseudo_view_min_valid_ratio) {
                    stats.pseudo_views_unobservable++;
                    continue;
                }

                CandidateRecord record;
                record.sphere_cell_id = cell.id;
                record.sphere_cell_lat_deg = cell.lat_deg;
                record.sphere_cell_lon_deg = cell.lon_deg;
                record.sphere_cell_angle_deg = sphere_cell_angle_deg;
                record.base_camera_index = base_index;
                record.base_camera_uid = static_cast<uint32_t>(train_cameras[static_cast<size_t>(base_index)]->uid());
                record.pose = pseudo_pose;
                record.sphere_cell_direction = cell.direction;
                record.surface_target = surface.centroid;
                record.surface_normal = surface.normal;
                record.surface_area = surface.area;
                record.surface_face_count = surface.face_count;
                record.surface_mean_cos = surface.mean_cos;
                record.radius = radius;
                record.radius_scale = radius / std::max(radius_q50, 1e-6f);
                record.front_face_pixels = front_mask.front_face_pixels;
                record.backface_pixels = front_mask.backface_pixels;
                record.front_face_ratio = front_mask.mesh_pixels > 0
                                              ? static_cast<float>(front_mask.front_face_pixels) /
                                                    static_cast<float>(front_mask.mesh_pixels)
                                              : 0.0f;
                record.normal_alignment = clamp01(normal_alignment);
                record.framing_score = framing_score;
                record.score = source.valid_ratio;
                record.source = std::move(source);
                candidates.push_back(std::move(record));
            }

            std::sort(candidates.begin(), candidates.end(), [](const CandidateRecord& a, const CandidateRecord& b) {
                if (std::abs(a.source.valid_ratio - b.source.valid_ratio) > 1e-6f) {
                    return a.source.valid_ratio > b.source.valid_ratio;
                }
                if (std::abs(a.source.pose_distance - b.source.pose_distance) > 1e-6f) {
                    return a.source.pose_distance < b.source.pose_distance;
                }
                if (std::abs(a.normal_alignment - b.normal_alignment) > 1e-6f) {
                    return a.normal_alignment > b.normal_alignment;
                }
                return a.sphere_cell_id < b.sphere_cell_id;
            });

            faces.clear();
            faces.shrink_to_fit();

            const bool minimal_metadata = params.optimization.pseudo_view_minimal_metadata;
            nlohmann::json manifest = pseudo_view::make_minimal_manifest_root("empty");
            if (!minimal_metadata) {
                manifest["output_dir"] = lfs::core::path_to_utf8(stats.output_dir);
                manifest["config"] = {
                    {"pose_generation_mode", PSEUDO_VIEW_POSE_GENERATION_MODE},
                    {"sphere_cell_angle_deg", sphere_cell_angle_deg},
                    {"sphere_support_angle_deg", sphere_support_angle_deg},
                    {"sphere_support_min_cos", sphere_support_cos},
                    {"mesh_front_angle_deg", mesh_front_angle_deg},
                    {"surface_front_angle_deg", surface_front_angle_deg},
                    {"surface_front_min_cos", surface_front_cos},
                    {"cell_radius_mode", "median"},
                    {"radius_q50", radius_q50},
                    {"min_valid_ratio", params.optimization.pseudo_view_min_valid_ratio},
                    {"min_valid_pixels", params.optimization.pseudo_view_min_valid_pixels},
                    {"top_k_source", params.optimization.pseudo_view_top_k_source},
                    {"rgb_face_angle_max_deg", params.optimization.pseudo_view_rgb_face_angle_max_deg},
                    {"source_rgb_face_angle_max_deg", source_rgb_face_angle_max_deg},
                    {"mask_erode_pixels", params.optimization.pseudo_view_mask_erode_pixels},
                    {"supervision_mask_valid_open_pixels", params.optimization.pseudo_view_mask_valid_open_pixels},
                    {"supervision_mask_erode_pixels", params.optimization.pseudo_view_mask_erode_pixels},
                    {"supervision_mask_invalid_dilate_pixels", params.optimization.pseudo_view_mask_invalid_dilate_pixels},
                    {"rgb_parallel_jobs", params.optimization.pseudo_view_rgb_parallel_jobs},
                    {"source_cache_budget_bytes", source_cache_budget}};
                manifest["debug"] = {
                    {"coverage_views", nlohmann::json::array()},
                    {"coverage_debug_skipped_reason", "cell_based_pose_generation"}};
            }
            std::vector<int> final_source_indices;
            final_source_indices.reserve(candidates.size());
            for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
                const auto& candidate = candidates[candidate_index];
                const int source_index = candidate.source.source_index;
                if (source_index < 0 || source_index >= static_cast<int>(train_cameras.size())) {
                    return std::unexpected("Pseudo-view final output encountered an invalid source index");
                }
                final_source_indices.push_back(source_index);
            }
            const auto source_groups =
                pseudo_view::group_candidate_indices_by_source(final_source_indices);

            std::vector<std::optional<SaveCandidateOutputResult>> save_results(candidates.size());
            int maximum_effective_parallel_jobs = 0;
            for (const auto& source_group : source_groups) {
                const int source_index = source_group.source_index;
                SourceCache transient_source;
                SourceCache* selected_source = source_cache.find(source_index);
                if (!selected_source) {
                    const auto cache_start = TimingClock::now();
                    auto cache_result = build_source_cache(
                        prepared_mesh,
                        *train_cameras[static_cast<size_t>(source_index)],
                        params);
                    add_timing(timing.source_cache_build_ms, timing.source_cache_build_count, cache_start);
                    if (!cache_result) {
                        return std::unexpected(cache_result.error());
                    }
                    const size_t bytes = source_cache_bytes(*cache_result);
                    if (bytes <= source_cache.budget_bytes()) {
                        selected_source = source_cache.insert(
                            source_index,
                            std::move(*cache_result),
                            bytes);
                        if (!selected_source) {
                            return std::unexpected(
                                "Pseudo-view source cache failed to retain an in-budget final source");
                        }
                    } else {
                        transient_source = std::move(*cache_result);
                        selected_source = &transient_source;
                    }
                }

                const auto rgb_load_start = TimingClock::now();
                auto rgb_result = load_source_rgb(
                    *selected_source,
                    *train_cameras[static_cast<size_t>(source_index)],
                    params);
                if (!rgb_result) {
                    return std::unexpected(rgb_result.error());
                }
                auto source_rgb = std::move(*rgb_result);
                add_timing(timing.source_rgb_load_ms, timing.source_rgb_load_count, rgb_load_start);

                const auto& group = source_group.candidate_indices;
                size_t single_job_bytes = 0;
                for (const size_t candidate_index : group) {
                    const auto& pose = candidates[candidate_index].pose;
                    single_job_bytes = std::max(
                        single_job_bytes,
                        pseudo_view::estimate_rgb_job_bytes(
                            pose.width,
                            pose.height,
                            prepared_mesh.vertex_count));
                }
                auto free_vram_result = cuda_free_memory_bytes("RGB concurrency budgeting");
                if (!free_vram_result) {
                    return std::unexpected(free_vram_result.error());
                }
                const int effective_parallel_jobs = pseudo_view::effective_rgb_parallel_jobs(
                    params.optimization.pseudo_view_rgb_parallel_jobs,
                    group.size(),
                    *free_vram_result,
                    single_job_bytes);
                const size_t transient_source_bytes = selected_source == &transient_source
                                                          ? source_cache_bytes(transient_source)
                                                          : 0;
                const size_t parallel_job_bytes = pseudo_view::saturating_multiply(
                    single_job_bytes,
                    static_cast<size_t>(effective_parallel_jobs));
                update_managed_gpu_peak(pseudo_view::saturating_add(
                    transient_source_bytes,
                    pseudo_view::saturating_add(
                        pooled_tensor_bytes(source_rgb),
                        parallel_job_bytes)));
                maximum_effective_parallel_jobs = std::max(
                    maximum_effective_parallel_jobs,
                    effective_parallel_jobs);

                if (effective_parallel_jobs <= 1) {
                    for (const size_t candidate_index : group) {
                        auto save_result = save_candidate_outputs(
                            candidates[candidate_index],
                            static_cast<int>(candidate_index),
                            prepared_mesh,
                            face_tensors.centers_cuda,
                            face_tensors.normals_cuda,
                            train_cameras,
                            params,
                            *selected_source,
                            source_rgb,
                            scene_radius,
                            source_rgb_face_angle_max_deg,
                            source_rgb_face_angle_min_cos,
                            stats.output_dir);
                        if (!save_result) {
                            return std::unexpected(save_result.error());
                        }
                        save_results[candidate_index] = std::move(*save_result);
                    }
                } else {
                    for (size_t group_offset = 0; group_offset < group.size();) {
                        std::vector<std::future<std::pair<size_t, std::expected<SaveCandidateOutputResult, std::string>>>> futures;
                        futures.reserve(static_cast<size_t>(effective_parallel_jobs));
                        for (int job = 0;
                             job < effective_parallel_jobs && group_offset < group.size();
                             ++job, ++group_offset) {
                            const size_t candidate_index = group[group_offset];
                            futures.push_back(std::async(
                                std::launch::async,
                                [&, candidate_index, selected_source]() {
                                    return std::make_pair(
                                        candidate_index,
                                        save_candidate_outputs(
                                            candidates[candidate_index],
                                            static_cast<int>(candidate_index),
                                            prepared_mesh,
                                            face_tensors.centers_cuda,
                                            face_tensors.normals_cuda,
                                            train_cameras,
                                            params,
                                            *selected_source,
                                            source_rgb,
                                            scene_radius,
                                            source_rgb_face_angle_max_deg,
                                            source_rgb_face_angle_min_cos,
                                            stats.output_dir));
                                }));
                        }
                        for (auto& future : futures) {
                            auto [candidate_index, save_result] = future.get();
                            if (!save_result) {
                                return std::unexpected(save_result.error());
                            }
                            save_results[candidate_index] = std::move(*save_result);
                        }
                    }
                }

                const cudaError_t group_sync_error = cudaDeviceSynchronize();
                if (group_sync_error != cudaSuccess) {
                    return std::unexpected(std::format(
                        "Pseudo-view source group {} synchronization failed: {}",
                        source_index,
                        cudaGetErrorString(group_sync_error)));
                }
                // RGB ownership is intentionally local to this source group.
                source_rgb = lfs::core::Tensor();
            }

            if (!minimal_metadata) {
                manifest["config"]["rgb_parallel_jobs_effective_max"] = maximum_effective_parallel_jobs;
                manifest["rejected_views"] = nlohmann::json::array();
            }
            for (size_t candidate_index = 0; candidate_index < save_results.size(); ++candidate_index) {
                auto& save_result = save_results[candidate_index];
                if (!save_result.has_value()) {
                    return std::unexpected("Pseudo-view final output failed to produce a save result");
                }
                merge_timing(timing, save_result->timing);
                if (save_result->accepted) {
                    manifest["views"].push_back(std::move(save_result->entry));
                    stats.pseudo_views_written++;
                } else {
                    stats.pseudo_views_unobservable++;
                    if (!minimal_metadata) {
                        manifest["rejected_views"].push_back({
                            {"pseudo_id", static_cast<int>(candidate_index)},
                            {"base_camera_uid", candidates[candidate_index].base_camera_uid},
                            {"reason", save_result->rejection_reason},
                            {"mesh_valid_pixels", save_result->mesh_valid_pixels},
                            {"raw_valid_pixels", save_result->raw_valid_pixels},
                            {"supervision_valid_pixels", save_result->supervision_valid_pixels},
                            {"post_morph_valid_ratio", save_result->post_morph_valid_ratio}});
                    }
                }
            }
            manifest["status"] = stats.pseudo_views_written > 0 ? "ok" : "empty";
            const size_t source_cache_peak_bytes = source_cache.peak_bytes();
            const size_t source_cache_evictions = source_cache.eviction_count();

            if (!minimal_metadata) {
                manifest["stats"] = {
                    {"total_train_cameras", stats.total_train_cameras},
                    {"sphere_cells", stats.sphere_cells},
                    {"occupied_sphere_cells", stats.occupied_sphere_cells},
                    {"empty_sphere_cells", stats.empty_sphere_cells},
                    {"supported_sphere_cells", stats.supported_sphere_cells},
                    {"pseudo_candidates", stats.pseudo_candidates},
                    {"accepted_cell_candidates", candidates.size()},
                    {"pseudo_views_written", stats.pseudo_views_written},
                    {"pseudo_views_unobservable", stats.pseudo_views_unobservable},
                    {"source_cache_peak_bytes", source_cache_peak_bytes},
                    {"source_cache_evictions", source_cache_evictions},
                    {"managed_gpu_memory_peak_estimate_bytes", managed_gpu_peak_estimate_bytes},
                    {"radius_q50", radius_q50},
                    {"scene_radius", scene_radius},
                    {"mesh_center", vec3_json(mesh_center)}};
                manifest["timing"] = timing_json(timing);
            }

            if (auto write_manifest = write_json_file(stats.manifest_path, manifest); !write_manifest) {
                return std::unexpected(write_manifest.error());
            }

            const cudaError_t pre_release_sync_error = cudaDeviceSynchronize();
            if (pre_release_sync_error != cudaSuccess) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute final CUDA synchronization failed: {}",
                    cudaGetErrorString(pre_release_sync_error)));
            }

            source_cache.clear();
            face_tensors.centers_cuda = lfs::core::Tensor();
            face_tensors.normals_cuda = lfs::core::Tensor();
            prepared_mesh.world_vertices_xyz = lfs::core::Tensor();
            prepared_mesh.world_vertex_normals_xyz = lfs::core::Tensor();
            prepared_mesh.triangle_indices = lfs::core::Tensor();
            lfs::core::Tensor::trim_memory_pool();

            const cudaError_t post_release_sync_error = cudaDeviceSynchronize();
            if (post_release_sync_error != cudaSuccess) {
                return std::unexpected(std::format(
                    "Pseudo-view precompute CUDA cleanup failed: {}",
                    cudaGetErrorString(post_release_sync_error)));
            }

            const double source_cache_peak_mib =
                static_cast<double>(source_cache_peak_bytes) /
                static_cast<double>(pseudo_view::MIB);
            const double managed_gpu_peak_estimate_mib =
                static_cast<double>(managed_gpu_peak_estimate_bytes) /
                static_cast<double>(pseudo_view::MIB);
            if (minimal_metadata) {
                LOG_INFO("Pseudo-view precompute complete: {} accepted, {} rejected, managed GPU memory peak estimate {:.1f} MiB",
                         stats.pseudo_views_written,
                         stats.pseudo_views_unobservable,
                         managed_gpu_peak_estimate_mib);
            } else {
                LOG_INFO("Pseudo-view precompute complete: {} written, {} unobservable, source cache peak {:.1f} MiB, managed GPU peak estimate {:.1f} MiB, manifest {}",
                         stats.pseudo_views_written,
                         stats.pseudo_views_unobservable,
                         source_cache_peak_mib,
                         managed_gpu_peak_estimate_mib,
                         lfs::core::path_to_utf8(stats.manifest_path));
            }
            return stats;
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Pseudo-view precompute failed: ") + e.what());
        }
    }

} // namespace lfs::training
