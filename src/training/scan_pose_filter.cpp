/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scan_pose_filter.hpp"

#include "core/tensor.hpp"
#include "kernels/scan_pose_zncc.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lfs::training {
    namespace {

        // These implementation constants are deliberately fixed in v1.  The
        // command exposes only the final ZNCC threshold; patch geometry and
        // photometric support are evaluated first and represented in the
        // compact patch scores below.
        constexpr int kMaxWorkingDimension = 512;
        constexpr int kRegionCellSize = 32;
        constexpr int kPatchSize = 11;
        // Each approximately 32x32 cell contributes a complete 11x11 patch.
        constexpr int kSamplesPerAxis = kPatchSize;
        constexpr int kMinValidSamplesPerRegion = 24;
        constexpr int kPeerCount = 4;
        constexpr int kCacheCapacity = kPeerCount + 2;
        constexpr float kRelativeDepthEpsilon = 0.01f;
        constexpr float kVarianceEpsilon = 1.0e-6f;

        struct CameraInfo {
            std::shared_ptr<lfs::core::Camera> camera;
            int width = 0;
            int height = 0;
            float fx = 0.0f;
            float fy = 0.0f;
            float cx = 0.0f;
            float cy = 0.0f;
            std::array<float, 16> c2w{};
            lfs::core::Tensor c2w_cuda;
        };

        struct WorkFrame {
            lfs::core::Tensor rgb;
            lfs::core::Tensor depth;
        };

        struct CacheEntry {
            WorkFrame work;
            std::list<std::size_t>::iterator lru_position;
        };

        class CameraOverrideGuard {
        public:
            ~CameraOverrideGuard() {
                for (const auto& entry : cameras_) {
                    if (entry.camera) {
                        entry.camera->restore_intrinsics();
                        entry.camera->set_image_dimensions(entry.width, entry.height);
                    }
                }
            }

            void add(lfs::core::Camera& camera) {
                cameras_.push_back({&camera, camera.image_width(), camera.image_height()});
            }

        private:
            struct Entry {
                lfs::core::Camera* camera = nullptr;
                int width = 0;
                int height = 0;
            };
            std::vector<Entry> cameras_;
        };

        std::array<float, 16> read_w2c(const lfs::core::Camera& camera) {
            auto transform = camera.world_view_transform()
                                 .squeeze(0)
                                 .to(lfs::core::Device::CPU)
                                 .to(lfs::core::DataType::Float32)
                                 .contiguous();
            if (!transform.is_valid() || transform.ndim() != 2 ||
                transform.shape()[0] != 4 || transform.shape()[1] != 4) {
                throw std::runtime_error("scan pose filter: camera transform must be a 4x4 matrix");
            }

            std::array<float, 16> result{};
            const float* ptr = transform.ptr<float>();
            if (!ptr) {
                throw std::runtime_error("scan pose filter: camera transform has no data");
            }
            std::copy_n(ptr, result.size(), result.begin());
            for (const float value : result) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("scan pose filter: camera transform contains non-finite values");
                }
            }
            return result;
        }

        std::array<float, 16> rigid_inverse(const std::array<float, 16>& w2c) {
            std::array<float, 16> c2w{};
            // R^-1 = R^T. Scanner poses are rigid c2w transforms, and Camera
            // stores their inverse w2c form. The app-level scanner-bin adapter
            // has already converted translations into mesh working units.
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    c2w[row * 4 + col] = w2c[col * 4 + row];
                }
            }
            for (int row = 0; row < 3; ++row) {
                c2w[row * 4 + 3] = -(
                    c2w[row * 4 + 0] * w2c[3] +
                    c2w[row * 4 + 1] * w2c[7] +
                    c2w[row * 4 + 2] * w2c[11]);
            }
            c2w[12] = 0.0f;
            c2w[13] = 0.0f;
            c2w[14] = 0.0f;
            c2w[15] = 1.0f;
            return c2w;
        }

        float camera_translation_distance_sq(
            const CameraInfo& lhs,
            const CameraInfo& rhs) {
            const float dx = lhs.c2w[3] - rhs.c2w[3];
            const float dy = lhs.c2w[7] - rhs.c2w[7];
            const float dz = lhs.c2w[11] - rhs.c2w[11];
            return dx * dx + dy * dy + dz * dz;
        }

        float median(std::vector<float> values) {
            if (values.empty()) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            std::sort(values.begin(), values.end());
            const std::size_t middle = values.size() / 2;
            if ((values.size() & 1u) != 0u) {
                return values[middle];
            }
            return 0.5f * (values[middle - 1] + values[middle]);
        }

        void validate_rgb(const lfs::core::Tensor& rgb, const CameraInfo& info) {
            if (!rgb.is_valid() || rgb.is_empty() || rgb.ndim() != 3 ||
                rgb.shape()[0] < 3 || rgb.shape()[1] != static_cast<size_t>(info.height) ||
                rgb.shape()[2] != static_cast<size_t>(info.width)) {
                throw std::runtime_error(std::format(
                    "scan pose filter: RGB tensor shape does not match camera (expected >=3x{}x{}, got rank {})",
                    info.height,
                    info.width,
                    rgb.ndim()));
            }
            if (rgb.device() != lfs::core::Device::CUDA ||
                rgb.dtype() != lfs::core::DataType::Float32 ||
                !rgb.is_contiguous()) {
                throw std::runtime_error("scan pose filter: RGB loader returned a non-contiguous CUDA float32 tensor");
            }
        }

        WorkFrame load_work_frame(
            const CameraInfo& info,
            const PreparedMesh& prepared_mesh,
            const lfs::core::param::TrainingParameters& render_params) {

            auto rgb = info.camera->load_and_get_image(-1, kMaxWorkingDimension)
                           .to(lfs::core::Device::CUDA)
                           .to(lfs::core::DataType::Float32)
                           .contiguous();
            validate_rgb(rgb, info);

            auto targets = render_mesh_supervision_targets_for_camera(
                prepared_mesh,
                *info.camera,
                render_params,
                MeshSupervisionOutputRequest::depth_only(),
                false);
            if (!targets) {
                throw std::runtime_error(targets.error());
            }
            if (!targets->depth.is_valid() || targets->depth.is_empty() ||
                targets->depth.ndim() != 3 || targets->depth.shape()[0] != 1 ||
                targets->depth.shape()[1] != static_cast<size_t>(info.height) ||
                targets->depth.shape()[2] != static_cast<size_t>(info.width)) {
                throw std::runtime_error("scan pose filter: mesh depth shape does not match RGB resolution");
            }

            WorkFrame result;
            result.rgb = std::move(rgb);
            result.depth = std::move(targets->depth).contiguous();
            if (!result.depth.is_contiguous()) {
                throw std::runtime_error("scan pose filter: mesh depth tensor is not contiguous");
            }
            return result;
        }

        std::optional<ScanPoseFilterPairScore> score_pair(
            const CameraInfo& target_info,
            const WorkFrame& target,
            const CameraInfo& source_info,
            const WorkFrame& source) {

            const int region_count_x =
                (target_info.width + kRegionCellSize - 1) / kRegionCellSize;
            const int region_count_y =
                (target_info.height + kRegionCellSize - 1) / kRegionCellSize;
            const int region_count = region_count_x * region_count_y;

            auto region_scores = lfs::core::Tensor::empty(
                {static_cast<size_t>(region_count)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            // Keep all compact integer diagnostics in one contiguous tensor so
            // each pair needs one statistics transfer back to the CPU.
            constexpr std::size_t kRegionStatisticCount = 4;
            auto region_statistics = lfs::core::Tensor::empty(
                {kRegionStatisticCount, static_cast<size_t>(region_count)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Int32);
            int32_t* statistics_ptr = region_statistics.ptr<int32_t>();
            float* scores_device_ptr = region_scores.ptr<float>();
            if (!statistics_ptr || !scores_device_ptr) {
                throw std::runtime_error("scan pose filter: failed to allocate ZNCC result buffers");
            }
            int32_t* region_photometric_samples = statistics_ptr;
            int32_t* region_target_samples = statistics_ptr + region_count;
            int32_t* region_projected_samples = statistics_ptr + 2 * region_count;
            int32_t* region_depth_consistent_samples = statistics_ptr + 3 * region_count;

            kernels::launch_scan_pose_zncc_regions(
                target.rgb.ptr<float>(),
                target.depth.ptr<float>(),
                target_info.c2w_cuda.ptr<float>(),
                target_info.width,
                target_info.height,
                target_info.fx,
                target_info.fy,
                target_info.cx,
                target_info.cy,
                source.rgb.ptr<float>(),
                source.depth.ptr<float>(),
                source_info.camera->world_view_transform_ptr(),
                source_info.width,
                source_info.height,
                source_info.fx,
                source_info.fy,
                source_info.cx,
                source_info.cy,
                region_count_x,
                region_count_y,
                kSamplesPerAxis,
                kMinValidSamplesPerRegion,
                kRelativeDepthEpsilon,
                kVarianceEpsilon,
                scores_device_ptr,
                region_photometric_samples,
                region_scores.stream(),
                region_target_samples,
                region_projected_samples,
                region_depth_consistent_samples);

            const cudaError_t launch_error = cudaGetLastError();
            if (launch_error != cudaSuccess) {
                throw std::runtime_error(std::format(
                    "scan pose filter: ZNCC CUDA launch failed: {}",
                    cudaGetErrorString(launch_error)));
            }

            auto scores_cpu = region_scores.to(lfs::core::Device::CPU)
                                  .to(lfs::core::DataType::Float32)
                                  .contiguous();
            auto statistics_cpu = region_statistics.to(lfs::core::Device::CPU)
                                      .to(lfs::core::DataType::Int32)
                                      .contiguous();
            const float* scores_ptr = scores_cpu.ptr<float>();
            const int32_t* statistics_cpu_ptr = statistics_cpu.ptr<int32_t>();
            if (!statistics_cpu_ptr) {
                throw std::runtime_error("scan pose filter: failed to read ZNCC sample statistics");
            }
            const int32_t* photometric_ptr = statistics_cpu_ptr;
            const int32_t* target_ptr = statistics_cpu_ptr + region_count;
            const int32_t* projected_ptr = statistics_cpu_ptr + 2 * region_count;
            const int32_t* depth_ptr = statistics_cpu_ptr + 3 * region_count;
            if (!scores_ptr || !photometric_ptr || !target_ptr || !projected_ptr || !depth_ptr) {
                throw std::runtime_error("scan pose filter: failed to read ZNCC region results");
            }

            std::vector<float> effective_scores;
            std::vector<float> photometric_scores;
            effective_scores.reserve(static_cast<size_t>(region_count));
            photometric_scores.reserve(static_cast<size_t>(region_count));
            std::size_t valid_samples = 0;
            std::size_t target_samples = 0;
            std::size_t projected_samples = 0;
            std::size_t depth_consistent_samples = 0;
            for (int region = 0; region < region_count; ++region) {
                valid_samples += static_cast<std::size_t>(std::max(0, photometric_ptr[region]));
                target_samples += static_cast<std::size_t>(std::max(0, target_ptr[region]));
                projected_samples += static_cast<std::size_t>(std::max(0, projected_ptr[region]));
                depth_consistent_samples += static_cast<std::size_t>(std::max(0, depth_ptr[region]));

                if (target_ptr[region] < kMinValidSamplesPerRegion) {
                    // The target mesh projection is the validity mask. Patch
                    // cells with no sufficient mesh-supported area are
                    // background and must not contribute a synthetic score.
                    continue;
                }

                // Only RGB correspondence and ZNCC determine the score. Target
                // mesh depth is used to establish the 3D correspondence, but
                // source mesh depth is not an additional rejection criterion.
                // Samples outside the target mesh mask or outside the source
                // image are simply excluded from this patch's ZNCC.
                const bool enough_photo_samples =
                    photometric_ptr[region] >= kMinValidSamplesPerRegion;
                const bool photo_score_valid = std::isfinite(scores_ptr[region]);
                if (enough_photo_samples && photo_score_valid) {
                    photometric_scores.push_back(scores_ptr[region]);
                    effective_scores.push_back(scores_ptr[region]);
                }
            }
            ScanPoseFilterPairScore result;
            result.valid_samples = valid_samples;
            result.target_samples = target_samples;
            result.projected_samples = projected_samples;
            result.depth_consistent_samples = depth_consistent_samples;
            if (effective_scores.empty()) {
                // No target-mask-supported patch produced a valid RGB ZNCC.
                // Omit this pair rather than penalizing the frame with a
                // synthetic -1 score from background or out-of-view pixels.
                return std::nullopt;
            }

            result.informative_regions = effective_scores.size();
            result.zncc = median(std::move(effective_scores));
            if (!photometric_scores.empty()) {
                result.photometric_zncc = median(std::move(photometric_scores));
            }
            return result;
        }

    } // namespace

    std::expected<ScanPoseFilterResult, std::string> score_scan_pose_frames_zncc(
        const PreparedMesh& prepared_mesh,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const float zncc_threshold) {

        if (!std::isfinite(zncc_threshold) || zncc_threshold < -1.0f || zncc_threshold > 1.0f) {
            return std::unexpected("scan pose filter: ZNCC threshold must be finite and within [-1, 1]");
        }
        if (!prepared_mesh.is_valid()) {
            return std::unexpected("scan pose filter: prepared mesh is empty or invalid");
        }
        if (cameras.empty()) {
            return std::unexpected("scan pose filter: no scanner cameras were provided");
        }

        try {
            CameraOverrideGuard override_guard;
            std::vector<CameraInfo> infos;
            infos.reserve(cameras.size());
            for (const auto& camera : cameras) {
                if (!camera) {
                    return std::unexpected("scan pose filter: camera list contains a null camera");
                }
                if (camera->has_pyramid_intrinsics_override()) {
                    return std::unexpected(
                        "scan pose filter: cameras must not have a pre-existing pyramid intrinsics override");
                }

                override_guard.add(*camera);
                const int source_width = camera->camera_width();
                const int source_height = camera->camera_height();
                if (source_width <= 0 || source_height <= 0) {
                    return std::unexpected("scan pose filter: camera has invalid source image dimensions");
                }
                const auto [source_fx, source_fy, source_cx, source_cy] = camera->get_intrinsics();
                camera->load_image_size(-1, kMaxWorkingDimension);
                const int width = camera->image_width();
                const int height = camera->image_height();
                if (width < kPatchSize || height < kPatchSize || width <= 0 || height <= 0) {
                    return std::unexpected("scan pose filter: camera image is too small for sparse region sampling");
                }
                const float scale_x = static_cast<float>(width) / static_cast<float>(source_width);
                const float scale_y = static_cast<float>(height) / static_cast<float>(source_height);
                const float fx = source_fx * scale_x;
                const float fy = source_fy * scale_y;
                const float cx = source_cx * scale_x;
                const float cy = source_cy * scale_y;
                if (!std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.0f || fy <= 0.0f ||
                    !std::isfinite(cx) || !std::isfinite(cy)) {
                    return std::unexpected("scan pose filter: camera has invalid pinhole intrinsics");
                }

                const auto w2c = read_w2c(*camera);
                CameraInfo info;
                info.camera = camera;
                info.width = width;
                info.height = height;
                info.fx = fx;
                info.fy = fy;
                info.cx = cx;
                info.cy = cy;
                info.c2w = rigid_inverse(w2c);
                info.c2w_cuda = lfs::core::Tensor::from_vector(
                    std::vector<float>(info.c2w.begin(), info.c2w.end()),
                    {size_t{4}, size_t{4}},
                    lfs::core::Device::CUDA);
                infos.push_back(std::move(info));
            }

            for (const auto& info : infos) {
                info.camera->set_intrinsics_for_pyramid(
                    info.fx,
                    info.fy,
                    info.cx,
                    info.cy,
                    info.width,
                    info.height);
            }

            lfs::core::param::TrainingParameters render_params;
            render_params.dataset.resize_factor = -1;
            render_params.dataset.max_width = kMaxWorkingDimension;

            std::unordered_map<std::size_t, CacheEntry> cache;
            std::list<std::size_t> lru;
            auto get_work_frame = [&](const std::size_t index) -> WorkFrame& {
                if (auto it = cache.find(index); it != cache.end()) {
                    lru.erase(it->second.lru_position);
                    lru.push_front(index);
                    it->second.lru_position = lru.begin();
                    return it->second.work;
                }

                if (cache.size() >= static_cast<size_t>(kCacheCapacity)) {
                    const std::size_t evicted = lru.back();
                    lru.pop_back();
                    cache.erase(evicted);
                }

                auto work = load_work_frame(infos[index], prepared_mesh, render_params);
                lru.push_front(index);
                auto [it, inserted] = cache.emplace(
                    index,
                    CacheEntry{std::move(work), lru.begin()});
                if (!inserted) {
                    throw std::runtime_error("scan pose filter: duplicate work-frame cache insertion");
                }
                return it->second.work;
            };

            ScanPoseFilterResult result;
            result.frames.resize(infos.size());
            for (std::size_t target_index = 0; target_index < infos.size(); ++target_index) {
                auto& frame_result = result.frames[target_index];
                WorkFrame& target_work = get_work_frame(target_index);

                // Select a fixed number of peers by camera-center translation
                // distance. This changes only which pairs are evaluated; no
                // distance threshold or trajectory-continuity score is used.
                std::vector<std::pair<float, std::size_t>> peer_candidates;
                peer_candidates.reserve(infos.size() - 1);
                for (std::size_t peer_index = 0; peer_index < infos.size(); ++peer_index) {
                    if (peer_index == target_index) {
                        continue;
                    }
                    const float distance_sq = camera_translation_distance_sq(
                        infos[target_index],
                        infos[peer_index]);
                    if (std::isfinite(distance_sq)) {
                        peer_candidates.emplace_back(distance_sq, peer_index);
                    }
                }
                const auto by_distance = [](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first) {
                        return lhs.first < rhs.first;
                    }
                    return lhs.second < rhs.second;
                };
                const std::size_t peer_count = std::min(
                    static_cast<std::size_t>(kPeerCount),
                    peer_candidates.size());
                if (peer_candidates.size() > peer_count) {
                    std::partial_sort(
                        peer_candidates.begin(),
                        peer_candidates.begin() + static_cast<std::ptrdiff_t>(peer_count),
                        peer_candidates.end(),
                        by_distance);
                } else {
                    std::sort(peer_candidates.begin(), peer_candidates.end(), by_distance);
                }

                for (std::size_t peer_slot = 0; peer_slot < peer_count; ++peer_slot) {
                    const std::size_t peer_index = peer_candidates[peer_slot].second;
                    WorkFrame& source_work = get_work_frame(peer_index);
                    auto pair = score_pair(
                        infos[target_index],
                        target_work,
                        infos[peer_index],
                        source_work);
                    if (!pair) {
                        continue;
                    }
                    pair->peer_index = peer_index;
                    frame_result.pair_scores.push_back(std::move(*pair));
                }

                if (frame_result.pair_scores.empty()) {
                    // No neighboring pair was available at all; without a
                    // finite pair score the frame remains uncertain.
                    frame_result.keep = true;
                    continue;
                }

                std::vector<float> pair_scores;
                std::vector<float> photometric_pair_scores;
                pair_scores.reserve(frame_result.pair_scores.size());
                photometric_pair_scores.reserve(frame_result.pair_scores.size());
                for (const auto& pair : frame_result.pair_scores) {
                    if (std::isfinite(pair.zncc)) {
                        pair_scores.push_back(pair.zncc);
                    }
                    if (pair.photometric_zncc && std::isfinite(*pair.photometric_zncc)) {
                        photometric_pair_scores.push_back(*pair.photometric_zncc);
                    }
                }
                if (pair_scores.empty()) {
                    frame_result.keep = true;
                    continue;
                }

                frame_result.zncc = median(std::move(pair_scores));
                if (!photometric_pair_scores.empty()) {
                    frame_result.photometric_zncc = median(std::move(photometric_pair_scores));
                }
                // This is the sole frame-level rejection comparison in v1.
                frame_result.keep = keep_scan_pose_frame_by_zncc(
                    frame_result.zncc,
                    zncc_threshold);
                if (frame_result.keep) {
                    ++result.kept_count;
                } else {
                    ++result.dropped_count;
                }
            }

            // Frames retained as uncertain were not counted in the branch above.
            result.kept_count = result.frames.size() - result.dropped_count;
            return result;
        } catch (const std::exception& error) {
            return std::unexpected(std::format("scan pose filter failed: {}", error.what()));
        }
    }

} // namespace lfs::training
