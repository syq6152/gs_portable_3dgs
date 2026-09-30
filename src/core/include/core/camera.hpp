/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "core/cuda/undistort/undistort.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <future>
#include <string>
#include <utility>

namespace lfs::core {

    class LFS_CORE_API Camera {
    public:
        Camera() = default;

        Camera(const Tensor& R,
               const Tensor& T,
               float focal_x, float focal_y,
               float center_x, float center_y,
               Tensor radial_distortion,
               Tensor tangential_distortion,
               CameraModelType camera_model_type,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               const std::filesystem::path& mask_path,
               int camera_width, int camera_height,
               int uid,
               int camera_id = 0);
        Camera(const Camera&, const Tensor& transform);

        // Destructor to clean up CUDA stream
        ~Camera();

        // Delete copy, define proper move semantics
        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
        Camera(Camera&& other) noexcept;
        Camera& operator=(Camera&& other) noexcept;

        // Initialize GPU tensors on demand
        void initialize_cuda_tensors();

        // Load image from disk and return it
        Tensor load_and_get_image(int resize_factor = -1, int max_width = 3840);

        // Load mask from disk, process it, and return it (cached)
        Tensor load_and_get_mask(int resize_factor = -1, int max_width = 3840,
                                 bool invert_mask = false, float mask_threshold = 0.5f);

        // Clear cached mask tensor to release long-lived GPU memory when callers only
        // need transient mask access (e.g. mesh GT precompute).
        void clear_cached_mask() noexcept;

        // Load image from disk just to populate _image_width/_image_height
        void load_image_size(int resize_factor = -1, int max_width = 3840);

        // Get number of bytes in the image file
        size_t get_num_bytes_from_file(int resize_factor = -1, int max_width = 3840) const;
        size_t get_num_bytes_from_file() const;

        // Accessors - now return const references to avoid copies
        const Tensor& world_view_transform() const {
            return _world_view_transform;
        }
        const Tensor& cam_position() const {
            return _cam_position;
        }

        // Direct GPU pointer access (tensors are already contiguous on CUDA)
        const float* world_view_transform_ptr() const {
            return _world_view_transform.ptr<float>();
        }
        const float* cam_position_ptr() const {
            return _cam_position.ptr<float>();
        }

        const Tensor& R() const { return _R; }
        const Tensor& T() const { return _T; }
        void set_world_to_camera_pose(const Tensor& R, const Tensor& T);
        void set_world_to_camera_pose_cpu(const std::array<float, 9>& R, const std::array<float, 3>& T);

        Tensor K() const;

        std::tuple<float, float, float, float> get_intrinsics() const;

        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        void set_image_dimensions(int width, int height) noexcept {
            _image_width = width;
            _image_height = height;
        }

        /// Temporarily override intrinsics for pyramid training (coarse-to-fine).
        /// Input intrinsics are effective values for the provided image size.
        /// Saves original values so they can be restored with restore_intrinsics().
        void set_intrinsics_for_pyramid(float fx, float fy, float cx, float cy,
                                        int width, int height) noexcept {
            if (!_intrinsics_saved) {
                _saved_focal_x = _focal_x;
                _saved_focal_y = _focal_y;
                _saved_center_x = _center_x;
                _saved_center_y = _center_y;
                _saved_image_width = _image_width;
                _saved_image_height = _image_height;
                _intrinsics_saved = true;
            }

            _focal_x = fx;
            _focal_y = fy;
            _center_x = cx;
            _center_y = cy;
            _image_width = width;
            _image_height = height;
        }

        /// Restore original intrinsics after pyramid override.
        void restore_intrinsics() noexcept {
            if (_intrinsics_saved) {
                _focal_x = _saved_focal_x;
                _focal_y = _saved_focal_y;
                _center_x = _saved_center_x;
                _center_y = _saved_center_y;
                _image_width = _saved_image_width;
                _image_height = _saved_image_height;
                _intrinsics_saved = false;
            }
        }
        bool has_pyramid_intrinsics_override() const noexcept { return _intrinsics_saved; }
        int camera_height() const noexcept { return _camera_height; }
        int camera_width() const noexcept { return _camera_width; }
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        float center_x() const noexcept { return _center_x; }
        float center_y() const noexcept { return _center_y; }
        Tensor radial_distortion() const noexcept { return _radial_distortion; }
        Tensor tangential_distortion() const noexcept { return _tangential_distortion; }
        CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        const std::filesystem::path& image_path() const noexcept { return _image_path; }
        const std::filesystem::path& mask_path() const noexcept { return _mask_path; }
        bool has_mask() const noexcept { return !_mask_path.empty() && std::filesystem::exists(_mask_path); }
        // Update the on-disk mask path for this camera (e.g. when masks are
        // synthesized at training start by the project_mesh mask mode). Clears
        // any cached processed mask so the new file is picked up on next load.
        void set_mask_path(std::filesystem::path path) {
            _mask_path = std::move(path);
            clear_cached_mask();
        }
        bool has_alpha() const noexcept { return _has_alpha; }
        void set_has_alpha(bool v) noexcept { _has_alpha = v; }
        int uid() const noexcept { return _uid; }
        int camera_id() const noexcept { return _camera_id; }

        // Pseudo-view supervision tagging. Pseudo cameras are synthesized by the
        // pseudo-view precompute pipeline and contribute to training with a reduced
        // loss weight so they cannot dominate real views.
        bool is_pseudo() const noexcept { return _is_pseudo; }
        float supervision_weight() const noexcept { return _supervision_weight; }
        void set_pseudo(bool is_pseudo, float weight = 1.0f) noexcept {
            _is_pseudo = is_pseudo;
            _supervision_weight = weight;
        }
        int pseudo_base_camera_uid() const noexcept { return _pseudo_base_camera_uid; }
        void set_pseudo_base_camera_uid(int uid) noexcept { _pseudo_base_camera_uid = uid; }

        // Stable identity of the real frame whose RGB was reprojected to build
        // this pseudo view. This is intentionally separate from
        // pseudo_base_camera_uid (pose following) and camera_id() (the camera
        // template used by the loader). The UID is only a same-run diagnostic;
        // image_name + camera_id form the persistent source key.
        [[nodiscard]] bool has_pseudo_rgb_source() const noexcept {
            return !_pseudo_rgb_source_image_name.empty() && _pseudo_rgb_source_camera_id != -1;
        }
        const std::string& pseudo_rgb_source_image_name() const noexcept {
            return _pseudo_rgb_source_image_name;
        }
        int pseudo_rgb_source_camera_id() const noexcept { return _pseudo_rgb_source_camera_id; }
        int pseudo_rgb_source_uid() const noexcept { return _pseudo_rgb_source_uid; }
        void set_pseudo_rgb_source(std::string image_name, int camera_id, int uid = -1) {
            _pseudo_rgb_source_image_name = std::move(image_name);
            _pseudo_rgb_source_camera_id = camera_id;
            _pseudo_rgb_source_uid = uid;
        }

        float FoVx() const noexcept { return _FoVx; }
        float FoVy() const noexcept { return _FoVy; }

        void precompute_undistortion(float blank_pixels = 0.0f);
        bool is_undistort_precomputed() const noexcept { return _undistort_precomputed; }
        void prepare_undistortion(float blank_pixels = 0.0f);
        bool is_undistort_prepared() const noexcept { return _undistort_prepared; }
        bool has_distortion() const noexcept;
        const UndistortParams& undistort_params() const noexcept { return _undistort_params; }

    private:
        // IDs
        float _FoVx = 0.f;
        float _FoVy = 0.f;
        int _uid = -1;
        int _camera_id = 0;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        // redundancy with _world_view_transform, but save calculation and passing from GPU 2 CPU
        Tensor _R;
        Tensor _T;

        Tensor _radial_distortion;
        Tensor _tangential_distortion;
        CameraModelType _camera_model_type = CameraModelType::PINHOLE;

        // Image info
        std::string _image_name;
        std::filesystem::path _image_path;
        std::filesystem::path _mask_path;
        bool _has_alpha = false;
        int _camera_width = 0;
        int _camera_height = 0;
        int _image_width = 0;
        int _image_height = 0;

        // GPU tensors (computed on demand)
        Tensor _world_view_transform;
        Tensor _cam_position;

        // Mask caching (processed mask stored on GPU)
        Tensor _cached_mask;
        bool _mask_loaded = false;

        // Undistortion state
        bool _undistort_precomputed = false;
        bool _undistort_prepared = false;
        UndistortParams _undistort_params{};

        // CUDA stream for async operations
        cudaStream_t _stream = nullptr;

        // Pseudo-view tagging (default = real training camera)
        bool _is_pseudo = false;
        float _supervision_weight = 1.0f;
        int _pseudo_base_camera_uid = -1;
        std::string _pseudo_rgb_source_image_name;
        int _pseudo_rgb_source_camera_id = -1;
        int _pseudo_rgb_source_uid = -1;

        // Saved intrinsics for pyramid training (restore after pyramid override)
        bool _intrinsics_saved = false;
        float _saved_focal_x = 0.f;
        float _saved_focal_y = 0.f;
        float _saved_center_x = 0.f;
        float _saved_center_y = 0.f;
        int _saved_image_width = 0;
        int _saved_image_height = 0;
    };
    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace lfs::core
