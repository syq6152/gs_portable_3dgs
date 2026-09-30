/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::training {

    /// Load pseudo-view cameras described by `<output>/pseudo_views/manifest.json`.
    ///
    /// For every entry with status == "ok" this reads the per-view pose JSON
    /// (R / T / intrinsics / image size) and constructs a `lfs::core::Camera`
    /// whose image_path points at the cached pseudo RGB. The mask_path points
    /// at supervision_mask_path when present, and falls back to valid_mask_path
    /// for older manifests. Each camera is tagged `is_pseudo() == true` and
    /// assigned `supervision_weight = loss_weight` so the trainer can attenuate
    /// its loss/gradient relative to real views.
    ///
    /// Manifest `base_camera_uid` is preserved on each pseudo camera so pose
    /// refinement can keep pseudo poses attached to the real camera used as
    /// their pose-generation base. It is not necessarily the RGB source.
    ///
    /// Manifest `source_image_name` + `source_camera_id` are preserved as the
    /// stable identity of the real frame whose RGB was reprojected. This source
    /// is independent of `base_camera_uid` and of the pseudo camera's own
    /// `camera_id()`. `source_uid` is retained only as an optional same-run
    /// diagnostic. Older caches without these fields remain loadable and expose
    /// `Camera::has_pseudo_rgb_source() == false`.
    ///
    /// `base_camera` supplies camera_model_type / distortion tensors / camera_id
    /// for distortion-free cloning; pass any real training camera (typically
    /// the first one) so the resulting cameras play nicely with the existing
    /// rasterization pipeline (which expects PINHOLE / no distortion for
    /// pseudo views).
    ///
    /// `starting_uid` is the first uid handed out to pseudo cameras; subsequent
    /// pseudo cameras get monotonically decreasing negative uids so they cannot
    /// collide with real-camera uids (which are non-negative).
    ///
    /// `min_valid_pixels` is an optional absolute floor applied to the decoded
    /// supervision mask. Zero disables the absolute floor, but an all-zero mask
    /// is always rejected.
    ///
    /// `log_diagnostics` controls per-entry cache warnings. Callers using strict
    /// minimal metadata can disable them so paths and processing details are not
    /// exposed through runtime logs.
    std::expected<std::vector<std::shared_ptr<lfs::core::Camera>>, std::string>
    load_pseudo_view_cameras(
        const std::filesystem::path& manifest_path,
        const lfs::core::Camera& base_camera,
        float loss_weight,
        int starting_uid = -100000,
        std::size_t min_valid_pixels = 0,
        bool log_diagnostics = true);

} // namespace lfs::training
