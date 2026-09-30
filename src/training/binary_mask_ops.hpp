/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfs::training {

    // Row-major binary image. Non-zero means set.
    //
    // Extracted from pseudo_view_precompute.cpp so the project_mesh mask path can reuse
    // the same morphology, and so these operations become unit-testable in isolation.
    struct BinaryMask {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> values;

        [[nodiscard]] bool is_valid() const noexcept {
            return width > 0 && height > 0 &&
                   values.size() == static_cast<size_t>(width) * static_cast<size_t>(height);
        }
        [[nodiscard]] size_t index(const int x, const int y) const noexcept {
            return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
        }
        [[nodiscard]] bool at(const int x, const int y) const noexcept {
            return values[index(x, y)] != 0;
        }
        void set(const int x, const int y, const bool value) noexcept {
            values[index(x, y)] = value ? uint8_t{1} : uint8_t{0};
        }
    };

    [[nodiscard]] BinaryMask make_binary_mask(int width, int height, bool initial = false);

    // Square structuring element of side (2 * radius + 1). radius <= 0 returns the input.
    // Erosion treats samples outside the image as unset, matching
    // launch_project_mesh_depth_mask_erode; dilation does not wrap around.
    [[nodiscard]] BinaryMask erode_binary_mask(const BinaryMask& mask, int radius);
    [[nodiscard]] BinaryMask dilate_binary_mask(const BinaryMask& mask, int radius);

    // open = erode then dilate (removes speckles).
    [[nodiscard]] BinaryMask open_binary_mask(const BinaryMask& mask, int radius);
    // close = dilate then erode (seals gaps narrower than 2 * radius). Border-safe: the
    // result always contains the input, including along the image edge.
    [[nodiscard]] BinaryMask close_binary_mask(const BinaryMask& mask, int radius);

    [[nodiscard]] BinaryMask union_binary_mask(const BinaryMask& a, const BinaryMask& b);

    // Shape mismatch yields an all-unset mask sized like `a`: an intersection with an
    // unusable operand has no pixels either side can vouch for.
    [[nodiscard]] BinaryMask intersect_binary_mask(const BinaryMask& a, const BinaryMask& b);

    [[nodiscard]] size_t count_binary_mask(const BinaryMask& mask);

    // Connected components of the UNSET pixels of `mask` — i.e. the gaps in coverage.
    //
    // Uses 8-connectivity, which is the conservative choice here: a diagonal gap lets a
    // component reach the image border, so fewer components are reported as enclosed and
    // fewer holes get filled. Components are labeled from 0; `labels[i] < 0` marks a set
    // pixel that belongs to no component.
    struct BackgroundComponents {
        int width = 0;
        int height = 0;
        std::vector<int32_t> labels;      // per pixel, -1 for set pixels
        std::vector<size_t> areas;        // per component, pixel count
        std::vector<uint8_t> touches_border; // per component, 1 when it reaches the image border

        [[nodiscard]] size_t component_count() const noexcept { return areas.size(); }
        [[nodiscard]] int32_t label_at(const int x, const int y) const noexcept {
            return labels[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
        }
        // A component fully surrounded by set pixels. Geometrically, for a silhouette
        // coverage mask, these are the interior holes of the silhouette.
        [[nodiscard]] bool is_enclosed(const int32_t label) const noexcept {
            return label >= 0 &&
                   static_cast<size_t>(label) < touches_border.size() &&
                   touches_border[static_cast<size_t>(label)] == 0;
        }
    };

    [[nodiscard]] BackgroundComponents label_background_components(const BinaryMask& mask);

} // namespace lfs::training
