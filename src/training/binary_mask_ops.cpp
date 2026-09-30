/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "binary_mask_ops.hpp"

#include <algorithm>
#include <numeric>

namespace lfs::training {

    namespace {
        // Square-window morphology. `dilate` picks the direction: dilation sets a pixel when
        // any sample in the window is set, erosion clears it when any sample is unset.
        // `outside` is the value assumed for samples that fall off the image edge.
        BinaryMask morph_binary_mask(
            const BinaryMask& mask,
            const int radius,
            const bool dilate,
            const bool outside) {

            if (radius <= 0 || !mask.is_valid()) {
                return mask;
            }

            BinaryMask out = mask;
            for (int y = 0; y < mask.height; ++y) {
                for (int x = 0; x < mask.width; ++x) {
                    // Dilation looks for a set sample, erosion for an unset one, so the
                    // value that triggers is `dilate` either way.
                    bool found = false;
                    for (int dy = -radius; dy <= radius && !found; ++dy) {
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int yy = y + dy;
                            const int xx = x + dx;
                            const bool in_bounds =
                                yy >= 0 && yy < mask.height && xx >= 0 && xx < mask.width;
                            const bool sample = in_bounds ? mask.at(xx, yy) : outside;
                            if (sample == dilate) {
                                found = true;
                                break;
                            }
                        }
                    }
                    out.set(x, y, dilate ? found : !found);
                }
            }
            return out;
        }
    } // namespace

    BinaryMask make_binary_mask(const int width, const int height, const bool initial) {
        BinaryMask mask;
        mask.width = std::max(0, width);
        mask.height = std::max(0, height);
        mask.values.assign(
            static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height),
            initial ? uint8_t{1} : uint8_t{0});
        return mask;
    }

    BinaryMask erode_binary_mask(const BinaryMask& mask, const int radius) {
        return morph_binary_mask(mask, radius, false, false);
    }

    BinaryMask dilate_binary_mask(const BinaryMask& mask, const int radius) {
        return morph_binary_mask(mask, radius, true, false);
    }

    BinaryMask open_binary_mask(const BinaryMask& mask, const int radius) {
        return dilate_binary_mask(erode_binary_mask(mask, radius), radius);
    }

    BinaryMask close_binary_mask(const BinaryMask& mask, const int radius) {
        // The closing erosion treats outside as SET, unlike the standalone erosion. With
        // outside unset it would trim a `radius`-wide ring off the image edge, which breaks
        // the property closing exists for: close(A) must contain A. The hole-fill path
        // relies on that — a stripped border ring reads as background, connects to the
        // image edge, and can reclassify a genuine interior hole as open.
        return morph_binary_mask(dilate_binary_mask(mask, radius), radius, false, true);
    }

    BinaryMask union_binary_mask(const BinaryMask& a, const BinaryMask& b) {
        if (!a.is_valid()) {
            return b;
        }
        if (!b.is_valid() || a.width != b.width || a.height != b.height) {
            return a;
        }
        BinaryMask out = a;
        for (size_t i = 0; i < out.values.size(); ++i) {
            out.values[i] = (a.values[i] != 0 || b.values[i] != 0) ? uint8_t{1} : uint8_t{0};
        }
        return out;
    }

    BinaryMask intersect_binary_mask(const BinaryMask& a, const BinaryMask& b) {
        if (!a.is_valid() || !b.is_valid() || a.width != b.width || a.height != b.height) {
            return make_binary_mask(a.width, a.height);
        }
        BinaryMask out = a;
        for (size_t i = 0; i < out.values.size(); ++i) {
            out.values[i] = (a.values[i] != 0 && b.values[i] != 0) ? uint8_t{1} : uint8_t{0};
        }
        return out;
    }

    size_t count_binary_mask(const BinaryMask& mask) {
        return static_cast<size_t>(
            std::count_if(mask.values.begin(), mask.values.end(),
                          [](const uint8_t value) { return value != 0; }));
    }

    BackgroundComponents label_background_components(const BinaryMask& mask) {
        BackgroundComponents result;
        if (!mask.is_valid()) {
            return result;
        }

        result.width = mask.width;
        result.height = mask.height;
        result.labels.assign(mask.values.size(), int32_t{-1});

        // Iterative flood fill. Recursion would blow the stack on full-resolution images
        // where a single background component can span millions of pixels.
        std::vector<size_t> stack;
        constexpr int kNeighborX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        constexpr int kNeighborY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

        for (int seed_y = 0; seed_y < mask.height; ++seed_y) {
            for (int seed_x = 0; seed_x < mask.width; ++seed_x) {
                const size_t seed_index = mask.index(seed_x, seed_y);
                if (mask.values[seed_index] != 0 || result.labels[seed_index] >= 0) {
                    continue;
                }

                const auto label = static_cast<int32_t>(result.areas.size());
                result.areas.push_back(0);
                result.touches_border.push_back(0);

                stack.clear();
                stack.push_back(seed_index);
                result.labels[seed_index] = label;

                while (!stack.empty()) {
                    const size_t current = stack.back();
                    stack.pop_back();

                    const int x = static_cast<int>(current % static_cast<size_t>(mask.width));
                    const int y = static_cast<int>(current / static_cast<size_t>(mask.width));
                    ++result.areas[static_cast<size_t>(label)];
                    if (x == 0 || y == 0 || x == mask.width - 1 || y == mask.height - 1) {
                        result.touches_border[static_cast<size_t>(label)] = 1;
                    }

                    for (int n = 0; n < 8; ++n) {
                        const int nx = x + kNeighborX[n];
                        const int ny = y + kNeighborY[n];
                        if (nx < 0 || ny < 0 || nx >= mask.width || ny >= mask.height) {
                            continue;
                        }
                        const size_t neighbor = mask.index(nx, ny);
                        if (mask.values[neighbor] != 0 || result.labels[neighbor] >= 0) {
                            continue;
                        }
                        result.labels[neighbor] = label;
                        stack.push_back(neighbor);
                    }
                }
            }
        }

        return result;
    }

} // namespace lfs::training
