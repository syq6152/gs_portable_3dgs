/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_normal_utils.hpp"
#include "training/kernels/depth_to_normal_forward.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lfs::training {

    namespace {
        core::Tensor to_depth_hw_cpu(const core::Tensor& depth_map) {
            if (!depth_map.is_valid() || depth_map.is_empty()) {
                return {};
            }

            core::Tensor depth = depth_map;
            if (depth.ndim() == 4 && depth.shape()[0] == 1) {
                depth = depth.squeeze(0);
            }
            if (depth.ndim() == 3 && depth.shape()[0] == 1) {
                depth = depth.squeeze(0);
            } else if (depth.ndim() == 3 && depth.shape()[2] == 1) {
                depth = depth.squeeze(2);
            }

            if (depth.ndim() != 2) {
                throw std::runtime_error("Depth tensor must be [H,W], [1,H,W], [H,W,1], or [1,H,W,1]");
            }

            return depth.to(core::Device::CPU).to(core::DataType::Float32).contiguous();
        }

        inline bool is_valid_depth(const float z) {
            return std::isfinite(z) && z > 0.0f;
        }

        inline std::array<float, 3> to_camera_point(
            const int x,
            const int y,
            const float z,
            const float fx,
            const float fy,
            const float cx,
            const float cy) {
            return {
                (static_cast<float>(x) - cx) * z / fx,
                (static_cast<float>(y) - cy) * z / fy,
                z};
        }
    } // namespace

    core::Tensor depth_to_normal_map(
        const core::Tensor& depth_map,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {

        auto depth = to_depth_hw_cpu(depth_map);
        if (!depth.is_valid() || depth.is_empty()) {
            return {};
        }

        if (std::abs(fx) < 1e-8f || std::abs(fy) < 1e-8f) {
            throw std::runtime_error("Invalid intrinsics: fx/fy must be non-zero");
        }

        const int h = static_cast<int>(depth.shape()[0]);
        const int w = static_cast<int>(depth.shape()[1]);
        auto normal = core::Tensor::zeros(
            {3, static_cast<size_t>(h), static_cast<size_t>(w)},
            core::Device::CPU,
            core::DataType::Float32);

        const float* d = depth.ptr<float>();
        float* nx = normal.ptr<float>();
        float* ny = nx + (h * w);
        float* nz = ny + (h * w);

        auto idx = [w](const int yy, const int xx) {
            return yy * w + xx;
        };

        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const int center = idx(y, x);

                const float zc = d[center];
                const float zl = d[idx(y, x - 1)];
                const float zr = d[idx(y, x + 1)];
                const float zu = d[idx(y - 1, x)];
                const float zd = d[idx(y + 1, x)];

                if (!is_valid_depth(zc) || !is_valid_depth(zl) || !is_valid_depth(zr) ||
                    !is_valid_depth(zu) || !is_valid_depth(zd)) {
                    continue;
                }

                const auto pl = to_camera_point(x - 1, y, zl, fx, fy, cx, cy);
                const auto pr = to_camera_point(x + 1, y, zr, fx, fy, cx, cy);
                const auto pu = to_camera_point(x, y - 1, zu, fx, fy, cx, cy);
                const auto pd = to_camera_point(x, y + 1, zd, fx, fy, cx, cy);

                const float vx0 = pr[0] - pl[0];
                const float vy0 = pr[1] - pl[1];
                const float vz0 = pr[2] - pl[2];

                const float vx1 = pd[0] - pu[0];
                const float vy1 = pd[1] - pu[1];
                const float vz1 = pd[2] - pu[2];

                float nnx = vy0 * vz1 - vz0 * vy1;
                float nny = vz0 * vx1 - vx0 * vz1;
                float nnz = vx0 * vy1 - vy0 * vx1;

                const float len = std::sqrt(nnx * nnx + nny * nny + nnz * nnz);
                if (!(len > 1e-8f) || !std::isfinite(len)) {
                    continue;
                }

                nnx /= len;
                nny /= len;
                nnz /= len;

                // Keep normals facing roughly toward the camera for stable visualization.
                if (nnz > 0.0f) {
                    nnx = -nnx;
                    nny = -nny;
                    nnz = -nnz;
                }

                nx[center] = nnx;
                ny[center] = nny;
                nz[center] = nnz;
            }
        }

        return normal;
    }

    core::Tensor depth_to_normal_map_gpu(
        const core::Tensor& depth_map,
        const float fx,
        const float fy,
        const float cx,
        const float cy) {

        if (!depth_map.is_valid() || depth_map.is_empty()) {
            return {};
        }
        if (std::abs(fx) < 1e-8f || std::abs(fy) < 1e-8f) {
            throw std::runtime_error("Invalid intrinsics: fx/fy must be non-zero");
        }

        // Ensure depth is [H,W] on CUDA
        core::Tensor depth = depth_map;
        if (depth.device() != core::Device::CUDA) {
            depth = depth.cuda();
        }
        if (depth.ndim() == 4 && depth.shape()[0] == 1) depth = depth.squeeze(0);
        if (depth.ndim() == 3 && depth.shape()[0] == 1) depth = depth.squeeze(0);
        else if (depth.ndim() == 3 && depth.shape()[2] == 1) depth = depth.squeeze(2);
        depth = depth.contiguous();

        const int h = static_cast<int>(depth.shape()[0]);
        const int w = static_cast<int>(depth.shape()[1]);
        auto normal = core::Tensor::zeros(
            {3, static_cast<size_t>(h), static_cast<size_t>(w)}, core::Device::CUDA);

        kernels::launch_depth_to_normal_forward(
            depth.ptr<float>(), normal.ptr<float>(),
            h, w, fx, fy, cx, cy, nullptr);

        return normal;
    }

    core::Tensor colorize_depth_map(const core::Tensor& depth_map) {
        auto depth = to_depth_hw_cpu(depth_map);
        if (!depth.is_valid() || depth.is_empty()) {
            return {};
        }

        const int h = static_cast<int>(depth.shape()[0]);
        const int w = static_cast<int>(depth.shape()[1]);
        const int n = h * w;

        const float* d = depth.ptr<float>();
        float dmin = std::numeric_limits<float>::infinity();
        float dmax = 0.0f;

        for (int i = 0; i < n; ++i) {
            const float z = d[i];
            if (!is_valid_depth(z)) {
                continue;
            }
            dmin = std::min(dmin, z);
            dmax = std::max(dmax, z);
        }

        auto color = core::Tensor::zeros(
            {3, static_cast<size_t>(h), static_cast<size_t>(w)},
            core::Device::CPU,
            core::DataType::Float32);

        if (!(dmax > dmin) || !std::isfinite(dmin) || !std::isfinite(dmax)) {
            return color;
        }

        float* r = color.ptr<float>();
        float* g = r + n;
        float* b = g + n;
        const float denom = dmax - dmin;

        for (int i = 0; i < n; ++i) {
            const float z = d[i];
            if (!is_valid_depth(z)) {
                continue;
            }

            const float t = std::clamp((z - dmin) / denom, 0.0f, 1.0f);

            float rr = 0.0f;
            float gg = 0.0f;
            float bb = 0.0f;
            if (t < 0.25f) {
                rr = 0.0f;
                gg = 4.0f * t;
                bb = 1.0f;
            } else if (t < 0.5f) {
                rr = 0.0f;
                gg = 1.0f;
                bb = 1.0f - 4.0f * (t - 0.25f);
            } else if (t < 0.75f) {
                rr = 4.0f * (t - 0.5f);
                gg = 1.0f;
                bb = 0.0f;
            } else {
                rr = 1.0f;
                gg = 1.0f - 4.0f * (t - 0.75f);
                bb = 0.0f;
            }

            r[i] = std::clamp(rr, 0.0f, 1.0f);
            g[i] = std::clamp(gg, 0.0f, 1.0f);
            b[i] = std::clamp(bb, 0.0f, 1.0f);
        }

        return color;
    }

    core::Tensor colorize_normal_map(const core::Tensor& normal_map) {
        if (!normal_map.is_valid() || normal_map.is_empty()) {
            return {};
        }

        core::Tensor normal = normal_map;
        if (normal.ndim() == 4 && normal.shape()[0] == 1) {
            normal = normal.squeeze(0);
        }
        if (normal.ndim() == 3 && normal.shape()[2] == 3) {
            normal = normal.permute({2, 0, 1});
        }
        if (normal.ndim() != 3 || normal.shape()[0] != 3) {
            throw std::runtime_error("Normal tensor must be [3,H,W], [H,W,3], or [1,3,H,W]");
        }

        normal = normal.to(core::Device::CPU).to(core::DataType::Float32).contiguous();
        return (normal * 0.5f + 0.5f).clamp(0.0f, 1.0f);
    }

} // namespace lfs::training
