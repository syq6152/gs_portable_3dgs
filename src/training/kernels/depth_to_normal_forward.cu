/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_to_normal_forward.hpp"
#include <cmath>

namespace lfs::training::kernels {
    namespace {

        constexpr int BLOCK_SIZE = 256;

        /**
         * GPU forward kernel for depth_to_normal_map (GGGS formula 24).
         *
         * For each interior pixel (y,x) in [1..H-2, 1..W-2]:
         *   Unproject 4 neighbors → camera-space points
         *   dx = P_right - P_left,  dy = P_down - P_up
         *   raw = cross(dx, dy)
         *   n = normalize(raw),  flip if n.z > 0
         *
         * Output: [3, H, W] CHW normal map. Boundary pixels remain zero.
         */
        __global__ void depth_to_normal_forward_kernel(
            const float* __restrict__ depth,   // [H, W]
            float* __restrict__ normal,        // [3, H, W] CHW output
            const int H,
            const int W,
            const float fx,
            const float fy,
            const float cx,
            const float cy) {

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int inner_h = H - 2;
            const int inner_w = W - 2;
            const int inner_total = inner_h * inner_w;
            if (tid >= inner_total) return;

            const int iy = tid / inner_w;
            const int ix = tid % inner_w;
            const int y = iy + 1;
            const int x = ix + 1;
            const int HW = H * W;
            const int center = y * W + x;

            // Read 5 depth values
            const float zc = depth[center];
            const float zl = depth[y * W + (x - 1)];
            const float zr = depth[y * W + (x + 1)];
            const float zu = depth[(y - 1) * W + x];
            const float zd = depth[(y + 1) * W + x];

            // Validity: all 5 depths must be positive and finite
            if (!(zc > 0.0f) || !isfinite(zc) ||
                !(zl > 0.0f) || !isfinite(zl) ||
                !(zr > 0.0f) || !isfinite(zr) ||
                !(zu > 0.0f) || !isfinite(zu) ||
                !(zd > 0.0f) || !isfinite(zd)) {
                normal[center] = 0.0f;
                normal[HW + center] = 0.0f;
                normal[2 * HW + center] = 0.0f;
                return;
            }

            // Unproject to camera-space coordinates
            const float px_l = (static_cast<float>(x - 1) - cx) / fx;
            const float px_r = (static_cast<float>(x + 1) - cx) / fx;
            const float px_c = (static_cast<float>(x) - cx) / fx;
            const float py_c = (static_cast<float>(y) - cy) / fy;
            const float py_u = (static_cast<float>(y - 1) - cy) / fy;
            const float py_d = (static_cast<float>(y + 1) - cy) / fy;

            // dx = P_right - P_left
            const float dx0 = px_r * zr - px_l * zl;
            const float dx1 = py_c * zr - py_c * zl;
            const float dx2 = zr - zl;

            // dy = P_down - P_up
            const float dy0 = px_c * zd - px_c * zu;
            const float dy1 = py_d * zd - py_u * zu;
            const float dy2 = zd - zu;

            // raw = cross(dx, dy)
            float nx = dx1 * dy2 - dx2 * dy1;
            float ny = dx2 * dy0 - dx0 * dy2;
            float nz = dx0 * dy1 - dx1 * dy0;

            const float len = sqrtf(nx * nx + ny * ny + nz * nz);
            if (!(len > 1e-8f) || !isfinite(len)) {
                normal[center] = 0.0f;
                normal[HW + center] = 0.0f;
                normal[2 * HW + center] = 0.0f;
                return;
            }

            nx /= len;
            ny /= len;
            nz /= len;

            // Flip to face camera (camera looks along -Z, so normal should have nz < 0)
            if (nz > 0.0f) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            normal[center] = nx;
            normal[HW + center] = ny;
            normal[2 * HW + center] = nz;
        }

    } // anonymous namespace

    void launch_depth_to_normal_forward(
        const float* depth,
        float* normal,
        int H,
        int W,
        float fx,
        float fy,
        float cx,
        float cy,
        cudaStream_t stream) {

        const int inner_total = (H - 2) * (W - 2);
        if (inner_total <= 0) return;

        const int num_blocks = (inner_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        depth_to_normal_forward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            depth, normal,
            H, W, fx, fy, cx, cy);
    }

} // namespace lfs::training::kernels
