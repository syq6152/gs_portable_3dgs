/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_to_normal_backward.hpp"
#include <cmath>

namespace lfs::training::kernels {
    namespace {

        constexpr int BLOCK_SIZE = 256;
        constexpr float DEPTH_EPS = 1e-8f;

        /**
         * Per-pixel kernel: for valid pixel (y,x) in [1..H-2, 1..W-2],
         * recompute the forward of depth_to_normal_map and backpropagate
         * grad_depth_normal → grad for the 4 neighbor depths via atomicAdd.
         *
         * Forward recap:
         *   P_l = ((x-1-cx)/fx * zl, (y-cy)/fy * zl, zl)
         *   P_r = ((x+1-cx)/fx * zr, (y-cy)/fy * zr, zr)
         *   P_u = ((x-cx)/fx * zu, (y-1-cy)/fy * zu, zu)
         *   P_d = ((x-cx)/fx * zd, (y+1-cy)/fy * zd, zd)
         *   dx = P_r - P_l,  dy = P_d - P_u
         *   raw = dx × dy
         *   n = raw / |raw|,  flip if n.z > 0
         */
        __global__ void depth_to_normal_backward_kernel(
            const float* __restrict__ grad_dn, // [3, H, W]
            const float* __restrict__ depth,   // [H, W]
            float* __restrict__ grad_depth,    // [H, W] output, atomicAdd
            const int H,
            const int W,
            const float fx,
            const float fy,
            const float cx,
            const float cy) {

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int HW = H * W;
            // Only process interior pixels [1..H-2, 1..W-2]
            const int inner_h = H - 2;
            const int inner_w = W - 2;
            const int inner_total = inner_h * inner_w;
            if (tid >= inner_total) return;

            const int iy = tid / inner_w;  // [0, inner_h)
            const int ix = tid % inner_w;  // [0, inner_w)
            const int y = iy + 1;
            const int x = ix + 1;
            const int center = y * W + x;

            // Check validity (same as forward)
            const float zc = depth[center];
            const float zl = depth[y * W + (x - 1)];
            const float zr = depth[y * W + (x + 1)];
            const float zu = depth[(y - 1) * W + x];
            const float zd = depth[(y + 1) * W + x];

            const bool valid = (zc > 0.0f) && isfinite(zc) &&
                               (zl > 0.0f) && isfinite(zl) &&
                               (zr > 0.0f) && isfinite(zr) &&
                               (zu > 0.0f) && isfinite(zu) &&
                               (zd > 0.0f) && isfinite(zd);
            if (!valid) return;

            // Recompute forward: unproject 4 neighbors
            const float px_l = (static_cast<float>(x - 1) - cx) / fx;
            const float py_c = (static_cast<float>(y) - cy) / fy;
            const float px_r = (static_cast<float>(x + 1) - cx) / fx;
            const float px_c = (static_cast<float>(x) - cx) / fx;
            const float py_u = (static_cast<float>(y - 1) - cy) / fy;
            const float py_d = (static_cast<float>(y + 1) - cy) / fy;

            // dx = P_right - P_left
            const float dx0 = px_r * zr - px_l * zl;
            const float dx1 = py_c * zr - py_c * zl;  // = py_c * (zr - zl)
            const float dx2 = zr - zl;

            // dy = P_down - P_up
            const float dy0 = px_c * zd - px_c * zu;  // = px_c * (zd - zu)
            const float dy1 = py_d * zd - py_u * zu;
            const float dy2 = zd - zu;

            // raw = dx × dy
            const float raw_x = dx1 * dy2 - dx2 * dy1;
            const float raw_y = dx2 * dy0 - dx0 * dy2;
            const float raw_z = dx0 * dy1 - dx1 * dy0;

            const float len = sqrtf(raw_x * raw_x + raw_y * raw_y + raw_z * raw_z);
            if (len < DEPTH_EPS) return;

            // n = raw / len
            float nx = raw_x / len;
            float ny = raw_y / len;
            float nz = raw_z / len;

            // Flip: if nz > 0, negate
            const bool flipped = (nz > 0.0f);

            // Read incoming gradient
            float gx = grad_dn[center];
            float gy = grad_dn[HW + center];
            float gz = grad_dn[2 * HW + center];

            // If we flipped in forward, also negate the incoming grad
            if (flipped) {
                gx = -gx;
                gy = -gy;
                gz = -gz;
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            // Backward through normalize: n = raw / |raw|
            // d(n)/d(raw) = (I - n*n^T) / |raw|
            const float proj = gx * nx + gy * ny + gz * nz;
            const float inv_len = 1.0f / len;
            const float g_raw_x = (gx - nx * proj) * inv_len;
            const float g_raw_y = (gy - ny * proj) * inv_len;
            const float g_raw_z = (gz - nz * proj) * inv_len;

            // Backward through cross product: raw = dx × dy
            // d(raw)/d(dx) applied to g_raw gives: g_dx = dy × g_raw
            // d(raw)/d(dy) applied to g_raw gives: g_dy = g_raw × dx
            const float g_dx0 = dy1 * g_raw_z - dy2 * g_raw_y;
            const float g_dx1 = dy2 * g_raw_x - dy0 * g_raw_z;
            const float g_dx2 = dy0 * g_raw_y - dy1 * g_raw_x;

            const float g_dy0 = g_raw_y * dx2 - g_raw_z * dx1;
            const float g_dy1 = g_raw_z * dx0 - g_raw_x * dx2;
            const float g_dy2 = g_raw_x * dx1 - g_raw_y * dx0;

            // Backward through dx = P_right - P_left:
            //   dx0 = px_r * zr - px_l * zl  → dL/dzr += px_r * g_dx0, dL/dzl += -px_l * g_dx0
            //   dx1 = py_c * (zr - zl)       → dL/dzr += py_c * g_dx1, dL/dzl += -py_c * g_dx1
            //   dx2 = zr - zl                 → dL/dzr += g_dx2,        dL/dzl += -g_dx2
            float g_zl = -px_l * g_dx0 - py_c * g_dx1 - g_dx2;
            float g_zr =  px_r * g_dx0 + py_c * g_dx1 + g_dx2;

            // Backward through dy = P_down - P_up:
            //   dy0 = px_c * (zd - zu)        → dL/dzd += px_c * g_dy0, dL/dzu += -px_c * g_dy0
            //   dy1 = py_d * zd - py_u * zu   → dL/dzd += py_d * g_dy1, dL/dzu += -py_u * g_dy1
            //   dy2 = zd - zu                  → dL/dzd += g_dy2,        dL/dzu += -g_dy2
            float g_zu = -px_c * g_dy0 - py_u * g_dy1 - g_dy2;
            float g_zd =  px_c * g_dy0 + py_d * g_dy1 + g_dy2;

            // Scatter to 4 neighbor depth pixels
            atomicAdd(&grad_depth[y * W + (x - 1)], g_zl);
            atomicAdd(&grad_depth[y * W + (x + 1)], g_zr);
            atomicAdd(&grad_depth[(y - 1) * W + x], g_zu);
            atomicAdd(&grad_depth[(y + 1) * W + x], g_zd);
        }

    } // anonymous namespace

    void launch_depth_to_normal_backward(
        const float* grad_depth_normal,
        const float* depth,
        float* grad_depth,
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
        depth_to_normal_backward_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            grad_depth_normal, depth, grad_depth,
            H, W, fx, fy, cx, cy);
    }

} // namespace lfs::training::kernels
