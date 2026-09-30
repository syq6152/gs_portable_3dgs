/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_rasterization.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <math_functions.h>

namespace lfs::training::kernels {

    namespace {
        constexpr uint32_t INVALID_TRIANGLE_ID = 0xFFFFFFFFu;

        __device__ __forceinline__ float edge_function(
            const float ax,
            const float ay,
            const float bx,
            const float by,
            const float px,
            const float py) {
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
        }

        __device__ __forceinline__ float min3(const float a, const float b, const float c) {
            return fminf(a, fminf(b, c));
        }

        __device__ __forceinline__ float max3(const float a, const float b, const float c) {
            return fmaxf(a, fmaxf(b, c));
        }

        __device__ __forceinline__ void normalize3(float& x, float& y, float& z) {
            const float n2 = x * x + y * y + z * z;
            if (!(n2 > 1e-20f) || !isfinite(n2)) {
                x = 0.0f;
                y = 0.0f;
                z = -1.0f;
                return;
            }
            const float inv = rsqrtf(n2);
            x *= inv;
            y *= inv;
            z *= inv;
        }

        __device__ __forceinline__ unsigned long long pack_depth_key(const float depth, const uint32_t tri_id) {
            return (static_cast<unsigned long long>(__float_as_uint(depth)) << 32ULL) |
                   static_cast<unsigned long long>(tri_id);
        }

        __global__ void mesh_depth_key_init_kernel(
            int64_t* depth_keys,
            const int pixel_count) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pixel_count) {
                return;
            }

            auto* keys = reinterpret_cast<unsigned long long*>(depth_keys);
            keys[idx] = pack_depth_key(CUDART_INF_F, INVALID_TRIANGLE_ID);
        }

        __global__ void mesh_transform_vertices_normals_kernel(
            const float* world_vertices_xyz,
            const float* world_normals_xyz,
            const int vertex_count,
            const float* w2c_4x4,
            float* cam_vertices_xyz,
            float* cam_normals_xyz) {
            const int v = blockIdx.x * blockDim.x + threadIdx.x;
            if (v >= vertex_count) {
                return;
            }

            const float wx = world_vertices_xyz[v * 3 + 0];
            const float wy = world_vertices_xyz[v * 3 + 1];
            const float wz = world_vertices_xyz[v * 3 + 2];

            const float x = w2c_4x4[0] * wx + w2c_4x4[1] * wy + w2c_4x4[2] * wz + w2c_4x4[3];
            const float y = w2c_4x4[4] * wx + w2c_4x4[5] * wy + w2c_4x4[6] * wz + w2c_4x4[7];
            const float z = w2c_4x4[8] * wx + w2c_4x4[9] * wy + w2c_4x4[10] * wz + w2c_4x4[11];

            cam_vertices_xyz[v * 3 + 0] = x;
            cam_vertices_xyz[v * 3 + 1] = y;
            cam_vertices_xyz[v * 3 + 2] = z;

            if (cam_normals_xyz == nullptr || world_normals_xyz == nullptr) {
                return;
            }

            const float wnx = world_normals_xyz[v * 3 + 0];
            const float wny = world_normals_xyz[v * 3 + 1];
            const float wnz = world_normals_xyz[v * 3 + 2];

            float cnx = w2c_4x4[0] * wnx + w2c_4x4[1] * wny + w2c_4x4[2] * wnz;
            float cny = w2c_4x4[4] * wnx + w2c_4x4[5] * wny + w2c_4x4[6] * wnz;
            float cnz = w2c_4x4[8] * wnx + w2c_4x4[9] * wny + w2c_4x4[10] * wnz;
            normalize3(cnx, cny, cnz);

            cam_normals_xyz[v * 3 + 0] = cnx;
            cam_normals_xyz[v * 3 + 1] = cny;
            cam_normals_xyz[v * 3 + 2] = cnz;
        }

        __global__ void mesh_rasterize_triangles_kernel(
            const float* cam_vertices_xyz,
            const int32_t* triangle_indices,
            const int face_count,
            const int width,
            const int height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float near_z,
            int64_t* depth_keys) {
            const int tri_id = blockIdx.x * blockDim.x + threadIdx.x;
            if (tri_id >= face_count) {
                return;
            }

            const int32_t i0 = triangle_indices[tri_id * 3 + 0];
            const int32_t i1 = triangle_indices[tri_id * 3 + 1];
            const int32_t i2 = triangle_indices[tri_id * 3 + 2];
            if (i0 < 0 || i1 < 0 || i2 < 0) {
                return;
            }

            const float x0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 0];
            const float y0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 1];
            const float z0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 2];
            const float x1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 0];
            const float y1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 1];
            const float z1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 2];
            const float x2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 0];
            const float y2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 1];
            const float z2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 2];

            if (!(z0 > near_z && z1 > near_z && z2 > near_z)) {
                return;
            }

            const float x0s = fx * (x0 / z0) + cx;
            const float y0s = fy * (y0 / z0) + cy;
            const float x1s = fx * (x1 / z1) + cx;
            const float y1s = fy * (y1 / z1) + cy;
            const float x2s = fx * (x2 / z2) + cx;
            const float y2s = fy * (y2 / z2) + cy;

            const float area = edge_function(x0s, y0s, x1s, y1s, x2s, y2s);
            if (!isfinite(area) || fabsf(area) < 1e-8f) {
                return;
            }
            const float inv_area = 1.0f / area;

            const int min_x = max(0, static_cast<int>(floorf(min3(x0s, x1s, x2s))));
            const int max_x = min(width - 1, static_cast<int>(ceilf(max3(x0s, x1s, x2s))));
            const int min_y = max(0, static_cast<int>(floorf(min3(y0s, y1s, y2s))));
            const int max_y = min(height - 1, static_cast<int>(ceilf(max3(y0s, y1s, y2s))));
            if (min_x > max_x || min_y > max_y) {
                return;
            }

            const float invz0 = 1.0f / z0;
            const float invz1 = 1.0f / z1;
            const float invz2 = 1.0f / z2;
            constexpr float INSIDE_EPS = -1e-5f;

            auto* keys = reinterpret_cast<unsigned long long*>(depth_keys);
            for (int py = min_y; py <= max_y; ++py) {
                const float sy = static_cast<float>(py) + 0.5f;
                for (int px = min_x; px <= max_x; ++px) {
                    const float sx = static_cast<float>(px) + 0.5f;

                    const float l0 = edge_function(x1s, y1s, x2s, y2s, sx, sy) * inv_area;
                    const float l1 = edge_function(x2s, y2s, x0s, y0s, sx, sy) * inv_area;
                    const float l2 = edge_function(x0s, y0s, x1s, y1s, sx, sy) * inv_area;
                    if (l0 < INSIDE_EPS || l1 < INSIDE_EPS || l2 < INSIDE_EPS) {
                        continue;
                    }

                    const float inv_z = l0 * invz0 + l1 * invz1 + l2 * invz2;
                    if (!(inv_z > 1e-8f) || !isfinite(inv_z)) {
                        continue;
                    }

                    const float depth = 1.0f / inv_z;
                    const int pixel_idx = py * width + px;
                    const unsigned long long key = pack_depth_key(depth, static_cast<uint32_t>(tri_id));
                    atomicMin(keys + pixel_idx, key);
                }
            }
        }

        __global__ void mesh_finalize_depth_normal_kernel(
            const int64_t* depth_keys,
            const float* cam_vertices_xyz,
            const float* cam_normals_xyz,
            const int32_t* triangle_indices,
            const int face_count,
            const int width,
            const int height,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            float* depth_out,
            float* normal_out) {
            const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int numel = width * height;
            if (pixel_idx >= numel) {
                return;
            }

            const auto* keys = reinterpret_cast<const unsigned long long*>(depth_keys);
            const unsigned long long key = keys[pixel_idx];
            const uint32_t tri_id_u32 = static_cast<uint32_t>(key & 0xFFFFFFFFULL);

            if (tri_id_u32 == INVALID_TRIANGLE_ID || tri_id_u32 >= static_cast<uint32_t>(face_count)) {
                depth_out[pixel_idx] = 0.0f;
                if (normal_out != nullptr) {
                    normal_out[pixel_idx] = 0.0f;
                    normal_out[numel + pixel_idx] = 0.0f;
                    normal_out[numel * 2 + pixel_idx] = 0.0f;
                }
                return;
            }

            // The winning key already contains the perspective-correct depth.
            // A depth-only request can finish here without camera-space normals
            // or the more expensive per-pixel normal interpolation path.
            if (normal_out == nullptr) {
                depth_out[pixel_idx] = __uint_as_float(static_cast<uint32_t>(key >> 32ULL));
                return;
            }

            const int tri_id = static_cast<int>(tri_id_u32);
            const int32_t i0 = triangle_indices[tri_id * 3 + 0];
            const int32_t i1 = triangle_indices[tri_id * 3 + 1];
            const int32_t i2 = triangle_indices[tri_id * 3 + 2];

            const float x0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 0];
            const float y0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 1];
            const float z0 = cam_vertices_xyz[static_cast<size_t>(i0) * 3 + 2];
            const float x1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 0];
            const float y1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 1];
            const float z1 = cam_vertices_xyz[static_cast<size_t>(i1) * 3 + 2];
            const float x2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 0];
            const float y2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 1];
            const float z2 = cam_vertices_xyz[static_cast<size_t>(i2) * 3 + 2];

            const float x0s = fx * (x0 / z0) + cx;
            const float y0s = fy * (y0 / z0) + cy;
            const float x1s = fx * (x1 / z1) + cx;
            const float y1s = fy * (y1 / z1) + cy;
            const float x2s = fx * (x2 / z2) + cx;
            const float y2s = fy * (y2 / z2) + cy;

            const float area = edge_function(x0s, y0s, x1s, y1s, x2s, y2s);
            if (!isfinite(area) || fabsf(area) < 1e-8f) {
                depth_out[pixel_idx] = 0.0f;
                normal_out[pixel_idx] = 0.0f;
                normal_out[numel + pixel_idx] = 0.0f;
                normal_out[numel * 2 + pixel_idx] = 0.0f;
                return;
            }
            const float inv_area = 1.0f / area;

            const int py = pixel_idx / width;
            const int px = pixel_idx - py * width;
            const float sx = static_cast<float>(px) + 0.5f;
            const float sy = static_cast<float>(py) + 0.5f;

            const float l0 = edge_function(x1s, y1s, x2s, y2s, sx, sy) * inv_area;
            const float l1 = edge_function(x2s, y2s, x0s, y0s, sx, sy) * inv_area;
            const float l2 = edge_function(x0s, y0s, x1s, y1s, sx, sy) * inv_area;

            const float invz0 = 1.0f / z0;
            const float invz1 = 1.0f / z1;
            const float invz2 = 1.0f / z2;
            const float inv_z = l0 * invz0 + l1 * invz1 + l2 * invz2;
            if (!(inv_z > 1e-8f) || !isfinite(inv_z)) {
                depth_out[pixel_idx] = 0.0f;
                normal_out[pixel_idx] = 0.0f;
                normal_out[numel + pixel_idx] = 0.0f;
                normal_out[numel * 2 + pixel_idx] = 0.0f;
                return;
            }

            const float depth = 1.0f / inv_z;
            const float w0 = (l0 * invz0) / inv_z;
            const float w1 = (l1 * invz1) / inv_z;
            const float w2 = (l2 * invz2) / inv_z;

            float nx = w0 * cam_normals_xyz[static_cast<size_t>(i0) * 3 + 0] +
                       w1 * cam_normals_xyz[static_cast<size_t>(i1) * 3 + 0] +
                       w2 * cam_normals_xyz[static_cast<size_t>(i2) * 3 + 0];
            float ny = w0 * cam_normals_xyz[static_cast<size_t>(i0) * 3 + 1] +
                       w1 * cam_normals_xyz[static_cast<size_t>(i1) * 3 + 1] +
                       w2 * cam_normals_xyz[static_cast<size_t>(i2) * 3 + 1];
            float nz = w0 * cam_normals_xyz[static_cast<size_t>(i0) * 3 + 2] +
                       w1 * cam_normals_xyz[static_cast<size_t>(i1) * 3 + 2] +
                       w2 * cam_normals_xyz[static_cast<size_t>(i2) * 3 + 2];
            normalize3(nx, ny, nz);

            if (nz > 0.0f) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }

            depth_out[pixel_idx] = depth;
            normal_out[pixel_idx] = nx;
            normal_out[numel + pixel_idx] = ny;
            normal_out[numel * 2 + pixel_idx] = nz;
        }

        __global__ void mesh_apply_mask_depth_normal_kernel(
            float* depth,
            float* normal,
            const float* mask,
            const int numel) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numel) {
                return;
            }

            const float mv = mask[idx];
            if (!isfinite(mv) || mv <= 0.0f) {
                depth[idx] = 0.0f;
                if (normal != nullptr) {
                    normal[idx] = 0.0f;
                    normal[numel + idx] = 0.0f;
                    normal[numel * 2 + idx] = 0.0f;
                }
            }
        }

        __global__ void mesh_extract_tri_id_kernel(
            const int64_t* depth_keys,
            const int face_count,
            const int numel,
            int32_t* tri_id_out) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numel) {
                return;
            }

            const auto* keys = reinterpret_cast<const unsigned long long*>(depth_keys);
            const unsigned long long key = keys[idx];
            const uint32_t tri_id_u32 = static_cast<uint32_t>(key & 0xFFFFFFFFULL);

            if (tri_id_u32 == INVALID_TRIANGLE_ID || tri_id_u32 >= static_cast<uint32_t>(face_count)) {
                tri_id_out[idx] = -1;
            } else {
                tri_id_out[idx] = static_cast<int32_t>(tri_id_u32);
            }
        }

        __global__ void mesh_apply_mask_tri_id_kernel(
            int32_t* tri_id,
            const float* mask,
            const int numel) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numel) {
                return;
            }

            const float mv = mask[idx];
            if (!isfinite(mv) || mv <= 0.0f) {
                tri_id[idx] = -1;
            }
        }

        // ---- GPU vertex normal computation ----
        // Pass 1: Each thread handles one face, computes face normal, atomicAdd to vertices
        __global__ void mesh_accumulate_face_normals_kernel(
            const float* vertices_xyz,
            const int32_t* triangle_indices,
            const int face_count,
            const int vertex_count,
            float* vertex_normals_xyz) {
            const int f = blockIdx.x * blockDim.x + threadIdx.x;
            if (f >= face_count) {
                return;
            }

            const int32_t i0 = triangle_indices[f * 3 + 0];
            const int32_t i1 = triangle_indices[f * 3 + 1];
            const int32_t i2 = triangle_indices[f * 3 + 2];
            if (i0 < 0 || i1 < 0 || i2 < 0 ||
                i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count) {
                return;
            }

            const float x0 = vertices_xyz[i0 * 3 + 0];
            const float y0 = vertices_xyz[i0 * 3 + 1];
            const float z0 = vertices_xyz[i0 * 3 + 2];
            const float x1 = vertices_xyz[i1 * 3 + 0];
            const float y1 = vertices_xyz[i1 * 3 + 1];
            const float z1 = vertices_xyz[i1 * 3 + 2];
            const float x2 = vertices_xyz[i2 * 3 + 0];
            const float y2 = vertices_xyz[i2 * 3 + 1];
            const float z2 = vertices_xyz[i2 * 3 + 2];

            const float e1x = x1 - x0;
            const float e1y = y1 - y0;
            const float e1z = z1 - z0;
            const float e2x = x2 - x0;
            const float e2y = y2 - y0;
            const float e2z = z2 - z0;

            float nx = e1y * e2z - e1z * e2y;
            float ny = e1z * e2x - e1x * e2z;
            float nz = e1x * e2y - e1y * e2x;

            const float n2 = nx * nx + ny * ny + nz * nz;
            if (!(n2 > 1e-20f) || !isfinite(n2)) {
                return;
            }
            const float inv = rsqrtf(n2);
            nx *= inv;
            ny *= inv;
            nz *= inv;

            atomicAdd(vertex_normals_xyz + i0 * 3 + 0, nx);
            atomicAdd(vertex_normals_xyz + i0 * 3 + 1, ny);
            atomicAdd(vertex_normals_xyz + i0 * 3 + 2, nz);
            atomicAdd(vertex_normals_xyz + i1 * 3 + 0, nx);
            atomicAdd(vertex_normals_xyz + i1 * 3 + 1, ny);
            atomicAdd(vertex_normals_xyz + i1 * 3 + 2, nz);
            atomicAdd(vertex_normals_xyz + i2 * 3 + 0, nx);
            atomicAdd(vertex_normals_xyz + i2 * 3 + 1, ny);
            atomicAdd(vertex_normals_xyz + i2 * 3 + 2, nz);
        }

        // Pass 2: Each thread normalizes one vertex normal
        __global__ void mesh_normalize_vertex_normals_kernel(
            float* vertex_normals_xyz,
            const int vertex_count) {
            const int v = blockIdx.x * blockDim.x + threadIdx.x;
            if (v >= vertex_count) {
                return;
            }

            float nx = vertex_normals_xyz[v * 3 + 0];
            float ny = vertex_normals_xyz[v * 3 + 1];
            float nz = vertex_normals_xyz[v * 3 + 2];
            normalize3(nx, ny, nz);
            vertex_normals_xyz[v * 3 + 0] = nx;
            vertex_normals_xyz[v * 3 + 1] = ny;
            vertex_normals_xyz[v * 3 + 2] = nz;
        }
    } // namespace

    void launch_mesh_depth_key_init(
        int64_t* depth_keys,
        const int pixel_count,
        cudaStream_t stream) {
        constexpr int BLOCK_SIZE = 256;
        const int grid = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_depth_key_init_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(depth_keys, pixel_count);
    }

    void launch_mesh_transform_vertices_normals(
        const float* world_vertices_xyz,
        const float* world_normals_xyz,
        const int vertex_count,
        const float* w2c_4x4,
        float* cam_vertices_xyz,
        float* cam_normals_xyz,
        cudaStream_t stream) {
        constexpr int BLOCK_SIZE = 256;
        const int grid = (vertex_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_transform_vertices_normals_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            world_vertices_xyz,
            world_normals_xyz,
            vertex_count,
            w2c_4x4,
            cam_vertices_xyz,
            cam_normals_xyz);
    }

    void launch_mesh_rasterize_triangles(
        const float* cam_vertices_xyz,
        const float* cam_normals_xyz,
        const int32_t* triangle_indices,
        const int face_count,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_z,
        int64_t* depth_keys,
        cudaStream_t stream) {
        (void)cam_normals_xyz;
        constexpr int BLOCK_SIZE = 128;
        const int grid = (face_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_rasterize_triangles_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            cam_vertices_xyz,
            triangle_indices,
            face_count,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_z,
            depth_keys);
    }

    void launch_mesh_finalize_depth_normal(
        const int64_t* depth_keys,
        const float* cam_vertices_xyz,
        const float* cam_normals_xyz,
        const int32_t* triangle_indices,
        const int face_count,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        float* depth_out,
        float* normal_out,
        cudaStream_t stream) {
        const int numel = width * height;
        constexpr int BLOCK_SIZE = 256;
        const int grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_finalize_depth_normal_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            depth_keys,
            cam_vertices_xyz,
            cam_normals_xyz,
            triangle_indices,
            face_count,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            depth_out,
            normal_out);
    }

    void launch_mesh_apply_mask_depth_normal(
        float* depth,
        float* normal,
        const float* mask,
        const int width,
        const int height,
        cudaStream_t stream) {
        const int numel = width * height;
        constexpr int BLOCK_SIZE = 256;
        const int grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_apply_mask_depth_normal_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            depth,
            normal,
            mask,
            numel);
    }

    void launch_mesh_extract_tri_id(
        const int64_t* depth_keys,
        const int face_count,
        const int width,
        const int height,
        int32_t* tri_id_out,
        cudaStream_t stream) {
        const int numel = width * height;
        constexpr int BLOCK_SIZE = 256;
        const int grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_extract_tri_id_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            depth_keys,
            face_count,
            numel,
            tri_id_out);
    }

    void launch_mesh_apply_mask_tri_id(
        int32_t* tri_id,
        const float* mask,
        const int width,
        const int height,
        cudaStream_t stream) {
        const int numel = width * height;
        constexpr int BLOCK_SIZE = 256;
        const int grid = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mesh_apply_mask_tri_id_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            tri_id,
            mask,
            numel);
    }

    void launch_mesh_compute_vertex_normals(
        const float* vertices_xyz,
        const int32_t* triangle_indices,
        const int vertex_count,
        const int face_count,
        float* vertex_normals_xyz,
        cudaStream_t stream) {
        // vertex_normals_xyz must be pre-zeroed by caller

        // Pass 1: accumulate face normals to vertices via atomicAdd
        {
            constexpr int BLOCK_SIZE = 256;
            const int grid = (face_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            mesh_accumulate_face_normals_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                vertices_xyz,
                triangle_indices,
                face_count,
                vertex_count,
                vertex_normals_xyz);
        }

        // Pass 2: normalize each vertex normal
        {
            constexpr int BLOCK_SIZE = 256;
            const int grid = (vertex_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            mesh_normalize_vertex_normals_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                vertex_normals_xyz,
                vertex_count);
        }
    }

} // namespace lfs::training::kernels
