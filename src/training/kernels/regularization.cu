/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/regularization.cuh"
#include <algorithm>
#include <cstdint>

namespace lfs::training::kernels {

    // =============================================================================
    // SCALE REGULARIZATION (exp-based)
    // =============================================================================

    /**
     * Fused kernel: computes exp(x), accumulates gradient, and sums for loss
     * OPTIMIZED: Uses warp-level reductions (5-10× faster than CUB!)
     */
    __global__ void fused_scale_regularization_kernel(
        const float* __restrict__ params,
        float* __restrict__ param_grads,
        float* __restrict__ partial_sums,
        size_t n,
        float grad_scale) { // weight / n

        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < n;
             idx += blockDim.x * gridDim.x) {

            const float x = params[idx];
            const float exp_x = expf(x); // exp(scaling_raw)

            // Accumulate for loss
            local_sum += exp_x;

            // Accumulate gradient: ∂L/∂x = grad_scale * exp(x)
            atomicAdd(&param_grads[idx], grad_scale * exp_x);
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    /**
     * Final reduction kernel
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void final_scale_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float weight,
        size_t n) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = weight * (sum / static_cast<float>(n)); // loss = weight * mean
        }
    }

    void launch_fused_scale_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream) {

        if (n == 0 || weight == 0.0f) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = std::min((n + block_size - 1) / block_size, size_t(1024));

        float grad_scale = weight / static_cast<float>(n);

        // Launch fused kernel
        fused_scale_regularization_kernel<<<num_blocks, block_size, 0, stream>>>(
            params, param_grads, temp_buffer, n, grad_scale);

        // Launch final reduction
        final_scale_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, weight, n);
    }

    // =============================================================================
    // OPACITY REGULARIZATION (sigmoid-based)
    // =============================================================================

    /**
     * Fused kernel: computes sigmoid(x), accumulates gradient, and sums for loss
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void fused_opacity_regularization_kernel(
        const float* __restrict__ params,
        float* __restrict__ param_grads,
        float* __restrict__ partial_sums,
        size_t n,
        float grad_scale) { // weight / n

        float local_sum = 0.0f;

        // Grid-stride loop for coalesced memory access
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < n;
             idx += blockDim.x * gridDim.x) {

            const float x = params[idx];
            const float sigmoid_x = 1.0f / (1.0f + expf(-x)); // sigmoid(opacity_raw)

            // Accumulate for loss
            local_sum += sigmoid_x;

            // Gradient: ∂L/∂x = grad_scale * sigmoid(x) * (1 - sigmoid(x))
            atomicAdd(&param_grads[idx], grad_scale * sigmoid_x * (1.0f - sigmoid_x));
        }

        // Block-level warp reduction (tiny-cuda-nn style - much faster!)
        local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);

        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = local_sum;
        }
    }

    /**
     * Final reduction kernel
     * OPTIMIZED: Uses warp-level reductions
     */
    __global__ void final_opacity_reduce_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ result,
        int num_blocks,
        float weight,
        size_t n) {

        // Grid-stride loop to handle more than blockDim.x partial sums
        float sum = 0.0f;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += partial_sums[i];
        }

        // Block-level warp reduction
        sum = lfs::core::warp_ops::block_reduce_sum(sum);

        if (threadIdx.x == 0) {
            result[0] = weight * (sum / static_cast<float>(n)); // loss = weight * mean
        }
    }

    void launch_fused_opacity_regularization(
        const float* params,
        float* param_grads,
        float* loss_out,
        float* temp_buffer,
        size_t n,
        float weight,
        cudaStream_t stream) {

        if (n == 0 || weight == 0.0f) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = std::min((n + block_size - 1) / block_size, size_t(1024));

        float grad_scale = weight / static_cast<float>(n);

        // Launch fused kernel
        fused_opacity_regularization_kernel<<<num_blocks, block_size, 0, stream>>>(
            params, param_grads, temp_buffer, n, grad_scale);

        // Launch final reduction
        final_opacity_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer, loss_out, num_blocks, weight, n);
    }

    // =============================================================================
    // MESH SURFACE REGULARIZATION
    // =============================================================================

    namespace {
        struct Mat3 {
            float m11, m12, m13;
            float m21, m22, m23;
            float m31, m32, m33;
        };

        __device__ inline float3 f3_add(const float3 a, const float3 b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }

        __device__ inline float3 f3_sub(const float3 a, const float3 b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }

        __device__ inline float3 f3_mul(const float3 a, const float s) {
            return make_float3(a.x * s, a.y * s, a.z * s);
        }

        __device__ inline float f3_dot(const float3 a, const float3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        __device__ inline float3 f3_cross(const float3 a, const float3 b) {
            return make_float3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
        }

        __device__ inline float3 f3_safe_normalize(const float3 v) {
            const float n2 = f3_dot(v, v);
            if (n2 < 1e-20f) {
                return make_float3(0.0f, 0.0f, 0.0f);
            }
            return f3_mul(v, rsqrtf(n2));
        }

        __device__ inline float3 load_vertex(
            const float* __restrict__ verts,
            const int32_t vertex_idx) {
            return make_float3(
                verts[vertex_idx * 3 + 0],
                verts[vertex_idx * 3 + 1],
                verts[vertex_idx * 3 + 2]);
        }

        __device__ inline void load_triangle(
            const float* __restrict__ verts,
            const int32_t* __restrict__ indices,
            const int32_t face,
            float3& a,
            float3& b,
            float3& c) {
            a = load_vertex(verts, indices[face * 3 + 0]);
            b = load_vertex(verts, indices[face * 3 + 1]);
            c = load_vertex(verts, indices[face * 3 + 2]);
        }

        __device__ inline float3 closest_point_on_triangle(
            const float3 p,
            const float3 a,
            const float3 b,
            const float3 c) {
            const float3 ab = f3_sub(b, a);
            const float3 ac = f3_sub(c, a);
            const float3 ap = f3_sub(p, a);

            const float d1 = f3_dot(ab, ap);
            const float d2 = f3_dot(ac, ap);
            if (d1 <= 0.0f && d2 <= 0.0f) {
                return a;
            }

            const float3 bp = f3_sub(p, b);
            const float d3 = f3_dot(ab, bp);
            const float d4 = f3_dot(ac, bp);
            if (d3 >= 0.0f && d4 <= d3) {
                return b;
            }

            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                const float v = d1 / (d1 - d3);
                return f3_add(a, f3_mul(ab, v));
            }

            const float3 cp = f3_sub(p, c);
            const float d5 = f3_dot(ab, cp);
            const float d6 = f3_dot(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) {
                return c;
            }

            const float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                const float w = d2 / (d2 - d6);
                return f3_add(a, f3_mul(ac, w));
            }

            const float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                return f3_add(b, f3_mul(f3_sub(c, b), w));
            }

            const float denom = 1.0f / (va + vb + vc);
            const float v = vb * denom;
            const float w = vc * denom;
            return f3_add(a, f3_add(f3_mul(ab, v), f3_mul(ac, w)));
        }

        __device__ inline float evaluate_face_distance(
            const float3 p,
            const int32_t face,
            const float* __restrict__ verts,
            const int32_t* __restrict__ indices,
            float3& q_out) {
            float3 a, b, c;
            load_triangle(verts, indices, face, a, b, c);
            q_out = closest_point_on_triangle(p, a, b, c);
            const float3 d = f3_sub(p, q_out);
            return f3_dot(d, d);
        }

        __device__ inline int32_t walk_nearest_face(
            const float3 p,
            int32_t face,
            const int32_t* __restrict__ edge_neighbors,
            const float* __restrict__ verts,
            const int32_t* __restrict__ indices,
            const size_t n_faces,
            const int walk_steps,
            float3& best_q,
            float& best_d2) {
            best_d2 = evaluate_face_distance(p, face, verts, indices, best_q);
            int32_t best_face = face;

            for (int step = 0; step < walk_steps; ++step) {
                int32_t next_face = best_face;
                float3 next_q = best_q;
                float next_d2 = best_d2;

                for (int edge = 0; edge < 3; ++edge) {
                    const int32_t candidate = edge_neighbors[best_face * 3 + edge];
                    if (candidate < 0 || static_cast<size_t>(candidate) >= n_faces) {
                        continue;
                    }
                    float3 q;
                    const float d2 = evaluate_face_distance(p, candidate, verts, indices, q);
                    if (d2 + 1e-12f < next_d2) {
                        next_d2 = d2;
                        next_q = q;
                        next_face = candidate;
                    }
                }

                if (next_face == best_face) {
                    break;
                }
                best_face = next_face;
                best_q = next_q;
                best_d2 = next_d2;
            }

            return best_face;
        }

        __device__ inline Mat3 rotation_from_raw_quaternion(
            const float* __restrict__ rotation_raw,
            const size_t idx,
            float& qr,
            float& qx,
            float& qy,
            float& qz,
            float& q_norm_sq_safe,
            float& qxx,
            float& qyy,
            float& qzz,
            float& qxy,
            float& qxz,
            float& qyz,
            float& qrx,
            float& qry,
            float& qrz) {
            qr = rotation_raw[idx * 4 + 0];
            qx = rotation_raw[idx * 4 + 1];
            qy = rotation_raw[idx * 4 + 2];
            qz = rotation_raw[idx * 4 + 3];

            const float qrr_raw = qr * qr;
            const float qxx_raw = qx * qx;
            const float qyy_raw = qy * qy;
            const float qzz_raw = qz * qz;
            q_norm_sq_safe = fmaxf(qrr_raw + qxx_raw + qyy_raw + qzz_raw, 1e-7f);

            qxx = 2.0f * qxx_raw / q_norm_sq_safe;
            qyy = 2.0f * qyy_raw / q_norm_sq_safe;
            qzz = 2.0f * qzz_raw / q_norm_sq_safe;
            qxy = 2.0f * qx * qy / q_norm_sq_safe;
            qxz = 2.0f * qx * qz / q_norm_sq_safe;
            qyz = 2.0f * qy * qz / q_norm_sq_safe;
            qrx = 2.0f * qr * qx / q_norm_sq_safe;
            qry = 2.0f * qr * qy / q_norm_sq_safe;
            qrz = 2.0f * qr * qz / q_norm_sq_safe;

            return Mat3{
                1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
                qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
                qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
        }

        __device__ inline float4 rotation_matrix_grad_to_raw_quaternion_grad(
            const Mat3 dR,
            const float qr,
            const float qx,
            const float qy,
            const float qz,
            const float q_norm_sq_safe,
            const float qxx,
            const float qyy,
            const float qzz,
            const float qxy,
            const float qxz,
            const float qyz,
            const float qrx,
            const float qry,
            const float qrz) {
            const float dL_dqxx = -dR.m22 - dR.m33;
            const float dL_dqyy = -dR.m11 - dR.m33;
            const float dL_dqzz = -dR.m11 - dR.m22;
            const float dL_dqxy = dR.m12 + dR.m21;
            const float dL_dqxz = dR.m13 + dR.m31;
            const float dL_dqyz = dR.m23 + dR.m32;
            const float dL_dqrx = dR.m32 - dR.m23;
            const float dL_dqry = dR.m13 - dR.m31;
            const float dL_dqrz = dR.m21 - dR.m12;
            const float dL_dq_norm_helper =
                qxx * dL_dqxx + qyy * dL_dqyy + qzz * dL_dqzz +
                qxy * dL_dqxy + qxz * dL_dqxz + qyz * dL_dqyz +
                qrx * dL_dqrx + qry * dL_dqry + qrz * dL_dqrz;
            const float scale = 2.0f / q_norm_sq_safe;
            return make_float4(
                (qx * dL_dqrx + qy * dL_dqry + qz * dL_dqrz - qr * dL_dq_norm_helper) * scale,
                (2.0f * qx * dL_dqxx + qy * dL_dqxy + qz * dL_dqxz + qr * dL_dqrx - qx * dL_dq_norm_helper) * scale,
                (2.0f * qy * dL_dqyy + qx * dL_dqxy + qz * dL_dqyz + qr * dL_dqry - qy * dL_dq_norm_helper) * scale,
                (2.0f * qz * dL_dqzz + qx * dL_dqxz + qy * dL_dqyz + qr * dL_dqrz - qz * dL_dq_norm_helper) * scale);
        }

        __global__ void mark_visible_mesh_primitives_kernel(
            const uint32_t* __restrict__ visible_primitive_indices,
            const size_t n_visible_primitives,
            const int32_t* __restrict__ current_faces,
            int32_t* __restrict__ visible_mask,
            int32_t* __restrict__ visible_count,
            const size_t n_primitives) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n_visible_primitives;
                 idx += blockDim.x * gridDim.x) {
                const uint32_t primitive_idx = visible_primitive_indices[idx];
                if (static_cast<size_t>(primitive_idx) >= n_primitives) {
                    continue;
                }
                if (current_faces[primitive_idx] < 0) {
                    continue;
                }
                const int32_t old = atomicExch(&visible_mask[primitive_idx], 1);
                if (old == 0) {
                    atomicAdd(visible_count, 1);
                }
            }
        }

        __global__ void count_mesh_surface_primitives_kernel(
            const int32_t* __restrict__ current_faces,
            const int32_t* __restrict__ visible_mask,
            int32_t* __restrict__ count_out,
            const size_t n_primitives,
            const size_t n_faces) {
            int local_count = 0;
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n_primitives;
                 idx += blockDim.x * gridDim.x) {
                if (visible_mask != nullptr && visible_mask[idx] == 0) {
                    continue;
                }
                const int32_t face = current_faces[idx];
                if (face >= 0 && static_cast<size_t>(face) < n_faces) {
                    ++local_count;
                }
            }

            local_count = lfs::core::warp_ops::block_reduce_sum(local_count);
            if (threadIdx.x == 0 && local_count > 0) {
                atomicAdd(count_out, local_count);
            }
        }

        __global__ void compute_mesh_face_normals_kernel(
            const float* __restrict__ mesh_vertices,
            const int32_t* __restrict__ mesh_indices,
            float* __restrict__ face_normals,
            const size_t n_faces) {
            for (size_t face = blockIdx.x * blockDim.x + threadIdx.x;
                 face < n_faces;
                 face += blockDim.x * gridDim.x) {
                float3 a, b, c;
                load_triangle(mesh_vertices, mesh_indices, static_cast<int32_t>(face), a, b, c);
                const float3 n = f3_safe_normalize(f3_cross(f3_sub(b, a), f3_sub(c, a)));
                face_normals[face * 3 + 0] = n.x;
                face_normals[face * 3 + 1] = n.y;
                face_normals[face * 3 + 2] = n.z;
            }
        }

        __device__ inline float mesh_constraint_spatial_weight(
            const float dist,
            const float signed_side,
            const float inside_constraint_distance,
            const float inside_constraint_fade_ratio) {
            if (!(inside_constraint_distance > 0.0f) || signed_side >= 0.0f) {
                return 1.0f;
            }
            if (dist >= inside_constraint_distance) {
                return 0.0f;
            }

            const float fade_width = inside_constraint_distance * inside_constraint_fade_ratio;
            const float full_width = inside_constraint_distance - fade_width;
            if (!(fade_width > 0.0f) || dist <= full_width) {
                return 1.0f;
            }

            const float t = fminf(fmaxf(
                                      (inside_constraint_distance - dist) / fade_width, 0.0f),
                                  1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        __global__ void prepare_mesh_constraint_regions_kernel(
            const float* __restrict__ means,
            const int32_t* __restrict__ visible_mask,
            int32_t* __restrict__ current_faces,
            const int32_t* __restrict__ edge_neighbors,
            const float* __restrict__ mesh_vertices,
            const int32_t* __restrict__ mesh_indices,
            const float* __restrict__ face_normals,
            float* __restrict__ surface_points,
            float* __restrict__ spatial_weights,
            float* __restrict__ soft_weight_sum,
            const size_t n_primitives,
            const size_t n_faces,
            const int walk_steps,
            const float inside_constraint_distance,
            const float inside_constraint_fade_ratio) {
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n_primitives;
                 idx += blockDim.x * gridDim.x) {
                int32_t face = current_faces[idx];
                if (face < 0 || static_cast<size_t>(face) >= n_faces) {
                    spatial_weights[idx] = 0.0f;
                    continue;
                }

                const float3 p = make_float3(
                    means[idx * 3 + 0], means[idx * 3 + 1], means[idx * 3 + 2]);
                float3 q;
                float dist2 = 0.0f;
                face = walk_nearest_face(
                    p, face, edge_neighbors, mesh_vertices, mesh_indices,
                    n_faces, walk_steps > 1 ? walk_steps : 1, q, dist2);
                current_faces[idx] = face;
                surface_points[idx * 3 + 0] = q.x;
                surface_points[idx * 3 + 1] = q.y;
                surface_points[idx * 3 + 2] = q.z;

                const float3 displacement = f3_sub(p, q);
                const float3 face_normal = make_float3(
                    face_normals[face * 3 + 0],
                    face_normals[face * 3 + 1],
                    face_normals[face * 3 + 2]);
                const float weight = mesh_constraint_spatial_weight(
                    sqrtf(fmaxf(dist2, 0.0f)),
                    f3_dot(displacement, face_normal),
                    inside_constraint_distance,
                    inside_constraint_fade_ratio);
                spatial_weights[idx] = weight;
                if (weight > 0.0f && (visible_mask == nullptr || visible_mask[idx] != 0)) {
                    atomicAdd(soft_weight_sum, weight);
                }
            }
        }

        __global__ void mesh_surface_regularization_kernel(
            const float* __restrict__ means,
            const float* __restrict__ scaling_raw,
            const float* __restrict__ rotation_raw,
            float* __restrict__ grad_means,
            float* __restrict__ grad_scaling_raw,
            float* __restrict__ grad_rotation_raw,
            const int32_t* __restrict__ visible_mask,
            const int32_t* __restrict__ soft_count,
            const float* __restrict__ soft_weight_sum,
            const int32_t* __restrict__ tracked_count,
            int32_t* __restrict__ current_faces,
            const int32_t* __restrict__ edge_neighbors,
            const float* __restrict__ mesh_vertices,
            const int32_t* __restrict__ mesh_indices,
            const float* __restrict__ face_normals,
            const float* __restrict__ surface_points,
            const float* __restrict__ spatial_weights,
            float* __restrict__ partial_sums,
            const size_t n_primitives,
            const size_t n_faces,
            const int walk_steps,
            const float lambda_project,
            const float lambda_outside_barrier,
            const float outside_distance_threshold,
            const float lambda_scale_min,
            const float lambda_scale_max,
            const float rho,
            const float lambda_normal) {
            const int32_t soft_count_i = soft_count[0];
            const float soft_weight_sum_f = soft_weight_sum != nullptr ? soft_weight_sum[0] : 0.0f;
            const int32_t tracked_count_i = tracked_count[0];
            if (soft_count_i <= 0 && !(soft_weight_sum_f > 0.0f) && tracked_count_i <= 0) {
                if (threadIdx.x == 0) {
                    partial_sums[blockIdx.x] = 0.0f;
                    partial_sums[gridDim.x + blockIdx.x] = 0.0f;
                }
                return;
            }

            const bool need_project = lambda_project > 0.0f;
            const bool need_outside_barrier = lambda_outside_barrier > 0.0f;
            const bool need_normal = lambda_normal > 0.0f;
            const bool need_scale = lambda_scale_min > 0.0f ||
                                    lambda_scale_max > 0.0f ||
                                    need_normal;
            const bool use_spatial_weights = spatial_weights != nullptr;
            const float inv_soft_count = use_spatial_weights
                                             ? (soft_weight_sum_f > 0.0f ? 1.0f / soft_weight_sum_f : 0.0f)
                                             : (soft_count_i > 0 ? 1.0f / static_cast<float>(soft_count_i) : 0.0f);
            const float inv_tracked_count = tracked_count_i > 0
                                                ? 1.0f / static_cast<float>(tracked_count_i)
                                                : 0.0f;
            float local_soft_sum = 0.0f;
            float local_barrier_sum = 0.0f;

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n_primitives;
                 idx += blockDim.x * gridDim.x) {
                int32_t face = current_faces[idx];
                if (face < 0 || static_cast<size_t>(face) >= n_faces) {
                    continue;
                }

                const bool soft_active = visible_mask == nullptr || visible_mask[idx] != 0;
                const float spatial_weight = use_spatial_weights ? spatial_weights[idx] : 1.0f;
                const bool weighted_soft_active = soft_active && spatial_weight > 0.0f;
                const bool need_surface_walk = surface_points != nullptr || need_outside_barrier || (need_project && soft_active);
                float3 p = make_float3(0.0f, 0.0f, 0.0f);
                float3 q = make_float3(0.0f, 0.0f, 0.0f);
                float dist2 = 0.0f;
                float dist = 0.0f;
                float3 displacement = make_float3(0.0f, 0.0f, 0.0f);

                if (surface_points != nullptr) {
                    p = make_float3(
                        means[idx * 3 + 0],
                        means[idx * 3 + 1],
                        means[idx * 3 + 2]);
                    q = make_float3(
                        surface_points[idx * 3 + 0],
                        surface_points[idx * 3 + 1],
                        surface_points[idx * 3 + 2]);
                    displacement = f3_sub(p, q);
                    dist2 = f3_dot(displacement, displacement);
                    dist = sqrtf(fmaxf(dist2, 0.0f));
                } else if (need_surface_walk) {
                    p = make_float3(
                        means[idx * 3 + 0],
                        means[idx * 3 + 1],
                        means[idx * 3 + 2]);
                    face = walk_nearest_face(
                        p, face, edge_neighbors, mesh_vertices, mesh_indices,
                        n_faces, walk_steps > 1 ? walk_steps : 1, q, dist2);
                    current_faces[idx] = face;
                    dist = sqrtf(fmaxf(dist2, 0.0f));
                    displacement = f3_sub(p, q);
                }

                if (need_project && weighted_soft_active) {
                    local_soft_sum += lambda_project * spatial_weight * dist;
                    if (dist > 1e-12f) {
                        const float grad_scale = lambda_project * spatial_weight * inv_soft_count / dist;
                        grad_means[idx * 3 + 0] += displacement.x * grad_scale;
                        grad_means[idx * 3 + 1] += displacement.y * grad_scale;
                        grad_means[idx * 3 + 2] += displacement.z * grad_scale;
                    }
                }

                if (need_outside_barrier) {
                    const float3 face_normal = make_float3(
                        face_normals[face * 3 + 0],
                        face_normals[face * 3 + 1],
                        face_normals[face * 3 + 2]);
                    const float signed_side = f3_dot(displacement, face_normal);
                    const float excess = dist - outside_distance_threshold;
                    if (signed_side > 0.0f && excess > 0.0f) {
                        local_barrier_sum += lambda_outside_barrier * excess * excess /
                                             (2.0f * outside_distance_threshold);
                        if (dist > 1e-12f) {
                            const float grad_scale = lambda_outside_barrier * inv_tracked_count *
                                                     excess / (outside_distance_threshold * dist);
                            grad_means[idx * 3 + 0] += displacement.x * grad_scale;
                            grad_means[idx * 3 + 1] += displacement.y * grad_scale;
                            grad_means[idx * 3 + 2] += displacement.z * grad_scale;
                        }
                    }
                }

                if (!weighted_soft_active) {
                    continue;
                }

                int min_axis = 0;
                int max_axis = 0;
                if (need_scale) {
                    const float s0 = expf(scaling_raw[idx * 3 + 0]);
                    const float s1 = expf(scaling_raw[idx * 3 + 1]);
                    const float s2 = expf(scaling_raw[idx * 3 + 2]);

                    float min_scale = s0;
                    if (s1 < min_scale) {
                        min_axis = 1;
                        min_scale = s1;
                    }
                    if (s2 < min_scale) {
                        min_axis = 2;
                        min_scale = s2;
                    }

                    float max_scale = s0;
                    if (s1 > max_scale) {
                        max_axis = 1;
                        max_scale = s1;
                    }
                    if (s2 > max_scale) {
                        max_axis = 2;
                        max_scale = s2;
                    }

                    if (lambda_scale_min > 0.0f) {
                        local_soft_sum += lambda_scale_min * spatial_weight * min_scale;
                        grad_scaling_raw[idx * 3 + min_axis] += lambda_scale_min * spatial_weight * min_scale * inv_soft_count;
                    }
                    if (lambda_scale_max > 0.0f) {
                        const float diff = max_scale - rho;
                        if (diff > 0.0f) {
                            local_soft_sum += lambda_scale_max * spatial_weight * diff;
                            grad_scaling_raw[idx * 3 + max_axis] += lambda_scale_max * spatial_weight * max_scale * inv_soft_count;
                        }
                    }
                }

                if (need_normal) {
                    const float3 face_normal = make_float3(
                        face_normals[face * 3 + 0],
                        face_normals[face * 3 + 1],
                        face_normals[face * 3 + 2]);

                    float qr, qx, qy, qz, q_norm_sq_safe;
                    float qxx, qyy, qzz, qxy, qxz, qyz, qrx, qry, qrz;
                    const Mat3 rotation = rotation_from_raw_quaternion(
                        rotation_raw, idx, qr, qx, qy, qz, q_norm_sq_safe,
                        qxx, qyy, qzz, qxy, qxz, qyz, qrx, qry, qrz);

                    float3 gs_normal;
                    if (min_axis == 0) {
                        gs_normal = make_float3(rotation.m11, rotation.m21, rotation.m31);
                    } else if (min_axis == 1) {
                        gs_normal = make_float3(rotation.m12, rotation.m22, rotation.m32);
                    } else {
                        gs_normal = make_float3(rotation.m13, rotation.m23, rotation.m33);
                    }

                    const float normal_dot = f3_dot(gs_normal, face_normal);
                    local_soft_sum += lambda_normal * spatial_weight * (1.0f - normal_dot);

                    const float3 dcol = f3_mul(face_normal, -lambda_normal * spatial_weight * inv_soft_count);
                    Mat3 dR{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    if (min_axis == 0) {
                        dR.m11 = dcol.x;
                        dR.m21 = dcol.y;
                        dR.m31 = dcol.z;
                    } else if (min_axis == 1) {
                        dR.m12 = dcol.x;
                        dR.m22 = dcol.y;
                        dR.m32 = dcol.z;
                    } else {
                        dR.m13 = dcol.x;
                        dR.m23 = dcol.y;
                        dR.m33 = dcol.z;
                    }

                    const float4 dq = rotation_matrix_grad_to_raw_quaternion_grad(
                        dR, qr, qx, qy, qz, q_norm_sq_safe,
                        qxx, qyy, qzz, qxy, qxz, qyz, qrx, qry, qrz);
                    grad_rotation_raw[idx * 4 + 0] += dq.x;
                    grad_rotation_raw[idx * 4 + 1] += dq.y;
                    grad_rotation_raw[idx * 4 + 2] += dq.z;
                    grad_rotation_raw[idx * 4 + 3] += dq.w;
                }
            }

            local_soft_sum = lfs::core::warp_ops::block_reduce_sum(local_soft_sum);
            __syncthreads();
            local_barrier_sum = lfs::core::warp_ops::block_reduce_sum(local_barrier_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_soft_sum;
                partial_sums[gridDim.x + blockIdx.x] = local_barrier_sum;
            }
        }

        __global__ void final_mesh_surface_reduce_kernel(
            const float* __restrict__ partial_sums,
            const int32_t* __restrict__ soft_count,
            const float* __restrict__ soft_weight_sum,
            const int32_t* __restrict__ tracked_count,
            float* __restrict__ result,
            const int num_blocks) {
            float soft_sum = 0.0f;
            float barrier_sum = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
                soft_sum += partial_sums[i];
                barrier_sum += partial_sums[num_blocks + i];
            }
            soft_sum = lfs::core::warp_ops::block_reduce_sum(soft_sum);
            __syncthreads();
            barrier_sum = lfs::core::warp_ops::block_reduce_sum(barrier_sum);
            if (threadIdx.x == 0) {
                const float soft_denominator = soft_weight_sum != nullptr
                                                   ? soft_weight_sum[0]
                                                   : static_cast<float>(soft_count[0]);
                const float normalized_soft = soft_denominator > 0.0f
                                                  ? soft_sum / soft_denominator
                                                  : 0.0f;
                const float normalized_barrier = tracked_count[0] > 0
                                                     ? barrier_sum / static_cast<float>(tracked_count[0])
                                                     : 0.0f;
                result[0] = normalized_soft + normalized_barrier;
            }
        }
    } // namespace

    void launch_mark_visible_mesh_primitives(
        const uint32_t* visible_primitive_indices,
        size_t n_visible_primitives,
        const int32_t* current_faces,
        int32_t* visible_mask,
        int32_t* visible_count,
        size_t n_primitives,
        cudaStream_t stream) {
        if (visible_primitive_indices == nullptr || current_faces == nullptr ||
            visible_mask == nullptr || visible_count == nullptr ||
            n_visible_primitives == 0 || n_primitives == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = static_cast<int>(std::min(
            (n_visible_primitives + block_size - 1) / block_size,
            size_t(1024)));
        mark_visible_mesh_primitives_kernel<<<num_blocks, block_size, 0, stream>>>(
            visible_primitive_indices,
            n_visible_primitives,
            current_faces,
            visible_mask,
            visible_count,
            n_primitives);
    }

    void launch_count_mesh_surface_primitives(
        const int32_t* current_faces,
        const int32_t* visible_mask,
        int32_t* count_out,
        size_t n_primitives,
        size_t n_faces,
        cudaStream_t stream) {
        if (current_faces == nullptr || count_out == nullptr ||
            n_primitives == 0 || n_faces == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = static_cast<int>(std::min(
            (n_primitives + block_size - 1) / block_size,
            size_t(1024)));
        count_mesh_surface_primitives_kernel<<<num_blocks, block_size, 0, stream>>>(
            current_faces,
            visible_mask,
            count_out,
            n_primitives,
            n_faces);
    }

    void launch_compute_mesh_face_normals(
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        float* face_normals,
        size_t n_faces,
        cudaStream_t stream) {
        if (mesh_vertices == nullptr || mesh_indices == nullptr ||
            face_normals == nullptr || n_faces == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = static_cast<int>(std::min(
            (n_faces + block_size - 1) / block_size,
            size_t(1024)));
        compute_mesh_face_normals_kernel<<<num_blocks, block_size, 0, stream>>>(
            mesh_vertices,
            mesh_indices,
            face_normals,
            n_faces);
    }

    void launch_prepare_mesh_constraint_regions(
        const float* means,
        const int32_t* visible_mask,
        int32_t* current_faces,
        const int32_t* edge_neighbors,
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        const float* face_normals,
        float* surface_points,
        float* spatial_weights,
        float* soft_weight_sum,
        size_t n_primitives,
        size_t n_faces,
        int walk_steps,
        float inside_constraint_distance,
        float inside_constraint_fade_ratio,
        cudaStream_t stream) {
        if (means == nullptr || current_faces == nullptr || edge_neighbors == nullptr ||
            mesh_vertices == nullptr || mesh_indices == nullptr || face_normals == nullptr ||
            surface_points == nullptr || spatial_weights == nullptr || soft_weight_sum == nullptr ||
            n_primitives == 0 || n_faces == 0 || !(inside_constraint_distance > 0.0f)) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = static_cast<int>(std::min(
            (n_primitives + block_size - 1) / block_size,
            size_t(1024)));
        prepare_mesh_constraint_regions_kernel<<<num_blocks, block_size, 0, stream>>>(
            means,
            visible_mask,
            current_faces,
            edge_neighbors,
            mesh_vertices,
            mesh_indices,
            face_normals,
            surface_points,
            spatial_weights,
            soft_weight_sum,
            n_primitives,
            n_faces,
            walk_steps > 1 ? walk_steps : 1,
            inside_constraint_distance,
            inside_constraint_fade_ratio);
    }

    void launch_mesh_surface_regularization(
        const float* means,
        const float* scaling_raw,
        const float* rotation_raw,
        float* grad_means,
        float* grad_scaling_raw,
        float* grad_rotation_raw,
        const int32_t* visible_mask,
        const int32_t* soft_count,
        const float* soft_weight_sum,
        const int32_t* tracked_count,
        int32_t* current_faces,
        const int32_t* edge_neighbors,
        const float* mesh_vertices,
        const int32_t* mesh_indices,
        const float* face_normals,
        const float* surface_points,
        const float* spatial_weights,
        float* loss_out,
        float* temp_buffer,
        size_t n_primitives,
        size_t n_faces,
        int walk_steps,
        float lambda_project,
        float lambda_outside_barrier,
        float outside_distance_threshold,
        float lambda_scale_min,
        float lambda_scale_max,
        float rho,
        float lambda_normal,
        cudaStream_t stream) {
        if (means == nullptr || scaling_raw == nullptr || rotation_raw == nullptr ||
            grad_means == nullptr || grad_scaling_raw == nullptr || grad_rotation_raw == nullptr ||
            soft_count == nullptr || tracked_count == nullptr || current_faces == nullptr ||
            edge_neighbors == nullptr || mesh_vertices == nullptr || mesh_indices == nullptr ||
            loss_out == nullptr || temp_buffer == nullptr ||
            n_primitives == 0 || n_faces == 0) {
            return;
        }

        const int block_size = 256;
        const int num_blocks = static_cast<int>(std::min(
            (n_primitives + block_size - 1) / block_size,
            size_t(1024)));

        mesh_surface_regularization_kernel<<<num_blocks, block_size, 0, stream>>>(
            means,
            scaling_raw,
            rotation_raw,
            grad_means,
            grad_scaling_raw,
            grad_rotation_raw,
            visible_mask,
            soft_count,
            soft_weight_sum,
            tracked_count,
            current_faces,
            edge_neighbors,
            mesh_vertices,
            mesh_indices,
            face_normals,
            surface_points,
            spatial_weights,
            temp_buffer,
            n_primitives,
            n_faces,
            walk_steps > 1 ? walk_steps : 1,
            lambda_project,
            lambda_outside_barrier,
            outside_distance_threshold,
            lambda_scale_min,
            lambda_scale_max,
            rho,
            lambda_normal);

        final_mesh_surface_reduce_kernel<<<1, block_size, 0, stream>>>(
            temp_buffer,
            soft_count,
            soft_weight_sum,
            tracked_count,
            loss_out,
            num_blocks);
    }

} // namespace lfs::training::kernels
