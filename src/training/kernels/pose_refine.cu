/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pose_refine.hpp"

#include <cuda_runtime.h>

namespace lfs::training::kernels {
    namespace {

        __device__ void set_identity(float* m) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                m[i] = 0.0f;
            }
            m[0] = 1.0f;
            m[5] = 1.0f;
            m[10] = 1.0f;
            m[15] = 1.0f;
        }

        __device__ void mat4_mul(const float* a, const float* b, float* out) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        v += a[r * 4 + k] * b[k * 4 + c];
                    }
                    out[r * 4 + c] = v;
                }
            }
        }

        __device__ void exp_so3(const float* w, float* R) {
            const float wx = w[0];
            const float wy = w[1];
            const float wz = w[2];
            const float theta2 = wx * wx + wy * wy + wz * wz;
            float a = 1.0f;
            float b = 0.5f;
            if (theta2 > 1.0e-12f) {
                const float theta = sqrtf(theta2);
                a = sinf(theta) / theta;
                b = (1.0f - cosf(theta)) / theta2;
            }

            const float k00 = 0.0f;
            const float k01 = -wz;
            const float k02 = wy;
            const float k10 = wz;
            const float k11 = 0.0f;
            const float k12 = -wx;
            const float k20 = -wy;
            const float k21 = wx;
            const float k22 = 0.0f;

            const float k2_00 = k01 * k10 + k02 * k20;
            const float k2_01 = k02 * k21;
            const float k2_02 = k01 * k12;
            const float k2_10 = k12 * k20;
            const float k2_11 = k10 * k01 + k12 * k21;
            const float k2_12 = k10 * k02;
            const float k2_20 = k21 * k10;
            const float k2_21 = k20 * k01;
            const float k2_22 = k20 * k02 + k21 * k12;

            R[0] = 1.0f + a * k00 + b * k2_00;
            R[1] = a * k01 + b * k2_01;
            R[2] = a * k02 + b * k2_02;
            R[3] = a * k10 + b * k2_10;
            R[4] = 1.0f + a * k11 + b * k2_11;
            R[5] = a * k12 + b * k2_12;
            R[6] = a * k20 + b * k2_20;
            R[7] = a * k21 + b * k2_21;
            R[8] = 1.0f + a * k22 + b * k2_22;
        }

        __device__ void skew_matrix(const float* w, float* K) {
            K[0] = 0.0f;
            K[1] = -w[2];
            K[2] = w[1];
            K[3] = w[2];
            K[4] = 0.0f;
            K[5] = -w[0];
            K[6] = -w[1];
            K[7] = w[0];
            K[8] = 0.0f;
        }

        __device__ void mat3_mul(const float* a, const float* b, float* out) {
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        v += a[r * 3 + k] * b[k * 3 + c];
                    }
                    out[r * 3 + c] = v;
                }
            }
        }

        __device__ void right_jacobian_so3(const float* w, float* Jr) {
            float K[9];
            float K2[9];
            skew_matrix(w, K);
            mat3_mul(K, K, K2);

            const float theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            float a = 0.5f;
            float b = 1.0f / 6.0f;
            if (theta2 > 1.0e-8f) {
                const float theta = sqrtf(theta2);
                a = (1.0f - cosf(theta)) / theta2;
                b = (theta - sinf(theta)) / (theta2 * theta);
            }

            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                Jr[i] = -a * K[i] + b * K2[i];
            }
            Jr[0] += 1.0f;
            Jr[4] += 1.0f;
            Jr[8] += 1.0f;
        }

        __device__ void compose_c2w_from_delta(
            const float* base_c2w,
            const float* relative_c2w,
            const float* delta6,
            float* c2w_out) {

            float R_delta[9];
            exp_so3(delta6 + 3, R_delta);

            float corrected[16];
            set_identity(corrected);

            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        v += R_delta[r * 3 + k] * base_c2w[k * 4 + c];
                    }
                    corrected[r * 4 + c] = v;
                }
            }

            corrected[3] = base_c2w[3] + delta6[0];
            corrected[7] = base_c2w[7] + delta6[1];
            corrected[11] = base_c2w[11] + delta6[2];

            if (relative_c2w != nullptr) {
                mat4_mul(corrected, relative_c2w, c2w_out);
            } else {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    c2w_out[i] = corrected[i];
                }
            }
        }

        __global__ void accumulate_w2c_kernel(
            const float* grad_w2c,
            const float* base_c2w,
            const float* relative_c2w,
            const float* delta,
            float* grad_delta,
            const int camera_index) {
            if (threadIdx.x != 0 || blockIdx.x != 0) {
                return;
            }

            const float* delta_local = delta + camera_index * 6;

            float c2w[16];
            compose_c2w_from_delta(base_c2w, relative_c2w, delta_local, c2w);

            const float c_eff[3] = {c2w[3], c2w[7], c2w[11]};

            float g_c_eff[3];
            float G_R_eff[9];

            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                float v = 0.0f;
                #pragma unroll
                for (int i = 0; i < 3; ++i) {
                    v -= c2w[r * 4 + i] * grad_w2c[i * 4 + 3];
                }
                g_c_eff[r] = v;
            }

            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    G_R_eff[r * 3 + c] =
                        grad_w2c[c * 4 + r] - c_eff[r] * grad_w2c[c * 4 + 3];
                }
            }

            float g_c_base[3] = {g_c_eff[0], g_c_eff[1], g_c_eff[2]};
            float G_R_base[9];

            if (relative_c2w != nullptr) {
                const float c_rel[3] = {relative_c2w[3], relative_c2w[7], relative_c2w[11]};

                #pragma unroll
                for (int r = 0; r < 3; ++r) {
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        float v = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 3; ++k) {
                            v += G_R_eff[r * 3 + k] * relative_c2w[c * 4 + k];
                        }
                        G_R_base[r * 3 + c] = v + g_c_eff[r] * c_rel[c];
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 9; ++i) {
                    G_R_base[i] = G_R_eff[i];
                }
            }

            float R0[9];
            R0[0] = base_c2w[0];
            R0[1] = base_c2w[1];
            R0[2] = base_c2w[2];
            R0[3] = base_c2w[4];
            R0[4] = base_c2w[5];
            R0[5] = base_c2w[6];
            R0[6] = base_c2w[8];
            R0[7] = base_c2w[9];
            R0[8] = base_c2w[10];

            float E[9];
            exp_so3(delta_local + 3, E);

            float G_E[9];
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        v += G_R_base[r * 3 + k] * R0[c * 3 + k];
                    }
                    G_E[r * 3 + c] = v;
                }
            }

            float M[9];
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 3; ++k) {
                        v += E[k * 3 + r] * G_E[k * 3 + c];
                    }
                    M[r * 3 + c] = v;
                }
            }

            const float b[3] = {
                M[7] - M[5],
                M[2] - M[6],
                M[3] - M[1]};

            float Jr[9];
            right_jacobian_so3(delta_local + 3, Jr);

            float grad_rot[3];
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                grad_rot[i] = Jr[i] * b[0] + Jr[3 + i] * b[1] + Jr[6 + i] * b[2];
            }

            float* g = grad_delta + camera_index * 6;
            atomicAdd(&g[0], g_c_base[0]);
            atomicAdd(&g[1], g_c_base[1]);
            atomicAdd(&g[2], g_c_base[2]);
            atomicAdd(&g[3], grad_rot[0]);
            atomicAdd(&g[4], grad_rot[1]);
            atomicAdd(&g[5], grad_rot[2]);
        }

        __global__ void regularization_kernel(
            const float* delta,
            float* grad_delta,
            float* loss,
            const int num_cameras,
            const float trans_l2,
            const float rot_l2) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= num_cameras) {
                return;
            }

            const float inv_n = num_cameras > 0 ? 1.0f / static_cast<float>(num_cameras) : 0.0f;
            const float* d = delta + i * 6;
            float* g = grad_delta + i * 6;

            const float trans_norm = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + 1.0e-20f);
            const float rot_norm = sqrtf(d[3] * d[3] + d[4] * d[4] + d[5] * d[5] + 1.0e-20f);
            const float trans_weight = trans_l2 * inv_n;
            const float rot_weight = rot_l2 * inv_n;

            if (trans_l2 > 0.0f) {
                atomicAdd(loss, trans_weight * trans_norm);
                atomicAdd(&g[0], trans_weight * d[0] / trans_norm);
                atomicAdd(&g[1], trans_weight * d[1] / trans_norm);
                atomicAdd(&g[2], trans_weight * d[2] / trans_norm);
            }
            if (rot_l2 > 0.0f) {
                atomicAdd(loss, rot_weight * rot_norm);
                atomicAdd(&g[3], rot_weight * d[3] / rot_norm);
                atomicAdd(&g[4], rot_weight * d[4] / rot_norm);
                atomicAdd(&g[5], rot_weight * d[5] / rot_norm);
            }
        }

        __global__ void adam_step_kernel(
            float* delta,
            float* exp_avg,
            float* exp_avg_sq,
            float* grad_delta,
            const int num_cameras,
            const float lr_trans,
            const float lr_rot,
            const float beta1,
            const float beta2,
            const float eps,
            const float bias_correction1_rcp,
            const float bias_correction2_sqrt_rcp,
            const float max_trans,
            const float max_rot_rad) {
            const int camera_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (camera_index >= num_cameras) {
                return;
            }

            float* d = delta + camera_index * 6;
            float* m = exp_avg + camera_index * 6;
            float* v = exp_avg_sq + camera_index * 6;
            float* g = grad_delta + camera_index * 6;

            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                const float lr = (k < 3) ? lr_trans : lr_rot;
                const float grad = g[k];
                m[k] = beta1 * m[k] + (1.0f - beta1) * grad;
                v[k] = beta2 * v[k] + (1.0f - beta2) * grad * grad;
                if (lr > 0.0f) {
                    const float denom = sqrtf(v[k]) * bias_correction2_sqrt_rcp + eps;
                    d[k] -= lr * (m[k] * bias_correction1_rcp) / denom;
                }
                g[k] = 0.0f;
            }

            if (max_trans > 0.0f) {
                const float n = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                if (n > max_trans) {
                    const float scale = max_trans / fmaxf(n, 1.0e-20f);
                    d[0] *= scale;
                    d[1] *= scale;
                    d[2] *= scale;
                }
            }
            if (max_rot_rad > 0.0f) {
                const float n = sqrtf(d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
                if (n > max_rot_rad) {
                    const float scale = max_rot_rad / fmaxf(n, 1.0e-20f);
                    d[3] *= scale;
                    d[4] *= scale;
                    d[5] *= scale;
                }
            }
        }

    } // namespace

    void launch_pose_refine_accumulate_w2c(
        const float* grad_w2c,
        const float* base_c2w,
        const float* relative_c2w,
        const float* delta,
        float* grad_delta,
        const int camera_index,
        cudaStream_t stream) {
        if (!grad_w2c || !base_c2w || !delta || !grad_delta || camera_index < 0) {
            return;
        }
        accumulate_w2c_kernel<<<1, 1, 0, stream>>>(
            grad_w2c,
            base_c2w,
            relative_c2w,
            delta,
            grad_delta,
            camera_index);
    }

    void launch_pose_refine_regularization(
        const float* delta,
        float* grad_delta,
        float* loss,
        const int num_cameras,
        const float trans_l2,
        const float rot_l2,
        cudaStream_t stream) {
        if (!delta || !grad_delta || !loss || num_cameras <= 0) {
            return;
        }
        const int threads = 128;
        const int blocks = (num_cameras + threads - 1) / threads;
        regularization_kernel<<<blocks, threads, 0, stream>>>(
            delta, grad_delta, loss, num_cameras, trans_l2, rot_l2);
    }

    void launch_pose_refine_adam_step(
        float* delta,
        float* exp_avg,
        float* exp_avg_sq,
        float* grad_delta,
        const int num_cameras,
        const float lr_trans,
        const float lr_rot,
        const float beta1,
        const float beta2,
        const float eps,
        const int step,
        const float max_trans,
        const float max_rot_rad,
        cudaStream_t stream) {
        if (!delta || !exp_avg || !exp_avg_sq || !grad_delta || num_cameras <= 0 || step <= 0) {
            return;
        }
        const float bias_correction1_rcp = 1.0f / (1.0f - powf(beta1, static_cast<float>(step)));
        const float bias_correction2_sqrt_rcp = 1.0f / sqrtf(1.0f - powf(beta2, static_cast<float>(step)));
        const int threads = 128;
        const int blocks = (num_cameras + threads - 1) / threads;
        adam_step_kernel<<<blocks, threads, 0, stream>>>(
            delta,
            exp_avg,
            exp_avg_sq,
            grad_delta,
            num_cameras,
            lr_trans,
            lr_rot,
            beta1,
            beta2,
            eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp,
            max_trans,
            max_rot_rad);
    }

} // namespace lfs::training::kernels
