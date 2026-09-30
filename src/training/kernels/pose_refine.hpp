/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_pose_refine_accumulate_w2c(
        const float* grad_w2c,
        const float* base_c2w,
        const float* relative_c2w,
        const float* delta,
        float* grad_delta,
        int camera_index,
        cudaStream_t stream);

    void launch_pose_refine_regularization(
        const float* delta,
        float* grad_delta,
        float* loss,
        int num_cameras,
        float trans_l2,
        float rot_l2,
        cudaStream_t stream);

    void launch_pose_refine_adam_step(
        float* delta,
        float* exp_avg,
        float* exp_avg_sq,
        float* grad_delta,
        int num_cameras,
        float lr_trans,
        float lr_rot,
        float beta1,
        float beta2,
        float eps,
        int step,
        float max_trans,
        float max_rot_rad,
        cudaStream_t stream);

} // namespace lfs::training::kernels
