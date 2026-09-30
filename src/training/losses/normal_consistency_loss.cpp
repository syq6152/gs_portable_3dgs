/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "normal_consistency_loss.hpp"
#include "lfs/kernels/normal_consistency.cuh"
#include <format>

namespace lfs::training::losses {

    std::expected<std::pair<lfs::core::Tensor, NormalConsistencyLoss::Context>, std::string>
    NormalConsistencyLoss::forward(
        const lfs::core::Tensor& render_normal,
        const lfs::core::Tensor& depth_normal,
        const lfs::core::Tensor& depth,
        const Params& params) {
        try {
            if (params.lambda <= 0.0f) {
                Context ctx;
                ctx.loss_value = 0.0f;
                return std::pair{lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA), ctx};
            }

            // Validate render_normal: [3, H, W] on CUDA
            if (!render_normal.is_valid() || render_normal.is_empty()) {
                return std::unexpected("render_normal is invalid or empty");
            }
            if (render_normal.ndim() != 3 || render_normal.shape()[0] != 3) {
                return std::unexpected(std::format("render_normal must be [3,H,W], got ndim={}", render_normal.ndim()));
            }

            const int H = static_cast<int>(render_normal.shape()[1]);
            const int W = static_cast<int>(render_normal.shape()[2]);

            // Ensure depth_normal is [3, H, W] on CUDA
            auto dn = depth_normal;
            if (!dn.is_valid() || dn.is_empty()) {
                return std::unexpected("depth_normal is invalid or empty");
            }
            if (dn.device() != lfs::core::Device::CUDA) {
                dn = dn.cuda();
            }
            dn = dn.contiguous();
            if (dn.ndim() != 3 || dn.shape()[0] != 3 ||
                static_cast<int>(dn.shape()[1]) != H || static_cast<int>(dn.shape()[2]) != W) {
                return std::unexpected(std::format("depth_normal shape mismatch: expected [3,{},{}]", H, W));
            }

            // Ensure depth is [H, W] on CUDA
            auto d = depth;
            if (!d.is_valid() || d.is_empty()) {
                return std::unexpected("depth is invalid or empty");
            }
            if (d.device() != lfs::core::Device::CUDA) {
                d = d.cuda();
            }
            // Squeeze leading dimensions
            if (d.ndim() == 3 && d.shape()[0] == 1) {
                d = d.squeeze(0);
            }
            if (d.ndim() == 4 && d.shape()[0] == 1) {
                d = d.squeeze(0);
                if (d.shape()[0] == 1) d = d.squeeze(0);
            }
            d = d.contiguous();
            if (d.ndim() != 2 || static_cast<int>(d.shape()[0]) != H || static_cast<int>(d.shape()[1]) != W) {
                return std::unexpected(std::format("depth shape mismatch: expected [{},{}]", H, W));
            }

            // Ensure render_normal is contiguous on CUDA
            auto rn = render_normal;
            if (rn.device() != lfs::core::Device::CUDA) {
                rn = rn.cuda();
            }
            rn = rn.contiguous();

            // Allocate outputs
            auto grad_rn = lfs::core::Tensor::empty({3, static_cast<size_t>(H), static_cast<size_t>(W)}, lfs::core::Device::CUDA);
            auto grad_dn = lfs::core::Tensor::empty({3, static_cast<size_t>(H), static_cast<size_t>(W)}, lfs::core::Device::CUDA);
            auto loss_tensor = lfs::core::Tensor::empty({1}, lfs::core::Device::CUDA);

            // Temp buffers for block-level reduction
            const int HW = H * W;
            const int num_blocks = (HW + 255) / 256;
            auto temp_sum = lfs::core::Tensor::empty({static_cast<size_t>(num_blocks)}, lfs::core::Device::CUDA);
            auto temp_count = lfs::core::Tensor::empty({static_cast<size_t>(num_blocks)}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);

            kernels::launch_normal_consistency_loss(
                rn.ptr<float>(),
                dn.ptr<float>(),
                d.ptr<float>(),
                grad_rn.ptr<float>(),
                grad_dn.ptr<float>(),
                loss_tensor.ptr<float>(),
                temp_sum.ptr<float>(),
                temp_count.ptr<int>(),
                H, W,
                nullptr);

            Context ctx;
            ctx.grad_render_normal = grad_rn;
            ctx.grad_depth_normal = grad_dn;
            return std::pair{loss_tensor, ctx};

        } catch (const std::exception& e) {
            return std::unexpected(std::format("NormalConsistencyLoss::forward error: {}", e.what()));
        }
    }

} // namespace lfs::training::losses
