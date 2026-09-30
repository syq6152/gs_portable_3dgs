/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <tuple>

namespace lfs::core {
    namespace cuda {

        /**
         * @brief GPU-accelerated k-means clustering using lfs::core::Tensor
         *
         * @param data Input data tensor [N, D] where N is number of points, D is dimensions
         * @param k Number of clusters
         * @param iterations Maximum number of iterations
         * @param tolerance Convergence tolerance (stop if centroids move less than this)
         * @return Tuple of (centroids [k, D], labels [N])
         */
        std::tuple<Tensor, Tensor> kmeans_new(
            const Tensor& data,
            int k,
            int iterations = 10,
            float tolerance = 1e-4f);

        /**
         * @brief GPU-accelerated 1D k-means clustering with optimal initialization
         *
         * @param data Input data tensor [N] or [N, 1]
         * @param k Number of clusters (typically 256 for SOG)
         * @param iterations Maximum number of iterations
         * @return Tuple of (sorted centroids [k, 1], labels [N])
         */
        std::tuple<Tensor, Tensor> kmeans_1d_new(
            const Tensor& data,
            int k = 256,
            int iterations = 10);

    } // namespace cuda
} // namespace lfs::core
