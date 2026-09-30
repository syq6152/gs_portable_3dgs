/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <tuple>

namespace lfs::core::cuda {

    /**
     * @brief K-means clustering using KD-tree for O(n log k) assignment
     *
     * @param data Input data [n_points, n_dims]
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @param tolerance Convergence tolerance
     * @return Tuple of (centroids [k, n_dims], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_kdtree(
        const Tensor& data,
        int k,
        int iterations = 10,
        float tolerance = 1e-4f);

    /**
     * @brief 1D k-means using binary search for O(n log k) assignment
     *
     * @param data Input data [n_points] or [n_points, 1]
     * @param k Number of clusters
     * @param iterations Maximum iterations
     * @return Tuple of (sorted centroids [k, 1], labels [n_points])
     */
    std::tuple<Tensor, Tensor> kmeans_1d_binary(
        const Tensor& data,
        int k,
        int iterations = 10);

} // namespace lfs::core::cuda
