/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2001-2002 Enthought, Inc. 2003, SciPy Developers
 * SPDX-FileCopyrightText: 2001-2020 Free Software Foundation, Inc.
 * SPDX-FileCopyrightText: 1994 Hewlett-Packard Company
 * SPDX-FileCopyrightText: 1997 Silicon Graphics Computer Systems, Inc.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Specialized non-periodic, exact, Euclidean k=1 adaptation of SciPy 1.15.3
 * spatial/ckdtree/src/{build,query}.cxx (BSD-3-Clause). The frozen Windows
 * SciPy wheel uses GCC 10.3; its median-partition ordering is reproduced
 * explicitly, since MSVC's std::nth_element chooses different duplicate keys.
 * See docs/swaptexture_m4/scipy_notice.txt for the upstream BSD notice.
 *
 * Selection ordering follows GCC 10.3 libstdc++ stl_algo.h/stl_heap.h
 * (GPL-3.0-or-later WITH GCC-exception-3.1); see
 * docs/swaptexture_m4/gcc_selection_notice.txt for upstream notices.
 */
#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace lfs::preprocess::detail {
    // Semantic adaptation of libstdc++ 10.3 __heap_select for the index
    // permutation used by this tree.  Keeping the child-selection and
    // equal-key comparisons explicit avoids MSVC's implementation-dependent
    // heap tie ordering in the SciPy compatibility path.
    template <class Less>
    inline void scipy_gcc_heap_select(std::vector<size_t>& order, size_t first, size_t middle, size_t last, Less less) {
        const size_t length = middle - first;
        if (length == 0)
            return;
        auto adjust = [&](size_t hole, size_t count, size_t value) {
            const size_t top = hole;
            size_t child = hole;
            while (child < (count - 1) / 2) {
                child = 2 * (child + 1);
                if (less(order[first + child], order[first + child - 1]))
                    --child;
                order[first + hole] = order[first + child];
                hole = child;
            }
            if ((count & 1U) == 0 && child == (count - 2) / 2) {
                child = 2 * (child + 1);
                order[first + hole] = order[first + child - 1];
                hole = child - 1;
            }
            while (hole > top) {
                const size_t parent = (hole - 1) / 2;
                if (!less(order[first + parent], value))
                    break;
                order[first + hole] = order[first + parent];
                hole = parent;
            }
            order[first + hole] = value;
        };
        if (length > 1)
            if (length > 1)
                for (size_t parent = (length - 2) / 2;; --parent) {
                    const size_t value = order[first + parent];
                    adjust(parent, length, value);
                    if (parent == 0)
                        break;
                }
        for (size_t i = middle; i < last; ++i) {
            if (!less(order[i], order[first]))
                continue;
            const size_t value = order[i];
            order[i] = order[first];
            adjust(0, length, value);
        }
    }

    // Equal-distance nearest neighbors are NOT replaced. Their first encounter
    // is observable when SIFT has several descriptors at the same pixel.
    class ScipyNearestTree {
        using Point = std::array<double, 3>;
        struct Node {
            size_t begin = 0, end = 0, left = 0, right = 0;
            int axis = -1;
            double split = 0;
        };
        const std::vector<Point>& points_;
        std::vector<size_t> order_;
        std::vector<Node> nodes_;
        Point minimum_{}, maximum_{};

        template <class Less>
        void select_median(size_t begin, size_t middle, size_t end, Less less) {
            size_t budget = 2 * (std::bit_width(end - begin) - 1);
            while (end - begin > 3) {
                if (budget-- == 0) {
                    // This is GCC 10.3's __introselect fallback: heap-select
                    // [first, nth + 1), then swap first and nth.  Do not use
                    // std::partial_sort here; MSVC may choose different ties.
                    scipy_gcc_heap_select(order_, begin, middle + 1, end, less);
                    std::swap(order_[begin], order_[middle]);
                    return;
                }
                const size_t a = begin + 1, b = begin + (end - begin) / 2, c = end - 1;
                size_t pivot;
                if (less(order_[a], order_[b])) {
                    pivot = less(order_[b], order_[c]) ? b : less(order_[a], order_[c]) ? c
                                                                                        : a;
                } else {
                    pivot = less(order_[a], order_[c]) ? a : less(order_[b], order_[c]) ? c
                                                                                        : b;
                }
                std::swap(order_[begin], order_[pivot]);
                size_t left = begin + 1, right = end;
                for (;;) {
                    while (less(order_[left], order_[begin]))
                        ++left;
                    do {
                        --right;
                    } while (less(order_[begin], order_[right]));
                    if (left >= right)
                        break;
                    std::swap(order_[left++], order_[right]);
                }
                if (left <= middle)
                    begin = left;
                else
                    end = left;
            }
            for (size_t i = begin + 1; i < end; ++i) {
                const auto value = order_[i];
                size_t pos = i;
                while (pos > begin && less(value, order_[pos - 1])) {
                    order_[pos] = order_[pos - 1];
                    --pos;
                }
                order_[pos] = value;
            }
        }
        size_t partition(size_t begin, size_t end, int axis, double pivot) {
            for (;;) {
                while (begin < end && points_[order_[begin]][axis] < pivot)
                    ++begin;
                if (begin == end)
                    return begin;
                do {
                    --end;
                } while (begin < end && points_[order_[end]][axis] >= pivot);
                if (begin == end)
                    return begin;
                std::swap(order_[begin++], order_[end]);
            }
        }
        size_t build(size_t begin, size_t end) {
            const size_t index = nodes_.size();
            nodes_.push_back({begin, end});
            if (end - begin <= 16)
                return index;
            Point lo = points_[order_[begin]], hi = lo;
            for (size_t i = begin + 1; i < end; ++i)
                for (int axis = 0; axis < 3; ++axis) {
                    lo[axis] = std::min(lo[axis], points_[order_[i]][axis]);
                    hi[axis] = std::max(hi[axis], points_[order_[i]][axis]);
                }
            int axis = 0;
            double spread = 0;
            for (int k = 0; k < 3; ++k)
                if (hi[k] - lo[k] > spread) {
                    axis = k;
                    spread = hi[k] - lo[k];
                }
            if (spread == 0)
                return index;
            const auto less = [&](size_t a, size_t b) { return points_[a][axis] < points_[b][axis]; };
            const auto middle = begin + (end - begin) / 2;
            select_median(begin, middle, end, less);
            double split = points_[order_[middle]][axis];
            auto boundary = partition(begin, middle, axis, split);
            if (boundary == begin) {
                const auto min = *std::min_element(order_.begin() + begin, order_.begin() + end, less);
                split = std::nextafter(points_[min][axis], std::numeric_limits<double>::infinity());
                boundary = partition(begin, end, axis, split);
            }
            if (boundary == end) {
                const auto max = *std::max_element(order_.begin() + begin, order_.begin() + end, less);
                split = points_[max][axis];
                boundary = partition(begin, end, axis, split);
            }
            if (boundary == begin || boundary == end)
                return index;
            const auto left = build(begin, boundary), right = build(boundary, end);
            nodes_[index].axis = axis;
            nodes_[index].split = split;
            nodes_[index].left = left;
            nodes_[index].right = right;
            return index;
        }
        struct Visit {
            size_t node = 0;
            double minimum = 0;
            Point sides{};
        };
        // SciPy's priority queue retains its exact equal-priority behavior;
        // std::priority_queue is not specified to choose the same tied child.
        static void push(std::vector<Visit>& heap, Visit item) {
            heap.push_back(item);
            size_t i = heap.size() - 1;
            while (i && heap[i].minimum < heap[(i - 1) / 2].minimum) {
                std::swap(heap[i], heap[(i - 1) / 2]);
                i = (i - 1) / 2;
            }
        }
        static Visit pop(std::vector<Visit>& heap) {
            const auto result = heap.front();
            heap.front() = heap.back();
            heap.pop_back();
            size_t i = 0;
            for (;;) {
                const auto left = 2 * i + 1, right = left + 1;
                if (left >= heap.size())
                    break;
                const auto child = right < heap.size() && heap[left].minimum > heap[right].minimum ? right : left;
                if (!(heap[i].minimum > heap[child].minimum))
                    break;
                std::swap(heap[i], heap[child]);
                i = child;
            }
            return result;
        }

    public:
        explicit ScipyNearestTree(const std::vector<Point>& points) : points_(points), order_(points.size()) {
            std::iota(order_.begin(), order_.end(), 0);
            if (points.empty())
                return;
            minimum_ = maximum_ = points.front();
            for (const auto& p : points)
                for (int k = 0; k < 3; ++k) {
                    minimum_[k] = std::min(minimum_[k], p[k]);
                    maximum_[k] = std::max(maximum_[k], p[k]);
                }
            build(0, points.size());
        }
        size_t knnSearch(const double* query, size_t count, size_t* index, double* squared_distance) const {
            if (count != 1 || points_.empty())
                return 0;
            Visit current;
            for (int k = 0; k < 3; ++k) {
                const double distance = std::max({0., minimum_[k] - query[k], query[k] - maximum_[k]});
                current.sides[k] = distance * distance;
                current.minimum += current.sides[k];
            }
            std::vector<Visit> queue;
            queue.reserve(32);
            double best = std::numeric_limits<double>::infinity();
            size_t winner = 0;
            for (;;) {
                const auto& node = nodes_[current.node];
                if (node.axis < 0) {
                    for (size_t i = node.begin; i < node.end; ++i) {
                        const auto& p = points_[order_[i]];
                        double d = 0;
                        for (int k = 0; k < 3; ++k) {
                            const double delta = p[k] - query[k];
                            d += delta * delta;
                        }
                        if (d < best) {
                            best = d;
                            winner = order_[i];
                        }
                    }
                    if (queue.empty())
                        break;
                    current = pop(queue);
                } else {
                    if (current.minimum > best)
                        break;
                    auto far = current;
                    const auto axis = node.axis;
                    if (query[axis] < node.split) {
                        current.node = node.left;
                        far.node = node.right;
                    } else {
                        current.node = node.right;
                        far.node = node.left;
                    }
                    const double delta = query[axis] - node.split, side = delta * delta;
                    far.minimum += side - far.sides[axis];
                    far.sides[axis] = side;
                    if (current.minimum > far.minimum)
                        std::swap(current, far);
                    if (far.minimum <= best)
                        push(queue, far);
                }
            }
            *index = winner;
            *squared_distance = best;
            return 1;
        }
    };
} // namespace lfs::preprocess::detail
