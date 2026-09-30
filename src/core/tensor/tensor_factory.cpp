/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"

namespace lfs::core {

    Tensor Tensor::empty(TensorShape shape, Device device, DataType dtype, bool use_pinned) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.use_pinned = use_pinned;
        args.args = std::monostate{};
        return load(LoadOp::Empty, args);
    }

    Tensor Tensor::empty_unpinned(TensorShape shape, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = Device::CPU;
        args.dtype = dtype;
        args.use_pinned = false;
        args.args = std::monostate{};
        return load(LoadOp::Empty, args);
    }

    Tensor Tensor::zeros(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = 0.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::ones(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = 1.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::full(TensorShape shape, float value, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = value;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::full_bool(TensorShape shape, bool value, Device device) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = DataType::Bool;
        args.args = value ? 1.0f : 0.0f;
        return load(LoadOp::Const, args);
    }

    Tensor Tensor::zeros_bool(TensorShape shape, Device device) {
        return full_bool(shape, false, device);
    }

    Tensor Tensor::ones_bool(TensorShape shape, Device device) {
        return full_bool(shape, true, device);
    }

    Tensor Tensor::rand(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{0.0f, 1.0f};
        return load(LoadOp::Random, args);
    }

    Tensor Tensor::randn(TensorShape shape, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{0.0f, 1.0f};
        return load(LoadOp::Normal, args);
    }

    Tensor Tensor::uniform(TensorShape shape, float low, float high, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{low, high};
        return load(LoadOp::Random, args);
    }

    Tensor Tensor::normal(TensorShape shape, float mean, float std, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<float, float>{mean, std};
        return load(LoadOp::Normal, args);
    }

    Tensor Tensor::randint(TensorShape shape, int low, int high, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = std::pair<int, int>{low, high};
        return load(LoadOp::Randint, args);
    }

    Tensor Tensor::bernoulli(TensorShape shape, float p, Device device, DataType dtype) {
        LoadArgs args;
        args.shape = shape;
        args.device = device;
        args.dtype = dtype;
        args.args = p;
        return load(LoadOp::Bernoulli, args);
    }

    Tensor Tensor::arange(float end) {
        LoadArgs args;
        args.shape = TensorShape{};
        args.device = Device::CUDA;
        args.dtype = DataType::Float32;
        args.args = std::tuple<float, float, float>{0.0f, end, 1.0f};
        return load(LoadOp::Arange, args);
    }

    Tensor Tensor::arange(float start, float end, float step) {
        LoadArgs args;
        args.shape = TensorShape{};
        args.device = Device::CUDA;
        args.dtype = DataType::Float32;
        args.args = std::tuple<float, float, float>{start, end, step};
        return load(LoadOp::Arange, args);
    }

    Tensor Tensor::eye(size_t n, Device device) {
        LoadArgs args;
        args.shape = TensorShape{n, n};
        args.device = device;
        args.dtype = DataType::Float32;
        args.args = std::monostate{};
        return load(LoadOp::Eye, args);
    }

    Tensor Tensor::eye(size_t m, size_t n, Device device) {
        LoadArgs args;
        args.shape = TensorShape{m, n};
        args.device = device;
        args.dtype = DataType::Float32;
        args.args = std::monostate{};
        return load(LoadOp::Eye, args);
    }

} // namespace lfs::core
