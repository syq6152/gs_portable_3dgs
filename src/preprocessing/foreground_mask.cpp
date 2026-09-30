/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2020 Daniel Gatis
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Native adaptation of rembg 2.0.67 u2netp normalization/postprocessing.
// Original MIT notice: docs/swaptexture_m5/rembg_notice.txt.
#include "preprocessing/foreground_mask.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <bcrypt.h>
#endif

namespace lfs::preprocess {
    namespace {
        constexpr std::size_t model_bytes = 4574861;
        constexpr int model_size = 320;
        constexpr std::size_t plane_size = model_size * model_size;
        bool valid_shape(const CpuImage& image) {
            return image.width > 0 && image.height > 0 && (image.channels == 1 || image.channels == 3) &&
                   uint64_t(image.width) * image.height * 3 <= 1024ull * 1024 * 1024 &&
                   image.pixels.size() == std::size_t(image.width) * image.height * image.channels;
        }
        std::expected<void, Error> progress(const ExecutionContext& context, float fraction) {
            if (context.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Foreground mask cancelled"});
            try {
                if (context.on_progress &&
                    !context.on_progress(Stage::ImageProcessing, fraction, "Generating incremental foreground mask"))
                    return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Foreground mask callback cancelled"});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::CallbackFailure, .message = "Foreground mask callback failed"});
            }
            return {};
        }
        std::expected<std::vector<char>, Error> verified_model(const std::filesystem::path& path) {
#ifdef _WIN32
            std::error_code ec;
            if (!std::filesystem::is_regular_file(path, ec))
                return std::unexpected(Error{.code = ErrorCode::RuntimeMissing, .message = "Local u2netp ONNX model is missing"});
            if (std::filesystem::file_size(path, ec) != model_bytes || ec)
                return std::unexpected(Error{.code = ErrorCode::RuntimeManifestMismatch, .message = "u2netp model size differs from the frozen model"});
            std::ifstream input(path, std::ios::binary);
            std::vector<char> bytes(model_bytes);
            input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            if (!input || input.peek() != std::char_traits<char>::eof())
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot read complete u2netp model"});
            struct Hash {
                BCRYPT_ALG_HANDLE algorithm = nullptr;
                BCRYPT_HASH_HANDLE hash = nullptr;
                ~Hash() {
                    if (hash)
                        BCryptDestroyHash(hash);
                    if (algorithm)
                        BCryptCloseAlgorithmProvider(algorithm, 0);
                }
            } hash;
            std::array<unsigned char, 32> digest{};
            if (BCryptOpenAlgorithmProvider(&hash.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
                BCryptCreateHash(hash.algorithm, &hash.hash, nullptr, 0, nullptr, 0, 0) < 0 ||
                BCryptHashData(hash.hash, reinterpret_cast<PUCHAR>(bytes.data()), static_cast<ULONG>(bytes.size()), 0) < 0 ||
                BCryptFinishHash(hash.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0)
                return std::unexpected(Error{.code = ErrorCode::IncompatibleRuntime, .message = "Cannot verify u2netp SHA-256"});
            std::string actual;
            for (const auto byte : digest) {
                actual += "0123456789abcdef"[byte >> 4];
                actual += "0123456789abcdef"[byte & 15];
            }
            if (actual != u2netp_model_sha256)
                return std::unexpected(Error{.code = ErrorCode::RuntimeManifestMismatch, .message = "u2netp SHA-256 differs from the frozen model"});
            return bytes;
#else
            (void)path;
            return std::unexpected(Error{.code = ErrorCode::PlatformUnsupported, .message = "Verified foreground provider currently requires Windows"});
#endif
        }
        class U2netpCpuProvider final : public IForegroundMaskProvider {
            Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "lfs_foreground"};
            Ort::SessionOptions options_;
            Ort::Session session_{nullptr};
            std::string input_name_, output_name_;
            std::mutex mutex_;

        public:
            explicit U2netpCpuProvider(const std::vector<char>& model) {
                // No execution provider is appended: ORT's built-in CPU provider.
                options_.SetIntraOpNumThreads(1);
                options_.SetInterOpNumThreads(1);
                session_ = Ort::Session(environment_, model.data(), model.size(), options_);
                if (session_.GetInputCount() != 1 || session_.GetOutputCount() < 1)
                    throw std::runtime_error("Unexpected u2netp input/output count");
                const auto input_info = session_.GetInputTypeInfo(0);
                const auto input_tensor = input_info.GetTensorTypeAndShapeInfo();
                const auto output_info = session_.GetOutputTypeInfo(0);
                const auto output_tensor = output_info.GetTensorTypeAndShapeInfo();
                if (input_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    input_tensor.GetShape() != std::vector<int64_t>{1, 3, model_size, model_size} ||
                    output_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    output_tensor.GetShape() != std::vector<int64_t>{1, 1, model_size, model_size})
                    throw std::runtime_error("Unexpected u2netp tensor shape/type");
                Ort::AllocatorWithDefaultOptions allocator;
                input_name_ = session_.GetInputNameAllocated(0, allocator).get();
                output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
            }
            std::expected<CpuImage, Error> mask(const CpuImage& image, const ForegroundMaskOptions& options,
                                                const ExecutionContext& context) override {
                std::unique_lock lock(mutex_, std::try_to_lock);
                if (!lock)
                    return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Foreground provider is already in use"});
                if (options.erosion_pixels < 0 || options.erosion_pixels > 4096 || context.timeout.count() < 0)
                    return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Invalid foreground erosion or timeout"});
                if (auto check = progress(context, 0); !check)
                    return std::unexpected(check.error());
                auto input = u2netp_input_tensor(image);
                if (!input)
                    return std::unexpected(input.error());
                if (auto check = progress(context, .2F); !check)
                    return std::unexpected(check.error());
                std::atomic_bool timed_out = false;
                try {
                    Ort::RunOptions run_options;
                    auto terminate = [&] {
                        try {
                            run_options.SetTerminate();
                        } catch (...) {
                            // Cancellation remains visible through the stop token.
                        }
                    };
                    std::stop_callback cancellation(context.stop_token, terminate);
                    std::mutex timer_mutex;
                    std::condition_variable_any timer_wake;
                    std::jthread timer;
                    if (context.timeout.count() > 0)
                        timer = std::jthread([&](std::stop_token stop) {
                            std::unique_lock timer_lock(timer_mutex);
                            timer_wake.wait_for(timer_lock, stop, context.timeout, [] { return false; });
                            if (!stop.stop_requested()) {
                                timed_out = true;
                                terminate();
                            }
                        });
                    const std::array<int64_t, 4> shape{1, 3, model_size, model_size};
                    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                    auto tensor = Ort::Value::CreateTensor<float>(memory, input->data(), input->size(), shape.data(), shape.size());
                    const char* inputs[]{input_name_.c_str()};
                    const char* outputs[]{output_name_.c_str()};
                    auto result = session_.Run(run_options, inputs, &tensor, 1, outputs, 1);
                    timer.request_stop();
                    if (timer.joinable())
                        timer.join();
                    if (timed_out)
                        return std::unexpected(Error{.code = ErrorCode::ProcessTimeout, .message = "Foreground inference timed out"});
                    if (auto check = progress(context, .8F); !check)
                        return std::unexpected(check.error());
                    if (result.size() != 1 || !result[0].IsTensor() ||
                        result[0].GetTensorTypeAndShapeInfo().GetElementCount() != plane_size)
                        return std::unexpected(Error{.code = ErrorCode::IncompatibleRuntime, .message = "Unexpected foreground output tensor"});
                    auto mask = u2netp_output_mask({result[0].GetTensorData<float>(), plane_size}, image.width, image.height, options);
                    if (!mask)
                        return mask;
                    if (auto check = progress(context, 1); !check)
                        return std::unexpected(check.error());
                    return mask;
                } catch (const std::exception& e) {
                    if (context.stop_token.stop_requested())
                        return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Foreground inference cancelled"});
                    if (timed_out)
                        return std::unexpected(Error{.code = ErrorCode::ProcessTimeout, .message = "Foreground inference timed out"});
                    return std::unexpected(Error{.code = ErrorCode::ProcessFailed, .message = std::string("Foreground inference failed: ") + e.what()});
                }
            }
        };
    } // namespace

    std::expected<std::vector<float>, Error> u2netp_input_tensor(const CpuImage& image) try {
        if (!valid_shape(image))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Foreground input must be a valid RGB or grayscale image"});
        CpuImage rgb = image;
        if (image.channels == 1) {
            rgb.channels = 3;
            rgb.pixels.resize(image.pixels.size() * 3);
            for (std::size_t i = 0; i < image.pixels.size(); ++i)
                std::fill_n(rgb.pixels.begin() + i * 3, 3, image.pixels[i]);
        }
        auto resized = resize_lanczos(rgb, model_size, model_size);
        if (!resized)
            return std::unexpected(resized.error());
        const double maximum = std::max(double(*std::max_element(resized->pixels.begin(), resized->pixels.end())), 1e-6);
        constexpr double mean[]{.485, .456, .406}, deviation[]{.229, .224, .225};
        std::vector<float> result(plane_size * 3);
        for (std::size_t i = 0; i < plane_size; ++i)
            for (std::size_t channel = 0; channel < 3; ++channel)
                result[channel * plane_size + i] = static_cast<float>((double(resized->pixels[i * 3 + channel]) / maximum - mean[channel]) / deviation[channel]);
        return result;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = e.what()});
    }

    std::expected<CpuImage, Error> u2netp_output_mask(
        std::span<const float> prediction, int width, int height, const ForegroundMaskOptions& options) try {
        if (prediction.size() != plane_size || width <= 0 || height <= 0 ||
            uint64_t(width) * height > 1024ull * 1024 * 1024 || options.erosion_pixels < 0 || options.erosion_pixels > 4096 ||
            !std::all_of(prediction.begin(), prediction.end(), [](float value) { return std::isfinite(value); }))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Invalid foreground output dimensions, prediction or erosion"});
        const auto [minimum, maximum] = std::minmax_element(prediction.begin(), prediction.end());
        const float range = *maximum - *minimum;
        if (!std::isfinite(range))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Nonfinite foreground prediction range"});
        CpuImage alpha{model_size, model_size, 1, std::vector<uint8_t>(plane_size, 0)};
        if (range > 0)
            for (std::size_t i = 0; i < plane_size; ++i) {
                const float normalized = (prediction[i] - *minimum) / range;
                const float scaled = normalized * 255.F;
                alpha.pixels[i] = static_cast<uint8_t>(std::clamp(scaled, 0.F, 255.F));
            }
        auto mask = resize_lanczos(alpha, width, height);
        if (!mask)
            return mask;
        for (auto& pixel : mask->pixels)
            pixel = pixel > 0 ? 255 : 0;
        if (options.erosion_pixels) {
            cv::Mat data(height, width, CV_8UC1, mask->pixels.data());
            const int kernel = options.erosion_pixels * 2 + 1;
            cv::erode(data, data, cv::getStructuringElement(cv::MORPH_RECT, {kernel, kernel}), {-1, -1}, 1, cv::BORDER_REPLICATE);
        }
        return mask;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = e.what()});
    }

    std::expected<std::unique_ptr<IForegroundMaskProvider>, Error> create_u2netp_cpu_provider(
        const std::filesystem::path& path, const ExecutionContext& context) try {
        if (context.stop_token.stop_requested())
            return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Foreground provider creation cancelled"});
        auto model = verified_model(path);
        if (!model)
            return std::unexpected(model.error());
        auto provider = std::make_unique<U2netpCpuProvider>(*model);
        if (context.stop_token.stop_requested())
            return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Foreground provider creation cancelled"});
        return provider;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IncompatibleRuntime, .message = std::string("Cannot initialize CPU foreground provider: ") + e.what()});
    }
} // namespace lfs::preprocess
