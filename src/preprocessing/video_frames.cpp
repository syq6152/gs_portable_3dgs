/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/video_frames.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

namespace lfs::preprocess {
    namespace {
        namespace fs = std::filesystem;
        using Clock = std::chrono::steady_clock;
        Error invalid(std::string message) {
            return {.code = ErrorCode::InvalidDataset, .message = std::move(message)};
        }
        std::string utf8(const fs::path& path) {
            const auto value = path.u8string();
            return {reinterpret_cast<const char*>(value.data()), value.size()};
        }
        std::expected<void, Error> check_progress(const ExecutionContext& ctx, Clock::time_point start, float fraction) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Video extraction cancelled"});
            if (ctx.timeout.count() > 0 && Clock::now() - start >= ctx.timeout)
                return std::unexpected(Error{.code = ErrorCode::ProcessTimeout, .message = "Video extraction timed out"});
            try {
                if (ctx.on_progress && !ctx.on_progress(ctx.stage, fraction, "Extract video frames"))
                    return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Video callback cancelled"});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::CallbackFailure, .message = "Video progress callback failed"});
            }
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Video extraction cancelled"});
            if (ctx.timeout.count() > 0 && Clock::now() - start >= ctx.timeout)
                return std::unexpected(Error{.code = ErrorCode::ProcessTimeout, .message = "Video extraction timed out"});
            return {};
        }
        int round_even(double value) {
            const double lo = std::floor(value), fraction = value - lo;
            return static_cast<int>(lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2) != 0)));
        }
        std::expected<void, Error> save_png(const fs::path& path, const cv::Mat& pixels) {
            std::vector<unsigned char> encoded;
            if (!cv::imencode(".png", pixels, encoded))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot encode extracted video PNG"});
            if (fs::exists(path))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Refusing to overwrite video output"});
            std::ofstream stream(path, std::ios::binary | std::ios::noreplace);
            if (!stream)
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot open extracted video PNG"});
            stream.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
            stream.close();
            if (!stream)
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot write extracted video PNG"});
            return {};
        }
    } // namespace

    std::expected<std::vector<std::int64_t>, Error> video_sample_indices(std::int64_t total_frames, int requested) {
        if (total_frames <= 0 || requested <= 0)
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Video frame counts must be positive"});
        const auto count = std::min<std::int64_t>(requested, total_frames);
        std::vector<std::int64_t> indices;
        indices.reserve(static_cast<std::size_t>(count));
        if (count == 1) {
            indices.push_back(0);
            return indices;
        }
        // Keep the same double multiply then floor sequence as NumPy linspace.
        // Integer division is subtly different when the floating step rounds down.
        const double step = double(total_frames - 1) / double(count - 1);
        for (std::int64_t i = 0; i < count; ++i) {
            const auto index = i + 1 == count ? total_frames - 1 : static_cast<std::int64_t>(std::floor(double(i) * step));
            if (indices.empty() || indices.back() != index)
                indices.push_back(index);
        }
        return indices;
    }

    std::expected<VideoExtractionResult, Error> extract_video_frames(
        const fs::path& video, const fs::path& destination,
        const VideoExtractionOptions& options, const ExecutionContext& ctx) {
        const auto start = Clock::now();
        if (options.target_frame_count <= 0)
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Video target frame count must be positive"});
        if (auto status = check_progress(ctx, start, 0); !status)
            return std::unexpected(status.error());
        try {
            if (!fs::is_regular_file(video))
                return std::unexpected(invalid("Video input is not a regular local file"));
            if (destination.empty() || fs::exists(destination))
                return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = "Video output directory must be fresh"});
            if (!cv::videoio_registry::hasBackend(cv::CAP_FFMPEG))
                return std::unexpected(Error{.code = ErrorCode::RuntimeMissing, .message = "Native OpenCV FFmpeg video backend is unavailable"});
            if (!cv::haveImageWriter(".png"))
                return std::unexpected(Error{.code = ErrorCode::RuntimeMissing, .message = "Native PNG codec is unavailable"});
            cv::VideoCapture capture;
            const auto path = utf8(fs::absolute(video));
            std::vector<int> parameters;
            if (ctx.timeout.count() > 0) {
                const int timeout = static_cast<int>(std::min<std::int64_t>(ctx.timeout.count(), std::numeric_limits<int>::max()));
                parameters = {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeout, cv::CAP_PROP_READ_TIMEOUT_MSEC, timeout};
            }
            if (!capture.open(path, cv::CAP_FFMPEG, parameters)) {
                if (auto status = check_progress(ctx, start, 0); !status)
                    return std::unexpected(status.error());
                return std::unexpected(invalid("Cannot open video using the native FFmpeg backend"));
            }
            VideoExtractionResult result;
            const double reported_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
            if (std::isfinite(reported_frames) && reported_frames > 0 && reported_frames < double(std::numeric_limits<std::int64_t>::max()))
                result.total_frames = static_cast<std::int64_t>(reported_frames);
            if (result.total_frames <= 0) {
                result.counted_frames = true;
                while (true) {
                    if (auto status = check_progress(ctx, start, 0); !status)
                        return std::unexpected(status.error());
                    if (!capture.grab())
                        break;
                    ++result.total_frames;
                }
                // Some elementary streams report seek success without rewinding.
                // Reopening the same read-only local file guarantees a real reset.
                capture.release();
                if (!capture.open(path, cv::CAP_FFMPEG, parameters))
                    return std::unexpected(invalid("Cannot reopen video after counting frames"));
            }
            if (result.total_frames <= 0)
                return std::unexpected(invalid("Video has zero decodable frames"));
            auto indices = video_sample_indices(result.total_frames, options.target_frame_count);
            if (!indices)
                return std::unexpected(indices.error());
            result.sample_indices = std::move(*indices);
            if (auto status = check_progress(ctx, start, 0); !status)
                return std::unexpected(status.error());
            // The caller provides a fresh child of its owned stage. Never clear or reuse an existing directory.
            if (!fs::create_directories(destination))
                return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = "Cannot create fresh video output directory"});
            std::size_t target = 0;
            std::int64_t source_index = 0;
            while (target < result.sample_indices.size()) {
                if (auto status = check_progress(ctx, start, float(target) / result.sample_indices.size()); !status)
                    return std::unexpected(status.error());
                if (!capture.grab())
                    break;
                if (source_index++ != result.sample_indices[target])
                    continue;
                cv::Mat frame;
                if (!capture.retrieve(frame) || frame.empty())
                    break;
                if (frame.type() != CV_8UC3)
                    return std::unexpected(invalid("Video decoder did not return an 8-bit BGR image"));
                const int longest = std::max(frame.cols, frame.rows);
                if (options.longest_side > 0 && longest > options.longest_side) {
                    const double scale = double(options.longest_side) / longest;
                    const int width = std::max(1, round_even(frame.cols * scale));
                    const int height = std::max(1, round_even(frame.rows * scale));
                    cv::Mat resized;
                    cv::resize(frame, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
                    frame = std::move(resized);
                }
                const auto name = std::format("video_{:04}.png", result.frames.size());
                if (auto saved = save_png(destination / name, frame); !saved)
                    return std::unexpected(saved.error());
                result.frames.push_back({name, result.sample_indices[target], frame.cols, frame.rows});
                ++target;
            }
            if (result.frames.empty())
                return std::unexpected(invalid("Video extraction produced no images"));
            result.incomplete = result.frames.size() < result.sample_indices.size();
            if (auto status = check_progress(ctx, start, 1); !status)
                return std::unexpected(status.error());
            return result;
        } catch (const cv::Exception& exception) {
            return std::unexpected(invalid(std::string("Native video backend: ") + exception.what()));
        } catch (const fs::filesystem_error& exception) {
            return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = exception.what()});
        } catch (const std::exception& exception) {
            return std::unexpected(invalid(exception.what()));
        }
    }
} // namespace lfs::preprocess
