/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2024 Reflct
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Native adaptation of sharp-frames 0.3.1 CLI scoring/outlier selection.
// Original MIT notice and source hashes: docs/swaptexture_m5/sharp_frames_notice.txt.
// NumPy-compatible reduction notice: docs/swaptexture_m5/numpy_notice.txt.
#include "preprocessing/blur_filter.hpp"
#include "preprocessing/runtime.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

namespace lfs::preprocess {
    namespace {
        std::expected<void, Error> validate_options(const BlurFilterOptions& options) {
            if (options.window_size < 1 || options.sensitivity < 0 || options.sensitivity > 100)
                return std::unexpected(Error{.code = ErrorCode::InvalidRequest,
                                             .message = "Blur window must be positive and sensitivity in [0,100]"});
            return {};
        }
        std::expected<void, Error> progress(const ExecutionContext& context, float fraction) {
            if (context.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Blur filtering cancelled"});
            try {
                if (context.on_progress &&
                    !context.on_progress(Stage::ImageProcessing, fraction, "Selecting sharp images"))
                    return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Blur callback cancelled"});
            } catch (...) {
                return std::unexpected(Error{.code = ErrorCode::CallbackFailure,
                                             .message = "Blur progress callback failed"});
            }
            return {};
        }
        std::string extension(const std::filesystem::path& path) {
            auto result = path.extension().string();
            std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            return result;
        }
        bool wrapper_supports(const std::filesystem::path& path) {
            static const std::set<std::string> formats{".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"};
            return formats.contains(extension(path));
        }
        bool scorer_supports(const std::filesystem::path& path) {
            const auto ext = extension(path);
            return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
        }
        // NumPy's contiguous double reduction uses 128-element pairwise blocks.
        // Reproduce the summation order for boundary-sensitive variance scores.
        double pairwise_sum(std::span<const double> values) {
            if (values.size() < 8) {
                double result = -0.;
                for (const auto value : values)
                    result += value;
                return result;
            }
            if (values.size() <= 128) {
                double sums[8];
                std::copy_n(values.begin(), 8, sums);
                std::size_t index = 8;
                for (; index + 7 < values.size(); index += 8)
                    for (std::size_t lane = 0; lane < 8; ++lane)
                        sums[lane] += values[index + lane];
                double result = ((sums[0] + sums[1]) + (sums[2] + sums[3])) +
                                ((sums[4] + sums[5]) + (sums[6] + sums[7]));
                for (; index < values.size(); ++index)
                    result += values[index];
                return result;
            }
            const auto split = (values.size() / 2) & ~std::size_t(7);
            return pairwise_sum(values.first(split)) + pairwise_sum(values.subspan(split));
        }
        BlurScoreSelection select_scores(std::span<const double> scores, const BlurFilterOptions& options,
                                         std::size_t source_count) {
            BlurScoreSelection result;
            result.selected.assign(scores.size(), true);
            result.threshold_percent = (100 - options.sensitivity) / 4.;
            result.effective_window = std::max<std::size_t>(3, std::min<std::size_t>(options.window_size, source_count));
            if (result.effective_window % 2 == 0)
                ++result.effective_window;
            if (!options.enabled || source_count < 3 || scores.empty()) {
                result.skipped = true;
                return result;
            }
            const auto [minimum, maximum] = std::minmax_element(scores.begin(), scores.end());
            result.global_range = *maximum - *minimum;
            // The old CLI selector treats BOTH endpoints as keep-all.
            if (result.global_range == 0 || options.sensitivity == 0 || options.sensitivity == 100)
                return result;
            const auto half = result.effective_window / 2;
            for (std::size_t index = 0; index < scores.size(); ++index) {
                const auto begin = index > half ? index - half : 0;
                const auto end = index + std::min(half + 1, scores.size() - index);
                const auto count = end - begin - 1;
                if (count < 3)
                    continue;
                double sum = 0.;
                for (std::size_t neighbor = begin; neighbor < end; ++neighbor)
                    if (neighbor != index)
                        sum += scores[neighbor];
                const double average = sum / count;
                const double difference_percent = ((average - scores[index]) / result.global_range) * 100.;
                result.selected[index] = !(scores[index] < average && difference_percent > result.threshold_percent);
            }
            return result;
        }
    } // namespace

    std::expected<BlurScoreSelection, Error> select_blur_outliers(
        std::span<const double> scores, const BlurFilterOptions& options) {
        if (auto valid = validate_options(options); !valid)
            return std::unexpected(valid.error());
        if (std::any_of(scores.begin(), scores.end(), [](double score) { return !std::isfinite(score) || score < 0; }))
            return std::unexpected(Error{.code = ErrorCode::InvalidRequest,
                                         .message = "Blur scores must be finite and nonnegative"});
        return select_scores(scores, options, scores.size());
    }

    std::expected<double, Error> blur_sharpness_score(const std::filesystem::path& path) try {
        // Opening with fs::path supports Unicode on Windows; imdecode retains the
        // grayscale codec conversion used by cv2.imread, unlike RGB->cvtColor.
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Cannot read blur image: " + path_utf8(path)});
        const auto length = stream.tellg();
        if (length <= 0 || length > std::numeric_limits<int>::max())
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Invalid encoded blur image size"});
        std::vector<uint8_t> bytes(static_cast<std::size_t>(length));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream)
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Cannot read complete blur image"});
        const auto gray = cv::imdecode(bytes, cv::IMREAD_GRAYSCALE);
        if (gray.empty() || gray.cols < 2 || gray.rows < 2)
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Cannot decode or halve blur image: " + path_utf8(path)});
        cv::Mat half, laplacian;
        cv::resize(gray, half, {gray.cols / 2, gray.rows / 2}, 0, 0, cv::INTER_AREA);
        cv::Laplacian(half, laplacian, CV_64F);
        const auto count = laplacian.total();
        std::vector<double> values(laplacian.ptr<double>(), laplacian.ptr<double>() + count);
        const double mean = pairwise_sum(values) / count;
        for (auto& value : values) {
            const double deviation = value - mean;
            value = deviation * deviation;
        }
        return pairwise_sum(values) / count;
    } catch (const std::exception& error) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = error.what()});
    }

    std::expected<BlurFilterResult, Error> select_blur_images(
        const std::filesystem::path& image_directory, const BlurFilterOptions& options,
        const ExecutionContext& context, const std::optional<std::filesystem::path>& pose_directory) try {
        if (auto valid = validate_options(options); !valid)
            return std::unexpected(valid.error());
        if (auto check = progress(context, 0); !check)
            return std::unexpected(check.error());
        if (!std::filesystem::is_directory(image_directory))
            return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = "Blur input is not an image directory"});
        BlurFilterResult result;
        for (const auto& entry : std::filesystem::directory_iterator(image_directory))
            if (entry.is_regular_file() && wrapper_supports(entry.path())) {
                BlurImageRecord record{.source = entry.path(), .name = path_utf8(entry.path().filename())};
                if (pose_directory)
                    record.pose = *pose_directory / entry.path().filename().replace_extension(".pose");
                result.records.push_back(std::move(record));
            }
        // Python sharp-frames sorts string paths, not numeric stems. UTF-8 byte
        // order preserves Unicode code point order for valid filenames.
        std::sort(result.records.begin(), result.records.end(), [](const auto& a, const auto& b) {
            return a.name < b.name;
        });
        result.threshold_percent = (100 - options.sensitivity) / 4.;
        result.skipped = !options.enabled || result.records.size() < 3;
        if (!result.skipped) {
            // A wrongly built dependency must not masquerade as the legacy
            // all-corrupt-images fallback. Prepared product images are PNG.
            if (!cv::haveImageWriter(".png") || !cv::haveImageWriter(".jpg"))
                return std::unexpected(Error{.code = ErrorCode::RuntimeMissing,
                                             .message = "Native blur filtering requires OpenCV PNG and JPEG codec support"});
            std::vector<double> scores;
            std::vector<std::size_t> scored_indices;
            for (std::size_t index = 0; index < result.records.size(); ++index) {
                auto& record = result.records[index];
                if (scorer_supports(record.source)) {
                    auto score = blur_sharpness_score(record.source);
                    if (score) {
                        record.sharpness_score = *score;
                        scores.push_back(*score);
                        scored_indices.push_back(index);
                    } else if (context.on_log) {
                        try {
                            context.on_log(Stage::ImageProcessing, "Blur scoring skipped " + record.name + ": " + score.error().message);
                        } catch (...) {
                            return std::unexpected(Error{.code = ErrorCode::CallbackFailure, .message = "Blur log callback failed"});
                        }
                    }
                }
                if (auto check = progress(context, static_cast<float>(index + 1) / result.records.size()); !check)
                    return std::unexpected(check.error());
            }
            const auto selection = select_scores(scores, options, result.records.size());
            result.global_range = selection.global_range;
            result.effective_window = selection.effective_window;
            result.kept_all_fallback = std::none_of(selection.selected.begin(), selection.selected.end(), [](bool keep) { return keep; });
            if (!result.kept_all_fallback) {
                for (auto& record : result.records)
                    record.selected = false;
                for (std::size_t index = 0; index < scored_indices.size(); ++index)
                    result.records[scored_indices[index]].selected = selection.selected[index];
            }
        }
        for (const auto& record : result.records)
            (record.selected ? result.selected_names : result.removed_names).push_back(record.name);
        if (auto check = progress(context, 1); !check)
            return std::unexpected(check.error());
        return result;
    } catch (const std::filesystem::filesystem_error& error) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = error.what()});
    } catch (const std::exception& error) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = error.what()});
    }
} // namespace lfs::preprocess
