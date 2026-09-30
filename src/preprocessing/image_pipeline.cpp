/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/image_pipeline.hpp"
#include "preprocessing/runtime.hpp"
#include "preprocessing/workspace.hpp"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <set>

namespace lfs::preprocess {
    namespace {
        void check(bool value, const char* message) {
            if (!value)
                throw std::runtime_error(message);
        }
        Error error(const std::exception& e) { return {.code = ErrorCode::InvalidDataset, .message = e.what()}; }
        void shape(const CpuImage& image) {
            check(image.width > 0 && image.height > 0 && (image.channels == 1 || image.channels == 3) &&
                      uint64_t(image.width) * image.height * image.channels == image.pixels.size(),
                  "Invalid CPU image shape");
        }
        int round_even(double value) {
            double lo = std::floor(value), fraction = value - lo;
            return static_cast<int>(lo + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lo, 2) != 0)));
        }
        uint8_t truncate(float value) { return static_cast<uint8_t>(std::clamp(value, 0.F, 255.F)); }
        std::expected<void, Error> progress(const ExecutionContext& ctx, float fraction) {
            if (ctx.stop_token.stop_requested())
                return std::unexpected(Error{.code = ErrorCode::ProcessCancelled, .message = "Image stage cancelled"});
            try {
                if (ctx.on_progress && !ctx.on_progress(Stage::ImageProcessing, fraction, "Image processing"))
                    return std::unexpected(
                        Error{.code = ErrorCode::ProcessCancelled, .message = "Image callback cancelled"});
            } catch (...) {
                return std::unexpected(
                    Error{.code = ErrorCode::CallbackFailure, .message = "Image progress callback failed"});
            }
            return {};
        }
        CpuImage orient(const CpuImage& source, int orientation) {
            if (orientation < 2 || orientation > 8)
                return source;
            CpuImage result{orientation >= 5 ? source.height : source.width,
                            orientation >= 5 ? source.width : source.height,
                            source.channels,
                            {}};
            result.pixels.resize(source.pixels.size());
            for (int y = 0; y < source.height; ++y)
                for (int x = 0; x < source.width; ++x) {
                    int dx = x, dy = y;
                    switch (orientation) {
                    case 2: dx = source.width - 1 - x; break;
                    case 3:
                        dx = source.width - 1 - x;
                        dy = source.height - 1 - y;
                        break;
                    case 4: dy = source.height - 1 - y; break;
                    case 5:
                        dx = y;
                        dy = x;
                        break;
                    case 6:
                        dx = source.height - 1 - y;
                        dy = x;
                        break;
                    case 7:
                        dx = source.height - 1 - y;
                        dy = source.width - 1 - x;
                        break;
                    case 8:
                        dx = y;
                        dy = source.width - 1 - x;
                        break;
                    }
                    for (int c = 0; c < source.channels; ++c)
                        result.pixels[(size_t(dy) * result.width + dx) * source.channels + c] =
                            source.pixels[(size_t(y) * source.width + x) * source.channels + c];
                }
            return result;
        }
        struct Filter {
            int start;
            std::vector<int32_t> weights;
        };
        std::vector<Filter> filters(int input, int output) {
            std::vector<Filter> result;
            const double scale = double(input) / output, stretch = std::max(1., scale), support = 3 * stretch;
            for (int i = 0; i < output; ++i) {
                double center = (i + 0.5) * scale;
                int lo = std::max(0, static_cast<int>(center - support + 0.5)),
                    hi = std::min(input, static_cast<int>(center + support + 0.5));
                std::vector<double> values;
                double sum = 0;
                for (int j = lo; j < hi; ++j) {
                    double x = (j - center + 0.5) / stretch;
                    double w = x == 0 ? 1
                                      : (std::abs(x) >= 3
                                             ? 0
                                             : std::sin(std::numbers::pi * x) * std::sin(std::numbers::pi * x / 3) /
                                                   (std::numbers::pi * std::numbers::pi * x * x / 3));
                    values.push_back(w);
                    sum += w;
                }
                Filter f{lo, {}};
                for (double w : values) {
                    w = w / sum * (1 << 22);
                    f.weights.push_back(static_cast<int32_t>(w + (w < 0 ? -0.5 : 0.5)));
                }
                result.push_back(std::move(f));
            }
            return result;
        }
        cv::Mat bgr(const CpuImage& image) {
            shape(image);
            cv::Mat source(image.height, image.width, image.channels == 1 ? CV_8UC1 : CV_8UC3,
                           const_cast<uint8_t*>(image.pixels.data())),
                result;
            cv::cvtColor(source, result, image.channels == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_RGB2BGR);
            return result;
        }
        CpuImage rgb(const cv::Mat& input) {
            cv::Mat out;
            cv::cvtColor(input, out, cv::COLOR_BGR2RGB);
            CpuImage result{out.cols, out.rows, 3, {}};
            result.pixels.assign(out.data, out.data + out.total() * 3);
            return result;
        }
        float percentile(std::vector<float> values, float percent) {
            check(!values.empty(), "Empty statistics");
            std::sort(values.begin(), values.end());
            const double index = std::clamp(double(percent), 0., 100.) / 100 * (values.size() - 1);
            auto lo = static_cast<size_t>(index), hi = static_cast<size_t>(std::ceil(index));
            return static_cast<float>(values[lo] + (values[hi] - values[lo]) * (index - lo));
        }
        cv::Mat sample(const std::filesystem::path& file, int max_size) {
            auto read = read_image(file, false);
            if (!read)
                throw std::runtime_error(read.error().message);
            auto image = bgr(*read);
            if (std::max(image.cols, image.rows) > max_size) {
                const auto size = resized_dimensions(image.cols, image.rows, max_size);
                cv::resize(image, image, {size[0], size[1]}, 0, 0, cv::INTER_AREA);
            }
            return image;
        }
        void gains(cv::Mat& image, const std::array<float, 3>& gain) {
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    for (int c = 0; c < 3; ++c)
                        image.at<cv::Vec3b>(y, x)[c] = truncate(float(image.at<cv::Vec3b>(y, x)[c]) * gain[c]);
        }
        bool supported(const std::filesystem::path& path) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return std::set<std::string>{".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}.contains(ext);
        }
        std::filesystem::path contained_name(const std::string& name) {
            auto p = std::filesystem::u8path(name);
            check(!p.empty() && !p.is_absolute() && !p.has_root_name() && name.find(':') == std::string::npos &&
                      name.find('\0') == std::string::npos,
                  "Invalid image relative name");
            for (const auto& part : p)
                check(part != "..", "Image path escapes root");
            return p;
        }
        void options_valid(const EnhancementOptions& o) {
            for (float v : {o.wb_min, o.wb_max, o.luma_target, o.relative_luma_keep, o.tone_blend, o.contrast_gain,
                            o.saturation_gain, o.sharpen_amount, o.low_percentile, o.high_percentile})
                check(std::isfinite(v), "Nonfinite enhancement option");
            check(o.stats_max_size > 0 && o.wb_min > 0 && o.wb_max >= o.wb_min, "Invalid enhancement bounds");
        }
    } // namespace
    std::expected<void, Error> copy_image(const std::filesystem::path& source, const std::filesystem::path& destination,
                                          const ExecutionContext& ctx) try {
        if (auto isolated = check_output_isolation(destination, {source}); !isolated)
            return isolated;
        if (auto p = progress(ctx, 0); !p)
            return p;
        check(std::filesystem::is_regular_file(source) && supported(source), "Unsupported/missing source image");
        std::filesystem::copy_file(source, destination, std::filesystem::copy_options::none);
        return progress(ctx, 1);
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
    std::expected<CpuImage, Error> read_image(const std::filesystem::path& path, bool apply_exif) try {
        OIIO::ImageSpec config;
        config.attribute("oiio:UnassociatedAlpha", 1);
        auto input = OIIO::ImageInput::open(path_utf8(path), &config);
        if (!input)
            throw std::runtime_error("Cannot decode image: " + OIIO::geterror());
        const auto& spec = input->spec();
        check(spec.width > 0 && spec.height > 0 && spec.nchannels >= 1 && spec.nchannels <= 4 &&
                  uint64_t(spec.width) * spec.height * spec.nchannels <= 1024ull * 1024 * 1024,
              "Unsupported/oversized image");
        std::vector<uint8_t> data(size_t(spec.width) * spec.height * spec.nchannels);
        check(input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::UINT8, data.data()), "Image decode failed");
        CpuImage image{spec.width, spec.height, spec.nchannels == 1 ? 1 : 3, {}};
        image.pixels.resize(size_t(image.width) * image.height * image.channels);
        for (size_t i = 0; i < size_t(image.width) * image.height; ++i)
            for (int c = 0; c < image.channels; ++c)
                image.pixels[i * image.channels + c] = data[i * spec.nchannels + (spec.nchannels == 2 ? 0 : c)];
        const int orientation = spec.get_int_attribute("Orientation", 1);
        input->close();
        return apply_exif ? orient(image, orientation) : image;
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<std::array<int, 2>, Error> image_dimensions(const std::filesystem::path& path, bool apply_exif) try {
        auto input = OIIO::ImageInput::open(path_utf8(path));
        if (!input)
            throw std::runtime_error("Cannot read image header: " + OIIO::geterror());
        const auto& s = input->spec();
        check(s.width > 0 && s.height > 0, "Invalid image dimensions");
        std::array<int, 2> result{s.width, s.height};
        if (apply_exif && s.get_int_attribute("Orientation", 1) >= 5)
            std::swap(result[0], result[1]);
        input->close();
        return result;
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<void, Error> write_png(const std::filesystem::path& path, const CpuImage& image) try {
        shape(image);
        check(!std::filesystem::exists(path), "PNG output must not overwrite an existing file");
        auto output = OIIO::ImageOutput::create("png");
        check(bool(output), "PNG codec missing");
        OIIO::ImageSpec spec(image.width, image.height, image.channels, OIIO::TypeDesc::UINT8);
        spec.attribute("Orientation", 1);
        check(output->open(path_utf8(path), spec) && output->write_image(OIIO::TypeDesc::UINT8, image.pixels.data()) &&
                  output->close(),
              "PNG write failed");
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
    std::array<int, 2> resized_dimensions(int width, int height, int longest_side) {
        check(width > 0 && height > 0, "Invalid resize dimensions");
        if (longest_side <= 0 || std::max(width, height) <= longest_side)
            return {width, height};
        const double scale = double(longest_side) / std::max(width, height);
        return {std::max(1, round_even(width * scale)), std::max(1, round_even(height * scale))};
    }
    std::expected<CpuImage, Error> resize_lanczos(const CpuImage& image, int width, int height) try {
        shape(image);
        check(width > 0 && height > 0 && uint64_t(width) * height * image.channels <= 1024ull * 1024 * 1024,
              "Invalid resize size");
        if (width == image.width && height == image.height)
            return image;
        CpuImage temp{width, image.height, image.channels, {}}, out{width, height, image.channels, {}};
        temp.pixels.resize(size_t(width) * image.height * image.channels);
        out.pixels.resize(size_t(width) * height * image.channels);
        auto horizontal = filters(image.width, width), vertical = filters(image.height, height);
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < image.channels; ++c) {
                    int64_t sum = 1 << 21;
                    const auto& f = horizontal[x];
                    for (size_t j = 0; j < f.weights.size(); ++j)
                        sum += int64_t(image.pixels[(size_t(y) * image.width + f.start + j) * image.channels + c]) *
                               f.weights[j];
                    temp.pixels[(size_t(y) * width + x) * image.channels + c] =
                        static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
                }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < image.channels; ++c) {
                    int64_t sum = 1 << 21;
                    const auto& f = vertical[y];
                    for (size_t j = 0; j < f.weights.size(); ++j)
                        sum += int64_t(temp.pixels[((f.start + j) * width + x) * image.channels + c]) * f.weights[j];
                    out.pixels[(size_t(y) * width + x) * image.channels + c] =
                        static_cast<uint8_t>(std::clamp<int64_t>(sum >> 22, 0, 255));
                }
        return out;
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    CpuImage rotate_ccw(const CpuImage& image) {
        shape(image);
        return orient(image, 8);
    }
    std::vector<std::size_t> uniform_sample(std::size_t total, std::size_t requested) {
        std::vector<std::size_t> result;
        if (!total || !requested)
            return result;
        const auto n = std::min(total, requested);
        if (n == 1)
            return {0};
        check(total <= uint64_t(INT32_MAX), "Sample size too large");
        const double step = double(total - 1) / (n - 1);
        for (size_t i = 0; i < n; ++i)
            result.push_back(i == n - 1 ? total - 1 : static_cast<size_t>(double(i) * step));
        return result;
    }
    std::string incremental_image_name(std::size_t index) { return std::format("inc_{:04}.png", index); }
    std::expected<std::vector<ImageImportRecord>, Error> import_images(const std::filesystem::path& source,
                                                                       const std::filesystem::path& destination,
                                                                       const ImageImportOptions& options,
                                                                       const ExecutionContext& ctx) try {
        if (auto isolated = check_output_isolation(destination, {source}); !isolated)
            return std::unexpected(isolated.error());
        if (auto p = progress(ctx, 0); !p)
            return std::unexpected(p.error());
        check(std::filesystem::is_directory(source) && !std::filesystem::exists(destination),
              "Image import needs input directory and fresh output");
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(source))
            if (entry.is_regular_file() && supported(entry.path()))
                files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        check(!files.empty(), "No input images");
        std::vector<std::array<int, 2>> sizes;
        std::map<std::array<int, 2>, size_t> original_counts, resized_counts;
        std::set<std::array<int, 2>> unordered;
        for (const auto& file : files) {
            auto size = image_dimensions(file, true);
            if (!size)
                return std::unexpected(size.error());
            sizes.push_back(*size);
            ++original_counts[*size];
            auto resized = resized_dimensions((*size)[0], (*size)[1], options.longest_side);
            ++resized_counts[resized];
            if (resized[0] > resized[1])
                std::swap(resized[0], resized[1]);
            unordered.insert(resized);
        }
        std::optional<std::array<int, 2>> selected, target;
        size_t count = 0;
        for (const auto& size : sizes)
            if (original_counts[size] > count) {
                count = original_counts[size];
                selected = size;
            }
        if (options.normalize_swapped_orientation && resized_counts.size() > 1 && unordered.size() == 1) {
            count = 0;
            for (const auto& size : sizes) {
                auto r = resized_dimensions(size[0], size[1], options.longest_side);
                if (resized_counts[r] > count) {
                    count = resized_counts[r];
                    target = r;
                }
            }
        }
        std::filesystem::create_directories(destination);
        std::vector<ImageImportRecord> records;
        for (size_t i = 0; i < files.size(); ++i) {
            if (options.largest_size_group_only && sizes[i] != *selected)
                continue;
            if (auto p = progress(ctx, float(i) / files.size()); !p)
                return std::unexpected(p.error());
            auto decoded = read_image(files[i]);
            if (!decoded)
                return std::unexpected(decoded.error());
            auto size = resized_dimensions(decoded->width, decoded->height, options.longest_side);
            auto resized = resize_lanczos(*decoded, size[0], size[1]);
            if (!resized)
                return std::unexpected(resized.error());
            bool rotated = false;
            if (!options.largest_size_group_only && target && size != *target && size[0] == (*target)[1] &&
                size[1] == (*target)[0]) {
                *resized = rotate_ccw(*resized);
                rotated = true;
            }
            const auto name = incremental_image_name(records.size());
            if (auto written = write_png(destination / name, *resized); !written)
                return std::unexpected(written.error());
            records.push_back({files[i], name, resized->width, resized->height, rotated});
        }
        if (auto p = progress(ctx, 1); !p)
            return std::unexpected(p.error());
        return records;
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<EnhancementStats, Error> enhancement_stats(const std::filesystem::path& root,
                                                             const std::vector<std::string>& names,
                                                             const EnhancementOptions& o,
                                                             const ExecutionContext& ctx) try {
        options_valid(o);
        EnhancementStats stats;
        if (names.empty())
            return stats;
        std::array<std::vector<float>, 3> medians;
        for (size_t i = 0; i < names.size(); ++i) {
            if (auto p = progress(ctx, float(i) / (2 * names.size())); !p)
                return std::unexpected(p.error());
            auto image = sample(root / contained_name(names[i]), o.stats_max_size);
            std::array<std::vector<float>, 3> values;
            for (int y = 0; y < image.rows; ++y)
                for (int x = 0; x < image.cols; ++x)
                    for (int c = 0; c < 3; ++c)
                        values[c].push_back(image.at<cv::Vec3b>(y, x)[c]);
            for (int c = 0; c < 3; ++c)
                medians[c].push_back(percentile(std::move(values[c]), 50));
        }
        for (int c = 0; c < 3; ++c)
            stats.channel_reference[c] = percentile(std::move(medians[c]), 50);
        float gray = 0;
        for (float v : stats.channel_reference)
            gray += std::max(1.F, v);
        gray /= 3;
        if (o.white_balance)
            for (int c = 0; c < 3; ++c)
                stats.wb_gains[c] = std::clamp(gray / std::max(1.F, stats.channel_reference[c]), o.wb_min, o.wb_max);
        std::array<std::vector<float>, 3> sequence;
        for (size_t i = 0; i < names.size(); ++i) {
            if (auto p = progress(ctx, 0.5F + float(i) / (2 * names.size())); !p)
                return std::unexpected(p.error());
            auto image = sample(root / contained_name(names[i]), o.stats_max_size);
            gains(image, stats.wb_gains);
            cv::Mat lab;
            cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
            std::vector<float> values;
            for (int y = 0; y < lab.rows; ++y)
                for (int x = 0; x < lab.cols; ++x)
                    values.push_back(lab.at<cv::Vec3b>(y, x)[0]);
            std::array<float, 3> row{percentile(values, std::clamp(o.low_percentile, 0.F, 49.F)),
                                     percentile(values, 50),
                                     percentile(values, std::clamp(o.high_percentile, 51.F, 100.F))};
            stats.image_luma.emplace(names[i], row);
            for (int c = 0; c < 3; ++c)
                sequence[c].push_back(row[c]);
        }
        for (int c = 0; c < 3; ++c)
            stats.sequence_luma[c] = percentile(std::move(sequence[c]), 50);
        if (auto p = progress(ctx, 1); !p)
            return std::unexpected(p.error());
        return stats;
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<CpuImage, Error> denoise_image(const CpuImage& image) try {
        auto color = bgr(image);
        cv::Mat out;
        cv::fastNlMeansDenoisingColored(color, out, 5, 3, 7, 21);
        return rgb(out);
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<CpuImage, Error> enhance_image(const CpuImage& image, const std::string& name,
                                                 const EnhancementStats& stats, const EnhancementOptions& o) try {
        options_valid(o);
        auto work = bgr(image);
        if (o.white_balance)
            gains(work, stats.wb_gains);
        if (o.luma_tone) {
            auto frame = stats.image_luma.contains(name) ? stats.image_luma.at(name) : stats.sequence_luma;
            const double mid = std::clamp(double(o.luma_target) + std::clamp(double(o.relative_luma_keep), 0., 1.) *
                                                                      (frame[1] - stats.sequence_luma[1]),
                                          24., 235.);
            const double contrast = o.contrast ? std::max(0., double(o.contrast_gain)) : 1;
            const double low = std::clamp(mid - std::max(1., double(frame[1] - frame[0])) * contrast, 1., mid - 1),
                         high = std::clamp(mid + std::max(1., double(frame[2] - frame[1])) * contrast, mid + 1, 254.);
            float slo = std::clamp(frame[0], 1.F, 253.F), smid = std::clamp(frame[1], slo + 1, 254.F),
                  shi = std::clamp(frame[2], smid + 1, 255.F);
            std::array<float, 5> src{0, slo, smid, shi, 255}, dst{0, float(low), float(mid), float(high), 255};
            std::array<float, 256> lut{};
            for (int i = 0; i < 256; ++i) {
                int j = 0;
                while (j < 3 && i >= src[j + 1])
                    ++j;
                lut[i] = src[j + 1] == src[j]
                             ? dst[j + 1]
                             : float(dst[j] + (double(i) - src[j]) * (dst[j + 1] - dst[j]) / (src[j + 1] - src[j]));
            }
            lut[255] = 255;
            cv::Mat lab;
            cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
            const float blend = std::clamp(o.tone_blend, 0.F, 1.F);
            for (int y = 0; y < lab.rows; ++y)
                for (int x = 0; x < lab.cols; ++x) {
                    auto& l = lab.at<cv::Vec3b>(y, x)[0];
                    l = truncate((1.F - blend) * float(l) + blend * lut[l]);
                }
            cv::cvtColor(lab, work, cv::COLOR_Lab2BGR);
        }
        const float sat = std::max(0.F, o.saturation_gain);
        if (o.saturation && std::abs(sat - 1) > 1e-6) {
            cv::Mat hsv;
            cv::cvtColor(work, hsv, cv::COLOR_BGR2HSV);
            for (int y = 0; y < hsv.rows; ++y)
                for (int x = 0; x < hsv.cols; ++x) {
                    auto& s = hsv.at<cv::Vec3b>(y, x)[1];
                    s = truncate(s * sat);
                }
            cv::cvtColor(hsv, work, cv::COLOR_HSV2BGR);
        }
        if (o.sharpen && o.sharpen_amount > 1e-6) {
            cv::Mat blurred;
            cv::GaussianBlur(work, blurred, {0, 0}, 1.);
            cv::addWeighted(work, 1. + o.sharpen_amount, blurred, -double(o.sharpen_amount), 0, work);
        }
        return rgb(work);
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
    std::expected<void, Error> enhance_images(const std::filesystem::path& input, const std::filesystem::path& output,
                                              const std::vector<std::string>& names, bool denoise,
                                              const EnhancementOptions& o, const ExecutionContext& ctx) try {
        if (auto isolated = check_output_isolation(output, {input}); !isolated)
            return std::unexpected(isolated.error());
        options_valid(o);
        if (auto p = progress(ctx, 0); !p)
            return p;
        check(!names.empty() && !std::filesystem::exists(output), "Enhancement needs images and fresh output");
        std::set<std::string> unique;
        for (const auto& name : names) {
            auto file = contained_name(name);
            check(*file.begin() != ".denoised", "Reserved enhancement path");
            check(unique.insert(name).second, "Duplicate enhancement name");
        }
        std::filesystem::create_directories(output);
        const auto temp = output / ".denoised";
        if (denoise)
            std::filesystem::create_directory(temp);
        for (size_t i = 0; denoise && i < names.size(); ++i) {
            if (auto p = progress(ctx, 0.3F * float(i) / names.size()); !p)
                return p;
            auto file = contained_name(names[i]);
            auto image = read_image(input / file, false);
            if (!image)
                return std::unexpected(image.error());
            auto filtered = denoise_image(*image);
            if (!filtered)
                return std::unexpected(filtered.error());
            std::filesystem::create_directories((temp / file).parent_path());
            if (auto written = write_png(temp / file, *filtered); !written)
                return written;
        }
        auto stats_ctx = ctx;
        stats_ctx.on_progress = [&](Stage, float fraction, std::string) {
            auto p = progress(ctx, 0.3F + 0.3F * fraction);
            if (!p) {
                if (p.error().code == ErrorCode::CallbackFailure)
                    throw std::runtime_error(p.error().message);
                return false;
            }
            return true;
        };
        auto stats = enhancement_stats(denoise ? temp : input, names, o, stats_ctx);
        if (!stats)
            return std::unexpected(stats.error());
        for (size_t i = 0; i < names.size(); ++i) {
            if (auto p = progress(ctx, 0.6F + 0.4F * float(i) / names.size()); !p)
                return p;
            auto file = contained_name(names[i]);
            auto image = read_image((denoise ? temp : input) / file, false);
            if (!image)
                return std::unexpected(image.error());
            auto enhanced = enhance_image(*image, names[i], *stats, o);
            if (!enhanced)
                return std::unexpected(enhanced.error());
            std::filesystem::create_directories((output / file).parent_path());
            if (auto written = write_png(output / file, *enhanced); !written)
                return written;
        }
        // Denoised images are an optional stage artifact in the frozen M0 contract.
        // Keep them for audit; Workspace owns cleanup of an unsuccessful stage.
        return progress(ctx, 1);
    } catch (const std::exception& e) { return std::unexpected(error(e)); }
} // namespace lfs::preprocess
