/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

namespace lfs::app {
    // Own the single log writer for both reconstruction and in-process training.
    class ReconstructionLog {
        class TeeBuffer : public std::streambuf {
        public:
            TeeBuffer(std::streambuf* console, std::streambuf* file, std::mutex& mutex)
                : console_(console), file_(file), mutex_(mutex) {}

        protected:
            std::streamsize xsputn(const char* text, std::streamsize size) override {
                std::lock_guard lock(mutex_);
                const auto first = console_->sputn(text, size);
                const auto second = file_->sputn(text, size);
                return (std::min)(first, second);
            }
            int_type overflow(int_type value) override {
                if (traits_type::eq_int_type(value, traits_type::eof()))
                    return traits_type::not_eof(value);
                const char ch = traits_type::to_char_type(value);
                return xsputn(&ch, 1) == 1 ? value : traits_type::eof();
            }
            int sync() override {
                std::lock_guard lock(mutex_);
                const int first = console_->pubsync();
                const int second = file_->pubsync();
                return first == 0 && second == 0 ? 0 : -1;
            }

        private:
            std::streambuf* console_;
            std::streambuf* file_;
            std::mutex& mutex_;
        };

        static std::filesystem::path open_log(std::ofstream& file, const std::filesystem::path& directory,
                                              std::chrono::system_clock::time_point now) {
            const auto time = std::chrono::system_clock::to_time_t(now);
            std::tm local{};
#ifdef _WIN32
            const bool converted = localtime_s(&local, &time) == 0;
#else
            const bool converted = localtime_r(&time, &local) != nullptr;
#endif
            char stamp[32]{};
            if (!converted || std::strftime(stamp, sizeof(stamp), "%Y%m%d%H%M", &local) == 0)
                throw std::runtime_error("Cannot format reconstruction log date");
            for (int attempt = 0; attempt < 10000; ++attempt) {
                const auto suffix = attempt == 0 ? std::string{} : "_" + std::to_string(attempt + 1);
                const auto path = directory / ("swap_texture_" + std::string(stamp) + suffix + ".log");
                file.open(path, std::ios::binary | std::ios::noreplace);
                if (file)
                    return path;
                // Exclusive creation handles concurrent same-minute runs too.
                if (!std::filesystem::exists(path))
                    throw std::filesystem::filesystem_error("Cannot create reconstruction log", path,
                                                            std::make_error_code(std::errc::io_error));
                file.clear();
            }
            throw std::filesystem::filesystem_error("Too many reconstruction logs for this minute", directory,
                                                    std::make_error_code(std::errc::file_exists));
        }

    public:
        explicit ReconstructionLog(const std::filesystem::path& directory,
                                   std::chrono::system_clock::time_point now = std::chrono::system_clock::now())
            : path_(open_log(file_, directory, now)),
              stdout_(std::cout.rdbuf(), file_.rdbuf(), mutex_),
              stderr_(std::cerr.rdbuf(), file_.rdbuf(), mutex_) {
            original_stdout_ = std::cout.rdbuf(&stdout_);
            original_stderr_ = std::cerr.rdbuf(&stderr_);
        }
        ~ReconstructionLog() {
            const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started_at_;
            std::cout << std::format("总耗时：{:.3f}s\n", elapsed.count());
            std::cout.flush();
            std::cerr.flush();
            std::cout.rdbuf(original_stdout_);
            std::cerr.rdbuf(original_stderr_);
        }

        [[nodiscard]] const std::filesystem::path& path() const { return path_; }

    private:
        std::chrono::steady_clock::time_point started_at_ = std::chrono::steady_clock::now();
        std::ofstream file_;
        std::filesystem::path path_;
        std::mutex mutex_;
        TeeBuffer stdout_, stderr_;
        std::streambuf* original_stdout_;
        std::streambuf* original_stderr_;
    };
} // namespace lfs::app
