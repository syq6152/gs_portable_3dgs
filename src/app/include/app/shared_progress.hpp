/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <string>

namespace lfs::app {
    class SharedProgressMemory {
    public:
        SharedProgressMemory();
        SharedProgressMemory(const wchar_t* mapping_name, const wchar_t* mutex_name);
        ~SharedProgressMemory();
        SharedProgressMemory(const SharedProgressMemory&) = delete;
        SharedProgressMemory& operator=(const SharedProgressMemory&) = delete;
        void write(float fraction);

    private:
#ifdef _WIN32
        void* mapping_ = nullptr;
        void* mutex_ = nullptr;
        char* view_ = nullptr;
#else
        std::string value_;
#endif
    };
} // namespace lfs::app
