/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/shared_progress.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
namespace lfs::app {
    SharedProgressMemory::SharedProgressMemory()
#ifdef _WIN32
        : SharedProgressMemory(L"/swap_texture_shared_memory", L"swap_texture_mutex")
#else
        : SharedProgressMemory(nullptr, nullptr)
#endif
    {}
    SharedProgressMemory::SharedProgressMemory(const wchar_t* mapping_name, const wchar_t* mutex_name) {
#ifdef _WIN32
        mapping_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 1024, mapping_name);
        if (mapping_)
            view_ = static_cast<char*>(MapViewOfFile(static_cast<HANDLE>(mapping_), FILE_MAP_ALL_ACCESS, 0, 0, 1024));
        mutex_ = CreateMutexW(nullptr, FALSE, mutex_name);
        if (view_) std::fill(view_, view_ + 1024, '\0');
#else
        value_.assign(1024, '\0');
#endif
    }
    SharedProgressMemory::~SharedProgressMemory() {
#ifdef _WIN32
        if (view_) UnmapViewOfFile(view_);
        if (mutex_) CloseHandle(static_cast<HANDLE>(mutex_));
        if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
#endif
    }
    void SharedProgressMemory::write(float fraction) {
        char payload[64]{};
        std::snprintf(payload, sizeof(payload), "%.9g", std::max(0.0F, std::min(1.0F, fraction)));
#ifdef _WIN32
        if (!view_) return;
        if (mutex_) WaitForSingleObject(static_cast<HANDLE>(mutex_), INFINITE);
        std::fill(view_, view_ + 1024, '\0');
        std::memcpy(view_, payload, std::strlen(payload));
        if (mutex_) ReleaseMutex(static_cast<HANDLE>(mutex_));
#else
        value_ = payload;
#endif
    }
} // namespace lfs::app
