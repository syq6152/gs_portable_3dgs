/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace fast_lfs {

    inline void check_cuda_pointer(const void* ptr, const char* name) {
        if (!ptr) {
            throw std::runtime_error("Null pointer for " + std::string(name));
        }

        cudaPointerAttributes attrs{};
        check_cuda(cudaPointerGetAttributes(&attrs, ptr),
                   "pointer_validation." + std::string(name));
        if (attrs.type != cudaMemoryTypeDevice) {
            throw std::runtime_error(std::string(name) + " is not a device pointer");
        }
    }

} // namespace fast_lfs

#define CHECK_CUDA_PTR(ptr, name)                  \
    do {                                           \
        ::fast_lfs::check_cuda_pointer(ptr, name); \
    } while (false);

#define CHECK_CUDA_PTR_OPTIONAL(ptr, name) \
    if (ptr) {                             \
        CHECK_CUDA_PTR(ptr, name)          \
    }

// Simple validation for CUDA pointers without throwing
inline bool is_valid_cuda_ptr(const void* ptr) {
    if (!ptr)
        return false;

    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        return false;
    }

    return attrs.type == cudaMemoryTypeDevice;
}

// Validate that a pointer is accessible from device (more lenient check)
inline bool is_device_accessible(const void* ptr) {
    if (!ptr)
        return false;

    // Try a simple test read
    float test_value;
    cudaError_t err = cudaMemcpy(&test_value, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}
