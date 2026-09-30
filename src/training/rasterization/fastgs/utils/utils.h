/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fast_lfs {

    class CudaRuntimeError final : public std::runtime_error {
    public:
        CudaRuntimeError(std::string_view stage, cudaError_t error)
            : std::runtime_error(make_message(stage, error)),
              stage_(stage),
              error_(error) {}

        [[nodiscard]] const std::string& stage() const noexcept { return stage_; }
        [[nodiscard]] cudaError_t error() const noexcept { return error_; }

    private:
        static std::string make_message(std::string_view stage, cudaError_t error) {
            const char* const name = cudaGetErrorName(error);
            const char* const description = cudaGetErrorString(error);
            return "CUDA failure at stage=" + std::string(stage) +
                   ": " + (name ? std::string(name) : std::string("unknown")) +
                   " (" + (description ? std::string(description) : std::string("unknown")) + ")";
        }

        std::string stage_;
        cudaError_t error_ = cudaSuccess;
    };

    inline void check_cuda(cudaError_t error, std::string_view stage) {
        if (error != cudaSuccess) {
            throw CudaRuntimeError(stage, error);
        }
    }

    // Check without consuming an error produced by work that predates the
    // current FastGS call.  This prevents a stale OOM/launch failure from being
    // misreported as a CUB sizing or arena-allocation problem.
    inline void check_no_pending_cuda_error(std::string_view stage) {
        const cudaError_t error = cudaPeekAtLastError();
        if (error != cudaSuccess) {
            throw CudaRuntimeError(stage, error);
        }
    }

    inline void check_cuda_kernel(std::string_view stage, bool synchronize) {
        // cudaGetLastError is intentional here: it captures and consumes the
        // error from the launch that immediately precedes this check.
        check_cuda(cudaGetLastError(), stage);
        if (synchronize) {
            check_cuda(cudaDeviceSynchronize(), stage);
        }
    }

} // namespace fast_lfs

// Launch errors must also be checked in Release. Debug builds additionally
// synchronize so asynchronous execution failures are attributed to the exact
// kernel that caused them.
#define CHECK_CUDA(debug, name)                      \
    do {                                             \
        ::fast_lfs::check_cuda_kernel(name, debug); \
    } while (false);

template <typename T>
inline __host__ __device__ T div_round_up(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}
