/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <atomic>
#include <string_view>
#include "preprocessing/types.hpp"

namespace lfs::app {
    enum class ReconstructionInputMode { Scan, Registered, NoGs };

    struct ReconstructionProgressWeights {
        float main_end;
        float output_end;
        float training_end;
        float final_end;
    };

    class ReconstructionProgress {
    public:
        explicit ReconstructionProgress(ReconstructionInputMode mode, bool has_video = false, bool has_enhancement = false);
        static ReconstructionProgressWeights weights(ReconstructionInputMode mode);
        float update_stage(std::string_view name, float local_fraction);
        float preprocessing_done();
        float training(float local_fraction);
        float complete();
        float value() const { return value_.load(std::memory_order_relaxed); }

    private:
        float update_range(float start, float end, float local_fraction);
        ReconstructionInputMode mode_;
        bool has_video_;
        bool has_enhancement_;
        std::atomic<float> value_{0.0F};
    };
} // namespace lfs::app
