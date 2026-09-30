/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/reconstruction_progress.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::app {
    namespace {
        constexpr ReconstructionProgressWeights SCAN{0.18F, 0.38F, 0.98F, 1.0F};
        constexpr ReconstructionProgressWeights REGISTERED{0.60F, 0.63F, 0.98F, 1.0F};
        constexpr ReconstructionProgressWeights NO_GS{0.92F, 0.98F, 0.98F, 1.0F};
        float clamp(float value) { return std::isfinite(value) ? std::clamp(value, 0.0F, 1.0F) : 0.0F; }
    }
    ReconstructionProgress::ReconstructionProgress(ReconstructionInputMode mode, bool has_video, bool has_enhancement)
        : mode_(mode), has_video_(has_video), has_enhancement_(has_enhancement) {}
    ReconstructionProgressWeights ReconstructionProgress::weights(ReconstructionInputMode mode) {
        return mode == ReconstructionInputMode::Scan ? SCAN : mode == ReconstructionInputMode::Registered ? REGISTERED : NO_GS;
    }
    float ReconstructionProgress::update_range(float start, float end, float local_fraction) {
        const float candidate = start + (end - start) * clamp(local_fraction);
        float current = value_.load(std::memory_order_relaxed);
        while (candidate > current && !value_.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {}
        return value();
    }
    float ReconstructionProgress::update_stage(std::string_view name, float local_fraction) {
        const auto w = weights(mode_);
        if (name == "video_frames") return update_range(0.0F, 0.05F, local_fraction);
        if (mode_ == ReconstructionInputMode::Scan) {
            if (name == "scan") return update_range(has_video_ ? 0.05F : 0.0F, w.main_end, local_fraction);
            if (name == "scan_processed") return update_range(w.main_end, has_enhancement_ ? 0.27F : 0.36F, local_fraction);
            if (name == "gs_input_scan") return update_range(has_enhancement_ ? 0.27F : 0.36F, w.output_end, local_fraction);
        } else if (mode_ == ReconstructionInputMode::Registered) {
            if (name == "prepared") return update_range(has_video_ ? 0.05F : 0.0F, 0.15F, local_fraction);
            if (name == "scan_features") return update_range(0.15F, 0.30F, local_fraction);
            if (name == "seed") return update_range(0.30F, 0.45F, local_fraction);
            if (name == "registration") return update_range(0.45F, w.main_end, local_fraction);
            if (name == "assembled") return update_range(w.main_end, w.output_end, local_fraction);
        } else {
            return update_range(0.0F, w.main_end, local_fraction);
        }
        return value();
    }
    float ReconstructionProgress::preprocessing_done() {
        const auto w = weights(mode_);
        return update_range(w.output_end, w.output_end, 1.0F);
    }
    float ReconstructionProgress::training(float local_fraction) {
        const auto w = weights(mode_);
        return update_range(w.output_end, w.training_end, local_fraction);
    }
    float ReconstructionProgress::complete() { return update_range(weights(mode_).final_end, 1.0F, 1.0F); }
} // namespace lfs::app
