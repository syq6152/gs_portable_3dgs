/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
#include <chrono>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::app {
    enum class ReconstructionInputMode;

    [[nodiscard]] std::vector<std::string> reconstruction_stage_order(
        ReconstructionInputMode mode, bool has_video, bool has_image_processing);

    // SwapTexture's runtime flag selects the output style in every build.
    [[nodiscard]] constexpr bool detailed_reconstruction_output(const bool configured) noexcept {
        return configured;
    }

    // Human-readable status uses the same sink as the unified reconstruction
    // console tee. An optional secondary sink is available for other callers.
    class ReconstructionOutput {
    public:
        ReconstructionOutput(bool detailed, std::ostream& terminal,
                             std::vector<std::string> stage_order, bool includes_training);
        void set_training_log(std::ostream* stream);
        void stage_progress(std::string_view name, float local_fraction, float overall_fraction);
        void training_started(std::size_t total_iterations, float overall_fraction);
        void training_progress(int iteration, std::size_t total_iterations, float loss, int gaussians,
                               float overall_fraction);
        void training_finished(std::string_view outcome, float overall_fraction);
        void completed(float overall_fraction);

    private:
        void emit(const std::string& message, bool training = false);
        [[nodiscard]] std::size_t stage_step(std::string_view name) const;
        [[nodiscard]] std::string concise_progress(std::size_t step, float overall_fraction) const;
        bool detailed_;
        std::ostream& terminal_;
        std::ostream* training_log_ = nullptr;
        std::unordered_map<std::string, std::size_t> stage_steps_;
        std::size_t total_steps_ = 1;
        std::size_t training_step_ = 0;
        std::size_t completion_step_ = 1;
        std::size_t current_step_ = 0;
        std::unordered_map<std::string, std::chrono::steady_clock::time_point> stage_started_;
        std::unordered_map<std::string, int> stage_bucket_;
        std::unordered_set<std::string> completed_stages_;
        int training_bucket_ = -1;
        int last_training_box_iteration_ = 0;
        std::chrono::steady_clock::time_point training_started_at_ = std::chrono::steady_clock::now();
    };

} // namespace lfs::app
