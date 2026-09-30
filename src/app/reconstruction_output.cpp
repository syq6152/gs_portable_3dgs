/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/reconstruction_output.hpp"
#include "app/reconstruction_progress.hpp"
#include <algorithm>
#include <cmath>
#include <format>

namespace lfs::app {
#ifndef DEBUG_BUILD
    namespace {
        std::string training_duration(std::chrono::duration<double> elapsed) {
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (seconds >= 3600)
                return std::format("{:02}h:{:02}m:{:02}s", seconds / 3600, seconds / 60 % 60, seconds % 60);
            return std::format("{:02}m:{:02}s", seconds / 60, seconds % 60);
        }
    } // namespace
#endif

    std::vector<std::string> reconstruction_stage_order(
        ReconstructionInputMode mode, bool has_video, bool has_image_processing) {
        if (mode == ReconstructionInputMode::Scan) {
            std::vector<std::string> stages{"scan"};
            if (has_image_processing)
                stages.emplace_back("scan_processed");
            stages.emplace_back("gs_input_scan");
            return stages;
        }
        if (mode == ReconstructionInputMode::Registered) {
            std::vector<std::string> stages;
            if (has_video)
                stages.emplace_back("video_frames");
            stages.insert(stages.end(), {"prepared", "scan_features", "seed", "registration", "assembled"});
            return stages;
        }
        return {"prepared", "scan_features", "triangulated", "mesh_export"};
    }

    ReconstructionOutput::ReconstructionOutput(bool detailed, std::ostream& terminal,
                                               std::vector<std::string> stage_order, bool includes_training)
        : detailed_(detailed), terminal_(terminal) {
        std::size_t next_step = 1;
        for (auto& stage : stage_order) {
            if (!stage_steps_.contains(stage))
                stage_steps_.emplace(std::move(stage), next_step++);
        }
        if (includes_training)
            training_step_ = next_step++;
        completion_step_ = next_step++;
        total_steps_ = next_step - 1;
    }

    void ReconstructionOutput::set_training_log(std::ostream* stream) { training_log_ = stream; }

    void ReconstructionOutput::emit(const std::string& message, bool training) {
        terminal_ << message << std::endl;
        if (training && training_log_)
            *training_log_ << message << std::endl;
    }

    std::size_t ReconstructionOutput::stage_step(std::string_view name) const {
        const auto found = stage_steps_.find(std::string(name));
        return found == stage_steps_.end() ? 0 : found->second;
    }

    std::string ReconstructionOutput::concise_progress(std::size_t step, float overall_fraction) const {
        const auto percent = std::clamp(std::lround(overall_fraction * 100.0F), 0L, 100L);
        return std::format("Step {}/{}, progress: {}%", step, total_steps_, percent);
    }

    void ReconstructionOutput::stage_progress(std::string_view name, float local_fraction, float overall_fraction) {
        const std::string stage(name);
        if (completed_stages_.contains(stage))
            return;
        const auto planned_step = stage_step(stage);
        if (planned_step == 0 && !detailed_)
            return;
        const auto step = planned_step != 0 ? planned_step : std::max<std::size_t>(current_step_, 1);
        if (planned_step != 0)
            current_step_ = std::max(current_step_, planned_step);
        const auto now = std::chrono::steady_clock::now();
        if (!stage_started_.contains(stage)) {
            stage_started_[stage] = now;
            stage_bucket_[stage] = -1;
            if (detailed_)
                emit(std::format("Step {}/{}: Starting {} (overall {:.1f}%)",
                                 step, total_steps_, stage, overall_fraction * 100.0F));
        }
        const int percent = std::clamp(static_cast<int>(std::floor(local_fraction * 100.0F)), 0, 100);
        if (percent >= 100) {
            const auto elapsed = std::chrono::duration<double>(now - stage_started_.at(stage)).count();
            if (detailed_)
                emit(std::format("Step {}/{}: Completed {} ({:.1f}s, overall {:.1f}%)",
                                 step, total_steps_, stage, elapsed, overall_fraction * 100.0F));
            else
                emit(concise_progress(step, overall_fraction));
            completed_stages_.insert(stage);
            stage_started_.erase(stage);
            stage_bucket_.erase(stage);
        } else if (detailed_) {
            const int bucket = percent / 10;
            if (bucket > stage_bucket_.at(stage) && bucket > 0) {
                stage_bucket_[stage] = bucket;
                emit(std::format("Step {}/{}: {} {}% (overall {:.1f}%)",
                                 step, total_steps_, stage, bucket * 10, overall_fraction * 100.0F));
            }
        }
    }

    void ReconstructionOutput::training_started(std::size_t total_iterations, float overall_fraction) {
        if (training_step_ == 0)
            return;
        current_step_ = training_step_;
        training_bucket_ = -1;
        last_training_box_iteration_ = 0;
        training_started_at_ = std::chrono::steady_clock::now();
        if (detailed_)
            emit(std::format("Step {}/{}: GS training started ({} iterations, overall {:.1f}%)",
                             training_step_, total_steps_, total_iterations, overall_fraction * 100.0F), true);
        else
            emit(concise_progress(training_step_, overall_fraction), true);
    }

    void ReconstructionOutput::training_progress(int iteration, std::size_t total_iterations, float loss, int gaussians,
                                                  float overall_fraction) {
        if (training_step_ == 0 || total_iterations <= 0)
            return;
        const double fraction = std::clamp(static_cast<double>(iteration) / static_cast<double>(total_iterations), 0.0, 1.0);
        const int percent = static_cast<int>(std::floor(fraction * 100.0));
#ifndef DEBUG_BUILD
        // Debug already renders the trainer's original live progress bar.
        // Keep the Licht training box as a log-friendly snapshot. Its cadence
        // is independent of the existing numbered stage status below.
        constexpr int training_box_interval = 100;
        if (detailed_ && iteration > last_training_box_iteration_ &&
            (iteration % training_box_interval == 0 || fraction >= 1.0)) {
            constexpr std::size_t bar_width = 40;
            const auto filled = static_cast<std::size_t>(fraction * bar_width);
            std::string bar(filled, '=');
            if (filled < bar_width) {
                bar += '>';
                bar.append(bar_width - filled - 1, ' ');
            }
            const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - training_started_at_;
            const auto remaining = elapsed * ((1.0 - fraction) / fraction);
            emit(std::format("Training [{}] {}% [{}<{}] {}/{} | Loss: {:.4f} | Splats: {}",
                             bar, percent, training_duration(elapsed), training_duration(remaining),
                             iteration, total_iterations, loss, gaussians), true);
            last_training_box_iteration_ = iteration;
        }
#endif
        const int interval = detailed_ ? 5 : 10;
        const int bucket = percent / interval;
        if (bucket <= training_bucket_ || (bucket == 0 && iteration < total_iterations))
            return;
        training_bucket_ = bucket;
        if (detailed_)
            emit(std::format("Step {}/{}: GS training {}/{} ({}%, overall {:.1f}%, loss={:.6g}, gaussians={})",
                             training_step_, total_steps_, iteration, total_iterations, percent,
                             overall_fraction * 100.0F, loss, gaussians), true);
        else
            emit(concise_progress(training_step_, overall_fraction), true);
    }

    void ReconstructionOutput::training_finished(std::string_view outcome, float overall_fraction) {
        if (training_step_ == 0)
            return;
        if (detailed_)
            emit(std::format("Step {}/{}: GS training {} (overall {:.1f}%)",
                             training_step_, total_steps_, outcome, overall_fraction * 100.0F), true);
        else
            emit(concise_progress(training_step_, overall_fraction), true);
    }

    void ReconstructionOutput::completed(float overall_fraction) {
        current_step_ = completion_step_;
        if (detailed_)
            emit(std::format("Step {}/{}: Processing completed (overall {:.1f}%)",
                             completion_step_, total_steps_, overall_fraction * 100.0F), true);
        else
            emit(concise_progress(completion_step_, overall_fraction), true);
    }
} // namespace lfs::app
