/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scheduler.hpp"
#include "adam_optimizer.hpp"
#include "core/logger.hpp"
#include <cmath>
#include <string>
#include <unordered_map>

namespace lfs::training {

    WarmupExponentialLR::WarmupExponentialLR(
        AdamOptimizer& optimizer,
        double gamma,
        int warmup_steps,
        double warmup_start_factor,
        std::vector<ParamType> params_to_update)
        : optimizer_(optimizer),
          gamma_(gamma),
          warmup_steps_(warmup_steps),
          warmup_start_factor_(warmup_start_factor),
          current_step_(0),
          params_to_update_(params_to_update) {
        // Store initial learning rate
        initial_lr_ = optimizer.get_lr();
    }

    void ExponentialLR::step() {
        // Calculate decay factor
        double decay_factor = gamma_;

        // Map param type to name for logging
        static const std::unordered_map<ParamType, std::string> param_names = {
            {ParamType::Means, "means"},
            {ParamType::Sh0, "sh0"},
            {ParamType::ShN, "shN"},
            {ParamType::Scaling, "scaling"},
            {ParamType::Rotation, "rotation"},
            {ParamType::Opacity, "opacity"}};

        if (params_to_update_.empty()) {
            // Default behavior (MCMC): Update ONLY global LR (means uses this)
            double current_lr = optimizer_.get_lr();
            double new_lr = current_lr * decay_factor;
            LOG_DEBUG("ExponentialLR::step() - Global LR: {:.6e} → {:.6e} (gamma={:.6f})",
                      current_lr, new_lr, decay_factor);
            optimizer_.set_lr(static_cast<float>(new_lr));

            // Log other params for visibility (they stay constant)
            for (auto param_type : AdamOptimizer::all_param_types()) {
                if (optimizer_.has_param_lr(param_type)) {
                    LOG_DEBUG("  {} LR: {:.6e} (constant)",
                              param_names.at(param_type),
                              optimizer_.get_param_lr(param_type));
                }
            }
        } else {
            for (auto param_type : params_to_update_) {
                if (optimizer_.has_param_lr(param_type)) {
                    // Use double precision to match legacy behavior
                    double current_param_lr = optimizer_.get_param_lr(param_type);
                    double new_param_lr = current_param_lr * decay_factor;
                    optimizer_.set_param_lr(param_type, new_param_lr);

                    LOG_DEBUG("ExponentialLR::step() - {} LR: {:.15e} → {:.15e}",
                              param_names.at(param_type), current_param_lr, new_param_lr);
                }
            }

            // Also update global LR if it's being used by any param
            double current_lr = optimizer_.get_lr();
            double new_lr = current_lr * decay_factor;
            optimizer_.set_lr(static_cast<float>(new_lr));
        }
    }

    void WarmupExponentialLR::step() {
        current_step_++;

        // Get current LR BEFORE updating
        double old_global_lr = optimizer_.get_lr();

        double new_global_lr;
        double scale_factor; // How much to scale LRs (relative to initial)

        const char* phase = nullptr;
        if (current_step_ <= warmup_steps_) {
            // Linear warmup from start_factor to 1.0
            double progress = static_cast<double>(current_step_) / warmup_steps_;
            scale_factor = warmup_start_factor_ + (1.0 - warmup_start_factor_) * progress;
            new_global_lr = initial_lr_ * scale_factor;
            phase = "warmup";
        } else {
            // Exponential decay after warmup
            int decay_steps = current_step_ - warmup_steps_;
            scale_factor = std::pow(gamma_, decay_steps);
            new_global_lr = initial_lr_ * scale_factor;
            phase = "decay";
        }

        // Map param type to name for logging
        static const std::unordered_map<ParamType, std::string> param_names = {
            {ParamType::Means, "means"},
            {ParamType::Sh0, "sh0"},
            {ParamType::ShN, "shN"},
            {ParamType::Scaling, "scaling"},
            {ParamType::Rotation, "rotation"},
            {ParamType::Opacity, "opacity"}};

        if (params_to_update_.empty()) {
            // Default behavior: Update ONLY global LR
            LOG_DEBUG("WarmupExponentialLR::step() [{}] - step {}/{}: Global LR: {:.6e} → {:.6e} (scale={:.6f})",
                      phase, current_step_, warmup_steps_, old_global_lr, new_global_lr, scale_factor);
            optimizer_.set_lr(static_cast<float>(new_global_lr));

            // Log other params for visibility (they stay constant)
            for (auto param_type : AdamOptimizer::all_param_types()) {
                if (optimizer_.has_param_lr(param_type)) {
                    LOG_DEBUG("  {} LR: {:.6e} (constant)",
                              param_names.at(param_type),
                              optimizer_.get_param_lr(param_type));
                }
            }
        } else {
            // Update specified per-parameter LRs
            double lr_ratio = new_global_lr / old_global_lr;
            LOG_DEBUG("WarmupExponentialLR::step() [{}] - step {}/{}: ratio={:.6f}",
                      phase, current_step_, warmup_steps_, lr_ratio);

            for (auto param_type : params_to_update_) {
                if (optimizer_.has_param_lr(param_type)) {
                    // Use double precision to avoid LR drift!
                    double current_param_lr = optimizer_.get_param_lr(param_type);
                    double new_param_lr = current_param_lr * lr_ratio;
                    LOG_DEBUG("  {} LR: {:.15e} → {:.15e} (ratio: {:.6f})",
                              param_names.at(param_type), current_param_lr, new_param_lr, lr_ratio);
                    optimizer_.set_param_lr(param_type, new_param_lr);
                } else {
                    LOG_WARN("  {} LR: not explicitly set, cannot update", param_names.at(param_type));
                }
            }

            // Also update global LR
            optimizer_.set_lr(static_cast<float>(new_global_lr));
            LOG_DEBUG("  Global LR: {:.6e} → {:.6e}", old_global_lr, new_global_lr);
        }
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t SCHED_EXPONENTIAL_MAGIC = 0x4C465345; // "LFSE"
        constexpr uint32_t SCHED_WARMUP_MAGIC = 0x4C465357;      // "LFSW"
        constexpr uint32_t SCHED_VERSION = 1;
    } // namespace

    void ExponentialLR::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&SCHED_EXPONENTIAL_MAGIC), sizeof(SCHED_EXPONENTIAL_MAGIC));
        os.write(reinterpret_cast<const char*>(&SCHED_VERSION), sizeof(SCHED_VERSION));
        os.write(reinterpret_cast<const char*>(&gamma_), sizeof(gamma_));

        // Write params_to_update
        uint32_t num_params = static_cast<uint32_t>(params_to_update_.size());
        os.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        for (const auto& param : params_to_update_) {
            uint8_t param_val = static_cast<uint8_t>(param);
            os.write(reinterpret_cast<const char*>(&param_val), sizeof(param_val));
        }

        LOG_DEBUG("Serialized ExponentialLR: gamma={}", gamma_);
    }

    void ExponentialLR::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != SCHED_EXPONENTIAL_MAGIC) {
            throw std::runtime_error("Invalid ExponentialLR checkpoint: wrong magic");
        }
        if (version != SCHED_VERSION) {
            throw std::runtime_error("Unsupported ExponentialLR checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&gamma_), sizeof(gamma_));

        uint32_t num_params;
        is.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        params_to_update_.resize(num_params);
        for (uint32_t i = 0; i < num_params; ++i) {
            uint8_t param_val;
            is.read(reinterpret_cast<char*>(&param_val), sizeof(param_val));
            params_to_update_[i] = static_cast<ParamType>(param_val);
        }

        LOG_DEBUG("Deserialized ExponentialLR: gamma={}", gamma_);
    }

    void WarmupExponentialLR::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&SCHED_WARMUP_MAGIC), sizeof(SCHED_WARMUP_MAGIC));
        os.write(reinterpret_cast<const char*>(&SCHED_VERSION), sizeof(SCHED_VERSION));

        // Write state
        os.write(reinterpret_cast<const char*>(&gamma_), sizeof(gamma_));
        os.write(reinterpret_cast<const char*>(&warmup_steps_), sizeof(warmup_steps_));
        os.write(reinterpret_cast<const char*>(&warmup_start_factor_), sizeof(warmup_start_factor_));
        os.write(reinterpret_cast<const char*>(&current_step_), sizeof(current_step_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));

        // Write params_to_update
        uint32_t num_params = static_cast<uint32_t>(params_to_update_.size());
        os.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        for (const auto& param : params_to_update_) {
            uint8_t param_val = static_cast<uint8_t>(param);
            os.write(reinterpret_cast<const char*>(&param_val), sizeof(param_val));
        }

        LOG_DEBUG("Serialized WarmupExponentialLR: step={}, initial_lr={}", current_step_, initial_lr_);
    }

    void WarmupExponentialLR::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != SCHED_WARMUP_MAGIC) {
            throw std::runtime_error("Invalid WarmupExponentialLR checkpoint: wrong magic");
        }
        if (version != SCHED_VERSION) {
            throw std::runtime_error("Unsupported WarmupExponentialLR checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&gamma_), sizeof(gamma_));
        is.read(reinterpret_cast<char*>(&warmup_steps_), sizeof(warmup_steps_));
        is.read(reinterpret_cast<char*>(&warmup_start_factor_), sizeof(warmup_start_factor_));
        is.read(reinterpret_cast<char*>(&current_step_), sizeof(current_step_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));

        uint32_t num_params;
        is.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        params_to_update_.resize(num_params);
        for (uint32_t i = 0; i < num_params; ++i) {
            uint8_t param_val;
            is.read(reinterpret_cast<char*>(&param_val), sizeof(param_val));
            params_to_update_[i] = static_cast<ParamType>(param_val);
        }

        LOG_DEBUG("Deserialized WarmupExponentialLR: step={}, initial_lr={}", current_step_, initial_lr_);
    }

} // namespace lfs::training
