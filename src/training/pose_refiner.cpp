/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pose_refiner.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "kernels/pose_refine.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <istream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <stdexcept>
#include <string>

namespace lfs::training {
    namespace {

        constexpr float kDegToRad = 0.017453292519943295769f;
        constexpr float kHalfPi = 1.5707963267948966192f;
        constexpr uint32_t kPoseRefinerMagic = 0x45534F50u; // "POSE"
        constexpr uint32_t kPoseRefinerVersion = 1u;

        // nerfstudio ExponentialDecayScheduler: optional sin() warmup ramp from
        // lr_pre_warmup up to lr_init over warmup_steps, then a log-linear decay from
        // lr_init to lr_final over [warmup_steps, max_steps]. lr_final <= 0 (or >=
        // lr_init) keeps the LR flat. Measured on the global training iteration.
        float scheduled_lr(
            const int step,
            const float lr_init,
            const float lr_final,
            const float lr_pre_warmup,
            const int warmup_steps,
            const int max_steps) {
            if (!(lr_init > 0.0f)) {
                return 0.0f;
            }
            const int s = step < 0 ? 0 : step;
            if (warmup_steps > 0 && s < warmup_steps) {
                const float frac = std::clamp(
                    static_cast<float>(s) / static_cast<float>(warmup_steps), 0.0f, 1.0f);
                return lr_pre_warmup + (lr_init - lr_pre_warmup) * std::sin(kHalfPi * frac);
            }
            const float target = (lr_final > 0.0f && lr_final < lr_init) ? lr_final : lr_init;
            if (target == lr_init) {
                return lr_init; // decay disabled
            }
            const int span = max_steps - warmup_steps;
            if (span <= 0) {
                return target;
            }
            const float t = std::clamp(
                static_cast<float>(s - warmup_steps) / static_cast<float>(span), 0.0f, 1.0f);
            return std::exp(std::log(lr_init) * (1.0f - t) + std::log(target) * t);
        }

        struct PoseRefinerCheckpointHeader {
            uint32_t magic = kPoseRefinerMagic;
            uint32_t version = kPoseRefinerVersion;
            int32_t num_cameras = 0;
            int32_t adam_step = 0;
        };

        void check_cuda(cudaError_t err, const char* what) {
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
            }
        }

        template <typename T>
        void write_pod(std::ostream& os, const T& value) {
            os.write(reinterpret_cast<const char*>(&value), sizeof(T));
        }

        template <typename T>
        std::expected<T, std::string> read_pod(std::istream& is, const char* label) {
            T value{};
            is.read(reinterpret_cast<char*>(&value), sizeof(T));
            if (!is) {
                return std::unexpected(std::format("Failed to read pose refine {}", label));
            }
            return value;
        }

        std::expected<PoseRefinerCheckpointHeader, std::string> read_pose_refiner_header(std::istream& is) {
            auto header = read_pod<PoseRefinerCheckpointHeader>(is, "checkpoint header");
            if (!header) {
                return std::unexpected(header.error());
            }
            if (header->magic != kPoseRefinerMagic) {
                return std::unexpected("Invalid pose refine checkpoint block magic");
            }
            if (header->version != kPoseRefinerVersion) {
                return std::unexpected(std::format("Unsupported pose refine checkpoint block version {}", header->version));
            }
            if (header->num_cameras < 0) {
                return std::unexpected("Invalid pose refine checkpoint camera count");
            }
            return header;
        }

        std::vector<float> copy_tensor_to_host(
            const lfs::core::Tensor& tensor,
            const size_t count,
            cudaStream_t stream,
            const char* label) {
            std::vector<float> values(count, 0.0f);
            if (count == 0) {
                return values;
            }
            if (!tensor.is_valid() || tensor.numel() < count) {
                throw std::runtime_error(std::format("PoseRefiner {} tensor is invalid", label));
            }
            check_cuda(cudaStreamSynchronize(stream), "PoseRefiner tensor sync");
            check_cuda(cudaMemcpy(
                           values.data(),
                           tensor.ptr<float>(),
                           sizeof(float) * count,
                           cudaMemcpyDeviceToHost),
                       label);
            return values;
        }

        float norm3(const float* values) {
            return std::sqrt(values[0] * values[0] +
                             values[1] * values[1] +
                             values[2] * values[2]);
        }

        struct NormStats {
            double sum = 0.0;
            double sum_sq = 0.0;
            float max = 0.0f;
            size_t count = 0;

            void add(const float value) {
                sum += static_cast<double>(value);
                sum_sq += static_cast<double>(value) * static_cast<double>(value);
                max = std::max(max, value);
                ++count;
            }

            [[nodiscard]] float mean() const {
                return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
            }

            [[nodiscard]] float l2() const {
                return static_cast<float>(std::sqrt(sum_sq));
            }
        };

        bool row_is_finite(const float* row, const int count) {
            for (int i = 0; i < count; ++i) {
                if (!std::isfinite(row[i])) {
                    return false;
                }
            }
            return true;
        }

        void fill_delta_stats(
            PoseRefiner::Diagnostics& diagnostics,
            const std::vector<float>& deltas,
            const float max_trans,
            const float max_rot_rad) {
            NormStats trans;
            NormStats rot;
            size_t trans_clamped = 0;
            size_t rot_clamped = 0;
            const float trans_eps = max_trans > 0.0f ? std::max(1.0e-8f, max_trans * 1.0e-4f) : 0.0f;
            const float rot_eps = max_rot_rad > 0.0f ? std::max(1.0e-8f, max_rot_rad * 1.0e-4f) : 0.0f;

            for (size_t i = 0; i < diagnostics.real_count; ++i) {
                const float* d = deltas.data() + i * 6;
                const float trans_norm = norm3(d);
                const float rot_norm = norm3(d + 3);
                trans.add(trans_norm);
                rot.add(rot_norm);
                diagnostics.finite = diagnostics.finite && row_is_finite(d, 6);
                if (max_trans > 0.0f && trans_norm >= max_trans - trans_eps) {
                    ++trans_clamped;
                }
                if (max_rot_rad > 0.0f && rot_norm >= max_rot_rad - rot_eps) {
                    ++rot_clamped;
                }
            }

            diagnostics.delta_trans_mean = trans.mean();
            diagnostics.delta_trans_max = trans.max;
            diagnostics.delta_rot_mean_deg = rot.mean() / kDegToRad;
            diagnostics.delta_rot_max_deg = rot.max / kDegToRad;
            diagnostics.clamp_trans_rows = trans_clamped;
            diagnostics.clamp_rot_rows = rot_clamped;

            if (diagnostics.base_index >= 0 &&
                static_cast<size_t>(diagnostics.base_index) < diagnostics.real_count) {
                const float* selected = deltas.data() + static_cast<size_t>(diagnostics.base_index) * 6;
                diagnostics.selected_delta_trans = norm3(selected);
                diagnostics.selected_delta_rot_deg = norm3(selected + 3) / kDegToRad;
            }
        }

        void fill_grad_stats(
            PoseRefiner::Diagnostics& diagnostics,
            const std::vector<float>& grads) {
            NormStats trans;
            NormStats rot;
            size_t nonzero_rows = 0;

            for (size_t i = 0; i < diagnostics.real_count; ++i) {
                const float* g = grads.data() + i * 6;
                const float trans_norm = norm3(g);
                const float rot_norm = norm3(g + 3);
                trans.add(trans_norm);
                rot.add(rot_norm);
                diagnostics.finite = diagnostics.finite && row_is_finite(g, 6);
                if (trans_norm > 0.0f || rot_norm > 0.0f) {
                    ++nonzero_rows;
                }
            }

            diagnostics.grad_trans_l2 = trans.l2();
            diagnostics.grad_trans_mean = trans.mean();
            diagnostics.grad_trans_max = trans.max;
            diagnostics.grad_rot_l2 = rot.l2();
            diagnostics.grad_rot_mean = rot.mean();
            diagnostics.grad_rot_max = rot.max;
            diagnostics.nonzero_grad_rows = nonzero_rows;

            if (diagnostics.base_index >= 0 &&
                static_cast<size_t>(diagnostics.base_index) < diagnostics.real_count) {
                const float* selected = grads.data() + static_cast<size_t>(diagnostics.base_index) * 6;
                diagnostics.selected_grad_trans = norm3(selected);
                diagnostics.selected_grad_rot = norm3(selected + 3);
            }
        }

        void fill_step_stats(
            PoseRefiner::Diagnostics& diagnostics,
            const std::vector<float>& deltas_after) {
            if (diagnostics.delta_before.size() != deltas_after.size()) {
                diagnostics.finite = false;
                return;
            }

            NormStats trans;
            NormStats rot;
            for (size_t i = 0; i < diagnostics.real_count; ++i) {
                float step[6]{};
                for (int j = 0; j < 6; ++j) {
                    const size_t idx = i * 6 + static_cast<size_t>(j);
                    step[j] = deltas_after[idx] - diagnostics.delta_before[idx];
                }
                const float trans_norm = norm3(step);
                const float rot_norm = norm3(step + 3);
                trans.add(trans_norm);
                rot.add(rot_norm);
                diagnostics.finite = diagnostics.finite && row_is_finite(step, 6);
            }

            diagnostics.step_trans_mean = trans.mean();
            diagnostics.step_trans_max = trans.max;
            diagnostics.step_rot_mean_deg = rot.mean() / kDegToRad;
            diagnostics.step_rot_max_deg = rot.max / kDegToRad;

            if (diagnostics.base_index >= 0 &&
                static_cast<size_t>(diagnostics.base_index) < diagnostics.real_count) {
                float selected[6]{};
                for (int j = 0; j < 6; ++j) {
                    const size_t idx = static_cast<size_t>(diagnostics.base_index) * 6 + static_cast<size_t>(j);
                    selected[j] = deltas_after[idx] - diagnostics.delta_before[idx];
                }
                diagnostics.selected_step_trans = norm3(selected);
                diagnostics.selected_step_rot_deg = norm3(selected + 3) / kDegToRad;
            }
        }

        std::array<float, 16> identity4() {
            return {1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f};
        }

        std::array<float, 16> mat4_mul(
            const std::array<float, 16>& a,
            const std::array<float, 16>& b) {
            std::array<float, 16> out{};
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    float v = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        v += a[r * 4 + k] * b[k * 4 + c];
                    }
                    out[r * 4 + c] = v;
                }
            }
            return out;
        }

        std::array<float, 16> invert_rigid(const std::array<float, 16>& c2w) {
            auto w2c = identity4();
            w2c[0] = c2w[0];
            w2c[1] = c2w[4];
            w2c[2] = c2w[8];
            w2c[4] = c2w[1];
            w2c[5] = c2w[5];
            w2c[6] = c2w[9];
            w2c[8] = c2w[2];
            w2c[9] = c2w[6];
            w2c[10] = c2w[10];

            const float cx = c2w[3];
            const float cy = c2w[7];
            const float cz = c2w[11];
            w2c[3] = -(w2c[0] * cx + w2c[1] * cy + w2c[2] * cz);
            w2c[7] = -(w2c[4] * cx + w2c[5] * cy + w2c[6] * cz);
            w2c[11] = -(w2c[8] * cx + w2c[9] * cy + w2c[10] * cz);
            return w2c;
        }

        std::array<float, 9> exp_so3(const std::array<float, 3>& w) {
            const float wx = w[0];
            const float wy = w[1];
            const float wz = w[2];
            const float theta2 = wx * wx + wy * wy + wz * wz;
            float a = 1.0f;
            float b = 0.5f;
            if (theta2 > 1.0e-12f) {
                const float theta = std::sqrt(theta2);
                a = std::sin(theta) / theta;
                b = (1.0f - std::cos(theta)) / theta2;
            }

            const float k01 = -wz;
            const float k02 = wy;
            const float k10 = wz;
            const float k12 = -wx;
            const float k20 = -wy;
            const float k21 = wx;

            return {
                1.0f + b * (k01 * k10 + k02 * k20), a * k01 + b * (k02 * k21), a * k02 + b * (k01 * k12),
                a * k10 + b * (k12 * k20), 1.0f + b * (k10 * k01 + k12 * k21), a * k12 + b * (k10 * k02),
                a * k20 + b * (k21 * k10), a * k21 + b * (k20 * k01), 1.0f + b * (k20 * k02 + k21 * k12)};
        }

        std::array<float, 16> compose_c2w_so3xr3(
            const std::array<float, 16>& base_c2w,
            const std::array<float, 6>& delta) {
            const std::array<float, 3> rot_delta = {delta[3], delta[4], delta[5]};
            const auto R_delta = exp_so3(rot_delta);
            auto out = identity4();

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        v += R_delta[r * 3 + k] * base_c2w[k * 4 + c];
                    }
                    out[r * 4 + c] = v;
                }
            }
            out[3] = base_c2w[3] + delta[0];
            out[7] = base_c2w[7] + delta[1];
            out[11] = base_c2w[11] + delta[2];
            return out;
        }

        std::array<float, 16> camera_c2w(const lfs::core::Camera& camera) {
            auto R_cpu = camera.R().cpu().contiguous();
            auto T_cpu = camera.T().cpu().contiguous();
            const float* R = R_cpu.ptr<float>();
            const float* T = T_cpu.ptr<float>();

            auto c2w = identity4();
            c2w[0] = R[0];
            c2w[1] = R[3];
            c2w[2] = R[6];
            c2w[4] = R[1];
            c2w[5] = R[4];
            c2w[6] = R[7];
            c2w[8] = R[2];
            c2w[9] = R[5];
            c2w[10] = R[8];

            c2w[3] = -(c2w[0] * T[0] + c2w[1] * T[1] + c2w[2] * T[2]);
            c2w[7] = -(c2w[4] * T[0] + c2w[5] * T[1] + c2w[6] * T[2]);
            c2w[11] = -(c2w[8] * T[0] + c2w[9] * T[1] + c2w[10] * T[2]);
            return c2w;
        }

        std::array<float, 9> rotation_from_w2c(const std::array<float, 16>& w2c) {
            return {w2c[0], w2c[1], w2c[2],
                    w2c[4], w2c[5], w2c[6],
                    w2c[8], w2c[9], w2c[10]};
        }

        std::array<float, 3> translation_from_w2c(const std::array<float, 16>& w2c) {
            return {w2c[3], w2c[7], w2c[11]};
        }

        nlohmann::json mat4_json(const std::array<float, 16>& m) {
            return nlohmann::json::array({
                nlohmann::json::array({m[0], m[1], m[2], m[3]}),
                nlohmann::json::array({m[4], m[5], m[6], m[7]}),
                nlohmann::json::array({m[8], m[9], m[10], m[11]}),
                nlohmann::json::array({m[12], m[13], m[14], m[15]})});
        }

        nlohmann::json mat3_json(const std::array<float, 9>& m) {
            return nlohmann::json::array({
                nlohmann::json::array({m[0], m[1], m[2]}),
                nlohmann::json::array({m[3], m[4], m[5]}),
                nlohmann::json::array({m[6], m[7], m[8]})});
        }

        nlohmann::json vec3_json(const std::array<float, 3>& v) {
            return nlohmann::json::array({v[0], v[1], v[2]});
        }

        nlohmann::json vec6_json(const std::array<float, 6>& v) {
            return nlohmann::json::array({v[0], v[1], v[2], v[3], v[4], v[5]});
        }

        std::array<double, 4> rotmat_to_qvec(const std::array<float, 9>& R) {
            const double r00 = R[0], r01 = R[1], r02 = R[2];
            const double r10 = R[3], r11 = R[4], r12 = R[5];
            const double r20 = R[6], r21 = R[7], r22 = R[8];
            std::array<double, 4> q{};

            const double trace = r00 + r11 + r22;
            if (trace > 0.0) {
                const double s = std::sqrt(trace + 1.0) * 2.0;
                q[0] = 0.25 * s;
                q[1] = (r21 - r12) / s;
                q[2] = (r02 - r20) / s;
                q[3] = (r10 - r01) / s;
            } else if (r00 > r11 && r00 > r22) {
                const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
                q[0] = (r21 - r12) / s;
                q[1] = 0.25 * s;
                q[2] = (r01 + r10) / s;
                q[3] = (r02 + r20) / s;
            } else if (r11 > r22) {
                const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
                q[0] = (r02 - r20) / s;
                q[1] = (r01 + r10) / s;
                q[2] = 0.25 * s;
                q[3] = (r12 + r21) / s;
            } else {
                const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
                q[0] = (r10 - r01) / s;
                q[1] = (r02 + r20) / s;
                q[2] = (r12 + r21) / s;
                q[3] = 0.25 * s;
            }

            const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (norm > 0.0) {
                for (double& v : q) {
                    v /= norm;
                }
            }
            if (q[0] < 0.0) {
                for (double& v : q) {
                    v = -v;
                }
            }
            return q;
        }

        std::expected<void, std::string> write_json_file(
            const std::filesystem::path& path,
            const nlohmann::json& json) {
            std::ofstream out(path, std::ios::binary);
            if (!out) {
                return std::unexpected("Failed to open pose refine export file: " +
                                       lfs::core::path_to_utf8(path));
            }
            out << json.dump(2);
            if (!out) {
                return std::unexpected("Failed to write pose refine export file: " +
                                       lfs::core::path_to_utf8(path));
            }
            return {};
        }

    } // namespace

    PoseRefiner::PoseRefiner(const lfs::core::param::OptimizationParameters& params)
        : enabled_(params.refine_camera_pose),
          params_(params) {}

    std::expected<void, std::string> PoseRefiner::initialize(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        if (!enabled_) {
            return {};
        }

        real_states_.clear();
        pseudo_states_.clear();
        real_index_by_uid_.clear();
        pseudo_index_by_uid_.clear();
        adam_step_ = 0;

        for (const auto& cam : cameras) {
            if (!cam || cam->is_pseudo()) {
                continue;
            }
            const int index = static_cast<int>(real_states_.size());
            real_index_by_uid_[cam->uid()] = index;
            real_states_.push_back(RealState{
                .uid = cam->uid(),
                .base_c2w = camera_c2w(*cam)});
        }

        if (real_states_.empty()) {
            return std::unexpected("Camera pose refine requires at least one real training camera");
        }

        std::vector<float> base_c2w_flat;
        base_c2w_flat.reserve(real_states_.size() * 16);
        for (const auto& state : real_states_) {
            base_c2w_flat.insert(base_c2w_flat.end(), state.base_c2w.begin(), state.base_c2w.end());
        }
        base_c2w_device_ = lfs::core::Tensor::from_vector(
            base_c2w_flat,
            {real_states_.size(), std::size_t{16}},
            lfs::core::Device::CPU).to(lfs::core::Device::CUDA).contiguous();

        std::vector<float> relative_flat;
        for (const auto& cam : cameras) {
            if (!cam || !cam->is_pseudo()) {
                continue;
            }
            const int base_uid = cam->pseudo_base_camera_uid();
            const auto base_it = real_index_by_uid_.find(base_uid);
            if (base_uid < 0 || base_it == real_index_by_uid_.end()) {
                LOG_WARN("[PoseRefine] pseudo camera '{}' has no trainable base_camera_uid {}; it will keep its absolute pose",
                         cam->image_name(), base_uid);
                continue;
            }

            const auto pseudo_c2w = camera_c2w(*cam);
            const auto base_inv = invert_rigid(real_states_[base_it->second].base_c2w);
            const auto relative = mat4_mul(base_inv, pseudo_c2w);
            const int relative_index = static_cast<int>(pseudo_states_.size());
            pseudo_index_by_uid_[cam->uid()] = relative_index;
            pseudo_states_.push_back(PseudoState{
                .uid = cam->uid(),
                .base_uid = base_uid,
                .base_index = base_it->second,
                .relative_index = relative_index,
                .relative_c2w = relative});
            relative_flat.insert(relative_flat.end(), relative.begin(), relative.end());
        }

        if (!relative_flat.empty()) {
            pseudo_relative_c2w_device_ = lfs::core::Tensor::from_vector(
                relative_flat,
                {pseudo_states_.size(), std::size_t{16}},
                lfs::core::Device::CPU).to(lfs::core::Device::CUDA).contiguous();
        }

        delta_ = lfs::core::Tensor::zeros({real_states_.size(), std::size_t{6}}, lfs::core::Device::CUDA);
        grad_delta_ = lfs::core::Tensor::zeros({real_states_.size(), std::size_t{6}}, lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros({real_states_.size(), std::size_t{6}}, lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros({real_states_.size(), std::size_t{6}}, lfs::core::Device::CUDA);
        regularization_loss_ = lfs::core::Tensor::zeros({std::size_t{1}}, lfs::core::Device::CUDA);

        LOG_INFO("[PoseRefine] initialized real_cameras={} pseudo_followers={} mode={} stop_iter={} lr_trans={:.2e} lr_rot={:.2e} lr_final_trans={:.2e} lr_final_rot={:.2e} lr_pre_warmup={:.2e} lr_warmup_steps={} lr_max_steps={} l2_trans={:.2e} l2_rot={:.2e} max_trans={:.4e} max_rot_deg={:.4e} log_every={}",
                 real_states_.size(),
                 pseudo_states_.size(),
                 params_.pose_refine_mode,
                 params_.pose_refine_stop_iter,
                 params_.pose_refine_lr_trans,
                 params_.pose_refine_lr_rot,
                 params_.pose_refine_lr_final_trans,
                 params_.pose_refine_lr_final_rot,
                 params_.pose_refine_lr_pre_warmup,
                 params_.pose_refine_lr_warmup_steps,
                 params_.pose_refine_lr_max_steps,
                 params_.pose_refine_l2_trans,
                 params_.pose_refine_l2_rot,
                 params_.pose_refine_max_trans,
                 params_.pose_refine_max_rot_deg,
                 params_.pose_refine_log_every);
        if (params_.pose_refine_lr_trans == 0.0f && params_.pose_refine_lr_rot == 0.0f) {
            LOG_WARN("[PoseRefine] enabled but both pose_refine_lr_trans and pose_refine_lr_rot are zero; pose deltas will not update");
        }
        return {};
    }

    bool PoseRefiner::is_active(const int iteration) const noexcept {
        if (!enabled_ || real_states_.empty()) {
            return false;
        }
        if (params_.pose_refine_stop_iter > 0 &&
            static_cast<size_t>(iteration) >= params_.pose_refine_stop_iter) {
            return false;
        }
        return true;
    }

    void PoseRefiner::zero_grad() {
        zero_grad_pending_ = true;
    }

    void PoseRefiner::use_stream(cudaStream_t stream) {
        pose_stream_ = stream;
        if (base_c2w_device_.is_valid()) {
            base_c2w_device_.set_stream(pose_stream_);
        }
        if (pseudo_relative_c2w_device_.is_valid()) {
            pseudo_relative_c2w_device_.set_stream(pose_stream_);
        }
        if (delta_.is_valid()) {
            delta_.set_stream(pose_stream_);
        }
        if (grad_delta_.is_valid()) {
            grad_delta_.set_stream(pose_stream_);
        }
        if (exp_avg_.is_valid()) {
            exp_avg_.set_stream(pose_stream_);
        }
        if (exp_avg_sq_.is_valid()) {
            exp_avg_sq_.set_stream(pose_stream_);
        }
        if (regularization_loss_.is_valid()) {
            regularization_loss_.set_stream(pose_stream_);
        }
        if (grad_w2c_scratch_.is_valid()) {
            grad_w2c_scratch_.set_stream(pose_stream_);
        }
    }

    void PoseRefiner::zero_grad_if_pending() {
        if (!zero_grad_pending_) {
            return;
        }
        if (grad_delta_.is_valid()) {
            grad_delta_.zero_();
        }
        zero_grad_pending_ = false;
    }

    lfs::core::Tensor& PoseRefiner::grad_w2c_scratch(cudaStream_t stream) {
        use_stream(stream);
        zero_grad_if_pending();
        if (!grad_w2c_scratch_.is_valid() || grad_w2c_scratch_.numel() != 16) {
            grad_w2c_scratch_ = lfs::core::Tensor::zeros({std::size_t{4}, std::size_t{4}}, lfs::core::Device::CUDA);
            grad_w2c_scratch_.set_stream(pose_stream_);
        }
        return grad_w2c_scratch_;
    }

    void PoseRefiner::accumulate_w2c_gradient(
        const lfs::core::Camera& camera,
        const lfs::core::Tensor& grad_w2c) {
        if (!enabled_ || !grad_w2c.is_valid() || grad_w2c.numel() < 16) {
            return;
        }
        use_stream(grad_w2c.stream());
        zero_grad_if_pending();

        int base_index = -1;
        const float* relative_ptr = nullptr;
        if (camera.is_pseudo()) {
            const auto pseudo_it = pseudo_index_by_uid_.find(camera.uid());
            if (pseudo_it == pseudo_index_by_uid_.end()) {
                return;
            }
            const auto& pseudo = pseudo_states_[pseudo_it->second];
            base_index = pseudo.base_index;
            if (pseudo_relative_c2w_device_.is_valid()) {
                relative_ptr = pseudo_relative_c2w_device_.ptr<float>() + pseudo.relative_index * 16;
            }
        } else {
            const auto real_it = real_index_by_uid_.find(camera.uid());
            if (real_it == real_index_by_uid_.end()) {
                return;
            }
            base_index = real_it->second;
        }

        if (base_index < 0) {
            return;
        }

        kernels::launch_pose_refine_accumulate_w2c(
            grad_w2c.ptr<float>(),
            base_c2w_device_.ptr<float>() + base_index * 16,
            relative_ptr,
            delta_.ptr<float>(),
            grad_delta_.ptr<float>(),
            base_index,
            grad_w2c.stream());
    }

    lfs::core::Tensor PoseRefiner::regularization_loss() {
        if (!enabled_ || real_states_.empty()) {
            return {};
        }
        use_stream(pose_stream_);
        zero_grad_if_pending();
        if (!regularization_loss_.is_valid()) {
            regularization_loss_ = lfs::core::Tensor::zeros({std::size_t{1}}, lfs::core::Device::CUDA);
            regularization_loss_.set_stream(pose_stream_);
        } else {
            regularization_loss_.zero_();
        }
        kernels::launch_pose_refine_regularization(
            delta_.ptr<float>(),
            grad_delta_.ptr<float>(),
            regularization_loss_.ptr<float>(),
            static_cast<int>(real_states_.size()),
            params_.pose_refine_l2_trans,
            params_.pose_refine_l2_rot,
            regularization_loss_.stream());
        return regularization_loss_;
    }

    void PoseRefiner::step(const int iteration) {
        if (!enabled_ || real_states_.empty()) {
            return;
        }
        use_stream(pose_stream_);
        ++adam_step_;

        // nerfstudio-style schedule, measured on the global training iteration.
        // Pose refinement is active for the whole run when enabled.
        int max_steps = static_cast<int>(params_.pose_refine_lr_max_steps);
        if (max_steps <= 0) {
            max_steps = static_cast<int>(params_.iterations);
        }
        const int warmup = static_cast<int>(params_.pose_refine_lr_warmup_steps);
        const float pre_warmup = params_.pose_refine_lr_pre_warmup;
        last_lr_trans_ = scheduled_lr(
            iteration, params_.pose_refine_lr_trans, params_.pose_refine_lr_final_trans,
            pre_warmup, warmup, max_steps);
        last_lr_rot_ = scheduled_lr(
            iteration, params_.pose_refine_lr_rot, params_.pose_refine_lr_final_rot,
            pre_warmup, warmup, max_steps);

        kernels::launch_pose_refine_adam_step(
            delta_.ptr<float>(),
            exp_avg_.ptr<float>(),
            exp_avg_sq_.ptr<float>(),
            grad_delta_.ptr<float>(),
            static_cast<int>(real_states_.size()),
            last_lr_trans_,
            last_lr_rot_,
            0.9f,
            0.999f,
            1.0e-8f,
            adam_step_,
            params_.pose_refine_max_trans,
            params_.pose_refine_max_rot_deg * kDegToRad,
            delta_.stream());
    }

    PoseRefiner::Diagnostics PoseRefiner::capture_diagnostics_pre_step(
        const lfs::core::Camera& camera,
        const float regularization_loss) const {
        Diagnostics diagnostics;
        diagnostics.camera_uid = camera.uid();
        diagnostics.camera_name = camera.image_name();
        diagnostics.camera_is_pseudo = camera.is_pseudo();
        diagnostics.real_count = real_states_.size();
        diagnostics.pseudo_count = pseudo_states_.size();
        diagnostics.reg_loss = regularization_loss;
        diagnostics.adam_step = adam_step_ + 1;
        diagnostics.finite = std::isfinite(regularization_loss);

        if (!enabled_ || real_states_.empty() || !delta_.is_valid() || !grad_delta_.is_valid()) {
            diagnostics.finite = false;
            return diagnostics;
        }

        if (camera.is_pseudo()) {
            const auto pseudo_it = pseudo_index_by_uid_.find(camera.uid());
            if (pseudo_it != pseudo_index_by_uid_.end()) {
                const auto& pseudo = pseudo_states_[pseudo_it->second];
                diagnostics.base_uid = pseudo.base_uid;
                diagnostics.base_index = pseudo.base_index;
            }
        } else {
            diagnostics.base_uid = camera.uid();
            const auto real_it = real_index_by_uid_.find(camera.uid());
            if (real_it != real_index_by_uid_.end()) {
                diagnostics.base_index = real_it->second;
            }
        }

        const size_t value_count = real_states_.size() * 6;
        diagnostics.delta_before = copy_tensor_to_host(
            delta_,
            value_count,
            pose_stream_,
            "PoseRefiner diagnostic delta copy");
        const auto grads = copy_tensor_to_host(
            grad_delta_,
            value_count,
            pose_stream_,
            "PoseRefiner diagnostic gradient copy");
        fill_grad_stats(diagnostics, grads);
        fill_delta_stats(
            diagnostics,
            diagnostics.delta_before,
            params_.pose_refine_max_trans,
            params_.pose_refine_max_rot_deg * kDegToRad);
        return diagnostics;
    }

    void PoseRefiner::capture_diagnostics_post_step(Diagnostics& diagnostics) const {
        if (!enabled_ || real_states_.empty() || !delta_.is_valid()) {
            diagnostics.finite = false;
            return;
        }

        diagnostics.lr_trans = last_lr_trans_;
        diagnostics.lr_rot = last_lr_rot_;
        const size_t value_count = real_states_.size() * 6;
        const auto deltas_after = copy_tensor_to_host(
            delta_,
            value_count,
            pose_stream_,
            "PoseRefiner diagnostic post-step delta copy");
        fill_step_stats(diagnostics, deltas_after);
        fill_delta_stats(
            diagnostics,
            deltas_after,
            params_.pose_refine_max_trans,
            params_.pose_refine_max_rot_deg * kDegToRad);
    }

    std::string PoseRefiner::diagnostics_summary(const Diagnostics& d) const {
        // Compact one-line summary. Conventions: grad t/r are L2 norms of the
        // selected camera's translation/rotation gradient; delta/step report the
        // max magnitude over all real cameras (translation in scene units, rotation
        // in degrees); act = cameras with nonzero grad this step / total.
        return std::format(
            "cam='{}'(uid={}{}) base={} reg={:.3e} | lr t={:.3e} r={:.3e} | grad t={:.3e} r={:.3e} | "
            "delta_max t={:.3e} r={:.3e}deg | step_max t={:.3e} r={:.3e}deg | "
            "act={}/{} clamp t={} r={} adam={}",
            d.camera_name,
            d.camera_uid,
            d.camera_is_pseudo ? ",pseudo" : "",
            d.base_uid,
            d.reg_loss,
            d.lr_trans,
            d.lr_rot,
            d.selected_grad_trans,
            d.selected_grad_rot,
            d.delta_trans_max,
            d.delta_rot_max_deg,
            d.step_trans_max,
            d.step_rot_max_deg,
            d.nonzero_grad_rows,
            d.real_count,
            d.clamp_trans_rows,
            d.clamp_rot_rows,
            d.adam_step);
    }

    std::string PoseRefiner::inactive_reason(
        const int iteration,
        const bool in_controller_phase) const {
        if (!enabled_) {
            return "disabled";
        }
        if (real_states_.empty()) {
            return "not_initialized";
        }
        if (params_.pose_refine_stop_iter > 0 &&
            static_cast<size_t>(iteration) >= params_.pose_refine_stop_iter) {
            return "past_stop_iter";
        }
        if (in_controller_phase) {
            return "controller_phase";
        }
        return "active";
    }

    std::string PoseRefiner::metrics_summary() const {
        if (!enabled_ || real_states_.empty() || !delta_.is_valid()) {
            return "pose_delta=disabled";
        }

        std::vector<float> deltas(real_states_.size() * 6);
        check_cuda(cudaStreamSynchronize(pose_stream_), "PoseRefiner metrics sync");
        check_cuda(cudaMemcpy(
                       deltas.data(),
                       delta_.ptr<float>(),
                       sizeof(float) * deltas.size(),
                       cudaMemcpyDeviceToHost),
                   "PoseRefiner metrics copy");

        float trans_sum = 0.0f;
        float rot_sum = 0.0f;
        float trans_max = 0.0f;
        float rot_max = 0.0f;
        for (size_t i = 0; i < real_states_.size(); ++i) {
            const float* d = deltas.data() + i * 6;
            const float trans_norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            const float rot_norm = std::sqrt(d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
            trans_sum += trans_norm;
            rot_sum += rot_norm;
            trans_max = std::max(trans_max, trans_norm);
            rot_max = std::max(rot_max, rot_norm);
        }

        const float inv_n = 1.0f / static_cast<float>(real_states_.size());
        constexpr float rad_to_deg = 57.29577951308232f;
        return std::format(
            "delta_trans_mean={:.4e} delta_trans_max={:.4e} delta_rot_mean_deg={:.4e} delta_rot_max_deg={:.4e}",
            trans_sum * inv_n,
            trans_max,
            rot_sum * inv_n * rad_to_deg,
            rot_max * rad_to_deg);
    }

    size_t PoseRefiner::serialized_size_bytes() const noexcept {
        const size_t n = real_states_.size();
        return sizeof(PoseRefinerCheckpointHeader) +
               n * sizeof(int32_t) +
               n * 6 * sizeof(float) * 3;
    }

    void PoseRefiner::serialize(std::ostream& os) const {
        if (!enabled_) {
            return;
        }

        const int32_t num_cameras = static_cast<int32_t>(real_states_.size());
        const size_t value_count = static_cast<size_t>(num_cameras) * 6;
        const auto delta_host = copy_tensor_to_host(delta_, value_count, pose_stream_, "PoseRefiner delta copy");
        const auto exp_avg_host = copy_tensor_to_host(exp_avg_, value_count, pose_stream_, "PoseRefiner exp_avg copy");
        const auto exp_avg_sq_host = copy_tensor_to_host(exp_avg_sq_, value_count, pose_stream_, "PoseRefiner exp_avg_sq copy");

        PoseRefinerCheckpointHeader header{};
        header.num_cameras = num_cameras;
        header.adam_step = adam_step_;
        write_pod(os, header);

        for (const auto& state : real_states_) {
            const int32_t uid = static_cast<int32_t>(state.uid);
            write_pod(os, uid);
        }

        if (!delta_host.empty()) {
            os.write(reinterpret_cast<const char*>(delta_host.data()),
                     static_cast<std::streamsize>(delta_host.size() * sizeof(float)));
            os.write(reinterpret_cast<const char*>(exp_avg_host.data()),
                     static_cast<std::streamsize>(exp_avg_host.size() * sizeof(float)));
            os.write(reinterpret_cast<const char*>(exp_avg_sq_host.data()),
                     static_cast<std::streamsize>(exp_avg_sq_host.size() * sizeof(float)));
        }
        if (!os) {
            throw std::runtime_error("Failed to write pose refine checkpoint block");
        }
    }

    std::expected<void, std::string> PoseRefiner::deserialize(std::istream& is) {
        try {
            if (!enabled_) {
                return std::unexpected("Cannot restore pose refine state when pose refine is disabled");
            }
            if (!delta_.is_valid() || !exp_avg_.is_valid() || !exp_avg_sq_.is_valid()) {
                return std::unexpected("Cannot restore pose refine state before PoseRefiner initialization");
            }

            auto header = read_pose_refiner_header(is);
            if (!header) {
                return std::unexpected(header.error());
            }

            const size_t saved_count = static_cast<size_t>(header->num_cameras);
            if (saved_count != real_states_.size()) {
                return std::unexpected(std::format(
                    "Pose refine checkpoint camera count mismatch: checkpoint={}, current={}",
                    saved_count,
                    real_states_.size()));
            }

            std::vector<int32_t> saved_uids(saved_count);
            for (size_t i = 0; i < saved_count; ++i) {
                auto uid = read_pod<int32_t>(is, "camera uid");
                if (!uid) {
                    return std::unexpected(uid.error());
                }
                saved_uids[i] = *uid;
            }

            const size_t value_count = saved_count * 6;
            std::vector<float> saved_delta(value_count);
            std::vector<float> saved_exp_avg(value_count);
            std::vector<float> saved_exp_avg_sq(value_count);
            if (value_count > 0) {
                is.read(reinterpret_cast<char*>(saved_delta.data()),
                        static_cast<std::streamsize>(value_count * sizeof(float)));
                is.read(reinterpret_cast<char*>(saved_exp_avg.data()),
                        static_cast<std::streamsize>(value_count * sizeof(float)));
                is.read(reinterpret_cast<char*>(saved_exp_avg_sq.data()),
                        static_cast<std::streamsize>(value_count * sizeof(float)));
                if (!is) {
                    return std::unexpected("Failed to read pose refine optimizer tensors");
                }
            }

            std::vector<float> delta_host(real_states_.size() * 6, 0.0f);
            std::vector<float> exp_avg_host(real_states_.size() * 6, 0.0f);
            std::vector<float> exp_avg_sq_host(real_states_.size() * 6, 0.0f);

            for (size_t saved_i = 0; saved_i < saved_count; ++saved_i) {
                const auto it = real_index_by_uid_.find(saved_uids[saved_i]);
                if (it == real_index_by_uid_.end()) {
                    return std::unexpected(std::format(
                        "Pose refine checkpoint references unknown camera uid {}",
                        saved_uids[saved_i]));
                }
                const size_t dst = static_cast<size_t>(it->second) * 6;
                const size_t src = saved_i * 6;
                std::copy_n(saved_delta.data() + src, 6, delta_host.data() + dst);
                std::copy_n(saved_exp_avg.data() + src, 6, exp_avg_host.data() + dst);
                std::copy_n(saved_exp_avg_sq.data() + src, 6, exp_avg_sq_host.data() + dst);
            }

            const size_t bytes = delta_host.size() * sizeof(float);
            if (bytes > 0) {
                check_cuda(cudaMemcpy(delta_.ptr<float>(), delta_host.data(), bytes, cudaMemcpyHostToDevice),
                           "PoseRefiner delta restore");
                check_cuda(cudaMemcpy(exp_avg_.ptr<float>(), exp_avg_host.data(), bytes, cudaMemcpyHostToDevice),
                           "PoseRefiner exp_avg restore");
                check_cuda(cudaMemcpy(exp_avg_sq_.ptr<float>(), exp_avg_sq_host.data(), bytes, cudaMemcpyHostToDevice),
                           "PoseRefiner exp_avg_sq restore");
            }
            if (grad_delta_.is_valid()) {
                grad_delta_.zero_();
            }
            zero_grad_pending_ = false;
            adam_step_ = header->adam_step;
            LOG_INFO("Pose refine state restored: {} cameras, Adam step {}",
                     real_states_.size(),
                     adam_step_);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Restore pose refine state failed: ") + e.what());
        }
    }

    std::expected<void, std::string> PoseRefiner::skip_serialized(std::istream& is) {
        auto header = read_pose_refiner_header(is);
        if (!header) {
            return std::unexpected(header.error());
        }

        const auto bytes_to_skip =
            static_cast<std::streamoff>(static_cast<size_t>(header->num_cameras) * sizeof(int32_t) +
                                        static_cast<size_t>(header->num_cameras) * 6 * sizeof(float) * 3);
        is.seekg(bytes_to_skip, std::ios::cur);
        if (!is) {
            return std::unexpected("Failed to skip pose refine checkpoint block");
        }
        return {};
    }

    std::array<float, 6> PoseRefiner::copy_delta_to_host(const int camera_index) const {
        std::array<float, 6> delta{};
        if (!delta_.is_valid() || camera_index < 0 ||
            camera_index >= static_cast<int>(real_states_.size())) {
            return delta;
        }
        check_cuda(cudaStreamSynchronize(pose_stream_), "PoseRefiner stream sync");
        check_cuda(cudaMemcpy(
                       delta.data(),
                       delta_.ptr<float>() + camera_index * 6,
                       sizeof(float) * delta.size(),
                       cudaMemcpyDeviceToHost),
                   "PoseRefiner delta copy");
        return delta;
    }

    std::array<float, 16> PoseRefiner::current_real_c2w(const int camera_index) const {
        return compose_c2w_so3xr3(real_states_[camera_index].base_c2w, copy_delta_to_host(camera_index));
    }

    void PoseRefiner::apply_c2w_to_camera(
        lfs::core::Camera& camera,
        const std::array<float, 16>& c2w) const {
        const auto w2c = invert_rigid(c2w);
        camera.set_world_to_camera_pose_cpu(rotation_from_w2c(w2c), translation_from_w2c(w2c));
    }

    void PoseRefiner::apply_to_camera(lfs::core::Camera& camera) {
        if (!enabled_ || real_states_.empty()) {
            return;
        }

        if (camera.is_pseudo()) {
            const auto pseudo_it = pseudo_index_by_uid_.find(camera.uid());
            if (pseudo_it == pseudo_index_by_uid_.end()) {
                return;
            }
            const auto& pseudo = pseudo_states_[pseudo_it->second];
            const auto base_current = current_real_c2w(pseudo.base_index);
            const auto pseudo_current = mat4_mul(base_current, pseudo.relative_c2w);
            apply_c2w_to_camera(camera, pseudo_current);
            return;
        }

        const auto real_it = real_index_by_uid_.find(camera.uid());
        if (real_it == real_index_by_uid_.end()) {
            return;
        }
        apply_c2w_to_camera(camera, current_real_c2w(real_it->second));
    }

    void PoseRefiner::apply_to_cameras(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        if (!enabled_ || real_states_.empty()) {
            return;
        }
        for (const auto& camera : cameras) {
            if (camera) {
                apply_to_camera(*camera);
            }
        }
    }

    std::expected<void, std::string> PoseRefiner::export_colmap_images_txt(
        const std::filesystem::path& images_txt_path,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) const {
        std::error_code ec;
        std::filesystem::create_directories(images_txt_path.parent_path(), ec);
        if (ec) {
            return std::unexpected("Failed to create COLMAP pose export directory '" +
                                   lfs::core::path_to_utf8(images_txt_path.parent_path()) + "': " + ec.message());
        }

        std::ofstream out(images_txt_path, std::ios::binary);
        if (!out) {
            return std::unexpected("Failed to open COLMAP pose export file: " +
                                   lfs::core::path_to_utf8(images_txt_path));
        }

        out << "# Refined camera poses exported by LichtFeld Studio PoseRefiner\n";
        out << "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n";
        out << "# POINTS2D[] is intentionally empty for pose-only export\n";
        out << std::setprecision(17);

        int sequential_id = 1;
        int written = 0;
        for (const auto& camera : cameras) {
            if (!camera || camera->is_pseudo()) {
                continue;
            }
            const auto real_it = real_index_by_uid_.find(camera->uid());
            if (real_it == real_index_by_uid_.end()) {
                continue;
            }

            const auto delta = copy_delta_to_host(real_it->second);
            const auto c2w = compose_c2w_so3xr3(real_states_[real_it->second].base_c2w, delta);
            const auto w2c = invert_rigid(c2w);
            const auto R = rotation_from_w2c(w2c);
            const auto T = translation_from_w2c(w2c);
            const auto q = rotmat_to_qvec(R);

            const int image_id = camera->uid() >= 0 ? camera->uid() + 1 : sequential_id;
            out << image_id << ' '
                << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << ' '
                << static_cast<double>(T[0]) << ' '
                << static_cast<double>(T[1]) << ' '
                << static_cast<double>(T[2]) << ' '
                << camera->camera_id() << ' '
                << camera->image_name() << "\n\n";

            ++sequential_id;
            ++written;
        }

        if (!out) {
            return std::unexpected("Failed to write COLMAP pose export file: " +
                                   lfs::core::path_to_utf8(images_txt_path));
        }

        LOG_INFO("Pose refine exported {} real camera poses to COLMAP images.txt '{}'",
                 written,
                 lfs::core::path_to_utf8(images_txt_path));
        return {};
    }

    std::expected<void, std::string> PoseRefiner::export_refined_poses(
        const std::filesystem::path& output_path,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const int iteration) const {
        if (!enabled_ || real_states_.empty()) {
            return {};
        }

        const auto pose_dir = output_path / "pose_refine";
        std::error_code ec;
        std::filesystem::create_directories(pose_dir, ec);
        if (ec) {
            return std::unexpected("Failed to create pose refine export directory '" +
                                   lfs::core::path_to_utf8(pose_dir) + "': " + ec.message());
        }

        nlohmann::json root;
        root["iteration"] = iteration;
        root["mode"] = params_.pose_refine_mode;
        root["real_camera_count"] = real_states_.size();
        root["pseudo_follower_count"] = pseudo_states_.size();
        root["note"] = "SO3xR3 refined poses exported from PoseRefiner; pseudo cameras follow base_camera_uid when tracked.";
        root["cameras"] = nlohmann::json::array();

        for (const auto& camera : cameras) {
            if (!camera) {
                continue;
            }

            bool tracked = false;
            std::array<float, 16> c2w{};
            std::array<float, 6> delta{};
            int base_uid = camera->pseudo_base_camera_uid();

            if (camera->is_pseudo()) {
                const auto pseudo_it = pseudo_index_by_uid_.find(camera->uid());
                if (pseudo_it != pseudo_index_by_uid_.end()) {
                    const auto& pseudo = pseudo_states_[pseudo_it->second];
                    c2w = mat4_mul(current_real_c2w(pseudo.base_index), pseudo.relative_c2w);
                    base_uid = pseudo.base_uid;
                    tracked = true;
                } else {
                    c2w = camera_c2w(*camera);
                }
            } else {
                const auto real_it = real_index_by_uid_.find(camera->uid());
                if (real_it != real_index_by_uid_.end()) {
                    delta = copy_delta_to_host(real_it->second);
                    c2w = compose_c2w_so3xr3(real_states_[real_it->second].base_c2w, delta);
                    tracked = true;
                } else {
                    c2w = camera_c2w(*camera);
                }
            }

            const auto w2c = invert_rigid(c2w);
            nlohmann::json entry;
            entry["uid"] = camera->uid();
            entry["camera_id"] = camera->camera_id();
            entry["image_name"] = camera->image_name();
            entry["is_pseudo"] = camera->is_pseudo();
            entry["tracked_by_pose_refiner"] = tracked;
            entry["base_camera_uid"] = base_uid;
            entry["supervision_weight"] = camera->supervision_weight();
            entry["delta_so3xr3"] = vec6_json(delta);
            entry["R"] = mat3_json(rotation_from_w2c(w2c));
            entry["T"] = vec3_json(translation_from_w2c(w2c));
            entry["camera_to_world"] = mat4_json(c2w);
            entry["world_to_camera"] = mat4_json(w2c);
            root["cameras"].push_back(std::move(entry));
        }

        const auto latest_path = pose_dir / "refined_cameras.json";
        if (auto result = write_json_file(latest_path, root); !result) {
            return result;
        }

        const auto snapshot_path = pose_dir / std::format("refined_cameras_iter_{:06d}.json", std::max(iteration, 0));
        if (auto result = write_json_file(snapshot_path, root); !result) {
            return result;
        }

        const auto colmap_images_path = pose_dir / "colmap" / "images.txt";
        if (auto result = export_colmap_images_txt(colmap_images_path, cameras); !result) {
            return result;
        }

        LOG_INFO("Pose refine exported {} camera poses to '{}'",
                 root["cameras"].size(),
                 lfs::core::path_to_utf8(latest_path));
        return {};
    }

} // namespace lfs::training
