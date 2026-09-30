/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

namespace lfs::training {

    /// MCMC-based optimization strategy. SplatData owned by Scene.
    class MCMC : public IStrategy {
    public:
        MCMC() = delete;
        /// SplatData must be owned by Scene
        explicit MCMC(lfs::core::SplatData& splat_data);

        // Reference member prevents copy/move
        MCMC(const MCMC&) = delete;
        MCMC& operator=(const MCMC&) = delete;
        MCMC(MCMC&&) = delete;
        MCMC& operator=(MCMC&&) = delete;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        // IStrategy interface - optimizer access
        AdamOptimizer& get_optimizer() override { return *_optimizer; }
        const AdamOptimizer& get_optimizer() const override { return *_optimizer; }
        ExponentialLR* get_scheduler() { return _scheduler.get(); }
        const ExponentialLR* get_scheduler() const { return _scheduler.get(); }
        MeshSurfaceState mesh_surface_state() override;

        // Serialization for checkpoints
        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "mcmc"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override {
            _params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);
        }

        // Exposed for testing
        int add_new_gs_test() { return add_new_gs(); }
        int add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs);
        int relocate_gs_test() { return relocate_gs(); }

    private:
        friend struct MCMCProvenanceTestAccess;

        // Helper functions
        lfs::core::Tensor multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement = true);
        int relocate_gs();
        int add_new_gs();
        void inject_noise();
        void update_optimizer_for_relocate(const lfs::core::Tensor& sampled_indices,
                                           const lfs::core::Tensor& dead_indices,
                                           ParamType param_type);
        lfs::core::Tensor get_sampling_weights() const;
        void ensure_densification_info_shape();
        void ensure_ratio_workspace_size(size_t required);

        // Optionally zero mesh-init opacity gradients. The older mean/rotation/scale
        // freeze calls are kept commented in the implementation as a reference path.
        void zero_mesh_init_gradients();

        // Mesh-surface constraint: snapshot before optimizer/noise updates, then surface-walk
        // constrained means back onto the mesh.
        void snapshot_mesh_surface_means();
        void project_mesh_init_means_to_surface();
        void ensure_mesh_face_normals();

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr; // Scene-owned
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        static constexpr float NOISE_LR = 5e5f;

        // State variables
        int _n_max = 0;                  // max relocation ratio
        lfs::core::Tensor _noise_buffer; // Reusable buffer for noise injection
        lfs::core::Tensor _ones_int32;   // Cached ones for ratio counting; grows with the live model.
        lfs::core::Tensor _error_score_max;
        int _error_score_windows = 0;

        // Mesh-init marker - bool tensor [max_cap], true = initialized from mesh.
        lfs::core::Tensor _mesh_init_mask;

        // Immutable hole-fill provenance - bool tensor [max_cap]. Relocated and
        // sampled rows inherit their source; unused capacity is false.
        lfs::core::Tensor _mesh_hole_fill_mask;

        // Mesh-surface constraint state (mirrors fields on SplatData; loaded in initialize()).
        //   _constraint_mesh_verts:   [V,3] Float32 CUDA - reference mesh vertex positions.
        //   _constraint_mesh_indices: [F,3] Int32   CUDA - reference mesh triangle indices.
        //   _birth_tri:               [max_cap] Int32 CUDA - per-Gaussian current face cache;
        //                                                    -1 for non-mesh-init slots.
        lfs::core::Tensor _constraint_mesh_verts;
        lfs::core::Tensor _constraint_mesh_indices;
        lfs::core::Tensor _birth_tri;
        lfs::core::Tensor _tri_edge_neighbors;
        lfs::core::Tensor _constraint_face_normals;
        lfs::core::Tensor _surface_prev_means;

        float _mesh_scene_radius = 1.0f;
        int32_t _surface_walk_steps = 8;
    };

} // namespace lfs::training
