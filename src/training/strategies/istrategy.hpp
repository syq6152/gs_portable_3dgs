/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <istream>
#include <memory>
#include <ostream>

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::training {

    class CameraDataset;

    struct MeshSurfaceState {
        lfs::core::Tensor* constraint_mesh_verts = nullptr;
        lfs::core::Tensor* constraint_mesh_indices = nullptr;
        lfs::core::Tensor* tri_edge_neighbors = nullptr;
        lfs::core::Tensor* constraint_face_normals = nullptr;
        lfs::core::Tensor* current_faces = nullptr;
        float scene_radius = 1.0f;
        int walk_steps = 1;

        [[nodiscard]] bool is_valid() const {
            return constraint_mesh_verts != nullptr &&
                   constraint_mesh_indices != nullptr &&
                   tri_edge_neighbors != nullptr &&
                   current_faces != nullptr &&
                   constraint_mesh_verts->is_valid() &&
                   constraint_mesh_indices->is_valid() &&
                   tri_edge_neighbors->is_valid() &&
                   current_faces->is_valid();
        }
    };

    /**
     * @brief Strategy interface for Gaussian splatting optimization.
     *
     * Strategies operate on a SplatData reference owned by the Scene.
     * This allows the same model to be used for both training and visualization.
     */
    class IStrategy {
    public:
        virtual ~IStrategy() = default;

        virtual void initialize(const lfs::core::param::OptimizationParameters& optimParams) = 0;

        virtual void pre_step(int /*iter*/, RenderOutput& /*render_output*/) {}

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        virtual bool is_refining(int iter) const = 0;

        // Get the underlying Gaussian model (reference to Scene-owned data)
        virtual lfs::core::SplatData& get_model() = 0;
        virtual const lfs::core::SplatData& get_model() const = 0;

        // Get the optimizer (for gradient access during backward pass)
        virtual class AdamOptimizer& get_optimizer() = 0;
        virtual const class AdamOptimizer& get_optimizer() const = 0;

        // Remove Gaussians based on mask
        virtual void remove_gaussians(const lfs::core::Tensor& mask) = 0;

        // Serialization for checkpoints
        virtual void serialize(std::ostream& os) const = 0;
        virtual void deserialize(std::istream& is) = 0;

        // Strategy type identifier for checkpoint compatibility
        virtual const char* strategy_type() const = 0;

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        virtual void reserve_optimizer_capacity(size_t capacity) = 0;

        // Update the strategy's cached optimization parameters after checkpoint params are resolved.
        virtual void set_optimization_params(const lfs::core::param::OptimizationParameters&) {}

        // Optional hook for strategies that need the training dataset (e.g., for view-based scoring)
        virtual void set_training_dataset(std::shared_ptr<CameraDataset>) {}

        virtual void set_image_loader(lfs::io::PipelinedImageLoader*) {}

        virtual MeshSurfaceState mesh_surface_state() { return {}; }
    };
} // namespace lfs::training
