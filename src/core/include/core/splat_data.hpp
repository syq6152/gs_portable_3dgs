/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"

#include <expected>
#include <filesystem>
#include <glm/fwd.hpp>
#include <string>
#include <vector>

namespace lfs::geometry {
    class BoundingBox;
}

namespace lfs::core {

    namespace param {
        struct TrainingParameters;
    }

    /**
     * @brief Core data structure for Gaussian splat representation
     *
     * Contains the fundamental attributes of a Gaussian splat scene:
     * - Positions (means)
     * - Spherical harmonics coefficients (sh0, shN)
     * - Scaling factors
     * - Rotation quaternions
     * - Opacity values
     *
     * Note: Gradients are managed by AdamOptimizer, not SplatData.
     */
    class LFS_CORE_API SplatData {
    public:
        SplatData() = default;
        ~SplatData();

        // Delete copy operations
        SplatData(const SplatData&) = delete;
        SplatData& operator=(const SplatData&) = delete;

        // Custom move operations
        SplatData(SplatData&& other) noexcept;
        SplatData& operator=(SplatData&& other) noexcept;

        // Constructor
        SplatData(int sh_degree,
                  Tensor means,
                  Tensor sh0,
                  Tensor shN,
                  Tensor scaling,
                  Tensor rotation,
                  Tensor opacity,
                  float scene_scale);

        // ========== Computed getters ==========
        Tensor get_means() const;
        Tensor get_opacity() const;  // Returns sigmoid(opacity_raw)
        Tensor get_rotation() const; // Returns normalized quaternions
        Tensor get_scaling() const;  // Returns exp(scaling_raw)
        Tensor get_shs() const;      // Returns concatenated sh0 + shN

        // ========== Simple inline getters ==========
        int get_active_sh_degree() const { return _active_sh_degree; }
        int get_max_sh_degree() const { return _max_sh_degree; }
        float get_scene_scale() const { return _scene_scale; }
        float get_mesh2splat_mean_max_scale() const { return _mesh2splat_mean_max_scale; }
        void set_mesh2splat_mean_max_scale(float value) { _mesh2splat_mean_max_scale = value; }
        unsigned long size() const { return static_cast<unsigned long>(_means.shape()[0]); }

        // ========== Raw tensor access (for optimization) ==========
        inline Tensor& means() { return _means; }
        inline const Tensor& means() const { return _means; }
        inline Tensor& means_raw() { return _means; }
        inline const Tensor& means_raw() const { return _means; }
        inline Tensor& opacity_raw() { return _opacity; }
        inline const Tensor& opacity_raw() const { return _opacity; }
        inline Tensor& rotation_raw() { return _rotation; }
        inline const Tensor& rotation_raw() const { return _rotation; }
        inline Tensor& scaling_raw() { return _scaling; }
        inline const Tensor& scaling_raw() const { return _scaling; }
        inline Tensor& sh0() { return _sh0; }
        inline const Tensor& sh0() const { return _sh0; }
        inline Tensor& sh0_raw() { return _sh0; }
        inline const Tensor& sh0_raw() const { return _sh0; }
        inline Tensor& shN() { return _shN; }
        inline const Tensor& shN() const { return _shN; }
        inline Tensor& shN_raw() { return _shN; }
        inline const Tensor& shN_raw() const { return _shN; }

        // ========== Mesh-init marking (tracks mesh2splat-initialized Gaussians) ==========
        Tensor& mesh_init_mask() { return _mesh_init_mask; }
        [[nodiscard]] const Tensor& mesh_init_mask() const { return _mesh_init_mask; }
        [[nodiscard]] bool has_mesh_init_mask() const { return _mesh_init_mask.is_valid(); }

        // Immutable row provenance for Gaussians sampled from a temporary mesh hole-fill
        // patch. Unlike birth_tri, this is copied through densification and never inferred
        // from the mutable mesh-surface face cache.
        Tensor& mesh_hole_fill_mask() { return _mesh_hole_fill_mask; }
        [[nodiscard]] const Tensor& mesh_hole_fill_mask() const { return _mesh_hole_fill_mask; }
        [[nodiscard]] bool has_mesh_hole_fill_mask() const { return _mesh_hole_fill_mask.is_valid(); }

        // ========== Mesh-surface constraint (surface-walk constrained means) ==========
        // Set together by mesh2splat. _birth_tri[i] is the global face index (into
        // _constraint_mesh_indices) the i-th Gaussian was rasterized from. MCMC copies it
        // into a mutable current-face cache and updates that cache while walking.
        Tensor& constraint_mesh_verts() { return _constraint_mesh_verts; }
        [[nodiscard]] const Tensor& constraint_mesh_verts() const { return _constraint_mesh_verts; }
        Tensor& constraint_mesh_indices() { return _constraint_mesh_indices; }
        [[nodiscard]] const Tensor& constraint_mesh_indices() const { return _constraint_mesh_indices; }
        Tensor& birth_tri() { return _birth_tri; }
        [[nodiscard]] const Tensor& birth_tri() const { return _birth_tri; }
        [[nodiscard]] bool has_constraint_mesh() const {
            return _constraint_mesh_verts.is_valid() &&
                   _constraint_mesh_indices.is_valid() &&
                   _birth_tri.is_valid();
        }

        // Per-triangle edge adjacency (set by mesh2splat alongside the constraint mesh).
        // [F,3] Int32 - tri_edge_neighbors[f, e] is the face id sharing edge e of face f,
        // or -1 for a boundary edge. Mesh-static; used directly by MCMC surface-walk.
        Tensor& tri_edge_neighbors() { return _tri_edge_neighbors; }
        [[nodiscard]] const Tensor& tri_edge_neighbors() const { return _tri_edge_neighbors; }
        [[nodiscard]] bool has_tri_edge_neighbors() const { return _tri_edge_neighbors.is_valid(); }

        // ========== Soft deletion (for undo/redo crop support) ==========
        Tensor& deleted() { return _deleted; }
        [[nodiscard]] const Tensor& deleted() const { return _deleted; }
        [[nodiscard]] bool has_deleted_mask() const { return _deleted.is_valid(); }
        [[nodiscard]] unsigned long visible_count() const;

        // Mark gaussians as deleted, returns previous state for undo
        Tensor soft_delete(const Tensor& mask);
        void undelete(const Tensor& mask);
        void clear_deleted();

        // Permanently remove deleted gaussians (compacts data)
        // Returns number of gaussians removed
        size_t apply_deleted();

        // ========== Capacity management ==========
        // Reserve capacity for parameter tensors (for MCMC densification)
        void reserve_capacity(size_t capacity);

        // ========== SH degree management ==========
        void increment_sh_degree();
        void set_active_sh_degree(int sh_degree);
        void set_max_sh_degree(int sh_degree) { _max_sh_degree = sh_degree; }

        // ========== Serialization ==========
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

    public:
        // Holds the magnitude of the screen space gradient (used for densification)
        Tensor _densification_info;

    private:
        int _active_sh_degree = 0;
        int _max_sh_degree = 0;
        float _scene_scale = 0.f;
        float _mesh2splat_mean_max_scale = 0.f;

        // Parameters
        Tensor _means;
        Tensor _sh0;
        Tensor _shN;
        Tensor _scaling;
        Tensor _rotation;
        Tensor _opacity;

        // Mesh-init mask: bool tensor [N], true = initialized from mesh.
        Tensor _mesh_init_mask;

        // Hole-fill provenance: bool tensor [N], true = sampled from a temporary
        // render-only hole-fill face. This is training provenance, not an export field.
        Tensor _mesh_hole_fill_mask;

        // Mesh-surface constraint data (set by mesh2splat path; CUDA tensors).
        //   _constraint_mesh_verts:   [V,3] Float32 — full mesh vertex positions
        //   _constraint_mesh_indices: [F,3] Int32   — full mesh triangle indices
        //   _birth_tri:               [N]   Int32   — per-Gaussian initial face id;
        //                                              -1 for non-mesh-init Gaussians.
        Tensor _constraint_mesh_verts;
        Tensor _constraint_mesh_indices;
        Tensor _birth_tri;

        // Face-level edge adjacency for surface walking.
        Tensor _tri_edge_neighbors; // [F,3] Int32 CUDA (-1 for boundary edges)

        // Soft deletion mask: bool tensor [N], true = hidden from rendering
        Tensor _deleted;

        // Allow free functions in splat_data_transform.cpp to access private members
        friend LFS_CORE_API SplatData& transform(SplatData&, const glm::mat4&);
        friend LFS_CORE_API SplatData crop_by_cropbox(const SplatData&, const lfs::geometry::BoundingBox&, bool);
        friend LFS_CORE_API SplatData extract_by_mask(const SplatData&, const Tensor&);
        friend LFS_CORE_API std::expected<SplatData, std::string> concat(const SplatData&, const SplatData&);
        friend LFS_CORE_API void random_choose(SplatData&, int, int);
    };

    // ========== Free function: Factory ==========

    /**
     * @brief Create SplatData from a PointCloud
     * @param params Training parameters (SH degree, init settings)
     * @param scene_center Center of the scene
     * @param point_cloud Source point cloud
     * @param capacity If > 0, pre-allocate for this many gaussians (bypasses memory pool)
     * @return SplatData on success, error string on failure
     */
    LFS_CORE_API std::expected<SplatData, std::string> init_model_from_pointcloud(
        const param::TrainingParameters& params,
        Tensor scene_center,
        const PointCloud& point_cloud,
        int capacity = 0);

} // namespace lfs::core
