/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "geometry/bounding_box.hpp"
#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"

namespace lfs::rendering {
    class RenderBoundingBox : public lfs::geometry::BoundingBox, public IBoundingBox {
    public:
        RenderBoundingBox();
        ~RenderBoundingBox() override = default;

        // Set the bounding box from min/max points
        void setBounds(const glm::vec3& min, const glm::vec3& max) override;

        // Initialize OpenGL resources - now returns Result
        Result<void> init();

        // Check if initialized
        bool isInitialized() const override { return initialized_; }

        // IBoundingBox interface implementation
        glm::vec3 getMinBounds() const override { return min_bounds_; }
        glm::vec3 getMaxBounds() const override { return max_bounds_; }
        glm::vec3 getCenter() const override { return BoundingBox::getCenter(); }
        glm::vec3 getSize() const override { return BoundingBox::getSize(); }
        glm::vec3 getLocalCenter() const override { return BoundingBox::getLocalCenter(); }

        void setworld2BBox(const lfs::geometry::EuclideanTransform& transform) override {
            BoundingBox::setworld2BBox(transform);
            box2world_mat4_ = transform.inv().toMat4();
            use_mat4_transform_ = false;
        }
        lfs::geometry::EuclideanTransform getworld2BBox() const override {
            return BoundingBox::getworld2BBox();
        }

        // Set transform using mat4 directly (preserves scale from parent nodes)
        void setWorld2BBoxMat4(const glm::mat4& world2box) {
            box2world_mat4_ = glm::inverse(world2box);
            use_mat4_transform_ = true;
        }

        // Set bounding box color
        void setColor(const glm::vec3& color) override { color_ = color; }

        // Set line width
        void setLineWidth(float width) override { line_width_ = width; }

        // Get color and line width
        glm::vec3 getColor() const override { return color_; }
        float getLineWidth() const override { return line_width_; }

        // Render the bounding box - now returns Result
        Result<void> render(const glm::mat4& view, const glm::mat4& projection);

    private:
        void createCubeGeometry();
        Result<void> setupVertexData();

        // Bounding box properties
        glm::vec3 color_;
        float line_width_;
        bool initialized_;

        // Mat4 transform for scale support (box-to-world)
        glm::mat4 box2world_mat4_{1.0f};
        bool use_mat4_transform_ = false;

        // OpenGL resources using RAII
        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        EBO ebo_;

        // Cube geometry data
        std::vector<glm::vec3> vertices_;
        std::vector<unsigned int> indices_;

        // Line indices for wireframe cube (12 edges, 24 indices)
        static const unsigned int cube_line_indices_[24];
    };
} // namespace lfs::rendering
