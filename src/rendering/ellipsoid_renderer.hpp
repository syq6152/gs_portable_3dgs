/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    class EllipsoidRenderer {
    public:
        EllipsoidRenderer();
        ~EllipsoidRenderer() = default;

        Result<void> init();
        bool isInitialized() const { return initialized_; }

        void setRadii(const glm::vec3& radii) { radii_ = radii; }
        void setTransform(const glm::mat4& transform) { transform_ = transform; }
        void setColor(const glm::vec3& color) { color_ = color; }
        void setLineWidth(float width) { line_width_ = width; }

        glm::vec3 getRadii() const { return radii_; }
        glm::vec3 getColor() const { return color_; }
        float getLineWidth() const { return line_width_; }

        Result<void> render(const glm::mat4& view, const glm::mat4& projection);

    private:
        void createSphereGeometry();
        Result<void> setupVertexData();

        static constexpr int LAT_SEGMENTS = 24;
        static constexpr int LON_SEGMENTS = 32;

        glm::vec3 radii_{1.0f, 1.0f, 1.0f};
        glm::mat4 transform_{1.0f};
        glm::vec3 color_{0.3f, 0.8f, 1.0f};
        float line_width_ = 2.0f;
        bool initialized_ = false;

        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        EBO ebo_;

        std::vector<glm::vec3> vertices_;
        std::vector<unsigned int> indices_;
    };

} // namespace lfs::rendering
