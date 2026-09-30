/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    class RenderInfiniteGrid {
    public:
        enum class GridPlane {
            YZ = 0,
            XZ = 1,
            XY = 2
        };

        RenderInfiniteGrid() = default;
        ~RenderInfiniteGrid() = default;

        Result<void> init();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        Result<void> render(const glm::mat4& view, const glm::mat4& projection, bool orthographic = false);

        void setOpacity(float opacity) { opacity_ = glm::clamp(opacity, 0.0f, 1.0f); }
        void setPlane(GridPlane plane) { plane_ = plane; }
        [[nodiscard]] GridPlane getPlane() const { return plane_; }

    private:
        void computeFrustumPerspective(const glm::mat4& view_inv, float fov_y, float aspect,
                                       glm::vec3& near_origin, glm::vec3& far_origin,
                                       glm::vec3& far_x, glm::vec3& far_y) const;

        void computeFrustumOrthographic(const glm::mat4& view_inv, float half_width, float half_height,
                                        glm::vec3& near_origin, glm::vec3& near_x, glm::vec3& near_y,
                                        glm::vec3& far_origin, glm::vec3& far_x, glm::vec3& far_y) const;

        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;

        GridPlane plane_ = GridPlane::XZ;
        float opacity_ = 1.0f;
        bool initialized_ = false;
    };

} // namespace lfs::rendering
