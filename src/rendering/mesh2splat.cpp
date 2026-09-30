/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/mesh2splat.hpp"
#include "rendering/mesh2splat_internal.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"

// clang-format off
#include <glad/glad.h>
// clang-format on

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace lfs::rendering {

    using core::DataType;
    using core::Device;
    using core::Mesh2SplatOptions;
    using core::Mesh2SplatProgressCallback;
    using core::MeshData;
    using core::SplatData;
    using core::Submesh;
    using core::Tensor;
    using core::TextureImage;

    namespace {

        constexpr float SH_C0 = 0.28209479177387814f;

        // Mirrors the GaussianVertex struct in converterFS.glsl SSBO layout
        struct GaussianVertex {
            glm::vec4 position;
            glm::vec4 color;
            glm::vec4 scale;
            glm::vec4 normal;
            glm::vec4 rotation;
            glm::vec4 pbr;
            // extra.x: global birth-triangle id packed via intBitsToFloat (see converterFS.glsl).
            glm::vec4 extra;
        };
        static_assert(sizeof(GaussianVertex) == 7 * sizeof(glm::vec4));

        // VBO layout matching the conversion vertex shader (17 floats per vertex)
        struct PerVertexData {
            glm::vec3 position;      // location 0
            glm::vec3 normal;        // location 1
            glm::vec4 tangent;       // location 2
            glm::vec2 uv;            // location 3
            glm::vec2 normalized_uv; // location 4
            glm::vec3 scale;         // location 5
            glm::vec4 color;         // location 6
        };
        static_assert(sizeof(PerVertexData) == 21 * sizeof(float));

        struct SubmeshDrawPlan {
            size_t start_index = 0;
            size_t index_count = 0;
            glm::vec3 bbox_min{std::numeric_limits<float>::max()};
            glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
            size_t material_index = 0;
            // Global face-index offset (into mesh.indices) of this submesh's first triangle.
            // Used to set u_triangleOffset so gl_PrimitiveIDIn maps to a global triangle id
            // for the mesh-surface-constraint birth_tri tracking.
            int32_t face_offset = 0;
        };

        struct GlSubmeshBuffer {
            GLuint vao = 0;
            GLuint vbo = 0;
            GLsizei vertex_count = 0;
            size_t material_index = 0;
            int32_t face_offset = 0;

            GlSubmeshBuffer() = default;
            GlSubmeshBuffer(const GlSubmeshBuffer&) = delete;
            GlSubmeshBuffer& operator=(const GlSubmeshBuffer&) = delete;

            GlSubmeshBuffer(GlSubmeshBuffer&& other) noexcept
                : vao(other.vao),
                  vbo(other.vbo),
                  vertex_count(other.vertex_count),
                  material_index(other.material_index),
                  face_offset(other.face_offset) {
                other.vao = 0;
                other.vbo = 0;
                other.vertex_count = 0;
            }

            GlSubmeshBuffer& operator=(GlSubmeshBuffer&& other) noexcept {
                if (this != &other) {
                    destroy();
                    vao = other.vao;
                    vbo = other.vbo;
                    vertex_count = other.vertex_count;
                    material_index = other.material_index;
                    face_offset = other.face_offset;
                    other.vao = 0;
                    other.vbo = 0;
                    other.vertex_count = 0;
                }
                return *this;
            }

            ~GlSubmeshBuffer() {
                destroy();
            }

            void destroy() {
                if (vbo) {
                    glDeleteBuffers(1, &vbo);
                    vbo = 0;
                }
                if (vao) {
                    glDeleteVertexArrays(1, &vao);
                    vao = 0;
                }
                vertex_count = 0;
            }
        };

        struct PreparedGeometry {
            std::vector<GlSubmeshBuffer> buffers;
            glm::vec3 bbox_min{std::numeric_limits<float>::max()};
            glm::vec3 bbox_max{std::numeric_limits<float>::lowest()};
            size_t total_vertices = 0;

            void release_buffers() {
                for (auto& buffer : buffers) {
                    buffer.destroy();
                }
                buffers.clear();
                buffers.shrink_to_fit();
            }
        };

        // RAII wrapper for a set of GL objects that need cleanup
        struct GlCleanup {
            GLuint program = 0;
            GLuint fbo = 0;
            GLuint rbo = 0;
            GLuint ssbo = 0;
            GLuint atomic_counter = 0;
            std::vector<GLuint> textures;

            ~GlCleanup() {
                for (auto tex : textures) {
                    if (tex)
                        glDeleteTextures(1, &tex);
                }
                if (atomic_counter)
                    glDeleteBuffers(1, &atomic_counter);
                if (ssbo)
                    glDeleteBuffers(1, &ssbo);
                if (rbo)
                    glDeleteRenderbuffers(1, &rbo);
                if (fbo)
                    glDeleteFramebuffers(1, &fbo);
                if (program)
                    glDeleteProgram(program);
            }

            GlCleanup() = default;
            GlCleanup(const GlCleanup&) = delete;
            GlCleanup& operator=(const GlCleanup&) = delete;
        };

        // Minimal GL state save/restore (cannot use lfs::rendering::GLStateGuard from lfs_core)
        struct GlStateSave {
            GLint viewport[4];
            GLint prev_program;
            GLint prev_fbo;
            GLint prev_vao;
            GLint prev_active_texture;
            GLboolean depth_test;
            GLboolean blend;
            GLboolean cull_face;

            GlStateSave() {
                glGetIntegerv(GL_VIEWPORT, viewport);
                glGetIntegerv(GL_CURRENT_PROGRAM, &prev_program);
                glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev_fbo);
                glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
                glGetIntegerv(GL_ACTIVE_TEXTURE, &prev_active_texture);
                depth_test = glIsEnabled(GL_DEPTH_TEST);
                blend = glIsEnabled(GL_BLEND);
                cull_face = glIsEnabled(GL_CULL_FACE);
            }

            ~GlStateSave() {
                glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                glUseProgram(prev_program);
                glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
                glBindVertexArray(prev_vao);
                glActiveTexture(prev_active_texture);
                if (depth_test)
                    glEnable(GL_DEPTH_TEST);
                else
                    glDisable(GL_DEPTH_TEST);
                if (blend)
                    glEnable(GL_BLEND);
                else
                    glDisable(GL_BLEND);
                if (cull_face)
                    glEnable(GL_CULL_FACE);
                else
                    glDisable(GL_CULL_FACE);
            }

            GlStateSave(const GlStateSave&) = delete;
            GlStateSave& operator=(const GlStateSave&) = delete;
        };

        // ========================================================================
        // Geometry extraction
        // ========================================================================

        struct MeshCpuData {
            Tensor vertices;
            Tensor indices;
            Tensor normals;
            Tensor tangents;
            Tensor texcoords;
            Tensor colors;

            const float* verts_ptr = nullptr;
            const int32_t* idx_ptr = nullptr;
            const float* normals_ptr = nullptr;
            const float* tangents_ptr = nullptr;
            const float* texcoords_ptr = nullptr;
            const float* colors_ptr = nullptr;
            int64_t vertex_count = 0;
            int64_t face_count = 0;
            std::vector<Submesh> submeshes;
        };

        glm::vec3 compute_face_normal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 n = glm::cross(edge1, edge2);
            float len = glm::length(n);
            return len > 1e-8f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
        }

        glm::vec4 compute_face_tangent(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                       const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2) {
            glm::vec3 dv1 = v1 - v0;
            glm::vec3 dv2 = v2 - v0;
            glm::vec2 duv1 = uv1 - uv0;
            glm::vec2 duv2 = uv2 - uv0;

            float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (std::abs(det) < 1e-8f) {
                return glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }

            float r = 1.0f / det;
            glm::vec3 t = glm::normalize((dv1 * duv2.y - dv2 * duv1.y) * r);
            return glm::vec4(t, 1.0f);
        }

        MeshCpuData make_mesh_cpu_data(const MeshData& mesh) {
            assert(mesh.vertices.is_valid());
            assert(mesh.indices.is_valid());
            assert(mesh.vertices.dtype() == DataType::Float32);
            assert(mesh.indices.dtype() == DataType::Int32);

            MeshCpuData data;
            data.vertices = mesh.vertices.device() == Device::CPU ? mesh.vertices : mesh.vertices.to(Device::CPU);
            data.indices = mesh.indices.device() == Device::CPU ? mesh.indices : mesh.indices.to(Device::CPU);
            data.verts_ptr = data.vertices.ptr<float>();
            data.idx_ptr = data.indices.ptr<int32_t>();
            data.vertex_count = static_cast<int64_t>(data.vertices.shape()[0]);
            data.face_count = static_cast<int64_t>(data.indices.shape()[0]);

            if (mesh.has_normals()) {
                data.normals = mesh.normals.device() == Device::CPU ? mesh.normals : mesh.normals.to(Device::CPU);
                data.normals_ptr = data.normals.ptr<float>();
                assert(data.normals.shape()[0] == data.vertices.shape()[0]);
            }

            if (mesh.has_tangents()) {
                data.tangents = mesh.tangents.device() == Device::CPU ? mesh.tangents : mesh.tangents.to(Device::CPU);
                data.tangents_ptr = data.tangents.ptr<float>();
                assert(data.tangents.shape()[0] == data.vertices.shape()[0]);
            }

            if (mesh.has_texcoords()) {
                data.texcoords = mesh.texcoords.device() == Device::CPU ? mesh.texcoords : mesh.texcoords.to(Device::CPU);
                data.texcoords_ptr = data.texcoords.ptr<float>();
                assert(data.texcoords.shape()[0] == data.vertices.shape()[0]);
            }

            if (mesh.has_colors()) {
                data.colors = mesh.colors.device() == Device::CPU ? mesh.colors : mesh.colors.to(Device::CPU);
                data.colors_ptr = data.colors.ptr<float>();
                assert(data.colors.shape()[0] == data.vertices.shape()[0]);
            }

            if (mesh.submeshes.empty()) {
                data.submeshes.push_back({0, static_cast<size_t>(data.face_count) * 3, 0});
            } else {
                data.submeshes = mesh.submeshes;
            }

            return data;
        }

        std::vector<SubmeshDrawPlan> build_submesh_draw_plans(
            const MeshCpuData& cpu,
            glm::vec3& global_min,
            glm::vec3& global_max,
            size_t& total_vertices) {

            std::vector<SubmeshDrawPlan> plans;
            plans.reserve(cpu.submeshes.size());
            global_min = glm::vec3(std::numeric_limits<float>::max());
            global_max = glm::vec3(std::numeric_limits<float>::lowest());
            total_vertices = 0;

            for (const auto& sub : cpu.submeshes) {
                assert(sub.index_count % 3 == 0);
                const size_t face_count = sub.index_count / 3;
                if (face_count == 0)
                    continue;

                SubmeshDrawPlan plan;
                plan.start_index = sub.start_index;
                plan.index_count = sub.index_count;
                plan.material_index = sub.material_index;
                plan.face_offset = static_cast<int32_t>(sub.start_index / 3);

                for (size_t f = 0; f < face_count; f++) {
                    const size_t base = sub.start_index + f * 3;
                    const int32_t indices[3] = {
                        cpu.idx_ptr[base + 0],
                        cpu.idx_ptr[base + 1],
                        cpu.idx_ptr[base + 2]};

                    for (int k = 0; k < 3; ++k) {
                        const int32_t vi = indices[k];
                        assert(vi >= 0 && vi < cpu.vertex_count);
                        const glm::vec3 pos{
                            cpu.verts_ptr[vi * 3],
                            cpu.verts_ptr[vi * 3 + 1],
                            cpu.verts_ptr[vi * 3 + 2]};
                        plan.bbox_min = glm::min(plan.bbox_min, pos);
                        plan.bbox_max = glm::max(plan.bbox_max, pos);
                    }
                }

                global_min = glm::min(global_min, plan.bbox_min);
                global_max = glm::max(global_max, plan.bbox_max);
                total_vertices += plan.index_count;
                plans.push_back(plan);
            }

            return plans;
        }

        void fill_submesh_vertices(
            const MeshCpuData& cpu,
            const SubmeshDrawPlan& plan,
            std::vector<PerVertexData>& scratch) {

            scratch.resize(plan.index_count);
            size_t out = 0;
            const size_t face_count = plan.index_count / 3;

            for (size_t f = 0; f < face_count; f++) {
                const size_t base = plan.start_index + f * 3;
                const int32_t indices[3] = {
                    cpu.idx_ptr[base + 0],
                    cpu.idx_ptr[base + 1],
                    cpu.idx_ptr[base + 2]};

                glm::vec3 pos[3];
                glm::vec3 nrm[3];
                glm::vec4 tan[3];
                glm::vec2 uv[3];
                glm::vec4 col[3] = {glm::vec4(1.0f), glm::vec4(1.0f), glm::vec4(1.0f)};

                for (int k = 0; k < 3; k++) {
                    int32_t vi = indices[k];
                    assert(vi >= 0 && vi < cpu.vertex_count);
                    pos[k] = {cpu.verts_ptr[vi * 3], cpu.verts_ptr[vi * 3 + 1], cpu.verts_ptr[vi * 3 + 2]};

                    if (cpu.texcoords_ptr) {
                        uv[k] = {cpu.texcoords_ptr[vi * 2], cpu.texcoords_ptr[vi * 2 + 1]};
                    } else {
                        uv[k] = {0.0f, 0.0f};
                    }

                    if (cpu.normals_ptr) {
                        nrm[k] = {cpu.normals_ptr[vi * 3], cpu.normals_ptr[vi * 3 + 1], cpu.normals_ptr[vi * 3 + 2]};
                    }

                    if (cpu.tangents_ptr) {
                        tan[k] = {cpu.tangents_ptr[vi * 4], cpu.tangents_ptr[vi * 4 + 1],
                                  cpu.tangents_ptr[vi * 4 + 2], cpu.tangents_ptr[vi * 4 + 3]};
                    }

                    if (cpu.colors_ptr) {
                        col[k] = {cpu.colors_ptr[vi * 4], cpu.colors_ptr[vi * 4 + 1],
                                  cpu.colors_ptr[vi * 4 + 2], cpu.colors_ptr[vi * 4 + 3]};
                    }
                }

                if (!cpu.normals_ptr) {
                    glm::vec3 fn = compute_face_normal(pos[0], pos[1], pos[2]);
                    nrm[0] = nrm[1] = nrm[2] = fn;
                }

                if (!cpu.tangents_ptr) {
                    glm::vec4 ft = compute_face_tangent(pos[0], pos[1], pos[2], uv[0], uv[1], uv[2]);
                    tan[0] = tan[1] = tan[2] = ft;
                }

                for (int k = 0; k < 3; k++) {
                    PerVertexData vtx;
                    vtx.position = pos[k];
                    vtx.normal = nrm[k];
                    vtx.tangent = tan[k];
                    vtx.uv = uv[k];
                    vtx.normalized_uv = {0.0f, 0.0f}; // GS ignores this; uses triplanar projection
                    vtx.scale = {0.0f, 0.0f, 0.0f};   // unused by GS
                    vtx.color = col[k];
                    scratch[out++] = vtx;
                }
            }

            assert(out == scratch.size());
        }

        void configure_submesh_vertex_layout() {
            constexpr GLsizei stride = sizeof(PerVertexData);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, position)));
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, normal)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, tangent)));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, uv)));
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, normalized_uv)));
            glEnableVertexAttribArray(4);
            glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, scale)));
            glEnableVertexAttribArray(5);
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride,
                                  reinterpret_cast<void*>(offsetof(PerVertexData, color)));
            glEnableVertexAttribArray(6);
        }

        GlSubmeshBuffer upload_submesh_buffer(
            const SubmeshDrawPlan& plan,
            const std::vector<PerVertexData>& vertices) {

            GlSubmeshBuffer buffer;
            buffer.vertex_count = static_cast<GLsizei>(vertices.size());
            buffer.material_index = plan.material_index;
            buffer.face_offset = plan.face_offset;

            glGenVertexArrays(1, &buffer.vao);
            glGenBuffers(1, &buffer.vbo);

            glBindVertexArray(buffer.vao);
            glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
            glBufferData(GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(vertices.size() * sizeof(PerVertexData)),
                         vertices.data(), GL_STATIC_DRAW);
            configure_submesh_vertex_layout();
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            return buffer;
        }

        PreparedGeometry prepare_geometry_buffers(const MeshData& mesh) {
            auto cpu = make_mesh_cpu_data(mesh);
            PreparedGeometry prepared;

            auto plans = build_submesh_draw_plans(
                cpu,
                prepared.bbox_min,
                prepared.bbox_max,
                prepared.total_vertices);

            prepared.buffers.reserve(plans.size());
            std::vector<PerVertexData> scratch;

            for (const auto& plan : plans) {
                fill_submesh_vertices(cpu, plan, scratch);
                if (!scratch.empty()) {
                    prepared.buffers.push_back(upload_submesh_buffer(plan, scratch));
                }
                scratch.clear();
            }

            return prepared;
        }

        // ========================================================================
        // Shader compilation
        // ========================================================================

        std::string read_file_contents(const std::filesystem::path& path) {
            std::ifstream file(path);
            if (!file.is_open()) {
                return {};
            }
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }

        GLuint compile_shader_stage(const std::string& source, GLenum type, const char* label) {
            GLuint shader = glCreateShader(type);
            assert(shader != 0);

            const char* src = source.c_str();
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);

            GLint status;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
            if (status != GL_TRUE) {
                char log[2048];
                GLsizei len;
                glGetShaderInfoLog(shader, sizeof(log), &len, log);
                LOG_ERROR("mesh2splat {} shader compile error: {}", label, log);
                glDeleteShader(shader);
                return 0;
            }

            return shader;
        }

        GLuint create_conversion_program() {
            auto shader_dir = core::getShadersDir() / "mesh2splat";

            auto vs_source = read_file_contents(shader_dir / "converterVS.glsl");
            auto gs_source = read_file_contents(shader_dir / "converterGS.glsl");
            auto fs_source = read_file_contents(shader_dir / "converterFS.glsl");

            if (vs_source.empty() || gs_source.empty() || fs_source.empty()) {
                LOG_ERROR("mesh2splat: failed to read shader files from {}", shader_dir.string());
                return 0;
            }

            GLuint vs = compile_shader_stage(vs_source, GL_VERTEX_SHADER, "vertex");
            if (!vs)
                return 0;

            GLuint gs = compile_shader_stage(gs_source, GL_GEOMETRY_SHADER, "geometry");
            if (!gs) {
                glDeleteShader(vs);
                return 0;
            }

            GLuint fs = compile_shader_stage(fs_source, GL_FRAGMENT_SHADER, "fragment");
            if (!fs) {
                glDeleteShader(vs);
                glDeleteShader(gs);
                return 0;
            }

            GLuint program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, gs);
            glAttachShader(program, fs);
            glLinkProgram(program);

            // Shaders can be deleted after linking
            glDeleteShader(vs);
            glDeleteShader(gs);
            glDeleteShader(fs);

            GLint status;
            glGetProgramiv(program, GL_LINK_STATUS, &status);
            if (status != GL_TRUE) {
                char log[2048];
                GLsizei len;
                glGetProgramInfoLog(program, sizeof(log), &len, log);
                LOG_ERROR("mesh2splat program link error: {}", log);
                glDeleteProgram(program);
                return 0;
            }

            return program;
        }

        // ========================================================================
        // Texture upload
        // ========================================================================

        GLuint upload_texture(const TextureImage& img, bool is_srgb = false) {
            assert(img.width > 0 && img.height > 0);
            assert(!img.pixels.empty());

            GLenum format;
            GLint internal_format;
            switch (img.channels) {
            case 1:
                format = GL_RED;
                internal_format = GL_R8;
                break;
            case 2:
                format = GL_RG;
                internal_format = GL_RG8;
                break;
            case 3:
                format = GL_RGB;
                internal_format = is_srgb ? GL_SRGB8 : GL_RGB8;
                break;
            case 4:
                format = GL_RGBA;
                internal_format = is_srgb ?  GL_SRGB8_ALPHA8 : GL_RGBA8;
                break;
            default: return 0;
            }

            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format,
                         img.width, img.height, 0, format, GL_UNSIGNED_BYTE, img.pixels.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            return tex;
        }

        // ========================================================================
        // FBO creation
        // ========================================================================

        bool create_fbo(int width, int height, GLuint& out_fbo, GLuint& out_rbo) {
            glGenFramebuffers(1, &out_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, out_fbo);

            glGenRenderbuffers(1, &out_rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, out_rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, out_rbo);

            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            return status == GL_FRAMEBUFFER_COMPLETE;
        }

        // ========================================================================
        // SSBO readback → SplatData conversion
        // ========================================================================

        [[maybe_unused]] std::unique_ptr<SplatData> build_splat_data(const GaussianVertex* data,
                                                                     const size_t N,
                                                                     float scale_multiplier,
                                                                     float scene_scale,
                                                                     int sh_degree,
                                                                     const MeshData& mesh) {
            assert(N > 0);

            auto means = Tensor::empty({N, 3}, Device::CPU);
            auto scaling_raw = Tensor::empty({N, 3}, Device::CPU);
            auto rotation_raw = Tensor::empty({N, 4}, Device::CPU);
            auto opacity_raw = Tensor::empty({N, 1}, Device::CPU);
            auto sh0 = Tensor::empty({N, 1, 3}, Device::CPU);
            auto birth_tri_cpu = Tensor::empty({N}, Device::CPU, DataType::Int32);

            float* m_ptr = means.ptr<float>();
            float* s_ptr = scaling_raw.ptr<float>();
            float* r_ptr = rotation_raw.ptr<float>();
            float* o_ptr = opacity_raw.ptr<float>();
            float* c_ptr = sh0.ptr<float>();
            int32_t* bt_ptr = birth_tri_cpu.ptr<int32_t>();

            // unsigmoid(0.999) — fully opaque but avoids infinity
            const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);
            double max_scale_sum = 0.0;

            for (size_t i = 0; i < N; i++) {
                const auto& g = data[i];

                // Position
                m_ptr[i * 3 + 0] = g.position.x;
                m_ptr[i * 3 + 1] = g.position.y;
                m_ptr[i * 3 + 2] = g.position.z;

                // Scale: log(linear_scale * sigma / raster_resolution)
                glm::vec3 ls(g.scale.x, g.scale.y, g.scale.z);
                ls *= scale_multiplier;
                ls = glm::max(ls, glm::vec3(1e-8f));
                max_scale_sum += static_cast<double>(std::max({ls.x, ls.y, ls.z}));
                s_ptr[i * 3 + 0] = std::log(ls.x);
                s_ptr[i * 3 + 1] = std::log(ls.y);
                s_ptr[i * 3 + 2] = std::log(ls.z);

                // Rotation: SSBO stores (w, x, y, z), SplatData stores (w, x, y, z) at [0,1,2,3]
                r_ptr[i * 4 + 0] = g.rotation.x;
                r_ptr[i * 4 + 1] = g.rotation.y;
                r_ptr[i * 4 + 2] = g.rotation.z;
                r_ptr[i * 4 + 3] = g.rotation.w;

                // Opacity (fully opaque)
                o_ptr[i] = opacity_logit;

                // SH DC: (color_linear - 0.5) / SH_C0
                c_ptr[i * 3 + 0] = (g.color.x - 0.5f) / SH_C0;
                c_ptr[i * 3 + 1] = (g.color.y - 0.5f) / SH_C0;
                c_ptr[i * 3 + 2] = (g.color.z - 0.5f) / SH_C0;

                // Birth triangle id, packed in extra.x via intBitsToFloat in the FS.
                int32_t tri_id;
                std::memcpy(&tri_id, &g.extra.x, sizeof(int32_t));
                bt_ptr[i] = tri_id;
            }

            means = means.to(Device::CUDA);
            scaling_raw = scaling_raw.to(Device::CUDA);
            rotation_raw = rotation_raw.to(Device::CUDA);
            opacity_raw = opacity_raw.to(Device::CUDA);
            sh0 = sh0.to(Device::CUDA);

            const size_t feature_shape = static_cast<size_t>((sh_degree + 1) * (sh_degree + 1));
            auto shN = Tensor::zeros({N, feature_shape > 1 ? feature_shape - 1 : 0, 3}, Device::CUDA);

            auto splat = std::make_unique<SplatData>(
                sh_degree,
                std::move(means),
                std::move(sh0),
                std::move(shN),
                std::move(scaling_raw),
                std::move(rotation_raw),
                std::move(opacity_raw),
                scene_scale);
            const float mean_max_scale = static_cast<float>(max_scale_sum / static_cast<double>(N));
            splat->set_mesh2splat_mean_max_scale(mean_max_scale);
            LOG_INFO("mesh2splat: mean initialized max scale={:.6g}", mean_max_scale);

            // Populate mesh-surface-constraint data: full mesh + per-Gaussian initial face.
            // MCMC uses the initial face as a seed for surface-walk projection.
            if (mesh.vertices.is_valid() && mesh.indices.is_valid()) {
                auto verts_cuda = mesh.vertices.device() == Device::CUDA
                                      ? mesh.vertices
                                      : mesh.vertices.to(Device::CUDA);
                auto idx_cuda = mesh.indices.device() == Device::CUDA
                                    ? mesh.indices
                                    : mesh.indices.to(Device::CUDA);
                splat->constraint_mesh_verts() = std::move(verts_cuda);
                splat->constraint_mesh_indices() = std::move(idx_cuda);
                splat->birth_tri() = birth_tri_cpu.to(Device::CUDA);

                // Build per-triangle edge adjacency (3 edge-neighbors per face, -1 for
                // boundary edges) on CPU, then upload. Edges are matched by quantized
                // vertex position instead of vertex index so UV seams / hard-normal splits
                // do not become artificial mesh boundaries for surface walking.
                {
                    auto verts_cpu = mesh.vertices.device() == Device::CPU
                                          ? mesh.vertices
                                          : mesh.vertices.to(Device::CPU);
                    auto idx_cpu = mesh.indices.device() == Device::CPU
                                       ? mesh.indices
                                       : mesh.indices.to(Device::CPU);
                    const float* vp = verts_cpu.ptr<float>();
                    const int32_t* fp = idx_cpu.ptr<int32_t>();
                    const auto F = static_cast<size_t>(idx_cpu.shape()[0]);
                    auto edge_neighbors_cpu = Tensor::empty({F, 3}, Device::CPU, DataType::Int32);
                    int32_t* en_ptr = edge_neighbors_cpu.ptr<int32_t>();
                    for (size_t i = 0; i < F * 3; ++i)
                        en_ptr[i] = -1;

                    struct QuantizedPoint {
                        int64_t x;
                        int64_t y;
                        int64_t z;

                        bool operator==(const QuantizedPoint& other) const noexcept {
                            return x == other.x && y == other.y && z == other.z;
                        }
                        bool operator<(const QuantizedPoint& other) const noexcept {
                            if (x != other.x)
                                return x < other.x;
                            if (y != other.y)
                                return y < other.y;
                            return z < other.z;
                        }
                    };

                    struct EdgeKey {
                        QuantizedPoint a;
                        QuantizedPoint b;

                        bool operator==(const EdgeKey& other) const noexcept {
                            return a == other.a && b == other.b;
                        }
                    };

                    struct EdgeKeyHash {
                        size_t operator()(const EdgeKey& key) const noexcept {
                            auto mix = [](size_t seed, int64_t value) {
                                const size_t h = std::hash<int64_t>{}(value);
                                return seed ^ (h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
                            };
                            size_t seed = 0;
                            seed = mix(seed, key.a.x);
                            seed = mix(seed, key.a.y);
                            seed = mix(seed, key.a.z);
                            seed = mix(seed, key.b.x);
                            seed = mix(seed, key.b.y);
                            seed = mix(seed, key.b.z);
                            return seed;
                        }
                    };

                    struct EdgeRecord {
                        int32_t face;
                        int32_t edge;
                        int32_t v0;
                        int32_t v1;
                    };

                    const double weld_eps = std::max(static_cast<double>(scene_scale) * 1e-6, 1e-7);
                    const double inv_weld_eps = 1.0 / weld_eps;
                    auto quantize_vertex = [&](int32_t vi) {
                        return QuantizedPoint{
                            static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 0]) * inv_weld_eps)),
                            static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 1]) * inv_weld_eps)),
                            static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 2]) * inv_weld_eps))};
                    };

                    auto edge_key = [&](int32_t a, int32_t b) {
                        auto qa = quantize_vertex(a);
                        auto qb = quantize_vertex(b);
                        if (qb < qa)
                            std::swap(qa, qb);
                        return EdgeKey{qa, qb};
                    };

                    // Map sorted geometric edge -> {face_id, opposite_local_edge}.
                    // Local edge e is the edge OPPOSITE vertex e (e=0 -> v1-v2, e=1 -> v2-v0, e=2 -> v0-v1).
                    std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> edge_map;
                    edge_map.reserve(F * 3);
                    size_t welded_duplicate_edges = 0;
                    size_t non_manifold_edges = 0;

                    for (size_t f = 0; f < F; ++f) {
                        const int32_t v0 = fp[f * 3 + 0];
                        const int32_t v1 = fp[f * 3 + 1];
                        const int32_t v2 = fp[f * 3 + 2];
                        const int32_t edges[3][2] = {{v1, v2}, {v2, v0}, {v0, v1}};
                        for (int e = 0; e < 3; ++e) {
                            auto key = edge_key(edges[e][0], edges[e][1]);
                            auto it = edge_map.find(key);
                            if (it == edge_map.end()) {
                                edge_map.emplace(key, EdgeRecord{static_cast<int32_t>(f), e, edges[e][0], edges[e][1]});
                            } else {
                                const auto& rec = it->second;
                                if (en_ptr[static_cast<size_t>(rec.face) * 3 + rec.edge] < 0) {
                                    en_ptr[f * 3 + e] = rec.face;
                                    en_ptr[static_cast<size_t>(rec.face) * 3 + rec.edge] = static_cast<int32_t>(f);
                                    const bool same_index_edge =
                                        (rec.v0 == edges[e][0] && rec.v1 == edges[e][1]) ||
                                        (rec.v0 == edges[e][1] && rec.v1 == edges[e][0]);
                                    if (!same_index_edge)
                                        ++welded_duplicate_edges;
                                } else {
                                    ++non_manifold_edges;
                                }
                            }
                        }
                    }

                    if (welded_duplicate_edges > 0 || non_manifold_edges > 0) {
                        LOG_INFO("mesh2splat: geometric edge adjacency welded {} seam edges (eps={:.3e}); skipped {} non-manifold incidents",
                                 welded_duplicate_edges, weld_eps, non_manifold_edges);
                    }

                    splat->tri_edge_neighbors() = edge_neighbors_cpu.to(Device::CUDA);
                }
            }

            return splat;
        }

        struct SplatCpuStaging {
            Tensor means;
            Tensor scaling_raw;
            Tensor rotation_raw;
            Tensor opacity_raw;
            Tensor sh0;
            Tensor birth_tri;
            Tensor mesh_hole_fill_mask;
            double max_scale_sum = 0.0;
            size_t invalid_render_face_id_count = 0;
        };

        SplatCpuStaging allocate_splat_cpu_staging(
            const size_t N,
            const bool include_hole_fill_provenance) {
            SplatCpuStaging staging;
            staging.means = Tensor::empty({N, 3}, Device::CPU);
            staging.scaling_raw = Tensor::empty({N, 3}, Device::CPU);
            staging.rotation_raw = Tensor::empty({N, 4}, Device::CPU);
            staging.opacity_raw = Tensor::empty({N, 1}, Device::CPU);
            staging.sh0 = Tensor::empty({N, 1, 3}, Device::CPU);
            staging.birth_tri = Tensor::empty({N}, Device::CPU, DataType::Int32);
            if (include_hole_fill_provenance) {
                staging.mesh_hole_fill_mask = Tensor::zeros_bool({N}, Device::CPU);
            }
            return staging;
        }

        void fill_splat_cpu_staging(
            SplatCpuStaging& staging,
            const GaussianVertex* data,
            const size_t dst_offset,
            const size_t count,
            const float scale_multiplier,
            const std::span<const int32_t> render_face_to_constraint_face,
            Mesh2SplatFaceStatistics* face_statistics) {

            float* m_ptr = staging.means.ptr<float>();
            float* s_ptr = staging.scaling_raw.ptr<float>();
            float* r_ptr = staging.rotation_raw.ptr<float>();
            float* o_ptr = staging.opacity_raw.ptr<float>();
            float* c_ptr = staging.sh0.ptr<float>();
            int32_t* bt_ptr = staging.birth_tri.ptr<int32_t>();
            bool* hole_ptr = staging.mesh_hole_fill_mask.is_valid()
                                 ? staging.mesh_hole_fill_mask.ptr<bool>()
                                 : nullptr;
            const float opacity_logit = -std::log(1.0f / 0.999f - 1.0f);

            for (size_t j = 0; j < count; ++j) {
                const size_t i = dst_offset + j;
                const auto& g = data[j];

                m_ptr[i * 3 + 0] = g.position.x;
                m_ptr[i * 3 + 1] = g.position.y;
                m_ptr[i * 3 + 2] = g.position.z;

                glm::vec3 ls(g.scale.x, g.scale.y, g.scale.z);
                ls *= scale_multiplier;
                ls = glm::max(ls, glm::vec3(1e-8f));
                staging.max_scale_sum += static_cast<double>(std::max({ls.x, ls.y, ls.z}));
                s_ptr[i * 3 + 0] = std::log(ls.x);
                s_ptr[i * 3 + 1] = std::log(ls.y);
                s_ptr[i * 3 + 2] = std::log(ls.z);

                r_ptr[i * 4 + 0] = g.rotation.x;
                r_ptr[i * 4 + 1] = g.rotation.y;
                r_ptr[i * 4 + 2] = g.rotation.z;
                r_ptr[i * 4 + 3] = g.rotation.w;

                o_ptr[i] = opacity_logit;

                c_ptr[i * 3 + 0] = (g.color.x - 0.5f) / SH_C0;
                c_ptr[i * 3 + 1] = (g.color.y - 0.5f) / SH_C0;
                c_ptr[i * 3 + 2] = (g.color.z - 0.5f) / SH_C0;

                int32_t tri_id;
                std::memcpy(&tri_id, &g.extra.x, sizeof(int32_t));
                const int32_t raw_render_face = tri_id;
                const size_t render_face_count = !render_face_to_constraint_face.empty()
                                                     ? render_face_to_constraint_face.size()
                                                 : face_statistics
                                                     ? face_statistics->gaussian_count_per_render_face.size()
                                                     : 0;
                if (render_face_count > 0) {
                    if (tri_id >= 0 && static_cast<size_t>(tri_id) < render_face_count) {
                        if (face_statistics) {
                            ++face_statistics->gaussian_count_per_render_face[static_cast<size_t>(tri_id)];
                            face_statistics->raw_render_face_per_gaussian[i] = raw_render_face;
                        }
                        if (!render_face_to_constraint_face.empty()) {
                            const auto mapped_face = detail::map_mesh2splat_render_face(
                                tri_id,
                                render_face_to_constraint_face);
                            if (!mapped_face) {
                                tri_id = -1;
                                ++staging.invalid_render_face_id_count;
                            } else {
                                tri_id = mapped_face->constraint_face;
                                hole_ptr[i] = mapped_face->hole_fill;
                            }
                        }
                    } else {
                        tri_id = -1;
                        ++staging.invalid_render_face_id_count;
                    }
                }
                bt_ptr[i] = tri_id;
            }
        }

        void build_mesh_edge_adjacency(SplatData& splat, const MeshData& mesh, const float scene_scale) {
            auto verts_cpu = mesh.vertices.device() == Device::CPU
                                 ? mesh.vertices
                                 : mesh.vertices.to(Device::CPU);
            auto idx_cpu = mesh.indices.device() == Device::CPU
                               ? mesh.indices
                               : mesh.indices.to(Device::CPU);
            const float* vp = verts_cpu.ptr<float>();
            const int32_t* fp = idx_cpu.ptr<int32_t>();
            const auto F = static_cast<size_t>(idx_cpu.shape()[0]);
            auto edge_neighbors_cpu = Tensor::empty({F, 3}, Device::CPU, DataType::Int32);
            int32_t* en_ptr = edge_neighbors_cpu.ptr<int32_t>();
            for (size_t i = 0; i < F * 3; ++i)
                en_ptr[i] = -1;

            struct QuantizedPoint {
                int64_t x;
                int64_t y;
                int64_t z;

                bool operator==(const QuantizedPoint& other) const noexcept {
                    return x == other.x && y == other.y && z == other.z;
                }
                bool operator<(const QuantizedPoint& other) const noexcept {
                    if (x != other.x)
                        return x < other.x;
                    if (y != other.y)
                        return y < other.y;
                    return z < other.z;
                }
            };

            struct EdgeKey {
                QuantizedPoint a;
                QuantizedPoint b;

                bool operator==(const EdgeKey& other) const noexcept {
                    return a == other.a && b == other.b;
                }
            };

            struct EdgeKeyHash {
                size_t operator()(const EdgeKey& key) const noexcept {
                    auto mix = [](size_t seed, int64_t value) {
                        const size_t h = std::hash<int64_t>{}(value);
                        return seed ^ (h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
                    };
                    size_t seed = 0;
                    seed = mix(seed, key.a.x);
                    seed = mix(seed, key.a.y);
                    seed = mix(seed, key.a.z);
                    seed = mix(seed, key.b.x);
                    seed = mix(seed, key.b.y);
                    seed = mix(seed, key.b.z);
                    return seed;
                }
            };

            struct EdgeRecord {
                int32_t face;
                int32_t edge;
                int32_t v0;
                int32_t v1;
            };

            const double weld_eps = std::max(static_cast<double>(scene_scale) * 1e-6, 1e-7);
            const double inv_weld_eps = 1.0 / weld_eps;
            auto quantize_vertex = [&](int32_t vi) {
                return QuantizedPoint{
                    static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 0]) * inv_weld_eps)),
                    static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 1]) * inv_weld_eps)),
                    static_cast<int64_t>(std::llround(static_cast<double>(vp[vi * 3 + 2]) * inv_weld_eps))};
            };

            auto edge_key = [&](int32_t a, int32_t b) {
                auto qa = quantize_vertex(a);
                auto qb = quantize_vertex(b);
                if (qb < qa)
                    std::swap(qa, qb);
                return EdgeKey{qa, qb};
            };

            std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> edge_map;
            edge_map.reserve(F * 3);
            size_t welded_duplicate_edges = 0;
            size_t non_manifold_edges = 0;

            for (size_t f = 0; f < F; ++f) {
                const int32_t v0 = fp[f * 3 + 0];
                const int32_t v1 = fp[f * 3 + 1];
                const int32_t v2 = fp[f * 3 + 2];
                const int32_t edges[3][2] = {{v1, v2}, {v2, v0}, {v0, v1}};
                for (int e = 0; e < 3; ++e) {
                    auto key = edge_key(edges[e][0], edges[e][1]);
                    auto it = edge_map.find(key);
                    if (it == edge_map.end()) {
                        edge_map.emplace(key, EdgeRecord{static_cast<int32_t>(f), e, edges[e][0], edges[e][1]});
                    } else {
                        const auto& rec = it->second;
                        if (en_ptr[static_cast<size_t>(rec.face) * 3 + rec.edge] < 0) {
                            en_ptr[f * 3 + e] = rec.face;
                            en_ptr[static_cast<size_t>(rec.face) * 3 + rec.edge] = static_cast<int32_t>(f);
                            const bool same_index_edge =
                                (rec.v0 == edges[e][0] && rec.v1 == edges[e][1]) ||
                                (rec.v0 == edges[e][1] && rec.v1 == edges[e][0]);
                            if (!same_index_edge)
                                ++welded_duplicate_edges;
                        } else {
                            ++non_manifold_edges;
                        }
                    }
                }
            }

            if (welded_duplicate_edges > 0 || non_manifold_edges > 0) {
                LOG_INFO("mesh2splat: geometric edge adjacency welded {} seam edges (eps={:.3e}); skipped {} non-manifold incidents",
                         welded_duplicate_edges, weld_eps, non_manifold_edges);
            }

            splat.tri_edge_neighbors() = edge_neighbors_cpu.to(Device::CUDA);
        }

        std::unique_ptr<SplatData> finalize_splat_data(
            SplatCpuStaging&& staging,
            const size_t N,
            const float scene_scale,
            const int sh_degree,
            const MeshData& constraint_mesh) {

            auto means_cuda = staging.means.to(Device::CUDA);
            staging.means = Tensor();
            auto scaling_cuda = staging.scaling_raw.to(Device::CUDA);
            staging.scaling_raw = Tensor();
            auto rotation_cuda = staging.rotation_raw.to(Device::CUDA);
            staging.rotation_raw = Tensor();
            auto opacity_cuda = staging.opacity_raw.to(Device::CUDA);
            staging.opacity_raw = Tensor();
            auto sh0_cuda = staging.sh0.to(Device::CUDA);
            staging.sh0 = Tensor();
            auto birth_tri_cpu = std::move(staging.birth_tri);
            auto mesh_hole_fill_mask_cpu = std::move(staging.mesh_hole_fill_mask);

            const size_t feature_shape = static_cast<size_t>((sh_degree + 1) * (sh_degree + 1));
            auto shN = Tensor::zeros({N, feature_shape > 1 ? feature_shape - 1 : 0, 3}, Device::CUDA);

            auto splat = std::make_unique<SplatData>(
                sh_degree,
                std::move(means_cuda),
                std::move(sh0_cuda),
                std::move(shN),
                std::move(scaling_cuda),
                std::move(rotation_cuda),
                std::move(opacity_cuda),
                scene_scale);
            const float mean_max_scale = static_cast<float>(staging.max_scale_sum / static_cast<double>(N));
            splat->set_mesh2splat_mean_max_scale(mean_max_scale);
            LOG_INFO("mesh2splat: mean initialized max scale={:.6g}", mean_max_scale);

            if (constraint_mesh.vertices.is_valid() && constraint_mesh.indices.is_valid()) {
                auto verts_cuda = constraint_mesh.vertices.device() == Device::CUDA
                                      ? constraint_mesh.vertices
                                      : constraint_mesh.vertices.to(Device::CUDA);
                auto idx_cuda = constraint_mesh.indices.device() == Device::CUDA
                                    ? constraint_mesh.indices
                                    : constraint_mesh.indices.to(Device::CUDA);
                splat->constraint_mesh_verts() = std::move(verts_cuda);
                splat->constraint_mesh_indices() = std::move(idx_cuda);
                splat->birth_tri() = birth_tri_cpu.to(Device::CUDA);
                birth_tri_cpu = Tensor();
                if (mesh_hole_fill_mask_cpu.is_valid()) {
                    splat->mesh_hole_fill_mask() = mesh_hole_fill_mask_cpu.to(Device::CUDA);
                    mesh_hole_fill_mask_cpu = Tensor();
                }

                build_mesh_edge_adjacency(*splat, constraint_mesh, scene_scale);
            }

            return splat;
        }

        std::unique_ptr<SplatData> build_splat_data_from_ssbo(
            GLuint ssbo,
            const uint32_t num_gaussians,
            const float scale_multiplier,
            const float scene_scale,
            const int sh_degree,
            const MeshData& constraint_mesh,
            const std::span<const int32_t> render_face_to_constraint_face,
            Mesh2SplatFaceStatistics* face_statistics) {

            const auto N = static_cast<size_t>(num_gaussians);
            assert(N > 0);
            if (face_statistics) {
                face_statistics->raw_render_face_per_gaussian.assign(N, -1);
            }
            auto staging = allocate_splat_cpu_staging(
                N,
                detail::mesh2splat_uses_hole_fill_provenance(render_face_to_constraint_face));
            const GLsizeiptr readback_bytes = static_cast<GLsizeiptr>(N * sizeof(GaussianVertex));

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, readback_bytes, GL_MAP_READ_BIT);
            if (mapped) {
                fill_splat_cpu_staging(
                    staging,
                    static_cast<const GaussianVertex*>(mapped),
                    0,
                    N,
                    scale_multiplier,
                    render_face_to_constraint_face,
                    face_statistics);
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            } else {
                constexpr size_t kReadbackChunkGaussians = 64 * 1024;
                std::vector<GaussianVertex> readback_chunk;
                readback_chunk.resize(std::min(kReadbackChunkGaussians, N));
                for (size_t offset = 0; offset < N;) {
                    const size_t count = std::min(readback_chunk.size(), N - offset);
                    glGetBufferSubData(
                        GL_SHADER_STORAGE_BUFFER,
                        static_cast<GLintptr>(offset * sizeof(GaussianVertex)),
                        static_cast<GLsizeiptr>(count * sizeof(GaussianVertex)),
                        readback_chunk.data());
                    fill_splat_cpu_staging(
                        staging,
                        readback_chunk.data(),
                        offset,
                        count,
                        scale_multiplier,
                        render_face_to_constraint_face,
                        face_statistics);
                    offset += count;
                }
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            if (staging.invalid_render_face_id_count > 0) {
                LOG_ERROR("mesh2splat: read back {} invalid raw render face ids",
                          staging.invalid_render_face_id_count);
                return nullptr;
            }

            return finalize_splat_data(std::move(staging), N, scene_scale, sh_degree, constraint_mesh);
        }

        // ========================================================================
        // Bind material textures for a submesh
        // ========================================================================

        void bind_submesh_textures(GLuint program, const MeshData& mesh, size_t material_index,
                                   GlCleanup& cleanup) {
            GLint has_albedo_loc = glGetUniformLocation(program, "hasAlbedoMap");
            GLint has_normal_loc = glGetUniformLocation(program, "hasNormalMap");
            GLint has_mr_loc = glGetUniformLocation(program, "hasMetallicRoughnessMap");

            if (has_albedo_loc >= 0)
                glUniform1i(has_albedo_loc, 0);
            if (has_normal_loc >= 0)
                glUniform1i(has_normal_loc, 0);
            if (has_mr_loc >= 0)
                glUniform1i(has_mr_loc, 0);

            if (material_index >= mesh.materials.size()) {
                LOG_DEBUG("mesh2splat: no material at index {}", material_index);
                return;
            }
            const auto& mat = mesh.materials[material_index];
            LOG_DEBUG("mesh2splat: material '{}' base_color=({},{},{},{}), albedo_tex={}, albedo_path='{}'",
                      mat.name, mat.base_color.r, mat.base_color.g, mat.base_color.b, mat.base_color.a,
                      mat.albedo_tex, mat.albedo_tex_path);

            // Upload all textures before binding — upload_texture() binds/unbinds
            // on the active texture unit, which would clobber earlier bindings.
            GLuint albedo_gl = 0, normal_gl = 0, mr_gl = 0;

            if (mat.has_albedo_texture() && mat.albedo_tex > 0 &&
                mat.albedo_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.albedo_tex - 1];
                if (!img.pixels.empty()) {
                    LOG_DEBUG("mesh2splat: uploading albedo texture {}x{} ({} ch, {} bytes)",
                              img.width, img.height, img.channels, img.pixels.size());
                    albedo_gl = upload_texture(img, true);
                    if (albedo_gl)
                        cleanup.textures.push_back(albedo_gl);
                }
            }

            if (mat.has_normal_texture() && mat.normal_tex > 0 &&
                mat.normal_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.normal_tex - 1];
                if (!img.pixels.empty()) {
                    normal_gl = upload_texture(img);
                    if (normal_gl)
                        cleanup.textures.push_back(normal_gl);
                }
            }

            if (mat.has_metallic_roughness_texture() && mat.metallic_roughness_tex > 0 &&
                mat.metallic_roughness_tex <= mesh.texture_images.size()) {
                const auto& img = mesh.texture_images[mat.metallic_roughness_tex - 1];
                if (!img.pixels.empty()) {
                    mr_gl = upload_texture(img);
                    if (mr_gl)
                        cleanup.textures.push_back(mr_gl);
                }
            }

            // Bind to texture units after all uploads are complete
            if (albedo_gl) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, albedo_gl);
                GLint loc = glGetUniformLocation(program, "albedoTexture");
                if (loc >= 0)
                    glUniform1i(loc, 0);
                if (has_albedo_loc >= 0)
                    glUniform1i(has_albedo_loc, 1);
            }

            if (normal_gl) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, normal_gl);
                GLint loc = glGetUniformLocation(program, "normalTexture");
                if (loc >= 0)
                    glUniform1i(loc, 1);
                if (has_normal_loc >= 0)
                    glUniform1i(has_normal_loc, 1);
            }

            if (mr_gl) {
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, mr_gl);
                GLint loc = glGetUniformLocation(program, "metallicRoughnessTexture");
                if (loc >= 0)
                    glUniform1i(loc, 2);
                if (has_mr_loc >= 0)
                    glUniform1i(has_mr_loc, 1);
            }
        }

    } // anonymous namespace

    namespace detail {

        std::vector<bool> build_mesh2splat_uniform_cap_keep_mask(
            const std::size_t row_count,
            const std::size_t target_row_count) {
            if (target_row_count == 0 || row_count <= target_row_count) {
                return std::vector<bool>(row_count, true);
            }

            std::vector<bool> keep_rows(row_count, false);
            const long double stride = static_cast<long double>(row_count) /
                                       static_cast<long double>(target_row_count);
            for (std::size_t output_row = 0; output_row < target_row_count; ++output_row) {
                const auto source_row = static_cast<std::size_t>(std::floor(
                    (static_cast<long double>(output_row) + 0.5L) * stride));
                keep_rows[std::min(source_row, row_count - 1)] = true;
            }
            return keep_rows;
        }

        std::expected<Mesh2SplatMappedFace, std::string>
        map_mesh2splat_render_face(
            const int32_t raw_render_face,
            const std::span<const int32_t> render_face_to_constraint_face) {
            if (raw_render_face < 0) {
                return std::unexpected("mesh2splat raw render face id is negative");
            }
            if (render_face_to_constraint_face.empty()) {
                return Mesh2SplatMappedFace{
                    .constraint_face = raw_render_face,
                    .hole_fill = false};
            }
            if (static_cast<size_t>(raw_render_face) >= render_face_to_constraint_face.size()) {
                return std::unexpected("mesh2splat raw render face id is out of range");
            }
            const int32_t constraint_face =
                render_face_to_constraint_face[static_cast<size_t>(raw_render_face)];
            return Mesh2SplatMappedFace{
                .constraint_face = constraint_face,
                .hole_fill = constraint_face < 0};
        }

        std::expected<void, std::string>
        validate_mesh2splat_face_mapping_contract(
            const std::size_t render_face_count,
            const std::size_t constraint_face_count,
            const bool uses_separate_constraint_mesh,
            const std::span<const int32_t> render_face_to_constraint_face) {
            if (render_face_to_constraint_face.empty()) {
                if (uses_separate_constraint_mesh) {
                    return std::unexpected(
                        "mesh2splat separate render/constraint meshes require a render-face mapping");
                }
                return {};
            }
            if (render_face_to_constraint_face.size() != render_face_count) {
                return std::unexpected(
                    "mesh2splat render-face mapping length must equal render mesh face count");
            }
            const bool invalid_mapping = std::any_of(
                render_face_to_constraint_face.begin(),
                render_face_to_constraint_face.end(),
                [constraint_face_count](const int32_t face) {
                    return face < -1 ||
                           (face >= 0 && static_cast<std::size_t>(face) >= constraint_face_count);
                });
            if (invalid_mapping) {
                return std::unexpected(
                    "mesh2splat render-face mapping contains an invalid constraint face id");
            }
            return {};
        }

    } // namespace detail

    // ============================================================================
    // Public API
    // ============================================================================

    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const Mesh2SplatInput& input,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {

        const MeshData& mesh = input.render_mesh;
        const MeshData& constraint_mesh = input.constraint_mesh ? *input.constraint_mesh : mesh;
        if (input.face_statistics) {
            input.face_statistics->gaussian_count_per_render_face.assign(
                static_cast<size_t>(mesh.face_count()), 0);
            input.face_statistics->raw_render_face_per_gaussian.clear();
            input.face_statistics->gaussian_count = 0;
            input.face_statistics->base_resolution = 0;
            input.face_statistics->requested_resolution = 0;
            input.face_statistics->final_resolution = 0;
        }

        auto report = [&](float pct, const std::string& stage) -> bool {
            if (progress)
                return progress(pct, stage);
            return true;
        };

        // Validate inputs
        if (!mesh.vertices.is_valid() || mesh.vertex_count() == 0)
            return std::unexpected("Mesh has no vertices");
        if (!mesh.indices.is_valid() || mesh.face_count() == 0)
            return std::unexpected("Mesh has no faces");
        if (!constraint_mesh.vertices.is_valid() || constraint_mesh.vertex_count() == 0)
            return std::unexpected("Constraint mesh has no vertices");
        if (!constraint_mesh.indices.is_valid() || constraint_mesh.face_count() == 0)
            return std::unexpected("Constraint mesh has no faces");
        if (auto mapping_contract = detail::validate_mesh2splat_face_mapping_contract(
                static_cast<std::size_t>(mesh.face_count()),
                static_cast<std::size_t>(constraint_mesh.face_count()),
                &constraint_mesh != &mesh,
                input.render_face_to_constraint_face);
            !mapping_contract) {
            return std::unexpected(mapping_contract.error());
        }
        if (options.resolution_target < Mesh2SplatOptions::kMinResolution)
            return std::unexpected("mesh2splat resolution_target must be at least kMinResolution");
        if (options.sampling_rate <= 0.0f || options.sampling_rate > 1.0f)
            return std::unexpected("mesh2splat sampling_rate must be in (0, 1]");
        if (options.target_max_gaussians < 0)
            return std::unexpected("mesh2splat target_max_gaussians must be non-negative");
        assert(options.sigma > 0.0f);

        GlStateSave state_save;
        GlCleanup cleanup;

        // Phase 1: Extract per-face geometry from indexed mesh and upload reusable VBOs.
        if (!report(0.0f, "Preparing mesh data"))
            return std::unexpected("Cancelled");

        auto prepared_geometry = prepare_geometry_buffers(mesh);
        if (prepared_geometry.buffers.empty())
            return std::unexpected("No geometry extracted");

        const glm::vec3 global_min = prepared_geometry.bbox_min;
        const glm::vec3 global_max = prepared_geometry.bbox_max;
        const size_t total_vertices = prepared_geometry.total_vertices;
        const float scene_scale = glm::length(global_max - global_min) * 0.5f;
        if (scene_scale <= 0.0f)
            return std::unexpected("Degenerate mesh: zero bounding box extent");

        const int base_res = options.resolution_target;
        const float sampling_rate = options.sampling_rate;
        const float resolution_scale = sampling_rate < 1.0f ? std::sqrt(sampling_rate) : 1.0f;
        const int requested_res = std::clamp(
            static_cast<int>(std::round(static_cast<float>(base_res) * resolution_scale)),
            Mesh2SplatOptions::kMinResolution,
            base_res);
        int res = requested_res;

        LOG_INFO("mesh2splat: {} submeshes, {} triangles, base_resolution={}, requested_resolution={}, sampling_rate={:.4f}, target_max_gaussians={}, bbox=[{:.2f},{:.2f},{:.2f}]-[{:.2f},{:.2f},{:.2f}]",
                 prepared_geometry.buffers.size(), total_vertices / 3, base_res, requested_res, sampling_rate,
                 options.target_max_gaussians,
                 global_min.x, global_min.y, global_min.z,
                 global_max.x, global_max.y, global_max.z);

        // Phase 3: GL pipeline — save state, compile shaders, run conversion
        if (!report(0.2f, "Compiling shaders"))
            return std::unexpected("Cancelled");

        cleanup.program = create_conversion_program();
        if (!cleanup.program)
            return std::unexpected("Failed to compile mesh2splat shaders");

        auto count_gaussians_at_resolution = [&](int probe_res) -> std::expected<uint32_t, std::string> {
            GlCleanup probe_cleanup;

            glGenBuffers(1, &probe_cleanup.ssbo);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, probe_cleanup.ssbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(sizeof(GaussianVertex)),
                         nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            glGenBuffers(1, &probe_cleanup.atomic_counter);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, probe_cleanup.atomic_counter);
            GLuint zero = 0;
            glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

            if (!create_fbo(probe_res, probe_res, probe_cleanup.fbo, probe_cleanup.rbo)) {
                return std::unexpected("Failed to create framebuffer for mesh2splat density probe");
            }

            glBindFramebuffer(GL_FRAMEBUFFER, probe_cleanup.fbo);
            glViewport(0, 0, probe_res, probe_res);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glDisable(GL_CULL_FACE);

            glUseProgram(cleanup.program);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, probe_cleanup.ssbo);
            glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, probe_cleanup.atomic_counter);

            GLint loc_count_only = glGetUniformLocation(cleanup.program, "u_countOnly");
            GLint loc_vertex_capacity = glGetUniformLocation(cleanup.program, "u_vertexCapacity");
            if (loc_count_only >= 0)
                glUniform1i(loc_count_only, 1);
            if (loc_vertex_capacity >= 0)
                glUniform1ui(loc_vertex_capacity, 1);

            for (size_t si = 0; si < prepared_geometry.buffers.size(); si++) {
                const auto& geo = prepared_geometry.buffers[si];
                glBindVertexArray(geo.vao);

                GLint loc_bbox_min = glGetUniformLocation(cleanup.program, "u_bboxMin");
                GLint loc_bbox_max = glGetUniformLocation(cleanup.program, "u_bboxMax");
                GLint loc_tri_offset = glGetUniformLocation(cleanup.program, "u_triangleOffset");
                if (loc_bbox_min >= 0)
                    glUniform3fv(loc_bbox_min, 1, glm::value_ptr(global_min));
                if (loc_bbox_max >= 0)
                    glUniform3fv(loc_bbox_max, 1, glm::value_ptr(global_max));
                if (loc_tri_offset >= 0)
                    glUniform1i(loc_tri_offset, geo.face_offset);

                GLenum err_before = glGetError();
                (void)err_before;
                glDrawArrays(GL_TRIANGLES, 0, geo.vertex_count);
                GLenum err_after = glGetError();
                if (err_after != GL_NO_ERROR)
                    LOG_ERROR("mesh2splat: GL error after density probe draw: 0x{:X}", err_after);
            }
            glBindVertexArray(0);

            glFinish();

            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, probe_cleanup.atomic_counter);
            uint32_t count = 0;
            glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &count);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
            return count;
        };

        uint32_t expected_final_count = 0;
        if (options.target_max_gaussians > 0) {
            const uint32_t target = static_cast<uint32_t>(options.target_max_gaussians);
            while (true) {
                if (!report(0.25f, "Probing mesh2splat density"))
                    return std::unexpected("Cancelled");

                auto probe_count = count_gaussians_at_resolution(res);
                if (!probe_count)
                    return std::unexpected(probe_count.error());
                expected_final_count = *probe_count;

                LOG_INFO("mesh2splat: density probe resolution={}, count={}, target_max_gaussians={}",
                         res, expected_final_count, target);

                if (expected_final_count == 0)
                    return std::unexpected("Conversion produced zero gaussians");
                if (expected_final_count <= target || res <= Mesh2SplatOptions::kMinResolution)
                    break;

                const long double scale =
                    std::sqrt(static_cast<long double>(target) /
                              static_cast<long double>(expected_final_count));
                int next_res = static_cast<int>(std::floor(static_cast<long double>(res) * scale));
                next_res = std::clamp(next_res, Mesh2SplatOptions::kMinResolution, res - 1);

                LOG_INFO("mesh2splat: reducing resolution {} -> {} to approach target_max_gaussians={}",
                         res, next_res, target);
                res = next_res;
            }

            if (expected_final_count > target) {
                if (detail::mesh2splat_enforces_combined_hard_cap(
                        input.render_face_to_constraint_face)) {
                    LOG_WARN("mesh2splat: minimum resolution {} still produces {} gaussians, exceeding target_max_gaussians={}; the complete output will be uniformly sampled to the hard global cap",
                             res, expected_final_count, target);
                } else {
                    LOG_WARN("mesh2splat: minimum resolution {} still produces {} gaussians, exceeding target_max_gaussians={}; continuing without post-generation pruning",
                             res, expected_final_count, target);
                }
            }
        }

        // SSBO for gaussian output. Each rasterized fragment writes one GaussianVertex.
        // For sparse meshes (few large triangles), res^2 * 6 is sufficient.
        // For dense meshes (many small triangles), fragments scale with triangle count.
        const auto triangle_count = static_cast<GLsizeiptr>(total_vertices / 3);
        const GLsizeiptr pixel_based = static_cast<GLsizeiptr>(res) * res * 6;
        const GLsizeiptr triangle_based = triangle_count * 2;
        const GLsizeiptr estimated_entries = std::max(pixel_based, triangle_based);
        const GLsizeiptr ssbo_entries = std::max<GLsizeiptr>(
            1,
            expected_final_count > 0
                ? static_cast<GLsizeiptr>(expected_final_count)
                : estimated_entries);
        const GLsizeiptr ssbo_size = ssbo_entries * static_cast<GLsizeiptr>(sizeof(GaussianVertex));

        glGenBuffers(1, &cleanup.ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cleanup.ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, ssbo_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Create atomic counter
        glGenBuffers(1, &cleanup.atomic_counter);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, cleanup.atomic_counter);
        GLuint zero = 0;
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

        // Create FBO at full resolution
        if (!create_fbo(res, res, cleanup.fbo, cleanup.rbo)) {
            return std::unexpected("Failed to create framebuffer");
        }

        // Set up GL state for conversion
        glBindFramebuffer(GL_FRAMEBUFFER, cleanup.fbo);
        glViewport(0, 0, res, res);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        glUseProgram(cleanup.program);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cleanup.ssbo);
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, cleanup.atomic_counter);
        if (GLint loc_count_only = glGetUniformLocation(cleanup.program, "u_countOnly"); loc_count_only >= 0)
            glUniform1i(loc_count_only, 0);
        if (GLint loc_vertex_capacity = glGetUniformLocation(cleanup.program, "u_vertexCapacity"); loc_vertex_capacity >= 0)
            glUniform1ui(loc_vertex_capacity, static_cast<GLuint>(ssbo_entries));

        // Phase 4: Process each submesh
        if (!report(0.3f, "Converting mesh to splats"))
            return std::unexpected("Cancelled");

        for (size_t si = 0; si < prepared_geometry.buffers.size(); si++) {
            const auto& geo = prepared_geometry.buffers[si];
            glBindVertexArray(geo.vao);

            // Set uniforms
            glm::vec4 material_factor(1.0f);
            float metallic_factor = 0.0f;
            float roughness_factor = 1.0f;
            if (geo.material_index < mesh.materials.size()) {
                const auto& mat = mesh.materials[geo.material_index];
                material_factor = mat.base_color;
                metallic_factor = mat.metallic;
                roughness_factor = mat.roughness;
            }

            GLint loc_material = glGetUniformLocation(cleanup.program, "u_materialFactor");
            GLint loc_metallic = glGetUniformLocation(cleanup.program, "u_metallicFactor");
            GLint loc_roughness = glGetUniformLocation(cleanup.program, "u_roughnessFactor");
            GLint loc_bbox_min = glGetUniformLocation(cleanup.program, "u_bboxMin");
            GLint loc_bbox_max = glGetUniformLocation(cleanup.program, "u_bboxMax");
            GLint loc_has_vtx_colors = glGetUniformLocation(cleanup.program, "hasVertexColors");
            GLint loc_tri_offset = glGetUniformLocation(cleanup.program, "u_triangleOffset");

            if (loc_material >= 0)
                glUniform4fv(loc_material, 1, glm::value_ptr(material_factor));
            if (loc_metallic >= 0)
                glUniform1f(loc_metallic, metallic_factor);
            if (loc_roughness >= 0)
                glUniform1f(loc_roughness, roughness_factor);
            if (loc_has_vtx_colors >= 0)
                glUniform1i(loc_has_vtx_colors, mesh.has_colors() ? 1 : 0);
            // Use GLOBAL bbox so all submeshes share the same orthogonal UV space.
            // The GS maps positions to FBO pixels via triplanar projection using bbox.
            // Shared bbox = shared UV space = no duplicate gaussians across submeshes.
            if (loc_bbox_min >= 0)
                glUniform3fv(loc_bbox_min, 1, glm::value_ptr(global_min));
            if (loc_bbox_max >= 0)
                glUniform3fv(loc_bbox_max, 1, glm::value_ptr(global_max));
            // Submesh's global face-index offset for birth-triangle tracking.
            if (loc_tri_offset >= 0)
                glUniform1i(loc_tri_offset, geo.face_offset);

            // Bind textures for this submesh
            bind_submesh_textures(cleanup.program, mesh, geo.material_index, cleanup);

            LOG_DEBUG("mesh2splat: submesh[{}] material_factor=({},{},{},{}), vertices={}, "
                      "uniform_locs: material={}, bbox_min={}, bbox_max={}",
                      si, material_factor.x, material_factor.y, material_factor.z, material_factor.w,
                      geo.vertex_count, loc_material, loc_bbox_min, loc_bbox_max);

            // Draw
            GLenum err_before = glGetError(); // clear any pre-existing errors
            (void)err_before;
            glDrawArrays(GL_TRIANGLES, 0, geo.vertex_count);
            GLenum err_after = glGetError();
            if (err_after != GL_NO_ERROR)
                LOG_ERROR("mesh2splat: GL error after draw: 0x{:X}", err_after);

            const float pct = 0.3f + 0.5f * static_cast<float>(si + 1) /
                                         static_cast<float>(prepared_geometry.buffers.size());
            if (!report(pct, "Converting submesh"))
                return std::unexpected("Cancelled");
        }
        glBindVertexArray(0);

        // Phase 5: Read back results
        glFinish();
        prepared_geometry.release_buffers();

        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, cleanup.atomic_counter);
        uint32_t num_gaussians = 0;
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &num_gaussians);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

        if (num_gaussians == 0) {
            return std::unexpected("Conversion produced zero gaussians");
        }

        const uint32_t max_gaussians = static_cast<uint32_t>(ssbo_size / sizeof(GaussianVertex));
        if (num_gaussians > max_gaussians) {
            LOG_WARN("mesh2splat: atomic counter ({}) exceeds SSBO capacity ({}), clamping",
                     num_gaussians, max_gaussians);
            num_gaussians = max_gaussians;
        }

        LOG_INFO("mesh2splat: produced {} gaussians (base_resolution={}, requested_resolution={}, final_resolution={}, sampling_rate={:.4f}, target_max_gaussians={})",
                 num_gaussians, base_res, requested_res, res, sampling_rate, options.target_max_gaussians);
        if (input.face_statistics) {
            input.face_statistics->gaussian_count = num_gaussians;
            input.face_statistics->base_resolution = base_res;
            input.face_statistics->requested_resolution = requested_res;
            input.face_statistics->final_resolution = res;
        }

        if (!report(0.85f, "Reading back data"))
            return std::unexpected("Cancelled");

        // Phase 6: Convert to SplatData
        if (!report(0.9f, "Building SplatData"))
            return std::unexpected("Cancelled");

        const float scale_multiplier = options.sigma / static_cast<float>(res);
        LOG_DEBUG("mesh2splat: scale_multiplier={:.6f} (sigma={}, res={})",
                  scale_multiplier, options.sigma, res);
        auto splat = build_splat_data_from_ssbo(
            cleanup.ssbo,
            num_gaussians,
            scale_multiplier,
            scene_scale,
            options.sh_degree,
            constraint_mesh,
            input.render_face_to_constraint_face,
            input.face_statistics);
        if (!splat) {
            return std::unexpected("mesh2splat readback contained invalid render face ids");
        }

        if (detail::mesh2splat_enforces_combined_hard_cap(
                input.render_face_to_constraint_face) &&
            options.target_max_gaussians > 0 &&
            splat->size() > static_cast<size_t>(options.target_max_gaussians)) {
            const size_t uncapped_count = splat->size();
            const size_t target_count = static_cast<size_t>(options.target_max_gaussians);
            const auto keep_rows = detail::build_mesh2splat_uniform_cap_keep_mask(
                uncapped_count,
                target_count);
            auto keep_tensor = Tensor::from_vector(
                                   keep_rows,
                                   {keep_rows.size()},
                                   Device::CPU)
                                   .to(splat->means().device());
            auto capped_splat = core::extract_by_mask(*splat, keep_tensor);
            if (capped_splat.size() != target_count) {
                return std::unexpected("mesh2splat hard global cap row selection failed");
            }
            splat = std::make_unique<SplatData>(std::move(capped_splat));

            if (input.face_statistics) {
                if (input.face_statistics->raw_render_face_per_gaussian.size() != uncapped_count) {
                    return std::unexpected(
                        "mesh2splat hard-cap raw render-face statistics shape mismatch");
                }
                std::vector<int32_t> capped_raw_faces;
                capped_raw_faces.reserve(target_count);
                std::fill(
                    input.face_statistics->gaussian_count_per_render_face.begin(),
                    input.face_statistics->gaussian_count_per_render_face.end(),
                    size_t{0});
                for (size_t row = 0; row < uncapped_count; ++row) {
                    if (!keep_rows[row]) {
                        continue;
                    }
                    const int32_t raw_face =
                        input.face_statistics->raw_render_face_per_gaussian[row];
                    if (raw_face < 0 ||
                        static_cast<size_t>(raw_face) >=
                            input.face_statistics->gaussian_count_per_render_face.size()) {
                        return std::unexpected(
                            "mesh2splat hard-cap encountered an invalid raw render face id");
                    }
                    capped_raw_faces.push_back(raw_face);
                    ++input.face_statistics->gaussian_count_per_render_face[
                        static_cast<size_t>(raw_face)];
                }
                input.face_statistics->raw_render_face_per_gaussian =
                    std::move(capped_raw_faces);
                input.face_statistics->gaussian_count = target_count;
            }
            LOG_INFO("mesh2splat: enforced hard global cap with uniform full-output sampling ({} -> {} gaussians)",
                     uncapped_count,
                     target_count);
        }

        // Log SplatData statistics
        if (core::Logger::get().is_enabled(core::LogLevel::Debug)) {
            auto sh0_cpu = splat->sh0_raw().to(Device::CPU);
            auto scale_cpu = splat->scaling_raw().to(Device::CPU);
            LOG_DEBUG("mesh2splat: SplatData sh0  min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
                      sh0_cpu.slice(2, 0, 1).min().item(), sh0_cpu.slice(2, 1, 2).min().item(),
                      sh0_cpu.slice(2, 2, 3).min().item(),
                      sh0_cpu.slice(2, 0, 1).max().item(), sh0_cpu.slice(2, 1, 2).max().item(),
                      sh0_cpu.slice(2, 2, 3).max().item());
            LOG_DEBUG("mesh2splat: SplatData scale(log) min=({:.3f},{:.3f},{:.3f}) max=({:.3f},{:.3f},{:.3f})",
                      scale_cpu.slice(1, 0, 1).min().item(), scale_cpu.slice(1, 1, 2).min().item(),
                      scale_cpu.slice(1, 2, 3).min().item(),
                      scale_cpu.slice(1, 0, 1).max().item(), scale_cpu.slice(1, 1, 2).max().item(),
                      scale_cpu.slice(1, 2, 3).max().item());
        }

        if (!report(1.0f, "Complete"))
            return std::unexpected("Cancelled");

        return splat;
    }

    std::expected<std::unique_ptr<SplatData>, std::string>
    mesh_to_splat(const MeshData& mesh,
                  const Mesh2SplatOptions& options,
                  Mesh2SplatProgressCallback progress) {
        return mesh_to_splat(Mesh2SplatInput{.render_mesh = mesh}, options, std::move(progress));
    }

} // namespace lfs::rendering
