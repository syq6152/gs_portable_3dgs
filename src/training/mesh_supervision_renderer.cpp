/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_supervision_renderer.hpp"

#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "depth_normal_utils.hpp"
#include "kernels/mesh_rasterization.hpp"
#include "training/kernels/grad_alpha.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr uint32_t MESH_TENSOR_FILE_MAGIC = 0x4D534754u; // "MSGT"
        constexpr uint32_t MESH_TENSOR_FILE_VERSION = 1u;
        constexpr uint32_t MESH_TENSOR_FILE_DTYPE_F32 = 1u;

#pragma pack(push, 1)
        struct MeshTensorFileHeader {
            uint32_t magic;
            uint32_t version;
            uint32_t ndim;
            uint32_t dtype;
            uint64_t shape0;
            uint64_t shape1;
            uint64_t shape2;
        };
#pragma pack(pop)

        // PreparedMesh is now defined in mesh_supervision_renderer.hpp

        std::expected<void, std::string> write_all_bytes(
            const std::filesystem::path& path,
            const MeshTensorFileHeader& header,
            const lfs::core::Tensor& tensor_cpu) {

            std::ofstream out;
            if (!lfs::core::open_file_for_write(path, std::ios::binary | std::ios::trunc, out)) {
                return std::unexpected(std::format("Failed to open '{}' for writing", lfs::core::path_to_utf8(path)));
            }

            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            if (!out.good()) {
                return std::unexpected(std::format("Failed to write header to '{}'", lfs::core::path_to_utf8(path)));
            }

            out.write(reinterpret_cast<const char*>(tensor_cpu.ptr<float>()), static_cast<std::streamsize>(tensor_cpu.bytes()));
            if (!out.good()) {
                return std::unexpected(std::format("Failed to write payload to '{}'", lfs::core::path_to_utf8(path)));
            }

            return {};
        }

        inline void normalize3(float& x, float& y, float& z) {
            const float n2 = x * x + y * y + z * z;
            if (!(n2 > 1e-20f) || !std::isfinite(n2)) {
                x = 0.0f;
                y = 0.0f;
                z = -1.0f;
                return;
            }
            const float inv = 1.0f / std::sqrt(n2);
            x *= inv;
            y *= inv;
            z *= inv;
        }

        std::string sanitize_token(const std::string& raw) {
            std::string out;
            out.reserve(raw.size());
            for (const unsigned char ch : raw) {
                if ((ch >= 'a' && ch <= 'z') ||
                    (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') ||
                    ch == '_' || ch == '-' || ch == '.') {
                    out.push_back(static_cast<char>(ch));
                } else {
                    out.push_back('_');
                }
            }
            if (out.empty()) {
                out = "camera";
            }
            return out;
        }

        std::string camera_cache_stem(const lfs::core::Camera& cam) {
            std::string image_stem = std::filesystem::path(cam.image_name()).stem().string();
            image_stem = sanitize_token(image_stem);
            return std::to_string(cam.uid()) + "_" + image_stem;
        }

        std::vector<float> compute_vertex_normals_cpu(
            lfs::core::Tensor& vertices_cpu,
            lfs::core::Tensor& indices_cpu) {

            const int64_t num_vertices = static_cast<int64_t>(vertices_cpu.shape()[0]);
            const int64_t num_faces = static_cast<int64_t>(indices_cpu.shape()[0]);

            std::vector<float> normals(static_cast<size_t>(num_vertices) * 3, 0.0f);
            auto vacc = vertices_cpu.accessor<float, 2>();
            auto iacc = indices_cpu.accessor<int32_t, 2>();

            for (int64_t f = 0; f < num_faces; ++f) {
                const int32_t i0 = iacc(static_cast<int>(f), 0);
                const int32_t i1 = iacc(static_cast<int>(f), 1);
                const int32_t i2 = iacc(static_cast<int>(f), 2);

                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices) {
                    continue;
                }

                const float x0 = vacc(i0, 0);
                const float y0 = vacc(i0, 1);
                const float z0 = vacc(i0, 2);
                const float x1 = vacc(i1, 0);
                const float y1 = vacc(i1, 1);
                const float z1 = vacc(i1, 2);
                const float x2 = vacc(i2, 0);
                const float y2 = vacc(i2, 1);
                const float z2 = vacc(i2, 2);

                const float e1x = x1 - x0;
                const float e1y = y1 - y0;
                const float e1z = z1 - z0;
                const float e2x = x2 - x0;
                const float e2y = y2 - y0;
                const float e2z = z2 - z0;

                float nx = e1y * e2z - e1z * e2y;
                float ny = e1z * e2x - e1x * e2z;
                float nz = e1x * e2y - e1y * e2x;

                const float n2 = nx * nx + ny * ny + nz * nz;
                if (!(n2 > 1e-20f) || !std::isfinite(n2)) {
                    continue;
                }

                const float inv = 1.0f / std::sqrt(n2);
                nx *= inv;
                ny *= inv;
                nz *= inv;

                normals[static_cast<size_t>(i0) * 3 + 0] += nx;
                normals[static_cast<size_t>(i0) * 3 + 1] += ny;
                normals[static_cast<size_t>(i0) * 3 + 2] += nz;
                normals[static_cast<size_t>(i1) * 3 + 0] += nx;
                normals[static_cast<size_t>(i1) * 3 + 1] += ny;
                normals[static_cast<size_t>(i1) * 3 + 2] += nz;
                normals[static_cast<size_t>(i2) * 3 + 0] += nx;
                normals[static_cast<size_t>(i2) * 3 + 1] += ny;
                normals[static_cast<size_t>(i2) * 3 + 2] += nz;
            }

            for (int64_t v = 0; v < num_vertices; ++v) {
                float nx = normals[static_cast<size_t>(v) * 3 + 0];
                float ny = normals[static_cast<size_t>(v) * 3 + 1];
                float nz = normals[static_cast<size_t>(v) * 3 + 2];
                normalize3(nx, ny, nz);
                normals[static_cast<size_t>(v) * 3 + 0] = nx;
                normals[static_cast<size_t>(v) * 3 + 1] = ny;
                normals[static_cast<size_t>(v) * 3 + 2] = nz;
            }

            return normals;
        }

        std::expected<PreparedMesh, std::string> prepare_visible_meshes(
            const std::vector<lfs::core::Scene::VisibleMesh>& visible_meshes) {

            std::vector<float> all_world_vertices_xyz;
            std::vector<float> all_world_normals_xyz;
            std::vector<int32_t> all_triangle_indices;

            for (const auto& vm : visible_meshes) {
                if (!vm.mesh) {
                    continue;
                }
                const auto& mesh = *vm.mesh;
                if (!mesh.vertices.is_valid() || !mesh.indices.is_valid() ||
                    mesh.vertex_count() <= 0 || mesh.face_count() <= 0) {
                    continue;
                }

                auto vertices_cpu = mesh.vertices.to(lfs::core::Device::CPU)
                                        .to(lfs::core::DataType::Float32)
                                        .contiguous();
                auto indices_cpu = mesh.indices.to(lfs::core::Device::CPU)
                                       .to(lfs::core::DataType::Int32)
                                       .contiguous();

                if (vertices_cpu.ndim() != 2 || vertices_cpu.shape()[1] != 3 ||
                    indices_cpu.ndim() != 2 || indices_cpu.shape()[1] != 3) {
                    return std::unexpected("Mesh tensor layout mismatch for supervision rasterization");
                }

                const int64_t num_vertices = static_cast<int64_t>(vertices_cpu.shape()[0]);
                const int64_t num_faces = static_cast<int64_t>(indices_cpu.shape()[0]);

                lfs::core::Tensor normals_cpu;
                const bool use_mesh_normals =
                    mesh.has_normals() &&
                    mesh.normals.ndim() == 2 &&
                    mesh.normals.shape()[1] == 3 &&
                    mesh.normals.shape()[0] == vertices_cpu.shape()[0];

                const float* normals_ptr = nullptr;
                if (use_mesh_normals) {
                    normals_cpu = mesh.normals.to(lfs::core::Device::CPU)
                                     .to(lfs::core::DataType::Float32)
                                     .contiguous();
                    normals_ptr = normals_cpu.ptr<float>();
                } else {
                    // Compute vertex normals on GPU: upload local vertices+indices,
                    // run two-pass kernel (accumulate face normals + normalize), download result
                    auto verts_cuda = vertices_cpu.to(lfs::core::Device::CUDA).contiguous();
                    auto inds_cuda = indices_cpu.to(lfs::core::Device::CUDA).contiguous();
                    auto normals_cuda = lfs::core::Tensor::zeros(
                        {static_cast<size_t>(num_vertices), size_t{3}},
                        lfs::core::Device::CUDA,
                        lfs::core::DataType::Float32);
                    kernels::launch_mesh_compute_vertex_normals(
                        verts_cuda.ptr<float>(),
                        inds_cuda.ptr<int32_t>(),
                        static_cast<int>(num_vertices),
                        static_cast<int>(num_faces),
                        normals_cuda.ptr<float>(),
                        normals_cuda.stream());
                    normals_cpu = normals_cuda.to(lfs::core::Device::CPU)
                                     .to(lfs::core::DataType::Float32)
                                     .contiguous();
                    normals_ptr = normals_cpu.ptr<float>();
                }

                const size_t vertex_offset = all_world_vertices_xyz.size() / 3;
                all_world_vertices_xyz.resize(all_world_vertices_xyz.size() + static_cast<size_t>(num_vertices) * 3);
                all_world_normals_xyz.resize(all_world_normals_xyz.size() + static_cast<size_t>(num_vertices) * 3);

                const glm::mat3 linear = glm::mat3(vm.transform);
                const float det = glm::determinant(linear);
                const glm::mat3 normal_transform =
                    std::abs(det) > 1e-12f ? glm::transpose(glm::inverse(linear)) : linear;

                auto vacc = vertices_cpu.accessor<float, 2>();
                for (int64_t v = 0; v < num_vertices; ++v) {
                    const glm::vec4 wp = vm.transform * glm::vec4(
                        vacc(static_cast<int>(v), 0),
                        vacc(static_cast<int>(v), 1),
                        vacc(static_cast<int>(v), 2),
                        1.0f);

                    const size_t dst = (vertex_offset + static_cast<size_t>(v)) * 3;
                    const size_t src = static_cast<size_t>(v) * 3;
                    all_world_vertices_xyz[dst + 0] = wp.x;
                    all_world_vertices_xyz[dst + 1] = wp.y;
                    all_world_vertices_xyz[dst + 2] = wp.z;

                    glm::vec3 wn(
                        normals_ptr[src + 0],
                        normals_ptr[src + 1],
                        normals_ptr[src + 2]);
                    wn = normal_transform * wn;

                    float nnx = wn.x;
                    float nny = wn.y;
                    float nnz = wn.z;
                    normalize3(nnx, nny, nnz);
                    all_world_normals_xyz[dst + 0] = nnx;
                    all_world_normals_xyz[dst + 1] = nny;
                    all_world_normals_xyz[dst + 2] = nnz;
                }

                auto iacc = indices_cpu.accessor<int32_t, 2>();
                for (int64_t f = 0; f < num_faces; ++f) {
                    const int32_t i0 = iacc(static_cast<int>(f), 0);
                    const int32_t i1 = iacc(static_cast<int>(f), 1);
                    const int32_t i2 = iacc(static_cast<int>(f), 2);

                    if (i0 < 0 || i1 < 0 || i2 < 0 ||
                        i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices) {
                        continue;
                    }

                    all_triangle_indices.push_back(static_cast<int32_t>(vertex_offset + static_cast<size_t>(i0)));
                    all_triangle_indices.push_back(static_cast<int32_t>(vertex_offset + static_cast<size_t>(i1)));
                    all_triangle_indices.push_back(static_cast<int32_t>(vertex_offset + static_cast<size_t>(i2)));
                }
            }

            if (all_triangle_indices.empty() || all_world_vertices_xyz.empty()) {
                return std::unexpected("Mesh supervision cache render skipped: mesh tensors are empty");
            }

            PreparedMesh prepared;
            prepared.vertex_count = static_cast<int>(all_world_vertices_xyz.size() / 3);
            prepared.face_count = static_cast<int>(all_triangle_indices.size() / 3);

            auto vertices_cpu = lfs::core::Tensor::from_vector(
                all_world_vertices_xyz,
                {static_cast<size_t>(prepared.vertex_count), size_t{3}},
                lfs::core::Device::CPU);
            auto normals_cpu = lfs::core::Tensor::from_vector(
                all_world_normals_xyz,
                {static_cast<size_t>(prepared.vertex_count), size_t{3}},
                lfs::core::Device::CPU);
            auto indices_cpu = lfs::core::Tensor::from_vector(
                all_triangle_indices,
                {static_cast<size_t>(prepared.face_count), size_t{3}},
                lfs::core::Device::CPU);

            prepared.world_vertices_xyz = vertices_cpu.to(lfs::core::Device::CUDA)
                                              .to(lfs::core::DataType::Float32)
                                              .contiguous();
            prepared.world_vertex_normals_xyz = normals_cpu.to(lfs::core::Device::CUDA)
                                                    .to(lfs::core::DataType::Float32)
                                                    .contiguous();
            prepared.triangle_indices = indices_cpu.to(lfs::core::Device::CUDA)
                                            .to(lfs::core::DataType::Int32)
                                            .contiguous();

            return prepared;
        }

        struct RasterizeResult {
            lfs::core::Tensor depth;
            lfs::core::Tensor normal;
            lfs::core::Tensor triangle_id; // optional, [H,W] int32, -1 invalid
        };

        std::expected<RasterizeResult, std::string>
        rasterize_depth_normal_for_camera_gpu(
            const PreparedMesh& mesh,
            lfs::core::Camera& cam,
            const MeshSupervisionOutputRequest& output_request =
                MeshSupervisionOutputRequest::depth_and_normal()) {

            const int width = cam.image_width();
            const int height = cam.image_height();
            if (width <= 0 || height <= 0) {
                return std::unexpected("Mesh supervision rasterization failed: invalid camera resolution");
            }
            if (!mesh.world_vertices_xyz.is_valid() || mesh.world_vertices_xyz.is_empty() ||
                !mesh.triangle_indices.is_valid() || mesh.triangle_indices.is_empty() ||
                mesh.vertex_count <= 0 || mesh.face_count <= 0) {
                return std::unexpected("Mesh supervision rasterization failed: invalid prepared mesh");
            }
            if (output_request.normal &&
                (!mesh.world_vertex_normals_xyz.is_valid() || mesh.world_vertex_normals_xyz.is_empty())) {
                return std::unexpected(
                    "Mesh supervision rasterization failed: normal output requires prepared vertex normals");
            }

            auto depth = lfs::core::Tensor::zeros(
                {size_t{1}, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            lfs::core::Tensor normal;
            if (output_request.normal) {
                normal = lfs::core::Tensor::zeros(
                    {size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
            }
            auto depth_keys = lfs::core::Tensor::empty(
                {static_cast<size_t>(width) * static_cast<size_t>(height)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Int64);
            auto cam_vertices = lfs::core::Tensor::empty(
                {static_cast<size_t>(mesh.vertex_count), size_t{3}},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
            lfs::core::Tensor cam_normals;
            if (output_request.normal) {
                cam_normals = lfs::core::Tensor::empty(
                    {static_cast<size_t>(mesh.vertex_count), size_t{3}},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
            }

            auto w2c = cam.world_view_transform()
                           .squeeze(0)
                           .to(lfs::core::Device::CUDA)
                           .to(lfs::core::DataType::Float32)
                           .contiguous();
            const float* w2c_ptr = w2c.ptr<float>();
            if (!w2c_ptr) {
                return std::unexpected("Mesh supervision rasterization failed: missing camera transform");
            }

            const auto [fx, fy, cx, cy] = cam.get_intrinsics();
            constexpr float NEAR_Z = 1e-4f;
            const cudaStream_t stream = depth.stream();

            kernels::launch_mesh_depth_key_init(
                depth_keys.ptr<int64_t>(),
                width * height,
                stream);

            kernels::launch_mesh_transform_vertices_normals(
                mesh.world_vertices_xyz.ptr<float>(),
                output_request.normal ? mesh.world_vertex_normals_xyz.ptr<float>() : nullptr,
                mesh.vertex_count,
                w2c_ptr,
                cam_vertices.ptr<float>(),
                output_request.normal ? cam_normals.ptr<float>() : nullptr,
                stream);

            kernels::launch_mesh_rasterize_triangles(
                cam_vertices.ptr<float>(),
                output_request.normal ? cam_normals.ptr<float>() : nullptr,
                mesh.triangle_indices.ptr<int32_t>(),
                mesh.face_count,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                NEAR_Z,
                depth_keys.ptr<int64_t>(),
                stream);

            kernels::launch_mesh_finalize_depth_normal(
                depth_keys.ptr<int64_t>(),
                cam_vertices.ptr<float>(),
                output_request.normal ? cam_normals.ptr<float>() : nullptr,
                mesh.triangle_indices.ptr<int32_t>(),
                mesh.face_count,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                depth.ptr<float>(),
                output_request.normal ? normal.ptr<float>() : nullptr,
                stream);

            RasterizeResult result;
            result.depth = std::move(depth);
            result.normal = std::move(normal);

            if (output_request.triangle_id) {
                auto tri_id = lfs::core::Tensor::empty(
                    {static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Int32);
                kernels::launch_mesh_extract_tri_id(
                    depth_keys.ptr<int64_t>(),
                    mesh.face_count,
                    width,
                    height,
                    tri_id.ptr<int32_t>(),
                    stream);
                result.triangle_id = std::move(tri_id);
            }

            return result;
        }

        lfs::core::Tensor to_mask_hw_cuda(const lfs::core::Tensor& mask) {
            if (!mask.is_valid() || mask.is_empty()) {
                return {};
            }

            lfs::core::Tensor m = mask;
            if (m.ndim() == 3 && m.shape()[0] == 1) {
                m = m.squeeze(0);
            } else if (m.ndim() == 3 && m.shape()[2] == 1) {
                m = m.squeeze(2);
            }
            if (m.ndim() != 2) {
                return {};
            }

            return m.to(lfs::core::Device::CUDA)
                .to(lfs::core::DataType::Float32)
                .contiguous();
        }

        lfs::core::Tensor resize_mask_to_resolution_cuda(
            const lfs::core::Tensor& mask,
            const int dst_h,
            const int dst_w) {

            auto src = to_mask_hw_cuda(mask);
            if (!src.is_valid() || src.is_empty()) {
                return {};
            }

            const int src_h = static_cast<int>(src.shape()[0]);
            const int src_w = static_cast<int>(src.shape()[1]);
            if (src_h == dst_h && src_w == dst_w) {
                return src;
            }

            auto dst = lfs::core::Tensor::empty(
                {size_t{1}, static_cast<size_t>(dst_h), static_cast<size_t>(dst_w)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);

            kernels::launch_bilinear_resize_chw(
                src.ptr<float>(),
                dst.ptr<float>(),
                1,
                src_h,
                src_w,
                dst_h,
                dst_w,
                dst.stream());

            return dst.reshape({dst_h, dst_w});
        }

        lfs::core::Tensor load_alpha_mask_like_eval(
            lfs::core::Camera& cam,
            const lfs::core::param::TrainingParameters& params) {

            auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                cam.image_path(),
                params.dataset.resize_factor,
                params.dataset.max_width);

            if (!img_data || channels != 4) {
                if (img_data) {
                    lfs::core::free_image(img_data);
                }
                return {};
            }

            const int64_t n = static_cast<int64_t>(width) * static_cast<int64_t>(height);
            std::vector<float> alpha_mask(static_cast<size_t>(n), 0.0f);

            for (int64_t i = 0; i < n; ++i) {
                alpha_mask[static_cast<size_t>(i)] = static_cast<float>(img_data[i * 4 + 3]) / 255.0f;
            }
            lfs::core::free_image(img_data);

            if (params.optimization.invert_masks) {
                for (float& v : alpha_mask) {
                    v = 1.0f - v;
                }
            }

            if (params.optimization.mask_threshold > 0.0f && params.optimization.mask_threshold < 1.0f) {
                for (float& v : alpha_mask) {
                    v = (v >= params.optimization.mask_threshold) ? 1.0f : 0.0f;
                }
            }

            auto mask = lfs::core::Tensor::from_vector(
                alpha_mask,
                {static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CPU);

            if (cam.is_undistort_prepared()) {
                const auto scaled = lfs::core::scale_undistort_params(
                    cam.undistort_params(),
                    width,
                    height);
                mask = lfs::core::undistort_mask(mask.to(lfs::core::Device::CUDA), scaled, nullptr)
                           .to(lfs::core::Device::CPU)
                           .to(lfs::core::DataType::Float32)
                           .contiguous();
            }

            return mask;
        }

        lfs::core::Tensor load_camera_mask_for_supervision(
            lfs::core::Camera& cam,
            const lfs::core::param::TrainingParameters& params,
            bool& masked_camera) {

            masked_camera = false;

            if (params.optimization.use_alpha_as_mask && cam.has_alpha()) {
                masked_camera = true;
                return load_alpha_mask_like_eval(cam, params);
            }

            if (cam.has_mask()) {
                masked_camera = true;
                auto mask = cam.load_and_get_mask(
                    params.dataset.resize_factor,
                    params.dataset.max_width,
                    params.optimization.invert_masks,
                    params.optimization.mask_threshold);
                cam.clear_cached_mask();
                return mask;
            }

            return {};
        }

        std::filesystem::path write_manifest(
            const std::filesystem::path& cache_dir,
            const nlohmann::json& manifest) {

            const std::filesystem::path manifest_path = cache_dir / "manifest.json";
            std::ofstream out;
            if (!lfs::core::open_file_for_write(manifest_path, std::ios::out | std::ios::trunc, out)) {
                throw std::runtime_error("Failed to open mesh supervision manifest for writing");
            }
            out << manifest.dump(2);
            return manifest_path;
        }
    } // namespace

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        lfs::core::Scene& scene,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionOutputRequest& output_request,
        const bool apply_camera_mask) {

        const auto visible_meshes = scene.getVisibleMeshes();
        if (visible_meshes.empty()) {
            return std::unexpected("Mesh supervision render skipped: no visible mesh nodes in scene");
        }

        auto prepared_mesh_result = prepare_visible_meshes(visible_meshes);
        if (!prepared_mesh_result) {
            return std::unexpected(prepared_mesh_result.error());
        }
        PreparedMesh prepared_mesh = std::move(*prepared_mesh_result);
        return render_mesh_supervision_targets_for_camera(
            prepared_mesh,
            camera,
            params,
            output_request,
            apply_camera_mask);
    }

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        lfs::core::Scene& scene,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const bool apply_camera_mask,
        const bool request_triangle_id) {
        return render_mesh_supervision_targets_for_camera(
            scene,
            camera,
            params,
            MeshSupervisionOutputRequest::depth_and_normal(request_triangle_id),
            apply_camera_mask);
    }

    std::expected<PreparedMesh, std::string> prepare_mesh_geometry(
        lfs::core::Scene& scene) {

        const auto visible_meshes = scene.getVisibleMeshes();
        if (visible_meshes.empty()) {
            return std::unexpected("Mesh supervision: no visible mesh nodes in scene");
        }

        return prepare_visible_meshes(visible_meshes);
    }

    std::expected<PreparedMesh, std::string> prepare_mesh_geometry(
        const lfs::core::MeshData& mesh) {
        return prepare_visible_meshes({lfs::core::Scene::VisibleMesh{
            .mesh = &mesh,
            .transform = glm::mat4(1.0f)}});
    }

    ProjectMaskMeshSource select_project_mask_mesh_source(
        const lfs::core::Scene& scene) {
        ProjectMaskMeshSource source;
        if (const auto* mesh_override = scene.getProjectMaskMeshOverride()) {
            source.mesh_override = mesh_override;
        } else {
            source.use_visible_scene_meshes = scene.hasVisibleMeshes();
        }
        // Independent of which mesh supplies coverage: external_pointcloud mode leaves the
        // mesh override unset (the original mesh is the base) and only adds the cloud.
        if (scene.hasProjectMaskPointCloudOverride()) {
            source.point_cloud_override = scene.getProjectMaskPointCloudOverride();
            source.point_cloud_spacing = scene.getProjectMaskPointCloudSpacing();
        }
        return source;
    }

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        const PreparedMesh& prepared_mesh,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionOutputRequest& output_request,
        const bool apply_camera_mask) {

        if (!camera.has_pyramid_intrinsics_override()) {
            camera.load_image_size(params.dataset.resize_factor, params.dataset.max_width);
        }

        const int width = camera.image_width();
        const int height = camera.image_height();
        if (width <= 0 || height <= 0) {
            return std::unexpected("Mesh supervision render failed: invalid camera resolution");
        }

        auto raster_result = rasterize_depth_normal_for_camera_gpu(prepared_mesh, camera, output_request);
        if (!raster_result) {
            return std::unexpected(raster_result.error());
        }

        MeshSupervisionTargets targets;
        targets.depth = std::move(raster_result->depth);
        targets.normal = std::move(raster_result->normal);
        targets.triangle_id = std::move(raster_result->triangle_id);

        if (apply_camera_mask) {
            auto mask = load_camera_mask_for_supervision(camera, params, targets.masked_camera);
            if (mask.is_valid() && !mask.is_empty()) {
                auto mask_cuda = resize_mask_to_resolution_cuda(mask, height, width);
                if (mask_cuda.is_valid() && !mask_cuda.is_empty()) {
                    kernels::launch_mesh_apply_mask_depth_normal(
                        targets.depth.ptr<float>(),
                        output_request.normal ? targets.normal.ptr<float>() : nullptr,
                        mask_cuda.ptr<float>(),
                        width,
                        height,
                        targets.depth.stream());
                    if (targets.triangle_id.is_valid() && !targets.triangle_id.is_empty()) {
                        kernels::launch_mesh_apply_mask_tri_id(
                            targets.triangle_id.ptr<int32_t>(),
                            mask_cuda.ptr<float>(),
                            width,
                            height,
                            targets.triangle_id.stream());
                    }
                    targets.mask = std::move(mask_cuda);
                }
            }
        }

        if (!targets.depth.is_valid() ||
            (output_request.normal && !targets.normal.is_valid()) ||
            (output_request.triangle_id && !targets.triangle_id.is_valid())) {
            return std::unexpected("Mesh supervision render produced invalid CUDA tensors");
        }

        return targets;
    }

    std::expected<MeshSupervisionTargets, std::string> render_mesh_supervision_targets_for_camera(
        const PreparedMesh& prepared_mesh,
        lfs::core::Camera& camera,
        const lfs::core::param::TrainingParameters& params,
        const bool apply_camera_mask,
        const bool request_triangle_id) {
        return render_mesh_supervision_targets_for_camera(
            prepared_mesh,
            camera,
            params,
            MeshSupervisionOutputRequest::depth_and_normal(request_triangle_id),
            apply_camera_mask);
    }

    std::expected<void, std::string> write_mesh_supervision_tensor_file(
        const std::filesystem::path& path,
        const lfs::core::Tensor& tensor) {

        if (!tensor.is_valid() || tensor.is_empty()) {
            return std::unexpected(std::format("Invalid tensor for '{}'", lfs::core::path_to_utf8(path)));
        }

        auto tensor_cpu = tensor.to(lfs::core::Device::CPU)
                              .to(lfs::core::DataType::Float32)
                              .contiguous();
        const size_t ndim = tensor_cpu.ndim();
        if (ndim != 2 && ndim != 3) {
            return std::unexpected(std::format(
                "Unsupported tensor rank {} for '{}' (expected 2D or 3D)",
                ndim,
                lfs::core::path_to_utf8(path)));
        }

        const auto& shape = tensor_cpu.shape();
        MeshTensorFileHeader header{};
        header.magic = MESH_TENSOR_FILE_MAGIC;
        header.version = MESH_TENSOR_FILE_VERSION;
        header.ndim = static_cast<uint32_t>(ndim);
        header.dtype = MESH_TENSOR_FILE_DTYPE_F32;
        header.shape0 = static_cast<uint64_t>(shape[0]);
        header.shape1 = static_cast<uint64_t>(shape[1]);
        header.shape2 = (ndim == 3) ? static_cast<uint64_t>(shape[2]) : 1ull;

        return write_all_bytes(path, header, tensor_cpu);
    }

    std::expected<lfs::core::Tensor, std::string> decode_mesh_supervision_tensor_bytes(
        const std::vector<uint8_t>& bytes) {

        if (bytes.size() < sizeof(MeshTensorFileHeader)) {
            return std::unexpected("Mesh tensor decode failed: payload is smaller than file header");
        }

        MeshTensorFileHeader header{};
        std::memcpy(&header, bytes.data(), sizeof(MeshTensorFileHeader));

        if (header.magic != MESH_TENSOR_FILE_MAGIC) {
            return std::unexpected("Mesh tensor decode failed: invalid magic");
        }
        if (header.version != MESH_TENSOR_FILE_VERSION) {
            return std::unexpected(std::format("Mesh tensor decode failed: unsupported version {}", header.version));
        }
        if (header.dtype != MESH_TENSOR_FILE_DTYPE_F32) {
            return std::unexpected(std::format("Mesh tensor decode failed: unsupported dtype {}", header.dtype));
        }
        if (header.ndim != 2 && header.ndim != 3) {
            return std::unexpected(std::format("Mesh tensor decode failed: invalid ndim {}", header.ndim));
        }

        std::vector<size_t> shape;
        shape.reserve(static_cast<size_t>(header.ndim));
        shape.push_back(static_cast<size_t>(header.shape0));
        shape.push_back(static_cast<size_t>(header.shape1));
        if (header.ndim == 3) {
            shape.push_back(static_cast<size_t>(header.shape2));
        }

        size_t numel = 1;
        for (const size_t dim : shape) {
            if (dim == 0) {
                return std::unexpected("Mesh tensor decode failed: zero-sized dimension");
            }
            if (numel > (std::numeric_limits<size_t>::max() / dim)) {
                return std::unexpected("Mesh tensor decode failed: tensor size overflow");
            }
            numel *= dim;
        }

        const size_t payload_bytes = numel * sizeof(float);
        const size_t expected_size = sizeof(MeshTensorFileHeader) + payload_bytes;
        if (bytes.size() != expected_size) {
            return std::unexpected(std::format(
                "Mesh tensor decode failed: payload size mismatch (expected {}, got {})",
                expected_size,
                bytes.size()));
        }

        auto tensor = lfs::core::Tensor::empty_unpinned(
            lfs::core::TensorShape(shape),
            lfs::core::DataType::Float32);
        std::memcpy(
            tensor.ptr<float>(),
            bytes.data() + sizeof(MeshTensorFileHeader),
            payload_bytes);

        return tensor.contiguous();
    }

    std::filesystem::path mesh_supervision_cache_dir(
        const std::filesystem::path& output_root,
        const std::string& cache_tag) {
        auto cache_dir = output_root / "mesh_supervision";
        if (!cache_tag.empty()) {
            cache_dir /= sanitize_token(cache_tag);
        }
        return cache_dir;
    }

    std::expected<MeshSupervisionRenderResult, std::string> render_mesh_supervision_cache(
        lfs::core::Scene& scene,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const lfs::core::param::TrainingParameters& params,
        const MeshSupervisionRenderOptions& options) {

        if (cameras.empty()) {
            return std::unexpected("Mesh supervision cache render skipped: no cameras available");
        }

        const auto visible_meshes = scene.getVisibleMeshes();
        if (visible_meshes.empty()) {
            return std::unexpected("Mesh supervision cache render skipped: no visible mesh nodes in scene");
        }

        auto prepared_mesh_result = prepare_visible_meshes(visible_meshes);
        if (!prepared_mesh_result) {
            return std::unexpected(prepared_mesh_result.error());
        }
        PreparedMesh prepared_mesh = std::move(*prepared_mesh_result);

        MeshSupervisionRenderResult result;
        auto& stats = result.stats;
        stats.total_cameras = cameras.size();
        stats.cache_dir = mesh_supervision_cache_dir(options.output_root, options.cache_tag);
        const bool save_preview_png = options.save_preview_png;

        try {
            std::filesystem::create_directories(stats.cache_dir);

            nlohmann::json manifest;
            manifest["version"] = 5;
            manifest["cache_dir"] = lfs::core::path_to_utf8(stats.cache_dir);
            manifest["camera_count"] = cameras.size();
            manifest["cache_tag"] = options.cache_tag;
            manifest["storage"] = "lfgt_f32";
            manifest["save_preview_png"] = save_preview_png;
            manifest["rasterizer"] = "gpu_cuda_triangle";
            manifest["mask_options"] = {
                {"use_alpha_as_mask", params.optimization.use_alpha_as_mask},
                {"invert_masks", params.optimization.invert_masks},
                {"mask_threshold", params.optimization.mask_threshold}};
            manifest["cameras"] = nlohmann::json::array();

            for (const auto& cam_ptr : cameras) {
                if (!cam_ptr) {
                    continue;
                }

                auto& cam = *cam_ptr;
                // Pyramid GT refresh temporarily overrides camera intrinsics + image size.
                // Keep that override intact so raster size and intrinsics stay consistent.
                if (!cam.has_pyramid_intrinsics_override()) {
                    cam.load_image_size(params.dataset.resize_factor, params.dataset.max_width);
                }
                const int width = cam.image_width();
                const int height = cam.image_height();
                if (width <= 0 || height <= 0) {
                    LOG_WARN("Skipping mesh supervision cache for camera '{}' due to invalid image size {}x{}",
                             cam.image_name(), width, height);
                    continue;
                }

                MeshSupervisionCameraCacheEntry cache_entry;
                cache_entry.camera_uid = static_cast<uint32_t>(cam.uid());
                cache_entry.image_name = cam.image_name();
                cache_entry.width = width;
                cache_entry.height = height;

                const std::string cam_stem = camera_cache_stem(cam);
                const auto depth_path = stats.cache_dir / (cam_stem + "_depth.lfgt");
                const auto normal_path = stats.cache_dir / (cam_stem + "_normal.lfgt");
                const auto preview_path = stats.cache_dir / (cam_stem + "_preview_depth_normal.png");

                bool masked_camera = false;
                auto raster_result = rasterize_depth_normal_for_camera_gpu(prepared_mesh, cam);
                if (!raster_result) {
                    return std::unexpected(raster_result.error());
                }
                auto depth = std::move(raster_result->depth);
                auto normal = std::move(raster_result->normal);

                auto mask = load_camera_mask_for_supervision(cam, params, masked_camera);
                if (mask.is_valid() && !mask.is_empty()) {
                    auto mask_cuda = resize_mask_to_resolution_cuda(mask, height, width);
                    if (mask_cuda.is_valid() && !mask_cuda.is_empty()) {
                        kernels::launch_mesh_apply_mask_depth_normal(
                            depth.ptr<float>(),
                            normal.ptr<float>(),
                            mask_cuda.ptr<float>(),
                            width,
                            height,
                            depth.stream());
                        mask = std::move(mask_cuda);
                    } else {
                        mask = {};
                    }
                } else {
                    mask = {};
                }

                if (masked_camera) {
                    stats.masked_cameras++;
                }

                if (!depth.is_valid() || !normal.is_valid()) {
                    return std::unexpected("Mesh supervision rasterization produced invalid CUDA tensors");
                }

                auto depth_cpu = depth.to(lfs::core::Device::CPU)
                                     .to(lfs::core::DataType::Float32)
                                     .contiguous();
                auto normal_cpu = normal.to(lfs::core::Device::CPU)
                                      .to(lfs::core::DataType::Float32)
                                      .contiguous();

                const float* depth_ptr = depth_cpu.ptr<float>();
                const size_t numel = static_cast<size_t>(width) * static_cast<size_t>(height);
                const size_t valid_depth_pixels = std::count_if(
                    depth_ptr,
                    depth_ptr + numel,
                    [](const float z) { return std::isfinite(z) && z > 0.0f; });

                if (auto write_depth = write_mesh_supervision_tensor_file(depth_path, depth_cpu); !write_depth) {
                    return std::unexpected(write_depth.error());
                }
                if (auto write_normal = write_mesh_supervision_tensor_file(normal_path, normal_cpu); !write_normal) {
                    return std::unexpected(write_normal.error());
                }

                if (save_preview_png) {
                    const auto depth_vis = colorize_depth_map(depth);
                    const auto normal_vis = colorize_normal_map(normal);
                    lfs::core::image_io::save_images_async(
                        preview_path,
                        {depth_vis, normal_vis},
                        true,
                        4);
                }

                stats.rendered_cameras++;

                cache_entry.depth_path = depth_path;
                cache_entry.normal_path = normal_path;
                result.camera_cache_entries[cache_entry.camera_uid] = std::move(cache_entry);

                nlohmann::json cam_entry;
                cam_entry["uid"] = cam.uid();
                cam_entry["image_name"] = cam.image_name();
                cam_entry["width"] = width;
                cam_entry["height"] = height;
                cam_entry["masked"] = masked_camera;
                cam_entry["valid_depth_pixels"] = valid_depth_pixels;
                cam_entry["depth_coverage"] =
                    numel > 0 ? static_cast<double>(valid_depth_pixels) / static_cast<double>(numel) : 0.0;
                cam_entry["depth_file"] = lfs::core::path_to_utf8(depth_path.filename());
                cam_entry["normal_file"] = lfs::core::path_to_utf8(normal_path.filename());
                if (save_preview_png) {
                    cam_entry["preview_file"] = lfs::core::path_to_utf8(preview_path.filename());
                }
                cam_entry["status"] = "rendered";
                manifest["cameras"].push_back(std::move(cam_entry));
            }

            if (save_preview_png) {
                lfs::core::image_io::wait_for_pending_saves();
            }

            const auto manifest_path = write_manifest(stats.cache_dir, manifest);
            LOG_INFO("Mesh supervision disk cache ready: {} (manifest: {})",
                     lfs::core::path_to_utf8(stats.cache_dir),
                     lfs::core::path_to_utf8(manifest_path));

            return result;
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Mesh supervision cache render failed: ") + e.what());
        }
    }

} // namespace lfs::training
