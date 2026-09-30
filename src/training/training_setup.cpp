/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training_setup.hpp"
#include "mesh_hole_fill_adapter.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "geometry/mesh_hole_fill.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "mesh_init_rgb_color_transfer.hpp"
#include "rendering/mesh2splat.hpp"

// clang-format off
#include <glad/glad.h>
// clang-format on
#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <glm/glm.hpp>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stb_image.h>

namespace lfs::training {

    namespace {
        constexpr size_t SH_CHANNELS = 3;
        // Meshes already use scene units, including scanner-bin meshes. Keep
        // external filled meshes and point clouds in the same coordinate system.
        constexpr float MESH_SCENE_POSITION_DIVISOR = 1.0f;

        struct ExplicitSplitSets {
            std::unordered_set<std::string> train_images;
            std::unordered_set<std::string> test_images;
        };

        std::optional<ExplicitSplitSets> build_explicit_split_sets(
            const lfs::core::param::OptimizationParameters& params) {

            if (!params.enable_eval || params.train_images.empty() || params.test_images.empty()) {
                return std::nullopt;
            }

            ExplicitSplitSets sets;
            sets.train_images.insert(params.train_images.begin(), params.train_images.end());
            sets.test_images.insert(params.test_images.begin(), params.test_images.end());
            return sets;
        }

        std::string split_image_name_for_camera(const std::shared_ptr<lfs::core::Camera>& camera) {
            std::string image_name = camera ? camera->image_name() : std::string{};
            const size_t dot_pos = image_name.find_last_of('.');
            if (dot_pos != std::string::npos) {
                image_name.resize(dot_pos);
            }
            return image_name;
        }

        bool split_set_contains_camera(
            const std::unordered_set<std::string>& image_names,
            const std::shared_ptr<lfs::core::Camera>& camera) {

            return image_names.find(split_image_name_for_camera(camera)) != image_names.end();
        }

        std::vector<std::shared_ptr<lfs::core::Camera>> select_train_cameras(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            const lfs::core::param::OptimizationParameters& params,
            const int test_every) {
            const bool enable_eval = params.enable_eval;
            const auto explicit_split_sets = build_explicit_split_sets(params);
            std::vector<std::shared_ptr<lfs::core::Camera>> train_cameras;
            train_cameras.reserve(cameras.size());
            for (size_t i = 0; i < cameras.size(); ++i) {
                const bool is_train = !enable_eval ||
                                      (explicit_split_sets
                                           ? split_set_contains_camera(explicit_split_sets->train_images, cameras[i])
                                           : (i % static_cast<size_t>(std::max(1, test_every))) != 0);
                if (is_train) {
                    train_cameras.push_back(cameras[i]);
                }
            }
            return train_cameras;
        }

        std::expected<void, std::string> normalize_mesh_positions_for_scene(
            lfs::core::MeshData& mesh,
            const float divisor) {

            if (!(divisor > 0.0f) || !std::isfinite(divisor)) {
                return std::unexpected("Mesh normalization divisor must be finite and > 0");
            }

            if (!mesh.vertices.is_valid() || mesh.vertices.ndim() != 2 || mesh.vertices.shape()[1] != 3) {
                return std::unexpected("Mesh vertices are invalid for scene normalization");
            }

            auto vertices_cpu = mesh.vertices.device() == lfs::core::Device::CPU
                                    ? mesh.vertices
                                    : mesh.vertices.to(lfs::core::Device::CPU);
            vertices_cpu = vertices_cpu.contiguous();

            const float inv_divisor = 1.0f / divisor;
            auto vacc = vertices_cpu.accessor<float, 2>();
            const size_t vertex_count = vertices_cpu.shape()[0];
            for (size_t i = 0; i < vertex_count; ++i) {
                vacc(static_cast<int>(i), 0) *= inv_divisor;
                vacc(static_cast<int>(i), 1) *= inv_divisor;
                vacc(static_cast<int>(i), 2) *= inv_divisor;
            }

            if (mesh.vertices.device() == lfs::core::Device::CPU) {
                mesh.vertices = std::move(vertices_cpu);
            } else {
                mesh.vertices = vertices_cpu.to(mesh.vertices.device());
            }

            mesh.mark_dirty();
            return {};
        }

        std::shared_ptr<lfs::core::PointCloud> createRandomPointCloud() {
            constexpr size_t N = 10000;
            auto positions = lfs::core::Tensor::rand({N, 3}, lfs::core::Device::CPU) * 2.0f - 1.0f;
            auto colors = lfs::core::Tensor::randint({N, 3}, 0, 256, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            return std::make_shared<lfs::core::PointCloud>(positions, colors);
        }

        void truncateSHDegree(lfs::core::SplatData& splat, const int target_degree) {
            if (target_degree < 0 || target_degree >= splat.get_max_sh_degree())
                return;

            if (target_degree == 0) {
                splat.shN() = lfs::core::Tensor{};
            } else {
                const size_t keep = static_cast<size_t>((target_degree + 1) * (target_degree + 1) - 1);
                auto& shN = splat.shN();
                if (shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > keep) {
                    const auto slice_end = static_cast<int64_t>(shN.ndim() == 3 ? keep : keep * SH_CHANNELS);
                    shN = shN.slice(1, 0, slice_end).contiguous();
                }
            }
            splat.set_max_sh_degree(target_degree);
            splat.set_active_sh_degree(target_degree);
        }

        std::expected<lfs::core::TextureImage, std::string> load_texture_image_once(
            const std::filesystem::path& texture_path,
            const std::string_view error_label) {

            const auto path_utf8 = lfs::core::path_to_utf8(texture_path);
            lfs::core::TextureImage image;
            uint8_t* pixels = stbi_load(path_utf8.c_str(), &image.width, &image.height, nullptr, 4);
            if (!pixels || image.width <= 0 || image.height <= 0) {
                if (pixels)
                    stbi_image_free(pixels);
                return std::unexpected(std::format(
                    "Failed to load {} '{}'",
                    error_label,
                    path_utf8));
            }

            image.channels = 4;
            const size_t byte_count = static_cast<size_t>(image.width) *
                                      static_cast<size_t>(image.height) *
                                      static_cast<size_t>(image.channels);
            image.pixels.resize(byte_count);
            std::memcpy(image.pixels.data(), pixels, byte_count);
            stbi_image_free(pixels);
            return image;
        }

        std::expected<void, std::string> apply_external_texture_to_mesh(
            lfs::core::MeshData& mesh,
            const std::filesystem::path& texture_path) {

            if (!mesh.has_texcoords()) {
                return std::unexpected(std::format(
                    "Mesh has no UV coordinates, cannot apply texture '{}'",
                    lfs::core::path_to_utf8(texture_path)));
            }

            auto texture_image = load_texture_image_once(texture_path, "mesh init texture");
            if (!texture_image) {
                return std::unexpected(texture_image.error());
            }

            const int texture_width = texture_image->width;
            const int texture_height = texture_image->height;
            const int texture_channels = texture_image->channels;

            mesh.texture_images.clear();
            mesh.texture_images.push_back(std::move(*texture_image));

            if (mesh.materials.empty()) {
                mesh.materials.emplace_back();
            }

            const auto tex_path_utf8 = lfs::core::path_to_utf8(texture_path);
            for (auto& mat : mesh.materials) {
                mat.albedo_tex = 1;
                mat.albedo_tex_path = tex_path_utf8;
                mat.base_color = glm::vec4(1.0f);
            }

            if (mesh.submeshes.empty() && mesh.face_count() > 0) {
                mesh.submeshes.push_back({0, static_cast<size_t>(mesh.face_count()) * 3, 0});
            }

            if (!mesh.materials.empty()) {
                const size_t max_material_index = mesh.materials.size() - 1;
                for (auto& submesh : mesh.submeshes) {
                    if (submesh.material_index > max_material_index) {
                        submesh.material_index = max_material_index;
                    }
                }
            }

            LOG_INFO("Applied mesh init texture '{}' ({}x{}, {} channels)",
                     tex_path_utf8,
                     texture_width,
                     texture_height,
                     texture_channels);
            return {};
        }

        lfs::core::MeshData make_mesh_geometry_only_copy_for_scene(const lfs::core::MeshData& mesh) {
            lfs::core::MeshData copy;
            copy.vertices = mesh.vertices.is_valid()
                                ? mesh.vertices.to(lfs::core::Device::CPU)
                                : mesh.vertices;
            copy.normals = mesh.normals.is_valid()
                               ? mesh.normals.to(lfs::core::Device::CPU)
                               : mesh.normals;
            copy.tangents = mesh.tangents.is_valid()
                                ? mesh.tangents.to(lfs::core::Device::CPU)
                                : mesh.tangents;
            copy.texcoords = mesh.texcoords.is_valid()
                                 ? mesh.texcoords.to(lfs::core::Device::CPU)
                                 : mesh.texcoords;
            copy.colors = mesh.colors.is_valid()
                              ? mesh.colors.to(lfs::core::Device::CPU)
                              : mesh.colors;
            copy.indices = mesh.indices.is_valid()
                               ? mesh.indices.to(lfs::core::Device::CPU)
                               : mesh.indices;
            copy.materials = mesh.materials;
            for (auto& mat : copy.materials) {
                mat.albedo_tex = 0;
                mat.normal_tex = 0;
                mat.metallic_roughness_tex = 0;
                mat.emissive_tex = 0;
                mat.ao_tex = 0;
                mat.albedo_tex_path.clear();
                mat.normal_tex_path.clear();
                mat.metallic_roughness_tex_path.clear();
            }
            copy.submeshes = mesh.submeshes;
            return copy;
        }

        std::string lower_extension(const std::filesystem::path& path) {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return ext;
        }

        bool mesh_has_albedo_texture(const lfs::core::MeshData& mesh) {
            if (mesh.texture_images.empty() || mesh.materials.empty()) {
                return false;
            }

            return std::any_of(mesh.materials.begin(), mesh.materials.end(), [&mesh](const auto& mat) {
                return mat.albedo_tex > 0 && mat.albedo_tex <= mesh.texture_images.size();
            });
        }

        // ------------------------------------------------------------------
        // External filled-mesh face mapping.
        //
        // The render mesh comes from an external hole-filling tool. Its faces must be
        // mapped back onto the original constraint mesh so mesh-surface losses know which
        // original face constrains each Gaussian (-1 means "new hole-fill geometry", which
        // the regularization kernel skips entirely).
        //
        // Two passes:
        //   1. Strict prefix identity — the cheap, historical contract. Faces
        //      [0, original_face_count) must be vertex-identical to the original mesh.
        //   2. Banded geometric matching — used only when pass 1 fails. External tools
        //      routinely move or retriangulate the rim while filling a hole, so matching
        //      is relaxed inside a band around the original hole rims and stays strict
        //      outside it. Strict-zone drift is an error, because it would desynchronize
        //      the project mask from the RGB the original UVs describe.
        // ------------------------------------------------------------------

        constexpr float kExternalPositionTolerance = 1e-4f;

        struct MeshCpuView {
            lfs::core::Tensor vertices;
            lfs::core::Tensor indices;
            const float* v = nullptr;
            const int32_t* i = nullptr;
            size_t vertex_count = 0;
            size_t face_count = 0;

            // The cached pointers alias buffers owned by the tensor members, so the view
            // must stay put once built. Copies and moves are deleted to enforce that;
            // fill_cpu_view() writes into an already-placed instance.
            MeshCpuView() = default;
            MeshCpuView(const MeshCpuView&) = delete;
            MeshCpuView& operator=(const MeshCpuView&) = delete;

            [[nodiscard]] glm::vec3 vertex(const int32_t index) const {
                const auto base = static_cast<size_t>(index) * 3;
                return glm::vec3(v[base + 0], v[base + 1], v[base + 2]);
            }
            [[nodiscard]] bool face_indices_valid(const size_t face) const {
                for (int corner = 0; corner < 3; ++corner) {
                    const auto index = i[face * 3 + corner];
                    if (index < 0 || static_cast<size_t>(index) >= vertex_count) {
                        return false;
                    }
                }
                return true;
            }
            [[nodiscard]] glm::vec3 centroid(const size_t face) const {
                return (vertex(i[face * 3 + 0]) + vertex(i[face * 3 + 1]) + vertex(i[face * 3 + 2])) / 3.0f;
            }
        };

        void fill_cpu_view(const lfs::core::MeshData& mesh, MeshCpuView& view) {
            view.vertices = mesh.vertices.to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32).contiguous();
            view.indices = mesh.indices.to(lfs::core::Device::CPU).to(lfs::core::DataType::Int32).contiguous();
            view.v = view.vertices.ptr<float>();
            view.i = view.indices.ptr<int32_t>();
            view.vertex_count = static_cast<size_t>(mesh.vertex_count());
            view.face_count = static_cast<size_t>(mesh.face_count());
        }

        float mesh_bbox_diagonal(const MeshCpuView& mesh) {
            if (mesh.vertex_count == 0) {
                return 0.0f;
            }
            glm::vec3 lo(std::numeric_limits<float>::max());
            glm::vec3 hi(std::numeric_limits<float>::lowest());
            for (size_t vertex = 0; vertex < mesh.vertex_count; ++vertex) {
                const glm::vec3 p(mesh.v[vertex * 3 + 0], mesh.v[vertex * 3 + 1], mesh.v[vertex * 3 + 2]);
                lo = glm::min(lo, p);
                hi = glm::max(hi, p);
            }
            return glm::length(hi - lo);
        }

        struct GridKey {
            int64_t x = 0;
            int64_t y = 0;
            int64_t z = 0;

            bool operator==(const GridKey& other) const noexcept {
                return x == other.x && y == other.y && z == other.z;
            }
        };

        struct GridKeyHash {
            size_t operator()(const GridKey& key) const noexcept {
                const auto mix = [](uint64_t value) noexcept {
                    value ^= value >> 33;
                    value *= 0xff51afd7ed558ccdULL;
                    value ^= value >> 33;
                    value *= 0xc4ceb9fe1a85ec53ULL;
                    value ^= value >> 33;
                    return value;
                };
                return static_cast<size_t>(
                    mix(static_cast<uint64_t>(key.x)) ^
                    (mix(static_cast<uint64_t>(key.y)) << 1) ^
                    (mix(static_cast<uint64_t>(key.z)) << 2));
            }
        };

        // Uniform spatial hash. The 27-cell neighbourhood scan is exact for query radii up
        // to one cell, so every grid is built with cell_size == the radius it is queried at.
        class SpatialGrid {
        public:
            explicit SpatialGrid(const float cell_size)
                : cell_size_(cell_size > 0.0f ? cell_size : 1.0f) {}

            void insert(const glm::vec3& point, const uint32_t id) {
                cells_[key_of(point)].push_back(id);
            }

            [[nodiscard]] bool empty() const noexcept { return cells_.empty(); }

            template <typename Fn>
            void for_each_near(const glm::vec3& point, Fn&& fn) const {
                const GridKey center = key_of(point);
                for (int64_t dz = -1; dz <= 1; ++dz) {
                    for (int64_t dy = -1; dy <= 1; ++dy) {
                        for (int64_t dx = -1; dx <= 1; ++dx) {
                            const auto it = cells_.find(
                                GridKey{center.x + dx, center.y + dy, center.z + dz});
                            if (it == cells_.end()) {
                                continue;
                            }
                            for (const auto id : it->second) {
                                fn(id);
                            }
                        }
                    }
                }
            }

        private:
            [[nodiscard]] GridKey key_of(const glm::vec3& point) const {
                return GridKey{
                    static_cast<int64_t>(std::floor(point.x / cell_size_)),
                    static_cast<int64_t>(std::floor(point.y / cell_size_)),
                    static_cast<int64_t>(std::floor(point.z / cell_size_))};
            }

            float cell_size_;
            std::unordered_map<GridKey, std::vector<uint32_t>, GridKeyHash> cells_;
        };

        // Boundary edges are matched by quantized vertex position rather than vertex index,
        // so UV seams and hard-normal splits do not masquerade as hole rims. This mirrors
        // the edge-adjacency construction in mesh2splat.cpp.
        std::vector<glm::vec3> collect_boundary_vertex_positions(
            const MeshCpuView& mesh,
            const float quantum) {

            struct EdgeKey {
                int64_t ax, ay, az, bx, by, bz;
                bool operator==(const EdgeKey& other) const noexcept {
                    return ax == other.ax && ay == other.ay && az == other.az &&
                           bx == other.bx && by == other.by && bz == other.bz;
                }
            };
            struct EdgeKeyHash {
                size_t operator()(const EdgeKey& key) const noexcept {
                    const GridKeyHash sub;
                    return sub(GridKey{key.ax, key.ay, key.az}) ^
                           (sub(GridKey{key.bx, key.by, key.bz}) << 1);
                }
            };

            const float inv_quantum = quantum > 0.0f ? 1.0f / quantum : 1.0f;
            const auto quantize = [inv_quantum](const glm::vec3& p) {
                return GridKey{
                    static_cast<int64_t>(std::llround(p.x * inv_quantum)),
                    static_cast<int64_t>(std::llround(p.y * inv_quantum)),
                    static_cast<int64_t>(std::llround(p.z * inv_quantum))};
            };

            std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_counts;
            edge_counts.reserve(mesh.face_count * 3);
            std::unordered_map<EdgeKey, std::pair<glm::vec3, glm::vec3>, EdgeKeyHash> edge_points;
            edge_points.reserve(mesh.face_count * 3);

            for (size_t face = 0; face < mesh.face_count; ++face) {
                if (!mesh.face_indices_valid(face)) {
                    continue;
                }
                for (int corner = 0; corner < 3; ++corner) {
                    const glm::vec3 a = mesh.vertex(mesh.i[face * 3 + corner]);
                    const glm::vec3 b = mesh.vertex(mesh.i[face * 3 + (corner + 1) % 3]);
                    GridKey qa = quantize(a);
                    GridKey qb = quantize(b);
                    bool swapped = false;
                    if (std::tie(qb.x, qb.y, qb.z) < std::tie(qa.x, qa.y, qa.z)) {
                        std::swap(qa, qb);
                        swapped = true;
                    }
                    const EdgeKey key{qa.x, qa.y, qa.z, qb.x, qb.y, qb.z};
                    ++edge_counts[key];
                    edge_points.try_emplace(key, swapped ? b : a, swapped ? a : b);
                }
            }

            std::vector<glm::vec3> boundary_positions;
            for (const auto& [key, count] : edge_counts) {
                if (count != 1) {
                    continue;
                }
                const auto it = edge_points.find(key);
                if (it == edge_points.end()) {
                    continue;
                }
                boundary_positions.push_back(it->second.first);
                boundary_positions.push_back(it->second.second);
            }
            return boundary_positions;
        }

        // Winding-preserving triangle equality across the three cyclic rotations.
        bool faces_match(
            const MeshCpuView& lhs, const size_t lhs_face,
            const MeshCpuView& rhs, const size_t rhs_face,
            const float tolerance) {

            for (int shift = 0; shift < 3; ++shift) {
                bool all_close = true;
                for (int corner = 0; corner < 3; ++corner) {
                    const glm::vec3 a = lhs.vertex(lhs.i[lhs_face * 3 + corner]);
                    const glm::vec3 b = rhs.vertex(rhs.i[rhs_face * 3 + ((corner + shift) % 3)]);
                    if (glm::length(a - b) > tolerance) {
                        all_close = false;
                        break;
                    }
                }
                if (all_close) {
                    return true;
                }
            }
            return false;
        }

        // Pass 1: the historical contract. Cheap, and the common case for tools that only
        // append hole-fill geometry.
        std::optional<std::vector<int32_t>> try_strict_prefix_identity(
            const MeshCpuView& filled,
            const MeshCpuView& original) {

            if (filled.face_count < original.face_count ||
                filled.vertex_count < original.vertex_count) {
                return std::nullopt;
            }
            for (size_t face = 0; face < original.face_count; ++face) {
                if (!filled.face_indices_valid(face) || !original.face_indices_valid(face)) {
                    return std::nullopt;
                }
                for (int corner = 0; corner < 3; ++corner) {
                    const glm::vec3 f = filled.vertex(filled.i[face * 3 + corner]);
                    const glm::vec3 o = original.vertex(original.i[face * 3 + corner]);
                    if (glm::length(f - o) > kExternalPositionTolerance) {
                        return std::nullopt;
                    }
                }
            }
            std::vector<int32_t> mapping(filled.face_count, -1);
            for (size_t face = 0; face < original.face_count; ++face) {
                mapping[face] = static_cast<int32_t>(face);
            }
            return mapping;
        }

        struct BandedMappingStats {
            size_t exact = 0;
            size_t band_remapped = 0;
            size_t hole_fill = 0;
            size_t band_faces = 0;
            size_t boundary_edges = 0;
        };

        // Pass 2: geometric matching with a relaxed band around the original hole rims.
        std::expected<std::vector<int32_t>, std::string> derive_banded_face_mapping(
            const MeshCpuView& filled,
            const MeshCpuView& original,
            const float band_ratio,
            const float band_tolerance_ratio,
            BandedMappingStats& stats) {

            if (original.face_count == 0) {
                return std::unexpected("Cannot derive external face mapping: the original mesh has no faces");
            }

            const float bbox_diagonal = mesh_bbox_diagonal(original);
            if (!(bbox_diagonal > 0.0f)) {
                return std::unexpected(
                    "Cannot derive external face mapping: the original mesh has a degenerate bounding box");
            }
            const float band_radius = band_ratio * bbox_diagonal;
            const float band_tolerance = std::max(
                band_tolerance_ratio * bbox_diagonal, kExternalPositionTolerance);
            const float weld_quantum = std::max(bbox_diagonal * 1e-6f, 1e-12f);

            // Hole rims of the ORIGINAL mesh define where relaxation is allowed.
            const auto boundary_positions = collect_boundary_vertex_positions(original, weld_quantum);
            stats.boundary_edges = boundary_positions.size() / 2;

            SpatialGrid boundary_grid(band_radius > 0.0f ? band_radius : 1.0f);
            for (size_t index = 0; index < boundary_positions.size(); ++index) {
                boundary_grid.insert(boundary_positions[index], static_cast<uint32_t>(index));
            }

            const auto near_boundary = [&](const glm::vec3& point) {
                if (boundary_positions.empty() || !(band_radius > 0.0f)) {
                    return false;
                }
                bool found = false;
                boundary_grid.for_each_near(point, [&](const uint32_t id) {
                    if (found) {
                        return;
                    }
                    if (glm::length(boundary_positions[id] - point) <= band_radius) {
                        found = true;
                    }
                });
                return found;
            };

            // Classify original faces once: a face is in the band when any vertex is.
            std::vector<bool> original_in_band(original.face_count, false);
            for (size_t face = 0; face < original.face_count; ++face) {
                if (!original.face_indices_valid(face)) {
                    return std::unexpected(
                        "Cannot derive external face mapping: the original mesh has invalid face indices");
                }
                for (int corner = 0; corner < 3; ++corner) {
                    if (near_boundary(original.vertex(original.i[face * 3 + corner]))) {
                        original_in_band[face] = true;
                        break;
                    }
                }
                if (original_in_band[face]) {
                    ++stats.band_faces;
                }
            }

            SpatialGrid centroid_grid(band_tolerance);
            for (size_t face = 0; face < original.face_count; ++face) {
                centroid_grid.insert(original.centroid(face), static_cast<uint32_t>(face));
            }

            std::vector<int32_t> mapping(filled.face_count, -1);
            std::vector<bool> original_matched(original.face_count, false);

            for (size_t face = 0; face < filled.face_count; ++face) {
                if (!filled.face_indices_valid(face)) {
                    return std::unexpected(
                        "Cannot derive external face mapping: the filled mesh has invalid face indices");
                }
                const glm::vec3 centroid = filled.centroid(face);

                int64_t exact_match = -1;
                int64_t nearest = -1;
                float nearest_distance = std::numeric_limits<float>::max();
                centroid_grid.for_each_near(centroid, [&](const uint32_t candidate) {
                    if (exact_match >= 0) {
                        return;
                    }
                    if (faces_match(filled, face, original, candidate, kExternalPositionTolerance)) {
                        exact_match = static_cast<int64_t>(candidate);
                        return;
                    }
                    const float distance = glm::length(original.centroid(candidate) - centroid);
                    if (distance < nearest_distance) {
                        nearest_distance = distance;
                        nearest = static_cast<int64_t>(candidate);
                    }
                });

                if (exact_match >= 0) {
                    mapping[face] = static_cast<int32_t>(exact_match);
                    original_matched[static_cast<size_t>(exact_match)] = true;
                    ++stats.exact;
                    continue;
                }

                if (!near_boundary(centroid)) {
                    return std::unexpected(std::format(
                        "Cannot derive external face mapping: filled face {} sits {:.6g} away from any "
                        "original face and is not near a hole rim (band radius {:.6g}). The external tool "
                        "modified geometry far from the holes, which would desynchronize the mesh project "
                        "mask from the original UV colors. Re-export it so only the hole neighbourhoods "
                        "change, or raise --mesh2splat-external-band-ratio if this region really is a rim.",
                        face,
                        nearest >= 0 ? nearest_distance : std::numeric_limits<float>::infinity(),
                        band_radius));
                }

                // Inside the band: keep the geometric constraint when the face still lies on
                // the original surface, otherwise treat it as new hole-fill geometry.
                if (nearest >= 0 && nearest_distance <= band_tolerance) {
                    mapping[face] = static_cast<int32_t>(nearest);
                    original_matched[static_cast<size_t>(nearest)] = true;
                    ++stats.band_remapped;
                } else {
                    mapping[face] = -1;
                    ++stats.hole_fill;
                }
            }

            // Strict-zone coverage: every original face away from a rim must survive in the
            // filled mesh. A missing one means the tool deleted or displaced real geometry.
            for (size_t face = 0; face < original.face_count; ++face) {
                if (!original_in_band[face] && !original_matched[face]) {
                    return std::unexpected(std::format(
                        "Cannot derive external face mapping: original face {} is not near a hole rim but "
                        "has no counterpart in the filled mesh. The external tool removed or moved geometry "
                        "outside the hole neighbourhoods, which would desynchronize the mesh project mask "
                        "from the original UV colors.",
                        face));
                }
            }

            return mapping;
        }

        std::expected<std::vector<int32_t>, std::string> derive_external_face_mapping(
            const lfs::core::MeshData& filled_mesh,
            const lfs::core::MeshData& original_mesh,
            const float band_ratio,
            const float band_tolerance_ratio) {

            MeshCpuView filled;
            MeshCpuView original;
            fill_cpu_view(filled_mesh, filled);
            fill_cpu_view(original_mesh, original);

            if (auto identity = try_strict_prefix_identity(filled, original)) {
                LOG_INFO("mesh_init external: face mapping resolved by strict prefix identity "
                         "({} original faces, {} appended hole-fill faces)",
                         original.face_count,
                         filled.face_count - original.face_count);
                return *identity;
            }

            BandedMappingStats stats;
            auto mapping = derive_banded_face_mapping(
                filled, original, band_ratio, band_tolerance_ratio, stats);
            if (!mapping) {
                return std::unexpected(mapping.error());
            }

            LOG_INFO("mesh_init external: face mapping resolved by banded matching "
                     "(rim edges={}, band faces={}/{}, exact={}, band-remapped={}, hole-fill={}, "
                     "band_ratio={:.4g}, band_tolerance_ratio={:.4g})",
                     stats.boundary_edges,
                     stats.band_faces,
                     original.face_count,
                     stats.exact,
                     stats.band_remapped,
                     stats.hole_fill,
                     band_ratio,
                     band_tolerance_ratio);
            return mapping;
        }

        // ------------------------------------------------------------------
        // ExternalPointCloud mode helpers.
        //
        // The original mesh stays the only render / constraint / color / project-mask
        // source. An external point cloud seeds Gaussians where the mesh is missing.
        // Those Gaussians get birth_tri = -1, so the mesh-surface losses skip them
        // entirely (see docs/Mesh_constrain/mesh2splat_external_filled_mesh_usage.md).
        // ------------------------------------------------------------------

        struct ExternalPointCloudStats {
            size_t loaded = 0;
            size_t after_density_match = 0;
            size_t rejected_near_surface = 0;
            size_t kept = 0;
            float mesh_spacing = 0.0f;
            float target_spacing = 0.0f;
            float actual_spacing = 0.0f;
            float bbox_overlap_ratio = 0.0f;
        };

        // 0.5 * sum |cross(p1-p0, p2-p0)| over all faces.
        double mesh_surface_area(const MeshCpuView& mesh) {
            double area = 0.0;
            for (size_t face = 0; face < mesh.face_count; ++face) {
                if (!mesh.face_indices_valid(face)) {
                    continue;
                }
                const glm::vec3 p0 = mesh.vertex(mesh.i[face * 3 + 0]);
                const glm::vec3 p1 = mesh.vertex(mesh.i[face * 3 + 1]);
                const glm::vec3 p2 = mesh.vertex(mesh.i[face * 3 + 2]);
                area += 0.5 * static_cast<double>(glm::length(glm::cross(p1 - p0, p2 - p0)));
            }
            return area;
        }

        struct Bounds {
            glm::vec3 lo{std::numeric_limits<float>::max()};
            glm::vec3 hi{std::numeric_limits<float>::lowest()};

            void add(const glm::vec3& p) {
                lo = glm::min(lo, p);
                hi = glm::max(hi, p);
            }
            [[nodiscard]] bool valid() const { return glm::all(glm::lessThanEqual(lo, hi)); }
            [[nodiscard]] float volume() const {
                if (!valid()) {
                    return 0.0f;
                }
                const glm::vec3 extent = hi - lo;
                return std::max(extent.x, 0.0f) * std::max(extent.y, 0.0f) * std::max(extent.z, 0.0f);
            }
        };

        // Fraction of the point-cloud bounding box that lies inside the mesh bounding box.
        // A near-zero overlap means the two inputs are in different coordinate systems,
        // which would otherwise produce silently misplaced Gaussians.
        float bbox_overlap_ratio(const Bounds& cloud, const Bounds& mesh) {
            if (!cloud.valid() || !mesh.valid()) {
                return 0.0f;
            }
            const glm::vec3 lo = glm::max(cloud.lo, mesh.lo);
            const glm::vec3 hi = glm::min(cloud.hi, mesh.hi);
            const glm::vec3 extent = glm::max(hi - lo, glm::vec3(0.0f));
            const float intersection = extent.x * extent.y * extent.z;
            const float cloud_volume = cloud.volume();
            if (!(cloud_volume > 0.0f)) {
                // Degenerate (planar or single-point) cloud: fall back to a containment
                // test on the cloud centroid instead of a volume ratio.
                const glm::vec3 center = (cloud.lo + cloud.hi) * 0.5f;
                const bool inside = glm::all(glm::greaterThanEqual(center, mesh.lo)) &&
                                    glm::all(glm::lessThanEqual(center, mesh.hi));
                return inside ? 1.0f : 0.0f;
            }
            return intersection / cloud_volume;
        }

        // Voxel downsample to a target spacing, keeping the centroid of each occupied
        // voxel. Only ever reduces density: a cloud sparser than the target is passed
        // through unchanged, because inventing points has no geometric justification.
        std::vector<glm::vec3> voxel_downsample(
            const std::vector<glm::vec3>& points,
            const float voxel_size) {

            if (!(voxel_size > 0.0f) || points.size() < 2) {
                return points;
            }

            struct Accum {
                glm::dvec3 sum{0.0};
                size_t count = 0;
            };
            std::unordered_map<GridKey, Accum, GridKeyHash> cells;
            cells.reserve(points.size());
            const float inv = 1.0f / voxel_size;
            for (const auto& p : points) {
                const GridKey key{
                    static_cast<int64_t>(std::floor(p.x * inv)),
                    static_cast<int64_t>(std::floor(p.y * inv)),
                    static_cast<int64_t>(std::floor(p.z * inv))};
                auto& cell = cells[key];
                cell.sum += glm::dvec3(p);
                ++cell.count;
            }

            std::vector<glm::vec3> out;
            out.reserve(cells.size());
            for (const auto& [key, cell] : cells) {
                const glm::dvec3 mean = cell.sum / static_cast<double>(cell.count);
                out.emplace_back(static_cast<float>(mean.x),
                                 static_cast<float>(mean.y),
                                 static_cast<float>(mean.z));
            }
            return out;
        }

        float point_triangle_distance(
            const glm::vec3& p,
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c) {

            // Standard closest-point-on-triangle: test the interior, then the three
            // edges, then the vertices via the barycentric regions.
            const glm::vec3 ab = b - a;
            const glm::vec3 ac = c - a;
            const glm::vec3 ap = p - a;
            const float d1 = glm::dot(ab, ap);
            const float d2 = glm::dot(ac, ap);
            if (d1 <= 0.0f && d2 <= 0.0f) {
                return glm::length(ap);
            }
            const glm::vec3 bp = p - b;
            const float d3 = glm::dot(ab, bp);
            const float d4 = glm::dot(ac, bp);
            if (d3 >= 0.0f && d4 <= d3) {
                return glm::length(bp);
            }
            const glm::vec3 cp = p - c;
            const float d5 = glm::dot(ab, cp);
            const float d6 = glm::dot(ac, cp);
            if (d6 >= 0.0f && d5 <= d6) {
                return glm::length(cp);
            }
            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                const float denom = d1 - d3;
                const float t = denom != 0.0f ? d1 / denom : 0.0f;
                return glm::length(p - (a + t * ab));
            }
            const float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                const float denom = d2 - d6;
                const float t = denom != 0.0f ? d2 / denom : 0.0f;
                return glm::length(p - (a + t * ac));
            }
            const float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                const float denom = (d4 - d3) + (d5 - d6);
                const float t = denom != 0.0f ? (d4 - d3) / denom : 0.0f;
                return glm::length(p - (b + t * (c - b)));
            }
            const float denom = va + vb + vc;
            if (!(std::abs(denom) > 0.0f)) {
                return glm::length(ap);
            }
            const float v = vb / denom;
            const float w = vc / denom;
            return glm::length(p - (a + ab * v + ac * w));
        }

        std::expected<std::vector<glm::vec3>, std::string> prepare_external_point_cloud(
            const std::filesystem::path& path,
            const MeshCpuView& mesh,
            const lfs::core::param::OptimizationParameters& opt,
            const float position_divisor,
            ExternalPointCloudStats& stats) {

            auto cloud_result = lfs::io::load_ply_point_cloud(path);
            if (!cloud_result) {
                return std::unexpected(std::format(
                    "Failed to load external point cloud '{}': {}",
                    lfs::core::path_to_utf8(path), cloud_result.error()));
            }

            auto means = cloud_result->means.to(lfs::core::Device::CPU)
                             .to(lfs::core::DataType::Float32)
                             .contiguous();
            if (means.ndim() != 2 || means.shape()[1] != 3 || means.shape()[0] == 0) {
                return std::unexpected("External point cloud must contain [N,3] positions");
            }
            const auto count = static_cast<size_t>(means.shape()[0]);
            const float* mp = means.ptr<float>();

            // Same scene normalization the mesh receives, so both share one space.
            const float inv_divisor = position_divisor != 0.0f ? 1.0f / position_divisor : 1.0f;
            std::vector<glm::vec3> points;
            points.reserve(count);
            Bounds cloud_bounds;
            for (size_t i = 0; i < count; ++i) {
                const glm::vec3 p(mp[i * 3 + 0], mp[i * 3 + 1], mp[i * 3 + 2]);
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                    return std::unexpected("External point cloud contains NaN or Inf coordinates");
                }
                const glm::vec3 scaled = p * inv_divisor;
                points.push_back(scaled);
                cloud_bounds.add(scaled);
            }
            stats.loaded = points.size();

            Bounds mesh_bounds;
            for (size_t v = 0; v < mesh.vertex_count; ++v) {
                mesh_bounds.add(glm::vec3(mesh.v[v * 3 + 0], mesh.v[v * 3 + 1], mesh.v[v * 3 + 2]));
            }

            stats.bbox_overlap_ratio = bbox_overlap_ratio(cloud_bounds, mesh_bounds);
            constexpr float kMinBboxOverlap = 0.05f;
            if (stats.bbox_overlap_ratio < kMinBboxOverlap) {
                return std::unexpected(std::format(
                    "External point cloud does not overlap the original mesh (bbox overlap {:.4g} < {:.4g}). "
                    "The point cloud must use the same coordinate system as the original mesh before scene "
                    "normalization. Cloud bbox [{:.4g},{:.4g},{:.4g}]..[{:.4g},{:.4g},{:.4g}], "
                    "mesh bbox [{:.4g},{:.4g},{:.4g}]..[{:.4g},{:.4g},{:.4g}].",
                    stats.bbox_overlap_ratio, kMinBboxOverlap,
                    cloud_bounds.lo.x, cloud_bounds.lo.y, cloud_bounds.lo.z,
                    cloud_bounds.hi.x, cloud_bounds.hi.y, cloud_bounds.hi.z,
                    mesh_bounds.lo.x, mesh_bounds.lo.y, mesh_bounds.lo.z,
                    mesh_bounds.hi.x, mesh_bounds.hi.y, mesh_bounds.hi.z));
            }

            return points;
        }

        // Density match + redundant-point rejection. `mesh_gaussian_count` is the count
        // mesh2splat actually produced, which is the only reliable density anchor.
        std::expected<std::vector<glm::vec3>, std::string> refine_external_point_cloud(
            std::vector<glm::vec3> points,
            const MeshCpuView& mesh,
            const size_t mesh_gaussian_count,
            const lfs::core::param::OptimizationParameters& opt,
            ExternalPointCloudStats& stats) {

            const double area = mesh_surface_area(mesh);
            if (!(area > 0.0) || mesh_gaussian_count == 0) {
                return std::unexpected(
                    "Cannot derive mesh Gaussian spacing for external point-cloud density matching");
            }
            const float mesh_spacing =
                static_cast<float>(std::sqrt(area / static_cast<double>(mesh_gaussian_count)));
            stats.mesh_spacing = mesh_spacing;

            float target_spacing = mesh_spacing;
            if (opt.mesh2splat_pointcloud_density_ratio > 0.0f) {
                target_spacing = mesh_spacing /
                                 std::sqrt(opt.mesh2splat_pointcloud_density_ratio);
                points = voxel_downsample(points, target_spacing);
            }
            stats.target_spacing = target_spacing;
            stats.after_density_match = points.size();

            if (opt.mesh2splat_pointcloud_surface_reject_ratio > 0.0f && !points.empty()) {
                const float reject_distance =
                    opt.mesh2splat_pointcloud_surface_reject_ratio * mesh_spacing;

                SpatialGrid face_grid(reject_distance);
                for (size_t face = 0; face < mesh.face_count; ++face) {
                    if (mesh.face_indices_valid(face)) {
                        face_grid.insert(mesh.centroid(face), static_cast<uint32_t>(face));
                    }
                }

                std::vector<glm::vec3> kept;
                kept.reserve(points.size());
                for (const auto& p : points) {
                    bool too_close = false;
                    face_grid.for_each_near(p, [&](const uint32_t face) {
                        if (too_close) {
                            return;
                        }
                        const glm::vec3 a = mesh.vertex(mesh.i[face * 3 + 0]);
                        const glm::vec3 b = mesh.vertex(mesh.i[face * 3 + 1]);
                        const glm::vec3 c = mesh.vertex(mesh.i[face * 3 + 2]);
                        if (point_triangle_distance(p, a, b, c) < reject_distance) {
                            too_close = true;
                        }
                    });
                    if (too_close) {
                        ++stats.rejected_near_surface;
                    } else {
                        kept.push_back(p);
                    }
                }
                points = std::move(kept);
            }

            stats.kept = points.size();
            if (points.empty()) {
                return std::unexpected(
                    "External point cloud has no points left after density matching and surface rejection; "
                    "lower --mesh2splat-pointcloud-surface-reject-ratio or check that the cloud covers the "
                    "regions the mesh is missing");
            }
            return points;
        }

        // White colors. The point cloud's own colors are deliberately ignored: they
        // usually come from a source unrelated to this training run. Photometric
        // supervision inside the filled project-mask corrects the initial white.
        lfs::core::PointCloud make_white_point_cloud(const std::vector<glm::vec3>& points) {
            const size_t count = points.size();
            auto means = lfs::core::Tensor::empty(
                {count, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto colors = lfs::core::Tensor::empty(
                {count, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            float* mp = means.ptr<float>();
            float* cp = colors.ptr<float>();
            for (size_t i = 0; i < count; ++i) {
                mp[i * 3 + 0] = points[i].x;
                mp[i * 3 + 1] = points[i].y;
                mp[i * 3 + 2] = points[i].z;
                cp[i * 3 + 0] = 1.0f;
                cp[i * 3 + 1] = 1.0f;
                cp[i * 3 + 2] = 1.0f;
            }
            return lfs::core::PointCloud(std::move(means), std::move(colors));
        }

        std::expected<void, std::string> validate_external_mesh_geometry(
            const lfs::core::MeshData& mesh) {
            if (mesh.vertices.ndim() != 2 || mesh.vertices.shape()[1] != 3 ||
                mesh.indices.ndim() != 2 || mesh.indices.shape()[1] != 3) {
                return std::unexpected("External filled mesh must contain [V,3] vertices and [F,3] triangle indices");
            }
            const auto vertices = mesh.vertices.to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32).contiguous();
            const auto indices = mesh.indices.to(lfs::core::Device::CPU).to(lfs::core::DataType::Int32).contiguous();
            const auto* v = vertices.ptr<float>();
            const auto* i = indices.ptr<int32_t>();
            for (size_t vertex = 0; vertex < static_cast<size_t>(mesh.vertex_count()) * 3; ++vertex) {
                if (!std::isfinite(v[vertex])) {
                    return std::unexpected("External filled mesh contains NaN or Inf vertex coordinates");
                }
            }
            for (size_t face = 0; face < static_cast<size_t>(mesh.face_count()); ++face) {
                const int32_t i0 = i[face * 3 + 0];
                const int32_t i1 = i[face * 3 + 1];
                const int32_t i2 = i[face * 3 + 2];
                if (i0 < 0 || i1 < 0 || i2 < 0 ||
                    static_cast<size_t>(i0) >= static_cast<size_t>(mesh.vertex_count()) ||
                    static_cast<size_t>(i1) >= static_cast<size_t>(mesh.vertex_count()) ||
                    static_cast<size_t>(i2) >= static_cast<size_t>(mesh.vertex_count())) {
                    return std::unexpected("External filled mesh contains an out-of-range triangle index");
                }
                const glm::vec3 p0(v[static_cast<size_t>(i0) * 3 + 0], v[static_cast<size_t>(i0) * 3 + 1], v[static_cast<size_t>(i0) * 3 + 2]);
                const glm::vec3 p1(v[static_cast<size_t>(i1) * 3 + 0], v[static_cast<size_t>(i1) * 3 + 1], v[static_cast<size_t>(i1) * 3 + 2]);
                const glm::vec3 p2(v[static_cast<size_t>(i2) * 3 + 0], v[static_cast<size_t>(i2) * 3 + 1], v[static_cast<size_t>(i2) * 3 + 2]);
                if (glm::length(glm::cross(p1 - p0, p2 - p0)) <= 1e-12f) {
                    return std::unexpected(std::format("External filled mesh contains degenerate triangle {}", face));
                }
            }
            return {};
        }

        struct HoleFillMeshInputStorage {
            std::vector<glm::vec3> vertices;
            std::vector<uint32_t> triangle_indices;
        };

        std::expected<HoleFillMeshInputStorage, std::string> make_hole_fill_input_storage(
            const lfs::core::MeshData& mesh) {
            if (!mesh.vertices.is_valid() || !mesh.indices.is_valid() ||
                mesh.vertices.ndim() != 2 || mesh.vertices.shape()[1] != 3 ||
                mesh.indices.ndim() != 2 || mesh.indices.shape()[1] != 3) {
                return std::unexpected("Mesh hole fill requires [V,3] vertices and [F,3] indices");
            }

            auto vertices_cpu = mesh.vertices.to(lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Float32)
                                    .contiguous();
            auto indices_cpu = mesh.indices.to(lfs::core::Device::CPU)
                                   .to(lfs::core::DataType::Int32)
                                   .contiguous();
            const size_t vertex_count = vertices_cpu.shape()[0];
            const size_t face_count = indices_cpu.shape()[0];
            if (vertex_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                return std::unexpected("Mesh hole fill input has too many vertices for uint32 indices");
            }

            HoleFillMeshInputStorage storage;
            storage.vertices.resize(vertex_count);
            storage.triangle_indices.resize(face_count * 3);
            const auto* vertex_ptr = vertices_cpu.ptr<float>();
            const auto* index_ptr = indices_cpu.ptr<int32_t>();
            for (size_t i = 0; i < vertex_count; ++i) {
                storage.vertices[i] = glm::vec3(
                    vertex_ptr[i * 3 + 0],
                    vertex_ptr[i * 3 + 1],
                    vertex_ptr[i * 3 + 2]);
            }
            for (size_t i = 0; i < face_count * 3; ++i) {
                if (index_ptr[i] < 0 || static_cast<size_t>(index_ptr[i]) >= vertex_count) {
                    return std::unexpected(std::format(
                        "Mesh hole fill input index {} is out of range", index_ptr[i]));
                }
                storage.triangle_indices[i] = static_cast<uint32_t>(index_ptr[i]);
            }
            return storage;
        }

        lfs::core::Tensor append_float_vertex_rows(
            const lfs::core::Tensor& source,
            const size_t source_row_count,
            const size_t appended_row_count,
            const size_t column_count,
            const std::span<const float> appended_values) {
            if (!source.is_valid()) {
                return {};
            }
            if (appended_values.size() != appended_row_count * column_count) {
                throw std::invalid_argument("Appended mesh attribute row count mismatch");
            }

            auto source_cpu = source.to(lfs::core::Device::CPU)
                                  .to(lfs::core::DataType::Float32)
                                  .contiguous();
            if (source_cpu.ndim() != 2 || source_cpu.shape()[0] != source_row_count ||
                source_cpu.shape()[1] != column_count) {
                throw std::invalid_argument("Original mesh attribute shape mismatch");
            }

            auto combined = lfs::core::Tensor::empty(
                {source_row_count + appended_row_count, column_count},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            std::memcpy(
                combined.ptr<float>(),
                source_cpu.ptr<float>(),
                source_row_count * column_count * sizeof(float));
            if (!appended_values.empty()) {
                std::memcpy(
                    combined.ptr<float>() + source_row_count * column_count,
                    appended_values.data(),
                    appended_values.size_bytes());
            }
            return combined;
        }

    } // namespace

    namespace mesh_hole_fill_adapter {

        std::expected<lfs::core::MeshData, std::string> assemble_hole_fill_render_mesh(
            const lfs::core::MeshData& original_mesh,
            const lfs::geometry::MeshHoleFillResult& fill_result) {
            const size_t original_vertex_count = static_cast<size_t>(original_mesh.vertex_count());
            const size_t original_face_count = static_cast<size_t>(original_mesh.face_count());
            const size_t appended_face_count = fill_result.appended_triangle_indices.size() / 3;
            const size_t patch_vertex_count = fill_result.appended_triangle_indices.size();
            if (fill_result.original_vertex_count != original_vertex_count ||
                fill_result.original_face_count != original_face_count ||
                fill_result.appended_triangle_indices.size() % 3 != 0 ||
                fill_result.appended_face_count != appended_face_count ||
                fill_result.appended_face_hole_ids.size() != appended_face_count) {
                return std::unexpected("Mesh hole fill result shape does not match the original mesh");
            }
            if (appended_face_count == 0) {
                return std::unexpected("Mesh hole fill returned no patch faces");
            }
            const size_t fill_source_vertex_count = original_vertex_count + fill_result.appended_vertices.size();
            const size_t combined_vertex_count = original_vertex_count + patch_vertex_count;
            if (combined_vertex_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                return std::unexpected("Filled render mesh exceeds int32 vertex index range");
            }

            auto original_vertices_cpu = original_mesh.vertices.to(lfs::core::Device::CPU)
                                             .to(lfs::core::DataType::Float32)
                                             .contiguous();
            auto original_indices_cpu = original_mesh.indices.to(lfs::core::Device::CPU)
                                            .to(lfs::core::DataType::Int32)
                                            .contiguous();

            lfs::core::MeshData render_mesh = original_mesh.to(lfs::core::Device::CPU);
            render_mesh.vertices = lfs::core::Tensor::empty(
                {combined_vertex_count, size_t{3}},
                lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
            std::memcpy(
                render_mesh.vertices.ptr<float>(),
                original_vertices_cpu.ptr<float>(),
                original_vertex_count * 3 * sizeof(float));
            float* combined_vertices = render_mesh.vertices.ptr<float>();
            const float* original_vertex_ptr = original_vertices_cpu.ptr<float>();
            for (size_t corner = 0; corner < patch_vertex_count; ++corner) {
                const uint32_t source_index = fill_result.appended_triangle_indices[corner];
                if (static_cast<size_t>(source_index) >= fill_source_vertex_count) {
                    return std::unexpected(std::format(
                        "Mesh hole fill output index {} is out of range", source_index));
                }
                glm::vec3 position;
                if (static_cast<size_t>(source_index) < original_vertex_count) {
                    position = glm::vec3(
                        original_vertex_ptr[static_cast<size_t>(source_index) * 3 + 0],
                        original_vertex_ptr[static_cast<size_t>(source_index) * 3 + 1],
                        original_vertex_ptr[static_cast<size_t>(source_index) * 3 + 2]);
                } else {
                    position = fill_result.appended_vertices[static_cast<size_t>(source_index) - original_vertex_count];
                }
                combined_vertices[(original_vertex_count + corner) * 3 + 0] = position.x;
                combined_vertices[(original_vertex_count + corner) * 3 + 1] = position.y;
                combined_vertices[(original_vertex_count + corner) * 3 + 2] = position.z;
            }

            render_mesh.indices = lfs::core::Tensor::empty(
                {original_face_count + appended_face_count, size_t{3}},
                lfs::core::Device::CPU,
                lfs::core::DataType::Int32);
            std::memcpy(
                render_mesh.indices.ptr<int32_t>(),
                original_indices_cpu.ptr<int32_t>(),
                original_face_count * 3 * sizeof(int32_t));
            int32_t* combined_indices = render_mesh.indices.ptr<int32_t>();
            for (size_t corner = 0; corner < patch_vertex_count; ++corner) {
                combined_indices[original_face_count * 3 + corner] =
                    static_cast<int32_t>(original_vertex_count + corner);
            }

            std::vector<float> patch_normals(patch_vertex_count * 3, 0.0f);
            for (size_t face = 0; face < appended_face_count; ++face) {
                const size_t index_offset = original_face_count * 3 + face * 3;
                const int32_t i0 = combined_indices[index_offset + 0];
                const int32_t i1 = combined_indices[index_offset + 1];
                const int32_t i2 = combined_indices[index_offset + 2];
                const glm::vec3 p0(
                    combined_vertices[static_cast<size_t>(i0) * 3 + 0],
                    combined_vertices[static_cast<size_t>(i0) * 3 + 1],
                    combined_vertices[static_cast<size_t>(i0) * 3 + 2]);
                const glm::vec3 p1(
                    combined_vertices[static_cast<size_t>(i1) * 3 + 0],
                    combined_vertices[static_cast<size_t>(i1) * 3 + 1],
                    combined_vertices[static_cast<size_t>(i1) * 3 + 2]);
                const glm::vec3 p2(
                    combined_vertices[static_cast<size_t>(i2) * 3 + 0],
                    combined_vertices[static_cast<size_t>(i2) * 3 + 1],
                    combined_vertices[static_cast<size_t>(i2) * 3 + 2]);
                glm::vec3 normal = glm::cross(p1 - p0, p2 - p0);
                const float length = glm::length(normal);
                normal = length > 1e-8f ? normal / length : glm::vec3(0.0f, 1.0f, 0.0f);
                for (size_t corner = 0; corner < 3; ++corner) {
                    const size_t local_vertex = face * 3 + corner;
                    patch_normals[local_vertex * 3 + 0] = normal.x;
                    patch_normals[local_vertex * 3 + 1] = normal.y;
                    patch_normals[local_vertex * 3 + 2] = normal.z;
                }
            }

            try {
                render_mesh.normals = append_float_vertex_rows(
                    original_mesh.normals,
                    original_vertex_count,
                    patch_vertex_count,
                    3,
                    patch_normals);
                std::vector<float> appended_tangents(patch_vertex_count * 4, 0.0f);
                for (size_t i = 0; i < patch_vertex_count; ++i) {
                    appended_tangents[i * 4 + 0] = 1.0f;
                    appended_tangents[i * 4 + 3] = 1.0f;
                }
                render_mesh.tangents = append_float_vertex_rows(
                    original_mesh.tangents,
                    original_vertex_count,
                    patch_vertex_count,
                    4,
                    appended_tangents);
                render_mesh.texcoords = append_float_vertex_rows(
                    original_mesh.texcoords,
                    original_vertex_count,
                    patch_vertex_count,
                    2,
                    std::vector<float>(patch_vertex_count * 2, 0.0f));
                render_mesh.colors = append_float_vertex_rows(
                    original_mesh.colors,
                    original_vertex_count,
                    patch_vertex_count,
                    4,
                    std::vector<float>(patch_vertex_count * 4, 1.0f));
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Failed to append filled mesh attributes: {}", e.what()));
            }

            if (render_mesh.materials.empty()) {
                lfs::core::Material original_default;
                original_default.name = "mesh2splat_original_default";
                render_mesh.materials.push_back(std::move(original_default));
            }
            if (render_mesh.submeshes.empty()) {
                render_mesh.submeshes.push_back(
                    {0, original_face_count * 3, 0});
            }
            lfs::core::Material patch_material;
            patch_material.name = "mesh2splat_hole_fill_patch";
            patch_material.base_color = glm::vec4(1.0f);
            const size_t patch_material_index = render_mesh.materials.size();
            render_mesh.materials.push_back(std::move(patch_material));
            render_mesh.submeshes.push_back(
                {original_face_count * 3, appended_face_count * 3, patch_material_index});
            render_mesh.mark_dirty();
            return std::move(render_mesh);
        }

        std::expected<HoleDensityCheck, std::string> check_hole_fill_density(
            const lfs::geometry::MeshHoleFillResult& fill_result,
            const lfs::rendering::Mesh2SplatFaceStatistics& statistics,
            const double min_density_ratio,
            const double max_density_ratio) {
            if (!std::isfinite(min_density_ratio) || min_density_ratio <= 0.0 ||
                !std::isfinite(max_density_ratio) || max_density_ratio <= 0.0 ||
                min_density_ratio > max_density_ratio) {
                return std::unexpected(
                    "Mesh hole-fill density ratio bounds must be finite, positive, and ordered");
            }
            const size_t render_face_count = fill_result.original_face_count + fill_result.appended_face_count;
            if (statistics.gaussian_count_per_render_face.size() != render_face_count ||
                fill_result.appended_face_hole_ids.size() != fill_result.appended_face_count) {
                return std::unexpected("Mesh hole-fill density statistics shape mismatch");
            }

            HoleDensityCheck check;
            for (const auto& diagnostic : fill_result.diagnostics) {
                if (diagnostic.status != lfs::geometry::HoleFillStatus::Filled) {
                    continue;
                }
                HoleDensityRecord record;
                record.hole_id = diagnostic.hole_id;
                for (size_t patch_face = 0; patch_face < fill_result.appended_face_count; ++patch_face) {
                    if (fill_result.appended_face_hole_ids[patch_face] == diagnostic.hole_id) {
                        record.patch_gaussian_count += statistics.gaussian_count_per_render_face[fill_result.original_face_count + patch_face];
                    }
                }
                for (const uint32_t neighbor_face : diagnostic.neighbor_face_indices) {
                    if (static_cast<size_t>(neighbor_face) >= fill_result.original_face_count) {
                        return std::unexpected(std::format(
                            "Mesh hole-fill diagnostic {} has invalid neighbor face {}",
                            diagnostic.hole_id,
                            neighbor_face));
                    }
                    record.neighbor_gaussian_count +=
                        statistics.gaussian_count_per_render_face[static_cast<size_t>(neighbor_face)];
                }

                if (diagnostic.patch_area > 0.0 && diagnostic.neighbor_area > 0.0) {
                    record.patch_density =
                        static_cast<double>(record.patch_gaussian_count) / diagnostic.patch_area;
                    record.neighbor_density =
                        static_cast<double>(record.neighbor_gaussian_count) / diagnostic.neighbor_area;
                }
                record.density_ratio = record.neighbor_density > 0.0
                                           ? record.patch_density / record.neighbor_density
                                           : std::numeric_limits<double>::infinity();
                record.accepted = std::isfinite(record.density_ratio) &&
                                  record.density_ratio >= min_density_ratio &&
                                  record.density_ratio <= max_density_ratio;
                if (!record.accepted) {
                    check.rejected_hole_ids.insert(record.hole_id);
                }
                check.records.push_back(record);
            }
            return check;
        }

        std::expected<std::vector<std::optional<double>>, std::string>
        build_density_retry_overrides(
            const lfs::geometry::MeshHoleFillResult& fill_result,
            const HoleDensityCheck& density_check,
            const double base_density_control_factor,
            const double min_density_ratio,
            const double max_density_ratio) {
            if (!std::isfinite(base_density_control_factor) ||
                base_density_control_factor <= 0.0) {
                return std::unexpected(
                    "Mesh hole-fill base density control factor must be finite and positive");
            }
            if (!std::isfinite(min_density_ratio) || min_density_ratio <= 0.0 ||
                !std::isfinite(max_density_ratio) || max_density_ratio <= 0.0 ||
                min_density_ratio > max_density_ratio) {
                return std::unexpected(
                    "Mesh hole-fill density ratio bounds must be finite, positive, and ordered");
            }

            const size_t hole_count = fill_result.diagnostics.size();
            std::vector<const lfs::geometry::HoleFillDiagnostic*> diagnostics_by_hole(
                hole_count,
                nullptr);
            for (const auto& diagnostic : fill_result.diagnostics) {
                if (diagnostic.hole_id >= hole_count ||
                    diagnostics_by_hole[diagnostic.hole_id] != nullptr) {
                    return std::unexpected(
                        "Mesh hole-fill diagnostics do not have unique, contiguous stable hole IDs");
                }
                diagnostics_by_hole[diagnostic.hole_id] = &diagnostic;
            }

            std::vector<const HoleDensityRecord*> records_by_hole(hole_count, nullptr);
            for (const auto& record : density_check.records) {
                if (record.hole_id >= hole_count || records_by_hole[record.hole_id] != nullptr) {
                    return std::unexpected(
                        "Mesh hole-fill density records have an invalid or duplicate hole ID");
                }
                records_by_hole[record.hole_id] = &record;
            }

            std::vector<std::optional<double>> overrides(hole_count);
            for (const size_t hole_id : density_check.rejected_hole_ids) {
                if (hole_id >= hole_count || diagnostics_by_hole[hole_id] == nullptr ||
                    diagnostics_by_hole[hole_id]->status !=
                        lfs::geometry::HoleFillStatus::Filled ||
                    records_by_hole[hole_id] == nullptr) {
                    return std::unexpected(std::format(
                        "Mesh hole-fill density retry references invalid hole {}",
                        hole_id));
                }

                const auto& record = *records_by_hole[hole_id];
                const double ratio = record.density_ratio;
                if (std::isnan(ratio) || ratio < 0.0) {
                    return std::unexpected(std::format(
                        "Mesh hole-fill density retry for hole {} has an invalid ratio",
                        hole_id));
                }

                double multiplier = 1.0;
                if (ratio == 0.0) {
                    multiplier = DENSITY_RETRY_MAX_FACTOR_MULTIPLIER;
                } else if (std::isinf(ratio)) {
                    multiplier = DENSITY_RETRY_MIN_FACTOR_MULTIPLIER;
                } else if (ratio < min_density_ratio) {
                    multiplier = std::sqrt(min_density_ratio / ratio);
                } else if (ratio > max_density_ratio) {
                    multiplier = std::sqrt(max_density_ratio / ratio);
                } else {
                    return std::unexpected(std::format(
                        "Mesh hole-fill density retry for hole {} is already inside the accepted interval",
                        hole_id));
                }
                multiplier = std::clamp(
                    multiplier,
                    DENSITY_RETRY_MIN_FACTOR_MULTIPLIER,
                    DENSITY_RETRY_MAX_FACTOR_MULTIPLIER);

                const long double candidate =
                    static_cast<long double>(base_density_control_factor) * multiplier;
                double retry_factor =
                    candidate > static_cast<long double>(std::numeric_limits<double>::max())
                        ? std::numeric_limits<double>::max()
                        : static_cast<double>(candidate);
                if (retry_factor <= 0.0) {
                    retry_factor = std::numeric_limits<double>::denorm_min();
                }
                if (!std::isfinite(retry_factor) || retry_factor <= 0.0) {
                    return std::unexpected(std::format(
                        "Mesh hole-fill density retry factor for hole {} is not representable",
                        hole_id));
                }
                overrides[hole_id] = retry_factor;
            }
            return overrides;
        }

        std::expected<lfs::geometry::MeshHoleFillResult, std::string>
        merge_density_retry_fill_results(
            const lfs::geometry::MeshHoleFillResult& first_fill_result,
            const lfs::geometry::MeshHoleFillResult& retry_fill_result,
            const std::unordered_set<size_t>& retry_hole_ids) {
            const auto valid_patch_shapes = [](const auto& result) {
                return result.appended_triangle_indices.size() % 3 == 0 &&
                       result.appended_face_count ==
                           result.appended_triangle_indices.size() / 3 &&
                       result.appended_face_hole_ids.size() ==
                           result.appended_face_count;
            };
            if (!valid_patch_shapes(first_fill_result) ||
                !valid_patch_shapes(retry_fill_result)) {
                return std::unexpected(
                    "Mesh hole-fill retry merge received malformed patch arrays");
            }
            if (first_fill_result.original_vertex_count !=
                    retry_fill_result.original_vertex_count ||
                first_fill_result.original_face_count !=
                    retry_fill_result.original_face_count ||
                first_fill_result.diagnostics.size() !=
                    retry_fill_result.diagnostics.size() ||
                first_fill_result.input_vertex_to_welded_representative !=
                    retry_fill_result.input_vertex_to_welded_representative) {
                return std::unexpected(
                    "Mesh hole-fill retry result does not match the first-pass topology");
            }

            const size_t hole_count = first_fill_result.diagnostics.size();
            std::vector<const lfs::geometry::HoleFillDiagnostic*> first_diagnostics(
                hole_count,
                nullptr);
            std::vector<const lfs::geometry::HoleFillDiagnostic*> retry_diagnostics(
                hole_count,
                nullptr);
            const auto index_diagnostics = [hole_count](
                                               const auto& diagnostics,
                                               auto& indexed) {
                for (const auto& diagnostic : diagnostics) {
                    if (diagnostic.hole_id >= hole_count ||
                        indexed[diagnostic.hole_id] != nullptr) {
                        return false;
                    }
                    indexed[diagnostic.hole_id] = &diagnostic;
                }
                return std::ranges::all_of(
                    indexed,
                    [](const auto* diagnostic) { return diagnostic != nullptr; });
            };
            if (!index_diagnostics(first_fill_result.diagnostics, first_diagnostics) ||
                !index_diagnostics(retry_fill_result.diagnostics, retry_diagnostics)) {
                return std::unexpected(
                    "Mesh hole-fill retry merge requires matching stable hole IDs");
            }
            for (const size_t hole_id : retry_hole_ids) {
                if (hole_id >= hole_count ||
                    first_diagnostics[hole_id]->status !=
                        lfs::geometry::HoleFillStatus::Filled) {
                    return std::unexpected(std::format(
                        "Mesh hole-fill retry merge references invalid hole {}",
                        hole_id));
                }
            }

            lfs::geometry::MeshHoleFillResult merged = first_fill_result;
            merged.appended_vertices.clear();
            merged.appended_triangle_indices.clear();
            merged.appended_face_hole_ids.clear();
            merged.appended_face_count = 0;
            merged.successful_hole_count = 0;
            merged.skipped_hole_count = 0;
            merged.failed_hole_count = 0;

            merged.diagnostics.clear();
            merged.diagnostics.reserve(hole_count);
            for (size_t hole_id = 0; hole_id < hole_count; ++hole_id) {
                if (retry_hole_ids.contains(hole_id)) {
                    auto replacement = *retry_diagnostics[hole_id];
                    if (replacement.status != lfs::geometry::HoleFillStatus::Filled) {
                        replacement.status = lfs::geometry::HoleFillStatus::Failed;
                        replacement.reason = std::format(
                            "density retry did not fill the hole: {}",
                            replacement.reason);
                    }
                    merged.diagnostics.push_back(std::move(replacement));
                } else {
                    merged.diagnostics.push_back(*first_diagnostics[hole_id]);
                }
            }

            const auto append_selected_faces = [&merged](
                                                   const auto& source,
                                                   const auto& select_hole) -> std::expected<void, std::string> {
                const size_t original_vertex_count = source.original_vertex_count;
                for (size_t face = 0; face < source.appended_face_count; ++face) {
                    const size_t hole_id = source.appended_face_hole_ids[face];
                    if (!select_hole(hole_id)) {
                        continue;
                    }
                    for (size_t corner = 0; corner < 3; ++corner) {
                        const uint32_t source_index =
                            source.appended_triangle_indices[face * 3 + corner];
                        if (static_cast<size_t>(source_index) < original_vertex_count) {
                            merged.appended_triangle_indices.push_back(source_index);
                            continue;
                        }
                        const size_t appended_index =
                            static_cast<size_t>(source_index) - original_vertex_count;
                        if (appended_index >= source.appended_vertices.size()) {
                            return std::unexpected(std::format(
                                "Mesh hole-fill retry merge has an invalid appended vertex {}",
                                source_index));
                        }
                        const size_t merged_index =
                            merged.original_vertex_count + merged.appended_vertices.size();
                        if (merged_index > std::numeric_limits<uint32_t>::max()) {
                            return std::unexpected(
                                "Mesh hole-fill retry merge exceeds uint32 vertex indices");
                        }
                        merged.appended_vertices.push_back(
                            source.appended_vertices[appended_index]);
                        merged.appended_triangle_indices.push_back(
                            static_cast<uint32_t>(merged_index));
                    }
                    merged.appended_face_hole_ids.push_back(hole_id);
                }
                return {};
            };

            if (auto appended = append_selected_faces(
                    first_fill_result,
                    [&](const size_t hole_id) { return !retry_hole_ids.contains(hole_id); });
                !appended) {
                return std::unexpected(appended.error());
            }
            if (auto appended = append_selected_faces(
                    retry_fill_result,
                    [&](const size_t hole_id) { return retry_hole_ids.contains(hole_id); });
                !appended) {
                return std::unexpected(appended.error());
            }
            merged.appended_face_count = merged.appended_face_hole_ids.size();

            std::vector<size_t> face_count_per_hole(hole_count, 0);
            for (const size_t hole_id : merged.appended_face_hole_ids) {
                if (hole_id >= hole_count) {
                    return std::unexpected(
                        "Mesh hole-fill retry merge produced an invalid face-to-hole mapping");
                }
                ++face_count_per_hole[hole_id];
            }
            for (auto& diagnostic : merged.diagnostics) {
                if (diagnostic.status == lfs::geometry::HoleFillStatus::Filled) {
                    if (face_count_per_hole[diagnostic.hole_id] == 0) {
                        return std::unexpected(std::format(
                            "Mesh hole-fill retry merge lost filled hole {}",
                            diagnostic.hole_id));
                    }
                    ++merged.successful_hole_count;
                } else if (diagnostic.status == lfs::geometry::HoleFillStatus::Skipped) {
                    ++merged.skipped_hole_count;
                } else {
                    ++merged.failed_hole_count;
                }
            }
            return merged;
        }

        lfs::geometry::MeshHoleFillResult filter_rejected_holes(
            const lfs::geometry::MeshHoleFillResult& fill_result,
            const std::unordered_set<size_t>& rejected_hole_ids) {
            lfs::geometry::MeshHoleFillResult filtered = fill_result;
            filtered.appended_triangle_indices.clear();
            filtered.appended_face_hole_ids.clear();
            filtered.appended_triangle_indices.reserve(fill_result.appended_triangle_indices.size());
            filtered.appended_face_hole_ids.reserve(fill_result.appended_face_hole_ids.size());

            for (size_t face = 0; face < fill_result.appended_face_count; ++face) {
                const size_t hole_id = fill_result.appended_face_hole_ids[face];
                if (rejected_hole_ids.contains(hole_id)) {
                    continue;
                }
                filtered.appended_face_hole_ids.push_back(hole_id);
                filtered.appended_triangle_indices.insert(
                    filtered.appended_triangle_indices.end(),
                    fill_result.appended_triangle_indices.begin() + static_cast<std::ptrdiff_t>(face * 3),
                    fill_result.appended_triangle_indices.begin() + static_cast<std::ptrdiff_t>(face * 3 + 3));
            }
            filtered.appended_face_count = filtered.appended_face_hole_ids.size();
            filtered.successful_hole_count =
                fill_result.successful_hole_count >= rejected_hole_ids.size()
                    ? fill_result.successful_hole_count - rejected_hole_ids.size()
                    : 0;
            filtered.failed_hole_count += rejected_hole_ids.size();
            for (auto& diagnostic : filtered.diagnostics) {
                if (rejected_hole_ids.contains(diagnostic.hole_id)) {
                    diagnostic.status = lfs::geometry::HoleFillStatus::Failed;
                    diagnostic.reason = "mesh2splat density ratio outside configured range";
                }
            }
            return filtered;
        }

        std::expected<std::vector<bool>, std::string> build_density_row_keep_mask(
            const lfs::geometry::MeshHoleFillResult& fill_result,
            const lfs::rendering::Mesh2SplatFaceStatistics& statistics,
            const std::unordered_set<size_t>& rejected_hole_ids) {
            if (statistics.raw_render_face_per_gaussian.size() != statistics.gaussian_count) {
                return std::unexpected(
                    "Mesh hole-fill raw render-face row statistics shape mismatch");
            }
            const size_t render_face_count =
                fill_result.original_face_count + fill_result.appended_face_count;
            if (fill_result.appended_face_hole_ids.size() != fill_result.appended_face_count) {
                return std::unexpected("Mesh hole-fill face-to-hole mapping shape mismatch");
            }

            std::vector<bool> keep_rows(statistics.gaussian_count, true);
            for (size_t row = 0; row < statistics.gaussian_count; ++row) {
                const int32_t raw_render_face =
                    statistics.raw_render_face_per_gaussian[row];
                if (raw_render_face < 0 ||
                    static_cast<size_t>(raw_render_face) >= render_face_count) {
                    return std::unexpected(std::format(
                        "Mesh hole-fill Gaussian row {} has invalid raw render face {}",
                        row,
                        raw_render_face));
                }
                const size_t render_face = static_cast<size_t>(raw_render_face);
                if (render_face < fill_result.original_face_count) {
                    continue;
                }
                const size_t patch_face = render_face - fill_result.original_face_count;
                keep_rows[row] = !rejected_hole_ids.contains(
                    fill_result.appended_face_hole_ids[patch_face]);
            }
            return keep_rows;
        }

    } // namespace mesh_hole_fill_adapter

    namespace {

        using mesh_hole_fill_adapter::assemble_hole_fill_render_mesh;
        using mesh_hole_fill_adapter::build_density_retry_overrides;
        using mesh_hole_fill_adapter::build_density_row_keep_mask;
        using mesh_hole_fill_adapter::check_hole_fill_density;
        using mesh_hole_fill_adapter::filter_rejected_holes;
        using mesh_hole_fill_adapter::HoleDensityCheck;
        using mesh_hole_fill_adapter::merge_density_retry_fill_results;

        struct ClosestTrianglePoint {
            glm::vec3 barycentric{1.0f, 0.0f, 0.0f};
            float squared_distance = std::numeric_limits<float>::infinity();
        };

        ClosestTrianglePoint closest_point_on_triangle(
            const glm::vec3& point,
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c) {
            const glm::vec3 ab = b - a;
            const glm::vec3 ac = c - a;
            const glm::vec3 ap = point - a;
            const float d1 = glm::dot(ab, ap);
            const float d2 = glm::dot(ac, ap);
            glm::vec3 barycentric;
            if (d1 <= 0.0f && d2 <= 0.0f) {
                barycentric = {1.0f, 0.0f, 0.0f};
            } else {
                const glm::vec3 bp = point - b;
                const float d3 = glm::dot(ab, bp);
                const float d4 = glm::dot(ac, bp);
                if (d3 >= 0.0f && d4 <= d3) {
                    barycentric = {0.0f, 1.0f, 0.0f};
                } else {
                    const float vc = d1 * d4 - d3 * d2;
                    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                        const float v = d1 / (d1 - d3);
                        barycentric = {1.0f - v, v, 0.0f};
                    } else {
                        const glm::vec3 cp = point - c;
                        const float d5 = glm::dot(ab, cp);
                        const float d6 = glm::dot(ac, cp);
                        if (d6 >= 0.0f && d5 <= d6) {
                            barycentric = {0.0f, 0.0f, 1.0f};
                        } else {
                            const float vb = d5 * d2 - d1 * d6;
                            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                                const float w = d2 / (d2 - d6);
                                barycentric = {1.0f - w, 0.0f, w};
                            } else {
                                const float va = d3 * d6 - d5 * d4;
                                if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                                    const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                                    barycentric = {0.0f, 1.0f - w, w};
                                } else {
                                    const float denominator = va + vb + vc;
                                    if (!(std::abs(denominator) > 1e-20f)) {
                                        barycentric = {1.0f, 0.0f, 0.0f};
                                    } else {
                                        const float inverse = 1.0f / denominator;
                                        const float v = vb * inverse;
                                        const float w = vc * inverse;
                                        barycentric = {1.0f - v - w, v, w};
                                    }
                                }
                            }
                        }
                    }
                }
            }
            const glm::vec3 closest = a * barycentric.x + b * barycentric.y + c * barycentric.z;
            return {.barycentric = barycentric,
                    .squared_distance = glm::dot(point - closest, point - closest)};
        }

        class NearestGaussianIndex {
        public:
            NearestGaussianIndex(
                const float* positions,
                std::vector<size_t> gaussian_indices)
                : positions_(positions), indices_(std::move(gaussian_indices)) {
                nodes_.reserve(indices_.size());
                root_ = build(0, indices_.size(), 0);
            }

            [[nodiscard]] std::optional<size_t> nearest(const glm::vec3& point) const {
                if (root_ < 0) {
                    return std::nullopt;
                }
                size_t best_index = 0;
                float best_distance = std::numeric_limits<float>::infinity();
                query(root_, point, best_index, best_distance);
                return best_index;
            }

        private:
            struct Node {
                size_t gaussian_index = 0;
                int left = -1;
                int right = -1;
                uint8_t axis = 0;
            };

            [[nodiscard]] float coordinate(const size_t gaussian_index, const uint8_t axis) const {
                return positions_[gaussian_index * 3 + axis];
            }

            int build(const size_t begin, const size_t end, const uint8_t depth) {
                if (begin >= end) {
                    return -1;
                }
                const uint8_t axis = depth % 3;
                const size_t middle = begin + (end - begin) / 2;
                std::nth_element(
                    indices_.begin() + static_cast<std::ptrdiff_t>(begin),
                    indices_.begin() + static_cast<std::ptrdiff_t>(middle),
                    indices_.begin() + static_cast<std::ptrdiff_t>(end),
                    [&](const size_t lhs, const size_t rhs) {
                        return coordinate(lhs, axis) < coordinate(rhs, axis);
                    });
                const int node_index = static_cast<int>(nodes_.size());
                nodes_.push_back(Node{.gaussian_index = indices_[middle], .axis = axis});
                const int left = build(begin, middle, static_cast<uint8_t>(depth + 1));
                const int right = build(middle + 1, end, static_cast<uint8_t>(depth + 1));
                nodes_[static_cast<size_t>(node_index)].left = left;
                nodes_[static_cast<size_t>(node_index)].right = right;
                return node_index;
            }

            void query(
                const int node_index,
                const glm::vec3& point,
                size_t& best_index,
                float& best_distance) const {
                if (node_index < 0) {
                    return;
                }
                const auto& node = nodes_[static_cast<size_t>(node_index)];
                const float dx = point.x - positions_[node.gaussian_index * 3 + 0];
                const float dy = point.y - positions_[node.gaussian_index * 3 + 1];
                const float dz = point.z - positions_[node.gaussian_index * 3 + 2];
                const float distance = dx * dx + dy * dy + dz * dz;
                if (distance < best_distance) {
                    best_distance = distance;
                    best_index = node.gaussian_index;
                }

                const float split_delta = point[node.axis] - coordinate(node.gaussian_index, node.axis);
                const int near_child = split_delta < 0.0f ? node.left : node.right;
                const int far_child = split_delta < 0.0f ? node.right : node.left;
                query(near_child, point, best_index, best_distance);
                if (split_delta * split_delta < best_distance) {
                    query(far_child, point, best_index, best_distance);
                }
            }

            const float* positions_ = nullptr;
            std::vector<size_t> indices_;
            std::vector<Node> nodes_;
            int root_ = -1;
        };

        float srgb_channel_to_linear(const float value) {
            return value <= 0.04045f
                       ? value / 12.92f
                       : std::pow((value + 0.055f) / 1.055f, 2.4f);
        }

        struct SampledAlbedo {
            glm::vec3 linear_rgb{0.0f};
            float alpha = 1.0f;
        };

        std::optional<SampledAlbedo> sample_albedo_texture_linear(
            const lfs::core::TextureImage& image,
            const glm::vec2 uv) {
            if (image.width <= 0 || image.height <= 0 || image.channels <= 0 ||
                image.pixels.size() < static_cast<size_t>(image.width) *
                                          static_cast<size_t>(image.height) *
                                          static_cast<size_t>(image.channels) ||
                !std::isfinite(uv.x) || !std::isfinite(uv.y)) {
                return std::nullopt;
            }

            const float x = std::clamp(uv.x, 0.0f, 1.0f) * static_cast<float>(image.width) - 0.5f;
            const float y = std::clamp(uv.y, 0.0f, 1.0f) * static_cast<float>(image.height) - 0.5f;
            const int x_floor = static_cast<int>(std::floor(x));
            const int y_floor = static_cast<int>(std::floor(y));
            const int x0 = std::clamp(x_floor, 0, image.width - 1);
            const int y0 = std::clamp(y_floor, 0, image.height - 1);
            const int x1 = std::clamp(x_floor + 1, 0, image.width - 1);
            const int y1 = std::clamp(y_floor + 1, 0, image.height - 1);
            const float tx = std::clamp(x - static_cast<float>(x_floor), 0.0f, 1.0f);
            const float ty = std::clamp(y - static_cast<float>(y_floor), 0.0f, 1.0f);

            auto fetch = [&](const int px, const int py) {
                const size_t offset =
                    (static_cast<size_t>(py) * static_cast<size_t>(image.width) + static_cast<size_t>(px)) *
                    static_cast<size_t>(image.channels);
                SampledAlbedo sample;
                for (int channel = 0; channel < 3; ++channel) {
                    const int source_channel = image.channels == 1 ? 0 : std::min(channel, image.channels - 1);
                    const float srgb = static_cast<float>(image.pixels[offset + static_cast<size_t>(source_channel)]) /
                                       255.0f;
                    sample.linear_rgb[channel] = srgb_channel_to_linear(srgb);
                }
                if (image.channels >= 4) {
                    sample.alpha = static_cast<float>(image.pixels[offset + 3]) / 255.0f;
                }
                return sample;
            };

            const auto top_left = fetch(x0, y0);
            const auto top_right = fetch(x1, y0);
            const auto bottom_left = fetch(x0, y1);
            const auto bottom_right = fetch(x1, y1);
            return SampledAlbedo{
                .linear_rgb = glm::mix(
                    glm::mix(top_left.linear_rgb, top_right.linear_rgb, tx),
                    glm::mix(bottom_left.linear_rgb, bottom_right.linear_rgb, tx),
                    ty),
                .alpha = glm::mix(
                    glm::mix(top_left.alpha, top_right.alpha, tx),
                    glm::mix(bottom_left.alpha, bottom_right.alpha, tx),
                    ty)};
        }

        void initialize_hole_fill_colors(
            lfs::core::SplatData& model,
            const lfs::core::MeshData& original_mesh,
            const bool transfer_all_rows = false) {
            if ((!transfer_all_rows && !model.has_mesh_hole_fill_mask()) || model.size() == 0) {
                return;
            }

            lfs::core::Tensor hole_mask;
            if (!transfer_all_rows) {
                hole_mask = model.mesh_hole_fill_mask().to(lfs::core::Device::CPU).contiguous();
            }
            if (!transfer_all_rows && (hole_mask.ndim() != 1 || hole_mask.shape()[0] != model.size())) {
                LOG_WARN("mesh_init hole fill: invalid hole mask shape; leaving converter colors unchanged");
                return;
            }
            const bool* hole_ptr = transfer_all_rows ? nullptr : hole_mask.ptr<bool>();
            const size_t hole_count = transfer_all_rows
                                          ? model.size()
                                          : std::count(hole_ptr, hole_ptr + model.size(), true);
            if (hole_count == 0) {
                return;
            }

            auto means_cpu = model.means().to(lfs::core::Device::CPU).contiguous();
            auto sh0_cpu = model.sh0().to(lfs::core::Device::CPU).contiguous();
            auto birth_tri_cpu = model.birth_tri().to(lfs::core::Device::CPU).contiguous();
            float* sh0_ptr = sh0_cpu.ptr<float>();
            const float* means_ptr = means_cpu.ptr<float>();
            const int32_t* birth_tri_ptr = birth_tri_cpu.ptr<int32_t>();

            auto vertices_cpu = original_mesh.vertices.to(lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Float32)
                                    .contiguous();
            auto indices_cpu = original_mesh.indices.to(lfs::core::Device::CPU)
                                   .to(lfs::core::DataType::Int32)
                                   .contiguous();
            lfs::core::Tensor texcoords_cpu;
            if (original_mesh.has_texcoords()) {
                texcoords_cpu = original_mesh.texcoords.to(lfs::core::Device::CPU)
                                    .to(lfs::core::DataType::Float32)
                                    .contiguous();
            }
            const float* vertex_ptr = vertices_cpu.ptr<float>();
            const int32_t* index_ptr = indices_cpu.ptr<int32_t>();
            const float* texcoord_ptr = texcoords_cpu.is_valid() ? texcoords_cpu.ptr<float>() : nullptr;
            const size_t face_count = static_cast<size_t>(original_mesh.face_count());

            std::vector<size_t> face_material(face_count, 0);
            for (const auto& submesh : original_mesh.submeshes) {
                const size_t first_face = std::min(submesh.start_index / 3, face_count);
                const size_t end_face = std::min(
                    (submesh.start_index + submesh.index_count) / 3,
                    face_count);
                for (size_t face = first_face; face < end_face; ++face) {
                    face_material[face] = submesh.material_index;
                }
            }

            constexpr size_t MAX_LINEAR_UV_TRIANGLE_TESTS = 50'000'000;
            const bool use_uv_triangle_search =
                texcoord_ptr &&
                (face_count == 0 || hole_count <= MAX_LINEAR_UV_TRIANGLE_TESTS / face_count);
            if (texcoord_ptr && !use_uv_triangle_search) {
                LOG_WARN("mesh_init hole fill: skipping linear UV search over {} hole GS x {} faces (limit={}); falling back to the nearest original GS index",
                         hole_count, face_count, MAX_LINEAR_UV_TRIANGLE_TESTS);
            }

            constexpr float SH_C0 = 0.28209479177387814f;
            std::vector<size_t> unresolved;
            unresolved.reserve(hole_count);
            size_t texture_color_count = 0;
            for (size_t gaussian = 0; gaussian < model.size(); ++gaussian) {
                if (!transfer_all_rows && !hole_ptr[gaussian]) {
                    continue;
                }

                const glm::vec3 point(
                    means_ptr[gaussian * 3 + 0],
                    means_ptr[gaussian * 3 + 1],
                    means_ptr[gaussian * 3 + 2]);
                float best_distance = std::numeric_limits<float>::infinity();
                glm::vec3 best_color;
                bool found_texture_color = false;
                if (use_uv_triangle_search) {
                    for (size_t face = 0; face < face_count; ++face) {
                        const size_t material_index = face_material[face];
                        if (material_index >= original_mesh.materials.size()) {
                            continue;
                        }
                        const auto& material = original_mesh.materials[material_index];
                        if (material.albedo_tex == 0 ||
                            material.albedo_tex > original_mesh.texture_images.size()) {
                            continue;
                        }
                        const int32_t i0 = index_ptr[face * 3 + 0];
                        const int32_t i1 = index_ptr[face * 3 + 1];
                        const int32_t i2 = index_ptr[face * 3 + 2];
                        const glm::vec3 p0(
                            vertex_ptr[static_cast<size_t>(i0) * 3 + 0],
                            vertex_ptr[static_cast<size_t>(i0) * 3 + 1],
                            vertex_ptr[static_cast<size_t>(i0) * 3 + 2]);
                        const glm::vec3 p1(
                            vertex_ptr[static_cast<size_t>(i1) * 3 + 0],
                            vertex_ptr[static_cast<size_t>(i1) * 3 + 1],
                            vertex_ptr[static_cast<size_t>(i1) * 3 + 2]);
                        const glm::vec3 p2(
                            vertex_ptr[static_cast<size_t>(i2) * 3 + 0],
                            vertex_ptr[static_cast<size_t>(i2) * 3 + 1],
                            vertex_ptr[static_cast<size_t>(i2) * 3 + 2]);
                        const auto closest = closest_point_on_triangle(point, p0, p1, p2);
                        if (!(closest.squared_distance < best_distance)) {
                            continue;
                        }
                        const glm::vec2 uv0(
                            texcoord_ptr[static_cast<size_t>(i0) * 2 + 0],
                            texcoord_ptr[static_cast<size_t>(i0) * 2 + 1]);
                        const glm::vec2 uv1(
                            texcoord_ptr[static_cast<size_t>(i1) * 2 + 0],
                            texcoord_ptr[static_cast<size_t>(i1) * 2 + 1]);
                        const glm::vec2 uv2(
                            texcoord_ptr[static_cast<size_t>(i2) * 2 + 0],
                            texcoord_ptr[static_cast<size_t>(i2) * 2 + 1]);
                        if (!mesh_hole_fill_adapter::is_transferable_uv_triangle(
                                uv0, uv1, uv2)) {
                            continue;
                        }
                        const glm::vec2 uv = uv0 * closest.barycentric.x +
                                             uv1 * closest.barycentric.y +
                                             uv2 * closest.barycentric.z;
                        const auto sampled_albedo = sample_albedo_texture_linear(
                            original_mesh.texture_images[material.albedo_tex - 1], uv);
                        // The converter does not train alpha/opacity from albedo. A fully
                        // transparent texel therefore has no meaningful transferable RGB;
                        // treat it as unavailable and continue to the original-GS fallback.
                        if (!sampled_albedo ||
                            !mesh_hole_fill_adapter::has_transferable_albedo_alpha(
                                sampled_albedo->alpha,
                                material.base_color.a)) {
                            continue;
                        }
                        auto linear_color = sampled_albedo->linear_rgb;
                        linear_color.r *= material.base_color.r;
                        linear_color.g *= material.base_color.g;
                        linear_color.b *= material.base_color.b;
                        best_color = glm::pow(
                            glm::clamp(linear_color, glm::vec3(0.0f), glm::vec3(1.0f)),
                            glm::vec3(1.0f / 2.2f));
                        best_distance = closest.squared_distance;
                        found_texture_color = true;
                    }
                }

                if (found_texture_color) {
                    sh0_ptr[gaussian * 3 + 0] = (best_color.r - 0.5f) / SH_C0;
                    sh0_ptr[gaussian * 3 + 1] = (best_color.g - 0.5f) / SH_C0;
                    sh0_ptr[gaussian * 3 + 2] = (best_color.b - 0.5f) / SH_C0;
                    ++texture_color_count;
                } else {
                    unresolved.push_back(gaussian);
                }
            }

            std::vector<size_t> original_gaussians;
            original_gaussians.reserve(model.size() - hole_count);
            for (size_t gaussian = 0; gaussian < model.size(); ++gaussian) {
                if ((!transfer_all_rows && !hole_ptr[gaussian]) ||
                    (transfer_all_rows && birth_tri_ptr[gaussian] >= 0)) {
                    original_gaussians.push_back(gaussian);
                }
            }
            const NearestGaussianIndex nearest_original_gaussian(
                means_ptr,
                original_gaussians);

            size_t nearest_gs_color_count = 0;
            size_t white_color_count = 0;
            constexpr float WHITE_SH0 = 0.5f / SH_C0;
            for (const size_t gaussian : unresolved) {
                const glm::vec3 point(
                    means_ptr[gaussian * 3 + 0],
                    means_ptr[gaussian * 3 + 1],
                    means_ptr[gaussian * 3 + 2]);
                if (const auto nearest = nearest_original_gaussian.nearest(point)) {
                    std::memcpy(
                        sh0_ptr + gaussian * 3,
                        sh0_ptr + *nearest * 3,
                        3 * sizeof(float));
                    ++nearest_gs_color_count;
                } else {
                    sh0_ptr[gaussian * 3 + 0] = WHITE_SH0;
                    sh0_ptr[gaussian * 3 + 1] = WHITE_SH0;
                    sh0_ptr[gaussian * 3 + 2] = WHITE_SH0;
                    ++white_color_count;
                }
            }

            model.sh0() = sh0_cpu.to(model.sh0().device());
            LOG_INFO("mesh_init hole fill: initialized {} hole colors from UV/albedo, {} from nearest original GS, {} to white",
                     texture_color_count, nearest_gs_color_count, white_color_count);
        }

        std::expected<void, std::string> verify_obj_mesh_texture_loaded(
            const lfs::core::MeshData& mesh,
            const std::filesystem::path& mesh_path) {

            if (!mesh.has_texcoords()) {
                return std::unexpected(std::format(
                    "Mesh init OBJ '{}' has no UV coordinates; check the OBJ contains vt entries",
                    lfs::core::path_to_utf8(mesh_path)));
            }

            if (!mesh_has_albedo_texture(mesh)) {
                return std::unexpected(std::format(
                    "Mesh init OBJ '{}' did not load an albedo texture; check mtllib/map_Kd and texture file paths",
                    lfs::core::path_to_utf8(mesh_path)));
            }

            return {};
        }

        std::expected<std::unique_ptr<lfs::core::SplatData>, std::string> create_init_model(
            const lfs::core::param::TrainingParameters& params,
            const lfs::core::Tensor& scene_center,
            const std::filesystem::path& init_file,
            const std::vector<std::shared_ptr<lfs::core::Camera>>& train_cameras,
            std::shared_ptr<lfs::core::PointCloud>* init_point_cloud_out = nullptr,
            std::shared_ptr<lfs::core::MeshData>* init_mesh_out = nullptr,
            std::shared_ptr<lfs::core::MeshData>* project_mask_mesh_out = nullptr,
            std::shared_ptr<lfs::core::PointCloud>* project_mask_point_cloud_out = nullptr,
            float* project_mask_point_spacing_out = nullptr) {

            if (init_point_cloud_out) {
                init_point_cloud_out->reset();
            }
            if (init_mesh_out) {
                init_mesh_out->reset();
            }
            if (project_mask_mesh_out) {
                project_mask_mesh_out->reset();
            }
            if (project_mask_point_cloud_out) {
                project_mask_point_cloud_out->reset();
            }
            if (project_mask_point_spacing_out) {
                *project_mask_point_spacing_out = 0.0f;
            }

            if (params.use_mesh_init) {
                if (!params.mesh_init_mesh_path.has_value() || params.mesh_init_mesh_path->empty()) {
                    return std::unexpected("Mesh init requires --mesh-init-gs-scene <ply_or_obj_file_or_legacy_directory>");
                }

                const auto mesh_init_path = lfs::core::utf8_to_path(*params.mesh_init_mesh_path);
                std::optional<std::filesystem::path> explicit_texture_path;
                if (params.mesh_init_texture_path.has_value() && !params.mesh_init_texture_path->empty()) {
                    explicit_texture_path = lfs::core::utf8_to_path(*params.mesh_init_texture_path);
                }
                const auto mesh_ext = lower_extension(mesh_init_path);

                // Try PLY per-face texcoord loading first (MeshLab-style UV seams).
                // This preserves per-face-corner UVs that Assimp's JoinIdenticalVertices
                // cannot represent, avoiding false-colour Gaussians at texture seams.
                std::shared_ptr<lfs::core::MeshData> mesh_data;
                {
                    if (mesh_ext == ".ply") {
                        auto ply_result = lfs::io::load_ply_mesh(mesh_init_path);
                        if (ply_result) {
                            mesh_data = std::make_shared<lfs::core::MeshData>(std::move(*ply_result));
                            LOG_INFO("mesh_init: loaded PLY with per-face UVs ({} verts, {} faces)",
                                     mesh_data->vertex_count(), mesh_data->face_count());
                        } else {
                            LOG_DEBUG("mesh_init: PLY has no per-face texcoords ({}), using generic loader",
                                      ply_result.error());
                        }
                    }
                }

                // Fall back to generic loader (Assimp) for OBJ/MTL or PLY without per-face UVs.
                if (!mesh_data) {
                    auto loader = lfs::io::Loader::create();
                    auto mesh_result = loader->load(mesh_init_path);
                    if (!mesh_result) {
                        return std::unexpected(std::format(
                            "Failed to load mesh init file '{}': {}",
                            lfs::core::path_to_utf8(mesh_init_path),
                            mesh_result.error().format()));
                    }

                    try {
                        mesh_data = std::get<std::shared_ptr<lfs::core::MeshData>>(mesh_result->data);
                    } catch (const std::bad_variant_access&) {
                        return std::unexpected(std::format(
                            "'{}' is not a mesh file result; disable --mesh-init-gs-scene or provide a mesh with faces",
                            lfs::core::path_to_utf8(mesh_init_path)));
                    }
                }

                if (!mesh_data) {
                    return std::unexpected("Mesh init returned null mesh data");
                }

                const auto color_source = params.optimization.mesh_init_color_source;
                const bool allow_texture =
                    color_source == lfs::core::param::MeshInitColorSource::Auto ||
                    color_source == lfs::core::param::MeshInitColorSource::Texture;
                if (allow_texture && mesh_ext == ".ply") {
                    std::filesystem::path texture_path;
                    if (explicit_texture_path.has_value()) {
                        texture_path = *explicit_texture_path;
                    } else {
                        const auto resolved_texture_path = lfs::io::resolve_ply_texture_file(mesh_init_path);
                        if (!resolved_texture_path) {
                            if (color_source == lfs::core::param::MeshInitColorSource::Texture) {
                                return std::unexpected(resolved_texture_path.error());
                            }
                            LOG_INFO("mesh_init: no PLY texture resolved; RGB-view color initialization will be used");
                            texture_path.clear();
                        } else {
                            texture_path = *resolved_texture_path;
                        }
                    }
                    if (!texture_path.empty()) {
                        if (const auto tex_result = apply_external_texture_to_mesh(*mesh_data, texture_path); !tex_result) {
                            if (color_source == lfs::core::param::MeshInitColorSource::Texture) {
                                return std::unexpected(tex_result.error());
                            }
                            LOG_WARN("mesh_init: PLY texture load failed ({}); RGB-view color initialization will be used",
                                     tex_result.error());
                        }
                    }
                } else if (allow_texture && explicit_texture_path.has_value()) {
                    if (const auto tex_result = apply_external_texture_to_mesh(*mesh_data, *explicit_texture_path); !tex_result) {
                        if (color_source == lfs::core::param::MeshInitColorSource::Texture) {
                            return std::unexpected(tex_result.error());
                        }
                        LOG_WARN("mesh_init: explicit texture load failed ({}); RGB-view color initialization will be used",
                                 tex_result.error());
                    }
                } else if (allow_texture && mesh_ext == ".obj") {
                    if (const auto texture_result = verify_obj_mesh_texture_loaded(*mesh_data, mesh_init_path); !texture_result) {
                        if (color_source == lfs::core::param::MeshInitColorSource::Texture) {
                            return std::unexpected(texture_result.error());
                        }
                        LOG_INFO("mesh_init: OBJ has no usable texture; RGB-view color initialization will be used");
                    }
                }

                const bool use_rgb_view_colors =
                    color_source == lfs::core::param::MeshInitColorSource::RgbViews ||
                    (color_source == lfs::core::param::MeshInitColorSource::Auto &&
                     (!mesh_data->has_texcoords() || !mesh_has_albedo_texture(*mesh_data)));

                if (const auto normalize_result = normalize_mesh_positions_for_scene(
                        *mesh_data, MESH_SCENE_POSITION_DIVISOR);
                    !normalize_result) {
                    return std::unexpected(std::format("Mesh init normalization failed: {}", normalize_result.error()));
                }

                LOG_INFO("Normalized mesh init geometry by 1/{} after texture/material setup",
                         MESH_SCENE_POSITION_DIVISOR);

                std::optional<lfs::core::MeshData> render_mesh;
                std::optional<lfs::geometry::MeshHoleFillResult> active_fill_result;
                std::optional<HoleFillMeshInputStorage> hole_fill_input_storage;
                std::optional<lfs::geometry::MeshHoleFillOptions> hole_fill_options;
                std::vector<int32_t> render_face_to_constraint_face;
                bool hole_fill_succeeded = false;
                const auto hole_fill_mode = params.optimization.resolved_mesh2splat_hole_fill_mode();
                const bool cgal_hole_fill_enabled = hole_fill_mode == lfs::core::param::Mesh2SplatHoleFillMode::Cgal;
                const bool external_hole_fill_enabled = hole_fill_mode == lfs::core::param::Mesh2SplatHoleFillMode::External;
                const bool external_pointcloud_enabled =
                    hole_fill_mode == lfs::core::param::Mesh2SplatHoleFillMode::ExternalPointCloud;
                bool external_mode = false;
                bool external_has_uv_albedo_mode = false;
                if (external_hole_fill_enabled) {
                    if (!params.mesh_init_external_filled_mesh_path || params.mesh_init_external_filled_mesh_path->empty()) {
                        return std::unexpected("External mesh2splat mode requires a filled mesh path");
                    }
                    const auto external_path = lfs::core::utf8_to_path(*params.mesh_init_external_filled_mesh_path);
                    auto loader = lfs::io::Loader::create();
                    auto external_result = loader->load(external_path);
                    if (!external_result) {
                        return std::unexpected(std::format(
                            "Failed to load external filled mesh '{}': {}",
                            lfs::core::path_to_utf8(external_path), external_result.error().format()));
                    }
                    std::shared_ptr<lfs::core::MeshData> external_mesh;
                    try {
                        external_mesh = std::get<std::shared_ptr<lfs::core::MeshData>>(external_result->data);
                    } catch (const std::bad_variant_access&) {
                        return std::unexpected("External filled mesh loader result is not mesh data");
                    }
                    if (!external_mesh || !external_mesh->vertices.is_valid() ||
                        !external_mesh->indices.is_valid() || external_mesh->vertex_count() == 0 ||
                        external_mesh->face_count() == 0) {
                        return std::unexpected("External filled mesh is empty or has no valid triangle indices");
                    }
                    if (const auto geometry_result = validate_external_mesh_geometry(*external_mesh);
                        !geometry_result) {
                        return std::unexpected(geometry_result.error());
                    }
                    if (const auto normalize_result = normalize_mesh_positions_for_scene(
                            *external_mesh, MESH_SCENE_POSITION_DIVISOR);
                        !normalize_result) {
                        return std::unexpected(std::format(
                            "External filled mesh normalization failed: {}", normalize_result.error()));
                    }
                    // External materials/textures are not authoritative. If the external
                    // mesh supplies UVs, bind the original mesh texture so those UVs sample
                    // the same source image; otherwise leave texturing disabled and transfer
                    // colors from the original mesh after readback.
                    external_has_uv_albedo_mode = external_mesh->has_texcoords() &&
                                                  mesh_has_albedo_texture(*mesh_data);
                    if (external_has_uv_albedo_mode) {
                        external_mesh->texture_images = mesh_data->texture_images;
                        if (external_mesh->materials.empty()) {
                            external_mesh->materials.emplace_back();
                        }
                        for (auto& material : external_mesh->materials) {
                            material.albedo_tex = 1;
                            material.albedo_tex_path = mesh_data->materials.empty()
                                                           ? std::string{}
                                                           : mesh_data->materials.front().albedo_tex_path;
                            material.base_color = glm::vec4(1.0f);
                            material.normal_tex = 0;
                            material.metallic_roughness_tex = 0;
                            material.emissive_tex = 0;
                            material.ao_tex = 0;
                        }
                    } else {
                        external_mesh->texture_images.clear();
                        for (auto& material : external_mesh->materials) {
                            material.albedo_tex = 0;
                            material.normal_tex = 0;
                            material.metallic_roughness_tex = 0;
                            material.emissive_tex = 0;
                            material.ao_tex = 0;
                            material.albedo_tex_path.clear();
                        }
                    }
                    auto mapping = derive_external_face_mapping(
                        *external_mesh,
                        *mesh_data,
                        params.optimization.mesh2splat_external_band_ratio,
                        params.optimization.mesh2splat_external_band_tolerance_ratio);
                    if (!mapping) {
                        return std::unexpected(mapping.error());
                    }
                    render_face_to_constraint_face = std::move(*mapping);
                    render_mesh.emplace(std::move(*external_mesh));
                    external_mode = true;
                    LOG_INFO("mesh_init external: using filled render mesh '{}' ({} verts, {} faces), original mesh remains the constraint/GT source",
                             lfs::core::path_to_utf8(external_path),
                             render_mesh->vertex_count(),
                             render_mesh->face_count());
                }
                if (cgal_hole_fill_enabled) {
                    auto input_storage = make_hole_fill_input_storage(*mesh_data);
                    if (!input_storage) {
                        LOG_WARN("mesh_init hole fill: {}; falling back to the original mesh",
                                 input_storage.error());
                    } else {
                        hole_fill_input_storage.emplace(std::move(*input_storage));
                        hole_fill_options.emplace();
                        auto& fill_options = *hole_fill_options;
                        fill_options.max_boundary_edges = static_cast<size_t>(
                            params.optimization.mesh2splat_hole_fill_max_boundary_edges);
                        fill_options.max_boundary_perimeter_bbox_ratio =
                            params.optimization.mesh2splat_hole_fill_max_boundary_perimeter_bbox_ratio;
                        fill_options.max_boundary_area_bbox_ratio =
                            params.optimization.mesh2splat_hole_fill_max_boundary_area_bbox_ratio;
                        fill_options.max_boundary_bbox_diagonal_ratio =
                            params.optimization.mesh2splat_hole_fill_max_boundary_bbox_diagonal_ratio;
                        fill_options.weld_epsilon_bbox_ratio =
                            params.optimization.mesh2splat_hole_fill_weld_epsilon_bbox_ratio;
                        fill_options.density_control_factor =
                            params.optimization.mesh2splat_hole_fill_density_control_factor;

                        auto fill_result = lfs::geometry::fill_mesh_holes(
                            lfs::geometry::MeshHoleFillInput{
                                .vertices = hole_fill_input_storage->vertices,
                                .triangle_indices = hole_fill_input_storage->triangle_indices},
                            fill_options);
                        if (!fill_result) {
                            LOG_WARN("mesh_init hole fill failed: {}; falling back to the original mesh",
                                     fill_result.error().message);
                        } else {
                            LOG_INFO("mesh_init hole fill topology: input_vertices={}, input_faces={}, welded_vertices={}, bbox_diagonal={:.6g}, weld_epsilon={:.6g}, appended_vertices={}, appended_faces={}",
                                     fill_result->original_vertex_count,
                                     fill_result->original_face_count,
                                     fill_result->welded_vertex_count,
                                     fill_result->mesh_bbox_diagonal,
                                     fill_result->weld_epsilon,
                                     fill_result->appended_vertices.size(),
                                     fill_result->appended_face_count);
                            for (const auto& diagnostic : fill_result->diagnostics) {
                                const char* status = "failed";
                                if (diagnostic.status == lfs::geometry::HoleFillStatus::Filled) {
                                    status = "filled";
                                } else if (diagnostic.status == lfs::geometry::HoleFillStatus::Skipped) {
                                    status = "skipped";
                                }
                                LOG_INFO("mesh_init hole fill: hole {} {}: {} (edges={}, perimeter_ratio={:.6g}, area_ratio={:.6g}, bbox_diagonal_ratio={:.6g}, density_factor={:.6g}, patch_faces={})",
                                         diagnostic.hole_id,
                                         status,
                                         diagnostic.reason,
                                         diagnostic.boundary_edge_count,
                                         diagnostic.boundary_perimeter_bbox_ratio,
                                         diagnostic.boundary_area_bbox_ratio,
                                         diagnostic.boundary_bbox_diagonal_ratio,
                                         diagnostic.density_control_factor,
                                         diagnostic.appended_face_count);
                            }
                            LOG_INFO("mesh_init hole fill: {} filled, {} skipped, {} failed",
                                     fill_result->successful_hole_count,
                                     fill_result->skipped_hole_count,
                                     fill_result->failed_hole_count);

                            if (fill_result->successful_hole_count > 0 &&
                                fill_result->appended_face_count > 0) {
                                auto assembled = assemble_hole_fill_render_mesh(*mesh_data, *fill_result);
                                if (!assembled) {
                                    LOG_WARN("mesh_init hole fill render-mesh assembly failed: {}; falling back to the original mesh",
                                             assembled.error());
                                } else {
                                    render_mesh.emplace(std::move(*assembled));
                                    render_face_to_constraint_face.resize(
                                        static_cast<size_t>(render_mesh->face_count()), -1);
                                    const size_t original_face_count =
                                        static_cast<size_t>(mesh_data->face_count());
                                    for (size_t face = 0; face < original_face_count; ++face) {
                                        render_face_to_constraint_face[face] = static_cast<int32_t>(face);
                                    }
                                    active_fill_result.emplace(std::move(*fill_result));
                                    hole_fill_succeeded = true;
                                    LOG_INFO("mesh_init hole fill: render mesh has {} original + {} patch faces",
                                             original_face_count,
                                             active_fill_result->appended_face_count);
                                }
                            } else {
                                LOG_WARN("mesh_init hole fill: no eligible hole was filled; using exact original mesh2splat path");
                            }
                        }
                    }
                }

                // ---- Use mesh2splat GPU pipeline for proper per-triangle rotation/scaling ----
                SDL_Window* m2s_window = nullptr;
                SDL_GLContext m2s_gl = nullptr;
                auto cleanup_gl = [&]() {
                    if (m2s_gl) {
                        SDL_GL_DestroyContext(m2s_gl);
                        m2s_gl = nullptr;
                    }
                    if (m2s_window) {
                        SDL_DestroyWindow(m2s_window);
                        m2s_window = nullptr;
                    }
                };

                if (!SDL_Init(SDL_INIT_VIDEO)) {
                    return std::unexpected(std::format("mesh_init mesh2splat: SDL_Init failed: {}", SDL_GetError()));
                }
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
                m2s_window = SDL_CreateWindow("mesh_init_m2s", 1, 1,
                                              SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
                if (!m2s_window) {
                    return std::unexpected(std::format("mesh_init mesh2splat: SDL_CreateWindow failed: {}", SDL_GetError()));
                }
                m2s_gl = SDL_GL_CreateContext(m2s_window);
                if (!m2s_gl) {
                    cleanup_gl();
                    return std::unexpected(std::format("mesh_init mesh2splat: SDL_GL_CreateContext failed: {}", SDL_GetError()));
                }
                SDL_GL_MakeCurrent(m2s_window, m2s_gl);
                if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
                    cleanup_gl();
                    return std::unexpected("mesh_init mesh2splat: gladLoadGLLoader failed");
                }

                LOG_INFO("mesh_init: using mesh2splat GPU pipeline for Gaussian initialization");
                lfs::core::Mesh2SplatOptions m2s_opts;
                m2s_opts.sh_degree = params.optimization.sh_degree;
                m2s_opts.sampling_rate = params.mesh_init_sampling_rate;
                m2s_opts.target_max_gaussians = params.optimization.max_cap > 0
                                                    ? params.optimization.max_cap
                                                    : 0;
                std::vector<int32_t> original_face_to_constraint_face;
                if (cgal_hole_fill_enabled || external_mode) {
                    original_face_to_constraint_face.resize(
                        static_cast<size_t>(mesh_data->face_count()));
                    for (size_t face = 0;
                         face < original_face_to_constraint_face.size();
                         ++face) {
                        original_face_to_constraint_face[face] =
                            static_cast<int32_t>(face);
                    }
                }
                auto run_mesh2splat = [&](lfs::rendering::Mesh2SplatFaceStatistics* statistics) {
                    if (hole_fill_succeeded || external_mode) {
                        return lfs::rendering::mesh_to_splat(
                            lfs::rendering::Mesh2SplatInput{
                                .render_mesh = *render_mesh,
                                .constraint_mesh = mesh_data.get(),
                                .render_face_to_constraint_face = render_face_to_constraint_face,
                                .face_statistics = statistics},
                            m2s_opts);
                    }
                    if (cgal_hole_fill_enabled) {
                        // Even when no patch survives, the enabled feature keeps
                        // target_max_gaussians as a strict global cap. An explicit
                        // identity mapping selects the cap-aware path while keeping
                        // every row associated with the exact original mesh.
                        return lfs::rendering::mesh_to_splat(
                            lfs::rendering::Mesh2SplatInput{
                                .render_mesh = *mesh_data,
                                .constraint_mesh = mesh_data.get(),
                                .render_face_to_constraint_face =
                                    original_face_to_constraint_face,
                                .face_statistics = statistics},
                            m2s_opts);
                    }
                    return lfs::rendering::mesh_to_splat(*mesh_data, m2s_opts);
                };

                lfs::rendering::Mesh2SplatFaceStatistics conversion_statistics;
                auto m2s_result = run_mesh2splat(
                    (hole_fill_succeeded || external_mode) ? &conversion_statistics : nullptr);

                auto convert_exact_original = [&](std::string reason) {
                    if (external_mode) {
                        return std::expected<std::unique_ptr<lfs::core::SplatData>, std::string>(
                            std::unexpected(std::move(reason)));
                    }
                    LOG_WARN("mesh_init hole fill: {}; converting cap-aware exact original mesh fallback",
                             reason);
                    active_fill_result.reset();
                    render_mesh.reset();
                    render_face_to_constraint_face.clear();
                    hole_fill_succeeded = false;
                    return run_mesh2splat(nullptr);
                };

                if (!m2s_result && hole_fill_succeeded) {
                    m2s_result = convert_exact_original(std::format(
                        "combined conversion failed ({})",
                        m2s_result.error()));
                }

                auto log_density_probe = [&](const std::string_view pass,
                                             const HoleDensityCheck& density_check) {
                    LOG_INFO("mesh_init hole fill density {}: render_faces={}, gaussians={}, resolution={} (requested={})",
                             pass,
                             render_mesh->face_count(),
                             conversion_statistics.gaussian_count,
                             conversion_statistics.final_resolution,
                             conversion_statistics.requested_resolution);
                    for (const auto& record : density_check.records) {
                        LOG_INFO("mesh_init hole fill density {}: hole {} patch={}/rho={:.6g}, neighbor={}/rho={:.6g}, ratio={:.6g}, accepted={}",
                                 pass,
                                 record.hole_id,
                                 record.patch_gaussian_count,
                                 record.patch_density,
                                 record.neighbor_gaussian_count,
                                 record.neighbor_density,
                                 record.density_ratio,
                                 record.accepted);
                    }
                };

                auto terminal_filter_density = [&](const HoleDensityCheck& density_check,
                                                   const size_t combined_converter_calls) {
                    if (density_check.rejected_hole_ids.empty()) {
                        LOG_INFO("mesh_init hole fill density: all retry-success holes accepted; combined converter calls={}",
                                 combined_converter_calls);
                        return;
                    }
                    for (const size_t hole_id : density_check.rejected_hole_ids) {
                        LOG_WARN("mesh_init hole fill density: terminally excluding hole {}", hole_id);
                    }

                    auto filtered_fill_result = filter_rejected_holes(
                        *active_fill_result,
                        density_check.rejected_hole_ids);
                    if (filtered_fill_result.appended_face_count == 0) {
                        m2s_result = convert_exact_original(
                            "no density-accepted patch remains after the bounded retry");
                        return;
                    }

                    auto filtered_render_mesh = assemble_hole_fill_render_mesh(
                        *mesh_data,
                        filtered_fill_result);
                    auto keep_rows = build_density_row_keep_mask(
                        *active_fill_result,
                        conversion_statistics,
                        density_check.rejected_hole_ids);
                    if (!filtered_render_mesh) {
                        m2s_result = convert_exact_original(std::format(
                            "accepted-patch render-mesh assembly failed ({})",
                            filtered_render_mesh.error()));
                        return;
                    }
                    if (!keep_rows) {
                        m2s_result = convert_exact_original(keep_rows.error());
                        return;
                    }

                    const size_t kept_gaussian_count = static_cast<size_t>(
                        std::count(keep_rows->begin(), keep_rows->end(), true));
                    const size_t unfiltered_gaussian_count = (*m2s_result)->size();
                    auto keep_tensor = lfs::core::Tensor::from_vector(
                                           *keep_rows,
                                           {keep_rows->size()},
                                           lfs::core::Device::CPU)
                                           .to((*m2s_result)->means().device());
                    auto filtered_model = lfs::core::extract_by_mask(
                        **m2s_result,
                        keep_tensor);
                    if (filtered_model.size() != kept_gaussian_count ||
                        kept_gaussian_count == 0) {
                        m2s_result = convert_exact_original(std::format(
                            "density row filtering produced {} rows, expected {}",
                            filtered_model.size(),
                            kept_gaussian_count));
                        return;
                    }

                    *m2s_result = std::make_unique<lfs::core::SplatData>(
                        std::move(filtered_model));
                    render_mesh.emplace(std::move(*filtered_render_mesh));
                    active_fill_result.emplace(std::move(filtered_fill_result));
                    render_face_to_constraint_face.assign(
                        static_cast<size_t>(render_mesh->face_count()), -1);
                    const size_t original_face_count =
                        static_cast<size_t>(mesh_data->face_count());
                    for (size_t face = 0; face < original_face_count; ++face) {
                        render_face_to_constraint_face[face] = static_cast<int32_t>(face);
                    }
                    LOG_INFO("mesh_init hole fill density: retained {} accepted holes and filtered {} rejected-hole Gaussians in-place ({} -> {} rows); combined converter calls={}, resolution={}",
                             active_fill_result->successful_hole_count,
                             unfiltered_gaussian_count - kept_gaussian_count,
                             unfiltered_gaussian_count,
                             kept_gaussian_count,
                             combined_converter_calls,
                             conversion_statistics.final_resolution);
                };

                if (m2s_result && hole_fill_succeeded && active_fill_result) {
                    auto first_density_check = check_hole_fill_density(
                        *active_fill_result,
                        conversion_statistics,
                        params.optimization.mesh2splat_hole_fill_min_density_ratio,
                        params.optimization.mesh2splat_hole_fill_max_density_ratio);
                    if (!first_density_check) {
                        m2s_result = convert_exact_original(first_density_check.error());
                    } else {
                        log_density_probe("probe 1/2", *first_density_check);
                        if (first_density_check->rejected_hole_ids.empty()) {
                            LOG_INFO("mesh_init hole fill density: all holes accepted; combined converter calls=1");
                        } else {
                            auto retry_overrides = build_density_retry_overrides(
                                *active_fill_result,
                                *first_density_check,
                                hole_fill_options->density_control_factor,
                                params.optimization.mesh2splat_hole_fill_min_density_ratio,
                                params.optimization.mesh2splat_hole_fill_max_density_ratio);
                            if (!retry_overrides) {
                                LOG_WARN("mesh_init hole fill density retry planning failed: {}; retaining first-pass accepted holes",
                                         retry_overrides.error());
                                terminal_filter_density(*first_density_check, 1);
                            } else {
                                auto retry_options = *hole_fill_options;
                                retry_options.density_control_factor_overrides =
                                    std::move(*retry_overrides);
                                retry_options.fill_only_density_control_factor_overrides = true;
                                for (const size_t hole_id : first_density_check->rejected_hole_ids) {
                                    LOG_INFO("mesh_init hole fill density retry: hole {} factor {:.6g} -> {:.6g}",
                                             hole_id,
                                             retry_options.density_control_factor,
                                             *retry_options.density_control_factor_overrides[hole_id]);
                                }

                                auto retry_fill_result = lfs::geometry::fill_mesh_holes(
                                    lfs::geometry::MeshHoleFillInput{
                                        .vertices = hole_fill_input_storage->vertices,
                                        .triangle_indices = hole_fill_input_storage->triangle_indices},
                                    retry_options);
                                if (retry_fill_result) {
                                    for (const auto& diagnostic : retry_fill_result->diagnostics) {
                                        const char* status = "failed";
                                        if (diagnostic.status ==
                                            lfs::geometry::HoleFillStatus::Filled) {
                                            status = "filled";
                                        } else if (diagnostic.status ==
                                                   lfs::geometry::HoleFillStatus::Skipped) {
                                            status = "skipped";
                                        }
                                        LOG_INFO("mesh_init hole fill density retry: hole {} {}: {} (density_factor={:.6g}, patch_faces={})",
                                                 diagnostic.hole_id,
                                                 status,
                                                 diagnostic.reason,
                                                 diagnostic.density_control_factor,
                                                 diagnostic.appended_face_count);
                                    }
                                }
                                if (!retry_fill_result) {
                                    LOG_WARN("mesh_init hole fill density retry failed: {}; retaining first-pass accepted holes",
                                             retry_fill_result.error().message);
                                    terminal_filter_density(*first_density_check, 1);
                                } else if (retry_fill_result->appended_face_count == 0 ||
                                           retry_fill_result->successful_hole_count == 0) {
                                    LOG_WARN("mesh_init hole fill density retry produced no patch; retaining first-pass accepted holes");
                                    terminal_filter_density(*first_density_check, 1);
                                } else {
                                    auto merged_fill_result = merge_density_retry_fill_results(
                                        *active_fill_result,
                                        *retry_fill_result,
                                        first_density_check->rejected_hole_ids);
                                    if (!merged_fill_result) {
                                        LOG_WARN("mesh_init hole fill density retry merge failed: {}; retaining first-pass accepted holes",
                                                 merged_fill_result.error());
                                        terminal_filter_density(*first_density_check, 1);
                                    } else {
                                        auto retry_render_mesh = assemble_hole_fill_render_mesh(
                                            *mesh_data,
                                            *merged_fill_result);
                                        if (!retry_render_mesh) {
                                            LOG_WARN("mesh_init hole fill density retry render-mesh assembly failed: {}; retaining first-pass accepted holes",
                                                     retry_render_mesh.error());
                                            terminal_filter_density(*first_density_check, 1);
                                        } else {
                                            render_mesh.emplace(std::move(*retry_render_mesh));
                                            active_fill_result.emplace(std::move(*merged_fill_result));
                                            render_face_to_constraint_face.assign(
                                                static_cast<size_t>(render_mesh->face_count()), -1);
                                            const size_t original_face_count =
                                                static_cast<size_t>(mesh_data->face_count());
                                            for (size_t face = 0; face < original_face_count; ++face) {
                                                render_face_to_constraint_face[face] =
                                                    static_cast<int32_t>(face);
                                            }

                                            conversion_statistics = {};
                                            m2s_result = run_mesh2splat(&conversion_statistics);
                                            if (!m2s_result) {
                                                m2s_result = convert_exact_original(std::format(
                                                    "density retry combined conversion failed ({})",
                                                    m2s_result.error()));
                                            } else {
                                                auto second_density_check = check_hole_fill_density(
                                                    *active_fill_result,
                                                    conversion_statistics,
                                                    params.optimization.mesh2splat_hole_fill_min_density_ratio,
                                                    params.optimization.mesh2splat_hole_fill_max_density_ratio);
                                                if (!second_density_check) {
                                                    m2s_result = convert_exact_original(
                                                        second_density_check.error());
                                                } else {
                                                    log_density_probe("probe 2/2", *second_density_check);
                                                    terminal_filter_density(*second_density_check, 2);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                cleanup_gl();

                if (!m2s_result) {
                    mesh_data->texture_images.clear();
                    mesh_data->texture_images.shrink_to_fit();
                    return std::unexpected(std::format("mesh_init mesh2splat failed: {}", m2s_result.error()));
                }

                auto model = std::move(*m2s_result);
                if (hole_fill_succeeded || (external_mode && !external_has_uv_albedo_mode)) {
                    if ((!external_mode && !model->has_mesh_hole_fill_mask()) ||
                        (!external_mode && (model->mesh_hole_fill_mask().ndim() != 1 ||
                                             model->mesh_hole_fill_mask().shape()[0] != model->size())) ||
                        !model->birth_tri().is_valid() ||
                        model->birth_tri().ndim() != 1 ||
                        model->birth_tri().shape()[0] != model->size()) {
                        return std::unexpected(
                            "mesh_init hole fill produced inconsistent provenance tensors");
                    }
                    initialize_hole_fill_colors(
                        *model,
                        *mesh_data,
                        external_mode && !external_has_uv_albedo_mode);
                }
                if (use_rgb_view_colors) {
                    auto transfer_result = transfer_mesh_init_rgb_colors(
                        *model,
                        *mesh_data,
                        train_cameras,
                        params);
                    if (!transfer_result) {
                        return std::unexpected(transfer_result.error());
                    }
                } else if (color_source == lfs::core::param::MeshInitColorSource::White) {
                    constexpr float WHITE_SH0 = 0.5f / 0.28209479177387814f;
                    model->sh0().fill_(WHITE_SH0);
                    LOG_INFO("mesh_init: initialized {} Gaussian colors to white (--mesh-init-color-source white)",
                             model->size());
                }
                if (init_mesh_out) {
                    // Mesh GT supervision only needs normalized geometry; texture pixels were only
                    // needed by mesh2splat and should not overlap with the initialized GS scene.
                    *init_mesh_out = std::make_shared<lfs::core::MeshData>(
                        make_mesh_geometry_only_copy_for_scene(*mesh_data));
                }
                if ((hole_fill_succeeded || external_mode) && project_mask_mesh_out) {
                    *project_mask_mesh_out = std::make_shared<lfs::core::MeshData>(
                        make_mesh_geometry_only_copy_for_scene(*render_mesh));
                } else if (external_pointcloud_enabled && project_mask_mesh_out) {
                    // The original mesh is the coverage base here, and it is already the
                    // only visible mesh node — but routing through the override matters:
                    // the scene path applies each node's world transform, while the
                    // override path renders with identity. The hole-filling point cloud is
                    // in normalized mesh space, so only the identity path is guaranteed to
                    // put the projected points in the pixels the depth buffer wrote.
                    *project_mask_mesh_out = std::make_shared<lfs::core::MeshData>(
                        make_mesh_geometry_only_copy_for_scene(*mesh_data));
                }

                mesh_data->texture_images.clear();
                mesh_data->texture_images.shrink_to_fit();
                if (render_mesh) {
                    render_mesh->texture_images.clear();
                    render_mesh->texture_images.shrink_to_fit();
                }

                // Mark all mesh-init Gaussians for means/rotation freeze during training
                model->mesh_init_mask() = lfs::core::Tensor::ones_bool(
                    {model->size()}, model->means().device());

                // ExternalPointCloud mode: seed additional Gaussians where the mesh is
                // missing. Appended AFTER the mesh_init_mask fill above so the point rows
                // keep mesh_init_mask=false and stay opacity-trainable even when
                // mesh2splat_opacity_no_grad is on. birth_tri=-1 exempts them from every
                // mesh-surface loss; see the usage doc's constraint-exemption chain.
                if (external_pointcloud_enabled) {
                    if (!params.mesh_init_external_pointcloud_path ||
                        params.mesh_init_external_pointcloud_path->empty()) {
                        return std::unexpected("External point-cloud mode requires a point cloud path");
                    }
                    const auto cloud_path =
                        lfs::core::utf8_to_path(*params.mesh_init_external_pointcloud_path);

                    MeshCpuView constraint_view;
                    fill_cpu_view(*mesh_data, constraint_view);

                    ExternalPointCloudStats pc_stats;
                    auto loaded = prepare_external_point_cloud(
                        cloud_path, constraint_view, params.optimization,
                        MESH_SCENE_POSITION_DIVISOR, pc_stats);
                    if (!loaded) {
                        return std::unexpected(loaded.error());
                    }

                    auto refined = refine_external_point_cloud(
                        std::move(*loaded), constraint_view,
                        static_cast<size_t>(model->size()), params.optimization, pc_stats);
                    if (!refined) {
                        return std::unexpected(refined.error());
                    }

                    const auto point_cloud = make_white_point_cloud(*refined);
                    auto point_model = lfs::core::init_model_from_pointcloud(
                        params, scene_center, point_cloud);
                    if (!point_model) {
                        return std::unexpected(std::format(
                            "External point-cloud Gaussian init failed: {}", point_model.error()));
                    }

                    // Hand the same point set to project_mesh masking. It must be the
                    // post-downsample, post-rejection cloud: masks derived from a denser
                    // or unrejected cloud would not match where Gaussians actually sit.
                    if (project_mask_point_cloud_out) {
                        *project_mask_point_cloud_out =
                            std::make_shared<lfs::core::PointCloud>(point_cloud);
                    }
                    if (project_mask_point_spacing_out) {
                        *project_mask_point_spacing_out = pc_stats.target_spacing;
                    }

                    // Provenance for the appended rows. birth_tri must be -1 wherever
                    // mesh_hole_fill_mask is true; strategy_utils validates that pairing.
                    const auto point_rows = static_cast<size_t>(point_model->size());
                    const auto device = point_model->means().device();
                    point_model->birth_tri() = lfs::core::Tensor::full(
                        {point_rows}, -1.0f, device, lfs::core::DataType::Int32);
                    point_model->mesh_hole_fill_mask() =
                        lfs::core::Tensor::ones_bool({point_rows}, device);
                    point_model->mesh_init_mask() =
                        lfs::core::Tensor::zeros_bool({point_rows}, device);

                    const size_t mesh_rows = static_cast<size_t>(model->size());
                    auto combined = lfs::core::concat(*model, *point_model);
                    if (!combined) {
                        return std::unexpected(std::format(
                            "Failed to append external point-cloud Gaussians: {}", combined.error()));
                    }
                    *model = std::move(*combined);

                    pc_stats.actual_spacing =
                        pc_stats.kept > 0 && pc_stats.target_spacing > 0.0f
                            ? pc_stats.target_spacing
                            : 0.0f;
                    LOG_INFO("mesh_init external point cloud: '{}' loaded={}, after_density_match={}, "
                             "rejected_near_surface={}, kept={}, mesh_spacing={:.6g}, target_spacing={:.6g}, "
                             "bbox_overlap={:.4g}",
                             lfs::core::path_to_utf8(cloud_path),
                             pc_stats.loaded,
                             pc_stats.after_density_match,
                             pc_stats.rejected_near_surface,
                             pc_stats.kept,
                             pc_stats.mesh_spacing,
                             pc_stats.target_spacing,
                             pc_stats.bbox_overlap_ratio);
                    LOG_INFO("mesh_init external point cloud: {} mesh Gaussians + {} point Gaussians = {} total "
                             "(point rows are white, unconstrained, opacity-trainable)",
                             mesh_rows, point_rows, model->size());

                    if (pc_stats.mesh_spacing > 0.0f &&
                        model->get_mesh2splat_mean_max_scale() > 0.0f) {
                        const float ratio =
                            pc_stats.mesh_spacing / model->get_mesh2splat_mean_max_scale();
                        if (ratio > 10.0f || ratio < 0.1f) {
                            LOG_WARN("mesh_init external point cloud: derived mesh spacing {:.6g} differs from "
                                     "mesh2splat mean max scale {:.6g} by {:.4g}x; check that the point cloud "
                                     "and mesh really share a coordinate system",
                                     pc_stats.mesh_spacing,
                                     model->get_mesh2splat_mean_max_scale(),
                                     ratio);
                        }
                    }
                }

                // Override SH0 to white if requested (skip PBR-baked colors).
                if (params.init_white_gs_color) {
                    constexpr float SH_C0_INV = 0.5f / 0.28209479177387814f; // ≈ 1.7725
                    auto& sh0 = model->sh0(); // [N, 1, 3]
                    sh0.fill_(SH_C0_INV);
                    LOG_INFO("mesh_init: overrode {} Gaussian colors to white (--init-white-gs-color)", model->size());
                }

                // Extract point cloud from mesh2splat result for downstream consumers.
                // Build manually because lfs::io::to_point_cloud doesn't populate .colors.
                if (init_point_cloud_out) {
                    constexpr float SH_C0 = 0.28209479177387814f;
                    auto means_cpu = model->means().cpu().contiguous();
                    // SH0 is [N, 1, 3] — squeeze to [N, 3], invert SH DC: rgb = sh0 * SH_C0 + 0.5
                    auto sh0_cpu = model->sh0().cpu().contiguous();
                    auto rgb = sh0_cpu.reshape({static_cast<int64_t>(model->size()), 3})
                                   .mul(SH_C0)
                                   .add(0.5f)
                                   .clamp(0.0f, 1.0f)
                                   .mul(255.0f)
                                   .to(lfs::core::DataType::UInt8);
                    *init_point_cloud_out = std::make_shared<lfs::core::PointCloud>(
                        std::move(means_cpu), std::move(rgb));
                }

                LOG_INFO("Initialized {} Gaussians from mesh {} via mesh2splat",
                         model->size(), lfs::core::path_to_utf8(mesh_init_path.filename()));
                return model;
            }

            const auto ext = init_file.extension().string();
            if (ext == ".ply" && !lfs::io::is_gaussian_splat_ply(init_file)) {
                auto pc_result = lfs::io::load_ply_point_cloud(init_file);
                if (!pc_result) {
                    return std::unexpected(std::format(
                        "Failed to load '{}': {}",
                        lfs::core::path_to_utf8(init_file),
                        pc_result.error()));
                }

                if (init_point_cloud_out) {
                    *init_point_cloud_out = std::make_shared<lfs::core::PointCloud>(*pc_result);
                }

                auto splat_result = lfs::core::init_model_from_pointcloud(
                    params,
                    scene_center,
                    *pc_result,
                    static_cast<int>(pc_result->size()));

                if (!splat_result) {
                    return std::unexpected(std::format("Init failed: {}", splat_result.error()));
                }

                auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
                LOG_INFO("Initialized {} Gaussians from {}",
                         model->size(), lfs::core::path_to_utf8(init_file.filename()));
                return model;
            }

            auto loader = lfs::io::Loader::create();
            auto splat_load_result = loader->load(init_file);
            if (!splat_load_result) {
                return std::unexpected(std::format(
                    "Failed to load '{}': {}",
                    lfs::core::path_to_utf8(init_file),
                    splat_load_result.error().format()));
            }

            try {
                auto splat_data = std::move(*std::get<std::shared_ptr<lfs::core::SplatData>>(splat_load_result->data));
                auto model = std::make_unique<lfs::core::SplatData>(std::move(splat_data));

                const int target_sh = params.optimization.sh_degree;
                if (target_sh >= 0 && target_sh < model->get_max_sh_degree()) {
                    LOG_INFO("Truncating SH: {} -> {}", model->get_max_sh_degree(), target_sh);
                    truncateSHDegree(*model, target_sh);
                }

                LOG_INFO("Loaded {} Gaussians from {} (sh={})",
                         model->size(), lfs::core::path_to_utf8(init_file.filename()), model->get_max_sh_degree());
                return model;
            } catch (const std::bad_variant_access&) {
                return std::unexpected(std::format("'{}': invalid SplatData", lfs::core::path_to_utf8(init_file)));
            }
        }
    } // namespace

    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        auto data_loader = lfs::io::Loader::create();

        const auto& data_path = params.dataset.data_path;
        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = false,
            .progress = [&data_path](float percentage, const std::string& message) {
                LOG_DEBUG("[{:5.1f}%] {}", percentage, message);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = data_path,
                    .progress = percentage,
                    .step = message}
                    .emit();
            }};

        LOG_INFO("Loading dataset from: {}", lfs::core::path_to_utf8(params.dataset.data_path));
        auto load_result = data_loader->load(params.dataset.data_path, load_options);
        if (!load_result) {
            return std::unexpected(std::format("Failed to load dataset: {}", load_result.error().format()));
        }

        LOG_INFO("Dataset loaded successfully using {} loader", load_result->loader_used);
        scene.setProjectMaskMeshOverride(nullptr);
        scene.setProjectMaskPointCloudOverride(nullptr, 0.0f);

        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                LOG_INFO("Loaded PLY directly into scene");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result->scene_center);
                scene.setImagesHaveAlpha(load_result->images_have_alpha);

                // Build dataset hierarchy in scene graph
                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (params.use_mesh_init || params.init_path.has_value()) {
                    const std::filesystem::path init_file = params.use_mesh_init
                                                               ? lfs::core::utf8_to_path(params.mesh_init_mesh_path.value())
                                                               : lfs::core::utf8_to_path(params.init_path.value());
                    std::shared_ptr<lfs::core::PointCloud> mesh_init_point_cloud;
                    std::shared_ptr<lfs::core::MeshData> mesh_init_mesh;
                    std::shared_ptr<lfs::core::MeshData> project_mask_mesh;
                    std::shared_ptr<lfs::core::PointCloud> project_mask_point_cloud;
                    float project_mask_point_spacing = 0.0f;
                    auto model_result = create_init_model(
                        params,
                        load_result->scene_center,
                        init_file,
                        select_train_cameras(data.cameras, params.optimization, params.dataset.test_every),
                        params.use_mesh_init ? &mesh_init_point_cloud : nullptr,
                        params.use_mesh_init ? &mesh_init_mesh : nullptr,
                        params.use_mesh_init ? &project_mask_mesh : nullptr,
                        params.use_mesh_init ? &project_mask_point_cloud : nullptr,
                        params.use_mesh_init ? &project_mask_point_spacing : nullptr);
                    if (!model_result) {
                        return std::unexpected(model_result.error());
                    }

                    if (mesh_init_mesh) {
                        std::string mesh_name = lfs::core::path_to_utf8(init_file.stem());
                        if (mesh_name.empty()) {
                            mesh_name = "mesh_init";
                        }
                        scene.addMesh(mesh_name, mesh_init_mesh, dataset_id);
                        LOG_INFO("Attached mesh init '{}' to scene for mesh supervision", mesh_name);
                    }
                    scene.setProjectMaskMeshOverride(std::move(project_mask_mesh));
                    scene.setProjectMaskPointCloudOverride(
                        std::move(project_mask_point_cloud), project_mask_point_spacing);

                    if (mesh_init_point_cloud && mesh_init_point_cloud->size() > 0) {
                        scene.setInitialPointCloud(mesh_init_point_cloud);
                        LOG_INFO("Using {} mesh-derived points/colors as in-memory sparse fallback",
                                 mesh_init_point_cloud->size());
                    }

                    scene.addSplat("Model", std::move(*model_result), dataset_id);
                    scene.setTrainingModelNode("Model");
                } else {
                    if (data.point_cloud && data.point_cloud->size() > 0) {
                        LOG_INFO("Adding {} points to scene", data.point_cloud->size());
                        scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                    } else {
                        LOG_INFO("No point cloud, using random initialization");
                        auto pc = createRandomPointCloud();
                        LOG_INFO("Adding {} random points to scene", pc->size());
                        scene.addPointCloud("PointCloud", pc, dataset_id);
                    }
                }

                const auto& cameras = data.cameras;
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;
                const auto explicit_split_sets = build_explicit_split_sets(params.optimization);

                size_t train_count = 0;
                size_t val_count = 0;
                size_t excluded_count = 0;
                size_t mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_train = !enable_eval ||
                                          (explicit_split_sets
                                               ? split_set_contains_camera(explicit_split_sets->train_images, cameras[i])
                                               : (i % test_every) != 0);
                    const bool is_val = enable_eval &&
                                        (explicit_split_sets
                                             ? split_set_contains_camera(explicit_split_sets->test_images, cameras[i])
                                             : (i % test_every) == 0);

                    if (is_train) {
                        ++train_count;
                    } else if (is_val) {
                        ++val_count;
                    } else {
                        ++excluded_count;
                    }

                    if ((is_train || is_val) && cameras[i]->has_mask()) {
                        mask_count++;
                    }
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);

                const auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count),
                    cameras_group_id,
                    train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_train = !enable_eval ||
                                          (explicit_split_sets
                                               ? split_set_contains_camera(explicit_split_sets->train_images, cameras[i])
                                               : (i % test_every) != 0);
                    if (is_train) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count),
                        cameras_group_id,
                        val_count);

                    for (size_t i = 0; i < cameras.size(); ++i) {
                        const bool is_val = explicit_split_sets
                                                ? split_set_contains_camera(explicit_split_sets->test_images, cameras[i])
                                                : (i % test_every) == 0;
                        if (is_val) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                const std::string eval_summary = enable_eval ? std::format(" + {} val", val_count) : "";
                const std::string excluded_summary = excluded_count > 0
                                                         ? std::format(" ({} excluded by explicit split)", excluded_count)
                                                         : "";
                const std::string mask_summary = mask_count > 0
                                                     ? std::format(" ({} with masks)", mask_count)
                                                     : "";

                LOG_INFO("Loaded dataset '{}' into scene: {} train{} cameras{}{}",
                         dataset_name, train_count,
                         eval_summary,
                         excluded_summary,
                         mask_summary);
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";

                if (const auto normalize_result = normalize_mesh_positions_for_scene(
                        *data, MESH_SCENE_POSITION_DIVISOR);
                    !normalize_result) {
                    return std::unexpected(std::format("Mesh scene normalization failed: {}", normalize_result.error()));
                }
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene (xyz normalized by 1/{})",
                         mesh_name,
                         MESH_SCENE_POSITION_DIVISOR);
                return {};

            } else {
                return std::unexpected("Unknown data type returned from loader");
            }
        },
                          load_result->data);
    }

    std::expected<void, std::string> initializeTrainingModel(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene) {

        if (scene.getTrainingModel() != nullptr) {
            return {};
        }

        lfs::core::NodeId point_cloud_node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
        const lfs::core::PointCloud* point_cloud = nullptr;
        glm::mat4 node_transform{1.0f};

        for (const auto* node : scene.getNodes()) {
            if (node->type == lfs::core::NodeType::POINTCLOUD && node->point_cloud) {
                point_cloud_node_id = node->id;
                parent_id = node->parent_id;
                node_transform = node->transform();
                point_cloud = node->point_cloud.get();
                break;
            }
        }

        lfs::core::PointCloud point_cloud_to_use;
        const int max_cap = params.optimization.max_cap;

        if (point_cloud && point_cloud->size() > 0) {
            const lfs::core::CropBoxData* cropbox_data = nullptr;
            lfs::core::NodeId cropbox_id = lfs::core::NULL_NODE;

            if (point_cloud_node_id != lfs::core::NULL_NODE) {
                cropbox_id = scene.getCropBoxForSplat(point_cloud_node_id);
                if (cropbox_id != lfs::core::NULL_NODE) {
                    cropbox_data = scene.getCropBoxData(cropbox_id);
                }
            }

            if (cropbox_data && cropbox_data->enabled) {
                const glm::mat4 world_to_cropbox = glm::inverse(scene.getWorldTransform(cropbox_id));
                const auto& means = point_cloud->means;
                const auto& colors = point_cloud->colors;
                const size_t num_points = point_cloud->size();

                auto means_cpu = means.cpu();
                auto colors_cpu = colors.cpu();
                const float* means_ptr = means_cpu.ptr<float>();
                const uint8_t* colors_ptr = colors_cpu.ptr<uint8_t>();

                std::vector<float> filtered_means;
                std::vector<uint8_t> filtered_colors;
                filtered_means.reserve(num_points * 3);
                filtered_colors.reserve(num_points * 3);

                for (size_t i = 0; i < num_points; ++i) {
                    const glm::vec3 pos(means_ptr[i * 3], means_ptr[i * 3 + 1], means_ptr[i * 3 + 2]);
                    const glm::vec4 local_pos = world_to_cropbox * glm::vec4(pos, 1.0f);
                    const glm::vec3 local = glm::vec3(local_pos) / local_pos.w;

                    bool inside = local.x >= cropbox_data->min.x && local.x <= cropbox_data->max.x &&
                                  local.y >= cropbox_data->min.y && local.y <= cropbox_data->max.y &&
                                  local.z >= cropbox_data->min.z && local.z <= cropbox_data->max.z;

                    if (cropbox_data->inverse)
                        inside = !inside;

                    if (inside) {
                        filtered_means.push_back(means_ptr[i * 3]);
                        filtered_means.push_back(means_ptr[i * 3 + 1]);
                        filtered_means.push_back(means_ptr[i * 3 + 2]);
                        filtered_colors.push_back(colors_ptr[i * 3]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 1]);
                        filtered_colors.push_back(colors_ptr[i * 3 + 2]);
                    }
                }

                const size_t filtered_count = filtered_means.size() / 3;
                LOG_INFO("CropBox filtering: {} -> {} points", num_points, filtered_count);

                if (filtered_count == 0) {
                    return std::unexpected("CropBox filtered out all points");
                }

                auto filtered_means_tensor = lfs::core::Tensor::from_vector(
                    filtered_means, {filtered_count, 3}, lfs::core::Device::CPU);
                auto filtered_colors_tensor = lfs::core::Tensor::zeros(
                    {filtered_count, 3}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                std::memcpy(filtered_colors_tensor.data_ptr(), filtered_colors.data(),
                            filtered_colors.size() * sizeof(uint8_t));

                point_cloud_to_use = lfs::core::PointCloud(filtered_means_tensor, filtered_colors_tensor);
            } else {
                point_cloud_to_use = *point_cloud;
                if (max_cap > 0) {
                    point_cloud_to_use.means = point_cloud_to_use.means.cpu();
                    point_cloud_to_use.colors = point_cloud_to_use.colors.cpu();
                }
            }
        } else {
            LOG_INFO("No point cloud provided, using random initialization");
            point_cloud_to_use = *createRandomPointCloud();
        }

        lfs::core::Tensor scene_center = scene.getSceneCenter();
        if (!scene_center.is_valid() || scene_center.numel() == 0) {
            LOG_WARN("No scene center from loader, computing from point cloud");
            if (point_cloud_to_use.size() > 0) {
                auto means_cpu = point_cloud_to_use.means.cpu();
                auto mean = means_cpu.mean({0});
                scene_center = max_cap > 0 ? mean : mean.cuda();
            } else {
                scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            }
        } else {
            scene_center = max_cap > 0 ? scene_center.cpu() : scene_center.cuda();
        }

        auto splat_result = lfs::core::init_model_from_pointcloud(
            params, scene_center, point_cloud_to_use, max_cap);

        if (!splat_result) {
            return std::unexpected(std::format("Failed to initialize model: {}", splat_result.error()));
        }

        if (max_cap > 0 && max_cap < static_cast<int>(splat_result->size())) {
            LOG_WARN("Max cap ({}) is less than initial splat count ({}), randomly selecting {} splats",
                     max_cap, splat_result->size(), max_cap);
            lfs::core::random_choose(*splat_result, max_cap);
        }

        if (point_cloud_node_id != lfs::core::NULL_NODE) {
            if (const auto* pc_node = scene.getNodeById(point_cloud_node_id)) {
                scene.removeNode(pc_node->name, false);
            }
        }

        auto model = std::make_unique<lfs::core::SplatData>(std::move(*splat_result));
        LOG_INFO("Created training model with {} gaussians", model->size());
        scene.addSplat("Model", std::move(model), parent_id);
        if (node_transform != glm::mat4{1.0f}) {
            scene.setNodeTransform("Model", node_transform);
        }
        scene.setTrainingModelNode("Model");

        return {};
    }

    std::expected<void, std::string> validateDatasetPath(
        const lfs::core::param::TrainingParameters& params) {

        auto data_loader = lfs::io::Loader::create();

        lfs::io::LoadOptions load_options{
            .resize_factor = params.dataset.resize_factor,
            .max_width = params.dataset.max_width,
            .images_folder = params.dataset.images,
            .validate_only = true};

        auto result = data_loader->load(params.dataset.data_path, load_options);
        if (!result) {
            return std::unexpected(result.error().format());
        }
        return {};
    }

    std::expected<void, std::string> applyLoadResultToScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::core::Scene& scene,
        lfs::io::LoadResult&& load_result) {

        scene.setProjectMaskMeshOverride(nullptr);
        scene.setProjectMaskPointCloudOverride(nullptr, 0.0f);
        return std::visit([&](auto&& data) -> std::expected<void, std::string> {
            using T = std::decay_t<decltype(data)>;

            if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::SplatData>>) {
                auto model = std::make_unique<lfs::core::SplatData>(std::move(*data));
                scene.addSplat("loaded_model", std::move(model));
                scene.setTrainingModelNode("loaded_model");
                return {};

            } else if constexpr (std::is_same_v<T, lfs::io::LoadedScene>) {
                scene.setInitialPointCloud(data.point_cloud);
                scene.setSceneCenter(load_result.scene_center);
                scene.setImagesHaveAlpha(load_result.images_have_alpha);

                std::string dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.filename());
                if (dataset_name.empty()) {
                    dataset_name = lfs::core::path_to_utf8(params.dataset.data_path.parent_path().filename());
                }
                if (dataset_name.empty()) {
                    dataset_name = "Dataset";
                }

                const auto dataset_id = scene.addDataset(dataset_name);

                if (params.use_mesh_init || params.init_path.has_value()) {
                    const std::filesystem::path init_file = params.use_mesh_init
                                                               ? lfs::core::utf8_to_path(params.mesh_init_mesh_path.value())
                                                               : lfs::core::utf8_to_path(params.init_path.value());
                    std::shared_ptr<lfs::core::PointCloud> mesh_init_point_cloud;
                    std::shared_ptr<lfs::core::MeshData> mesh_init_mesh;
                    std::shared_ptr<lfs::core::MeshData> project_mask_mesh;
                    std::shared_ptr<lfs::core::PointCloud> project_mask_point_cloud;
                    float project_mask_point_spacing = 0.0f;
                    auto model_result = create_init_model(
                        params,
                        load_result.scene_center,
                        init_file,
                        select_train_cameras(data.cameras, params.optimization, params.dataset.test_every),
                        params.use_mesh_init ? &mesh_init_point_cloud : nullptr,
                        params.use_mesh_init ? &mesh_init_mesh : nullptr,
                        params.use_mesh_init ? &project_mask_mesh : nullptr,
                        params.use_mesh_init ? &project_mask_point_cloud : nullptr,
                        params.use_mesh_init ? &project_mask_point_spacing : nullptr);
                    if (!model_result) {
                        return std::unexpected(model_result.error());
                    }

                    if (mesh_init_mesh) {
                        std::string mesh_name = lfs::core::path_to_utf8(init_file.stem());
                        if (mesh_name.empty()) {
                            mesh_name = "mesh_init";
                        }
                        scene.addMesh(mesh_name, mesh_init_mesh, dataset_id);
                        LOG_INFO("Attached mesh init '{}' to scene for mesh supervision", mesh_name);
                    }
                    scene.setProjectMaskMeshOverride(std::move(project_mask_mesh));
                    scene.setProjectMaskPointCloudOverride(
                        std::move(project_mask_point_cloud), project_mask_point_spacing);

                    if (mesh_init_point_cloud && mesh_init_point_cloud->size() > 0) {
                        scene.setInitialPointCloud(mesh_init_point_cloud);
                        LOG_INFO("Using {} mesh-derived points/colors as in-memory sparse fallback",
                                 mesh_init_point_cloud->size());
                    }

                    scene.addSplat("Model", std::move(*model_result), dataset_id);
                    scene.setTrainingModelNode("Model");
                } else if (data.point_cloud && data.point_cloud->size() > 0) {
                    scene.addPointCloud("PointCloud", data.point_cloud, dataset_id);
                } else {
                    scene.addPointCloud("PointCloud", createRandomPointCloud(), dataset_id);
                }

                const auto& cameras = data.cameras;
                const bool enable_eval = params.optimization.enable_eval;
                const int test_every = params.dataset.test_every;
                const auto explicit_split_sets = build_explicit_split_sets(params.optimization);

                size_t train_count = 0, val_count = 0, excluded_count = 0, mask_count = 0;
                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_train = !enable_eval ||
                                          (explicit_split_sets
                                               ? split_set_contains_camera(explicit_split_sets->train_images, cameras[i])
                                               : (i % test_every) != 0);
                    const bool is_val = enable_eval &&
                                        (explicit_split_sets
                                             ? split_set_contains_camera(explicit_split_sets->test_images, cameras[i])
                                             : (i % test_every) == 0);

                    if (is_train) {
                        ++train_count;
                    } else if (is_val) {
                        ++val_count;
                    } else {
                        ++excluded_count;
                    }
                    if ((is_train || is_val) && cameras[i]->has_mask())
                        ++mask_count;
                }

                const auto cameras_group_id = scene.addGroup("Cameras", dataset_id);
                const auto train_cameras_id = scene.addCameraGroup(
                    std::format("Training ({})", train_count), cameras_group_id, train_count);

                for (size_t i = 0; i < cameras.size(); ++i) {
                    const bool is_train = !enable_eval ||
                                          (explicit_split_sets
                                               ? split_set_contains_camera(explicit_split_sets->train_images, cameras[i])
                                               : (i % test_every) != 0);
                    if (is_train) {
                        scene.addCamera(cameras[i]->image_name(), train_cameras_id, cameras[i]);
                    }
                }

                if (enable_eval && val_count > 0) {
                    const auto val_cameras_id = scene.addCameraGroup(
                        std::format("Validation ({})", val_count), cameras_group_id, val_count);
                    for (size_t i = 0; i < cameras.size(); ++i) {
                        const bool is_val = explicit_split_sets
                                                ? split_set_contains_camera(explicit_split_sets->test_images, cameras[i])
                                                : (i % test_every) == 0;
                        if (is_val) {
                            scene.addCamera(cameras[i]->image_name(), val_cameras_id, cameras[i]);
                        }
                    }
                }

                const std::string eval_summary = enable_eval ? std::format(" + {} val", val_count) : "";
                const std::string excluded_summary = excluded_count > 0
                                                         ? std::format(" ({} excluded by explicit split)", excluded_count)
                                                         : "";
                const std::string mask_summary = mask_count > 0
                                                     ? std::format(" ({} masked)", mask_count)
                                                     : "";

                LOG_INFO("Dataset '{}': {} train{} cameras{}{}",
                         dataset_name, train_count,
                         eval_summary,
                         excluded_summary,
                         mask_summary);
                return {};

            } else if constexpr (std::is_same_v<T, std::shared_ptr<lfs::core::MeshData>>) {
                assert(data && "MeshData must not be null");
                std::string mesh_name = lfs::core::path_to_utf8(params.dataset.data_path.stem());
                if (mesh_name.empty())
                    mesh_name = "mesh";

                if (const auto normalize_result = normalize_mesh_positions_for_scene(
                        *data, MESH_SCENE_POSITION_DIVISOR);
                    !normalize_result) {
                    return std::unexpected(std::format("Mesh scene normalization failed: {}", normalize_result.error()));
                }
                scene.addMesh(mesh_name, data);
                LOG_INFO("Loaded mesh '{}' into scene (xyz normalized by 1/{})",
                         mesh_name,
                         MESH_SCENE_POSITION_DIVISOR);
                return {};

            } else {
                return std::unexpected("Unknown data type from loader");
            }
        },
                          load_result.data);
    }

} // namespace lfs::training
