/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/mesh_geometry.hpp"
#include "scipy_nearest_tree.hpp"
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
#include <algorithm>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <cmath>
#include <fstream>
#include <limits>
#include <nanoflann.hpp>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <set>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace lfs::preprocess {
    namespace {
        using Vec3 = std::array<double, 3>;
        struct Failure {
            Error error;
        };
        [[noreturn]] void fail(ErrorCode code, const std::string& message) { throw Failure{{code, message}}; }
        void check_stop(const ExecutionContext& ctx) {
            if (ctx.stop_token.stop_requested())
                fail(ErrorCode::ProcessCancelled, "Mesh geometry cancelled");
        }
        void progress(const ExecutionContext& ctx, float value, const std::string& message) {
            check_stop(ctx);
            try {
                if (ctx.on_progress && !ctx.on_progress(ctx.stage, value, message))
                    fail(ErrorCode::ProcessCancelled, "Mesh geometry cancelled by progress callback");
            } catch (const Failure&) { throw; } catch (const std::exception& e) {
                fail(ErrorCode::CallbackFailure, e.what());
            } catch (...) { fail(ErrorCode::CallbackFailure, "Mesh geometry progress callback failed"); }
        }
        Vec3 subtract(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
        double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
        bool finite(const Vec3& a) { return std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]); }
        Vec3 transform(const std::array<double, 9>& r, const Vec3& p) {
            return {r[0] * p[0] + r[1] * p[1] + r[2] * p[2], r[3] * p[0] + r[4] * p[1] + r[5] * p[2], r[6] * p[0] + r[7] * p[1] + r[8] * p[2]};
        }
        Vec3 transpose_transform(const std::array<double, 9>& r, const Vec3& p) {
            return {r[0] * p[0] + r[3] * p[1] + r[6] * p[2], r[1] * p[0] + r[4] * p[1] + r[7] * p[2], r[2] * p[0] + r[5] * p[1] + r[8] * p[2]};
        }
        Vec3 normalize(Vec3 p) {
            const double n = std::sqrt(dot(p, p));
            for (auto& v : p)
                v /= n;
            return p;
        }
        void validate_camera(const ColmapCamera& camera, const ColmapImage& pose) {
            if (camera.model != 1 || camera.parameters.size() != 4 || camera.width == 0 || camera.height == 0 ||
                camera.width > std::numeric_limits<int>::max() || camera.height > std::numeric_limits<int>::max())
                fail(ErrorCode::InvalidDataset, "Mesh projection requires a valid PINHOLE camera");
            if (!std::all_of(camera.parameters.begin(), camera.parameters.end(), [](double v) { return std::isfinite(v); }) ||
                camera.parameters[0] <= 0 || camera.parameters[1] <= 0 || !finite(pose.translation) ||
                !std::all_of(pose.rotation.begin(), pose.rotation.end(), [](double v) { return std::isfinite(v); }))
                fail(ErrorCode::InvalidDataset, "Invalid mesh projection camera or pose");
            const double qnorm = std::inner_product(pose.rotation.begin(), pose.rotation.end(), pose.rotation.begin(), 0.0);
            if (std::abs(qnorm - 1.0) > 1e-5)
                fail(ErrorCode::InvalidDataset, "Mesh projection quaternion is not normalized");
        }
        void validate_mesh(const CpuMesh& mesh, bool require_triangles) {
            if (mesh.vertices.empty() || (require_triangles && mesh.triangles.empty()))
                fail(ErrorCode::InvalidDataset, "Mesh contains no usable vertices/triangles");
            for (const auto& vertex : mesh.vertices)
                if (!finite(vertex))
                    fail(ErrorCode::InvalidDataset, "Mesh contains a non-finite vertex");
            for (const auto& triangle : mesh.triangles)
                for (auto index : triangle)
                    if (index >= mesh.vertices.size())
                        fail(ErrorCode::InvalidDataset, "Mesh triangle index is out of bounds");
        }
        template <class F>
        auto guarded(F&& action) -> std::expected<decltype(action()), Error> {
            try {
                return action();
            } catch (const Failure& error) { return std::unexpected(error.error); } catch (const std::exception& error) {
                return std::unexpected(Error{ErrorCode::InvalidDataset, error.what()});
            }
        }
        using Kernel = CGAL::Simple_cartesian<double>;
        using Triangle = Kernel::Triangle_3;
        using Primitive = CGAL::AABB_triangle_primitive_3<Kernel, std::vector<Triangle>::const_iterator>;
        using Tree = CGAL::AABB_tree<CGAL::AABB_traits_3<Kernel, Primitive>>;
        Kernel::Point_3 point(const Vec3& p) { return {p[0], p[1], p[2]}; }
        struct Raycaster {
            std::vector<Triangle> triangles;
            Tree tree;
            Raycaster(const CpuMesh& mesh, const ExecutionContext& ctx) {
                triangles.reserve(mesh.triangles.size());
                for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                    if ((i & 4095) == 0)
                        check_stop(ctx);
                    const auto& f = mesh.triangles[i];
                    // Open3D Tensor geometry and submitted rays are Float32.
                    auto fp = [&](uint32_t index) {auto v=mesh.vertices[index];for(auto& x:v)x=static_cast<float>(x);return point(v); };
                    Triangle triangle(fp(f[0]), fp(f[1]), fp(f[2]));
                    if (!triangle.is_degenerate())
                        triangles.push_back(triangle);
                }
                if (triangles.empty())
                    fail(ErrorCode::InvalidDataset, "Mesh contains only degenerate triangles");
                tree.insert(triangles.cbegin(), triangles.cend());
                tree.build(); // Finish lazy construction before parallel read-only queries.
                check_stop(ctx);
            }
            std::optional<Vec3> hit(const Vec3& origin, const Vec3& direction) const {
                Vec3 float_origin = origin, float_direction = direction;
                for (auto& x : float_origin)
                    x = static_cast<float>(x);
                for (auto& x : float_direction)
                    x = static_cast<float>(x);
                const auto hit = tree.first_intersection(Kernel::Ray_3(point(float_origin), Kernel::Vector_3(float_direction[0], float_direction[1], float_direction[2])));
                if (!hit)
                    return {};
                double distance = std::numeric_limits<double>::infinity();
                auto measure = [&](const Kernel::Point_3& p) {
                    const Vec3 delta{p.x() - float_origin[0], p.y() - float_origin[1], p.z() - float_origin[2]};
                    const double t = dot(delta, float_direction) / dot(float_direction, float_direction);
                    if (t >= 0)
                        distance = std::min(distance, t);
                };
                if (const auto* p = std::get_if<Kernel::Point_3>(&hit->first))
                    measure(*p);
                else if (const auto* s = std::get_if<Kernel::Segment_3>(&hit->first)) {
                    measure(s->source());
                    measure(s->target());
                }
                const float t = static_cast<float>(distance);
                if (!std::isfinite(t))
                    return {};
                return Vec3{origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t};
            }
        };
        struct Cloud {
            std::vector<Vec3> points;
            std::size_t kdtree_get_point_count() const { return points.size(); }
            double kdtree_get_pt(std::size_t index, std::size_t dimension) const { return points[index][dimension]; }
            template <class B>
            bool kdtree_get_bbox(B&) const { return false; }
        };
        using Index = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Cloud>, Cloud, 3, std::size_t>;
        struct ImageHits {
            uint32_t id = 0;
            const Keypoints* keypoints = nullptr;
            Cloud cloud;
            std::vector<uint32_t> indices;
            std::vector<std::array<uint8_t, 3>> colors;
            bool colors_valid = false;
            std::unique_ptr<detail::ScipyNearestTree> tree;
        };
        struct UnionFind {
            std::vector<std::size_t> parent;
            explicit UnionFind(std::size_t size) : parent(size) { std::iota(parent.begin(), parent.end(), 0); }
            std::size_t find(std::size_t index) {
                while (parent[index] != index) {
                    parent[index] = parent[parent[index]];
                    index = parent[index];
                }
                return index;
            }
            void join(std::size_t first, std::size_t second) {
                first = find(first);
                second = find(second);
                if (first != second)
                    parent[std::max(first, second)] = std::min(first, second);
            }
        };
    } // namespace

    std::expected<CpuMesh, Error> read_mesh_geometry(const std::filesystem::path& path, double scale, const ExecutionContext& ctx) {
        return guarded([&] {
            check_stop(ctx);
            if (!std::isfinite(scale) || scale <= 0)
                fail(ErrorCode::InvalidRequest, "Mesh coordinate scale must be positive");
            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            if (!stream)
                fail(ErrorCode::InvalidDataset, "Cannot read scanner mesh");
            const auto length = stream.tellg();
            if (length <= 0)
                fail(ErrorCode::InvalidDataset, "Scanner mesh is empty");
            stream.seekg(0);
            std::vector<char> buffer(static_cast<std::size_t>(length));
            if (!stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size())))
                fail(ErrorCode::IoFailure, "Cannot read scanner mesh payload");
            check_stop(ctx);
            Assimp::Importer importer;
            const auto extension = path.extension().string();
            const auto* scene = importer.ReadFileFromMemory(buffer.data(), buffer.size(), aiProcess_Triangulate | aiProcess_PreTransformVertices,
                                                            extension.empty() ? "" : extension.c_str() + 1);
            if (!scene || !scene->HasMeshes())
                fail(ErrorCode::InvalidDataset, std::string("Cannot decode scanner mesh: ") + importer.GetErrorString());
            CpuMesh result;
            for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
                check_stop(ctx);
                const auto* mesh = scene->mMeshes[m];
                const auto offset = result.vertices.size();
                if (offset + mesh->mNumVertices > std::numeric_limits<uint32_t>::max())
                    fail(ErrorCode::InvalidDataset, "Scanner mesh has too many vertices");
                for (unsigned v = 0; v < mesh->mNumVertices; ++v) {
                    const auto p = mesh->mVertices[v];
                    result.vertices.push_back({p.x * scale, p.y * scale, p.z * scale});
                }
                for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
                    const auto& face = mesh->mFaces[f];
                    if (face.mNumIndices != 3)
                        continue;
                    for (unsigned k = 0; k < 3; ++k)
                        if (face.mIndices[k] >= mesh->mNumVertices)
                            fail(ErrorCode::InvalidDataset, "Invalid imported mesh face");
                    result.triangles.push_back({static_cast<uint32_t>(offset + face.mIndices[0]), static_cast<uint32_t>(offset + face.mIndices[1]), static_cast<uint32_t>(offset + face.mIndices[2])});
                }
            }
            validate_mesh(result, true);
            check_stop(ctx);
            return result;
        });
    }

    std::expected<CpuImage, Error> project_mesh_overlay(const CpuMesh& mesh, const CpuImage& image, const ColmapCamera& camera,
                                                        const ColmapImage& pose, const ExecutionContext& ctx) {
        return guarded([&] {
            check_stop(ctx);
            validate_mesh(mesh, false);
            validate_camera(camera, pose);
            const auto width = static_cast<int>(camera.width), height = static_cast<int>(camera.height);
            const auto pixels = static_cast<std::size_t>(width) * height;
            if (image.width != width || image.height != height || image.channels != 3 ||
                pixels > std::numeric_limits<std::size_t>::max() / 3 || image.pixels.size() != pixels * 3)
                fail(ErrorCode::InvalidDataset, "Mesh overlay requires an RGB image matching the camera dimensions");
            progress(ctx, 0.F, "Project mesh debug overlay");
            CpuImage result = image;
            std::vector<uint8_t> touched(pixels, 0);
            const auto rotation = quaternion_rotation(pose.rotation);
            // NumPy round uses ties-to-even, unlike std::round. Do not depend on the
            // process floating-point rounding mode (nearbyint/rint) for pixel identity.
            const auto round_even = [](double value) {
                const auto lower = static_cast<int64_t>(std::floor(value));
                const auto fraction = value - static_cast<double>(lower);
                return lower + (fraction > .5 || (fraction == .5 && (lower & 1)) ? 1 : 0);
            };
            for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
                if ((i & 4095) == 0)
                    progress(ctx, static_cast<float>(i) / mesh.vertices.size(), "Project mesh debug overlay");
                auto p = transform(rotation, mesh.vertices[i]);
                for (int k = 0; k < 3; ++k)
                    p[k] += pose.translation[k];
                if (!(p[2] > 0))
                    continue;
                // Divide before scaling, matching frozen NumPy's separate x/y arrays.
                const double u = camera.parameters[0] * (p[0] / p[2]) + camera.parameters[2];
                const double v = camera.parameters[1] * (p[1] / p[2]) + camera.parameters[3];
                if (!(u >= 0 && u < width && v >= 0 && v < height))
                    continue;
                const auto x = std::min<int64_t>(round_even(u), width - 1);
                const auto y = std::min<int64_t>(round_even(v), height - 1);
                const auto pixel = static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
                if (touched[pixel])
                    continue;
                touched[pixel] = 1;
                // Pillow alpha_composite on an opaque RGB base. Each projected pixel
                // is assigned once, even for coincident or occluded mesh vertices.
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const unsigned green = channel == 1 ? 255 : 0;
                    result.pixels[pixel * 3 + channel] = static_cast<uint8_t>((95U * image.pixels[pixel * 3 + channel] + 160U * green + 127U) / 255U);
                }
            }
            progress(ctx, 1.F, "Mesh debug overlay complete");
            return result;
        });
    }

    std::expected<CpuImage, Error> project_mesh_mask(const CpuMesh& mesh, const ColmapCamera& camera, const ColmapImage& pose,
                                                     int erosion, const ExecutionContext& ctx) {
        return guarded([&] {
            check_stop(ctx);
            validate_mesh(mesh, false);
            validate_camera(camera, pose);
            if (erosion < 0 || erosion > 10000)
                fail(ErrorCode::InvalidRequest, "Invalid mask erosion radius");
            const int width = static_cast<int>(camera.width), height = static_cast<int>(camera.height);
            CpuImage result{width, height, 1, std::vector<uint8_t>(static_cast<std::size_t>(width) * height, 0)};
            const auto rotation = quaternion_rotation(pose.rotation);
            using Point2 = std::array<double, 2>;
            std::vector<Point2> uv;
            uv.reserve(mesh.vertices.size());
            for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
                if ((i & 4095) == 0)
                    check_stop(ctx);
                auto p = transform(rotation, mesh.vertices[i]);
                for (int k = 0; k < 3; ++k)
                    p[k] += pose.translation[k];
                if (p[2] <= 0)
                    continue;
                const double x = camera.parameters[0] * p[0] / p[2] + camera.parameters[2], y = camera.parameters[1] * p[1] / p[2] + camera.parameters[3];
                if (std::isfinite(x) && std::isfinite(y) && x >= 0 && x < width && y >= 0 && y < height)
                    uv.push_back({x, y});
            }
            if (uv.size() < 3)
                return result;
            std::sort(uv.begin(), uv.end());
            uv.erase(std::unique(uv.begin(), uv.end()), uv.end());
            auto cross = [](const Point2& a, const Point2& b, const Point2& c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); };
            std::vector<Point2> hull(2 * uv.size());
            std::size_t count = 0;
            for (const auto& p : uv) {
                while (count >= 2 && cross(hull[count - 2], hull[count - 1], p) <= 0)
                    --count;
                hull[count++] = p;
            }
            const auto lower = count;
            for (std::size_t i = uv.size() - 1; i > 0; --i) {
                const auto& p = uv[i - 1];
                while (count > lower && cross(hull[count - 2], hull[count - 1], p) <= 0)
                    --count;
                hull[count++] = p;
            }
            if (count > 1)
                --count;
            hull.resize(count);
            cv::Mat mask(height, width, CV_8UC1, result.pixels.data());
            if (hull.size() < 3) {
                for (const auto& p : uv)
                    result.pixels[static_cast<std::size_t>(static_cast<int>(p[1])) * width + static_cast<int>(p[0])] = 255;
            } else {
                std::vector<cv::Point> polygon;
                polygon.reserve(hull.size());
                for (const auto& p : hull)
                    polygon.emplace_back(static_cast<int>(p[0]), static_cast<int>(p[1]));
                // Pillow draws polygon scanline spans, not OpenCV's additional Bresenham
                // boundary pixels. Its edge intersections are float32, with half-pixel
                // ties rounded inward. Preserve that detail before SIFT masking.
                for (int y = 0; y < height; ++y) {
                    if ((y & 127) == 0)
                        check_stop(ctx);
                    float left = std::numeric_limits<float>::infinity();
                    float right = -std::numeric_limits<float>::infinity();
                    for (std::size_t i = 0; i < polygon.size(); ++i) {
                        const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
                        if (y < std::min(a.y, b.y) || y > std::max(a.y, b.y))
                            continue;
                        if (a.y == b.y) {
                            left = std::min(left, static_cast<float>(std::min(a.x, b.x)));
                            right = std::max(right, static_cast<float>(std::max(a.x, b.x)));
                        } else {
                            const float slope = static_cast<float>(b.x - a.x) / static_cast<float>(b.y - a.y);
                            const float intersection = static_cast<float>(a.x) + static_cast<float>(y - a.y) * slope;
                            left = std::min(left, intersection);
                            right = std::max(right, intersection);
                        }
                    }
                    if (!std::isfinite(left))
                        continue;
                    const int begin = std::clamp(static_cast<int>(std::floor(static_cast<double>(left) + .5)), 0, width - 1);
                    const int end = std::clamp(static_cast<int>(std::ceil(static_cast<double>(right) - .5)), 0, width - 1);
                    if (begin <= end)
                        std::fill(result.pixels.begin() + static_cast<std::size_t>(y) * width + begin,
                                  result.pixels.begin() + static_cast<std::size_t>(y) * width + end + 1, 255);
                }
            }
            if (erosion > 0)
                cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, {2 * erosion + 1, 2 * erosion + 1}), {-1, -1}, 1, cv::BORDER_REPLICATE);
            check_stop(ctx);
            return result;
        });
    }

    std::expected<MeshTriangulationResult, Error> triangulate_mesh(const CpuMesh& mesh, const ColmapModel& source,
                                                                   const std::map<uint32_t, Keypoints>& keypoints, const std::filesystem::path& image_root,
                                                                   const MeshTriangulationOptions& options, const ExecutionContext& ctx) {
        return guarded([&] {
            progress(ctx, 0, "Building mesh raycasting tree");
            validate_mesh(mesh, true);
            if (!std::isfinite(options.match_3d_threshold) || options.match_3d_threshold <= 0 || options.min_matches_per_pair == 0)
                fail(ErrorCode::InvalidRequest, "Invalid mesh triangulation matching options");
            if (auto valid = validate_model(source); !valid)
                throw Failure{valid.error()};
            Raycaster raycaster(mesh, ctx);
            MeshTriangulationResult result;
            std::vector<std::unique_ptr<ImageHits>> images;
            std::size_t image_number = 0;
            for (const auto& [id, pose] : source.images) {
                check_stop(ctx);
                const auto& camera = source.cameras.at(pose.camera_id);
                validate_camera(camera, pose);
                const auto found = keypoints.find(id);
                if (found == keypoints.end() || found->second.data.empty()) {
                    ++image_number;
                    continue;
                }
                const auto& keys = found->second;
                if ((keys.columns != 2 && keys.columns != 4 && keys.columns != 6) || keys.data.size() % keys.columns != 0)
                    fail(ErrorCode::InvalidDataset, "Malformed mesh triangulation keypoint matrix");
                if (!std::all_of(keys.data.begin(), keys.data.end(), [](float v) { return std::isfinite(v); }))
                    fail(ErrorCode::InvalidDataset, "Non-finite mesh triangulation keypoint");
                auto data = std::make_unique<ImageHits>();
                data->id = id;
                data->keypoints = &keys;
                const auto count = keys.data.size() / keys.columns;
                if (count > std::numeric_limits<uint32_t>::max())
                    fail(ErrorCode::InvalidDataset, "Too many mesh triangulation keypoints");
                std::vector<std::optional<Vec3>> hits(count);
                const auto rotation = quaternion_rotation(pose.rotation);
                Vec3 origin = transpose_transform(rotation, pose.translation);
                for (auto& v : origin)
                    v = -v;
                tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, 256), [&](const auto& range) {
                    for (std::size_t k = range.begin(); k < range.end(); ++k) {
                        if (ctx.stop_token.stop_requested())
                            return;
                        const double x = keys.data[k * keys.columns], y = keys.data[k * keys.columns + 1];
                        if (x < 0 || x >= camera.width || y < 0 || y >= camera.height)
                            continue;
                        const Vec3 direction = normalize(transpose_transform(rotation, normalize({(x - camera.parameters[2]) / camera.parameters[0], (y - camera.parameters[3]) / camera.parameters[1], 1})));
                        auto hit = raycaster.hit(origin, direction);
                        if (hit && transform(rotation, *hit)[2] + pose.translation[2] > 1e-6)
                            hits[k] = hit;
                    }
                });
                check_stop(ctx);
                std::optional<CpuImage> colors;
                if (options.mean_colors) {
                    auto decoded = read_image(image_root / std::filesystem::u8path(pose.name), false);
                    if (decoded)
                        colors = std::move(*decoded);
                }
                data->colors_valid = colors.has_value();
                for (std::size_t k = 0; k < count; ++k)
                    if (hits[k]) {
                        data->indices.push_back(static_cast<uint32_t>(k));
                        data->cloud.points.push_back(*hits[k]);
                        std::array<uint8_t, 3> color{255, 255, 255};
                        if (colors) {
                            const int x = std::clamp(static_cast<int>(std::nearbyint(keys.data[k * keys.columns])), 0, colors->width - 1);
                            const int y = std::clamp(static_cast<int>(std::nearbyint(keys.data[k * keys.columns + 1])), 0, colors->height - 1);
                            const auto base = (static_cast<std::size_t>(y) * colors->width + x) * colors->channels;
                            for (int c = 0; c < 3; ++c)
                                color[c] = colors->pixels[base + (colors->channels == 1 ? 0 : c)];
                        }
                        data->colors.push_back(color);
                    }
                result.hit_count += data->indices.size();
                if (!data->indices.empty())
                    data->tree = std::make_unique<detail::ScipyNearestTree>(data->cloud.points);
                images.push_back(std::move(data));
                progress(ctx, .35F * static_cast<float>(++image_number) / static_cast<float>(source.images.size()), "Mesh keypoint raycasting");
            }
            const double squared_radius = options.match_3d_threshold * options.match_3d_threshold;
            std::set<uint32_t> connected;
            for (std::size_t i = 0; i < images.size(); ++i) {
                const auto& first = *images[i];
                std::vector<std::optional<MeshPairMatches>> pairs(images.size());
                if (first.tree)
                    tbb::parallel_for(tbb::blocked_range<std::size_t>(i + 1, images.size(), 1), [&](const auto& range) {
                        for (std::size_t j = range.begin(); j < range.end(); ++j) {
                            if (ctx.stop_token.stop_requested())
                                return;
                            const auto& second = *images[j];
                            if (!second.tree)
                                continue;
                            MeshPairMatches pair{first.id, second.id, {}};
                            for (std::size_t k = 0; k < first.cloud.points.size(); ++k) {
                                if ((k & 1023) == 0 && ctx.stop_token.stop_requested())
                                    return;
                                std::size_t nearest = 0;
                                double distance = 0;
                                if (!second.tree->knnSearch(first.cloud.points[k].data(), 1, &nearest, &distance) || distance >= squared_radius)
                                    continue;
                                if (options.mutual_nearest) {
                                    std::size_t reverse = 0;
                                    double reverse_distance = 0;
                                    first.tree->knnSearch(second.cloud.points[nearest].data(), 1, &reverse, &reverse_distance);
                                    if (reverse != k)
                                        continue;
                                }
                                pair.matches.push_back({first.indices[k], second.indices[nearest]});
                            }
                            if (pair.matches.size() >= options.min_matches_per_pair)
                                pairs[j] = std::move(pair);
                        }
                    });
                check_stop(ctx);
                for (auto& pair : pairs)
                    if (pair) {
                        connected.insert(pair->first);
                        connected.insert(pair->second);
                        result.pairs.push_back(std::move(*pair));
                    }
                progress(ctx, .35F + .35F * static_cast<float>(i + 1) / static_cast<float>(std::max<std::size_t>(images.size(), 1)), "Matching mesh keypoints");
            }
            for (const auto& [id, pose] : source.images)
                if (!connected.contains(id))
                    result.dropped_names.push_back(pose.name);
            if (connected.size() < 2)
                fail(ErrorCode::InvalidDataset, "Mesh triangulation connectivity filter left fewer than 2 scan images with verified geometry");
            Cloud all;
            struct SourceHit {
                std::size_t image, index;
            };
            std::vector<SourceHit> origins;
            for (std::size_t i = 0; i < images.size(); ++i)
                if (connected.contains(images[i]->id)) {
                    for (std::size_t k = 0; k < images[i]->cloud.points.size(); ++k) {
                        all.points.push_back(images[i]->cloud.points[k]);
                        origins.push_back({i, k});
                    }
                    auto pose = source.images.at(images[i]->id);
                    pose.observations.clear();
                    const auto& keys = *images[i]->keypoints;
                    for (std::size_t k = 0; k < keys.data.size() / keys.columns; ++k)
                        pose.observations.push_back({keys.data[k * keys.columns], keys.data[k * keys.columns + 1], -1});
                    result.model.cameras.emplace(pose.camera_id, source.cameras.at(pose.camera_id));
                    result.model.images.emplace(pose.id, std::move(pose));
                }
            if (all.points.empty())
                fail(ErrorCode::InvalidDataset, "Mesh triangulation produced no ray hits");
            Index index(3, all, nanoflann::KDTreeSingleIndexAdaptorParams(16));
            UnionFind groups(all.points.size());
            std::vector<nanoflann::ResultItem<std::size_t, double>> neighbors;
            for (std::size_t i = 0; i < all.points.size(); ++i) {
                if ((i & 4095) == 0)
                    progress(ctx, .7F + .2F * static_cast<float>(i) / static_cast<float>(all.points.size()), "Clustering mesh ray hits");
                // nanoflann uses strict <; frozen scipy radius graph includes equality.
                index.radiusSearch(all.points[i].data(), std::nextafter(squared_radius, std::numeric_limits<double>::infinity()), neighbors, {0, false});
                for (const auto& neighbor : neighbors)
                    if (neighbor.first > i && neighbor.second <= squared_radius)
                        groups.join(i, neighbor.first);
            }
            struct Cluster {
                Vec3 sum{};
                std::size_t count = 0;
                std::map<uint32_t, SourceHit> tracks;
                uint64_t id = 0;
            };
            std::map<std::size_t, Cluster> clusters;
            uint64_t next_id = 1;
            for (std::size_t i = 0; i < all.points.size(); ++i) {
                if ((i & 4095) == 0)
                    check_stop(ctx);
                auto [it, inserted] = clusters.try_emplace(groups.find(i));
                auto& cluster = it->second;
                if (inserted)
                    cluster.id = next_id++;
                for (int k = 0; k < 3; ++k)
                    cluster.sum[k] += all.points[i][k];
                ++cluster.count;
                const auto origin = origins[i];
                cluster.tracks.try_emplace(images[origin.image]->id, origin);
            }
            result.model.points.emplace();
            for (const auto& [root, cluster] : clusters) {
                check_stop(ctx);
                if (cluster.tracks.size() < 2)
                    continue;
                ColmapPoint output;
                output.id = cluster.id;
                for (int k = 0; k < 3; ++k)
                    output.position[k] = cluster.sum[k] / static_cast<double>(cluster.count);
                std::array<std::size_t, 3> color_sum{};
                std::size_t color_count = 0;
                for (const auto& [image_id, origin] : cluster.tracks) {
                    const auto& data = *images[origin.image];
                    const auto keypoint = data.indices[origin.index];
                    output.track.push_back({image_id, keypoint});
                    result.model.images.at(image_id).observations[keypoint].point_id = static_cast<int64_t>(output.id);
                    if (data.colors_valid) {
                        ++color_count;
                        for (int k = 0; k < 3; ++k)
                            color_sum[k] += data.colors[origin.index][k];
                    }
                }
                for (int k = 0; k < 3; ++k)
                    output.color[k] = color_count ? static_cast<uint8_t>(color_sum[k] / color_count) : 255;
                result.model.points->emplace(output.id, std::move(output));
            }
            if (result.model.points->empty())
                fail(ErrorCode::InvalidDataset, "Mesh triangulation produced no multiview points");
            if (auto valid = validate_model(result.model); !valid)
                throw Failure{valid.error()};
            progress(ctx, 1, "Mesh triangulation complete");
            return result;
        });
    }
} // namespace lfs::preprocess
