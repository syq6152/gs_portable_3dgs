/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/mesh_hole_fill.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polygon_mesh_processing/border.h>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::geometry {
    namespace {

        using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
        using Point = Kernel::Point_3;
        using Mesh = CGAL::Surface_mesh<Point>;
        using Vertex = Mesh::Vertex_index;
        using Halfedge = Mesh::Halfedge_index;
        using Face = Mesh::Face_index;
        namespace PMP = CGAL::Polygon_mesh_processing;

        constexpr double NORMAL_EPSILON = 1e-12;

        struct CellKey {
            int64_t x;
            int64_t y;
            int64_t z;

            bool operator==(const CellKey&) const = default;
        };

        struct CellKeyHash {
            std::size_t operator()(const CellKey& key) const noexcept {
                auto mix = [](uint64_t value) {
                    value ^= value >> 30;
                    value *= 0xbf58476d1ce4e5b9ULL;
                    value ^= value >> 27;
                    value *= 0x94d049bb133111ebULL;
                    return value ^ (value >> 31);
                };
                return static_cast<std::size_t>(
                    mix(static_cast<uint64_t>(key.x)) ^
                    (mix(static_cast<uint64_t>(key.y)) << 1) ^
                    (mix(static_cast<uint64_t>(key.z)) << 2));
            }
        };

        struct WeldResult {
            std::vector<glm::dvec3> positions;
            std::vector<uint32_t> unique_to_input;
            std::vector<uint32_t> input_to_unique;
            std::vector<uint32_t> input_to_representative;
        };

        struct BoundaryMetrics {
            std::vector<Halfedge> halfedges;
            std::vector<Vertex> vertices;
            glm::dvec3 average_neighbor_normal{0.0};
            std::size_t neighbor_face_count = 0;
            std::vector<uint32_t> neighbor_face_indices;
            double perimeter = 0.0;
            double projected_area = 0.0;
            double bbox_diagonal = 0.0;
            double neighbor_area = 0.0;
            double neighbor_median_edge_length = 0.0;
            double neighbor_median_face_area = 0.0;
            bool topology_valid = false;
            bool self_intersects = false;
            bool normals_consistent = false;
            std::string failure_reason;
        };

        [[nodiscard]] glm::dvec3 to_dvec3(const glm::vec3& value) {
            return {static_cast<double>(value.x),
                    static_cast<double>(value.y),
                    static_cast<double>(value.z)};
        }

        [[nodiscard]] glm::dvec3 to_dvec3(const Point& value) {
            return {CGAL::to_double(value.x()),
                    CGAL::to_double(value.y()),
                    CGAL::to_double(value.z())};
        }

        [[nodiscard]] Point to_point(const glm::dvec3& value) {
            return {value.x, value.y, value.z};
        }

        [[nodiscard]] bool is_finite(const glm::dvec3& value) {
            return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
        }

        [[nodiscard]] double triangle_double_area(const glm::dvec3& a,
                                                  const glm::dvec3& b,
                                                  const glm::dvec3& c) {
            return glm::length(glm::cross(b - a, c - a));
        }

        [[nodiscard]] double median(std::vector<double> values) {
            if (values.empty()) {
                return 0.0;
            }
            const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
            std::nth_element(values.begin(), middle, values.end());
            const double upper = *middle;
            if ((values.size() & 1U) != 0U) {
                return upper;
            }
            const auto lower = std::max_element(values.begin(), middle);
            return 0.5 * (*lower + upper);
        }

        [[nodiscard]] std::array<Vertex, 3> face_vertices(const Mesh& mesh, const Face face) {
            const Halfedge h0 = mesh.halfedge(face);
            const Halfedge h1 = mesh.next(h0);
            const Halfedge h2 = mesh.next(h1);
            return {mesh.target(h0), mesh.target(h1), mesh.target(h2)};
        }

        [[nodiscard]] glm::dvec3 face_normal(const Mesh& mesh, const Face face) {
            const auto vertices = face_vertices(mesh, face);
            const auto a = to_dvec3(mesh.point(vertices[0]));
            const auto b = to_dvec3(mesh.point(vertices[1]));
            const auto c = to_dvec3(mesh.point(vertices[2]));
            return glm::cross(b - a, c - a);
        }

        [[nodiscard]] bool valid_options(const MeshHoleFillOptions& options,
                                         MeshHoleFillError& error) {
            if (options.max_boundary_edges < 3) {
                error = {MeshHoleFillErrorCode::InvalidOptions,
                         "max_boundary_edges must be at least 3"};
                return false;
            }
            const std::array<std::pair<double, const char*>, 5> positive_values = {{
                {options.max_boundary_perimeter_bbox_ratio, "max_boundary_perimeter_bbox_ratio"},
                {options.max_boundary_area_bbox_ratio, "max_boundary_area_bbox_ratio"},
                {options.max_boundary_bbox_diagonal_ratio, "max_boundary_bbox_diagonal_ratio"},
                {options.weld_epsilon_bbox_ratio, "weld_epsilon_bbox_ratio"},
                {options.density_control_factor, "density_control_factor"},
            }};
            for (const auto& [value, name] : positive_values) {
                if (!std::isfinite(value) || value <= 0.0) {
                    error = {MeshHoleFillErrorCode::InvalidOptions,
                             std::format("{} must be finite and positive", name)};
                    return false;
                }
            }
            for (std::size_t hole_id = 0;
                 hole_id < options.density_control_factor_overrides.size();
                 ++hole_id) {
                const auto& override_factor =
                    options.density_control_factor_overrides[hole_id];
                if (override_factor.has_value() &&
                    (!std::isfinite(*override_factor) || *override_factor <= 0.0)) {
                    error = {MeshHoleFillErrorCode::InvalidOptions,
                             std::format(
                                 "density_control_factor_overrides[{}] must be finite and positive",
                                 hole_id)};
                    return false;
                }
            }
            if (options.fill_only_density_control_factor_overrides &&
                std::ranges::none_of(
                    options.density_control_factor_overrides,
                    [](const auto& value) { return value.has_value(); })) {
                error = {MeshHoleFillErrorCode::InvalidOptions,
                         "fill_only_density_control_factor_overrides requires an override"};
                return false;
            }
            return true;
        }

        [[nodiscard]] std::expected<std::pair<glm::dvec3, glm::dvec3>, MeshHoleFillError>
        input_bounds(const MeshHoleFillInput& input) {
            if (input.vertices.empty()) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::InvalidInput, "mesh must contain at least one vertex"});
            }
            glm::dvec3 min_point(std::numeric_limits<double>::infinity());
            glm::dvec3 max_point(-std::numeric_limits<double>::infinity());
            for (const auto& vertex : input.vertices) {
                const auto point = to_dvec3(vertex);
                if (!is_finite(point)) {
                    return std::unexpected(MeshHoleFillError{
                        MeshHoleFillErrorCode::InvalidInput, "mesh contains a non-finite vertex"});
                }
                min_point = glm::min(min_point, point);
                max_point = glm::max(max_point, point);
            }
            return std::pair{min_point, max_point};
        }

        [[nodiscard]] CellKey cell_for(const glm::dvec3& point, const double cell_size) {
            return {static_cast<int64_t>(std::floor(point.x / cell_size)),
                    static_cast<int64_t>(std::floor(point.y / cell_size)),
                    static_cast<int64_t>(std::floor(point.z / cell_size))};
        }

        [[nodiscard]] WeldResult weld_vertices(const std::span<const glm::vec3> input,
                                               const double epsilon) {
            WeldResult result;
            result.input_to_unique.resize(input.size());
            result.input_to_representative.resize(input.size());

            std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash> cells;
            const double epsilon_squared = epsilon * epsilon;
            for (std::size_t input_index = 0; input_index < input.size(); ++input_index) {
                const glm::dvec3 point = to_dvec3(input[input_index]);
                const CellKey cell = cell_for(point, epsilon);
                uint32_t match = std::numeric_limits<uint32_t>::max();
                for (int dz = -1; dz <= 1 && match == std::numeric_limits<uint32_t>::max(); ++dz) {
                    for (int dy = -1; dy <= 1 && match == std::numeric_limits<uint32_t>::max(); ++dy) {
                        for (int dx = -1; dx <= 1 && match == std::numeric_limits<uint32_t>::max(); ++dx) {
                            const CellKey neighbor{cell.x + dx, cell.y + dy, cell.z + dz};
                            const auto found = cells.find(neighbor);
                            if (found == cells.end()) {
                                continue;
                            }
                            for (const uint32_t candidate : found->second) {
                                const glm::dvec3 delta = point - result.positions[candidate];
                                if (glm::dot(delta, delta) <= epsilon_squared) {
                                    match = candidate;
                                    break;
                                }
                            }
                        }
                    }
                }

                if (match == std::numeric_limits<uint32_t>::max()) {
                    match = static_cast<uint32_t>(result.positions.size());
                    result.positions.push_back(point);
                    result.unique_to_input.push_back(static_cast<uint32_t>(input_index));
                    cells[cell].push_back(match);
                }
                result.input_to_unique[input_index] = match;
                result.input_to_representative[input_index] = result.unique_to_input[match];
            }
            return result;
        }

        [[nodiscard]] int dominant_projection_axis(const glm::dvec3& normal) {
            const glm::dvec3 magnitude = glm::abs(normal);
            if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
                return 0;
            }
            return magnitude.y >= magnitude.z ? 1 : 2;
        }

        [[nodiscard]] glm::dvec2 project_point(const glm::dvec3& point, const int dropped_axis) {
            if (dropped_axis == 0) {
                return {point.y, point.z};
            }
            if (dropped_axis == 1) {
                return {point.x, point.z};
            }
            return {point.x, point.y};
        }

        [[nodiscard]] double orient_2d(const glm::dvec2& a,
                                       const glm::dvec2& b,
                                       const glm::dvec2& c) {
            return (b.x - a.x) * (c.y - a.y) -
                   (b.y - a.y) * (c.x - a.x);
        }

        [[nodiscard]] bool on_segment(const glm::dvec2& a,
                                      const glm::dvec2& b,
                                      const glm::dvec2& p,
                                      const double epsilon) {
            return p.x >= std::min(a.x, b.x) - epsilon &&
                   p.x <= std::max(a.x, b.x) + epsilon &&
                   p.y >= std::min(a.y, b.y) - epsilon &&
                   p.y <= std::max(a.y, b.y) + epsilon;
        }

        [[nodiscard]] int orientation_sign(const double value, const double epsilon) {
            if (value > epsilon) {
                return 1;
            }
            return value < -epsilon ? -1 : 0;
        }

        [[nodiscard]] bool segments_intersect(const glm::dvec2& a,
                                              const glm::dvec2& b,
                                              const glm::dvec2& c,
                                              const glm::dvec2& d,
                                              const double epsilon) {
            const double ab_c = orient_2d(a, b, c);
            const double ab_d = orient_2d(a, b, d);
            const double cd_a = orient_2d(c, d, a);
            const double cd_b = orient_2d(c, d, b);
            const int s1 = orientation_sign(ab_c, epsilon);
            const int s2 = orientation_sign(ab_d, epsilon);
            const int s3 = orientation_sign(cd_a, epsilon);
            const int s4 = orientation_sign(cd_b, epsilon);
            if (s1 != s2 && s3 != s4) {
                return true;
            }
            return (s1 == 0 && on_segment(a, b, c, epsilon)) ||
                   (s2 == 0 && on_segment(a, b, d, epsilon)) ||
                   (s3 == 0 && on_segment(c, d, a, epsilon)) ||
                   (s4 == 0 && on_segment(c, d, b, epsilon));
        }

        [[nodiscard]] bool polygon_self_intersects(const std::vector<glm::dvec3>& points,
                                                   const glm::dvec3& area_vector,
                                                   const double scale) {
            const int axis = dominant_projection_axis(area_vector);
            std::vector<glm::dvec2> projected;
            projected.reserve(points.size());
            for (const auto& point : points) {
                projected.push_back(project_point(point, axis));
            }
            const double epsilon = std::max(scale * scale * 1e-12, 1e-18);
            for (std::size_t i = 0; i < projected.size(); ++i) {
                const std::size_t i_next = (i + 1) % projected.size();
                for (std::size_t j = i + 1; j < projected.size(); ++j) {
                    const std::size_t j_next = (j + 1) % projected.size();
                    if (i == j || i_next == j || j_next == i) {
                        continue;
                    }
                    if (segments_intersect(projected[i], projected[i_next],
                                           projected[j], projected[j_next], epsilon)) {
                        return true;
                    }
                }
            }
            return false;
        }

        [[nodiscard]] BoundaryMetrics collect_boundary_metrics(const Mesh& mesh,
                                                               const Halfedge representative,
                                                               const double mesh_diagonal) {
            BoundaryMetrics metrics;
            std::vector<glm::dvec3> points;
            std::unordered_set<uint32_t> unique_vertices;
            std::unordered_set<uint32_t> unique_neighbor_faces;
            std::vector<glm::dvec3> neighbor_normals;
            std::vector<double> neighbor_edge_lengths;
            std::vector<double> neighbor_face_areas;

            Halfedge current = representative;
            const std::size_t halfedge_guard = mesh.number_of_halfedges() + 1;
            do {
                if (metrics.halfedges.size() >= halfedge_guard || !mesh.is_border(current)) {
                    metrics.failure_reason = "invalid or non-closing boundary cycle";
                    return metrics;
                }
                const Vertex vertex = mesh.target(current);
                if (!unique_vertices.insert(vertex.idx()).second) {
                    metrics.failure_reason = "boundary cycle repeats a vertex";
                    return metrics;
                }
                metrics.halfedges.push_back(current);
                metrics.vertices.push_back(vertex);
                points.push_back(to_dvec3(mesh.point(vertex)));

                const Face neighbor = mesh.face(mesh.opposite(current));
                if (neighbor == Mesh::null_face()) {
                    metrics.failure_reason = "boundary edge has no adjacent source face";
                    return metrics;
                }
                if (unique_neighbor_faces.insert(neighbor.idx()).second) {
                    metrics.neighbor_face_indices.push_back(neighbor.idx());
                    const glm::dvec3 normal = face_normal(mesh, neighbor);
                    const double double_area = glm::length(normal);
                    if (!std::isfinite(double_area) || double_area <= NORMAL_EPSILON) {
                        metrics.failure_reason = "boundary has a degenerate adjacent face";
                        return metrics;
                    }
                    neighbor_normals.push_back(normal / double_area);
                    neighbor_face_areas.push_back(0.5 * double_area);
                    metrics.neighbor_area += 0.5 * double_area;

                    const auto fv = face_vertices(mesh, neighbor);
                    for (std::size_t edge = 0; edge < 3; ++edge) {
                        const auto a = to_dvec3(mesh.point(fv[edge]));
                        const auto b = to_dvec3(mesh.point(fv[(edge + 1) % 3]));
                        neighbor_edge_lengths.push_back(glm::length(b - a));
                    }
                }
                current = mesh.next(current);
            } while (current != representative);

            if (points.size() < 3) {
                metrics.failure_reason = "boundary cycle has fewer than three edges";
                return metrics;
            }

            metrics.neighbor_face_count = unique_neighbor_faces.size();
            std::ranges::sort(metrics.neighbor_face_indices);
            metrics.neighbor_median_edge_length = median(neighbor_edge_lengths);
            metrics.neighbor_median_face_area = median(neighbor_face_areas);

            glm::dvec3 min_point(std::numeric_limits<double>::infinity());
            glm::dvec3 max_point(-std::numeric_limits<double>::infinity());
            glm::dvec3 area_vector(0.0);
            for (std::size_t i = 0; i < points.size(); ++i) {
                const auto& a = points[i];
                const auto& b = points[(i + 1) % points.size()];
                const double edge_length = glm::length(b - a);
                if (!std::isfinite(edge_length) || edge_length <= NORMAL_EPSILON) {
                    metrics.failure_reason = "boundary contains a zero-length edge";
                    return metrics;
                }
                metrics.perimeter += edge_length;
                min_point = glm::min(min_point, a);
                max_point = glm::max(max_point, a);
                area_vector += glm::cross(a, b);
            }
            metrics.projected_area = 0.5 * glm::length(area_vector);
            metrics.bbox_diagonal = glm::length(max_point - min_point);
            if (!std::isfinite(metrics.projected_area) ||
                metrics.projected_area <= mesh_diagonal * mesh_diagonal * 1e-14) {
                metrics.failure_reason = "boundary has a degenerate projected area";
                return metrics;
            }

            metrics.self_intersects = polygon_self_intersects(points, area_vector, metrics.bbox_diagonal);
            if (metrics.self_intersects) {
                metrics.failure_reason = "boundary polygon self-intersects";
                return metrics;
            }

            glm::dvec3 normal_sum(0.0);
            for (const auto& normal : neighbor_normals) {
                normal_sum += normal;
            }
            const double normal_sum_length = glm::length(normal_sum);
            if (!std::isfinite(normal_sum_length) || normal_sum_length <= NORMAL_EPSILON) {
                metrics.failure_reason = "boundary adjacent normals have no stable direction";
                return metrics;
            }
            metrics.average_neighbor_normal = normal_sum / normal_sum_length;
            metrics.normals_consistent = std::ranges::all_of(
                neighbor_normals,
                [&](const glm::dvec3& normal) {
                    return glm::dot(normal, metrics.average_neighbor_normal) > 0.0;
                });
            if (!metrics.normals_consistent) {
                metrics.failure_reason = "boundary adjacent normals are inconsistent";
                return metrics;
            }

            metrics.topology_valid = true;
            return metrics;
        }

        void copy_metrics(const BoundaryMetrics& source,
                          const double mesh_diagonal,
                          HoleFillDiagnostic& target) {
            target.boundary_edge_count = source.halfedges.size();
            target.boundary_perimeter = source.perimeter;
            target.boundary_projected_area = source.projected_area;
            target.boundary_bbox_diagonal = source.bbox_diagonal;
            target.boundary_perimeter_bbox_ratio = source.perimeter / mesh_diagonal;
            target.boundary_area_bbox_ratio = source.projected_area / (mesh_diagonal * mesh_diagonal);
            target.boundary_bbox_diagonal_ratio = source.bbox_diagonal / mesh_diagonal;
            target.neighbor_face_count = source.neighbor_face_count;
            target.neighbor_face_indices = source.neighbor_face_indices;
            target.neighbor_area = source.neighbor_area;
            target.neighbor_median_edge_length = source.neighbor_median_edge_length;
            target.neighbor_median_face_area = source.neighbor_median_face_area;
            target.topology_valid = source.topology_valid;
            target.boundary_self_intersects = source.self_intersects;
            target.normals_consistent = source.normals_consistent;
        }

        [[nodiscard]] std::string threshold_failure(const HoleFillDiagnostic& diagnostic,
                                                    const MeshHoleFillOptions& options) {
            std::vector<std::string> exceeded;
            if (diagnostic.boundary_edge_count > options.max_boundary_edges) {
                exceeded.emplace_back("max_boundary_edges");
            }
            if (diagnostic.boundary_perimeter_bbox_ratio >
                options.max_boundary_perimeter_bbox_ratio) {
                exceeded.emplace_back("max_boundary_perimeter_bbox_ratio");
            }
            if (diagnostic.boundary_area_bbox_ratio > options.max_boundary_area_bbox_ratio) {
                exceeded.emplace_back("max_boundary_area_bbox_ratio");
            }
            if (diagnostic.boundary_bbox_diagonal_ratio >
                options.max_boundary_bbox_diagonal_ratio) {
                exceeded.emplace_back("max_boundary_bbox_diagonal_ratio");
            }
            if (exceeded.empty()) {
                return {};
            }
            std::string reason = "boundary exceeds ";
            for (std::size_t i = 0; i < exceeded.size(); ++i) {
                if (i != 0) {
                    reason += ", ";
                }
                reason += exceeded[i];
            }
            return reason;
        }

        [[nodiscard]] bool has_non_manifold_vertex(const Mesh& mesh) {
            std::vector<Halfedge> non_manifold;
            PMP::non_manifold_vertices(mesh, std::back_inserter(non_manifold));
            return !non_manifold.empty();
        }

        [[nodiscard]] std::vector<Face> current_patch_faces(const Mesh& mesh,
                                                            const std::vector<Face>& output_faces) {
            std::vector<Face> result;
            std::unordered_set<uint32_t> seen;
            result.reserve(output_faces.size());
            for (const Face face : output_faces) {
                if (face == Mesh::null_face() || mesh.is_removed(face) ||
                    !seen.insert(face.idx()).second) {
                    continue;
                }
                result.push_back(face);
            }
            return result;
        }

        [[nodiscard]] bool validate_patch(const Mesh& mesh,
                                          const std::vector<Face>& patch_faces,
                                          const glm::dvec3& expected_normal,
                                          HoleFillDiagnostic& diagnostic) {
            if (patch_faces.empty()) {
                diagnostic.reason = "CGAL returned an empty patch";
                return false;
            }
            if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh)) {
                diagnostic.reason = "CGAL patch produced invalid triangle topology";
                return false;
            }

            diagnostic.patch_manifold = !has_non_manifold_vertex(mesh);
            if (!diagnostic.patch_manifold) {
                diagnostic.reason = "CGAL patch produced a non-manifold vertex";
                return false;
            }

            std::vector<double> patch_edge_lengths;
            for (const Face face : patch_faces) {
                const auto fv = face_vertices(mesh, face);
                const auto a = to_dvec3(mesh.point(fv[0]));
                const auto b = to_dvec3(mesh.point(fv[1]));
                const auto c = to_dvec3(mesh.point(fv[2]));
                const glm::dvec3 normal = glm::cross(b - a, c - a);
                const double double_area = glm::length(normal);
                if (!std::isfinite(double_area) || double_area <= NORMAL_EPSILON) {
                    diagnostic.reason = "CGAL patch contains a degenerate face";
                    return false;
                }
                const glm::dvec3 unit_normal = normal / double_area;
                if (glm::dot(unit_normal, expected_normal) <= 0.0) {
                    diagnostic.reason = "CGAL patch contains an inconsistent face normal";
                    return false;
                }
                diagnostic.patch_area += 0.5 * double_area;
                patch_edge_lengths.push_back(glm::length(b - a));
                patch_edge_lengths.push_back(glm::length(c - b));
                patch_edge_lengths.push_back(glm::length(a - c));
            }
            diagnostic.patch_normals_consistent = true;
            diagnostic.patch_median_edge_length = median(std::move(patch_edge_lengths));

            diagnostic.patch_self_intersects = PMP::does_self_intersect(mesh);
            if (diagnostic.patch_self_intersects) {
                diagnostic.reason = "CGAL patch self-intersects existing or patch geometry";
                return false;
            }
            return true;
        }

    } // namespace

    std::expected<MeshHoleFillResult, MeshHoleFillError>
    fill_mesh_holes(const MeshHoleFillInput& input,
                    const MeshHoleFillOptions& options) {
        MeshHoleFillError option_error;
        if (!valid_options(options, option_error)) {
            return std::unexpected(std::move(option_error));
        }
        if (input.triangle_indices.empty() || input.triangle_indices.size() % 3 != 0) {
            return std::unexpected(MeshHoleFillError{
                MeshHoleFillErrorCode::InvalidInput,
                "triangle index array must be non-empty and divisible by three"});
        }
        if (input.vertices.size() > std::numeric_limits<uint32_t>::max()) {
            return std::unexpected(MeshHoleFillError{
                MeshHoleFillErrorCode::InvalidInput,
                "mesh has too many vertices for 32-bit patch indices"});
        }

        const auto bounds = input_bounds(input);
        if (!bounds) {
            return std::unexpected(bounds.error());
        }
        const double mesh_diagonal = glm::length(bounds->second - bounds->first);
        if (!std::isfinite(mesh_diagonal) || mesh_diagonal <= NORMAL_EPSILON) {
            return std::unexpected(MeshHoleFillError{
                MeshHoleFillErrorCode::InvalidInput,
                "mesh bounding-box diagonal must be finite and positive"});
        }
        const double weld_epsilon = mesh_diagonal * options.weld_epsilon_bbox_ratio;
        const double minimum_double_area = mesh_diagonal * mesh_diagonal * 1e-14;

        for (std::size_t face_index = 0; face_index < input.triangle_indices.size() / 3;
             ++face_index) {
            const uint32_t ia = input.triangle_indices[face_index * 3];
            const uint32_t ib = input.triangle_indices[face_index * 3 + 1];
            const uint32_t ic = input.triangle_indices[face_index * 3 + 2];
            if (ia >= input.vertices.size() || ib >= input.vertices.size() ||
                ic >= input.vertices.size()) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::InvalidInput,
                    "triangle index is outside the input vertex array",
                    face_index,
                    true});
            }
            if (ia == ib || ib == ic || ic == ia ||
                triangle_double_area(to_dvec3(input.vertices[ia]),
                                     to_dvec3(input.vertices[ib]),
                                     to_dvec3(input.vertices[ic])) <= minimum_double_area) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::InvalidInput,
                    "mesh contains a degenerate triangle",
                    face_index,
                    true});
            }
        }

        WeldResult weld = weld_vertices(input.vertices, weld_epsilon);
        Mesh mesh;
        std::vector<Vertex> unique_to_cgal;
        unique_to_cgal.reserve(weld.positions.size());
        std::unordered_map<uint32_t, uint32_t> cgal_vertex_to_output;
        for (std::size_t unique_index = 0; unique_index < weld.positions.size(); ++unique_index) {
            const Vertex vertex = mesh.add_vertex(to_point(weld.positions[unique_index]));
            unique_to_cgal.push_back(vertex);
            cgal_vertex_to_output.emplace(vertex.idx(), weld.unique_to_input[unique_index]);
        }

        for (std::size_t face_index = 0; face_index < input.triangle_indices.size() / 3;
             ++face_index) {
            const uint32_t input_a = input.triangle_indices[face_index * 3];
            const uint32_t input_b = input.triangle_indices[face_index * 3 + 1];
            const uint32_t input_c = input.triangle_indices[face_index * 3 + 2];
            const uint32_t a = weld.input_to_unique[input_a];
            const uint32_t b = weld.input_to_unique[input_b];
            const uint32_t c = weld.input_to_unique[input_c];
            if (a == b || b == c || c == a ||
                triangle_double_area(weld.positions[a], weld.positions[b], weld.positions[c]) <=
                    minimum_double_area) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::InvalidInput,
                    "seam welding collapses a triangle",
                    face_index,
                    true});
            }
            if (mesh.add_face(unique_to_cgal[a], unique_to_cgal[b], unique_to_cgal[c]) ==
                Mesh::null_face()) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::InvalidTopology,
                    "triangle soup is not a consistently oriented two-manifold mesh after welding",
                    face_index,
                    true});
            }
        }

        if (!CGAL::is_valid_polygon_mesh(mesh) || !CGAL::is_triangle_mesh(mesh) ||
            has_non_manifold_vertex(mesh)) {
            return std::unexpected(MeshHoleFillError{
                MeshHoleFillErrorCode::InvalidTopology,
                "input is not a valid manifold triangle mesh after welding"});
        }
        try {
            if (PMP::does_self_intersect(mesh)) {
                return std::unexpected(MeshHoleFillError{
                    MeshHoleFillErrorCode::SelfIntersection,
                    "input mesh self-intersects after welding"});
            }
        } catch (const std::exception& error) {
            return std::unexpected(MeshHoleFillError{
                MeshHoleFillErrorCode::InvalidTopology,
                std::format("input topology validation failed: {}", error.what())});
        }

        MeshHoleFillResult result;
        result.original_vertex_count = input.vertices.size();
        result.original_face_count = input.triangle_indices.size() / 3;
        result.welded_vertex_count = weld.positions.size();
        result.mesh_bbox_diagonal = mesh_diagonal;
        result.weld_epsilon = weld_epsilon;
        result.input_vertex_to_welded_representative = std::move(weld.input_to_representative);

        // Snapshot boundary representatives before any patch changes topology.
        std::vector<Halfedge> boundaries;
        PMP::extract_boundary_cycles(mesh, std::back_inserter(boundaries));
        std::ranges::sort(
            boundaries,
            [](const Halfedge left, const Halfedge right) { return left.idx() < right.idx(); });

        for (std::size_t hole_id = 0; hole_id < boundaries.size(); ++hole_id) {
            HoleFillDiagnostic diagnostic;
            diagnostic.hole_id = hole_id;
            const bool has_density_override =
                hole_id < options.density_control_factor_overrides.size() &&
                options.density_control_factor_overrides[hole_id].has_value();
            diagnostic.density_control_factor = has_density_override
                                                    ? *options.density_control_factor_overrides[hole_id]
                                                    : options.density_control_factor;
            if (options.fill_only_density_control_factor_overrides &&
                !has_density_override) {
                diagnostic.status = HoleFillStatus::Skipped;
                diagnostic.reason = "not selected for the internal density retry";
                ++result.skipped_hole_count;
                result.diagnostics.push_back(std::move(diagnostic));
                continue;
            }

            const BoundaryMetrics metrics =
                collect_boundary_metrics(mesh, boundaries[hole_id], mesh_diagonal);
            copy_metrics(metrics, mesh_diagonal, diagnostic);
            if (!metrics.topology_valid) {
                diagnostic.status = HoleFillStatus::Failed;
                diagnostic.reason = metrics.failure_reason;
                ++result.failed_hole_count;
                result.diagnostics.push_back(std::move(diagnostic));
                continue;
            }

            if (const std::string reason = threshold_failure(diagnostic, options);
                !reason.empty()) {
                diagnostic.status = HoleFillStatus::Skipped;
                diagnostic.reason = reason;
                ++result.skipped_hole_count;
                result.diagnostics.push_back(std::move(diagnostic));
                continue;
            }

            Mesh trial = mesh;
            std::vector<Face> patch_face_output;
            std::vector<Vertex> patch_vertex_output;
            try {
                const double planar_threshold =
                    std::max(weld_epsilon, metrics.bbox_diagonal * 1e-3);
                PMP::triangulate_and_refine_hole(
                    trial,
                    boundaries[hole_id],
                    CGAL::parameters::face_output_iterator(std::back_inserter(patch_face_output))
                        .vertex_output_iterator(std::back_inserter(patch_vertex_output))
                        .use_delaunay_triangulation(true)
                        .use_2d_constrained_delaunay_triangulation(true)
                        .threshold_distance(planar_threshold)
                        .do_not_use_cubic_algorithm(false)
                        .density_control_factor(diagnostic.density_control_factor));

                const auto patch_faces = current_patch_faces(trial, patch_face_output);
                if (!validate_patch(trial, patch_faces, metrics.average_neighbor_normal,
                                    diagnostic)) {
                    diagnostic.status = HoleFillStatus::Failed;
                    ++result.failed_hole_count;
                    result.diagnostics.push_back(std::move(diagnostic));
                    continue;
                }

                const std::size_t combined_vertex_base =
                    input.vertices.size() + result.appended_vertices.size();
                std::vector<glm::vec3> patch_vertices;
                std::vector<uint32_t> patch_indices;
                patch_indices.reserve(patch_faces.size() * 3);
                auto trial_vertex_to_output = cgal_vertex_to_output;
                for (const Face face : patch_faces) {
                    for (const Vertex vertex : face_vertices(trial, face)) {
                        auto mapping = trial_vertex_to_output.find(vertex.idx());
                        if (mapping == trial_vertex_to_output.end()) {
                            const std::size_t combined_index =
                                combined_vertex_base + patch_vertices.size();
                            if (combined_index > std::numeric_limits<uint32_t>::max()) {
                                throw std::runtime_error(
                                    "patch exceeds the 32-bit combined vertex index space");
                            }
                            const auto point = to_dvec3(trial.point(vertex));
                            patch_vertices.emplace_back(
                                static_cast<float>(point.x),
                                static_cast<float>(point.y),
                                static_cast<float>(point.z));
                            mapping = trial_vertex_to_output
                                          .emplace(vertex.idx(),
                                                   static_cast<uint32_t>(combined_index))
                                          .first;
                        }
                        patch_indices.push_back(mapping->second);
                    }
                }

                // Allocate all result storage before changing its logical contents. A
                // failed allocation therefore leaves this hole's patch output empty.
                result.appended_vertices.reserve(
                    result.appended_vertices.size() + patch_vertices.size());
                result.appended_triangle_indices.reserve(
                    result.appended_triangle_indices.size() + patch_indices.size());
                result.appended_face_hole_ids.reserve(
                    result.appended_face_hole_ids.size() + patch_faces.size());

                result.appended_vertices.insert(
                    result.appended_vertices.end(), patch_vertices.begin(), patch_vertices.end());
                result.appended_triangle_indices.insert(
                    result.appended_triangle_indices.end(), patch_indices.begin(), patch_indices.end());
                result.appended_face_hole_ids.insert(
                    result.appended_face_hole_ids.end(), patch_faces.size(), hole_id);
                result.appended_face_count += patch_faces.size();
                diagnostic.appended_vertex_count = patch_vertices.size();
                diagnostic.appended_face_count = patch_faces.size();
                diagnostic.status = HoleFillStatus::Filled;
                diagnostic.reason = "filled";
                ++result.successful_hole_count;

                cgal_vertex_to_output = std::move(trial_vertex_to_output);
                mesh = std::move(trial);
            } catch (const std::exception& error) {
                diagnostic.status = HoleFillStatus::Failed;
                diagnostic.reason = std::format("CGAL hole filling failed: {}", error.what());
                ++result.failed_hole_count;
            } catch (...) {
                diagnostic.status = HoleFillStatus::Failed;
                diagnostic.reason = "CGAL hole filling failed with an unknown exception";
                ++result.failed_hole_count;
            }
            result.diagnostics.push_back(std::move(diagnostic));
        }

        return result;
    }

} // namespace lfs::geometry
