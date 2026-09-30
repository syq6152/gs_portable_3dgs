/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/colmap_model.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace lfs::preprocess {
    namespace {
        constexpr const char* names[] = {"SIMPLE_PINHOLE",
                                         "PINHOLE",
                                         "SIMPLE_RADIAL",
                                         "RADIAL",
                                         "OPENCV",
                                         "OPENCV_FISHEYE",
                                         "FULL_OPENCV",
                                         "FOV",
                                         "SIMPLE_RADIAL_FISHEYE",
                                         "RADIAL_FISHEYE",
                                         "THIN_PRISM_FISHEYE"};
        constexpr std::size_t counts[] = {3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12};
        void check(bool ok, const char* message) {
            if (!ok)
                throw std::runtime_error(message);
        }
        template <class T>
        T get(std::istream& stream) {
            std::array<unsigned char, sizeof(T)> bytes{};
            check(static_cast<bool>(stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size())),
                  "Truncated COLMAP binary");
            if constexpr (std::endian::native == std::endian::big)
                std::reverse(bytes.begin(), bytes.end());
            return std::bit_cast<T>(bytes);
        }
        template <class T>
        void put(std::ostream& stream, T value) {
            auto bytes = std::bit_cast<std::array<unsigned char, sizeof(T)>>(value);
            if constexpr (std::endian::native == std::endian::big)
                std::reverse(bytes.begin(), bytes.end());
            stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        uint64_t count(std::istream& stream, uint64_t minimum_bytes) {
            auto n = get<uint64_t>(stream);
            auto start = stream.tellg();
            stream.seekg(0, std::ios::end);
            auto end = stream.tellg();
            stream.seekg(start);
            check(start >= 0 && end >= start && n <= static_cast<uint64_t>(end - start) / minimum_bytes,
                  "Invalid COLMAP record count");
            return n;
        }
        void eof(std::istream& stream) {
            check(stream.peek() == std::char_traits<char>::eof(), "Trailing COLMAP binary bytes");
        }
        bool row(std::istream& stream, std::string& line, bool allow_empty = false) {
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                auto first = line.find_first_not_of(" \t");
                if (first != std::string::npos && line[first] == '#')
                    continue;
                if (allow_empty || first != std::string::npos)
                    return true;
            }
            return false;
        }
        template <class T>
        T token(std::istream& in) {
            T value{};
            check(static_cast<bool>(in >> value), "Malformed COLMAP text field");
            return value;
        }
        template <class Map, class Value>
        void insert(Map& map, Value value) {
            check(map.emplace(value.id, std::move(value)).second, "Duplicate COLMAP ID");
        }
        template <class Range>
        bool finite(const Range& values) {
            return std::ranges::all_of(values, [](double v) {
                return std::isfinite(v);
            });
        }
        std::ifstream input(const std::filesystem::path& file) {
            std::ifstream in(file, std::ios::binary);
            check(in.good(), "Cannot open COLMAP input");
            return in;
        }
        void read_text(const std::filesystem::path& root, ColmapModel& model) {
            std::string line;
            auto cameras = input(root / "cameras.txt");
            while (row(cameras, line)) {
                std::istringstream in(line);
                ColmapCamera c;
                c.id = token<uint32_t>(in);
                c.model = camera_model_id(token<std::string>(in));
                c.width = token<uint64_t>(in);
                c.height = token<uint64_t>(in);
                for (std::size_t i = 0; i < camera_parameter_count(c.model); ++i)
                    c.parameters.push_back(token<double>(in));
                in >> std::ws;
                check(in.eof(), "Extra camera fields");
                insert(model.cameras, std::move(c));
            }
            auto images = input(root / "images.txt");
            while (row(images, line)) {
                std::istringstream in(line);
                ColmapImage image;
                image.id = token<uint32_t>(in);
                for (auto& v : image.rotation)
                    v = token<double>(in);
                for (auto& v : image.translation)
                    v = token<double>(in);
                image.camera_id = token<uint32_t>(in);
                in >> std::ws;
                std::getline(in, image.name);
                check(row(images, line, true), "Missing image observations line");
                std::istringstream points(line);
                while (points >> std::ws && !points.eof()) {
                    Observation p;
                    p.x = token<double>(points);
                    p.y = token<double>(points);
                    p.point_id = token<int64_t>(points);
                    image.observations.push_back(p);
                }
                insert(model.images, std::move(image));
            }
            if (std::filesystem::exists(root / "points3D.txt")) {
                model.points.emplace();
                auto points = input(root / "points3D.txt");
                while (row(points, line)) {
                    std::istringstream in(line);
                    ColmapPoint p;
                    p.id = token<uint64_t>(in);
                    for (auto& v : p.position)
                        v = token<double>(in);
                    for (auto& v : p.color) {
                        auto channel = token<unsigned>(in);
                        check(channel <= 255, "Invalid point color");
                        v = static_cast<uint8_t>(channel);
                    }
                    p.error = token<double>(in);
                    while (in >> std::ws && !in.eof())
                        p.track.push_back({token<uint32_t>(in), token<uint32_t>(in)});
                    insert(*model.points, std::move(p));
                }
            }
        }
        void read_binary(const std::filesystem::path& root, ColmapModel& model) {
            auto cameras = input(root / "cameras.bin");
            auto n = count(cameras, 24);
            for (uint64_t i = 0; i < n; ++i) {
                ColmapCamera c;
                c.id = get<uint32_t>(cameras);
                c.model = get<int32_t>(cameras);
                c.width = get<uint64_t>(cameras);
                c.height = get<uint64_t>(cameras);
                for (std::size_t j = 0; j < camera_parameter_count(c.model); ++j)
                    c.parameters.push_back(get<double>(cameras));
                insert(model.cameras, std::move(c));
            }
            eof(cameras);
            auto images = input(root / "images.bin");
            n = count(images, 73);
            for (uint64_t i = 0; i < n; ++i) {
                ColmapImage image;
                image.id = get<uint32_t>(images);
                for (auto& v : image.rotation)
                    v = get<double>(images);
                for (auto& v : image.translation)
                    v = get<double>(images);
                image.camera_id = get<uint32_t>(images);
                for (char c = get<char>(images); c; c = get<char>(images)) {
                    check(image.name.size() < 32768, "Image name too long");
                    image.name += c;
                }
                const auto m = count(images, 24);
                for (uint64_t j = 0; j < m; ++j)
                    image.observations.push_back({get<double>(images), get<double>(images), get<int64_t>(images)});
                insert(model.images, std::move(image));
            }
            eof(images);
            if (std::filesystem::exists(root / "points3D.bin")) {
                model.points.emplace();
                auto points = input(root / "points3D.bin");
                n = count(points, 51);
                for (uint64_t i = 0; i < n; ++i) {
                    ColmapPoint p;
                    p.id = get<uint64_t>(points);
                    for (auto& v : p.position)
                        v = get<double>(points);
                    for (auto& v : p.color)
                        v = get<uint8_t>(points);
                    p.error = get<double>(points);
                    auto m = count(points, 8);
                    for (uint64_t j = 0; j < m; ++j)
                        p.track.push_back({get<uint32_t>(points), get<uint32_t>(points)});
                    insert(*model.points, std::move(p));
                }
                eof(points);
            }
        }
    } // namespace
    std::string camera_model_name(int id) {
        (void)camera_parameter_count(id);
        return names[id];
    }
    std::size_t camera_parameter_count(int id) {
        check(id >= 0 && id < static_cast<int>(std::size(counts)), "Unsupported COLMAP camera model");
        return counts[id];
    }
    int camera_model_id(const std::string& name) {
        for (int i = 0; i < static_cast<int>(std::size(names)); ++i)
            if (name == names[i])
                return i;
        throw std::runtime_error("Unsupported COLMAP camera model: " + name);
    }
    std::expected<void, Error> validate_model(const ColmapModel& model) try {
        check(!model.cameras.empty() && !model.images.empty(), "Model requires cameras and images");
        std::set<std::string> names_seen;
        for (const auto& [id, c] : model.cameras) {
            check(id == c.id && id > 0 && c.width > 0 && c.height > 0 && c.width <= INT32_MAX && c.height <= INT32_MAX,
                  "Invalid camera identity/dimensions");
            check(c.parameters.size() == camera_parameter_count(c.model) && finite(c.parameters),
                  "Invalid camera parameters");
            check(c.parameters[0] > 0 && (c.model == 0 || c.model == 2 || c.model == 3 || c.model == 8 ||
                                          c.model == 9 || c.parameters[1] > 0),
                  "Invalid focal length");
        }
        for (const auto& [id, image] : model.images) {
            check(id == image.id && id > 0 && id < INT32_MAX && model.cameras.contains(image.camera_id),
                  "Invalid image ID/camera reference");
            check(!image.name.empty() && image.name.find_first_of("\r\n") == std::string::npos &&
                      image.name.find('\0') == std::string::npos && names_seen.insert(image.name).second,
                  "Invalid/duplicate image name");
            check(finite(image.rotation) && finite(image.translation), "Nonfinite pose");
            double norm = 0;
            for (auto v : image.rotation)
                norm += v * v;
            check(std::abs(norm - 1) < 1e-5, "Quaternion is not unit length");
            for (const auto& p : image.observations) {
                check(std::isfinite(p.x) && std::isfinite(p.y) && p.point_id >= -1, "Invalid observation");
                if (model.points && p.point_id != -1)
                    check(model.points->contains(static_cast<uint64_t>(p.point_id)), "Missing observation point");
            }
        }
        if (model.points) {
            check(!model.points->empty(), "Present points3D must not be empty");
            for (const auto& [id, p] : *model.points) {
                check(id == p.id && id <= INT64_MAX && finite(p.position) && std::isfinite(p.error) && p.error >= 0,
                      "Invalid point");
                std::set<std::pair<uint32_t, uint32_t>> seen;
                for (const auto& t : p.track) {
                    check(model.images.contains(t.image_id) && seen.emplace(t.image_id, t.point_index).second,
                          "Invalid/duplicate point track");
                    const auto& obs = model.images.at(t.image_id).observations;
                    check(t.point_index < obs.size() && obs[t.point_index].point_id == static_cast<int64_t>(id),
                          "Point track/observation mismatch");
                }
            }
            // The two directions describe the same relation, not independent optional metadata.
            for (const auto& [id, image] : model.images)
                for (size_t index = 0; index < image.observations.size(); ++index) {
                    const auto point = image.observations[index].point_id;
                    if (point < 0)
                        continue;
                    const auto& track = model.points->at(static_cast<uint64_t>(point)).track;
                    check(std::ranges::any_of(track, [&](const TrackElement& t) {
                              return t.image_id == id && t.point_index == index;
                          }),
                          "Observation missing reciprocal point track");
                }
        }
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = e.what()});
    }
    std::expected<ColmapModel, Error> read_colmap_model(const std::filesystem::path& root) try {
        ColmapModel model;
        if (std::filesystem::exists(root / "cameras.bin") || std::filesystem::exists(root / "images.bin"))
            read_binary(root, model);
        else
            read_text(root, model);
        if (auto valid = validate_model(model); !valid)
            return std::unexpected(valid.error());
        return model;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = e.what()});
    }
    std::expected<void, Error> write_colmap_model(const std::filesystem::path& root, const ColmapModel& model,
                                                  ModelFormat format) try {
        if (auto valid = validate_model(model); !valid)
            return valid;
        check(!std::filesystem::exists(root), "Model output must be a fresh directory");
        std::filesystem::create_directories(root);
        const bool binary = format == ModelFormat::Binary;
        const std::string ext = binary ? ".bin" : ".txt";
        auto out = [&](const std::string& name) {
            std::ofstream f(root / (name + ext), std::ios::binary);
            f.exceptions(std::ios::badbit | std::ios::failbit);
            f << std::setprecision(17);
            return f;
        };
        auto cameras = out("cameras"), images = out("images");
        if (binary) {
            put<uint64_t>(cameras, model.cameras.size());
            put<uint64_t>(images, model.images.size());
        }
        for (const auto& [id, c] : model.cameras) {
            if (binary) {
                put(cameras, id);
                put<int32_t>(cameras, c.model);
                put(cameras, c.width);
                put(cameras, c.height);
                for (auto v : c.parameters)
                    put(cameras, v);
            } else {
                cameras << id << ' ' << camera_model_name(c.model) << ' ' << c.width << ' ' << c.height;
                for (auto v : c.parameters)
                    cameras << ' ' << v;
                cameras << '\n';
            }
        }
        for (const auto& [id, image] : model.images) {
            if (binary) {
                put(images, id);
                for (auto v : image.rotation)
                    put(images, v);
                for (auto v : image.translation)
                    put(images, v);
                put(images, image.camera_id);
                images.write(image.name.c_str(), image.name.size() + 1);
                put<uint64_t>(images, image.observations.size());
                for (const auto& p : image.observations) {
                    put(images, p.x);
                    put(images, p.y);
                    put(images, p.point_id);
                }
            } else {
                images << id;
                for (auto v : image.rotation)
                    images << ' ' << v;
                for (auto v : image.translation)
                    images << ' ' << v;
                images << ' ' << image.camera_id << ' ' << image.name << '\n';
                for (const auto& p : image.observations)
                    images << p.x << ' ' << p.y << ' ' << p.point_id << ' ';
                images << '\n';
            }
        }
        cameras.close();
        images.close();
        if (model.points) {
            auto points = out("points3D");
            if (binary)
                put<uint64_t>(points, model.points->size());
            for (const auto& [id, p] : *model.points) {
                if (binary) {
                    put(points, id);
                    for (auto v : p.position)
                        put(points, v);
                    for (auto v : p.color)
                        put(points, v);
                    put(points, p.error);
                    put<uint64_t>(points, p.track.size());
                    for (const auto& t : p.track) {
                        put(points, t.image_id);
                        put(points, t.point_index);
                    }
                } else {
                    points << id;
                    for (auto v : p.position)
                        points << ' ' << v;
                    for (auto v : p.color)
                        points << ' ' << unsigned(v);
                    points << ' ' << p.error;
                    for (const auto& t : p.track)
                        points << ' ' << t.image_id << ' ' << t.point_index;
                    points << '\n';
                }
            }
            points.close();
        }
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::IoFailure, .message = e.what()});
    }
    std::array<double, 9> quaternion_rotation(const std::array<double, 4>& q) {
        const auto [w, x, y, z] = q;
        return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
    }
    std::expected<ColmapImage, Error> scanner_pose_to_colmap(const std::array<float, 16>& c) try {
        check(finite(c) && std::abs(c[12]) + std::abs(c[13]) + std::abs(c[14]) + std::abs(c[15] - 1) < 1e-5,
              "Invalid homogeneous pose");
        std::array<double, 9> r{c[0], c[4], c[8], c[1], c[5], c[9], c[2], c[6], c[10]};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double dot = 0;
                for (int k = 0; k < 3; ++k)
                    dot += r[i * 3 + k] * r[j * 3 + k];
                check(std::abs(dot - (i == j ? 1 : 0)) < 1e-4, "Scanner rotation is not orthonormal");
            }
        const double det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) +
                           r[2] * (r[3] * r[7] - r[4] * r[6]);
        check(det > 0, "Scanner rotation contains reflection");
        ColmapImage image;
        auto& q = image.rotation;
        const double trace = r[0] + r[4] + r[8];
        if (trace > 0) {
            double s = std::sqrt(trace + 1) * 2;
            q = {s / 4, (r[7] - r[5]) / s, (r[2] - r[6]) / s, (r[3] - r[1]) / s};
        } else {
            int i = r[4] > r[0] ? 1 : 0;
            if (r[8] > r[i * 3 + i])
                i = 2;
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            double s = std::sqrt(1 + r[i * 3 + i] - r[j * 3 + j] - r[k * 3 + k]) * 2;
            q[0] = (r[k * 3 + j] - r[j * 3 + k]) / s;
            q[i + 1] = s / 4;
            q[j + 1] = (r[j * 3 + i] + r[i * 3 + j]) / s;
            q[k + 1] = (r[k * 3 + i] + r[i * 3 + k]) / s;
        }
        if (q[0] < 0)
            for (auto& v : q)
                v = -v;
        for (int i = 0; i < 3; ++i)
            image.translation[i] = -(r[i * 3] * c[3] + r[i * 3 + 1] * c[7] + r[i * 3 + 2] * c[11]) * 0.001;
        return image;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidDataset, .message = e.what()});
    }
    std::expected<ColmapCamera, Error> scale_camera(const ColmapCamera& camera, uint64_t width, uint64_t height) try {
        check(camera.width && camera.height && width && height &&
                  camera.parameters.size() == camera_parameter_count(camera.model),
              "Invalid camera scaling");
        ColmapCamera c = camera;
        const double sx = double(width) / c.width, sy = double(height) / c.height;
        const bool shared = c.model == 0 || c.model == 2 || c.model == 3 || c.model == 8 || c.model == 9;
        if (shared) {
            check(std::abs(sx - sy) < 1e-9, "Anisotropic scaling requires separate focal lengths");
            c.parameters[0] *= sx;
            c.parameters[1] *= sx;
            c.parameters[2] *= sy;
        } else {
            c.parameters[0] *= sx;
            c.parameters[1] *= sy;
            c.parameters[2] *= sx;
            c.parameters[3] *= sy;
        }
        c.width = width;
        c.height = height;
        return c;
    } catch (const std::exception& e) {
        return std::unexpected(Error{.code = ErrorCode::InvalidRequest, .message = e.what()});
    }
} // namespace lfs::preprocess
