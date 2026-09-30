/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/colmap_database.hpp"
#include "preprocessing/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <sqlite3.h>

namespace lfs::preprocess {
    namespace {
        void check(bool value, const char* message) {
            if (!value)
                throw std::runtime_error(message);
        }
        Error db_error(const std::exception& e) { return {.code = ErrorCode::InvalidDataset, .message = e.what()}; }
        void sql(sqlite3* db, const char* command) {
            char* error = nullptr;
            if (sqlite3_exec(db, command, nullptr, nullptr, &error) != SQLITE_OK) {
                std::string message = error ? error : sqlite3_errmsg(db);
                sqlite3_free(error);
                throw std::runtime_error(message);
            }
        }
        struct Statement {
            sqlite3_stmt* value = nullptr;
            Statement(sqlite3* db, const char* text) {
                if (sqlite3_prepare_v2(db, text, -1, &value, nullptr) != SQLITE_OK)
                    throw std::runtime_error(sqlite3_errmsg(db));
            }
            ~Statement() { sqlite3_finalize(value); }
            void integer(int index, int64_t v) {
                check(sqlite3_bind_int64(value, index, v) == SQLITE_OK, "SQLite integer bind failed");
            }
            void text(int index, const std::string& v) {
                check(sqlite3_bind_text(value, index, v.c_str(), static_cast<int>(v.size()), SQLITE_TRANSIENT) ==
                          SQLITE_OK,
                      "SQLite text bind failed");
            }
            void blob(int index, const void* data, std::size_t bytes) {
                check(bytes <= INT32_MAX, "SQLite blob too large");
                check(sqlite3_bind_blob(value, index, bytes ? data : "", static_cast<int>(bytes), SQLITE_TRANSIENT) ==
                          SQLITE_OK,
                      "SQLite blob bind failed");
            }
            template <class T>
            void optional(int index, const std::optional<T>& v) {
                if (v)
                    blob(index, v->data(), sizeof(typename T::value_type) * v->size());
                else
                    check(sqlite3_bind_null(value, index) == SQLITE_OK, "SQLite null bind failed");
            }
            bool next() {
                auto code = sqlite3_step(value);
                if (code == SQLITE_ROW)
                    return true;
                if (code == SQLITE_DONE)
                    return false;
                throw std::runtime_error(sqlite3_errmsg(sqlite3_db_handle(value)));
            }
            void done() { check(!next(), "Unexpected SQLite result row"); }
            int64_t integer(int index) const { return sqlite3_column_int64(value, index); }
            std::string text(int index) const {
                auto p = sqlite3_column_text(value, index);
                check(p != nullptr, "Missing database text");
                return {reinterpret_cast<const char*>(p), static_cast<std::size_t>(sqlite3_column_bytes(value, index))};
            }
            template <class T>
            std::vector<T> blob(int index, std::size_t expected) const {
                check(expected <= INT32_MAX / sizeof(T) &&
                          sqlite3_column_bytes(value, index) == static_cast<int>(expected * sizeof(T)),
                      "Invalid database blob dimensions");
                std::vector<T> result(expected);
                if (expected) {
                    const auto* p = sqlite3_column_blob(value, index);
                    check(p != nullptr, "Null database blob");
                    std::memcpy(result.data(), p, expected * sizeof(T));
                }
                return result;
            }
            template <std::size_t N>
            std::optional<std::array<double, N>> optional(int index) const {
                if (sqlite3_column_type(value, index) == SQLITE_NULL)
                    return {};
                auto values = blob<double>(index, N);
                std::array<double, N> out{};
                std::copy(values.begin(), values.end(), out.begin());
                check(std::ranges::all_of(out,
                                          [](double x) {
                                              return std::isfinite(x);
                                          }),
                      "Nonfinite geometry matrix");
                return out;
            }
        };
        struct Transaction {
            sqlite3* db;
            bool committed = false;
            explicit Transaction(sqlite3* d) : db(d) { sql(db, "BEGIN IMMEDIATE"); }
            ~Transaction() {
                if (!committed)
                    sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            }
            void commit() {
                sql(db, "COMMIT");
                committed = true;
            }
        };
        std::set<std::string> columns(sqlite3* db, const std::string& table) {
            // Only called with internal literal table names.
            Statement query(db, ("PRAGMA table_info(" + table + ")").c_str());
            std::set<std::string> result;
            while (query.next())
                result.insert(query.text(1));
            return result;
        }
        void validate_schema(sqlite3* db) {
            const std::map<std::string, std::set<std::string>> required{
                {"cameras", {"camera_id", "model", "width", "height", "params", "prior_focal_length"}},
                {"images", {"image_id", "name", "camera_id"}},
                {"keypoints", {"image_id", "rows", "cols", "data"}},
                {"matches", {"pair_id", "rows", "cols", "data"}},
                {"two_view_geometries", {"pair_id", "rows", "cols", "data", "config", "F", "E", "H", "qvec", "tvec"}}};
            for (const auto& [table, wanted] : required) {
                auto actual = columns(db, table);
                check(std::ranges::includes(actual, wanted), "Unsupported COLMAP database schema");
            }
            auto rigs = columns(db, "rigs"), frames = columns(db, "frames"), data = columns(db, "frame_data");
            if (!rigs.empty() || !frames.empty() || !data.empty()) {
                check(rigs.contains("rig_id") && rigs.contains("ref_sensor_id") && rigs.contains("ref_sensor_type") &&
                          frames.contains("frame_id") && frames.contains("rig_id") && data.contains("frame_id") &&
                          data.contains("data_id") && data.contains("sensor_id") && data.contains("sensor_type"),
                      "Unsupported/incomplete rig/frame schema");
            }
        }
        constexpr const char* base_schema = R"SQL(
CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,model INTEGER NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,params BLOB,prior_focal_length INTEGER NOT NULL);
CREATE TABLE images(image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,name TEXT NOT NULL UNIQUE,camera_id INTEGER NOT NULL,CHECK(image_id>=0 AND image_id<2147483647),FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY NOT NULL,rows INTEGER NOT NULL,cols INTEGER NOT NULL,data BLOB,FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY NOT NULL,type INTEGER NOT NULL,rows INTEGER NOT NULL,cols INTEGER NOT NULL,data BLOB,FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE matches(pair_id INTEGER PRIMARY KEY NOT NULL,rows INTEGER NOT NULL,cols INTEGER NOT NULL,data BLOB);
CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY NOT NULL,rows INTEGER NOT NULL,cols INTEGER NOT NULL,data BLOB,config INTEGER NOT NULL,F BLOB,E BLOB,H BLOB,qvec BLOB,tvec BLOB);
CREATE TABLE pose_priors(pose_prior_id INTEGER PRIMARY KEY NOT NULL,corr_data_id INTEGER NOT NULL,corr_sensor_id INTEGER NOT NULL,corr_sensor_type INTEGER NOT NULL,position BLOB,position_covariance BLOB,gravity BLOB,coordinate_system INTEGER NOT NULL);
)SQL";
        constexpr const char* rig_schema = R"SQL(
CREATE TABLE rigs(rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,ref_sensor_id INTEGER NOT NULL,ref_sensor_type INTEGER NOT NULL);
CREATE TABLE rig_sensors(rig_id INTEGER NOT NULL,sensor_id INTEGER NOT NULL,sensor_type INTEGER NOT NULL,sensor_from_rig BLOB,FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE TABLE frames(frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,rig_id INTEGER NOT NULL,FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE);
CREATE TABLE frame_data(frame_id INTEGER NOT NULL,data_id INTEGER NOT NULL,sensor_id INTEGER NOT NULL,sensor_type INTEGER NOT NULL,FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE);
)SQL";
        uint64_t checked_pair(uint32_t a, uint32_t b) {
            auto p = image_pair_id(a, b);
            if (!p)
                throw std::invalid_argument(p.error().message);
            return *p;
        }
        Matches read_matches_row(Statement& query, int rows = 0, int cols = 1, int data = 2) {
            const auto n = query.integer(rows);
            check(n >= 0 && n <= INT32_MAX / 8 && query.integer(cols) == 2, "Invalid match matrix");
            return query.blob<std::array<uint32_t, 2>>(data, static_cast<std::size_t>(n));
        }
    } // namespace
    struct ColmapDatabase::Impl {
        sqlite3* db = nullptr;
        ~Impl() {
            if (db)
                sqlite3_close_v2(db);
        }
        void open(const std::filesystem::path& path, int flags) {
            if (sqlite3_open_v2(path_utf8(path).c_str(), &db, flags, nullptr) != SQLITE_OK)
                throw std::runtime_error(sqlite3_errmsg(db));
            sqlite3_busy_timeout(db, 2000);
        }
    };
    ColmapDatabase::ColmapDatabase(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
    ColmapDatabase::~ColmapDatabase() = default;
    std::expected<uint64_t, Error> image_pair_id(uint32_t a, uint32_t b) {
        if (!a || !b || a >= colmap_max_image_id || b >= colmap_max_image_id || a == b)
            return std::unexpected(
                Error{.code = ErrorCode::InvalidRequest, .message = "Pair needs two distinct valid image IDs"});
        if (a > b)
            std::swap(a, b);
        return uint64_t(a) * colmap_max_image_id + b;
    }
    std::expected<std::unique_ptr<ColmapDatabase>, Error> ColmapDatabase::create(const std::filesystem::path& path,
                                                                                 DatabaseSchema schema) try {
        static_assert(std::endian::native == std::endian::little, "COLMAP SQLite blobs require little-endian encoding");
        std::ofstream reservation(path, std::ios::binary | std::ios::noreplace);
        check(reservation.good(), "Database output must be a fresh writable file");
        reservation.close();
        auto impl = std::make_unique<Impl>();
        impl->open(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX);
        sql(impl->db, "PRAGMA foreign_keys=ON");
        Transaction tx(impl->db);
        sql(impl->db, base_schema);
        if (schema == DatabaseSchema::RigFrames)
            sql(impl->db, rig_schema);
        tx.commit();
        return std::unique_ptr<ColmapDatabase>(new ColmapDatabase(std::move(impl)));
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<std::unique_ptr<ColmapDatabase>, Error>
    ColmapDatabase::clone(const std::filesystem::path& source, const std::filesystem::path& destination) try {
        auto original = std::make_unique<Impl>();
        original->open(source, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX);
        validate_schema(original->db);
        std::ofstream reservation(destination, std::ios::binary | std::ios::noreplace);
        check(reservation.good(), "Database clone destination must be fresh");
        reservation.close();
        auto impl = std::make_unique<Impl>();
        impl->open(destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX);
        auto* backup = sqlite3_backup_init(impl->db, "main", original->db, "main");
        check(backup != nullptr, "Cannot start database snapshot");
        const auto status = sqlite3_backup_step(backup, -1);
        const auto finish = sqlite3_backup_finish(backup);
        check(status == SQLITE_DONE && finish == SQLITE_OK, "Database snapshot failed");
        sql(impl->db, "PRAGMA foreign_keys=ON");
        return std::unique_ptr<ColmapDatabase>(new ColmapDatabase(std::move(impl)));
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<void, Error> ColmapDatabase::sync_metadata(const ColmapModel& model) {
        return sync_metadata_impl(model, false);
    }
    std::expected<void, Error> ColmapDatabase::synchronize_scan_metadata(const ColmapModel& model) {
        return sync_metadata_impl(model, true);
    }
    std::expected<void, Error> ColmapDatabase::sync_metadata_impl(const ColmapModel& model, bool rebuild_scan_rigs) try {
        if (auto result = validate_model(model); !result)
            return result;
        auto* db = impl_->db;
        Transaction tx(db);
        // Take the write snapshot before introspection: another connection may
        // have changed the schema since our previous statement was prepared.
        validate_schema(db);
        if (rebuild_scan_rigs) {
            Statement count(db, "SELECT COUNT(*) FROM images");
            check(count.next() && count.integer(0) == static_cast<int64_t>(model.images.size()) && !model.images.empty(),
                  "Scanner synchronization requires the complete extracted image set");
            if (!columns(db, "rigs").empty()) {
                Statement rigs(db, "SELECT rig_id,ref_sensor_id,ref_sensor_type FROM rigs");
                while (rigs.next())
                    check(rigs.integer(0) == rigs.integer(1) && rigs.integer(2) == 0,
                          "Scanner synchronization cannot replace nontrivial rigs");
                if (!columns(db, "rig_sensors").empty()) {
                    Statement sensors(db, "SELECT COUNT(*) FROM rig_sensors");
                    check(sensors.next() && sensors.integer(0) == 0,
                          "Scanner synchronization cannot replace multi-sensor rigs");
                }
                Statement frames(db, "SELECT f.frame_id,f.rig_id,i.camera_id FROM frames f "
                                     "LEFT JOIN images i ON i.image_id=f.frame_id");
                while (frames.next())
                    check(model.images.contains(static_cast<uint32_t>(frames.integer(0))) &&
                              frames.integer(1) == frames.integer(2),
                          "Scanner synchronization cannot replace nontrivial frames");
                Statement data(db, "SELECT fd.frame_id,fd.data_id,fd.sensor_id,fd.sensor_type,i.camera_id "
                                   "FROM frame_data fd LEFT JOIN images i ON i.image_id=fd.data_id");
                std::set<int64_t> seen;
                while (data.next())
                    check(model.images.contains(static_cast<uint32_t>(data.integer(1))) &&
                              data.integer(0) == data.integer(1) && data.integer(2) == data.integer(4) &&
                              data.integer(3) == 0 && seen.insert(data.integer(1)).second,
                          "Scanner synchronization cannot replace nontrivial frame data");
                const auto frame_columns = columns(db, "frames");
                for (const auto& [name, value] : std::map<std::string, std::string>{{"qw", "1"}, {"qx", "0"}, {"qy", "0"}, {"qz", "0"}, {"tx", "0"}, {"ty", "0"}, {"tz", "0"}})
                    if (frame_columns.contains(name)) {
                        Statement pose(db, ("SELECT COUNT(*) FROM frames WHERE " + name + " IS NOT " + value).c_str());
                        check(pose.next() && pose.integer(0) == 0,
                              "Scanner synchronization cannot replace nonidentity frame poses");
                    }
            }
        }
        // Reject remapping before modifying any row: existing feature IDs remain authoritative.
        for (const auto& [id, image] : model.images) {
            Statement q(db, "SELECT image_id,name FROM images WHERE image_id=? OR name=?");
            q.integer(1, id);
            q.text(2, image.name);
            bool found = false;
            while (q.next()) {
                check(q.integer(0) == id && q.text(1) == image.name, "Database image ID/name mismatch");
                found = true;
            }
            check(!rebuild_scan_rigs || found, "Scanner synchronization requires extracted image IDs");
        }
        if (rebuild_scan_rigs && !columns(db, "rigs").empty()) {
            sql(db, "DELETE FROM frame_data");
            sql(db, "DELETE FROM frames");
            sql(db, "DELETE FROM rigs");
        }
        for (const auto& [id, c] : model.cameras) {
            Statement q(db, "INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) "
                            "VALUES(?,?,?,?,?,1) ON CONFLICT(camera_id) DO UPDATE SET "
                            "model=excluded.model,width=excluded.width,height=excluded.height,params=excluded.params,"
                            "prior_focal_length=1");
            q.integer(1, id);
            q.integer(2, c.model);
            q.integer(3, c.width);
            q.integer(4, c.height);
            q.blob(5, c.parameters.data(), c.parameters.size() * sizeof(double));
            q.done();
        }
        for (const auto& [id, image] : model.images) {
            Statement q(db, "INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?) ON CONFLICT(image_id) DO UPDATE "
                            "SET camera_id=excluded.camera_id");
            q.integer(1, id);
            q.text(2, image.name);
            q.integer(3, image.camera_id);
            q.done();
        }
        if (!columns(db, "rigs").empty()) {
            // Only initialize trivial rigs/frames; never silently rewrite a multi-sensor input rig.
            for (const auto& [id, c] : model.cameras) {
                Statement old(db, "SELECT ref_sensor_id,ref_sensor_type FROM rigs WHERE rig_id=?");
                old.integer(1, id);
                if (old.next())
                    check(old.integer(0) == id && old.integer(1) == 0, "Nontrivial rig mapping unsupported");
                if (!columns(db, "rig_sensors").empty()) {
                    Statement sensors(db, "SELECT COUNT(*) FROM rig_sensors WHERE rig_id=?");
                    sensors.integer(1, id);
                    check(sensors.next() && sensors.integer(0) == 0, "Multi-sensor rig unsupported");
                }
                Statement q(db, "INSERT INTO rigs(rig_id,ref_sensor_id,ref_sensor_type) VALUES(?,?,0) ON "
                                "CONFLICT(rig_id) DO NOTHING");
                q.integer(1, id);
                q.integer(2, id);
                q.done();
            }
            const auto frame_columns = columns(db, "frames");
            for (const auto& [id, image] : model.images) {
                Statement old(db, "SELECT rig_id FROM frames WHERE frame_id=?");
                old.integer(1, id);
                if (old.next())
                    check(old.integer(0) == image.camera_id, "Existing frame rig mismatch");
                std::string statement = "INSERT INTO frames(frame_id,rig_id";
                std::string values = ") VALUES(?,?";
                for (const auto& [name, value] : std::map<std::string, std::string>{{"qw", "1"},
                                                                                    {"qx", "0"},
                                                                                    {"qy", "0"},
                                                                                    {"qz", "0"},
                                                                                    {"tx", "0"},
                                                                                    {"ty", "0"},
                                                                                    {"tz", "0"}})
                    if (frame_columns.contains(name)) {
                        statement += "," + name;
                        values += "," + value;
                    }
                statement += values + ") ON CONFLICT(frame_id) DO NOTHING";
                Statement q(db, statement.c_str());
                q.integer(1, id);
                q.integer(2, image.camera_id);
                q.done();
                Statement old_data(db, "SELECT frame_id,sensor_id,sensor_type FROM frame_data WHERE data_id=?");
                old_data.integer(1, id);
                bool found = false;
                while (old_data.next()) {
                    check(!found && old_data.integer(0) == id && old_data.integer(1) == image.camera_id &&
                              old_data.integer(2) == 0,
                          "Existing frame data mismatch");
                    found = true;
                }
                if (!found) {
                    Statement data(db,
                                   "INSERT INTO frame_data(frame_id,data_id,sensor_id,sensor_type) VALUES(?,?,?,0)");
                    data.integer(1, id);
                    data.integer(2, id);
                    data.integer(3, image.camera_id);
                    data.done();
                }
            }
        }
        tx.commit();
        return {};
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<ColmapModel, Error> ColmapDatabase::read_metadata() const try {
        ColmapModel model;
        Statement c(impl_->db, "SELECT camera_id,model,width,height,params FROM cameras ORDER BY camera_id");
        while (c.next()) {
            ColmapCamera camera;
            camera.id = static_cast<uint32_t>(c.integer(0));
            camera.model = static_cast<int>(c.integer(1));
            camera.width = c.integer(2);
            camera.height = c.integer(3);
            camera.parameters = c.blob<double>(4, camera_parameter_count(camera.model));
            model.cameras.emplace(camera.id, std::move(camera));
        }
        Statement i(impl_->db, "SELECT image_id,camera_id,name FROM images ORDER BY image_id");
        while (i.next()) {
            ColmapImage image;
            image.id = static_cast<uint32_t>(i.integer(0));
            image.camera_id = static_cast<uint32_t>(i.integer(1));
            image.name = i.text(2);
            model.images.emplace(image.id, std::move(image));
        }
        if (auto result = validate_model(model); !result)
            return std::unexpected(result.error());
        return model;
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<std::map<std::string, uint64_t>, Error> ColmapDatabase::statistics() const try {
        Transaction snapshot(impl_->db);
        validate_schema(impl_->db);
        std::map<std::string, uint64_t> counts;
        for (const auto* table : {"cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries",
                                  "rigs", "rig_sensors", "frames", "frame_data", "pose_priors"}) {
            if (columns(impl_->db, table).empty())
                continue;
            Statement query(impl_->db, (std::string("SELECT COUNT(*) FROM ") + table).c_str());
            check(query.next() && query.integer(0) >= 0, "Invalid database table row count");
            counts.emplace(table, static_cast<uint64_t>(query.integer(0)));
        }
        snapshot.commit();
        return counts;
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<void, Error> ColmapDatabase::write_keypoints(uint32_t image, const Keypoints& points) try {
        check((points.columns == 2 || points.columns == 4 || points.columns == 6) &&
                  points.data.size() % points.columns == 0,
              "Invalid keypoint shape");
        check(std::ranges::all_of(points.data,
                                  [](float v) {
                                      return std::isfinite(v);
                                  }),
              "Nonfinite keypoint");
        Transaction tx(impl_->db);
        Statement q(impl_->db, "INSERT OR REPLACE INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)");
        q.integer(1, image);
        q.integer(2, points.data.size() / points.columns);
        q.integer(3, points.columns);
        q.blob(4, points.data.data(), points.data.size() * sizeof(float));
        q.done();
        tx.commit();
        return {};
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<Keypoints, Error> ColmapDatabase::read_keypoints(uint32_t image) const try {
        Statement q(impl_->db, "SELECT rows,cols,data FROM keypoints WHERE image_id=?");
        q.integer(1, image);
        check(q.next(), "Missing keypoints");
        const auto rows = q.integer(0), cols = q.integer(1);
        check(rows >= 0 && rows <= INT32_MAX / 24 && (cols == 2 || cols == 4 || cols == 6),
              "Invalid keypoint dimensions");
        auto data = q.blob<float>(2, static_cast<std::size_t>(rows * cols));
        check(std::ranges::all_of(data, [](float value) { return std::isfinite(value); }), "Nonfinite keypoint blob");
        return Keypoints{static_cast<uint32_t>(cols), std::move(data)};
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<void, Error> ColmapDatabase::write_matches(uint32_t a, uint32_t b, const Matches& matches) try {
        const auto pair = checked_pair(a, b);
        Matches ordered = matches;
        if (a > b)
            for (auto& row : ordered)
                std::swap(row[0], row[1]);
        Transaction tx(impl_->db);
        Statement q(impl_->db, "INSERT OR REPLACE INTO matches(pair_id,rows,cols,data) VALUES(?,?,2,?)");
        q.integer(1, pair);
        q.integer(2, ordered.size());
        q.blob(3, ordered.data(), ordered.size() * sizeof(ordered[0]));
        q.done();
        tx.commit();
        return {};
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<Matches, Error> ColmapDatabase::read_matches(uint32_t a, uint32_t b) const try {
        Statement q(impl_->db, "SELECT rows,cols,data FROM matches WHERE pair_id=?");
        q.integer(1, checked_pair(a, b));
        check(q.next(), "Missing matches");
        auto result = read_matches_row(q);
        if (a > b)
            for (auto& row : result)
                std::swap(row[0], row[1]);
        return result;
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<void, Error> ColmapDatabase::write_geometry(uint32_t a, uint32_t b,
                                                              const TwoViewGeometry& geometry) try {
        check(a < b, "Geometry requires canonical image order");
        const auto pair = checked_pair(a, b);
        auto valid = [](const auto& value) {
            return !value || std::ranges::all_of(*value, [](double v) {
                return std::isfinite(v);
            });
        };
        check(valid(geometry.fundamental) && valid(geometry.essential) && valid(geometry.homography) &&
                  valid(geometry.rotation) && valid(geometry.translation),
              "Nonfinite geometry");
        Transaction tx(impl_->db);
        Statement q(impl_->db,
                    "INSERT OR REPLACE INTO two_view_geometries(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) "
                    "VALUES(?,?,2,?,?,?,?,?,?,?)");
        q.integer(1, pair);
        q.integer(2, geometry.matches.size());
        q.blob(3, geometry.matches.data(), geometry.matches.size() * sizeof(geometry.matches[0]));
        q.integer(4, geometry.configuration);
        q.optional(5, geometry.fundamental);
        q.optional(6, geometry.essential);
        q.optional(7, geometry.homography);
        q.optional(8, geometry.rotation);
        q.optional(9, geometry.translation);
        q.done();
        tx.commit();
        return {};
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
    std::expected<TwoViewGeometry, Error> ColmapDatabase::read_geometry(uint32_t a, uint32_t b) const try {
        check(a < b, "Geometry requires canonical image order");
        Statement q(impl_->db, "SELECT rows,cols,data,config,F,E,H,qvec,tvec FROM two_view_geometries WHERE pair_id=?");
        q.integer(1, checked_pair(a, b));
        check(q.next(), "Missing two-view geometry");
        TwoViewGeometry g;
        g.matches = read_matches_row(q);
        g.configuration = static_cast<int>(q.integer(3));
        g.fundamental = q.optional<9>(4);
        g.essential = q.optional<9>(5);
        g.homography = q.optional<9>(6);
        g.rotation = q.optional<4>(7);
        g.translation = q.optional<3>(8);
        return g;
    } catch (const std::exception& e) { return std::unexpected(db_error(e)); }
} // namespace lfs::preprocess
