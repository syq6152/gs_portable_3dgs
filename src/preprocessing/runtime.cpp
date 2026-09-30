/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "preprocessing/runtime.hpp"
#include "core/executable_path.hpp"
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#ifdef _WIN32
#include <bcrypt.h>
#endif

namespace lfs::preprocess {
    namespace {
        Error error(ErrorCode code, std::string message) { return {.code = code, .message = std::move(message)}; }
        std::filesystem::path relative_path(const std::string& text) {
            auto path = std::filesystem::u8path(text);
            if (path.empty() || path.is_absolute() || path.has_root_name())
                throw std::runtime_error("Expected relative manifest path");
            for (const auto& part : path)
                if (part == "..")
                    throw std::runtime_error("Manifest path escapes root");
            return path;
        }
        std::string sha256(const std::filesystem::path& path) {
#ifdef _WIN32
            struct Hash {
                BCRYPT_ALG_HANDLE algorithm = nullptr;
                BCRYPT_HASH_HANDLE hash = nullptr;
                ~Hash() {
                    if (hash)
                        BCryptDestroyHash(hash);
                    if (algorithm)
                        BCryptCloseAlgorithmProvider(algorithm, 0);
                }
            } state;
            if (BCryptOpenAlgorithmProvider(&state.algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
                BCryptCreateHash(state.algorithm, &state.hash, nullptr, 0, nullptr, 0, 0) < 0)
                throw std::runtime_error("Cannot initialize SHA-256");
            std::ifstream input(path, std::ios::binary);
            if (!input)
                throw std::runtime_error("Cannot read manifest file: " + path_utf8(path));
            std::array<char, 65536> buffer{};
            while (input) {
                input.read(buffer.data(), buffer.size());
                if (BCryptHashData(state.hash, reinterpret_cast<PUCHAR>(buffer.data()), static_cast<ULONG>(input.gcount()), 0) < 0)
                    throw std::runtime_error("SHA-256 update failed");
            }
            if (!input.eof())
                throw std::runtime_error("Runtime file read failed");
            std::array<unsigned char, 32> digest{};
            if (BCryptFinishHash(state.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0)
                throw std::runtime_error("SHA-256 finish failed");
            std::string out;
            for (auto byte : digest) {
                out += "0123456789abcdef"[byte >> 4];
                out += "0123456789abcdef"[byte & 15];
            }
            return out;
#else
            throw std::runtime_error("Windows runtime hashing unavailable on this platform");
#endif
        }
    } // namespace

    std::expected<RuntimePaths, Error> resolve_runtime_paths(
        const std::filesystem::path& executable_dir,
        std::optional<std::filesystem::path> colmap_override,
        std::optional<std::filesystem::path> super_resolution_override,
        std::optional<std::filesystem::path> manifest_override) {
        if ((colmap_override || super_resolution_override) && !manifest_override)
            return std::unexpected(error(ErrorCode::InvalidRequest, "Runtime overrides require an explicit matching manifest"));
        RuntimePaths paths;
        paths.root = (executable_dir.empty() ? core::getRuntimeDataDir() : std::filesystem::absolute(executable_dir)) / "third_party";
        paths.colmap_root = std::filesystem::absolute(colmap_override.value_or(paths.root / "colmap-cuda-cli"));
        paths.colmap_executable = paths.colmap_root / "bin" / "sfm.exe";
        paths.super_resolution_root = std::filesystem::absolute(super_resolution_override.value_or(paths.root / "SuperResolution"));
        paths.super_resolution_executable = paths.super_resolution_root / "preprocess.exe";
        paths.super_resolution_models = paths.super_resolution_root / "models";
        // The release package keeps provider payloads only. The source manifest
        // remains a build-time integrity contract; custom development overrides
        // may still provide an external matching manifest.
        paths.manifest = manifest_override ? std::filesystem::absolute(*manifest_override) : std::filesystem::path{};
        if (auto result = validate_runtime_paths(paths); !result)
            return std::unexpected(result.error());
        if (auto result = verify_runtime_manifest(paths); !result)
            return std::unexpected(result.error());
        return paths;
    }

    std::expected<void, Error> validate_runtime_paths(const RuntimePaths& paths) {
        for (const auto& path : {paths.colmap_executable, paths.super_resolution_executable}) {
            std::error_code ec;
            if (!std::filesystem::is_regular_file(path, ec))
                return std::unexpected(error(ErrorCode::RuntimeMissing, "Missing runtime: " + path_utf8(path)));
        }
        return {};
    }

    std::expected<void, Error> verify_runtime_manifest(const RuntimePaths& paths) {
        if (paths.manifest.empty())
            return {};
        try {
            std::ifstream input(paths.manifest);
            const auto manifest = nlohmann::json::parse(input);
            if (manifest.at("schema_version") != 1)
                throw std::runtime_error("Unsupported runtime manifest major");
            std::set<std::string> components;
            for (const auto& component : manifest.at("components")) {
                const auto id = component.at("component").get<std::string>();
                if (!components.insert(id).second)
                    throw std::runtime_error("Duplicate runtime component");
                const auto root = id == "colmap" ? paths.colmap_root : paths.super_resolution_root;
                if (id != "colmap" && id != "super_resolution")
                    throw std::runtime_error("Unknown component");
                std::set<std::filesystem::path> files;
                for (const auto& item : component.at("files")) {
                    const auto relative = relative_path(item.at("path").get<std::string>());
                    if (!files.insert(relative).second)
                        throw std::runtime_error("Duplicate runtime file");
                    const auto path = root / relative;
                    // Reject links/reparse points in payload paths; the manifest must describe real contained files.
                    auto cursor = root;
                    for (const auto& part : relative) {
                        cursor /= part;
                        if (std::filesystem::is_symlink(cursor))
                            throw std::runtime_error("Runtime symlink rejected");
#ifdef _WIN32
                        if (GetFileAttributesW(cursor.c_str()) != INVALID_FILE_ATTRIBUTES &&
                            (GetFileAttributesW(cursor.c_str()) & FILE_ATTRIBUTE_REPARSE_POINT))
                            throw std::runtime_error("Runtime reparse point rejected");
#endif
                    }
                    if (!std::filesystem::is_regular_file(path))
                        return std::unexpected(error(ErrorCode::RuntimeMissing, "Missing runtime file: " + path_utf8(path)));
                    if (std::filesystem::file_size(path) != item.at("byte_size").get<uintmax_t>() ||
                        sha256(path) != item.at("sha256").get<std::string>())
                        throw std::runtime_error("Runtime hash mismatch: " + path_utf8(path));
                }
                if (!files.contains(relative_path(component.at("entrypoint").get<std::string>())))
                    throw std::runtime_error("Entrypoint not covered by manifest");
                const auto expected_entry = id == "colmap" ? std::filesystem::path("bin/sfm.exe") : std::filesystem::path("preprocess.exe");
                if (relative_path(component.at("entrypoint").get<std::string>()) != expected_entry)
                    throw std::runtime_error("Unexpected runtime entrypoint");
                for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
                    if (item.is_regular_file() && !files.contains(item.path().lexically_relative(root)))
                        throw std::runtime_error("Unmanifested runtime file: " + path_utf8(item.path()));
                }
            }
            if (components != std::set<std::string>{"colmap", "super_resolution"})
                throw std::runtime_error("Manifest must contain both runtimes");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(error(ErrorCode::RuntimeManifestMismatch, e.what()));
        }
    }

    std::expected<void, Error> check_runtime_compatibility(bool require_cuda) {
#ifdef _WIN32
        struct Module {
            HMODULE value;
            ~Module() {
                if (value)
                    FreeLibrary(value);
            }
        };
        OSVERSIONINFOW version{};
        version.dwOSVersionInfoSize = sizeof(version);
        using RtlVersion = LONG(WINAPI*)(OSVERSIONINFOW*);
        auto rtl = reinterpret_cast<RtlVersion>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion"));
        if (sizeof(void*) != 8 || !rtl || rtl(&version) != 0 || version.dwMajorVersion < 10)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "Windows 10/11 x64 required"));
        RuntimeHostInfo host{.x64 = sizeof(void*) == 8, .windows_major = version.dwMajorVersion, .windows_build = version.dwBuildNumber};
        Module vc{LoadLibraryExW(L"vcruntime140.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)};
        if (!vc.value)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "VC 14 x64 runtime missing"));
        std::wstring vc_path(32768, 0);
        vc_path.resize(GetModuleFileNameW(vc.value, vc_path.data(), static_cast<DWORD>(vc_path.size())));
        DWORD unused = 0;
        const auto info_size = GetFileVersionInfoSizeW(vc_path.c_str(), &unused);
        std::vector<unsigned char> version_info(info_size);
        VS_FIXEDFILEINFO* fixed = nullptr;
        UINT fixed_size = 0;
        if (!info_size || !GetFileVersionInfoW(vc_path.c_str(), 0, info_size, version_info.data()) ||
            !VerQueryValueW(version_info.data(), L"\\", reinterpret_cast<void**>(&fixed), &fixed_size) || !fixed ||
            fixed_size < sizeof(VS_FIXEDFILEINFO))
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "Cannot determine VC runtime version"));
        host.vc_major = HIWORD(fixed->dwFileVersionMS);
        host.vc_minor = LOWORD(fixed->dwFileVersionMS);
        if (!require_cuda)
            return validate_runtime_host(host, false);
        Module cuda{LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)};
        if (!cuda.value)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "NVIDIA CUDA driver missing"));
        using Init = int(WINAPI*)(unsigned);
        using Query = int(WINAPI*)(int*);
        using Capability = int(WINAPI*)(int*, int*, int);
        auto init = reinterpret_cast<Init>(GetProcAddress(cuda.value, "cuInit"));
        auto driver = reinterpret_cast<Query>(GetProcAddress(cuda.value, "cuDriverGetVersion"));
        auto count = reinterpret_cast<Query>(GetProcAddress(cuda.value, "cuDeviceGetCount"));
        auto capability = reinterpret_cast<Capability>(GetProcAddress(cuda.value, "cuDeviceComputeCapability"));
        int driver_version = 0, devices = 0;
        if (!init || !driver || !count || !capability || init(0) || driver(&driver_version) || count(&devices) ||
            driver_version < 12080 || devices < 1)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "CUDA 12.8-compatible driver and GPU required"));
        // M0 observed sm_86. Broader support remains a release qualification task.
        int major = 0, minor = 0;
        if (capability(&major, &minor, 0) || major * 10 + minor < 75)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "GPU SM 7.5 or newer required"));
        host.cuda_driver = driver_version;
        host.gpu_sm = major * 10 + minor;
        return validate_runtime_host(host, true);
#else
        return std::unexpected(error(ErrorCode::PlatformUnsupported, "Bundled runtimes require Windows"));
#endif
    }

    std::expected<void, Error> validate_runtime_host(const RuntimeHostInfo& host, bool require_cuda) {
        if (!host.x64 || host.windows_major < 10 || host.windows_build < 19045)
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "Windows 10 22H2/11 x64 required"));
        if (host.vc_major < 14 || (host.vc_major == 14 && host.vc_minor < 44))
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "VC runtime 14.44 or newer required for this frozen payload"));
        if (require_cuda && (host.cuda_driver < 12080 || host.gpu_sm < 75))
            return std::unexpected(error(ErrorCode::IncompatibleRuntime, "CUDA 12.8 driver and GPU SM >= 7.5 required"));
        return {};
    }
} // namespace lfs::preprocess
