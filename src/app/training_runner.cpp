/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/training_runner.hpp"
#include "control/command_api.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"

#include "rendering/mesh2splat.hpp"
// clang-format off
#include <glad/glad.h>
// clang-format on
#include <SDL3/SDL.h>

#include <atomic>
#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <expected>
#include <format>
#include <mutex>
#include <rasterization_api.h>
#include <stb_image.h>
#include <string_view>

#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::app {

    namespace {

        std::atomic<bool> g_headless_interrupt_requested{false};
        std::atomic<lfs::training::Trainer*> g_interruptible_trainer{nullptr};

        struct LegacyConstraintMeshContext {
            core::Tensor vertices;
            core::Tensor indices;
            core::Tensor edge_neighbors;
            float mesh2splat_mean_max_scale = 0.0f;
        };

        std::expected<LegacyConstraintMeshContext, std::string> capture_legacy_constraint_mesh_context(
            const core::SplatData& model) {
            const auto& vertices = model.constraint_mesh_verts();
            const auto& indices = model.constraint_mesh_indices();
            const auto& edge_neighbors = model.tri_edge_neighbors();
            if (!vertices.is_valid() || vertices.ndim() != 2 || vertices.shape()[1] != 3 ||
                vertices.dtype() != core::DataType::Float32 ||
                !indices.is_valid() || indices.ndim() != 2 || indices.shape()[1] != 3 ||
                indices.dtype() != core::DataType::Int32 ||
                !edge_neighbors.is_valid() || edge_neighbors.ndim() != 2 ||
                edge_neighbors.shape()[1] != 3 || edge_neighbors.dtype() != core::DataType::Int32 ||
                edge_neighbors.shape()[0] != indices.shape()[0]) {
                return std::unexpected(
                    "fresh mesh-init model has an incompatible static constraint-mesh schema");
            }

            const size_t vertex_count = vertices.shape()[0];
            const size_t face_count = indices.shape()[0];
            for (const int vertex_id : indices.cpu().contiguous().to_vector_int()) {
                if (vertex_id < 0 || static_cast<size_t>(vertex_id) >= vertex_count) {
                    return std::unexpected(
                        "fresh mesh-init constraint mesh contains an out-of-range vertex index");
                }
            }
            for (const int neighbor_id : edge_neighbors.cpu().contiguous().to_vector_int()) {
                if (neighbor_id < -1 ||
                    (neighbor_id >= 0 && static_cast<size_t>(neighbor_id) >= face_count)) {
                    return std::unexpected(
                        "fresh mesh-init constraint adjacency contains an out-of-range face index");
                }
            }

            return LegacyConstraintMeshContext{
                .vertices = vertices,
                .indices = indices,
                .edge_neighbors = edge_neighbors,
                .mesh2splat_mean_max_scale = model.get_mesh2splat_mean_max_scale()};
        }

        std::expected<void, std::string> graft_legacy_constraint_mesh_context(
            core::SplatData& checkpoint_model,
            LegacyConstraintMeshContext context) {
            if (checkpoint_model.constraint_mesh_verts().is_valid() ||
                checkpoint_model.constraint_mesh_indices().is_valid() ||
                checkpoint_model.tri_edge_neighbors().is_valid() ||
                checkpoint_model.birth_tri().is_valid()) {
                return std::unexpected(
                    "checkpoint model contains a partial constraint-mesh context");
            }

            auto birth_placeholder = core::Tensor::full(
                {static_cast<size_t>(checkpoint_model.size())},
                -1.0f,
                checkpoint_model.means().device(),
                core::DataType::Int32);

            checkpoint_model.constraint_mesh_verts() = std::move(context.vertices);
            checkpoint_model.constraint_mesh_indices() = std::move(context.indices);
            checkpoint_model.tri_edge_neighbors() = std::move(context.edge_neighbors);
            checkpoint_model.birth_tri() = std::move(birth_placeholder);
            checkpoint_model.set_mesh2splat_mean_max_scale(context.mesh2splat_mean_max_scale);
            return {};
        }

        void request_headless_interrupt() {
            g_headless_interrupt_requested.store(true, std::memory_order_release);
            if (auto* trainer = g_interruptible_trainer.load(std::memory_order_acquire)) {
                trainer->request_stop();
            }
        }

        void headless_signal_handler(int signal) {
            if (signal == SIGINT || signal == SIGTERM) {
                request_headless_interrupt();
            }
        }

#ifdef WIN32
        BOOL WINAPI headless_console_ctrl_handler(DWORD ctrl_type) {
            switch (ctrl_type) {
            case CTRL_C_EVENT:
            case CTRL_BREAK_EVENT:
            case CTRL_CLOSE_EVENT:
            case CTRL_SHUTDOWN_EVENT:
                request_headless_interrupt();
                return TRUE;
            default:
                return FALSE;
            }
        }
#endif

        class ScopedHeadlessInterruptHandler {
        public:
            ScopedHeadlessInterruptHandler()
                : previous_sigint_(std::signal(SIGINT, headless_signal_handler)),
                  previous_sigterm_(std::signal(SIGTERM, headless_signal_handler)) {
                g_headless_interrupt_requested.store(false, std::memory_order_release);
#ifdef WIN32
                // Some Windows libraries flip the process into "ignore Ctrl+C" mode.
                // Clear that and install an explicit console handler for terminal runs.
                SetConsoleCtrlHandler(nullptr, FALSE);
                console_handler_installed_ = SetConsoleCtrlHandler(headless_console_ctrl_handler, TRUE) != FALSE;
#endif
            }

            ~ScopedHeadlessInterruptHandler() {
                clear_trainer();
                if (previous_sigint_ != SIG_ERR) {
                    std::signal(SIGINT, previous_sigint_);
                }
                if (previous_sigterm_ != SIG_ERR) {
                    std::signal(SIGTERM, previous_sigterm_);
                }
#ifdef WIN32
                if (console_handler_installed_) {
                    SetConsoleCtrlHandler(headless_console_ctrl_handler, FALSE);
                }
#endif
            }

            void set_trainer(lfs::training::Trainer& trainer) {
                g_interruptible_trainer.store(&trainer, std::memory_order_release);
                if (interrupted()) {
                    trainer.request_stop();
                }
            }

            void clear_trainer() {
                g_interruptible_trainer.store(nullptr, std::memory_order_release);
            }

            [[nodiscard]] bool interrupted() const {
                return g_headless_interrupt_requested.load(std::memory_order_acquire);
            }

        private:
            using SignalHandler = void (*)(int);

            SignalHandler previous_sigint_ = SIG_ERR;
            SignalHandler previous_sigterm_ = SIG_ERR;
#ifdef WIN32
            bool console_handler_installed_ = false;
#endif
        };

        bool checkCudaDriverVersion();

        bool isMesh2SplatStrategy(const std::string_view strategy) {
            return strategy == "mesh2splat";
        }

        class ScopedHeadlessGlContext {
        public:
            ~ScopedHeadlessGlContext() {
                if (gl_context_) {
                    SDL_GL_DestroyContext(gl_context_);
                    gl_context_ = nullptr;
                }
                if (window_) {
                    SDL_DestroyWindow(window_);
                    window_ = nullptr;
                }
                if (sdl_initialized_) {
                    SDL_Quit();
                    sdl_initialized_ = false;
                }
            }

            std::expected<void, std::string> initialize() {
                if (!SDL_Init(SDL_INIT_VIDEO)) {
                    return std::unexpected(std::format("SDL_Init failed: {}", SDL_GetError()));
                }
                sdl_initialized_ = true;

                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

                window_ = SDL_CreateWindow(
                    "LichtFeld Mesh2Splat",
                    1,
                    1,
                    SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
                if (!window_) {
                    return std::unexpected(std::format("SDL_CreateWindow failed: {}", SDL_GetError()));
                }

                gl_context_ = SDL_GL_CreateContext(window_);
                if (!gl_context_) {
                    return std::unexpected(std::format("SDL_GL_CreateContext failed: {}", SDL_GetError()));
                }

                SDL_GL_MakeCurrent(window_, gl_context_);
                if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
                    return std::unexpected("Failed to initialize GL loader (glad)");
                }

                return {};
            }

        private:
            bool sdl_initialized_ = false;
            SDL_Window* window_ = nullptr;
            SDL_GLContext gl_context_ = nullptr;
        };

        std::filesystem::path resolveMesh2SplatOutputPath(
            const lfs::core::param::TrainingParameters& params,
            const std::filesystem::path& mesh_path) {

            const std::filesystem::path default_filename =
                mesh_path.stem().string() + "_mesh2splat.ply";

            if (params.dataset.output_path.empty()) {
                return mesh_path.parent_path() / default_filename;
            }

            auto output_path = params.dataset.output_path;
            std::error_code ec;
            const bool path_exists = std::filesystem::exists(output_path, ec);
            const bool is_directory = path_exists && std::filesystem::is_directory(output_path, ec);

            if (is_directory || output_path.extension().empty()) {
                return output_path / default_filename;
            }

            if (output_path.extension() != ".ply") {
                LOG_WARN("mesh2splat output extension '{}' is not .ply; forcing .ply", output_path.extension().string());
                output_path.replace_extension(".ply");
            }
            return output_path;
        }

        std::expected<void, std::string> applyExternalTextureToMesh(
            lfs::core::MeshData& mesh,
            const std::filesystem::path& texture_path) {

            if (!mesh.has_texcoords()) {
                return std::unexpected(std::format(
                    "Mesh has no UV coordinates, cannot apply texture '{}'",
                    lfs::core::path_to_utf8(texture_path)));
            }

            auto texture_path_utf8 = lfs::core::path_to_utf8(texture_path);
            lfs::core::TextureImage texture_image;
            uint8_t* pixels = stbi_load(texture_path_utf8.c_str(),
                                        &texture_image.width,
                                        &texture_image.height,
                                        nullptr,
                                        4);
            if (!pixels || texture_image.width <= 0 || texture_image.height <= 0) {
                if (pixels) {
                    stbi_image_free(pixels);
                }
                return std::unexpected(std::format(
                    "Failed to load texture image '{}'",
                    texture_path_utf8));
            }

            texture_image.channels = 4;
            const size_t byte_count = static_cast<size_t>(texture_image.width) *
                                      static_cast<size_t>(texture_image.height) *
                                      static_cast<size_t>(texture_image.channels);
            texture_image.pixels.resize(byte_count);
            std::memcpy(texture_image.pixels.data(), pixels, byte_count);
            stbi_image_free(pixels);

            const int texture_width = texture_image.width;
            const int texture_height = texture_image.height;
            const int texture_channels = texture_image.channels;
            mesh.texture_images.clear();
            mesh.texture_images.push_back(std::move(texture_image));

            if (mesh.materials.empty()) {
                mesh.materials.emplace_back();
            }

            for (auto& mat : mesh.materials) {
                mat.albedo_tex = 1;
                mat.albedo_tex_path = texture_path_utf8;
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

            LOG_INFO("Applied external texture '{}' to mesh ({}x{}, {} channels)",
                     texture_path_utf8,
                     texture_width,
                     texture_height,
                     texture_channels);
            return {};
        }

        std::expected<void, std::string> runHeadlessMesh2Splat(
            const lfs::core::param::TrainingParameters& params) {

            if (params.resume_checkpoint.has_value()) {
                return std::unexpected("mesh2splat strategy does not support --resume");
            }

            if (!params.mesh2splat_mesh_path.has_value() || params.mesh2splat_mesh_path->empty() ||
                !params.mesh2splat_texture_path.has_value() || params.mesh2splat_texture_path->empty()) {
                return std::unexpected("mesh2splat strategy requires --mesh2splat-dir <directory_with_obj_mtl_png>");
            }

            const auto mesh_path = lfs::core::utf8_to_path(*params.mesh2splat_mesh_path);
            if (!std::filesystem::exists(mesh_path)) {
                return std::unexpected(std::format("Mesh file does not exist: {}", lfs::core::path_to_utf8(mesh_path)));
            }

            const auto texture_path = lfs::core::utf8_to_path(*params.mesh2splat_texture_path);
            if (!std::filesystem::exists(texture_path)) {
                return std::unexpected(std::format("Texture file does not exist: {}", lfs::core::path_to_utf8(texture_path)));
            }

            auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(mesh_path);
            if (!load_result) {
                return std::unexpected(std::format(
                    "Failed to load mesh '{}': {}",
                    lfs::core::path_to_utf8(mesh_path),
                    load_result.error().format()));
            }

            auto* mesh_ptr = std::get_if<std::shared_ptr<lfs::core::MeshData>>(&load_result->data);
            if (!mesh_ptr || !(*mesh_ptr)) {
                return std::unexpected(std::format(
                    "Input '{}' did not load as mesh data",
                    lfs::core::path_to_utf8(mesh_path)));
            }

            if (const auto tex_result = applyExternalTextureToMesh(**mesh_ptr, texture_path); !tex_result) {
                return std::unexpected(tex_result.error());
            }

            const auto output_path = resolveMesh2SplatOutputPath(params, mesh_path);
            if (const auto parent = output_path.parent_path(); !parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        lfs::core::path_to_utf8(parent),
                        ec.message()));
                }
            }

            ScopedHeadlessGlContext gl_context;
            if (const auto gl_result = gl_context.initialize(); !gl_result) {
                return std::unexpected(gl_result.error());
            }

            LOG_INFO("Running mesh2splat conversion: {}", lfs::core::path_to_utf8(mesh_path));

            auto splat_result = lfs::rendering::mesh_to_splat(**mesh_ptr);
            (*mesh_ptr)->texture_images.clear();
            (*mesh_ptr)->texture_images.shrink_to_fit();
            if (!splat_result) {
                return std::unexpected(std::format("mesh2splat conversion failed: {}", splat_result.error()));
            }

            auto splat_data = std::move(*splat_result);
            auto save_result = lfs::io::save_ply(
                *splat_data,
                {.output_path = output_path, .binary = true});
            if (!save_result) {
                return std::unexpected(std::format(
                    "Failed to save splat '{}': {}",
                    lfs::core::path_to_utf8(output_path),
                    save_result.error().format()));
            }

            LOG_INFO("mesh2splat output saved to {}", lfs::core::path_to_utf8(output_path));
            return {};
        }

        int run_training_impl(std::unique_ptr<lfs::core::param::TrainingParameters> params,
                              std::stop_token stop, const TrainingCallbacks& callbacks, TrainingRunResult& summary) {
            ScopedHeadlessInterruptHandler interrupt_handler;
            std::stop_callback cancellation(stop, [] { g_headless_interrupt_requested.store(true, std::memory_order_release); });
            if (stop.stop_requested())
                return 130;

            if (callbacks.on_progress)
                callbacks.on_progress(0.0F);

            if (isMesh2SplatStrategy(params->optimization.strategy)) {
                if (!checkCudaDriverVersion())
                    return 1;

                if (const auto result = runHeadlessMesh2Splat(*params); !result) {
                    LOG_ERROR("{}", result.error());
                    return 1;
                }

                LOG_INFO("Headless mesh2splat conversion completed");

                return 0;
            }

            if (params->dataset.data_path.empty() && !params->resume_checkpoint) {
                LOG_ERROR("Headless mode requires --data-path or --resume");
                return 1;
            }

            if (!checkCudaDriverVersion())
                return 1;
            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

            {
                core::Scene scene;

                if (params->resume_checkpoint) {
                    LOG_INFO("Resuming from checkpoint: {}", core::path_to_utf8(*params->resume_checkpoint));

                    auto params_result = core::load_checkpoint_params(*params->resume_checkpoint);
                    if (!params_result) {
                        LOG_ERROR("Failed to load checkpoint params: {}", params_result.error());
                        return 1;
                    }
                    auto checkpoint_params = std::move(*params_result);

                    if (!params->dataset.data_path.empty())
                        checkpoint_params.dataset.data_path = params->dataset.data_path;
                    if (!params->dataset.output_path.empty())
                        checkpoint_params.dataset.output_path = params->dataset.output_path;
                    if (params->use_mesh_init) {
                        checkpoint_params.use_mesh_init = true;
                        checkpoint_params.optimization.mesh2splat_hole_fill_mode =
                            params->optimization.mesh2splat_hole_fill_mode;
                        checkpoint_params.optimization.mesh2splat_hole_fill_enabled =
                            params->optimization.mesh2splat_hole_fill_enabled;
                        checkpoint_params.mesh_init_mesh_path = params->mesh_init_mesh_path;
                        checkpoint_params.mesh_init_texture_path = params->mesh_init_texture_path;
                        checkpoint_params.mesh_init_external_filled_mesh_path = params->mesh_init_external_filled_mesh_path;
                        checkpoint_params.mesh_init_external_pointcloud_path = params->mesh_init_external_pointcloud_path;
                        checkpoint_params.mesh_init_sampling_rate = params->mesh_init_sampling_rate;
                        checkpoint_params.init_white_gs_color = params->init_white_gs_color;
                    }

                    if (checkpoint_params.dataset.data_path.empty()) {
                        LOG_ERROR("Checkpoint has no dataset path and none provided via --data-path");
                        return 1;
                    }
                    if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                        LOG_ERROR("Dataset path does not exist: {}", core::path_to_utf8(checkpoint_params.dataset.data_path));
                        return 1;
                    }

                    if (const auto result = training::validateDatasetPath(checkpoint_params); !result) {
                        LOG_ERROR("Dataset validation failed: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    // Mesh-init reconstructs a fresh training Splat so the
                    // original mesh and project-mask source exist. The checkpoint
                    // model replaces only that transient Splat, not the mesh.
                    std::optional<LegacyConstraintMeshContext> legacy_constraint_context;
                    const std::string fresh_training_model = scene.getTrainingModelNodeName();
                    if (!fresh_training_model.empty()) {
                        const auto* node = scene.getNode(fresh_training_model);
                        if (node && node->type == core::NodeType::SPLAT) {
                            if (node->model && node->model->has_constraint_mesh() &&
                                node->model->has_tri_edge_neighbors()) {
                                auto context_result = capture_legacy_constraint_mesh_context(*node->model);
                                if (!context_result) {
                                    LOG_ERROR("Cannot preserve legacy checkpoint constraint context: {}",
                                              context_result.error());
                                    return 1;
                                }
                                legacy_constraint_context = std::move(*context_result);
                            } else if (checkpoint_params.use_mesh_init) {
                                LOG_ERROR("Fresh mesh-init model has no compatible constraint context");
                                return 1;
                            }
                            scene.removeNode(fresh_training_model, false);
                        }
                    } else if (checkpoint_params.use_mesh_init) {
                        LOG_ERROR("Mesh-init checkpoint resume did not create a fresh training model");
                        return 1;
                    }

                    for (const auto* node : scene.getNodes()) {
                        if (node->type == core::NodeType::POINTCLOUD) {
                            scene.removeNode(node->name, false);
                            break;
                        }
                    }

                    auto splat_result = core::load_checkpoint_splat_data(*params->resume_checkpoint);
                    if (!splat_result) {
                        LOG_ERROR("Failed to load checkpoint splat data: {}", splat_result.error());
                        return 1;
                    }
                    if (!splat_result->has_constraint_mesh() && legacy_constraint_context) {
                        if (auto graft_result = graft_legacy_constraint_mesh_context(
                                *splat_result, std::move(*legacy_constraint_context));
                            !graft_result) {
                            LOG_ERROR("Failed to graft legacy checkpoint constraint context: {}",
                                      graft_result.error());
                            return 1;
                        }
                        LOG_INFO("Restored legacy checkpoint constraint context from fresh mesh-init model");
                    }

                    auto splat_data = std::make_unique<core::SplatData>(std::move(*splat_result));
                    scene.addSplat("Model", std::move(splat_data), core::NULL_NODE);
                    scene.setTrainingModelNode("Model");

                    checkpoint_params.resume_checkpoint = *params->resume_checkpoint;

                    auto trainer = std::make_unique<training::Trainer>(scene);
                    interrupt_handler.set_trainer(*trainer);
                    struct Registration {
                        ScopedHeadlessInterruptHandler& handler;
                        ~Registration() { handler.clear_trainer(); }
                    } registration{interrupt_handler};
                    std::stop_callback trainer_stop(stop, [&] { trainer->request_stop(); });

                    // if (!params->python_scripts.empty()) {
                    //     trainer->set_python_scripts(params->python_scripts);
                    //     vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    // }

                    if (callbacks.on_parameters)
                        callbacks.on_parameters(checkpoint_params);
                    summary.gaussian_ply = checkpoint_params.dataset.output_path / "Gaussian.ply";
                    if (const auto result = trainer->initialize(checkpoint_params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    const auto ckpt_result = trainer->load_checkpoint(*params->resume_checkpoint);
                    if (!ckpt_result) {
                        LOG_ERROR("Failed to restore checkpoint state: {}", ckpt_result.error());
                        return 1;
                    }
                    LOG_INFO("Resumed from iteration {}", *ckpt_result);

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(); !result) {
                        interrupt_handler.clear_trainer();
                        LOG_ERROR("Training error: {}", result.error());
                        // if (!params->python_scripts.empty()) {
                        //     core::Tensor::shutdown_memory_pool();
                        //     core::PinnedMemoryAllocator::instance().shutdown();
                        //     python::finalize();
                        //     std::_Exit(1);
                        // }
                        return 1;
                    }
                    summary.final_iteration = trainer->get_current_iteration();
                    interrupt_handler.clear_trainer();
                    if (interrupt_handler.interrupted()) {
                        LOG_INFO("Headless training interrupted by terminal signal");
                        return 130;
                    }
                } else {
                    LOG_INFO("Starting headless training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);
                    interrupt_handler.set_trainer(*trainer);
                    struct Registration {
                        ScopedHeadlessInterruptHandler& handler;
                        ~Registration() { handler.clear_trainer(); }
                    } registration{interrupt_handler};
                    std::stop_callback trainer_stop(stop, [&] { trainer->request_stop(); });

                    // if (!params->python_scripts.empty()) {
                    //     trainer->set_python_scripts(params->python_scripts);
                    //     vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    // }

                    if (callbacks.on_parameters)
                        callbacks.on_parameters(*params);
                    summary.gaussian_ply = params->dataset.output_path / "Gaussian.ply";
                    // 初始化，读数据
                    if (const auto result = trainer->initialize(*params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(); !result) {
                        interrupt_handler.clear_trainer();
                        LOG_ERROR("Training error: {}", result.error());
                        // if (!params->python_scripts.empty()) {
                        //     core::Tensor::shutdown_memory_pool();
                        //     core::PinnedMemoryAllocator::instance().shutdown();
                        //     python::finalize();
                        //     std::_Exit(1);
                        // }
                        return 1;
                    }
                    summary.final_iteration = trainer->get_current_iteration();
                    interrupt_handler.clear_trainer();
                    if (interrupt_handler.interrupted()) {
                        LOG_INFO("Headless training interrupted by terminal signal");
                        return 130;
                    }
                }

                LOG_INFO("Headless training completed");
            }

            if (callbacks.on_progress)
                callbacks.on_progress(1.0F);

            // if (!params->python_scripts.empty()) {
            //     python::finalize();
            //     std::_Exit(0);
            // }
            return 0;
        }

        bool checkCudaDriverVersion() {
            const auto info = lfs::core::check_cuda_version();
            if (info.query_failed) {
                LOG_WARN("Failed to query CUDA driver version");
                return true;
            }

            LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
            if (!info.supported) {
                LOG_WARN("CUDA {}.{} unsupported. Requires 12.8+ (driver 570+)", info.major, info.minor);
                return false;
            }
            return true;
        }

    } // namespace

    std::expected<TrainingRunResult, TrainingError> run_training(
        core::param::TrainingParameters params, std::stop_token stop, TrainingCallbacks callbacks) {
        // The process has global cache/pool and terminal handlers; serialize runner sessions.
        static std::mutex training_mutex;
        std::unique_lock lock(training_mutex, std::try_to_lock);
        if (!lock.owns_lock())
            return std::unexpected(TrainingError{"Another training session is active"});
        struct Cleanup {
            ~Cleanup() {
                g_interruptible_trainer.store(nullptr, std::memory_order_release);
                core::Tensor::shutdown_memory_pool();
                core::PinnedMemoryAllocator::instance().shutdown();
                lfs::event::CommandCenterBridge::instance().set(nullptr);
            }
        } cleanup;
        TrainingRunResult summary;
        try {
            const auto total_iterations = params.optimization.iterations;
            lfs::event::ScopedHandler progress_events;
            progress_events.subscribe<core::events::state::TrainingProgress>([&callbacks, total_iterations](const auto& event) {
                if (callbacks.on_progress && total_iterations > 0)
                    callbacks.on_progress(std::clamp(static_cast<float>(event.iteration) / static_cast<float>(total_iterations), 0.0F, 1.0F));
                if (callbacks.on_iteration)
                    callbacks.on_iteration(event.iteration, event.loss, event.num_gaussians);
            });
            if (stop.stop_requested())
                return TrainingRunResult{.cancelled = true};
            lfs::io::CacheLoader::getInstance(params.dataset.loading_params.use_cpu_memory, params.dataset.loading_params.use_fs_cache);
            lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(
                    p.path, {.resize_factor = p.resize_factor, .max_width = p.max_width, .cuda_stream = p.stream});
            });
            const bool conversion = isMesh2SplatStrategy(params.optimization.strategy);
            const int code = run_training_impl(std::make_unique<core::param::TrainingParameters>(std::move(params)), stop, callbacks, summary);
            if (code == 130) {
                summary.cancelled = true;
                return summary;
            }
            if (code != 0)
                return std::unexpected(TrainingError{"Training failed; see logged stage diagnostics", code});
            if (!conversion && (!std::filesystem::is_regular_file(summary.gaussian_ply) || std::filesystem::file_size(summary.gaussian_ply) == 0))
                return std::unexpected(TrainingError{"Training completed without a nonempty Gaussian.ply"});
            return summary;
        } catch (const std::exception& e) {
            return std::unexpected(TrainingError{e.what()});
        } catch (...) {
            return std::unexpected(TrainingError{"Training/callback raised an unknown exception"});
        }
    }
} // namespace lfs::app
