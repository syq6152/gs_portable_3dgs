# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc photo imgcodecs videoio)
find_package(Protobuf CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# Headless native decoding is independent of the commented-out GUI/CUDA video
# implementation in src/io. No ffmpeg executable or Python runtime is invoked.
add_library(lfs_video STATIC video_frames.cpp)
target_include_directories(lfs_video PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lfs_video PRIVATE opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

add_library(lfs_preprocessing STATIC
    process_runner.cpp
    providers.cpp
    runtime.cpp
    pipeline.cpp
    colmap_model.cpp
    colmap_database.cpp
    image_pipeline.cpp
    workspace.cpp
    dataset_validation.cpp
    scan_pipeline.cpp
    registered_pipeline.cpp
    registered_twice.cpp
    mesh_geometry.cpp
    mesh_export.cpp
    mesh_export_pipeline.cpp
    debug_exports.cpp
    blur_filter.cpp
    registration_triplets.cpp
    foreground_mask.cpp
    ../io/formats/scanner_bin.cpp
)

target_include_directories(lfs_preprocessing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(lfs_preprocessing PRIVATE .. ../core/include ../io/include)
target_link_libraries(lfs_preprocessing PRIVATE nlohmann_json::nlohmann_json)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
target_link_libraries(lfs_preprocessing PRIVATE unofficial::sqlite3::sqlite3)
find_package(OpenImageIO CONFIG REQUIRED)
target_link_libraries(lfs_preprocessing PRIVATE OpenImageIO::OpenImageIO opencv_core opencv_imgproc opencv_photo opencv_imgcodecs
    lfs_video onnxruntime::onnxruntime)
find_package(assimp CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
target_include_directories(lfs_preprocessing PRIVATE ../../external)
target_link_libraries(lfs_preprocessing PRIVATE assimp::assimp CGAL::CGAL TBB::tbb)
if(WIN32)
    target_link_libraries(lfs_preprocessing PRIVATE bcrypt version)
endif()
if(MSVC)
    target_compile_options(lfs_preprocessing PRIVATE /utf-8 /W4)
    target_compile_options(lfs_video PRIVATE /utf-8 /W4)
endif()

set_target_properties(lfs_preprocessing lfs_video PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
