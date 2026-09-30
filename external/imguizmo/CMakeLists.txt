# ImGuizmo - 3D manipulation gizmo library
cmake_minimum_required(VERSION 3.18)

# Create ImGuizmo library
add_library(imguizmo STATIC
    ImGuizmo.cpp
    ImGuizmo.h
)

# ImGuizmo needs ImGui headers
target_link_libraries(imguizmo
    PUBLIC
        imgui::imgui
)

# Set include directories
target_include_directories(imguizmo
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set C++ standard
target_compile_features(imguizmo PUBLIC cxx_std_17)

# Create alias for consistent naming
add_library(imguizmo::imguizmo ALIAS imguizmo)
