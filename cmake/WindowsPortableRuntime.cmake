# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)

function(lfs_install_windows_runtime target)
    # The executable is the authoritative import root. The preprocessing
    # component is already copied beside it at build time; using its source
    # prefix here makes GET_RUNTIME_DEPENDENCIES report duplicate SDK ABIs.
    set(_roots)
    get_property(_declared GLOBAL PROPERTY LFS_PREPROCESSING_RUNTIME_FILES)
    foreach(_module nvimgcodec nvjpeg_ext nvjpeg2k_ext)
        if(TARGET ${_module})
            list(APPEND _roots "$<TARGET_FILE:${_module}>")
        endif()
    endforeach()
    set(_vcpkg_runtime_dir "")
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        set(_vcpkg_runtime_dir "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
    endif()
    # Discover linked Python through PE imports; never copy an SDK environment.
    install(CODE "
        set(_lfs_executable \"$<TARGET_FILE:${target}>\")
        set(_lfs_libraries \"${_roots}\")
        set(_lfs_declared_runtime \"${_declared}\")
        set(_lfs_search_dirs \"$<TARGET_FILE_DIR:${target}>;${_vcpkg_runtime_dir};${CUDAToolkit_BIN_DIR};${CUDAToolkit_BIN_DIR}/x64\")
        set(_lfs_bindir \"${CMAKE_INSTALL_BINDIR}\")
        set(_lfs_audit_dir \"${CMAKE_BINARY_DIR}/package-audit\")
        include(\"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/BundleWindowsRuntime.cmake\")
    " COMPONENT runtime)
    # Component installs deliberately are not complete application packages.
    install(CODE "
        set(PACKAGE_ROOT \"\${CMAKE_INSTALL_PREFIX}\")
        set(PACKAGE_AUDIT_DIR \"${CMAKE_BINARY_DIR}/package-audit\")
        include(\"${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FinalizeNativePackage.cmake\")
    " ALL_COMPONENTS)
endfunction()
