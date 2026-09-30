# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)
include(GNUInstallDirs)
get_filename_component(_runtime_default "${CMAKE_CURRENT_LIST_DIR}/../third_party/runtime" ABSOLUTE)
if((DEFINED LFS_COLMAP_RUNTIME_SOURCE_DIR OR DEFINED LFS_SUPERRES_RUNTIME_SOURCE_DIR)
   AND NOT DEFINED LFS_THIRD_PARTY_RUNTIME_MANIFEST)
    message(FATAL_ERROR "Runtime overrides require LFS_THIRD_PARTY_RUNTIME_MANIFEST")
endif()
set(LFS_COLMAP_RUNTIME_SOURCE_DIR "${_runtime_default}/colmap-cuda-cli" CACHE PATH "COLMAP payload")
set(LFS_SUPERRES_RUNTIME_SOURCE_DIR "${_runtime_default}/SuperResolution" CACHE PATH "SR payload")
set(LFS_THIRD_PARTY_RUNTIME_MANIFEST "${_runtime_default}/manifest.json" CACHE FILEPATH "Matching runtime manifest")
set(LFS_THIRD_PARTY_NOTICES_DIR "${_runtime_default}/licenses" CACHE PATH "Runtime notices")
set(LFS_RUNTIME_VERIFY_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/VerifyThirdPartyRuntime.cmake")
function(lfs_verify_runtime colmap superres)
    set(NOTICES_ROOT "${LFS_THIRD_PARTY_NOTICES_DIR}")
    set(MANIFEST "${LFS_THIRD_PARTY_RUNTIME_MANIFEST}")
    set(COLMAP_ROOT "${colmap}")
    set(SUPERRES_ROOT "${superres}")
    include("${LFS_RUNTIME_VERIFY_SCRIPT}")
endfunction()
lfs_verify_runtime("${LFS_COLMAP_RUNTIME_SOURCE_DIR}" "${LFS_SUPERRES_RUNTIME_SOURCE_DIR}")
function(lfs_attach_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" "-DMANIFEST=${LFS_THIRD_PARTY_RUNTIME_MANIFEST}"
            "-DNOTICES_ROOT=${LFS_THIRD_PARTY_NOTICES_DIR}"
            "-DCOLMAP_ROOT=${LFS_COLMAP_RUNTIME_SOURCE_DIR}" "-DSUPERRES_ROOT=${LFS_SUPERRES_RUNTIME_SOURCE_DIR}" -P "${LFS_RUNTIME_VERIFY_SCRIPT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:${target}>/third_party"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${LFS_COLMAP_RUNTIME_SOURCE_DIR}" "$<TARGET_FILE_DIR:${target}>/third_party/colmap-cuda-cli"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${LFS_SUPERRES_RUNTIME_SOURCE_DIR}" "$<TARGET_FILE_DIR:${target}>/third_party/SuperResolution"
        VERBATIM)
    add_custom_target(verify_third_party_runtime
        COMMAND "${CMAKE_COMMAND}" "-DMANIFEST=${LFS_THIRD_PARTY_RUNTIME_MANIFEST}"
            "-DNOTICES_ROOT=${LFS_THIRD_PARTY_NOTICES_DIR}"
            "-DCOLMAP_ROOT=$<TARGET_FILE_DIR:${target}>/third_party/colmap-cuda-cli"
            "-DSUPERRES_ROOT=$<TARGET_FILE_DIR:${target}>/third_party/SuperResolution" -P "${LFS_RUNTIME_VERIFY_SCRIPT}"
        DEPENDS ${target} VERBATIM)
endfunction()
install(CODE "
    set(NOTICES_ROOT \"${LFS_THIRD_PARTY_NOTICES_DIR}\")
    set(MANIFEST \"${LFS_THIRD_PARTY_RUNTIME_MANIFEST}\")
    set(COLMAP_ROOT \"${LFS_COLMAP_RUNTIME_SOURCE_DIR}\")
    set(SUPERRES_ROOT \"${LFS_SUPERRES_RUNTIME_SOURCE_DIR}\")
    include(\"${LFS_RUNTIME_VERIFY_SCRIPT}\")
" COMPONENT ThirdPartyRuntime)
install(DIRECTORY "${LFS_COLMAP_RUNTIME_SOURCE_DIR}/" DESTINATION "${CMAKE_INSTALL_BINDIR}/third_party/colmap-cuda-cli" COMPONENT ThirdPartyRuntime)
install(DIRECTORY "${LFS_SUPERRES_RUNTIME_SOURCE_DIR}/" DESTINATION "${CMAKE_INSTALL_BINDIR}/third_party/SuperResolution" COMPONENT ThirdPartyRuntime)
