# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
if(CMAKE_INSTALL_COMPONENT)
    return()
endif()
file(GLOB_RECURSE _lfs_files LIST_DIRECTORIES FALSE RELATIVE "${PACKAGE_ROOT}" "${PACKAGE_ROOT}/*")
list(SORT _lfs_files)
set(_lfs_inventory "[]")
set(_lfs_index 0)
foreach(_lfs_relative IN LISTS _lfs_files)
    string(TOLOWER "${_lfs_relative}" _lfs_lower)
    if((_lfs_lower MATCHES "[.]json$" AND NOT _lfs_lower STREQUAL "manifest.json")
       OR _lfs_lower MATCHES "(^|/)(licenses?|notices?|copying|copyright)([./_-]|$)"
       OR _lfs_lower MATCHES "(^|/)(readme|changelog|authors?|contributors?)([./_-]|$)"
       OR _lfs_lower MATCHES "[.](md|txt|rst|html?|pdf|xml|ya?ml)$"
       OR _lfs_lower MATCHES "^share/")
        message(FATAL_ERROR "Non-runtime package file: ${_lfs_relative}; use a fresh prefix")
    endif()
    if((_lfs_lower MATCHES "(^|/)(swaptexture[.]exe|_internal|swaptexture_python|scripts|tests|[.]git)(/|$)" AND NOT _lfs_lower STREQUAL "swaptexture.exe")
       OR _lfs_lower MATCHES "(^|/)gs/bin/"
       OR _lfs_lower MATCHES "(^|/)swaptexture_.*[.]py$")
        message(FATAL_ERROR "Forbidden legacy/test payload: ${_lfs_relative}; use a fresh prefix")
    endif()
    if(_lfs_lower MATCHES "[.]exe$" AND NOT _lfs_lower MATCHES
       "^(swaptexture[.]exe|bin/third_party/colmap-cuda-cli/bin/sfm[.]exe|bin/third_party/superresolution/preprocess[.]exe)$")
        message(FATAL_ERROR "Undeclared executable: ${_lfs_relative}")
    endif()
    if(IS_SYMLINK "${PACKAGE_ROOT}/${_lfs_relative}")
        message(FATAL_ERROR "External package payload: ${_lfs_relative}")
    endif()
    file(SIZE "${PACKAGE_ROOT}/${_lfs_relative}" _lfs_size)
    file(SHA256 "${PACKAGE_ROOT}/${_lfs_relative}" _lfs_hash)
    string(JSON _lfs_inventory SET "${_lfs_inventory}" ${_lfs_index}
        "{\"path\":\"${_lfs_relative}\",\"byte_size\":${_lfs_size},\"sha256\":\"${_lfs_hash}\"}")
    math(EXPR _lfs_index "${_lfs_index}+1")
endforeach()
foreach(_lfs_config GS_params_scan_fast.bin GS_params_scan_medium.bin GS_params_scan_quality.bin
        GS_params_inc_fast.bin GS_params_inc_medium.bin GS_params_inc_quality.bin Swaptexture_params.bin)
    if(NOT EXISTS "${PACKAGE_ROOT}/bin/${_lfs_config}")
        message(FATAL_ERROR "Missing encrypted configuration: ${_lfs_config}")
    endif()
endforeach()
if(NOT EXISTS "${PACKAGE_ROOT}/SwapTexture.exe" OR NOT EXISTS "${PACKAGE_ROOT}/manifest.json")
    message(FATAL_ERROR "Missing root SwapTexture.exe or product manifest.json")
endif()
file(READ "${CMAKE_CURRENT_LIST_DIR}/../resources/swaptexture_configs/manifest.json" _lfs_expected_manifest)
file(READ "${PACKAGE_ROOT}/manifest.json" _lfs_product_manifest)
string(JSON _lfs_manifest_matches EQUAL "${_lfs_expected_manifest}" "${_lfs_product_manifest}")
if(NOT _lfs_manifest_matches)
    message(FATAL_ERROR "Product manifest differs from canonical source")
endif()
file(MAKE_DIRECTORY "${PACKAGE_AUDIT_DIR}")
file(WRITE "${PACKAGE_AUDIT_DIR}/package-manifest.json"
    "{\"schema_version\":1,\"product\":\"LichtFeld native reconstruction\",\"files\":${_lfs_inventory}}\n")
