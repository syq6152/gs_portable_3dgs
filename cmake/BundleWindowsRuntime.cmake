# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_policy(PUSH)
cmake_policy(SET CMP0007 NEW)
list(FILTER _lfs_libraries EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _lfs_libraries)
get_filename_component(_lfs_app_bin "${_lfs_executable}" DIRECTORY)
foreach(_lfs_source IN LISTS _lfs_declared_runtime)
    if(NOT _lfs_source)
        continue()
    endif()
    get_filename_component(_lfs_name "${_lfs_source}" NAME)
    set(_lfs_local "${_lfs_app_bin}/${_lfs_name}")
    if(NOT EXISTS "${_lfs_local}")
        message(FATAL_ERROR "Build has not deployed declared runtime: ${_lfs_name}")
    endif()
    file(SHA256 "${_lfs_source}" _lfs_source_hash)
    file(SHA256 "${_lfs_local}" _lfs_local_hash)
    if(NOT _lfs_source_hash STREQUAL _lfs_local_hash)
        message(FATAL_ERROR "Stale app-local declared runtime: ${_lfs_name}; rebuild before installing")
    endif()
    list(APPEND _lfs_libraries "${_lfs_local}")
endforeach()
file(TO_CMAKE_PATH "$ENV{SystemRoot}/System32" _lfs_system32)
string(TOLOWER "${_lfs_system32}" _lfs_system32_lower)
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${_lfs_executable}"
    LIBRARIES ${_lfs_libraries}
    DIRECTORIES ${_lfs_search_dirs}
    PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*" "nvcuda[.]dll"
    POST_EXCLUDE_REGEXES ".*[Ss][Yy][Ss][Tt][Ee][Mm]32[/\\].*" ".*[Ww][Ii][Nn][Ss][Xx][Ss][/\\].*"
    RESOLVED_DEPENDENCIES_VAR _lfs_deps
    UNRESOLVED_DEPENDENCIES_VAR _lfs_unresolved
    CONFLICTING_DEPENDENCIES_PREFIX _lfs_conflicts)
if(_lfs_unresolved)
    message(FATAL_ERROR "Portable dependency closure failed: unresolved=${_lfs_unresolved}")
endif()
# On Windows, the executable scan can return a DLL without including all of
# that DLL's imports. Scan the resolved libraries as roots as well, so their
# transitive runtime dependencies become part of the installed inventory.
if(_lfs_deps)
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES ${_lfs_deps}
        DIRECTORIES ${_lfs_search_dirs}
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*" "nvcuda[.]dll"
        POST_EXCLUDE_REGEXES ".*[Ss][Yy][Ss][Tt][Ee][Mm]32[/\\].*" ".*[Ww][Ii][Nn][Ss][Xx][Ss][/\\].*"
        RESOLVED_DEPENDENCIES_VAR _lfs_indirect_deps
        UNRESOLVED_DEPENDENCIES_VAR _lfs_indirect_unresolved
        CONFLICTING_DEPENDENCIES_PREFIX _lfs_indirect_conflicts)
    if(_lfs_indirect_unresolved)
        message(FATAL_ERROR "Portable indirect dependency closure failed: unresolved=${_lfs_indirect_unresolved}")
    endif()
    list(APPEND _lfs_deps ${_lfs_indirect_deps})
    foreach(_lfs_conflict IN LISTS _lfs_indirect_conflicts_FILENAMES)
        list(APPEND _lfs_conflicts_${_lfs_conflict} ${_lfs_indirect_conflicts_${_lfs_conflict}})
    endforeach()
    list(APPEND _lfs_conflicts_FILENAMES ${_lfs_indirect_conflicts_FILENAMES})
endif()
# SDK roots and app-local copies may resolve the same name twice. Accept only
# byte-identical copies; never select an arbitrary ABI from conflicting SDKs.
foreach(_lfs_conflict IN LISTS _lfs_conflicts_FILENAMES)
    set(_lfs_previous_hash "")
    foreach(_lfs_candidate IN LISTS _lfs_conflicts_${_lfs_conflict})
        file(SHA256 "${_lfs_candidate}" _lfs_candidate_hash)
        if(_lfs_previous_hash AND NOT _lfs_previous_hash STREQUAL _lfs_candidate_hash)
            message(FATAL_ERROR "Conflicting runtime bytes for ${_lfs_conflict}: ${_lfs_conflicts_${_lfs_conflict}}")
        endif()
        set(_lfs_previous_hash "${_lfs_candidate_hash}")
    endforeach()
    list(GET _lfs_conflicts_${_lfs_conflict} 0 _lfs_candidate)
    list(APPEND _lfs_deps "${_lfs_candidate}")
endforeach()
list(APPEND _lfs_deps ${_lfs_libraries})
list(REMOVE_DUPLICATES _lfs_deps)
set(_lfs_records "[]")
set(_lfs_index 0)
foreach(_lfs_dll IN LISTS _lfs_deps)
    file(TO_CMAKE_PATH "${_lfs_dll}" _lfs_normalized)
    string(TOLOWER "${_lfs_normalized}" _lfs_lower)
    string(FIND "${_lfs_lower}" "${_lfs_system32_lower}/" _lfs_is_system)
    if(_lfs_is_system EQUAL 0)
        continue()
    endif()
    get_filename_component(_lfs_name "${_lfs_dll}" NAME)
    # Codec modules are data under bin/extensions; import DLLs accompany the EXE.
    if(_lfs_name MATCHES "^nvjpeg(2k)?_ext.*[.]dll$")
        set(_lfs_relative "${_lfs_bindir}/extensions/${_lfs_name}")
    else()
        set(_lfs_relative "${_lfs_name}")
    endif()
    get_filename_component(_lfs_destination "${_lfs_relative}" DIRECTORY)
    string(TOLOWER "${_lfs_relative}" _lfs_key)
    file(SHA256 "${_lfs_dll}" _lfs_hash)
    if(DEFINED _lfs_seen_${_lfs_key})
        if(NOT _lfs_seen_${_lfs_key} STREQUAL _lfs_hash)
            message(FATAL_ERROR "Conflicting installed runtime: ${_lfs_relative}")
        endif()
        continue()
    endif()
    set(_lfs_seen_${_lfs_key} "${_lfs_hash}")
    file(INSTALL "${_lfs_dll}" DESTINATION "${CMAKE_INSTALL_PREFIX}/${_lfs_destination}")
    string(JSON _lfs_records SET "${_lfs_records}" ${_lfs_index}
        "{\"path\":\"${_lfs_relative}\",\"sha256\":\"${_lfs_hash}\"}")
    math(EXPR _lfs_index "${_lfs_index}+1")
endforeach()
file(MAKE_DIRECTORY "${_lfs_audit_dir}")
file(WRITE "${_lfs_audit_dir}/runtime-dependencies.json"
    "{\"schema_version\":1,\"policy\":\"PE imports plus declared dynamic modules; Windows/NVIDIA driver external\",\"files\":${_lfs_records}}\n")
cmake_policy(POP)
