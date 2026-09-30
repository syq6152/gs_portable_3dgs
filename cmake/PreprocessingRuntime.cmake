# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc photo imgcodecs videoio)
# The local ORT SDK export references these targets without find_dependency.
find_package(Protobuf CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# vcpkg's app-local step searches its own prefix. During migration OpenCV may use
# a second prefix with a different zlib DLL name (z.dll vs zlib1.dll).
function(lfs_attach_preprocessing_runtime target)
    if(NOT WIN32)
        return()
    endif()
    set(_runtime $<TARGET_FILE:opencv_core> $<TARGET_FILE:opencv_imgproc> $<TARGET_FILE:opencv_photo>
        $<TARGET_FILE:opencv_imgcodecs> $<TARGET_FILE:opencv_videoio> $<TARGET_FILE:onnxruntime::onnxruntime>)
    foreach(_config RELEASE DEBUG)
        get_target_property(_location opencv_core IMPORTED_LOCATION_${_config})
        if(_location)
            get_filename_component(_bin "${_location}" DIRECTORY)
            foreach(_dependency z.dll zd.dll zlib1.dll zlibd1.dll jpeg62.dll libpng16.dll libpng16d.dll)
                if(EXISTS "${_bin}/${_dependency}")
                    if(_config STREQUAL "DEBUG")
                        list(APPEND _runtime "$<$<CONFIG:Debug>:${_bin}/${_dependency}>")
                    else()
                        list(APPEND _runtime "$<$<NOT:$<CONFIG:Debug>>:${_bin}/${_dependency}>")
                    endif()
                endif()
            endforeach()
            # VideoIO links the native FFmpeg DLLs from the same OpenCV prefix.
            # Do not redistribute or launch an ffmpeg executable.
            file(GLOB _ffmpeg_runtime
                "${_bin}/avcodec*.dll" "${_bin}/avformat*.dll" "${_bin}/avutil*.dll"
                "${_bin}/swscale*.dll" "${_bin}/swresample*.dll"
                "${_bin}/avfilter*.dll" "${_bin}/avdevice*.dll")
            foreach(_dependency IN LISTS _ffmpeg_runtime)
                if(_config STREQUAL "DEBUG")
                    list(APPEND _runtime "$<$<CONFIG:Debug>:${_dependency}>")
                else()
                    list(APPEND _runtime "$<$<NOT:$<CONFIG:Debug>>:${_dependency}>")
                endif()
            endforeach()
        endif()
    endforeach()
    foreach(_config RELEASE DEBUG)
        get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION_${_config})
        # An SDK may expose only Release. Match CMake's imported-target fallback;
        # this does not assert that a mixed/Debug build has been validated.
        if(NOT _ort_location)
            get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
        endif()
        if(_ort_location)
            get_filename_component(_ort_bin "${_ort_location}" DIRECTORY)
            file(GLOB _ort_runtime "${_ort_bin}/onnxruntime_providers_shared.dll"
                "${_ort_bin}/libprotobuf*.dll" "${_ort_bin}/re2.dll" "${_ort_bin}/abseil_dll.dll")
            foreach(_dependency IN LISTS _ort_runtime)
                if(_config STREQUAL "DEBUG")
                    list(APPEND _runtime "$<$<CONFIG:Debug>:${_dependency}>")
                else()
                    list(APPEND _runtime "$<$<NOT:$<CONFIG:Debug>>:${_dependency}>")
                endif()
            endforeach()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _runtime)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_runtime} $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS)
    if(target STREQUAL "Run-GS")
        # Explicit dynamic roots for full portable dependency closure.
        set_property(GLOBAL PROPERTY LFS_PREPROCESSING_RUNTIME_FILES "${_runtime}")
        # A named component permits isolated DLL verification without
        # requiring app, GUI, Python, COLMAP, or SR installation targets.
        # Omitting --component still includes these files in the full install.
        install(FILES ${_runtime} DESTINATION "." COMPONENT NativePreprocessing)
    endif()
endfunction()
