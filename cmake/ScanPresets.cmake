# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
function(lfs_attach_scan_presets target)
    foreach(quality fast medium quality)
        set(preset "${CMAKE_SOURCE_DIR}/resources/scan_presets/scan_${quality}.json")
        # Presets are inputs, not generated source. A change must relink/redeploy.
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${preset}")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${target}>/scan_presets"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${preset}"
                "$<TARGET_FILE_DIR:${target}>/scan_presets/scan_${quality}.json")
        install(FILES "${preset}" DESTINATION bin/scan_presets)
    endforeach()
endfunction()
