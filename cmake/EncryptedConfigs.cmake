# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)

set(LFS_SWAPTEXTURE_TRAINING_STRATEGY "igs+" CACHE STRING
    "Training strategy encrypted into the six packaged SwapTexture presets")
set_property(CACHE LFS_SWAPTEXTURE_TRAINING_STRATEGY PROPERTY STRINGS "igs+" "mcmc" "adc")

if(LFS_SWAPTEXTURE_TRAINING_STRATEGY STREQUAL "igs+")
    set(_lfs_swaptexture_config_prefix "improvedGSplus_optimization_params_pack")
elseif(LFS_SWAPTEXTURE_TRAINING_STRATEGY STREQUAL "mcmc")
    set(_lfs_swaptexture_config_prefix "mcmc_optimization_params_pack")
elseif(LFS_SWAPTEXTURE_TRAINING_STRATEGY STREQUAL "adc")
    set(_lfs_swaptexture_config_prefix "adc_optimization_params_pack")
else()
    message(FATAL_ERROR
        "LFS_SWAPTEXTURE_TRAINING_STRATEGY must be one of: igs+, mcmc, adc; "
        "got '${LFS_SWAPTEXTURE_TRAINING_STRATEGY}'")
endif()

function(lfs_attach_encrypted_configs target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_output_dir "${CMAKE_BINARY_DIR}/encrypted_configs")
    set(_inputs "${CMAKE_SOURCE_DIR}/resources/swaptexture_configs/swaptexture_params.json")
    set(_outputs "${_output_dir}/Swaptexture_params.bin")
    foreach(_mode scan inc)
        foreach(_quality fast medium quality)
            list(APPEND _inputs
                "${CMAKE_SOURCE_DIR}/resources/swaptexture_configs/eval/${_lfs_swaptexture_config_prefix}_${_mode}_${_quality}.json")
            list(APPEND _outputs "${_output_dir}/GS_params_${_mode}_${_quality}.bin")
        endforeach()
    endforeach()
    add_custom_command(OUTPUT ${_outputs}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/encrypt_eval_configs.py"
            --eval-dir "${CMAKE_SOURCE_DIR}/resources/swaptexture_configs/eval"
            --strategy "${LFS_SWAPTEXTURE_TRAINING_STRATEGY}"
            --output-dir "${_output_dir}"
            --swaptexture-config "${CMAKE_SOURCE_DIR}/resources/swaptexture_configs/swaptexture_params.json"
        DEPENDS ${_inputs}
            "${CMAKE_SOURCE_DIR}/scripts/encrypt_eval_configs.py"
            "${CMAKE_SOURCE_DIR}/scripts/encrypt_config_json.py"
        COMMENT "Encrypting ${LFS_SWAPTEXTURE_TRAINING_STRATEGY} reconstruction configuration"
        VERBATIM)
    add_custom_target(encrypted_configs DEPENDS ${_outputs})
    add_dependencies(${target} encrypted_configs)
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${_outputs})
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_outputs} "$<TARGET_FILE_DIR:${target}>"
        COMMAND_EXPAND_LISTS VERBATIM)
    install(FILES ${_outputs} DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()
