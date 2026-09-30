# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)
include(GNUInstallDirs)
set(LFS_INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}")
set(LFS_INSTALL_EXTENSIONS_DIR "extensions")
if(WIN32)
    # Windows resolves native imports before main; keep DLLs beside the EXE.
    set(LFS_INSTALL_RUNTIME_DIR ".")
    set(LFS_INSTALL_EXTENSIONS_DIR "${CMAKE_INSTALL_BINDIR}/extensions")
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/../resources/swaptexture_configs/manifest.json" DESTINATION ".")
endif()
