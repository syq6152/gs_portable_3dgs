/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "preprocessing/providers.hpp"
namespace lfs::preprocess {
    class IForegroundMaskProvider;
    std::expected<PreprocessResult, Error> preprocess_registered(
        const PreprocessRequest&, IColmapProvider*, const ExecutionContext& = {}, IForegroundMaskProvider* foreground_override = nullptr);
} // namespace lfs::preprocess
