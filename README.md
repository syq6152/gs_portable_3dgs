# GS Portable

用于 Windows 的 SwapTexture 原生前处理与 3D Gaussian Splatting 训练源码工作区。前处理由原 Python SwapTexture 流程迁移为 C++，随后在同一进程执行 LichtFeld 3DGS 训练；正常运行无需原 `swaptexture/main.py` 工程。

本目录保留现有源码相对路径，移除了研究资料、历史包、开发测试、构建缓存和日志。构建目标仍叫 `Run-GS`，生成的程序名为 **`SwapTexture.exe`**。

## 目录

| 路径 | 用途 |
|---|---|
| `src/app`、`src/preprocessing` | 命令行入口、前处理及训练编排 |
| `src/core`、`src/io`、`src/geometry`、`src/training` | 共享基础设施、数据读写、几何和 3DGS 训练 |
| `src/rendering` 及其余 `src` 模块 | mesh2splat、着色器及保留的原生源码/头文件；部分模块不参与当前交付构建 |
| `cmake`、`CMakeLists.txt` | 编译、配置加密、运行依赖收集与安装 |
| `external` | 第三方源码和头文件，沿用原路径 |
| `third_party/runtime` | 前处理需要的 COLMAP、超分程序、哈希清单和来源记录 |
| `resources/swaptexture_configs` | 当前正式配置：融合参数、18 个训练 preset 和产品 manifest |
| `scripts` | 构建分发、配置更新、加密、包校验及启动检查 |
| `.github/overlays` | vcpkg 清单引用的 overlay；不含 CI workflow |
| `docs/swaptexture_delivery` | 参数、构建分发、源码映射及迁移说明 |

保留完整 `src` 是为了维持头文件依赖和模块结构；精简工作区仍以现有原生交付目标为入口，不包含 GUI 构建配套文档及测试套件。构建所需的许可证和第三方来源记录保留在源码中。

## 构建与分发

项目要求 Windows x64、Visual Studio 2022 Community C++ 工具链、CMake 3.30+、Ninja、CUDA Toolkit 12.8+ 及匹配的 NVIDIA 驱动。以下版本要求来自当前 CMake。构建脚本默认使用：

- VS：`C:\Program Files\Microsoft Visual Studio\2022\Community`
- Python：`C:\Users\shiboke\AppData\Local\anaconda3\python.exe`（包校验器要求 Python 3.11+）
- vcpkg：工作区同级的 `..\vcpkg`，即本机 `C:\Users\shiboke\repos\vcpkg`

在 **CMD** 中运行；脚本会自动初始化 VS 开发环境。首次构建需要下载/编译 vcpkg 依赖，可能耗时较长。这个源码副本不携带旧的 CMake 缓存或已生成的安装包。

```bat
cd /d C:\Users\shiboke\repos\gs_portable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+
```

该命令配置并编译 Release，生成七个加密配置，安装到新 staging，执行包完整性审计和受限 PATH 启动检查，然后发布到 `SwapTexture`。本工作区未迁入 `tests`，脚本会明确跳过开发单元测试；包审计和启动检查仍照常执行。

```bat
rem 仅编译，不生成分发包
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+ -BuildOnly

rem 选择 MCMC，并生成 ZIP
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy mcmc -Archive
```

`-Strategy` 支持 `igs+`、`mcmc`、`adc`。默认每个策略保留独立缓存 `build-swaptexture-work/<策略>/build`（`igs+` 对应 `igsplus`）。成功分发后清理本轮 staging，保留构建缓存。完整参数见[构建分发说明](docs/swaptexture_delivery/build_package_audit.md)。

## 配置更新

修改 `resources/swaptexture_configs/swaptexture_params.json` 设置前处理；修改 `eval` 下对应策略的 `{inc,scan}_{fast,medium,quality}` 六个 JSON 设置训练。配置内容沿用迁移时工作区版本；JSON 内 `_run_command` 等说明字段可能保留旧程序名，实际命令以本文为准。

已有 `SwapTexture` 分发包时，单独更新配置无需重新编译 C++：

```bat
rem 只更新前处理配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\swaptexture_params.json

rem 更新当前策略全部七个配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -All -Strategy igs+
```

可以用 `-PackageDirectory "D:\delivery\SwapTexture"` 指定包根目录，以 `-WhatIf` 预览。策略应与目标包一致。脚本先校验所选策略全部七个输入，再替换选中的 BIN；更新后旧审计哈希失效，正式分发时重新执行完整构建分发流程。

## 运行

构建发布后的目录以 `SwapTexture\SwapTexture.exe` 为入口，DLL 与主程序同目录，运行配置、着色器和辅助程序位于包内 `bin`。请整体复制分发包。

```bat
rem 前处理，不训练
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --result_path "D:\output\registered"

rem registered 前处理并训练
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --enable_gs_train --gs-input-source registered --gs-training-mode medium --result_path "D:\output\Gaussian.ply"

rem scan-only 前处理并训练
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --enable_gs_train --gs-input-source scan --gs-training-mode medium --result_path "D:\output\Gaussian.ply"
```

训练 mesh 由 scanner bin 引用及回退路径自动解析；保留对应输入文件。训练时 `--result_path` 是 `.ply` 文件，不训练时是目录。启用增量前景 mask 需另行提供配置指定的 ONNX 模型。Release 分发关闭 CLI 帮助输出，参数以[参数说明](docs/swaptexture_delivery/parameter_audit.md)为准。

## 迁移与验证

本副本来自当前 LichtFeld-Studio 工作区，包含迁移时未提交的源码修改。原项目文件未移动或删除，新目录原有 Git 仓库保留。详见[迁移范围与验证记录](docs/swaptexture_delivery/migration.md)。

原项目许可证见 [LICENSE](LICENSE)，依赖信息见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
