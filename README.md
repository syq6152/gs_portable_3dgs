# GS Portable

## 1. 工程简介

GS Portable 是面向 Windows x64 的 C++ / CUDA 命令行工程，用于扫描数据前处理、增量图像注册和 3D Gaussian Splatting（3DGS）训练。程序读取 scanner `texture_data.bin` 及关联数据，完成图像处理、相机与几何数据准备，并可在同一进程中执行训练，输出 Gaussian PLY。

主要支持两种输入流程：

- **registered**：结合扫描数据与增量图片，完成注册、数据准备和可选的 GS 训练；增量图片也可通过配置中的视频抽帧提供。
- **scan**：仅使用扫描数据准备训练输入，无需增量图片。

训练支持 `igs+`、`mcmc`、`adc` 三套策略配置，每套包含两种输入源和 `fast`、`medium`、`quality` 三个档位。策略在构建打包时选择，输入源和档位在运行时选择。

CMake 构建目标为 `Run-GS`，输出程序为 **`SwapTexture.exe`**。分发包包含运行 DLL、七个加密配置，以及 COLMAP 和超分辅助程序。运行时请整体复制 `SwapTexture` 目录。

## 2. 编译构建

### 2.1 环境要求

请先参考 [Windows 构建指南](https://github.com/MrNeRF/LichtFeld-Studio/wiki/Build-Instructions-%E2%80%90-Windows)的 **Step 1、Step 2**，安装并验证 Visual Studio 2022 C++ 工具链和 CUDA Toolkit，并确保 NVIDIA 驱动满足下表要求。完成环境准备后，按本文的构建命令编译本工程。

| 组件 | 要求 |
|---|---|
| 操作系统 | Windows x64 |
| C++ 工具链 | Visual Studio 2022 Community，安装 C++ 桌面开发组件和 Windows SDK |
| 构建工具 | CMake 3.30+、Ninja |
| CUDA | CUDA Toolkit 12.8+、兼容的 NVIDIA GPU；当前 CMake 检查驱动版本不低于 570 |
| Python | Python 3.11+，用于配置加密和包校验；默认使用 conda base |
| 依赖管理 | 已准备好的 vcpkg checkout 和 `vcpkg.exe`，依赖由根目录 `vcpkg.json` 声明 |

构建脚本默认使用以下路径：

- Visual Studio：`C:\Program Files\Microsoft Visual Studio\2022\Community`，使用其中的 MSVC、CMake 和 Ninja。
- Python：`C:\Users\shiboke\AppData\Local\anaconda3\python.exe`。
- vcpkg：工程同级的 `..\vcpkg`。

如果本机安装位置不同，可通过 `-VsDevCmd`、`-CMakeExe`、`-NinjaExe`、`-PythonExe` 和 `-VcpkgRoot` 指定路径。首次构建需要下载或恢复 vcpkg 依赖。

### 2.2 完整构建与打包

在 **CMD** 中进入工程根目录执行，脚本会自动初始化 VS x64 开发环境：

```bat
cd /d C:\Users\shiboke\repos\gs_portable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+
```

该命令依次安装依赖、配置 Release / Ninja、编译程序、生成七个加密配置、安装分发包、校验文件和依赖哈希，并在发布前后执行受限 PATH 的真实程序启动检查。成功后生成 `SwapTexture\SwapTexture.exe`。

当前构建启用 `BUILD_PORTABLE=ON` 和 CUDA PTX，关闭 CLI 帮助输出。工程不包含开发测试套件，脚本会明确跳过开发单元测试；包审计和启动检查仍然执行。

其他常用构建方式：

```bat
rem 仅编译，不安装、打包或执行分发包检查
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+ -BuildOnly

rem 使用 MCMC 配置，并额外生成 ZIP
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy mcmc -Archive

rem 使用 ADC 配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy adc
```

| 构建参数 | 默认值或作用 |
|---|---|
| `-Strategy` | `igs+`；可选 `igs+`、`mcmc`、`adc`。目前请先选择 `igs+` 策略。 |
| `-Parallel` | 编译并发数，默认 `8` |
| `-VcpkgConcurrency` | vcpkg 构建并发数，默认 `8` |
| `-BuildOnly` | 仅构建，不执行安装和包验证 |
| `-PublishDirectory` | 分发目录，默认工程根目录下的 `SwapTexture` |
| `-Archive` | 额外生成 ZIP 和 SHA-256 文件 |
| `-ArtifactDirectory` | ZIP 输出目录，默认 `build-swaptexture-artifacts` |

各策略的构建缓存独立保存在 `build-swaptexture-work/<策略>/build`，其中 `igs+` 的目录名为 `igsplus`。重复执行会复用缓存；完整交付成功后清理本轮 staging，保留构建缓存和最终包。详细流程见[构建分发说明](docs/swaptexture_delivery/build_package_audit.md)。

## 3. 运行参数

### 3.1 命令行参数

以下参数直接放在 `SwapTexture.exe` 后，无需添加子命令。建议输入和输出路径使用绝对路径，并用双引号包裹。

| 参数 | 默认值 / 必填 | 说明 |
|---|---|---|
| `--bin_path <file>` | 必填 | scanner `texture_data.bin` 路径；同时保留其引用的图像、mesh 等关联文件 |
| `--images_inc_path <dir>` | registered 输入需要 | 增量图片目录；配置了增量视频时可由视频输入替代，scan 无需此参数 |
| `--enable_gs_train` | 默认关闭 | 传入此开关后，在前处理完成后执行 GS 训练；不附带 `true/false` 值 |
| `--gs-input-source registered\|scan` | `registered` | 选择输入源；需与融合配置中固定的输入模式一致 |
| `--gs-training-mode fast\|medium\|quality` | `medium` | 选择当前包内策略的训练档位 |
| `--result_path <path>` | 可选 | 训练时必须为 `.ply` 文件路径；仅前处理时为工作目录。省略时使用配置工作目录内的输出 |
| `--log_path <dir-or-log>` | 包内 `bin/outputs` | 指定日志目录；若传入 `.log` 文件路径，则使用其父目录 |
| `--mesh-init-gs-scene <mesh>` | 无需传入 | 此重建入口接受但忽略其值，初始化 mesh 根据 scanner bin 的引用和回退路径自动解析 |
| `--enable_inc_sim3_registration` | 不支持 | 当前会返回 `UnsupportedFeature`，请勿启用 |

发布构建关闭 `-h/--help` 输出。可单独执行 `SwapTexture\SwapTexture.exe --version` 查看版本，格式为 `Swaptexture v<packageVersion>`。版本取自 `resources/swaptexture_configs/manifest.json` 的 `extension.packageVersion`，修改后需重新构建。更多配置字段和直接训练入口见[参数说明](docs/swaptexture_delivery/parameter_audit.md)。

### 3.2 运行示例

以下命令在工程根目录执行。

**registered 前处理，不训练：**

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --result_path "D:\output\registered"
```

**registered 前处理并训练：**

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --enable_gs_train --gs-input-source registered --gs-training-mode medium --result_path "D:\output\Gaussian.ply"
```

**scan 前处理并训练：**

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --enable_gs_train --gs-input-source scan --gs-training-mode medium --result_path "D:\output\Gaussian.ply"
```

部分训练参数可以放在 `--` 后覆盖，例如在训练命令末尾添加 `-- --iter 2` 做两步调用检查。这需要同时传入 `--enable_gs_train`，且禁止覆盖流程管理的 dataset、output、config、strategy 和 mesh-init 等参数。两步检查不能用于评价训练质量。

**数据路径：**

测试数据存放在以下共享目录，可将所需样本整体复制到本地后运行：

```text
\\10.10.20.13\智能视觉事业群\00.项目文档\11.三维扫描产品线\TB2201\安卓版本\06算法\windows\AI\纹理替换\GS_portable_test_data
```

以样本 `Pika-红外纹理中物体-0.3mm-壁画花` 为例，部分文件结构如下：

```text
Pika-红外纹理中物体-0.3mm-壁画花/
├── tex_dump/
│   ├── texture_data.bin          # --bin_path 指向此文件
│   ├── mesh.ply                  # 扫描网格
│   ├── 0.png                     # 扫描图片
│   ├── ...
│   └── 243.png
├── 增量图/                        # --images_inc_path 指向此目录
│   ├── IMG_0838.JPG               # 补拍照片
│   └── ...

```

### 3.3 配置文件与模式约束

前处理参数由 `resources/swaptexture_configs/swaptexture_params.json` 管理，训练参数由 `resources/swaptexture_configs/eval` 下对应策略的六个 JSON 管理。构建时加密为包内 `bin/Swaptexture_params.bin` 和六个 `bin/GS_params_*.bin`，运行时读取这些 BIN。

- `reconstruction.input_mode` 默认 `auto`，跟随 CLI 输入源；固定为其他模式时，不能与显式 CLI 输入源冲突。
- `reconstruction.preprocess_only=true` 强制仅前处理，与 `--enable_gs_train` 互斥。scan 模式仅做前处理时必须设置此项。
- GS 训练需要可用的三角 mesh；mesh、点云和相机应处于相同坐标系与单位。
- 启用 `registered.incremental_mask` 时，需要自行提供 `registered.foreground_model` 指定的 ONNX 模型。
- `mesh-export` 通过 `reconstruction.input_mode` 选择，只导出数据；需关闭 `common.debug_mesh_overlay`，且不能传入训练开关或 `--gs-training-mode`。

修改 JSON 后需重新加密才会影响已生成的分发包。已有包可只更新配置，无需重新编译 C++：

```bat
rem 更新前处理配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\swaptexture_params.json

rem 更新 igs+ 的全部七个配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -All -Strategy igs+
```

更新时策略应与目标包一致；可用 `-PackageDirectory` 指定包根目录、`-WhatIf` 预览。修改配置会改变包内文件哈希，正式交付时需重新执行完整构建分发流程。

## 4. 代码文件结构

| 路径 | 职责 |
|---|---|
| `CMakeLists.txt` | 工程构建入口、依赖查找、目标链接与安装配置 |
| `cmake/` | 构建脚本和模板：配置加密、运行依赖收集、第三方文件校验、包布局及审计；属于源码 |
| `src/app/` | 程序入口与流程编排；`main.cpp` 分发运行模式，`reconstruction_command.cpp` 解析重建参数，`training_runner.cpp` 调用训练 |
| `src/preprocessing/` | scan / registered 流程、图像处理、COLMAP 数据准备、注册、mesh 导出和运行工作区管理 |
| `src/core/` | 参数解析、张量与 CUDA 基础设施、场景数据、日志和路径处理 |
| `src/io/` | scanner bin、图像、相机、mesh、点云及模型格式读写 |
| `src/geometry/` | 几何变换、包围盒和 mesh 孔洞填补 |
| `src/training/` | 训练循环、策略、优化器、损失函数及光栅化 CUDA 后端 |
| `src/rendering/` | 渲染相关源码；当前命令行目标使用其中的 `mesh2splat.cpp` |
| `src/visualizer/`、`src/sequencer/`、`src/mcp/`、`src/python/` | 可视化、序列、MCP 和 Python 相关源码；当前 CMake 未启用这些完整模块 |
| `external/` | 随工程提供的第三方源码和头文件 |
| `third_party/runtime/` | COLMAP、超分辅助程序及对应文件清单、哈希和来源记录 |
| `resources/swaptexture_configs/` | 融合配置、18 个训练配置和产品 manifest |
| `src/rendering/resources/shaders/` | mesh2splat 着色器资源 |
| `scripts/` | 构建分发、配置加密与更新、包审计和启动检查脚本 |
| `vcpkg.json`、`vcpkg-configuration.json`、`.github/overlays/` | vcpkg 依赖声明和 overlay 配置 |
| `docs/swaptexture_delivery/` | 参数、构建分发和源码映射等详细文档 |

生成内容与源码分开放置：`build-swaptexture-work/` 保存编译缓存和中间产物，`SwapTexture/` 保存最终运行包，`build-swaptexture-artifacts/` 保存可选 ZIP，`.planning/` 保存本地任务记录和验证报告。

许可证见 [LICENSE](LICENSE)，第三方依赖信息见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
