# Windows 构建、配置更新与交付

本手册适用于 `gs_portable` 的原生前处理和 3DGS Release portable 包。CMake target 是 `Run-GS`，生成和安装的程序是 `SwapTexture.exe`。

## 1. 构建环境

- Windows x64、Visual Studio 2022 Community 的 C++ 工具链、CMake 3.30+、Ninja。
- CUDA Toolkit 12.8+，以及运行 CUDA 程序所需的 NVIDIA GPU 和兼容驱动。
- Python 3.11+；默认使用 `C:\Users\shiboke\AppData\Local\anaconda3\python.exe`。加密优先使用 `cryptography`，Windows 上缺少该包时使用脚本内置的 PowerShell/.NET AES 实现。
- 可用的 vcpkg checkout 和 `vcpkg.exe`，默认在源码目录同级的 `vcpkg`；项目依赖按根 `vcpkg.json` 安装，首次构建可能需要下载。
- 源码内的 nvImageCodec 及其构建输入，以及 `third_party/runtime` 中匹配 manifest 的 COLMAP、SuperResolution payload。

`third_party/runtime/licenses/THIRD_PARTY_NOTICES.md` 是 provider 校验所需的源码材料，须保留。它不作为运行包 payload 安装。此构建路径不需要原 Python SwapTexture 项目，也不使用旧工作站的 LibTorch 下载脚本。

## 2. 构建和发布

在普通 CMD 中执行：

```bat
cd /d C:\Users\shiboke\repos\gs_portable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+
```

完整流程依次导入 VS 开发环境、检查 vcpkg 依赖、配置 Release/Ninja、生成七个密文配置、构建程序、安装新 staging、检查文件和依赖哈希、启动程序、发布并再次检查。配置设置包括 `BUILD_PORTABLE=ON`、`BUILD_TESTS=OFF`、`BUILD_CUDA_PTX_ONLY=ON`、`LFS_ENABLE_CLI_HELP=OFF`。

当 `tests/python` 不存在时，脚本明确提示跳过开发测试。本精简副本采用此路径；包审计和受限 PATH 启动检查仍属于完整交付流程。`-SkipTests` 只控制开发测试。`-BuildOnly` 在构建阶段结束，不执行 install、包审计、发布或发布版启动检查。

只构建，或选择其他策略：

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+ -BuildOnly
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy mcmc
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy adc -Archive
```

`-Archive` 需要 `tar.exe`，成功时另外生成 ZIP 和 `.sha256`。

### 参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `Strategy` | `igs+` | 选择 `igs+`、`mcmc` 或 `adc` 的六个正式训练配置。 |
| `Parallel` | `2` | CMake 构建并发，允许 1–64。 |
| `VcpkgConcurrency` | `1` | vcpkg 构建并发，允许 1–64。 |
| `BuildOnly` | 关闭 | 构建后结束，保留本次工作目录。 |
| `SkipTests` | 关闭 | 跳过开发测试；没有 `tests/python` 时自动提示并跳过。 |
| `Publish` | 开启 | 完整交付默认发布；单独关闭会报错，需使用 `BuildOnly`。 |
| `PublishDirectory` | `SwapTexture` | 最终包根目录；相对路径以源码根目录解析。 |
| `Archive` | 关闭 | 额外生成 ZIP 与 SHA-256。 |
| `ArtifactDirectory` | `build-swaptexture-artifacts` | 压缩包输出目录。 |
| `VsDevCmd` | VS 2022 Community 的 `Common7/Tools/VsDevCmd.bat` | 初始化 x64 MSVC 环境。 |
| `CMakeExe` | VS 自带 CMake | 可指定其他 CMake 3.30+。 |
| `NinjaExe` | VS 自带 Ninja | 指定 Ninja。 |
| `VcpkgRoot` | 源码目录同级 `vcpkg` | vcpkg checkout。 |
| `VcpkgExe` | `VcpkgRoot/vcpkg.exe` | vcpkg 可执行文件。 |
| `PythonExe` | conda base Python | 执行加密器、包校验和启动检查。 |

### 增量构建和失败恢复

缓存位于 `build-swaptexture-work/<策略>/build/`，其中 `igs+` 的目录名是 `igsplus`。再次执行相同命令会复用编译产物及该目录内的 vcpkg 安装前缀，但仍检查依赖并重新运行 CMake configure。不同策略使用独立缓存，同策略通过文件锁避免并发构建。

每次交付创建 `<策略>/runs/<时间戳>-<随机ID>/`，使用新的 staging 和审计快照。安装通过检查后，将已有发布目录移动到本次运行目录中的 `previous-publish`，再发布新目录。发布后复核失败时尝试恢复旧包，保留失败现场。

完整交付成功后删除本次运行目录，包括审计快照和旧包备份；构建缓存和最终包保留。失败或 `-BuildOnly` 时保留工作目录。缓存中的 `build/package-audit/` 保存最近一次 install 的两份清单；需要长期保留单次审核报告时，应在清理前另行保存或调整工作流。

发布目录不得与持久构建工作区重叠。本脚本没有 `BuildDirectory`、`StageDirectory`、`AuditDirectory` 或 `NoVcpkgInstall` 参数。

## 3. 配置来源与加密

正式输入位于 [resources/swaptexture_configs](../../resources/swaptexture_configs/README.md)。该目录包含一个产品 manifest、一个融合配置和三套各六个训练配置，共 20 个 JSON。

| 策略 | 训练输入前缀 |
|---|---|
| `igs+` | `eval/improvedGSplus_optimization_params_pack` |
| `mcmc` | `eval/mcmc_optimization_params_pack` |
| `adc` | `eval/adc_optimization_params_pack` |

每个前缀追加 `_{inc,scan}_{fast,medium,quality}.json`。一次构建选择六个训练输入，并加密为 `GS_params_{inc,scan}_{fast,medium,quality}.bin`；`swaptexture_params.json` 加密为 `Swaptexture_params.bin`。七个文件安装到包内 `bin`。`manifest.json` 单独明文安装到包根目录，不参与配置加密。

`cmake/EncryptedConfigs.cmake` 自动运行 `scripts/encrypt_eval_configs.py`，后者使用同目录的 `encrypt_config_json.py`。加密前验证整批 JSON 根类型、重复键和非有限数值，随后通过临时文件替换输出。配置字段和模式组合还会在原生程序读取时校验。

只检查输入，不写密文：

```bat
C:\Users\shiboke\AppData\Local\anaconda3\python.exe scripts\encrypt_eval_configs.py --strategy igs+ --eval-dir resources\swaptexture_configs\eval --swaptexture-config resources\swaptexture_configs\swaptexture_params.json --output-dir build\encrypted-config-check --dry-run
```

## 4. 更新已有包的配置

修改正式 JSON 后，可调用 [配置更新脚本](../../scripts/update_swaptexture_delivery_configs.ps1)。该操作不编译 C++，也不创建新包。

```bat
rem 更新融合配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\swaptexture_params.json

rem 更新一个 IGS+ 档位
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\eval\improvedGSplus_optimization_params_pack_scan_medium.json

rem 更新当前策略的全部七个配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -All -Strategy igs+

rem 只预览另一个包的替换目标
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -All -Strategy igs+ -PackageDirectory "D:\delivery\SwapTexture" -WhatIf
```

| 参数 | 行为 |
|---|---|
| `ConfigPath` | 当前策略的正式 JSON 路径；在 PowerShell 中可传入数组。与 `All` 互斥。 |
| `All` | 更新当前策略六个训练 BIN 和融合 BIN，不更新产品 manifest。 |
| `Strategy` | 默认 `igs+`；支持 `mcmc`、`adc`，应与目标包配置策略一致。 |
| `PackageDirectory` | 默认源码根目录下的 `SwapTexture`；传包根目录，不是 `bin`。 |
| `PythonExe` | 默认 conda base Python；可指定符合构建环境要求的其他解释器。 |
| `WhatIf` | 预览路径和目标，不加密、不写文件，也不验证 JSON 内容。 |

输入与包目录的相对路径均按源码根目录解析。脚本先验证并加密当前策略的全部七个 JSON，再替换选中的文件，因此未选中的同批 JSON 也必须有效。替换失败时尝试回滚；若回滚受文件占用等原因阻碍，会保留 `.lfs-config-*.bak` 并报告路径。

更新改变配置哈希，先前的包外清单和报告随之失效。正式发布请重新执行完整交付命令。

## 5. 包审计与启动检查

`scripts/swaptexture_m6_package_verify.py` 是当前交付使用的文件校验器。它检查产品 manifest、七个密文配置、provider 的文件集合及哈希、包外安装清单、依赖清单和禁止进入运行包的材料。

`scripts/check_portable_startup.py` 将 PATH 限制到包根目录和 Windows 系统目录，再从包目录执行 `SwapTexture.exe --version`，检查真实程序是否能加载依赖。完整交付在 staging 和发布后分别运行该检查。

单独检查已有包能否启动：

```bat
C:\Users\shiboke\AppData\Local\anaconda3\python.exe scripts\check_portable_startup.py --package-root SwapTexture
```

这些检查分别验证文件清单和程序启动；不替代完整训练、图像质量比较或其他机器上的运行验证。脚本的成功输出才是本次执行的证据，本文不记录预先通过的验收结论。
