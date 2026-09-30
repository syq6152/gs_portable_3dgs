# 当前 C++ 源码定位

原生交付路径由 `SwapTexture.exe` 在同一进程中执行前处理编排和 3DGS 训练。它不启动原 Python SwapTexture 项目，也不需要原项目的 `main.py`、Python 虚拟环境或另一个训练 EXE。构建机 Python 用于配置加密和交付检查。

```text
SwapTexture.exe
  -> 解析 CLI 和 bin/Swaptexture_params.bin
  -> scanner bin、scan 图片、增量图片或视频
  -> 图像处理、相机模型、SQLite、mesh 几何、注册与导出
  -> 必要时调用包内 COLMAP / SuperResolution provider
  -> 验证训练 dataset
  -> 同进程 run_training
  -> Gaussian PLY 输出和可选发布
```

前处理专用模式在数据导出后结束。COLMAP 和 SuperResolution 仍使用包内原生 EXE 和模型；“同进程”指前处理编排与 GS 训练之间没有第二个训练进程。

## 功能到源码

| 功能 | 主要文件 |
|---|---|
| 融合 CLI、密文配置读取、模式验证、训练衔接 | [src/app/reconstruction_command.cpp](../../src/app/reconstruction_command.cpp) |
| 原训练 CLI、JSON 训练配置解析 | [src/core/argument_parser.cpp](../../src/core/argument_parser.cpp) |
| 输入路径解密 | [src/core/path_crypto.cpp](../../src/core/path_crypto.cpp) |
| scanner bin 解析与关联 mesh 查找 | [src/io/formats/scanner_bin.cpp](../../src/io/formats/scanner_bin.cpp) |
| 前处理分流与 PLY 发布 | [src/preprocessing/pipeline.cpp](../../src/preprocessing/pipeline.cpp) |
| scan 帧采样、图片与 pose 准备 | [src/preprocessing/scan_pipeline.cpp](../../src/preprocessing/scan_pipeline.cpp) |
| 图像读写、尺寸处理、增强和降噪 | [src/preprocessing/image_pipeline.cpp](../../src/preprocessing/image_pipeline.cpp) |
| 视频均匀抽帧 | [src/preprocessing/video_frames.cpp](../../src/preprocessing/video_frames.cpp) |
| 模糊过滤 | [src/preprocessing/blur_filter.cpp](../../src/preprocessing/blur_filter.cpp) |
| COLMAP text/bin 模型、相机缩放、scanner pose 转换 | [src/preprocessing/colmap_model.cpp](../../src/preprocessing/colmap_model.cpp) |
| SQLite 相机、关键点、匹配与几何记录 | [src/preprocessing/colmap_database.cpp](../../src/preprocessing/colmap_database.cpp) |
| mesh 射线查询、投影 mask、mesh 三角化 | [src/preprocessing/mesh_geometry.cpp](../../src/preprocessing/mesh_geometry.cpp) |
| registered 主流程、匹配、注册、去畸变与对齐 | [src/preprocessing/registered_pipeline.cpp](../../src/preprocessing/registered_pipeline.cpp) |
| 二次注册合并 | [src/preprocessing/registered_twice.cpp](../../src/preprocessing/registered_twice.cpp) |
| 增量前景分割 | [src/preprocessing/foreground_mask.cpp](../../src/preprocessing/foreground_mask.cpp)，使用 ONNX Runtime CPU 与外部模型 |
| 注册三件套导出 | [src/preprocessing/registration_triplets.cpp](../../src/preprocessing/registration_triplets.cpp) |
| mesh-export 前处理与数据写出 | [mesh_export_pipeline.cpp](../../src/preprocessing/mesh_export_pipeline.cpp)、[mesh_export.cpp](../../src/preprocessing/mesh_export.cpp) |
| 投影调试图 | [src/preprocessing/debug_exports.cpp](../../src/preprocessing/debug_exports.cpp) |
| dataset 校验 | [src/preprocessing/dataset_validation.cpp](../../src/preprocessing/dataset_validation.cpp) |
| provider 定位及调用 | [providers.cpp](../../src/preprocessing/providers.cpp)、[runtime.cpp](../../src/preprocessing/runtime.cpp) |
| 子进程、取消与工作目录生命周期 | [process_runner.cpp](../../src/preprocessing/process_runner.cpp)、[workspace.cpp](../../src/preprocessing/workspace.cpp) |
| 同进程 GS 训练与回调 | [src/app/training_runner.cpp](../../src/app/training_runner.cpp) |
| 训练初始化与 mesh 输入 | [src/training/training_setup.cpp](../../src/training/training_setup.cpp) |
| 策略、训练循环与 CUDA 后端 | [src/training](../../src/training/)、[src/core](../../src/core/) |

这里使用文件链接，不固定可能随编辑漂移的函数行号。保留的 `src` 相对路径用于定位和维护，不能据目录存在推断某个 GUI、插件或研究功能进入当前 `Run-GS` 目标；构建成员以 CMake 配置为准。

## 构建与交付入口

| 内容 | 文件 |
|---|---|
| CMake 主目标和安装规则 | [CMakeLists.txt](../../CMakeLists.txt) |
| 构建、staging、审计与发布 | [scripts/build_swaptexture_delivery.ps1](../../scripts/build_swaptexture_delivery.ps1) |
| 已有包的配置更新 | [scripts/update_swaptexture_delivery_configs.ps1](../../scripts/update_swaptexture_delivery_configs.ps1) |
| 七个配置的加密生成 | [cmake/EncryptedConfigs.cmake](../../cmake/EncryptedConfigs.cmake)、[scripts/encrypt_eval_configs.py](../../scripts/encrypt_eval_configs.py)、[scripts/encrypt_config_json.py](../../scripts/encrypt_config_json.py) |
| 包布局与产品 manifest | [cmake/SwapTextureLayout.cmake](../../cmake/SwapTextureLayout.cmake)、[resources/swaptexture_configs/manifest.json](../../resources/swaptexture_configs/manifest.json) |
| provider 校验和复制 | [cmake/ThirdPartyRuntime.cmake](../../cmake/ThirdPartyRuntime.cmake)、[cmake/VerifyThirdPartyRuntime.cmake](../../cmake/VerifyThirdPartyRuntime.cmake) |
| Windows DLL 收集 | [cmake/WindowsPortableRuntime.cmake](../../cmake/WindowsPortableRuntime.cmake)、[cmake/BundleWindowsRuntime.cmake](../../cmake/BundleWindowsRuntime.cmake) |
| 安装包文件清单与依赖审计 | [scripts/swaptexture_m6_package_verify.py](../../scripts/swaptexture_m6_package_verify.py) |
| 实际 EXE 的受限 PATH 启动检查 | [scripts/check_portable_startup.py](../../scripts/check_portable_startup.py) |

Sim3/TEASER++ 尚无原生实现；启用对应兼容参数会返回明确错误。前景模型也不随包分发，开启增量 mask 需要显式提供。上述映射说明实现归属，不声明与旧 Python 算法或历史数据结果完全等价。
