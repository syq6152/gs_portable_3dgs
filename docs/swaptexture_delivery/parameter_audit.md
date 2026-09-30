# CLI 参数与加密配置

本文描述原生 `SwapTexture.exe` 入口。发布脚本设置 `LFS_ENABLE_CLI_HELP=OFF`，Release 包关闭 `-h/--help` 帮助输出；参数以本文和 [原生入口源码](../../src/app/reconstruction_command.cpp) 为准。

## 1. 根级 SwapTexture CLI

参数直接放在 EXE 后，不加 `reconstruct`：

| 参数 | 语义 |
|---|---|
| `--bin_path <file>` | 必填，scanner `texture_data.bin`；支持明文路径或已有路径密文解码。 |
| `--images_inc_path <dir>` | registered 模式的增量图片；配置增量视频时可由视频输入替代，scan 模式无需增量图片。 |
| `--enable_gs_train` | presence flag；存在时完成前处理后执行同进程 GS 训练。 |
| `--gs-input-source registered\|scan` | GS 输入源，默认 `registered`。 |
| `--gs-training-mode fast\|medium\|quality` | 从包内当前策略的六个配置中选择档位，默认 `medium`。 |
| `--mesh-init-gs-scene <mesh>` | 兼容参数；原生重建入口接受但忽略其值，mesh 从 scanner bin 引用及回退路径解析。 |
| `--result_path <path>` | 训练时是 `.ply` 文件；只做前处理时是目录。未提供时使用工作目录内的输出。 |
| `--log_path <dir-or-log>` | 日志目录；传入 `.log` 文件路径时使用其父目录。 |
| `--enable_inc_sim3_registration` | 保留参数名，但当前明确返回 UnsupportedFeature，未实现原生 Sim3/TEASER++。 |

训练覆盖参数放在 `--` 后，仅在 `--enable_gs_train` 时允许。dataset、output、config、strategy 与 mesh-init 等编排拥有的参数禁止覆盖。

从源码根目录运行一个两步训练检查：

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --enable_gs_train --gs-input-source registered --gs-training-mode fast --result_path "D:\output\Gaussian.ply" -- --iter 2
```

`--iter 2` 只适合检查调用链，不能用于评估完整训练质量。正常运行时去掉该覆盖。

## 2. 融合配置字段

正式明文源是 [swaptexture_params.json](../../resources/swaptexture_configs/swaptexture_params.json)，运行包从 `bin/Swaptexture_params.bin` 读取其密文。下表的“当前值”来自此源码副本的配置文件；修改 JSON 后需重新加密才能影响已生成的包。

| JSON 路径 | 类型或范围 | 作用与当前配置值 |
|---|---|---|
| `schema_version` | integer `1` | 必填 schema。 |
| `reconstruction.workspace_root` | string | 当前 `outputs`；相对路径以运行数据目录解析，安装包中为 `bin`。 |
| `reconstruction.input_mode` | `auto\|scan\|registered\|mesh-export` | 当前 `auto`，服从 CLI source；固定模式与显式 CLI source 冲突时拒绝。 |
| `reconstruction.retain_failed_stages` | boolean | 是否保留失败阶段，当前 false。 |
| `reconstruction.preprocess_only` | boolean | 强制仅前处理，当前 false；true 与 `--enable_gs_train` 互斥。 |
| `reconstruction.write_diagnostic_json` | boolean | 控制重建诊断 JSON 和此重建训练路径的参数 JSON 保存，当前 false。 |
| `common.debug_mesh_overlay` | boolean | 投影调试图，当前 true；mesh-export 模式不允许 true。 |
| `common.scan_blur_filter` | boolean | scan 模糊过滤，当前 false。 |
| `common.verbose_output` | boolean | 详细前处理及训练输出，当前 true；未配置该字段时原生入口默认 false。与帮助开关独立。 |
| `scan.sample_limit` | positive integer | scan 采样上限，当前 200。 |
| `scan.enhance` | boolean | scan 增强，当前 true。 |
| `scan.denoise` | boolean | scan 降噪，当前 false。 |
| `scan.super_resolution` | boolean | scan 超分，当前 true。 |
| `scan.sr_final_scale` | integer 1..4 | 最终缩放，当前 2。 |
| `registered.triangulation` | `mesh\|colmap` | registered 三角化路径，当前 `mesh`。 |
| `registered.sample_limit` | positive integer | registered 的 scan seed 上限，当前 120。 |
| `registered.match_3d_threshold` | positive finite number | mesh 匹配距离阈值，当前 0.0004。 |
| `registered.inc_blur_filter` | boolean | 增量图模糊过滤，当前 false。 |
| `registered.export_registration_triplets` | boolean | 输出注册三件套，当前 false。 |
| `registered.register_twice` | boolean | 二次注册，当前 false。 |
| `registered.face_mode` | boolean | 兼容 no-op，当前 false。 |
| `registered.incremental_mask` | boolean | 增量前景 mask，当前 false；开启时必须提供模型。 |
| `registered.foreground_model` | string | 外部 ONNX 模型路径，当前为空；模型不随包交付。 |
| `registered.incremental_video` | string | 增量视频路径，当前为空，使用 CLI 增量图片输入。 |
| `registered.video_frame_count` | positive integer | 视频均匀抽帧数，当前 80。 |
| `mesh_export.enhance` | boolean | mesh-export 增强，当前 true。 |
| `mesh_export.super_resolution` | boolean | mesh-export 超分，当前 false。 |
| `mesh_export.sr_final_scale` | integer 1..4 | 最终缩放，当前 2。 |
| `mesh_export.match_3d_threshold` | positive finite number | mesh-export 匹配阈值，当前 0.0004。 |

原生 loader 拒绝未知 section/field、重复键、错误类型、非有限数字和越界值。配置中的工作目录、外部模型、视频相对路径按运行数据目录解析；安装包为 `bin`，开发构建通常为 EXE 目录。CLI 路径使用调用时的工作目录规则，建议传绝对路径。

开发调试可设置 `LFS_SWAPTEXTURE_CONFIG` 指向 JSON 或 BIN；正常发布不需要该环境变量。它只替换融合配置来源，不会重写包内六个 GS 密文。

## 3. 六个训练档位与策略

正式 GS 输入均位于 [eval](../../resources/swaptexture_configs/eval)。每套策略的前缀如下：

| 策略 | JSON 文件前缀 |
|---|---|
| `igs+` | `improvedGSplus_optimization_params_pack` |
| `mcmc` | `mcmc_optimization_params_pack` |
| `adc` | `adc_optimization_params_pack` |

将前缀与下表后缀组合，即得到精确源文件名，例如 `improvedGSplus_optimization_params_pack_scan_medium.json`。

| CLI 输入源与档位 | JSON 后缀 | 包内密文 | 当前 IGS+ / MCMC / ADC 迭代数 |
|---|---|---|---|
| `registered + fast` | `_inc_fast.json` | `GS_params_inc_fast.bin` | 5000 / 5000 / 5000 |
| `registered + medium` | `_inc_medium.json` | `GS_params_inc_medium.bin` | 8000 / 8000 / 8000 |
| `registered + quality` | `_inc_quality.json` | `GS_params_inc_quality.bin` | 12000 / 10000 / 10000 |
| `scan + fast` | `_scan_fast.json` | `GS_params_scan_fast.bin` | 4000 / 4000 / 4000 |
| `scan + medium` | `_scan_medium.json` | `GS_params_scan_medium.bin` | 6000 / 6000 / 6000 |
| `scan + quality` | `_scan_quality.json` | `GS_params_scan_quality.bin` | 10000 / 8000 / 8000 |

构建时用 `build_swaptexture_delivery.ps1 -Strategy igs+`、`mcmc` 或 `adc` 选择整套策略；直接配置 CMake 时使用 `LFS_SWAPTEXTURE_TRAINING_STRATEGY`。运行时 source 和 mode 从已打包的六个 BIN 中选档，不在三种算法之间切换。不要仅修改 IGS+ JSON 中的 `strategy` 字段来代替切换正式策略输入。

配置目录的 18 个训练 JSON 加上融合配置共 19 个运行配置输入；另一个 `manifest.json` 是产品 metadata，明文复制到包根目录。配置中的 `_run_command` 为保留的说明字段，其中旧程序名不改变当前实际输出名 `SwapTexture.exe`。

## 4. 其他入口与模式边界

原 LichtFeld 训练/版本入口仍由 [argument_parser.cpp](../../src/core/argument_parser.cpp) 解析：

```bat
SwapTexture\SwapTexture.exe --version
SwapTexture\SwapTexture.exe -d "D:\dataset" -o "D:\training-output" --config "D:\config\training.json"
```

根级 `--bin_path` 属于 SwapTexture 入口；`reconstruct --bin-path` 属于原生重建入口，不能混用两套参数。帮助输出是否可用取决于构建的 `LFS_ENABLE_CLI_HELP`，本交付脚本显式关闭它。

模式约束：

- 默认 source 为 `registered`、档位为 `medium`，默认不启动 GS 训练。
- registered 需要增量图片或配置中的增量视频。scan 不需要增量图片；scan 仅前处理需要配置 `preprocess_only=true`。
- `mesh-export` 通过融合配置选择，只导出数据，不训练；还必须关闭 `common.debug_mesh_overlay`，且不能传 `--gs-training-mode`。
- 原生重建自动解析初始化 mesh；没有可用三角 mesh 时 GS 训练会报错。支持无贴图 mesh，由训练 RGB 视图初始化颜色。
- mesh、外部 mesh、点云与相机必须使用相同坐标系和单位。当前 `MESH_SCENE_POSITION_DIVISOR = 1.0f` 保留输入坐标，不需要旧 Python 调用链的乘 1000 / 除 1000 配对。
- 直接训练入口的 `--mesh-init-gs-scene` 仍表示实际 mesh 输入；兼容忽略只适用于重建入口。

修改融合/训练 JSON 后可以执行 [配置更新](build_package_audit.md#4-更新已有包的配置)；正式交付需重新生成包外审计记录。
