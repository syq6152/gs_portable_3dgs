# 正式交付配置

此目录是原生前处理与 3DGS 交付的明文配置入口，共保留 20 个 JSON，配置内容沿用当前源码。

| 文件 | 用途 |
|---|---|
| `manifest.json` | 产品 metadata，明文复制到运行包根目录。 |
| `swaptexture_params.json` | 前处理、注册、图像处理、导出和工作目录设置，加密为 `bin/Swaptexture_params.bin`。 |
| `eval/improvedGSplus_optimization_params_pack_{inc,scan}_{fast,medium,quality}.json` | IGS+ 六个正式训练档位。 |
| `eval/mcmc_optimization_params_pack_{inc,scan}_{fast,medium,quality}.json` | MCMC 六个正式训练档位。 |
| `eval/adc_optimization_params_pack_{inc,scan}_{fast,medium,quality}.json` | ADC 六个正式训练档位。 |

大括号表示文件名组合。一次交付通过 `-Strategy igs+|mcmc|adc` 选择一套策略，输出六个固定名称的 `GS_params_*.bin` 和一个融合 BIN。运行时按 input source 和训练档位选择已打包的配置。

[CMake 加密目标](../../cmake/EncryptedConfigs.cmake) 调用 [加密脚本](../../scripts/encrypt_eval_configs.py) 在构建树生成密文，安装阶段将它们复制到运行包 `bin`。训练和融合 JSON 不进入运行包。

在源码根目录的 CMD 中执行完整交付：

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+
```

只更新已有包的融合配置：

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\swaptexture_params.json
```

更新脚本会验证当前策略全部七个输入，但只替换所选 BIN。配置哈希变化后需重新执行完整交付以更新包外审计。产品 manifest 不属于更新脚本的 `-All` 范围。

字段和使用边界见 [参数说明](../../docs/swaptexture_delivery/parameter_audit.md)，构建与配置更新详情见 [操作手册](../../docs/swaptexture_delivery/build_package_audit.md)。
