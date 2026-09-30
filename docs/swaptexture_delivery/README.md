# 原生前处理与 3DGS 交付

`gs_portable` 保留从 scanner 数据前处理、增量图注册到同进程 3DGS 训练的源码和 Windows 交付流程。CMake 主目标名是 `Run-GS`，生成的程序名是 `SwapTexture.exe`。

## 构建和配置入口

从普通 CMD 进入源码目录后执行：

```bat
cd /d C:\Users\shiboke\repos\gs_portable
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_swaptexture_delivery.ps1 -Strategy igs+
```

脚本初始化 VS 2022 Community x64 环境，复用按策略保存的构建缓存，自动加密配置，然后安装、审计并发布到 `SwapTexture/`。此源码副本没有 `tests/python`，构建脚本会明确提示跳过开发测试；完整交付仍执行发布前后的包审计和受限 PATH 启动检查。

修改明文配置后，可更新已有包，无需重新编译：

```bat
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -ConfigPath resources\swaptexture_configs\swaptexture_params.json
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\update_swaptexture_delivery_configs.ps1 -All -Strategy igs+
```

配置更新会使旧包外哈希记录失效；正式交付应重新执行完整构建和审计流程。

## 运行示例

以下命令从源码目录运行，输入路径请替换为实际数据路径。

只做 registered 前处理和纹理数据导出：

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --result_path "D:\output\registered"
```

registered 输入并训练 GS：

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --images_inc_path "D:\input\images_inc" --enable_gs_train --gs-input-source registered --gs-training-mode medium --result_path "D:\output\Gaussian.ply"
```

scan 输入并训练 GS：

```bat
SwapTexture\SwapTexture.exe --bin_path "D:\input\texture_data.bin" --enable_gs_train --gs-input-source scan --gs-training-mode medium --result_path "D:\output\Gaussian.ply"
```

GS 初始化 mesh 从 scanner bin 引用及其回退路径解析；`--mesh-init-gs-scene` 在原生重建入口中仅保留兼容解析，其值不参与初始化。训练需要可用的三角 mesh 和对应输入数据。

不传 `--enable_gs_train` 时只执行前处理。scan 仅前处理还要求配置中的 `reconstruction.preprocess_only=true`；此设置与开启 GS 训练互斥。

## 运行包布局

```text
SwapTexture/
  SwapTexture.exe
  manifest.json
  *.dll
  bin/
    Swaptexture_params.bin
    GS_params_{inc,scan}_{fast,medium,quality}.bin
    extensions/
    resources/shaders/mesh2splat/
    third_party/
      colmap-cuda-cli/
      SuperResolution/
```

DLL 与主程序同目录；加密配置、nvImageCodec 插件、shader 和 provider 位于 `bin`。配置内的相对路径以运行数据目录解析，安装包中即 `bin`。根目录产品 `manifest.json` 明文复制；它与包外的 `package-manifest.json`、`runtime-dependencies.json` 哈希清单用途不同。

发布构建关闭 `-h/--help` 帮助输出，参数说明见下方文档；`SwapTexture.exe --version` 可用于查看版本。

## 继续阅读

- [构建、配置更新和包审计](build_package_audit.md)
- [CLI 参数与配置字段](parameter_audit.md)
- [当前 C++ 源码定位](source_mapping_audit.md)
- [正式明文配置目录](../../resources/swaptexture_configs/README.md)

这些文档描述源码与脚本的行为，不构成本副本已完成编译、真实数据训练或干净机器验收的记录。
