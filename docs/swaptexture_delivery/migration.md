# GS Portable 迁移记录

迁移日期：2026-09-29。来源：`C:\Users\shiboke\repos\LichtFeld-Studio` 当前工作区；目标：`C:\Users\shiboke\repos\gs_portable`。复制的是当前磁盘文件，包含尚未提交的修改，不是原仓库某个历史提交的导出。

## 保留范围

- 完整 `src`、`cmake` 和 `external` 源码/资源目录，保持相对路径。`main.cpp` 仍包含 Python 模块头文件，因此本次没有按当前链接目标裁剪这些模块。
- 完整 `third_party/runtime`：COLMAP、SuperResolution、清单及来源记录。运行载荷约 73 MiB，CMake 对其文件集合、大小及 SHA-256 做校验。
- 根 `CMakeLists.txt`、vcpkg 两个清单、`.github/overlays`、格式和换行约定、许可证。
- 六个当前交付脚本、20 个正式配置 JSON、配置及交付文档。
- 目标目录原有 `.git`；没有复制来源仓库的 Git 历史。

## 排除范围

`THEORY`、`ReferenceMethods`、`paper`、`tests`、`archive`、`eval`、`ConclusionTXT`、`PoseRefine`、`mesh_scan`，所有旧 `GS_V*` 和 `SwapTexture` 成品目录、压缩包、构建树、日志、缓存、IDE/Agent 状态、旧 planning 记录、无关文档和旧 `build_lichtfeld.ps1` 工作站构建入口。配置目录不包含 `V1.1.5` 及未用于正式打包的 baseline/汇总 preset。

`.gitignore` 重新整理为本工作区的构建和输出规则，避免旧的全局 `*.py` 规则漏掉必要 Python 脚本。

## 必要适配

1. 活跃的 `swaptexture_m6_package_verify.py` 从旧 `archive` 提升到 `scripts`，修正根目录定位和默认运行清单路径；其包校验逻辑不变。
2. 分发脚本两处校验器路径随之更新。在未包含 `tests/python` 时明确跳过开发单元测试；安装包审计和实际程序的受限 PATH 启动检查继续执行。
3. 重写入口 README 并整理四份交付说明，使用实际程序名 `SwapTexture.exe`，说明 CMake 目标仍为 `Run-GS`、运行数据位于 `bin`、Release 关闭帮助输出。

业务源码、CMake 构建逻辑、vcpkg 依赖声明和正式 JSON 参数均保持迁移时内容。本文不包含原工程的历史验收结果。

## 验证边界

逐文件哈希清单与本轮检查记录保存在来源工程的 `.planning/2026-09-29-gs`，不作为程序运行文件复制。完成后记录在下方。新目录首次正式构建应从自己的空构建缓存开始，不能复用旧 `CMakeCache.txt`。

待完成本轮复制与独立配置验证。
