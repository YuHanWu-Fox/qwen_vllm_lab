# qwen_vllm_lab

使用 Qwen 模型验证 vLLM 优化。

- 开发源码：vllm-v0.31 子模块
- 开发分支：project/v0.33
- 基线分支：baseline/v0.33
- 基线提交：configs/baseline.commit
- 基线工作区：.worktrees/vllm-baseline
- 两个工作区分别使用自己的 .venv

当前源码基于官方 `v0.33.0`（`4ba787ec9959e566d3b587a796dff07673324da5`）。
保留 `vllm-v0.31` 目录名以兼容已有路径，实际版本以提交记录为准。

官方 v0.33 已包含原补丁 `8d02c91f2d` 的 INC/PLE 排除逻辑和回归测试。
新分支通过合并保留原补丁的提交历史，源码使用官方 v0.33 实现，包括更新后的默认回退行为。
原 `project/v0.31`、`baseline/v0.31` 和 `study/v0.31` 分支继续保留。

切换 Git 分支不会更新 `.venv` 中的依赖或编译扩展。从 v0.31 升级时，两个环境都需要
按 v0.33 的依赖重新安装；`configs/precompiled.commit` 记录匹配的官方源码提交。
`requirements/constraints-cu130-py312.txt` 是旧 v0.31 环境快照，不能用于安装 v0.33
（v0.33 使用 PyTorch 2.14.0，旧快照固定为 2.13.0）。
