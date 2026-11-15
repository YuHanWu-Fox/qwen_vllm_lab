# qwen_vllm_lab

使用 Qwen 模型验证 vLLM 优化。

- 开发源码：vllm-v0.33 子模块
- 开发分支：project/v0.33
- 基线分支：baseline/v0.33
- 基线提交：configs/baseline.commit
- 基线工作区：.worktrees/vllm-baseline
- 两个工作区分别使用自己的 .venv

当前源码基于官方 `v0.33.0`（`4ba787ec9959e566d3b587a796dff07673324da5`）。
开发目录、子模块名称和编辑器配置统一使用 `vllm-v0.33`。

官方 v0.33 已包含原补丁 `8d02c91f2d` 的 INC/PLE 排除逻辑和回归测试。
新分支通过合并保留原补丁的提交历史，源码使用官方 v0.33 实现，包括更新后的默认回退行为。
原 `project/v0.31`、`baseline/v0.31` 和 `study/v0.31` 分支继续保留。

切换 Git 分支不会更新 `.venv` 中的依赖或编译扩展，使用下文的同步脚本更新环境。
`configs/precompiled.commit` 记录匹配的官方源码提交；依赖统一由外层 `uv.lock` 管理。

## 统一的依赖锁定

外层 `pyproject.toml` 声明运行依赖和开发、构建工具，`uv.lock` 锁定完整的传递依赖、
下载来源和文件哈希。`.python-version` 固定 Python 3.12.13。
当前安装目标为 Linux x86_64、Python 3.12、CUDA 13.0；PyTorch 包使用官方 cu130 索引。
宿主机的 NVIDIA 驱动、CUDA 开发工具链及模型权重不由 `uv.lock` 安装或管理。

两套环境使用同一份第三方依赖，并分别 editable 安装各自的 vLLM 源码。
`uv.lock` 中的 `vllm==0.33.0` 提供官方依赖元数据和匹配的编译组件信息；实际运行的
源码版本由子模块指针及 `configs/baseline.commit` 固定，本地安装版本可以带有 dev 后缀。
安装脚本跳过发行版 vLLM 包，从锁文件选择并校验官方 wheel，再用本地源码替换为 editable 安装。
vLLM 的构建工具也包含在锁文件中，editable 安装使用 `--no-deps --no-build-isolation`，
避免重新解析或升级已锁定依赖。

不要将两份同名 vLLM 声明为 uv workspace 的两个成员。这里保留两个独立 `.venv`，
通过下面的脚本指定安装目标。

## 新服务器初始化

安装 Git、curl、uv，并确认宿主机驱动支持 CUDA 13.0。如果使用 SSH 网络隧道，
先配置当前终端的 `ALL_PROXY` 等环境变量。

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/YuHanWu-Fox/qwen_vllm_lab.git
cd qwen_vllm_lab

# 保留外层仓库固定的源码提交，再关联远端分支。
git -C vllm-v0.33 switch -c project/v0.33
git -C vllm-v0.33 branch --set-upstream-to=origin/project/v0.33

git -C vllm-v0.33 remote add upstream https://github.com/vllm-project/vllm.git
git -C vllm-v0.33 fetch --no-tags upstream tag v0.33.0

mkdir -p .worktrees
git -C vllm-v0.33 worktree add -b baseline/v0.33 \
  ../.worktrees/vllm-baseline "$(cat configs/baseline.commit)"
git -C .worktrees/vllm-baseline branch --set-upstream-to=origin/baseline/v0.33

# 先查看计划，再安装。下载和安装可能需要较多磁盘空间及时间。
uv run --no-project --python 3.12.13 scripts/sync_envs.py all --dry-run
uv run --no-project --python 3.12.13 scripts/sync_envs.py all
```

以上 Git 初始化命令用于首次 clone，已有分支和 worktree 时无需重复创建。

## 日常使用

在外层项目根目录执行：

```bash
# 分别同步，或同时同步。操作会将所选环境更新为锁定依赖。
uv run --no-project --python 3.12.13 scripts/sync_envs.py project
uv run --no-project --python 3.12.13 scripts/sync_envs.py baseline
uv run --no-project --python 3.12.13 scripts/sync_envs.py all

# 使用已同步的环境运行同一个验证脚本。
vllm-v0.33/.venv/bin/python scripts/run_qwen.py
.worktrees/vllm-baseline/.venv/bin/python scripts/run_qwen.py

# 检查锁文件；主动升级第三方依赖时再更新锁文件。
uv lock --check
uv lock --upgrade
```

`run_qwen.py` 的模型路径目前是 `~/vllm-projects/models/Qwen3-0.6B`，
服务器运行前需要准备模型并调整为实际路径。

单独执行普通 `uv sync` 或 `uv run` 会使用外层默认环境和发行版 vLLM，
进行源码实验时使用上面的同步脚本和明确的解释器路径。
两份源码若修改依赖声明或 C++、CUDA、Rust 构建代码，需要更新锁定方案或重新编译；
脚本会检查这些文件，阻止混用不匹配的官方 wheel。
修改源码版本、依赖配置后，将子模块指针、基线记录、`pyproject.toml`、`.python-version`
和 `uv.lock` 的相应改动提交到外层 Git 仓库。

同步脚本从 `.gitmodules` 的 `submodule.vllm-v0.33.path` 读取开发源码位置。
今后修改子模块目录名时使用 `git mv`，并重新同步开发环境以更新 editable 路径。
