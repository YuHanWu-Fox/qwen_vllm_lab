# 双卡 A800-NVLink 的 Hybrid DBO 验证交接

用 `GDN_Mamba_DBO_Base` 验证 GDN 和 Mamba2 的 token 粒度 DBO，整理正式 PR 的正确性与性能证据。父仓库和 `vllm-v0.33` 子模块均有同名分支，子模块以父仓库记录的提交为准。`GDN_Mamba_DBO_agrs` 是双 4090 上的 CUDA AG/RS 实验分支，本文的旧数据来自该分支。

- A800 Base 的 GDN 修复提交 `214be07d6cb80f4b3dd74c48dd960251bc165bac`，基于 `90ee31022e`。
- 双 4090 AGRS 的对应修复提交 `e5821eb41cb0942dc252c5be4f26e34be573557e`，基于 `4672e58763`。

## 检出与环境

新服务器从远端检出 Base，并用仓库的锁文件初始化源码环境。已有检出时，更新父仓库及其固定的子模块指针即可。详细环境管理见 [环境与初始化](later.md)。

```bash
git clone --branch GDN_Mamba_DBO_Base --recurse-submodules \
  https://github.com/YuHanWu-Fox/qwen_vllm_lab.git
cd qwen_vllm_lab
git rev-parse HEAD
git submodule status
git -C vllm-v0.33 rev-parse HEAD
uv run --no-project --python 3.12.13 scripts/sync_envs.py project --dry-run
uv run --no-project --python 3.12.13 scripts/sync_envs.py project
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv
```

在子模块目录通过 `uv run --offline --no-project --python .venv/bin/python .venv/bin/python` 检查 `vllm.__file__`、PyTorch/CUDA/NCCL 版本，以及两个方向的 `torch.cuda.can_device_access_peer`。记录 GPU 间 `NV#` 拓扑、实际 EP 后端及其版本，并保留首个双 rank 运行的通信日志。

CUDA Base 的 DBO 配置接受 `deepep_high_throughput`、`deepep_low_latency` 和 `nixl_ep`。先在目标镜像上验证选中的后端有 SM80 kernel，再跑一个实际 DP2 + EP 的 dispatch/combine。已有 [仓库说明](README.md) 记录了默认 DeepEP 源码在 SM80 上的构建限制。能运行的 `deepep_high_throughput` 用于 eager。FULL CUDA Graph 另用通过捕获和重放验证的 `deepep_low_latency` 或 `nixl_ep`。记录每组使用的后端，分别在同一后端内比较 DBO 开关。

## GDN 修复与回归

`align` 下某个 DP rank 的第二个 eager microbatch 全是 padding 时，Hybrid runner 会给它构造 null block 的 dummy decode。多个 GDN cache group 使用相同 spec 时，缺少各组显式工作索引会让 metadata 复用进入 `GDNAttentionMetadataBuilder.update_block_table()`，触发 `mamba_aligned_state_indices is not None` 断言。修复在 `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` 给每组传入它自己的 dummy null state index。

在子模块目录运行回归。测试依赖涉及 `pytest` 和 `tblib`，缺少时可按 `requirements/test/cuda.in` 为项目虚拟环境补齐所需包。

```bash
cd vllm-v0.33
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
uv run --offline --no-project --python .venv/bin/python .venv/bin/python \
  -m pytest -q tests/v1/worker/test_gpu_ubatch_slicing.py::test_hybrid_ubatches_continue_state
```

`[empty_prefill-align]` 回归含两个同 spec 的 GDN 组和一个 Mamba2 组，验证 padding 的 slot mapping、null state index 和缓存内容，同时检查真实请求的状态和输出。

合入时在 4090 上，Base 和 AGRS 各通过这组 10 项回归。AGRS 的 ubatch slicing、GDN/Mamba2 metadata 与 partial prefix cache 扩展回归共通过 344 项。

## 模型与双卡驱动

两种模型均用本地配置和 BF16 随机权重。用 `LLM(load_format="dummy", dtype="bfloat16", seed=2701, skip_tokenizer_init=True)`，直接提交 `prompt_token_ids`，输出使用 `detokenize=False`。下列代码可在项目环境生成两份 `config.json`。

```python
from pathlib import Path

from transformers import GraniteMoeHybridConfig, Qwen3NextConfig

cfg_root = Path("/tmp/hybrid-dbo-configs")
layers = ["linear_attention"] * 3 + ["full_attention"]

Qwen3NextConfig(
    architectures=["Qwen3NextForCausalLM"],
    hidden_size=2048,
    num_hidden_layers=4,
    layer_types=layers,
    num_experts=16,
    num_experts_per_tok=4,
    max_position_embeddings=33792,
).save_pretrained(cfg_root / "gdn")

GraniteMoeHybridConfig(
    architectures=["GraniteMoeHybridForCausalLM"],
    hidden_size=2048,
    intermediate_size=512,
    shared_intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=16,
    num_key_value_heads=8,
    layer_types=layers,
    num_local_experts=16,
    num_experts_per_tok=4,
    mamba_expand=2,
    mamba_n_heads=64,
    mamba_n_groups=8,
    mamba_d_head=64,
    mamba_d_state=64,
    mamba_chunk_size=128,
    position_embedding_type="rope",
    max_position_embeddings=33792,
).save_pretrained(cfg_root / "mamba2")
```

用 `torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0` 启动用例，通过 `distributed_executor_backend="external_launcher"`、`data_parallel_size=2`、`tensor_parallel_size=1`、`enable_expert_parallel=True`、`async_scheduling=False` 驱动 `LLMEngine.step()`。设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 便于在测试脚本里记录 runner 的调度结果。两组采用相同的已验证 EP/MoE 后端、请求 token、权重种子和采样参数，仅切换 `enable_dbo`。先跑 GDN，再跑 Mamba2。

`none` 用 `enable_prefix_caching=False`，`align` 用 `enable_prefix_caching=True` 和 `mamba_cache_mode="align"`。初始 attention `block_size` 可沿用 GDN 256、Mamba2 512。断言和用例生成均读取设备上实际生效的 Mamba block size 及其 checkpoint 能力。`FULL_DECODE_ONLY` 组指定 decode capture sizes，并验证真实 replay。Eager 组可用 `enforce_eager=True`。

## 正确性验收

逐步记录两 rank 的实际 token 数、padding 后大小、每请求起点和长度、microbatch 切片、`num_ubatches`、`has_mamba_prefill_checkpoint`、图模式及 capture/replay 事件。可从 `dispatch_cg_and_sync_dp`、`UBatchRunner.prepare/run` 和 CUDA Graph replay 入口采集。正确性组用 `VLLM_GPU_SYNC_CHECK=error` 检查路径，在读取 GPU 状态做比较前显式同步。

| 用例 | 验收信号 |
| --- | --- |
| 长 prefill 跨切点、prefill 和 decode 混合 | `num_ubatches=2`，至少一个请求在两片都有有效 token。比较 DBO on/off 的有效 token logprobs、输出和后续 decode。混合步先推进已有 decode，再插入长 prefill。 |
| 纯 decode、多 batch 大小 | 记录 eager 与 FULL 的实际执行。覆盖对称 batch 和 `4/8`、`5/8` 等不对称形状，短 batch 可用 decode 阈值 2，FULL 捕获 2/4/8/16。分别验证第二片有 0 或 1 个有效 token 的形状。 |
| GDN `align` 的 eager 全 padding | 临时用 prefill 阈值 32。两 rank 各提交一条，令短 rank token 数达到阈值但不超过另一 rank 的一半，且长 prompt 小于有效 cache block size。4090 示例是 `71/759`。短 rank 两片均参加 collective，第二片有效 token 为 0。 |
| `align` prefix hit | 预热长度 `B + 17` 的请求，其中 `B` 是有效 cache block size。提交共享前 `B` tokens 的请求并断言 `num_cached_tokens == B`，再同 `skip_reading_prefix_cache=True` 的同输入比较。 |
| Mamba2 `align` 内部 checkpoint | 冷 prefill 用 `B + 89`，同一步另一 rank 用较短、已过阈值的 prompt。需要导出的 rank 否决切分，两个 rank 都用单 microbatch。验证 on/off 的该请求均一步完成，后续两组都命中 `B` tokens。若 GDN 的实际后端具备内部导出能力，覆盖同样的验收。 |
| 空 batch | 一侧 rank 无请求时确认两边都走单 microbatch，空 rank 不采样。两 rank 均空时无样本或 cache 更新。 |
| 默认阈值 | prefill 用两 rank 实际 token 数 `511/512` 和 `512/512`，纯 decode 用 `31/32` 和 `32/32`。检查 DP 最小值达到 `512` 或 `32` 时才会切分，且所有 rank 均同意。大型性能组另测阈值 8192。 |

需要内部 checkpoint 的 step 先按 cache spec 和位置做调度，再统一回退到整步 forward。`has_mamba_prefill_checkpoint` 只在启用 ubatching 时报告回退信号，验证 off 是否导出可查看后续 prefix hit。把 scheduler step 数与每步的 microbatch 数分开记录。

## 性能复测

关闭 GPU 同步调试和状态采样钩子。固定两 rank 的 CPU 线程数、KV cache 预算、模型长度及请求到达顺序。先做同后端 A/B 小 batch，再扩大到每 rank 8192、16384 和 32768 prompt tokens。每个规模跑 `32/64/128 × 256` 短请求合批及同总 token 数的单长请求。用 `numpy.random.default_rng(2701 + rank)` 从 `[10, 20000)` 生成 token。冷 prefill 每请求用贪心采样生成 1 token，设置 `ignore_eos=True` 和 `skip_reading_prefix_cache=True`。

4090 大 batch 参考参数是 `dbo_prefill_token_threshold=8192`、`max_num_batched_tokens=32768`、`max_model_len=33792`、`max_num_seqs=129`、每 rank KV cache 2 GiB。每组预热 6 次、测量 16 次。提交所有请求后同步 GPU 和 DP ranks，记录从首次 `LLMEngine.step()` 到整批出首 token 的耗时，每次取较慢的 rank，报告 P50/P95 和双卡 prompt tokens/s。再按需要补充混合负载及纯 decode 的逐步延迟、TPOT 和吞吐。若 EP 后端有 dispatch 容量上限，记录实际支持的规模以及报错配置。

## 已有双 4090 证据

CUDA AG/RS 实验环境经 PCIe 走 NCCL SHM，两卡 CUDA P2P 不可用。缩减模型覆盖 `none`、`align`、长 prefill、混合、decode、prefix hit、checkpoint 回退、空 batch、eager 和 FULL replay。对照矩阵共完成 832 个请求，另外用默认阈值完成 470 个边界请求。下表是阈值 8192 时的 P50 变化，负数表示时延降低。

| 模型 / Cache | 每卡 `64×256` | 每卡 `128×256` | 每卡 `1×16384` | 每卡 `1×32768` |
| --- | ---: | ---: | ---: | ---: |
| GDN `none` | −19.1% | −21.8% | −18.1% | −18.9% |
| GDN `align` | −18.1% | −20.9% | −16.0% | −17.4% |
| Mamba2 `none` | −19.4% | −22.8% | −20.6% | −23.6% |
| Mamba2 `align` | −17.6% | −21.3% | −0.2% | −0.1% |

GDN `align` 在 4090 上用 Triton/FLA，实际 cache block 为 768。两个长请求分别调度为 `16128 + 256` 和 `32256 + 512`，仅首段触发 DBO。Mamba2 `align` 两个长请求均一步导出 checkpoint，回退到单 microbatch。复测在 A800 上以实际 kernel、调度步和 EP 后端重新归因。

交付时保留环境与拓扑日志、两仓库 SHA、每个 case 的逐 rank 执行记录、DBO on/off 正确性对比及性能 CSV。PR 中按实测区分 eager、FULL replay 与 checkpoint 回退。
