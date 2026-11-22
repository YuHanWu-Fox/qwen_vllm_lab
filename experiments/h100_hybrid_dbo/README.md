# 双 H100 Hybrid DBO 验证

2026-11-22，在 Vast.ai 单机双 H100 80GB HBM3（SM90、NV6）上测试 `GDN_Mamba_DBO_Base`。父仓库基线是 `065e269b793e26a4dcfed4cdd4875651cc725892`，vLLM 子模块是 `214be07d6cb80f4b3dd74c48dd960251bc165bac`。采用 Model Runner V2、DP=2、TP=1、EP 和同步调度。

| 模型 | 权重及 revision |
| --- | --- |
| GDN | `Intel/Qwen3-Next-80B-A3B-Instruct-int4-AutoRound` (`c1123e9dc2de0a4ef1fef0ef791721bd04d2c202`) |
| Mamba2 | `ibm-granite/granite-4.0-h-tiny` (`791e0d3d28c86e106c9b6e0b4cecdee0375b6124`) |

PyTorch `2.14.0+cu130`，NCCL `2.30.7`，DeepEP `2.1.0+local`（`a56d6156febcd9976e55adc85b5155bfac9f28f8`），NIXL `1.5.2`。NIXL 使用 `UCX_NET_DEVICES=lo` 和 `UCX_TLS=tcp,cuda_copy,cuda_ipc,self`，GPU 数据路径由探针日志中的 `device(cuda_ipc/cuda)` 确认。 [DeepEP 环境](deepep-h100-base-214be07d6/environment.txt) · [NIXL 环境](nixl-ep-h100-base-214be07d6/environment.txt) · [UCX 探针](nixl-ep-h100-base-214be07d6/transport-gdn-loopback.log)

## 正确性

| 后端 | 模型和路径 | DBO on/off 配对 | 场景 | 完成请求 |
| --- | --- | ---: | ---: | ---: |
| DeepEP | Granite HT eager，Granite 和 Qwen LL eager + FULL decode | 6 | 159 | 4,860 |
| `nixl_ep` | Granite 和 Qwen eager + FULL decode | 4 | 106 | 3,240 |

两组均覆盖 `none`、`align`、请求内 prefill 切分、混合 batch、不同 batch 大小的纯 decode、prefix hit、内部 checkpoint 整步回退、空 rank、空 batch、第二微批 padding、默认阈值边界、FULL 捕获和实际回放。主矩阵记录了每 rank 的调度和回放信息，覆盖检查错误为 0。 [DeepEP 结果](deepep-h100-base-214be07d6/case-results.csv) · [NIXL 结果](nixl-ep-h100-base-214be07d6/case-results.csv)

Granite 的 `nixl_ep` 与 DeepEP LL 在对应 DBO 开关和缓存模式下的 token 和 logprobs 逐项相等。Qwen 的 `nixl_ep none` 和 `align` 生成 logprob A/B 差值 P99 分别为 0.415 和 0.403，DBO-off 同进程重跑的 P99 分别为 0.417 和 0.431。DeepEP 矩阵有 14/138 个贪心请求出现首处分叉，NIXL 矩阵有 11/92 个。分叉时两侧选出的 token 均在对侧 top 5。结果支持已测配置的功能结论，精度证据来自固定 token、logprobs 和运行路径。 [跨后端对比](nixl-ep-h100-base-214be07d6/cross-backend-logprobs.json) · [DeepEP 数值汇总](deepep-h100-base-214be07d6/summary.json) · [NIXL 数值汇总](nixl-ep-h100-base-214be07d6/summary.json)

四个 pytest 文件共 342 个不同用例通过。首次离线运行 271 个通过，71 个因缺少 `facebook/opt-125m` 或 `Qwen/Qwen3.5-0.8B` 的模型配置失败。缓存配置后重跑全部 71 个失败项，71 个通过。NIXL CUDA wheel import canary 另有 1 个通过。 [pytest 原始日志](deepep-h100-base-214be07d6/pytest-logs.zip) · [NIXL import 日志](nixl-ep-h100-base-214be07d6/pytest-import.log)

## 性能

在同一模型、后端、cache 模式和请求序列上切换 `enable_dbo`。用离线 `LLMEngine.step()` 测首 token 时间、TPOT 和吞吐，另跑一遍记录实际切分、padding、checkpoint 和 CUDA 图回放。完成 12 组 A/B、112 个配对负载、105,696 个计时请求。 [性能表](dbo-perf-h100-base-214be07d6/case-results.csv) · [原始统计和运行路径](dbo-perf-h100-base-214be07d6/summary.json) · [计时方法](dbo-perf-h100-base-214be07d6/method.txt)

Granite DeepEP HT 的大 prefill 组使用 8192 prefill 阈值。表内为开启相对关闭的首 token P50 变化，`none` 的区间来自正序与反序复测。

| 每 rank 的 prefill 负载 | `none` | `align` |
| --- | ---: | ---: |
| 8192（32×256） | +173% 至 +199% | +161% |
| 16384（64×256） | +37% 至 +48% | +45% |
| 24576（96×256） | −13.6% | — |
| 32768（128×256） | −20.6% 至 −22.9% | −15.3% |
| 32786（单长请求） | −17.2% 至 −29.5% | 约 530 ms，checkpoint 回退 |

32768 短请求合批的双卡 prompt 吞吐提升 26.0%–29.7%（`none`）和 18.1%（`align`）。在 DeepEP LL 和 `nixl_ep` 常规组，两模型均使用 512 prefill 阈值与 32 decode 阈值。32 请求/rank 的纯 decode 实际执行双微批 FULL 回放，输出吞吐下降 39.8%–44.9%。带 704-token prefill 波次的混合负载下降 33.2%–59.4%。Granite `nixl_ep none` 在 16384 单步预算下，大 prefill 的 8192 和 16384 各点首 token P50 上升 0.6%–6.3%。 [逐负载 A/B](dbo-perf-h100-base-214be07d6/case-results.csv)

- **Qwen DeepEP HT** 在 EP=2 时有 256 个本地专家/rank，当前单机 HT kernel 上限为 128。64 token/rank 的同几何通信探针触发断言。 [HT 探针](deepep-h100-base-214be07d6/transport-ht.log)
- **Qwen DeepEP LL 和 NIXL** 在 8192 预算时每卡通信缓冲估算为 65 GiB。LL 的 4096 预算触发缓冲索引断言，2048 预算的开启侧在 DeepEP dispatch 分配时 OOM。完整性能 A/B 采用 1536 预算，早期正确性回归采用显存设置不同的 2048 预算。 [缓冲估算](dbo-perf-h100-base-214be07d6/backend-buffers.txt) · [失败日志和复跑](dbo-perf-h100-base-214be07d6/raw-arms.zip)
- **Granite NIXL** 在 32786 预算的关闭侧完成所有负载，开启侧在首个 8192-token 点为第二微批分配 10.01 GiB MoE 工作区时 OOM。两侧改用 16384 预算完成对照。 [运行日志](dbo-perf-h100-base-214be07d6/matrix.log) · [失败与复跑记录](dbo-perf-h100-base-214be07d6/raw-arms.zip)

## Granite 8192-token 诊断

每 rank `32×256`，在请求边界切成两个 `16×256` 微批。串行对照沿用正常 DBO 的切片和 metadata，以一个线程依次执行两片。下表的 TTFT P50 来自 Nsight 开始采集前的计时。

| Cache | 整批前向 | 同切分串行 | 正常 DBO |
| --- | ---: | ---: | ---: |
| `none` | 132.20 ms | 203.81 ms | 404.14 ms |
| `align` | 146.11 ms | 211.62 ms | 409.77 ms |

串行对照从进入 step 到首个 DeepEP layout kernel 约 12 ms（`none`）和 24 ms（`align`），这一窗口包含 metadata 及第一层前半段。正常 DBO 中每 rank 的 dispatch 和 combine 各从 40 次增到 80 次，记录到 240 次线程交接。GPU 空闲且下一次 kernel 尚未由 CPU 提交的时间从整批的约 10–12 ms 增到约 202–241 ms，首个 layout 到最后一次 combine 的区间就占约 179–215 ms。通信与计算重叠约 3–7 ms，Mamba kernel 累计时间约 10.5–11.2 ms。主要增量落在模型逐层推进时的主机提交链路。该 trace 的粒度为整条链路。 [诊断摘要](granite-ht-8192-214be07d6/diagnostic-summary.txt) · [逐步指标](granite-ht-8192-214be07d6/diagnostic-summary.json) · [阶段耗时](granite-ht-8192-214be07d6/8192-timing.csv)

`align` 下的单条 8192-token 请求触发 checkpoint 回退，整批路径约 143 ms。32768 token/rank 的短请求合批在 `none` 下从约 487 ms 降到 406 ms。随微批计算变长，trace 观察到通信与计算重叠增加。 [负载对照](granite-ht-8192-214be07d6/diagnostic-summary.txt)

## 原始证据与复现

各组的 `raw-arms.zip` 保留按 arm 和 rank 分组的 JSON 和运行日志，性能组还包括命令和 GPU 遥测。Granite 诊断每个 ZIP 保留 rank 结果、日志和 8192-token 场景可供 `analyze.py` 重算的精简 Nsight SQLite，包含 CUDA kernel、CUDA API、NVTX、memcpy 和 memset 时间线。其他负载的 trace 汇总保留在原始 `diagnostic-summary.json`。完整性校验文件为 [SHA256SUMS](SHA256SUMS)。

在已安装实测依赖的 checkout 中，可先把归档解压到临时目录并重新生成表格。以下命令从父仓库根目录执行。

```bash
export DBO_VLLM_DIR="$PWD/vllm-v0.33"
py="$DBO_VLLM_DIR/.venv/bin/python"
record="$PWD/experiments/h100_hybrid_dbo"
(cd "$record" && sha256sum -c SHA256SUMS)
scratch=$(mktemp -d)
cp -R "$record/." "$scratch/"
for group in deepep-h100-base-214be07d6 nixl-ep-h100-base-214be07d6 dbo-perf-h100-base-214be07d6; do
    unzip -q "$scratch/$group/raw-arms.zip" -d "$scratch/$group"
done
deepep="$scratch/deepep-h100-base-214be07d6"
nixl="$scratch/nixl-ep-h100-base-214be07d6"
perf="$scratch/dbo-perf-h100-base-214be07d6"
trace="$scratch/granite-ht-8192-214be07d6"
uv run --offline --no-project --python "$py" "$py" "$deepep/summarize_deepep.py" "$deepep"
uv run --offline --no-project --python "$py" "$py" "$nixl/summarize_nixl.py" "$nixl"
uv run --offline --no-project --python "$py" "$py" "$perf/summarize.py" --directory "$perf"
for archive in "$trace"/*.zip; do
    unzip -q "$archive" -d "$trace"
done
uv run --offline --no-project --python "$py" "$py" "$trace/analyze.py" \
    --output "$trace/recomputed.json" --phase-csv "$trace/recomputed-timing.csv"
```

GPU 复跑时用 `DBO_MODELS_DIR` 指定已缓存模型的父目录，用 `DBO_EP_RUNTIME_DIR` 指定包含 `build/nvshmem/lib` 与 `flashinfer-kernels` 的目录。两者默认取本次 H100 的路径。各 arm 的参数可从归档中的 `command.txt` 或 `ENGINE_START` 读取。下面是三个单组复跑示例。

```bash
USE_FLASHINFER_KERNEL_WHEELS=1 ARM_SUFFIX=-recheck \
    "$deepep/run_arm.sh" gdn ll none on --max-batched-tokens 2048
ARM_SUFFIX=-recheck "$nixl/run_arm.sh" mamba2 align on --max-batched-tokens 2048
"$perf/run_arm.sh" mamba2 ht none off large -recheck \
    --max-batched-tokens 32786 --max-model-len 33792 --prefill-threshold 8192
"$trace/run.sh" none serial -recheck --trace --only-8192
```

单测在 vLLM 源码目录运行。离线环境先缓存测试使用的两个 Hugging Face 模型配置。

```bash
cd "$DBO_VLLM_DIR"
uv run --offline --no-project --python "$py" "$py" -m pytest -q \
    tests/v1/worker/test_gpu_ubatch_slicing.py \
    tests/v1/attention/test_gdn_metadata_builder.py \
    tests/v1/attention/test_mamba_metadata_builder.py \
    tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
```
