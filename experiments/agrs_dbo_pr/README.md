# CUDA AG/RS DBO: submission candidate and serving validation

The `GDN_Mamba_DBO_agrs` submodule includes the CUDA AG/RS backport for the
v0.34.0 Hybrid DBO branch. Use the pinned submodule revision for this experiment
branch. The patch files and serving results below refer to the original
upstream-main candidate.

The candidate targets vLLM MRV2 on upstream `main` at
[`3521784d8c98b04d54947edbb3704005e36bfffb`](https://github.com/vllm-project/vllm/commit/3521784d8c98b04d54947edbb3704005e36bfffb)
(November 18, 2026, 11:05 UTC), in `/root/autodl-tmp/agrs-dbo-pr-candidate`.
The measured A/B and complete regression used
[`81a8cb063e6247fe254d7caf7e98a6691d71022a`](https://github.com/vllm-project/vllm/commit/81a8cb063e6247fe254d7caf7e98a6691d71022a)
in `/root/autodl-tmp/agrs-dbo-submit`. The four upstream commits between the
two bases did not touch the candidate's files or native build inputs. Both
patch files apply byte-for-byte on either base.

Apply the two patches in this order in a clean vLLM worktree:

```bash
artifacts=/root/qwen_vllm_lab/experiments/agrs_dbo_pr
base=3521784d8c98b04d54947edbb3704005e36bfffb
git worktree add --detach /path/to/new/vllm "$base"
cd /path/to/new/vllm
git apply --index "$artifacts/upstream-prerequisites.patch"
git apply "$artifacts/cuda-agrs-dbo.patch"
```

`git diff` now shows the feature alone; `git diff --cached` shows the prerequisites.
Both prerequisite PRs were open when the candidate was prepared:

- [#66734](https://github.com/vllm-project/vllm/pull/66734): `a529be94c6d2` and `cb2eaee0876c` provide padding-only handling and keep MLA from reserving locked workspace. The Quark, multimodal and ROCm-only follow-ups are outside this local backport.
- [#65128](https://github.com/vllm-project/vllm/pull/65128): the MRV2 and common parts of `887dbc3ccf68` size each scratch workspace lane across microbatches during the memory profile. The candidate removes its own extra dummy forwards.

Use `cuda-agrs-dbo.patch` for feature review. It enables single-node TP1, DP+EP,
modular MoE with standard NCCL; gives the CUDA allocator ownership information
for the cross-stream tensors; preserves the existing ROCm path and TP-only DBO
schedule; and keeps captured collectives on the capture stream. It does not add
[#63706](https://github.com/vllm-project/vllm/pull/63706)'s new scope switch.
Once the prerequisites land, apply only the feature patch to a fresh upstream
base and rerun the regression. Until then, publish or review it as a dependent
change with attribution to those PRs, not as a standalone validated patch.

## Online serving results

DeepSeek-V2-Lite-Chat revision `85864749cd611b4353ce1decdb286193298f64c7`;
BF16; two RTX 4090 GPUs across NUMA nodes, without GPU P2P (NCCL `SHM/direct`);
TP1, DP2+EP, AG/RS,
Triton MoE, `TRITON_MLA`; PyTorch `2.14.0+cu132`, NCCL `2.30.7`. The native
extensions came from the nightly wheel at the exact source base
(`vllm-0.34.0rc1.dev787+g81a8cb063`, SHA-256
`25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9`).

Two otherwise identical `vllm serve` instances, run in off-then-on order,
received random prompts of 2,048 input and 8 output tokens from `vllm bench serve`.
Each arm and concurrency used three trials of 128 requests (seeds 42/43/44),
with a separate warmup of one request per concurrency slot before each trial.
The scheduler used 8,192 batched tokens, 64 sequences per rank, 4,096-token
context, no prefix cache, and 3 GiB of KV cache per rank in both arms.
The only arm flag was `--enable-dbo`. The thresholds were 512 for prefill and
8,192 for decode, with `FULL_DECODE_ONLY` graphs for 1, 2, 4, 8, 16, 32 and 64
requests. With regular decoding, a pure decode batch cannot meet the decode
threshold, while an eligible step with prefill can split.

Values are medians across the three trials. TTFT is time to first token, TPOT is
time per output token after the first, and E2EL is end-to-end request latency.

| Concurrency | Requests/s (off → on) | P99 TTFT, ms (off → on) | Median TPOT, ms (off → on) | P99 E2EL, ms (off → on) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 14.93 → 15.42 (+3.3%) | 2,104 → 1,970 (−6.4%) | 103.6 → 121.8 (+17.5%) | 2,790 → 2,649 (−5.0%) |
| 64 | 15.97 → 17.07 (+6.9%) | 4,022 → 3,739 (−7.0%) | 313.2 → 297.5 (−5.0%) | 6,462 → 6,117 (−5.3%) |

All 1,536 requests completed and no KV preemption was recorded, including warmup.
Both arms logged runtime `FULL` decode replays. At concurrency 64 all three
paired seeds improved requests/s. At 32 the throughput ranges overlap
(14.78–15.28 off, 14.94–15.76 on), and one paired seed was slower. Online TPOT
includes time spent waiting on interleaved prefill work, so it can change even
with identical pure decode graph policy. These results apply to this synthetic
prefill-heavy workload and topology. They do not show a speedup for split
pure decode; the earlier large-decode experiment is separate.

- [Trial summaries](serving-trials.jsonl), [aggregates and ranges](serving-summary.json), [raw JSON, commands and server logs](serving-raw.zip)
- [Benchmark driver](run_prefill_serving.py) (run with vLLM's Python environment; in this checkout, `CUDA_HOME` and `LD_LIBRARY_PATH` point to the PyTorch CUDA 13.2 toolkit)

```bash
.venv/bin/python "$artifacts/run_prefill_serving.py" --repo "$PWD" \
  --model /root/autodl-tmp/models/DeepSeek-V2-Lite-Chat \
  --output /path/to/new/serving-results
```

The diagnostic `probe-batch-dispatch.patch` logs the actual DP decision for
real steps. Apply it only for a separate probe, add `--arms on --concurrency 32
--repeats 1 --requests 64 --probe` to the benchmark command and use a new output
path, then reverse the diagnostic patch. Its timings are excluded from the
comparison above. `--probe` also enables `VLLM_GPU_SYNC_CHECK=error`. The
[probe results](dispatch-probe.json) counted 13 split prefill or mixed steps on
each rank. Pure decode had 18/19 `FULL` replays (ranks 0/1), plus three eager
fallback steps per rank; all pure decode steps were unsplit.

## Regression commands

Select the native nightly for the pinned commit, and use the vLLM test
dependencies. Both commands ran with `VLLM_GPU_SYNC_CHECK=error`. If the host
uses an HTTP proxy, set `NO_PROXY=127.0.0.1,localhost` for the server test.

```bash
.venv/bin/python -m pytest -q \
  tests/kernels/moe/test_naive_dp_ep_dbo.py \
  tests/v1/worker/test_gpu_ubatch_slicing.py \
  tests/v1/worker/test_workspace.py \
  tests/v1/worker/test_dbo_pure_tp.py \
  tests/v1/worker/test_gpu_model_runner_v2.py
.venv/bin/python -m pytest -q -s \
  tests/v1/distributed/test_dbo.py::test_dbo_cuda_ag_rs_prompt_logprobs
```

The five-file regression passed 130 tests. The two-GPU integration test passed
its prompt-token logprob comparison for uneven requests with `--enable-dbo` on
and off. The serving benchmark keeps pure decode unsplit, while the integration
test also has a capture configuration eligible for split decode. The CUDA
capability test failed as expected when run against the original implementation.
On `3521784d8c` a second two-GPU smoke test used automatic extensible KV sizing
at `--gpu-memory-utilization 0.80`. It completed 48 requests with uneven prompts;
each worker locked equal-sized workspaces for both microbatches and extended its
KV cache successfully. This smoke reused the `81a8cb063e` native extensions.
See the [regression and probe logs](regression-and-probe.zip) and the
[PR draft](PR_DRAFT.md).
