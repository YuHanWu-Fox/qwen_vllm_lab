# [CUDA][DBO] Support AG/RS all-to-all for expert parallelism

Enable Dual Batch Overlap for MRV2 CUDA deployments using
`allgather_reducescatter`: single node, DP+EP, TP1, modular MoE, regular decode,
and standard NCCL collectives. Eager dispatch and combine can now overlap the
other microbatch's expert compute. CUDA Graph capture keeps the collectives on
the capture stream.

This implements the CUDA backend work proposed in vllm-project/vllm#65121.
The feature diff uses the existing synchronous AG/RS handoff and records
buffers on each stream that uses them. It leaves the async prepare/finalize
contract unchanged and lets NCCL schedule its own CTAs without setting a
DeepGEMM SM quota. Other all-to-all backends and the existing ROCm and TP-only
schedules retain their behavior.

The tested stack uses the common workspace sizing from vllm-project/vllm#65128
(`887dbc3ccf68`, MRV2 portion) and eager padding-only handling from
vllm-project/vllm#66734 (`a529be94c6d2`, `cb2eaee0876c`). This feature diff does
not duplicate their workspace or padding logic. It also works with the existing
prefill/decode thresholds, independently of the new CLI scope in
vllm-project/vllm#63706. The shared fixes are prerequisites; their authorship
belongs with their upstream changes.

### Validation

Runtime validation: upstream `81a8cb063e6247fe254d7caf7e98a6691d71022a` plus the
prerequisites and this feature, with the native nightly for that exact commit.
The review candidate also applies byte-for-byte to upstream `3521784d8c`;
the four intervening commits do not touch the feature or native build inputs.
An additional two-GPU smoke on that source revision completed 48 requests
with the automatic extensible KV cache and equal workspace sizes in both slots.
Two RTX 4090 GPUs on different NUMA nodes; NCCL uses `SHM/direct` without GPU
P2P. The following checks ran with `VLLM_GPU_SYNC_CHECK=error`:

```bash
pytest -q tests/kernels/moe/test_naive_dp_ep_dbo.py \
  tests/v1/worker/test_gpu_ubatch_slicing.py tests/v1/worker/test_workspace.py \
  tests/v1/worker/test_dbo_pure_tp.py tests/v1/worker/test_gpu_model_runner_v2.py
pytest -q -s tests/v1/distributed/test_dbo.py::test_dbo_cuda_ag_rs_prompt_logprobs
```

130 tests passed in the first command. The second passed a real two-GPU
`TRITON_MLA` test comparing prompt logprobs for uneven prefills, with FULL
decode CUDA Graph capture, tolerance 0.04 for BF16 rounding. A separate dispatch
probe with the full model observed split eager prefill and ordinary unsplit FULL
decode replays. The older `db8e4658e` prototype also recorded a GPU trace of
517/1,040 communication-stream AG/RS kernels overlapping Triton expert kernels
on each rank, and GSM8K 5-shot strict-match scores of 164/256 off, 167/256 on.
The trace and GSM8K results are from the older base, not a rerun on this stack.
The H200 and 8-GPU configurations mentioned in the RFC have not been tested.

Full checkpoint serving: `deepseek-ai/DeepSeek-V2-Lite-Chat` BF16 (revision
`85864749cd611b4353ce1decdb286193298f64c7`), 2,048 input/8 output tokens,
128 requests per trial, three seeds per arm and concurrency after warmup.
TP1, DP2+EP, Triton MoE and `TRITON_MLA`, 8,192 tokens/step, max 64 sequences
per rank, 3 GiB KV/rank, and no prefix cache. Both arms use `FULL_DECODE_ONLY`
and thresholds 512 (prefill) and 8,192 (decode); only one arm sets
`--enable-dbo`. Regular decode-only batches therefore remain unsplit. All
1,536 requests completed and no KV preemption was recorded.

| Max concurrency | Requests/s off → on | P99 TTFT (ms) off → on | Median TPOT (ms) off → on |
| ---: | ---: | ---: | ---: |
| 32 | 14.93 → 15.42 (+3.3%) | 2,104 → 1,970 (−6.4%) | 103.6 → 121.8 (+17.5%) |
| 64 | 15.97 → 17.07 (+6.9%) | 4,022 → 3,739 (−7.0%) | 313.2 → 297.5 (−5.0%) |

These results describe this prefill-heavy workload. At concurrency 32 the
throughput ranges overlap. Online TPOT includes interference from interleaved
prefill, and the measurements do not establish an advantage for splitting
pure decode. The default behavior with DBO disabled remains unchanged.

AI assistance was used for implementation, testing and drafting. The human
submitter should review every changed line, run the relevant tests and sign off
on the submission.
