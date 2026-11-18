"""Large, rank-aligned MoE batches on the local patched vLLM checkout.

Run one GPU-pair A/B arm at a time; example:
  /tmp/agrs_dbo_main_env/bin/python -u bench_large.py --graph full --dbo off
"""
import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import random
from statistics import fmean, median
import subprocess
import time

os.environ.update({
    'VLLM_NO_USAGE_STATS': '1',
    'HF_HUB_OFFLINE': '1',
    'OMP_NUM_THREADS': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_USE_FLASHINFER_SAMPLER': '0',
    'VLLM_LOGGING_LEVEL': 'INFO',
    'NCCL_DEBUG': 'INFO',
    'NCCL_DEBUG_SUBSYS': 'INIT,ENV,GRAPH',
})

from vllm import AsyncEngineArgs, SamplingParams
from vllm.inputs import tokens_input
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.attention.backends.registry import AttentionBackendEnum

ROWS = []


class StepLog(StatLoggerBase):
    def __init__(self, vllm_config, engine_index=0):
        self.engine_index = engine_index

    def log_engine_initialized(self):
        pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None,
               engine_idx=None):
        cg = getattr(scheduler_stats, 'cudagraph_stats', None)
        if cg is None:
            return
        if iteration_stats is None:
            gen, prompt, itls, preemptions = 0, 0, [], 0
        else:
            gen = iteration_stats.num_generation_tokens
            prompt = iteration_stats.prompt_token_stats.computed
            itls = iteration_stats.inter_token_latencies_iter
            preemptions = iteration_stats.num_preempted_reqs
        ROWS.append({
            'rank': self.engine_index if engine_idx is None else engine_idx,
            'real_tokens': cg.num_unpadded_tokens,
            'padded_tokens': cg.num_padded_tokens,
            'mode': cg.runtime_mode,
            'gen_tokens': gen,
            'prompt_tokens': prompt,
            'itl_count': len(itls),
            'itl_s': fmean(itls) if itls else None,
            'preemptions': preemptions,
        })


def shape_counts(rows):
    return [dict(Counter(f'{r["mode"]}:{r["real_tokens"]}/{r["padded_tokens"]}'
                         for r in rows if r['rank'] == rank)) for rank in (0, 1)]


async def wave(engine, name, kind, scale, variant, prompt_length, output_length):
    if kind == 'decode':
        per_rank, n_input, n_output = scale, prompt_length, output_length
    else:
        n_input, n_output = 1024, 1
        assert scale % n_input == 0
        per_rank = scale // n_input
    rng = random.Random(42000 + 31 * scale + variant)
    prompts = [[rng.randrange(1000, 25000) for _ in range(n_input)]
               for _ in range(2 * per_rank)]
    params = SamplingParams(temperature=0, max_tokens=n_output,
                            ignore_eos=True, detokenize=False,
                            output_kind=RequestOutputKind.FINAL_ONLY)
    await engine.pause_generation(mode='keep', clear_cache=False)
    ROWS.clear()
    collectors = []
    # Rank interleaving and the scheduler hold ensure both queues are full at t0.
    for i in range(per_rank):
        for rank in (0, 1):
            collectors.append(await engine.add_request(
                f'{name}-r{rank}-{i}', tokens_input(prompts[2*i+rank]), params,
                data_parallel_rank=rank,
            ))
    t0 = time.perf_counter()
    await engine.resume_generation()
    outputs = await asyncio.wait_for(
        asyncio.gather(*(x.get() for x in collectors)), 180)
    wall = time.perf_counter() - t0
    rows = ROWS.copy()
    counts = shape_counts(rows)
    ranks = [sum(r['rank'] == n for r in rows) for n in (0, 1)]
    # A wave is valid only if every response completed with exactly its fixed token count.
    assert len(outputs) == 2 * per_rank and all(
        o.finished and len(o.outputs) == 1 and
        len(o.outputs[0].token_ids) == n_output and
        o.outputs[0].finish_reason == 'length' and
        len(o.prompt_token_ids) == n_input for o in outputs
    ), (name, 'unexpected output')
    assert not any(r['preemptions'] for r in rows), (name, rows[:2])
    result = {'name': name, 'kind': kind, 'tokens_per_rank': scale,
              'requests_per_rank': per_rank, 'input_tokens_per_request': n_input,
              'output_tokens_per_request': n_output,
              'wave_s': wall, 'step_shapes': counts, 'steps_per_rank': ranks,
              'rows': rows}
    if kind == 'decode':
        duration_by_rank = []
        decode_count = n_output - 1
        steady_steps = decode_count - 16
        assert steady_steps >= 8
        for rank in (0, 1):
            candidates = [r for r in rows if r['rank'] == rank and
                          r['real_tokens'] == r['padded_tokens'] == per_rank and
                          r['prompt_tokens'] == 0 and r['gen_tokens'] == per_rank and
                          r['itl_count'] == per_rank]
            if len(candidates) != decode_count:
                raise RuntimeError(
                    f'{name} rank {rank}: {len(candidates)} exact decode steps, '
                    f'expected {decode_count}; shapes {counts}; first rows {rows[:4]}')
            selected = candidates[8:-8]
            duration_by_rank.append(sum(r['itl_s'] for r in selected))
        # ITLs come from the engine timestamps, excluding prefill and phase edges.
        result['decode_steps_per_rank'] = steady_steps
        result['decode_steps_duration_rank_s'] = duration_by_rank
        result['mean_decode_step_ms'] = 1000 * max(duration_by_rank) / steady_steps
        result['tokens_per_s'] = 2 * per_rank * steady_steps / max(duration_by_rank)
    else:
        wanted = {'NONE:' + str(scale) + '/' + str(scale): 1}
        if any(dict(c) != wanted for c in counts):
            raise RuntimeError(f'{name} is not a single aligned prefill step: {counts}')
        result['tokens_per_s'] = 2 * scale / wall
    print(f'{name}: {result["tokens_per_s"]:.2f} tok/s, '
          f'wall={wall:.4f}s, shapes={counts}', flush=True)
    return result


async def main(args):
    opt = dict(
        model=args.model, trust_remote_code=True, dtype='bfloat16',
        max_model_len=4096, max_num_seqs=64, max_num_batched_tokens=args.max_batched,
        kv_cache_memory_bytes=2 * 1024**3, enable_extensible_kv_cache=False,
        enable_prefix_caching=False, data_parallel_size=2, enable_expert_parallel=True,
        all2all_backend='allgather_reducescatter', moe_backend='triton',
        attention_backend=AttentionBackendEnum.TRITON_MLA,
        enable_dbo=args.dbo == 'on', dbo_decode_token_threshold=32,
        dbo_prefill_token_threshold=512, cudagraph_metrics=True,
        enforce_eager=args.graph == 'eager',
    )
    if args.graph == 'full':
        opt['compilation_config'] = dict(mode=0, cudagraph_mode='FULL_DECODE_ONLY',
                                         decode_cudagraph_capture_request_sizes=[32, 64])
    print('config:', opt, flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**opt), stat_loggers=[StepLog])
    print('engine ready', flush=True)
    cases = []
    if args.phase in ('both', 'decode'):
        cases += [('decode', 32), ('decode', 64)]
    if args.phase in ('both', 'prefill'):
        cases += [('prefill', size) for size in (4096, 8192) if size <= args.max_batched]
    if not cases:
        raise ValueError('no case fits these settings')
    raw = { 'config': opt, 'graph': args.graph, 'dbo': args.dbo,
            'vllm_commit': subprocess.check_output(['git', '-C', args.repo, 'rev-parse', 'HEAD'], text=True).strip(),
            'warmups': [], 'measurements': [] }
    path = Path(__file__).parent / f'{args.label or (args.graph+"-"+args.dbo)}.json'
    def save():
        path.write_text(json.dumps(raw, default=str, indent=1))
    try:
        for i in range(args.warmups):
            for kind, scale in cases:
                raw['warmups'].append(await wave(engine, f'warm{i}-{kind}-{scale}',
                    kind, scale, 1000+i, args.prompt_length, args.output_length))
                save()
        for rep in range(args.reps):
            for kind, scale in cases[rep % len(cases):] + cases[:rep % len(cases)]:
                raw['measurements'].append(await wave(engine, f'meas{rep}-{kind}-{scale}',
                    kind, scale, rep, args.prompt_length, args.output_length))
                save()
        for k,s in cases:
            vals = [x['tokens_per_s'] for x in raw['measurements']
                    if (x['kind'],x['tokens_per_rank']) == (k,s)]
            print(f'{k} {s}: median {median(vals):.2f} tok/s; samples {vals}', flush=True)
        print('completed:', path, flush=True)
    finally:
        engine.shutdown(timeout=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='/tmp/agrs-dbo-upstream-check')
    parser.add_argument('--model', default='/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat')
    parser.add_argument('--graph', choices=['full','eager'], required=True)
    parser.add_argument('--dbo', choices=['off','on'], required=True)
    parser.add_argument('--phase', choices=['both','decode','prefill'], default='both')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--max-batched', type=int, default=8192)
    parser.add_argument('--prompt-length', type=int, default=64)
    parser.add_argument('--output-length', type=int, default=65)
    parser.add_argument('--label')
    asyncio.run(main(parser.parse_args()))
