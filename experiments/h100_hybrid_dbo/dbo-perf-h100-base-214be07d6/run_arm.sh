#!/usr/bin/env bash
set -euo pipefail

run_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir="${DBO_VLLM_DIR:-$(cd "$run_root/../../.." && pwd)/vllm-v0.33}"
runtime_root="${DBO_EP_RUNTIME_DIR:-/workspace/qwen_vllm_lab_runs/deepep-h100-base-214be07d6}"
model=$1
backend=$2
cache=$3
dbo=$4
suite=$5
tag=$6
shift 6
arm="$run_root/arms/$model-$backend-$cache-$suite-$dbo$tag"
if [[ -e "$arm/run.log" ]]; then
    printf 'Use a new tag for an existing arm: %s\n' "$arm" >&2
    exit 2
fi
mkdir -p "$arm"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0a
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
unset CUDA_LAUNCH_BLOCKING VLLM_GPU_SYNC_CHECK
if [[ "$backend" == ht || "$backend" == ll ]]; then
    export LD_LIBRARY_PATH="$runtime_root/build/nvshmem/lib:$source_dir/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "$backend" == ll ]]; then
    export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-8192}"
fi
if [[ "$backend" == nixl ]]; then
    export UCX_TLS=tcp,cuda_copy,cuda_ipc,self
    export UCX_NET_DEVICES=lo
fi
if [[ "$model" == gdn ]]; then
    export PYTHONPATH="$runtime_root/flashinfer-kernels${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$source_dir"
{
    date -u -Iseconds
    git rev-parse HEAD
    git status --short --branch
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
    nvidia-smi topo -m
    taskset -pc $$
    env | sort | rg '^(CUDA_HOME|CUDA_VISIBLE_DEVICES|VLLM_ENABLE_V1_MULTIPROCESSING|VLLM_NO_USAGE_STATS|VLLM_DBO_COMM_SMS|NVSHMEM_QP_DEPTH|NVSHMEM_VERSION|UCX_TLS|UCX_NET_DEVICES|OMP_NUM_THREADS|MKL_NUM_THREADS)='
} > "$arm/environment.txt"
printf '%q ' "$run_root/run_arm.sh" "$model" "$backend" "$cache" "$dbo" "$suite" "$tag" "$@" > "$arm/command.txt"
printf '\n' >> "$arm/command.txt"
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu --format=csv -l 1 > "$arm/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
timeout --kill-after=20s "${ARM_TIMEOUT:-15m}" \
    uv run --offline --no-project --python .venv/bin/python .venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0 \
    "$run_root/bench_dbo.py" --model "$model" --backend "$backend" \
    --cache "$cache" --dbo "$dbo" --suite "$suite" --output "$arm" "$@" > "$arm/run.log" 2>&1
