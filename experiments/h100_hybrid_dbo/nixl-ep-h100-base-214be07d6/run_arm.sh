#!/usr/bin/env bash
set -euo pipefail

run_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir="${DBO_VLLM_DIR:-$(cd "$run_root/../../.." && pwd)/vllm-v0.33}"
runtime_root="${DBO_EP_RUNTIME_DIR:-/workspace/qwen_vllm_lab_runs/deepep-h100-base-214be07d6}"
model=$1
cache=$2
dbo=$3
shift 3
arm=$run_root/arms/"$model-nixl-$cache-$dbo${ARM_SUFFIX:-}"
if [[ -f "$arm/run.log" ]]; then
    printf 'A run log already exists at %s. Use ARM_SUFFIX for another trial.\n' "$arm/run.log" >&2
    exit 2
fi
mkdir -p "$arm"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0a
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy,cuda_ipc,self}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-lo}"
if [[ "$model" == gdn ]]; then
    flashinfer_wheels="$runtime_root/flashinfer-kernels"
    export PYTHONPATH="$flashinfer_wheels${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$source_dir"
timeout --kill-after=15s "${ARM_TIMEOUT:-7m}" \
    uv run --offline --no-project --python .venv/bin/python .venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0 \
    "$run_root/validate_nixl.py" --model "$model" --cache "$cache" \
    --dbo "$dbo" --output "$arm" "$@" >"$arm/run.log" 2>&1
