#!/usr/bin/env bash
set -euo pipefail

run_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir="${DBO_VLLM_DIR:-$(cd "$run_root/../../.." && pwd)/vllm-v0.33}"
runtime_root="${DBO_EP_RUNTIME_DIR:-/workspace/qwen_vllm_lab_runs/deepep-h100-base-214be07d6}"
model=$1
backend=$2
cache=$3
dbo=$4
shift 4
arm=$run_root/arms/"$model-$backend-$cache-$dbo${ARM_SUFFIX:-}"
mkdir -p "$arm"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST=9.0a
export LD_LIBRARY_PATH="$runtime_root/build/nvshmem/lib:$source_dir/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "$backend" == ll ]]; then
    export NVSHMEM_QP_DEPTH="${NVSHMEM_QP_DEPTH:-8192}"
fi
if [[ "${USE_FLASHINFER_KERNEL_WHEELS:-0}" == 1 ]]; then
    export PYTHONPATH="$runtime_root/flashinfer-kernels${PYTHONPATH:+:$PYTHONPATH}"
fi

cd "$source_dir"
timeout --kill-after=20s "${ARM_TIMEOUT:-12m}" \
    uv run --offline --no-project --python .venv/bin/python .venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0 \
    "$run_root/validate_deepep.py" --model "$model" --backend "$backend" \
    --cache "$cache" --dbo "$dbo" --output "$arm" "$@" >"$arm/run.log" 2>&1
