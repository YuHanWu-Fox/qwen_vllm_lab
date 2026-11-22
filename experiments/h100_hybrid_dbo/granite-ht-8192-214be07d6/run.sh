#!/usr/bin/env bash
set -euo pipefail

run_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir="${DBO_VLLM_DIR:-$(cd "$run_root/../../.." && pwd)/vllm-v0.33}"
runtime_root="${DBO_EP_RUNTIME_DIR:-/workspace/qwen_vllm_lab_runs/deepep-h100-base-214be07d6}"
cache=$1
variant=$2
tag=$3
shift 3
arm="$run_root/$cache-$variant$tag"
mkdir "$arm"

export LD_LIBRARY_PATH="$runtime_root/build/nvshmem/lib:$source_dir/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST=9.0a
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
unset CUDA_LAUNCH_BLOCKING VLLM_GPU_SYNC_CHECK

cd "$source_dir"
printf '%q ' "$run_root/run.sh" "$cache" "$variant" "$tag" "$@" > "$arm/command.txt"
printf '\n' >> "$arm/command.txt"
prefix=()
if [[ " ${*} " == *' --trace '* ]]; then
    prefix=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
        --osrt-threshold=10000 --capture-range=cudaProfilerApi --capture-range-end=none
        --wait=all --output="$arm/trace")
fi
timeout --kill-after=15s 5m "${prefix[@]}" \
    uv run --offline --no-project --python .venv/bin/python .venv/bin/python \
    -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0 \
    "$run_root/diagnose.py" --cache "$cache" --variant "$variant" --output "$arm" "$@" > "$arm/run.log" 2>&1
if [[ -f "$arm/trace.nsys-rep" ]]; then
    nsys export --type sqlite --output "$arm/trace.sqlite" "$arm/trace.nsys-rep" > "$arm/export.log" 2>&1
fi
