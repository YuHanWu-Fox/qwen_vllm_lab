#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

run() {
    local status=0
    printf '%s START %s\n' "$(date -u -Iseconds)" "$*" | tee -a matrix.log
    ./run_arm.sh "$@" || status=$?
    printf '%s END status=%s %s\n' "$(date -u -Iseconds)" "$status" "$*" | tee -a matrix.log
    return "$status"
}

run_ht() {
    local common=(--max-batched-tokens 32786 --max-model-len 33792 --prefill-threshold 8192)
    run mamba2 ht none off large -r2 "${common[@]}"
    run mamba2 ht none on large -r2 "${common[@]}"
    run mamba2 ht align on large -r2 "${common[@]}"
    run mamba2 ht align off large -r2 "${common[@]}"
}

run_regular() {
    local backend=$1 model cache cap min_cap depth tag max_len cases
    shift
    local models=("$@")
    if (( ${#models[@]} == 0 )); then
        models=(gdn mamba2)
    fi
    for model in "${models[@]}"; do
        cap=4096
        min_cap=2048
        max_len=8192
        if [[ "$model" == gdn ]]; then
            cap=1536
            min_cap=1024
            if [[ "$backend" == nixl ]]; then
                max_len=33792
            fi
        fi
        for cache in none align; do
            local order=(off on)
            if [[ "$cache" == align || "$model" == gdn ]]; then
                order=(on off)
            fi
            while :; do
                depth=8192
                if ((cap > 2048)); then
                    depth=16384
                fi
                tag="-cap$cap"
                local options=(--max-batched-tokens "$cap" --max-model-len "$max_len" --prefill-threshold 512 --max-seqs 64 --kv-gib 4)
                if [[ "$model" == gdn ]]; then
                    cases="prefill_1x256,prefill_2x256,prefill_4x256,prefill_1x$cap,decode_16x32,decode_32x32,mixed_32decode_2x320"
                    if ((cap >= 1536)); then
                        cases="$cases,prefill_6x256,mixed_32decode_2x704"
                    fi
                    if ((max_len > 32786)); then
                        cases="$cases,prefill_chunked_1x16384,prefill_chunked_1x32786"
                    fi
                    options+=(--cases "$cases")
                elif [[ "$backend" == nixl ]]; then
                    cases="prefill_1x256,prefill_2x256,prefill_8x256,prefill_1x$cap"
                    cases="$cases,decode_16x32,decode_32x32,decode_64x32,mixed_32decode_2x704"
                    options+=(--cases "$cases")
                fi
                if NVSHMEM_QP_DEPTH="$depth" run "$model" "$backend" "$cache" "${order[0]}" regular "$tag" "${options[@]}" &&
                   NVSHMEM_QP_DEPTH="$depth" run "$model" "$backend" "$cache" "${order[1]}" regular "$tag" "${options[@]}"; then
                    break
                fi
                if ((cap == min_cap)); then
                    return 1
                fi
                cap=$min_cap
            done
        done
    done
}

run_ht_repeat() {
    local common=(--max-batched-tokens 32786 --max-model-len 33792 --prefill-threshold 8192
        --prefill-totals 8192,16384,24576,32768,32786
        --cases prefill_32x256,prefill_64x256,prefill_1x16384,prefill_96x256,prefill_128x256,prefill_1x32786)
    run mamba2 ht none on large -reverse "${common[@]}"
    run mamba2 ht none off large -reverse "${common[@]}"
}

run_nixl_large() {
    local cap
    for cap in 32786 16384 8192; do
        local common=(--max-batched-tokens "$cap" --max-model-len 33792 --prefill-threshold 8192
            --prefill-totals 8192,16384,32768,32786 --warmup 3 --trials 8)
        if run mamba2 nixl none off large "-cap$cap" "${common[@]}" &&
           run mamba2 nixl none on large "-cap$cap" "${common[@]}"; then
            return 0
        fi
    done
    return 1
}

for phase in "$@"; do
    case "$phase" in
        ht) run_ht ;;
        ht-repeat) run_ht_repeat ;;
        nixl-large) run_nixl_large ;;
        ll-gdn) run_regular ll gdn ;;
        ll-mamba2) run_regular ll mamba2 ;;
        nixl) run_regular nixl ;;
        *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2 ;;
    esac
done
