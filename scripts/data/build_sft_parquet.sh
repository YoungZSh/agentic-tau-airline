#!/usr/bin/env bash
# Build the multi-turn SFT parquet from the AReaL airline SFT subset.
# -> data/tau2_airline_sft/{train,val}.parquet (one row per dialog, 999 dialogs).
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

TAU2_ENV=${TAU2_ENV:-tau2verl}
OUT_DIR=${OUT_DIR:-data/tau2_airline_sft}
SRC=${SRC:-local_datasets/tau2_sft_train.jsonl}
MAX_TOKENS=${MAX_TOKENS:-32768}   # drop dialogs longer than this (0 disables)

conda run -n "$TAU2_ENV" python -m tau2_airline_verl.data.build_sft_parquet \
    --out_dir "$OUT_DIR" --src "$SRC" --max_tokens "$MAX_TOKENS"
