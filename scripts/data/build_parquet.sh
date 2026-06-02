#!/usr/bin/env bash
# Build verl GRPO parquet datasets (train=30 / test=20) from tau2 airline tasks.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_NAME="${TAU2_ENV:-tau2verl}"
OUT_DIR="${1:-data/tau3_airline}"
conda run -n "$ENV_NAME" python -m tau2_airline_verl.data.build_parquet --out_dir "$OUT_DIR"
