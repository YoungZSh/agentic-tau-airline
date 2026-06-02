#!/usr/bin/env bash
# Smoke test for the GRPO pipeline: tiny batch / short trajectories / 1 step.
# Proves the loop runs end-to-end (rollout -> reward -> GRPO update -> LoRA sync)
# without OOM. NOT a real training run. Reuses run_grpo.sh via env overrides.
#
# Still needs 2×A100 + a vLLM server + gpt-5 API (user simulator). Expect a few
# minutes of startup. Uses a 2-task subset parquet so a step finishes fast.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_NAME="${TAU2_ENV:-tau2verl}"

# Build a 2-task smoke parquet (first 2 train tasks) if absent.
SMOKE_DIR="${SMOKE_DIR:-data/tau3_airline/smoke}"
if [ ! -f "$SMOKE_DIR/train.parquet" ]; then
  conda run -n "$ENV_NAME" python - "$SMOKE_DIR" <<'PY'
import sys, datasets
from tau2_airline_verl.data.build_parquet import _build_rows
from tau2_airline_verl.agents.qwen3_prompt import build_system_prompt
from tau2_airline_verl.data.splits import load_train_tasks
import os
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
policy = build_system_prompt()
tasks = load_train_tasks()[:2]
rows = _build_rows(tasks, "train", policy)
datasets.Dataset.from_list(rows).to_parquet(f"{out}/train.parquet")
datasets.Dataset.from_list(rows).to_parquet(f"{out}/test.parquet")
print(f"smoke parquet: {len(rows)} tasks -> {out}")
PY
fi

TRAIN_FILES="$SMOKE_DIR/train.parquet" \
VAL_FILES="$SMOKE_DIR/test.parquet" \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
ROLLOUT_N=2 \
MAX_RESPONSE_LENGTH=1024 \
MAX_ASSISTANT_TURNS=4 \
MAX_USER_TURNS=4 \
AGENT_NUM_WORKERS=2 \
ROLLOUT_GPU_MEM_UTIL=0.4 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
VAL_BEFORE_TRAIN=False \
LOGGER='["console"]' \
EXPERIMENT_NAME="smoke_$(date +%s)" \
exec bash "$HERE/scripts/train/run_grpo.sh" "$@"
