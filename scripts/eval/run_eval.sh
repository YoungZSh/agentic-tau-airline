#!/usr/bin/env bash
# Stage 3 held-out eval on the TEST split. Same machinery as run_base_rollout.sh;
# point LORA_PATH at a trained adapter to evaluate the GRPO policy, or leave it
# empty to get the base TEST baseline (for the Base vs GRPO comparison).
#
#   LORA_PATH=ckpts/grpo_airline/global_step_30/actor/lora_adapter \
#   bash scripts/eval/run_eval.sh 8
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_TRIALS="${1:-8}"; shift || true
SPLIT=test \
OUT="${OUT:-notes/stage3_held_out_eval.md}" \
exec bash "$HERE/scripts/eval/run_base_rollout.sh" "$NUM_TRIALS" "$@"
