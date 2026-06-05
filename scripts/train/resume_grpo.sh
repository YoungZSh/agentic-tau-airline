#!/usr/bin/env bash
# Resume the crashed GRPO run from its latest checkpoint, with hardened gpt-5
# (user-simulator) networking so a transient API timeout no longer kills the run.
#
# What broke (run 20260603_152218): training reached step 35, then a gpt-5
# user-sim call hit httpx.PoolTimeout — the shared sync connection pool drained
# under concurrent rollouts against the slow reasoning model. verl's
# asyncio.gather has no per-rollout tolerance, so the step (and process) died.
#
# Fix (all env-driven; consumed by src/.../utils/litellm_setup.py and
# env/airline_interaction.py): bigger sync http pool + larger per-call timeout +
# more retries. Nothing about the training dynamics changes.
#
# verl resume_mode defaults to `auto`: pointing RUN_DIR at the existing run dir
# makes it load checkpoints/global_step_30 and continue to total_epochs (100 steps).
#
# Usage:
#   bash scripts/train/resume_grpo.sh                 # resume the default run
#   RUN_DIR=outputs/<other_run> bash scripts/train/resume_grpo.sh
#   TAU2_USER_NUM_RETRIES=8 bash scripts/train/resume_grpo.sh   # override a knob
#
# PRECONDITION: GPUs 1 & 3 must be free. The crashed run's wrapper left gpu_hold.py
# decoys pinning them (masquerading as VLLM::EngineCore) — stop those first, e.g.:
#   pkill -f gpu_hold.py        # or: kill the old run_grpo_then_hold.sh wrapper
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- the run to resume (its checkpoints/ holds global_step_30) ---
export RUN_DIR="${RUN_DIR:-$HERE/outputs/qwen3_8b_lora_tau2_airline_20260603_152218}"
# Fresh log-file stamp for the resumed segment so we don't clobber the crash log
# (run_grpo.sh names the log train_${RUN_TIMESTAMP}.log; RUN_DIR fixes the folder).
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-resume_$(date +%Y%m%d_%H%M%S)}"

# --- gpt-5 / litellm hardening (override any via env) ---
export TAU2_USER_TIMEOUT="${TAU2_USER_TIMEOUT:-600}"          # secs per attempt (read + pool wait)
export TAU2_USER_NUM_RETRIES="${TAU2_USER_NUM_RETRIES:-10}"   # was tau2 DEFAULT_MAX_RETRIES=3
export TAU2_LLM_MAX_CONNECTIONS="${TAU2_LLM_MAX_CONNECTIONS:-256}"
export TAU2_LLM_MAX_KEEPALIVE="${TAU2_LLM_MAX_KEEPALIVE:-64}"

if [ ! -f "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ]; then
    echo "[resume_grpo] ERROR: no checkpoint under $RUN_DIR/checkpoints" >&2
    exit 1
fi
echo "[resume_grpo] resuming $RUN_DIR (resume_mode=auto -> global_step $(cat "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"))"
echo "[resume_grpo] gpt-5 hardening: timeout=${TAU2_USER_TIMEOUT}s retries=${TAU2_USER_NUM_RETRIES} pool=${TAU2_LLM_MAX_CONNECTIONS}/${TAU2_LLM_MAX_KEEPALIVE}"

# Reuse the train-then-hold wrapper (GPUs 1 & 3; re-pins them if training exits).
exec bash "$HERE/scripts/train/run_grpo_then_hold.sh" "$@"
