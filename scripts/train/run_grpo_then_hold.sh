#!/usr/bin/env bash
# Run GRPO training on GPUs 1 & 3, then ALWAYS re-occupy those GPUs via
# gpu_hold.py — whether training succeeds, fails, or is interrupted (Ctrl-C /
# SIGTERM).
#
# Why: gpu_hold.py pins each card (masquerading as VLLM::EngineCore) so the
# slots aren't grabbed by someone else on this shared box. We free the GPUs for
# the training run, then re-occupy them the moment training exits.
#
# Usage:
#   bash scripts/train/run_grpo_then_queue.sh [args passed through to run_grpo.sh]
#
# Env overrides:
#   TRAIN_SCRIPT          which training script to run before holding (relative paths resolve
#                         from repo root). default: scripts/train/run_grpo.sh (colocated);
#                         set scripts/train/run_grpo_fully_async.sh for the disaggregated entry.
#   GPU_LIST              space-separated physical GPU ids for training (default: "1 3")
#   HOLD_DEVICES          comma-separated ids to re-occupy (default: derived from GPU_LIST -> "1,3")
#   HOLD_COMPUTE_PERCENT  fake SM utilisation % for the hold (default: 75)
#   HOLD_PYTHON           python with torch+setproctitle (default: slam-bat env)
#   GPU_HOLD              path to gpu_hold.py
#
# The hold runs in the FOREGROUND and holds the cards until you Ctrl-C it.
# gpu_hold.py with no --mem_percent/--mem_gb auto-fills memory (leaves ~512 MiB).
#
# NOTE: GPU_LIST defaults to two cards, matching run_grpo.sh's NGPUS_PER_NODE=2
# and ROLLOUT_TP=2. If you change the card count, sync those two knobs too.
#
# NOTE: deliberately NO `set -e` — a non-zero exit from run_grpo.sh must still
# reach the hold step. The `trap ... EXIT` guarantees the hold runs even on
# signals or a `set -u` abort.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- ensure the tau2verl conda env is active (the train scripts call bare `python3`; a fresh
# non-activated shell would hit "No module named verl"). .env supplies TAU2_ENV. Idempotent:
# a no-op when already active. set +u around conda's init (it references unset vars). ---
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
if [ "${CONDA_DEFAULT_ENV:-}" != "${TAU2_ENV:-tau2verl}" ]; then
    set +u
    source /ssd/home/zc/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
        && conda activate "${TAU2_ENV:-tau2verl}" 2>/dev/null \
        || echo "[run_grpo_then_hold] WARN: could not auto-activate ${TAU2_ENV:-tau2verl}; ensure it is active" >&2
    set -u
fi

# --- single source of truth: which physical GPUs this run owns ---
# Default 3 cards "0 1 2" for the fully_async 1-train + 2-rollout layout (see TRAIN_SCRIPT below).
# export GPU_LIST so the fully_async child sees the SAME list (it re-derives CUDA_VISIBLE_DEVICES
# from it) -> CVD and HOLD_DEVICES both cover all 3 cards. For colocated run_grpo.sh, pass GPU_LIST="<2 cards>".
GPU_LIST="${GPU_LIST:-0 1 2}"                     # physical ids, space-separated
export GPU_LIST
export CUDA_VISIBLE_DEVICES="${GPU_LIST// /,}"    # -> "0,1,2" : verl/vLLM see exactly these (logical 0,1,2)

GPU_HOLD="${GPU_HOLD:-/ssd/home/zc/yzs/omini/tools/gpu_hold.py}"
HOLD_PYTHON="${HOLD_PYTHON:-/ssd/home/zc/miniconda3/envs/slam-bat/bin/python}"
HOLD_DEVICES="${HOLD_DEVICES:-${GPU_LIST// /,}}"  # follow this script's GPUs -> "1,3"
HOLD_COMPUTE_PERCENT="${HOLD_COMPUTE_PERCENT:-75}"

# Which training script to run before the hold. Default = disaggregated fully_async (1 train + 2
# rollout, 3 cards). Set TRAIN_SCRIPT=scripts/train/run_grpo.sh (+ GPU_LIST="<2 cards>") for the
# legacy colocated entry. Relative paths resolve from repo root so the wrapper works regardless of CWD.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$HERE/scripts/train/run_grpo_fully_async.sh}"
case "$TRAIN_SCRIPT" in /*) ;; *) TRAIN_SCRIPT="$HERE/$TRAIN_SCRIPT" ;; esac

_hold_gpus() {
    if [ -f "$GPU_HOLD" ]; then
        echo "[run_grpo_then_queue] holding GPUs ${HOLD_DEVICES} @ compute ${HOLD_COMPUTE_PERCENT}% (foreground; Ctrl-C to release)"
        # gpu_hold.py addresses GPUs by LOGICAL cuda index, so CUDA_VISIBLE_DEVICES
        # must be cleared here — otherwise the ids would be remapped instead of
        # meaning the physical cards.
        env -u CUDA_VISIBLE_DEVICES "$HOLD_PYTHON" "$GPU_HOLD" \
            --devices "$HOLD_DEVICES" --compute_percent "$HOLD_COMPUTE_PERCENT" \
            || echo "[run_grpo_then_queue] WARN: gpu_hold.py exited non-zero" >&2
    else
        echo "[run_grpo_then_queue] ERROR: gpu_hold.py not found at ${GPU_HOLD}" >&2
    fi
}
# Fires once on any exit path (normal, training failure, or interrupt).
trap _hold_gpus EXIT

echo "[run_grpo_then_queue] training on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} : ${TRAIN_SCRIPT##*/} $*"
bash "$TRAIN_SCRIPT" "$@"
train_rc=$?
echo "[run_grpo_then_queue] training exited with code ${train_rc}"

# Preserve training's exit code as this wrapper's exit code (the EXIT trap still
# runs the hold step first); a CI / caller can still tell if training failed.
exit "$train_rc"
