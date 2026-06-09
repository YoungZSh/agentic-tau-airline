#!/usr/bin/env bash
# AReaL-recipe reproduction probe (LoRA, 2-GPU) — see docs/grpo_flat_curve_and_areal_recipe.md.
#
# Goal: a cheap (~25 yuan, ~15 steps) test of whether the AReaL airline recipe moves the
# policy at all on our stack. It mirrors AReaL's examples/tau2/config_8b_airline.yaml on
# every knob we can match on 2 GPUs:
#   - RL-zero from base Qwen3-8B (init_from_scratch:false; NOT our SFT checkpoint)
#   - plain GRPO  (adv_estimator=grpo; GDPO reverted)
#   - NO KL       (use_kl_loss=False -> ref model not loaded; AReaL kl_ctl=0)
#   - wide clip   (clip_ratio=0.4;     AReaL eps_clip=0.4)
#   - full-batch  (train_batch_size=40 = our whole train split; AReaL batch=30=their split)
#                 ppo_mini_batch_size=40 -> 1 update/step (AReaL ppo_n_minibatches=1)
# The one knob we DON'T match is full-param vs LoRA (2-GPU limit) — that's the variable
# under test: if this climbs, LoRA+recipe is enough; if flat, LoRA capacity is the suspect.
# LR is bumped to 1e-4 (LoRA needs ~10x the full-param 1.7e-5; counters the grad_norm~0.02 freeze).
# user-sim stays DeepSeek V4 (.env). No >0.95 group filter (won't fire from a ~0.2 RL-zero start).
#
# Usage (run inside tmux; the trap re-occupies the GPUs after training exits):
#   GPU_LIST="7 5" bash scripts/train/run_grpo_areal_probe.sh
# Override any single knob at launch, e.g.:
#   GPU_LIST="7 5" ACTOR_LR=5e-5 TOTAL_EPOCHS=20 bash scripts/train/run_grpo_areal_probe.sh

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- AReaL-recipe defaults (each still overridable from the caller's env) ---
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_lora_areal_probe}"
export INIT_MODEL_PATH="${INIT_MODEL_PATH:-$HERE/ckpts/Qwen3-8B}"   # RL-zero base
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"                       # plain GRPO (not GDPO)
export USE_KL_LOSS="${USE_KL_LOSS:-False}"                          # no KL, no ref model
export CLIP_RATIO="${CLIP_RATIO:-0.4}"                              # wide clip = trust region
export ACTOR_LR="${ACTOR_LR:-1e-4}"                                 # LoRA needs ~10x full-param LR
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-40}"                   # full-batch = whole train split
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-40}"             # 1 update/step, on-policy
export ROLLOUT_N="${ROLLOUT_N:-8}"                                  # group size
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"                           # full-batch -> 1 step/epoch -> 15 steps
export TEST_FREQ="${TEST_FREQ:-2}"                                  # val every 2 steps
export SAVE_FREQ="${SAVE_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"                 # measure the base baseline first
# FSDP offload: REQUIRED for full-batch on 2 colocated GPUs. Without it the actor + its
# 320-rollout batch starves vLLM's wake_up KV re-alloc -> "CUDA OOM at wake_up" (the
# crash on runs ..._135840 and ..._152750). Offloads actor params+optim to CPU during
# rollout; ~negligible time cost vs the multi-minute steps. AReaL omits it (dedicated gpus).
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"

echo "[areal_probe] init=${INIT_MODEL_PATH}"
echo "[areal_probe] adv=${ADV_ESTIMATOR} use_kl_loss=${USE_KL_LOSS} clip=${CLIP_RATIO} lr=${ACTOR_LR}"
echo "[areal_probe] batch=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} n=${ROLLOUT_N} epochs=${TOTAL_EPOCHS}"
[ -d "$INIT_MODEL_PATH" ] || { echo "[areal_probe] base model not found: $INIT_MODEL_PATH" >&2; exit 1; }

# Delegate to the GPU-holding wrapper (sets CUDA_VISIBLE_DEVICES from GPU_LIST, re-occupies on exit).
exec bash "$HERE/scripts/train/run_grpo_then_hold.sh" "$@"
