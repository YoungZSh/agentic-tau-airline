#!/usr/bin/env bash
# GRPO/GDPO + LoRA + multi-turn — Qwen3-8B, DISAGGREGATED fully-async on 2 GPUs
# (1 GPU FSDP train + 1 GPU vLLM rollout). Sibling of run_grpo.sh, which is the
# COLOCATED HybridEngine entry (verl.trainer.main_ppo, 2 GPUs time-sharing).
#
# Why this script exists: colocated rollout is ~65% of step time and the train card
# idles during it. fully_async_policy puts rollout and train on SEPARATE cards so
# rollout(N+1) overlaps train(N). Entry is verl.experimental.fully_async_policy.
# fully_async_main, which loads fully_async_ppo_trainer.yaml (extends ppo_trainer).
#
# Our Tau2AirlineAgentLoop + tau2 reward + GDPO are ENGINE-AGNOSTIC and reused as-is
# (no src/ changes): the rollouter drives the same AgentLoop registry / server_manager,
# our loop already emits aligned response_logprobs (needed by use_rollout_log_probs),
# and GDPO/extract_reward read reward_extra_info exactly as in the colocated path.
#
# DEFAULTS = faithful disaggregated reproduction of the validated areal_probe run
# (adv_estimator=grpo single-scalar, base Qwen3-8B RL-zero, clip 0.4, no KL, lr 1e-4),
# at ONE-STEP-OFF async (trigger_sync=1, staleness=1, partial rollout) so data lags the
# policy by at most ~1 optimizer step. Raise STALENESS_THRESHOLD to trade freshness for
# throughput once this config is shown to learn.
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

# --- which physical GPUs this run owns: first = ... they're split by Ray into
# trainer (TRAIN_GPUS) + rollout (ROLLOUT_GPUS). Default 3 cards -> 1 train + 2 rollout (DP=2),
# matching AReaL's 2:1 inference:train ratio. The 2nd rollout card ~1.8x'd gen throughput
# (validated 2026-06-11: timing_s/gen 177->100s on n=2/8192) and at the heavy config (n=8/12288)
# both rollout cards saturate (~95-100%). Train side stays 1 GPU (fsdp_size=1 -> decoupled patch).
# Via run_grpo_then_hold.sh, pass GPU_LIST="0 1 2" so the wrapper's HOLD_DEVICES matches all 3.
GPU_LIST=${GPU_LIST:-"0 1 2"}
export CUDA_VISIBLE_DEVICES="${GPU_LIST// /,}"
TRAIN_GPUS=${TRAIN_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
NNODES=${NNODES:-1}

project_name=${PROJECT_NAME:-verl_grpo_airline}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_lora_fa_areal}
logger=${LOGGER:-'["console","wandb"]'}
run_name=${RUN_NAME:-$experiment_name}
run_timestamp=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
run_dir=${RUN_DIR:-"$HERE/outputs/${run_name}_${run_timestamp}"}

# Resume in place (same as run_grpo.sh): RESUME_FROM=<prev run dir>.
RESUME_FROM=${RESUME_FROM:-}
if [ -n "$RESUME_FROM" ]; then
    run_dir="$RESUME_FROM"
    [ -d "$run_dir/checkpoints" ] || { echo "[run_grpo_fa] RESUME_FROM has no checkpoints/: $run_dir" >&2; exit 1; }
    echo "[run_grpo_fa] RESUMING from $run_dir"
fi
resume_mode=${RESUME_MODE:-auto}
default_local_dir=${DEFAULT_LOCAL_DIR:-"$run_dir/checkpoints"}
logs_dir="$run_dir/logs"
mkdir -p "$logs_dir"

LOG_FILE=${LOG_FILE:-"$logs_dir/train_${run_timestamp}.log"}
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[run_grpo_fa] logging to $LOG_FILE"
fi

# NCCL OOB on loopback (single node; ib0 auto-select deadlocks init). The cross-process
# trainer<->rollouter param sync (NCCL) also rides this.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
# fully_async async rollout requires the vLLM v1 engine + raw chat to the agent loop.
export VLLM_USE_V1=1

: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env (user-sim + NL judge; default DeepSeek V4)}"

# Init checkpoint. Default = base Qwen3-8B (RL-zero), matching the validated areal_probe
# run (which trained from base, adv_estimator=grpo). Override INIT_MODEL_PATH for the
# SFT-merged init or another base.
MODEL_PATH="${INIT_MODEL_PATH:-${QWEN3_8B_PATH:-$HERE/ckpts/Qwen3-8B}}"
[ -d "$MODEL_PATH" ] || { echo "[run_grpo_fa] init checkpoint not found: $MODEL_PATH — set INIT_MODEL_PATH"; exit 1; }

TRAIN_FILES=${TRAIN_FILES:-data/tau3_airline/train.parquet}
VAL_FILES=${VAL_FILES:-data/tau3_airline/test.parquet}

# --- batch / streaming sizing (fully_async semantics) ---
# data.train_batch_size is INEFFECTIVE in fully async (set 0); gen_batch_size=1 = streaming
# sample-by-sample production. The trainer trains on require_batches*ppo_mini_batch_size
# prompts per local update; total work = rollout.total_rollout_steps prompts over the run.
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-40}     # prompts per trainer update (validated run used 40)
require_batches=${REQUIRE_BATCHES:-1}              # mini-batches fetched per local update
# total prompts over the whole run ~= ppo_mini_batch_size * (#sync-cycles). 40*15 ~= the
# validated 15-step run. Override TOTAL_ROLLOUT_STEPS for longer/smoke.
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-1200}
rollout_n=${ROLLOUT_N:-8}                          # GRPO group size

max_prompt_length=${MAX_PROMPT_LENGTH:-6144}
max_response_length=${MAX_RESPONSE_LENGTH:-12288}   # raised back from 10240 (2026-06-12): at 10240 step:1 showed response_length/clip_ratio=10% — a tenth of trajectories truncated, mostly the long/hard ones. The old "12288 = knife's edge ~79.7GB" note predates the real fixes; with the micro-batch graph-retention leak fixed (TRAIN_METRICS_DETACH) + bf16 master + fused-CE, the actor update peaks at ~17GB, so 18432-token micro-batches have huge headroom. fused-CE keeps the logits term flat as token len grows — keep USE_FUSED_KERNELS on when raising this further.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}   # MUST be >= longest single sequence (prompt+response): dynamic-bsz can't split one seq across micro-batches, and seqlen_balancing asserts max_token_len >= max_seq_len. Auto-size to the cap -> assert always holds AND smallest micro-batch -> least backward activation on the single train card. (response 8192 -> 14336; validated.)

# --- algorithm (defaults match validated areal_probe: grpo, no KL, clip 0.4, lr 1e-4) ---
adv_estimator=${ADV_ESTIMATOR:-grpo}              # grpo = single-scalar (validated); set gdpo to test subscore path
gdpo_reward_keys=${GDPO_REWARD_KEYS:-'[db,comm,db_comm]'}
gdpo_reward_weights=${GDPO_REWARD_WEIGHTS:-'[1.0,1.0,1.0]'}
actor_lr=${ACTOR_LR:-1e-4}
use_kl_loss=${USE_KL_LOSS:-False}                 # AReaL kl_ctl=0; with use_kl_in_reward=False the ref model is never loaded
kl_loss_coef=${KL_LOSS_COEF:-0.0}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio=${CLIP_RATIO:-0.4}                     # AReaL eps_clip=0.4

lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}

# Fused linear cross-entropy (docs/sft_fused_kernels.md): never materialize the full
# [tokens × vocab] logits — at ppo_max_token_len 16384 × vocab 151936 that's the ~5GB
# allocation the 2026-06-11 OOMs died on. Natively wired in this engine path
# (engine_workers reads model.use_fused_kernels -> FSDPEngineWithLMHead fused forward),
# numerically validated 1e-6 on SFT (full-param). Default OFF here: LoRA + fused on the
# RL update path is not yet smoke-validated; flip to True if expandable_segments alone
# doesn't hold (escalation order: expandable -> fused -> smaller ppo_max_token_len).
use_fused_kernels=${USE_FUSED_KERNELS:-False}

# --- rollout (DEDICATED card -> bigger KV than colocated 0.5; TP=1 on one GPU) ---
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}   # 0.9 OOMs param-sync: NCCL bucketed weight transfer needs a 2GB staging buffer on THIS card (update_weights_bucket_megabytes=2048); 0.9 leaves <1GB free -> torch.empty(2GB) OOM. 0.8 (validated in dsmoke5) leaves ~11GB headroom.
temperature=${TEMPERATURE:-1.0}                      # AReaL tau2 uses temperature=1.0 for training rollout
max_assistant_turns=${MAX_ASSISTANT_TURNS:-24}       # match the validated colocated areal_probe (24/12): the 0612 fa run's 32/32 changed the MDP (trajectories up to 48 turns vs 37) and made curves non-comparable; revisit longer horizons only after fa reproduces colocated learning
max_user_turns=${MAX_USER_TURNS:-12}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}
agent_num_workers=${AGENT_NUM_WORKERS:-16}

# --- FSDP (1 train GPU -> fsdp_size=1). param_offload=True here is COARSE whole-model offload
# (offload_fsdp_model_to_cpu = model.cpu(): the frozen 8B base ~16GB + the tiny LoRA adapter,
# moved to CPU between phases, reloaded before fwd/bwd), NOT FSDP2 per-layer CPUOffloadPolicy
# (that's the separate `offload_policy` knob, off; only forward_only engines use it). The actor
# update's backward still has the 16GB base resident, so the OOM fix was really the auto-sized
# (smaller) micro-batch above; param_offload's role is freeing that 16GB during the trainer's
# ~55% idle wait + the empty_cache() defrag (run#2 OOM had 8.88GB reserved-but-unallocated).
# Validated end-to-end 2026-06-10. optimizer_offload stays False (LoRA optim state is tiny). ---
param_offload=${PARAM_OFFLOAD:-True}
optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
fsdp_size=${FSDP_SIZE:-1}
# Master-param + Adam-state dtype on the train card. fp32 (verl default, what the validated
# colocated areal_probe ran) — NOT bf16. The earlier "fp32 buys nothing with LoRA" note was
# wrong about the half that matters: model_dtype also sets the TRAINABLE LoRA params and
# their Adam moments, and this task's updates are tiny (grad_norm 0.012-0.018, lr 1e-4) —
# below bf16's ~0.4% relative precision, so `w += update` rounds away and exp_avg_sq
# degrades. Suspected co-cause of the flat 0612 fa curve alongside trigger_sync=4.
# Memory is fine post the real OOM fixes (metrics detach + expandable_segments): actor
# update peaks ~17GB, +32GB fp32 base still has headroom. Compute stays bf16 either way
# (mixed_precision param_dtype=bf16).
model_dtype=${MODEL_DTYPE:-fp32}

# --- async knobs. DEFAULT = one-step-off (trigger=1, staleness=1, partial rollout):
# weights sync after EVERY optimizer step, rollouter may run at most 1 version ahead, so
# data lags <=1-2 optimizer steps. The 2026-06-12 run with trigger=4 had ~78% stale
# trajectories (effective lag 4-8 optimizer steps in AReaL units) and a flat score curve
# at the sample budget where colocated areal_probe reached 0.38 — sync costs only ~18s
# vs ~1900s/optimizer step and rollouter idle_ratio was already ~0.4, so trigger>1 buys
# nothing here. Climb staleness via STALENESS_THRESHOLD (fractional OK), not TRIGGER_SYNC_STEP. ---
staleness_threshold=${STALENESS_THRESHOLD:-1}
trigger_parameter_sync_step=${TRIGGER_SYNC_STEP:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}
use_rollout_log_probs=${USE_ROLLOUT_LOG_PROBS:-True}   # our loop emits aligned logprobs; on-policy this == training logprobs
# old_log_prob mode. bypass_mode=True (default) -> old_log_prob = rollout_log_probs (the
# behavior-policy logprob; standard PPO ratio pi_theta/pi_rollout, a valid off-policy
# correction even under staleness>0). bypass_mode=False -> "decoupled PPO" (3 policies,
# recomputes old_log_prob on the training engine + rollout_is reweighting), which
# approximates AReaL's Decoupled PPO. It normally needs FSDP2-sharded DTensor params (>=2 train
# GPUs), but we UNLOCK it on a single train GPU (fsdp_size=1) via the sitecustomize monkeypatch
# (auto-engaged below when bypass_mode=False): it falls back to a plain trainable-param CPU snapshot
# when no DTensor is present. So decoupled PPO + token-level rollout-IS is the DEFAULT here.
# rollout_is valid values: null | token | sequence (only used when bypass_mode=False).
bypass_mode=${BYPASS_MODE:-False}        # AReaL-style decoupled PPO (3-policy) by default
rollout_is=${ROLLOUT_IS:-token}          # token-level truncated importance sampling (clamp = rollout_is_threshold below)
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
# Rejection sampling (rollout_rs): masks too-off-policy samples out of the loss. Works in
# bypass mode using only rollout logprobs (1-GPU-safe), unlike rollout_is. e.g. seq_mean_k3 / 0.01.
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

# --- sitecustomize monkeypatches (src/tau2_airline_verl/patches/). The dir is always on
# PYTHONPATH; each patch only engages via its env var below, so this line alone is inert. ---
export PYTHONPATH="${HERE}/src/tau2_airline_verl/patches${PYTHONPATH:+:${PYTHONPATH}}"

# Decoupled-on-1-GPU. bypass_mode=False (true 3-policy decoupled + rollout_is) needs the
# proximal-anchor snapshot, which verl implements only for FSDP2-sharded DTensor params
# (>=2 train GPUs). On our single train GPU we engage a sitecustomize monkeypatch that falls
# back to a plain trainable-param (LoRA adapter) state_dict copy. Auto-on when
# bypass_mode=False; off (and zero effect) otherwise.
if [ "${bypass_mode}" = "False" ] || [ "${bypass_mode}" = "false" ]; then
    export VERL_DECOUPLED_1GPU_PATCH=1
    echo "[run_grpo_fa] decoupled-on-1GPU monkeypatch ENABLED (bypass_mode=False)"
fi

# Expandable segments on the TRAIN worker only (default on). All three 2026-06-11 fa_areal runs
# OOM'd in the actor update's backward with ~10GB reserved-but-unallocated (fragmentation).
# A global `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the run_sft.sh fix) is NOT
# safe here: ray workers inherit it and vLLM's sleep-mode CuMemAllocator hard-asserts against it
# -> both rollout servers would crash at startup. The sitecustomize patch instead flips the
# allocator at runtime inside TrainingWorker.__init__ (trainer process only).
train_expandable_segments=${TRAIN_EXPANDABLE_SEGMENTS:-True}
if [ "${train_expandable_segments}" = "True" ] || [ "${train_expandable_segments}" = "true" ]; then
    export VERL_TRAIN_EXPANDABLE_SEGMENTS=1
    echo "[run_grpo_fa] trainer expandable_segments monkeypatch ENABLED"
fi

# Detach metric tensors in the actor loss (default on). verl's ppo_loss stores the live
# pg_loss/ppo_kl tensors (with autograd graph) in the per-micro-batch metrics dict, retained
# until the whole mini-batch finishes — profiled at ~0.22GiB leaked per training micro-batch
# (the actual cause of the 2026-06-11 update_actor OOMs after the static fixes landed).
# Metrics are reporting-only, so detaching is behavior-neutral.
train_metrics_detach=${TRAIN_METRICS_DETACH:-True}
if [ "${train_metrics_detach}" = "True" ] || [ "${train_metrics_detach}" = "true" ]; then
    export VERL_TRAIN_METRICS_DETACH=1
    echo "[run_grpo_fa] loss-metrics detach monkeypatch ENABLED"
fi

# Diagnostic: record allocator history on the trainer and dump a torch memory snapshot to
# $logs_dir on actor-update OOM (TRAIN_MEM_SNAPSHOT=1). Off by default.
train_mem_snapshot=${TRAIN_MEM_SNAPSHOT:-0}
if [ "${train_mem_snapshot}" = "1" ] || [ "${train_mem_snapshot}" = "True" ]; then
    export VERL_TRAIN_MEM_SNAPSHOT=1
    export VERL_MEM_SNAPSHOT_DIR="${logs_dir}"
    echo "[run_grpo_fa] trainer memory-snapshot diagnostics ENABLED -> ${logs_dir}"
fi

save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-2}
max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-8}
val_before_train=${VAL_BEFORE_TRAIN:-True}
total_epochs=${TOTAL_EPOCHS:-30}   # total_rollout_steps counts PROMPTS (1 RolloutSample = 1 prompt = n trajectories = 1 GRPO group). With the 40-prompt train split, total_rollout_steps=1200 == exactly 30 epochs, so this ceiling and total_rollout_steps bind TOGETHER at the defaults. total sync-steps = total_rollout_steps / (ppo_mini * trigger) = 1200/(40*1) = 30 (each = 1 optimizer step at trigger=1).
rollout_data_dir=${ROLLOUT_DATA_DIR:-"$logs_dir/rollouts"}
val_data_dir=${VAL_DATA_DIR:-"$logs_dir/val_rollouts"}

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator="${adv_estimator}" \
    ++algorithm.gdpo_reward_keys="${gdpo_reward_keys}" \
    ++algorithm.gdpo_reward_weights="${gdpo_reward_weights}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=${use_fused_kernels} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${model_dtype} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.clip_ratio=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_user_turns} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.agent.num_workers=${agent_num_workers} \
    actor_rollout_ref.rollout.agent.default_agent_loop=tau2_airline \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${HERE}/configs/tau2_agent_loop.yaml" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger="${logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.resume_mode=${resume_mode} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.total_epochs=${total_epochs} \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.validation_data_dir="${val_data_dir}" \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_trainer_do_validate=False \
    "$@"
