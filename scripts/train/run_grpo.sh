#!/usr/bin/env bash
# GRPO/GDPO + LoRA + multi-turn async rollout — Qwen3-8B on 2×A100 80GB (plan stage 2).
# Entry: verl.trainer.main_ppo over the default ppo_trainer.yaml; the scalar reward flows
# from our tau2_airline AgentLoop via the `naive` manager, and the DB/COMMUNICATE subscores
# flow alongside it (reward_extra_info) so the default GDPO advantage can normalize each
# dimension separately (see adv_estimator below). Every knob is env-overridable so
# run_grpo_smoke.sh can reuse this.
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

# Per-run folder outputs/<run_name>_<timestamp>/{checkpoints,logs} — one timestamp
# so a run's weights and logs stay together and never collide across runs.
project_name=${PROJECT_NAME:-verl_grpo_airline}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_lora_grpo_tau2_airline}
logger=${LOGGER:-'["console","wandb"]'}
run_name=${RUN_NAME:-$experiment_name}
run_timestamp=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
run_dir=${RUN_DIR:-"$HERE/outputs/${run_name}_${run_timestamp}"}
default_local_dir=${DEFAULT_LOCAL_DIR:-"$run_dir/checkpoints"}
logs_dir="$run_dir/logs"
mkdir -p "$logs_dir"

# Tee the full console log to logs/ while still streaming to the terminal (survives
# dropped SSH). exec-redirection (not a pipe) keeps $? intact for callers. LOG_FILE= disables.
LOG_FILE=${LOG_FILE:-"$logs_dir/train_${run_timestamp}.log"}
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[run_grpo] logging to $LOG_FILE"
fi

# Pin NCCL OOB bootstrap to loopback: auto-selecting ib0 hangs the per-rank
# allgather and deadlocks init (single node; data plane still uses NVLink/SHM).
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env (gpt-5 user simulator)}"
# Init checkpoint. Default = the merged SFT cold-start checkpoint, so `bash run_grpo.sh`
# realizes the SFT→GDPO combo: RL starts from the cold-started weights and the LoRA adapters
# train on top of the SFT-merged base (not base Qwen3). Override INIT_MODEL_PATH for a
# different init, e.g. INIT_MODEL_PATH="${QWEN3_8B_PATH}" for a base+GDPO control run.
MODEL_PATH="${INIT_MODEL_PATH:-$HERE/outputs/qwen3_8b_full_sft_tau2_airline_20260605_152720/hf_merged}"
[ -d "$MODEL_PATH" ] || { echo "[run_grpo] init checkpoint not found: $MODEL_PATH — set INIT_MODEL_PATH"; exit 1; }

NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
NNODES=${NNODES:-1}

TRAIN_FILES=${TRAIN_FILES:-data/tau3_airline/train.parquet}
VAL_FILES=${VAL_FILES:-data/tau3_airline/test.parquet}

train_batch_size=${TRAIN_BATCH_SIZE:-8}       # 40 tasks -> 5 steps/epoch; 40%8==0, real batch 8*n.
                                              # Floor ~8: RL-zero cold-start risks an all-uninformative step.
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8} # = train_batch_size -> 1 update/step, fully on-policy.
max_prompt_length=${MAX_PROMPT_LENGTH:-6144}  # full initial prompt (policy+14 tools+turn) ~4.8k
max_response_length=${MAX_RESPONSE_LENGTH:-12288}
# MUST be >= max_prompt_length + max_response_length or a packed seq won't fit under
# use_dynamic_bsz. Feeds actor.ppo / rollout.log_prob / ref.log_prob. Watch OOM.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}

actor_lr=${ACTOR_LR:-1e-5}
kl_loss_coef=${KL_LOSS_COEF:-0.01}
entropy_coeff=${ENTROPY_COEFF:-0}

# Advantage estimator. Default GDPO (Group reward-Decoupled Normalization): rather than
# normalizing the single DB×COMMUNICATE product (vanilla GRPO), it normalizes DB,
# COMMUNICATE, and their product as 3 separate dimensions within each group, then sums.
# This yields a gradient on a dimension that still varies even when the product is all-zero
# across a group (e.g. COMMUNICATE flips 0/1 while DB is stuck at 0 → product all 0 → GRPO
# sees no signal, GDPO still learns the COMMUNICATE dimension — exactly the SFT-rescued
# "must report numbers" tasks). The subscores are emitted by our AgentLoop into
# reward_extra_info → non_tensor_batch; the keys/weights below must match those names.
# Eval is unaffected (tau2 native product), so base/trained numbers stay comparable.
# Set ADV_ESTIMATOR=grpo to revert to single-scalar GRPO (the keys are then ignored).
# NOTE: ppo_trainer.yaml's `algorithm:` node is a struct-mode dict that does NOT declare
# gdpo_reward_keys/weights (they live only in the AlgoConfig dataclass), so the overrides
# below MUST use the `++` force-append prefix — a plain `algorithm.gdpo_reward_keys=` is
# rejected with "Key 'gdpo_reward_keys' is not in struct". GDPO reads them via .get().
adv_estimator=${ADV_ESTIMATOR:-gdpo}
gdpo_reward_keys=${GDPO_REWARD_KEYS:-'[db,comm,db_comm]'}
gdpo_reward_weights=${GDPO_REWARD_WEIGHTS:-'[1.0,1.0,1.0]'}

lora_rank=${LORA_RANK:-32}                       # plan §6.3 (r32/alpha64, ratio 2)
lora_alpha=${LORA_ALPHA:-64}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
rollout_n=${ROLLOUT_N:-8}                         # GRPO group size (plan §11.4: >=8)
max_assistant_turns=${MAX_ASSISTANT_TURNS:-24}
max_user_turns=${MAX_USER_TURNS:-12}
# verl default 256 truncates get_user_details JSON and drops `reservations`, failing
# the task; raise it so the agent can see the user's bookings.
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-2048}
agent_num_workers=${AGENT_NUM_WORKERS:-8}

total_epochs=${TOTAL_EPOCHS:-20}              # batch=8 -> 5 steps/epoch, so 20 epochs = 100 steps
# Save/eval every 10 steps -> ~10 curve points. Each eval runs 10 held-out tasks
# through tau2 + gpt-5, so it isn't free; drop TEST_FREQ to 5 for a finer curve.
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}
max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-8}       # keep 8 newest (steps 30..100) to pick the best eval step
val_before_train=${VAL_BEFORE_TRAIN:-False}
# Dump per-step train / held-out rollouts (one JSON line per sample) to SEPARATE
# dirs — verl names both <global_step>.jsonl, so a shared dir would clobber. *_DATA_DIR= disables.
rollout_data_dir=${ROLLOUT_DATA_DIR:-"$logs_dir/rollouts"}
val_data_dir=${VAL_DATA_DIR:-"$logs_dir/val_rollouts"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${adv_estimator}" \
    ++algorithm.gdpo_reward_keys="${gdpo_reward_keys}" \
    ++algorithm.gdpo_reward_weights="${gdpo_reward_weights}" \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger="${logger}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.total_epochs=${total_epochs} \
    trainer.rollout_data_dir="${rollout_data_dir}" \
    trainer.validation_data_dir="${val_data_dir}" \
    "$@"
