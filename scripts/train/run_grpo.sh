#!/usr/bin/env bash
# GRPO + LoRA + multi-turn async rollout — Qwen3-8B on 2×A100 80GB (plan stage 2).
#
# Entry: verl.trainer.main_ppo (verl 0.9.0.dev), inheriting the default
# ppo_trainer.yaml; we only override what differs. Reward comes from our
# tau2_airline AgentLoop (reward_score -> rm_scores[-1]); the default `naive`
# reward manager returns those directly (no custom_reward_function needed).
#
# Every knob is env-overridable so scripts/train/run_grpo_smoke.sh can reuse this.
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

# tau2 user simulator (gpt-5) needs the API; surfaced here as a guard.
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env (gpt-5 user simulator)}"
MODEL_PATH="${QWEN3_8B_PATH:?set QWEN3_8B_PATH in .env}"

NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
NNODES=${NNODES:-1}

TRAIN_FILES=${TRAIN_FILES:-data/tau3_airline/train.parquet}
VAL_FILES=${VAL_FILES:-data/tau3_airline/test.parquet}

train_batch_size=${TRAIN_BATCH_SIZE:-30}      # 30 train tasks -> 1 GRPO step/epoch
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-15}
max_prompt_length=${MAX_PROMPT_LENGTH:-6144}  # full initial prompt (policy+14 tools+turn) ~4.8k
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

actor_lr=${ACTOR_LR:-3e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.01}
entropy_coeff=${ENTROPY_COEFF:-0}

lora_rank=${LORA_RANK:-16}                     # plan §6.3 first version
lora_alpha=${LORA_ALPHA:-32}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
rollout_n=${ROLLOUT_N:-8}                       # GRPO group size (plan §11.4: >=8)
max_assistant_turns=${MAX_ASSISTANT_TURNS:-12}
max_user_turns=${MAX_USER_TURNS:-12}
agent_num_workers=${AGENT_NUM_WORKERS:-8}

total_epochs=${TOTAL_EPOCHS:-30}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-False}

project_name=${PROJECT_NAME:-verl_grpo_airline}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_lora_tau2_airline}
logger=${LOGGER:-'["console","wandb"]'}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    "$@"
