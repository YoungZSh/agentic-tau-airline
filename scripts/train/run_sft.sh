#!/usr/bin/env bash
# Full-parameter (non-LoRA) multi-turn SFT cold-start — Qwen3-8B, pure ZeRO-3 (FSDP).
# Entry: verl.trainer.sft_trainer over sft_trainer_engine.yaml (engine=fsdp). The
# dataset is our KeepThinkMultiTurnSFTDataset (data.custom_cls) which patches the
# Qwen3 chat template to keep <think> on every assistant turn and injects the
# airline tool schemas — so the SFT token layout matches RL rollout. Every knob is
# env-overridable so run_sft_smoke.sh can reuse this.
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

project_name=${PROJECT_NAME:-verl_sft_airline}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_full_sft_tau2_airline}
logger=${LOGGER:-'["console","wandb"]'}
run_name=${RUN_NAME:-$experiment_name}
run_timestamp=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
run_dir=${RUN_DIR:-"$HERE/outputs/${run_name}_${run_timestamp}"}
default_local_dir=${DEFAULT_LOCAL_DIR:-"$run_dir/checkpoints"}
logs_dir="$run_dir/logs"
mkdir -p "$logs_dir"

LOG_FILE=${LOG_FILE:-"$logs_dir/sft_${run_timestamp}.log"}
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[run_sft] logging to $LOG_FILE"
fi

# Pin NCCL OOB bootstrap to loopback (auto-selecting ib0 deadlocks init on this box).
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}

# Cut CUDA allocator fragmentation at the 32k backward peak (a prior run stranded
# ~8.79GB as reserved-but-unallocated right before OOMing). expandable_segments lets
# the allocator grow/reuse one segment instead of orphaning fixed blocks — often the
# difference between OOM and fit when the peak sits right at the 80GB ceiling.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# TiledMLP: chunk the MLP fwd/bwd over tokens to cut the activation peak. verl's
# FSDP engine never wires model.tiled_mlp through to apply_monkey_patch (only the
# megatron/veomni backends do), so it's enabled via our custom_cls
# (keepthink_dataset.py), which reads these env vars at import. Default OFF: since
# verl doesn't wire it on this path (so it's unvalidated end-to-end here), we stay
# on the conservative route — fused kernels only, which IS natively wired — and
# leave tiled MLP opt-in (TILED_MLP=True). export so torchrun workers inherit it.
export TILED_MLP=${TILED_MLP:-False}
export TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-4}

TAU2_ENV=${TAU2_ENV:-tau2verl}
MODEL_PATH="${QWEN3_8B_PATH:?set QWEN3_8B_PATH in .env}"
# Pure ZeRO-3 data parallel: any GPU count works (no num_heads divisibility
# constraint — that only applies to Ulysses SP, which is OFF; see sp_size below).
# More GPUs = thinner static-state shard = more headroom for activations. Pick the
# free GPUs via CUDA_VISIBLE_DEVICES (e.g. 4,6 for two, or 4,6,7 for three).
NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}
NNODES=${NNODES:-1}

TRAIN_FILES=${TRAIN_FILES:-data/tau2_airline_sft/train.parquet}
VAL_FILES=${VAL_FILES:-data/tau2_airline_sft/val.parquet}
CUSTOM_DS_PATH=${CUSTOM_DS_PATH:-"$HERE/src/tau2_airline_verl/sft/keepthink_dataset.py"}
CUSTOM_DS_NAME=${CUSTOM_DS_NAME:-KeepThinkMultiTurnSFTDataset}

# Conversations are long (median ~15k tok). Dialogs > 32k are already dropped at
# build time (build_sft_parquet --max_tokens); max_length re-caps + right-truncates
# the rare residual as a second guard.
max_length=${MAX_LENGTH:-32768}
truncation=${TRUNCATION:-right}
pad_mode=${PAD_MODE:-no_padding}
# Batch: with pure ZeRO-3, DP = NGPUS (each GPU takes different convs). train_batch_size
# is the global batch (gradient-accumulated across DP ranks); dynamic bsz packs the
# variable-length convs by token up to max_token_len_per_gpu. No SP, so each GPU holds
# the full sequence's activations (≤32k) — gradient checkpointing keeps the peak in check.
train_batch_size=${TRAIN_BATCH_SIZE:-16}        # ~54 steps/epoch over ~870 dialogs
micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-$max_length}
train_max_samples=${TRAIN_MAX_SAMPLES:--1}
val_max_samples=${VAL_MAX_SAMPLES:--1}

# Cold-start full-param SFT: low lr + few epochs to avoid forgetting Qwen3's base
# thinking/tool ability (needed by the downstream RL).
lr=${LR:-1e-5}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-cosine}
lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO:-0.03}
weight_decay=${WEIGHT_DECAY:-0.0}
total_epochs=${TOTAL_EPOCHS:-2}
# Full-param checkpoint is ~92GB each (model + fp32 optim), vs grpo's ~32GB LoRA —
# so do NOT copy grpo's save_freq=10 (that would write ~10×92GB and fill the disk).
# Default: save only the final model (for the downstream RL to load). To pick the
# best epoch instead, set SAVE_FREQ≈steps/epoch (~54) + TEST_FREQ to watch val loss;
# max_ckpt_to_keep caps how many of those 92GB checkpoints survive.
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}
max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-2}

# Memory for full-param 8B + 32k, NO CPU offload (per requirement). The 128GB of
# static state (params 16 + grad 16 + fp32 AdamW 96) is sharded by FSDP ZeRO-3 to
# ~42.7GB/GPU on 3 GPUs / ~64GB/GPU on 2 (that is sharding, not offload). With SP off
# each GPU holds the full-sequence activations, so the levers against OOM are
# max_token_len_per_gpu (cap the per-GPU packed tokens) and/or more GPUs — never offload.
# Ulysses SP is OFF (sp_size=1): pure ZeRO-3, no num_heads%sp_size constraint. Only set
# SP_SIZE>1 to re-enable SP, and then it MUST divide num_heads (32 → 1,2,4,8,16,32).
sp_size=${SP_SIZE:-1}
fsdp_strategy=${FSDP_STRATEGY:-fsdp}
param_offload=${PARAM_OFFLOAD:-false}
optimizer_offload=${OPTIMIZER_OFFLOAD:-false}
enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True}
enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-False}

# Fused linear cross-entropy: never materialize the full [tokens × vocab] logits
# (~10GB at 32k — the exact tensor that OOM'd the backward). Numerically verified
# equivalent to the standard path (logprob/entropy/grad diff ~1e-6, well under 1e-4).
# On by default; the FSDP engine wires this through apply_monkey_patch.
use_fused_kernels=${USE_FUSED_KERNELS:-True}

# Run inside the tau2verl conda env (self-contained, mirrors the eval scripts).
# Explicit master port (not --standalone) so this can coexist with other torchrun
# jobs on a shared box; CUDA_VISIBLE_DEVICES selects which GPUs to use.
conda run -n "$TAU2_ENV" --no-capture-output \
torchrun --nnodes="${NNODES}" --nproc_per_node="${NGPUS_PER_NODE}" \
    --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29577}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.custom_cls.path="${CUSTOM_DS_PATH}" \
    data.custom_cls.name="${CUSTOM_DS_NAME}" \
    data.ignore_input_ids_mismatch=True \
    data.pad_mode="${pad_mode}" \
    data.max_length=${max_length} \
    data.truncation="${truncation}" \
    data.train_batch_size=${train_batch_size} \
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    data.use_dynamic_bsz=${use_dynamic_bsz} \
    data.max_token_len_per_gpu=${max_token_len_per_gpu} \
    data.train_max_samples=${train_max_samples} \
    data.val_max_samples=${val_max_samples} \
    model.path="${MODEL_PATH}" \
    model.lora_rank=0 \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=${enable_gradient_checkpointing} \
    model.enable_activation_offload=${enable_activation_offload} \
    model.use_fused_kernels=${use_fused_kernels} \
    engine=fsdp \
    engine.strategy="${fsdp_strategy}" \
    engine.ulysses_sequence_parallel_size=${sp_size} \
    engine.param_offload=${param_offload} \
    engine.optimizer_offload=${optimizer_offload} \
    optim.lr=${lr} \
    optim.weight_decay=${weight_decay} \
    optim.lr_scheduler_type=${lr_scheduler_type} \
    optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger="${logger}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.total_epochs=${total_epochs} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.max_ckpt_to_keep=${max_ckpt_to_keep} \
    "$@"
