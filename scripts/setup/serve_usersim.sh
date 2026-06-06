#!/usr/bin/env bash
# Serve the local user-sim model (Qwen3.6-35B-A3B-FP8) as an OpenAI-compatible
# endpoint via vllm 0.19.1 (cu128). Replaces the gpt-5 user simulator.
# Env overrides: USER_SIM_PATH / USER_SIM_GPUS / USER_SIM_PORT / USER_SIM_SERVED_NAME.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

MODEL="${USER_SIM_PATH:-$HERE/ckpts/Qwen3.6-35B-A3B-FP8}"
GPUS="${USER_SIM_GPUS:-3,4}"
PORT="${USER_SIM_PORT:-8011}"
NAME="${USER_SIM_SERVED_NAME:-qwen3.6-usersim}"
VLLM_BIN="/ssd/home/zc/miniconda3/envs/vllm/bin/vllm"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_SOCKET_IFNAME=lo          # avoid ib0 OOB hang on multi-GPU collectives
export VLLM_LOGGING_LEVEL=INFO

# CUDA graphs ON by default (much higher decode throughput for the MoE under load).
# Set USER_SIM_EAGER=1 for a faster-startup smoke (skips graph capture).
# max-model-len = prompt budget + up to 8k NEW tokens (vllm has no separate output
# cap; total is bounded here, per-request output by `max_tokens`). Well below the
# native 256K so KV-cache profiling stays sane.
EAGER_FLAG=""
[ "${USER_SIM_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
MAXLEN="${USER_SIM_MAX_MODEL_LEN:-16384}"

# Reasoning parser: the user-sim now runs thinking ON (airline_interaction.py sends
# enable_thinking=True). vllm must split the <think>...</think> block into
# `reasoning_content` so the OpenAI `content` field is the clean reply — that is what
# tau2 reads as the user turn. Without this the raw <think> would leak into the
# policy's prompt. `qwen3` is the parser for Qwen3.x thinking models; it also strips
# the empty <think></think> emitted when a request disables thinking (the NL judge),
# so that path stays correct too. Set USER_SIM_REASONING_PARSER="" to disable.
REASONING_PARSER="${USER_SIM_REASONING_PARSER:-qwen3}"
PARSER_FLAG=""
[ -n "$REASONING_PARSER" ] && PARSER_FLAG="--reasoning-parser $REASONING_PARSER"

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.93 \
  $EAGER_FLAG \
  $PARSER_FLAG \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "$PORT"
