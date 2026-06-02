#!/usr/bin/env bash
# Stage 1 GO/NO-GO: base Qwen3-8B rollout on the airline TRAIN split via tau2's
# official runner. Starts a local vLLM OpenAI server (hermes tool parser), runs
# the eval, then tears the server down.
#
#   bash scripts/eval/run_base_rollout.sh [num_trials] [extra run_tau2_eval args...]
#
# Smoke a couple of tasks first:
#   bash scripts/eval/run_base_rollout.sh 2 --task-ids 0 1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
ENV_NAME="${TAU2_ENV:-tau2verl}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env (gpt-5 user simulator)}"
MODEL_PATH="${QWEN3_8B_PATH:?set QWEN3_8B_PATH in .env}"

NUM_TRIALS="${1:-8}"; shift || true
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
TP="${ROLLOUT_TP:-2}"
LORA_PATH="${LORA_PATH:-}"     # set to a trained adapter dir for held-out eval
SPLIT="${SPLIT:-train}"
OUT="${OUT:-notes/stage1_base_rollout.md}"

LORA_ARGS=()
[ -n "$LORA_PATH" ] && LORA_ARGS=(--enable-lora --lora-modules "${MODEL_NAME}=${LORA_PATH}")

echo ">>> starting vLLM server (model=$MODEL_PATH tp=$TP port=$PORT)"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"   # lower it on a shared/partly-occupied GPU
conda run -n "$ENV_NAME" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$TP" --port "$PORT" \
    --gpu-memory-utilization "$VLLM_GPU_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    "${LORA_ARGS[@]}" > outputs/vllm_server.log 2>&1 &
VLLM_PID=$!
trap 'echo ">>> stopping vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

echo ">>> waiting for vLLM /v1/models ..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then echo "ready"; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; see outputs/vllm_server.log"; exit 1; fi
    sleep 5
done

echo ">>> running tau2 airline eval (split=$SPLIT trials=$NUM_TRIALS)"
conda run -n "$ENV_NAME" python -m tau2_airline_verl.evaluation.run_tau2_eval \
    --split "$SPLIT" --num-trials "$NUM_TRIALS" \
    --model-name "$MODEL_NAME" --api-base "http://localhost:${PORT}/v1" \
    --out "$OUT" "$@"
