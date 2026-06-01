#!/usr/bin/env bash
# Install tau2-bench (editable) into the cloned conda env `tau2verl`.
# Uses pip (NOT uv) to avoid re-resolving verl's torch/sglang/vllm.
set -euo pipefail

ENV_PY="/home/yzs/miniconda3/envs/tau2verl/bin/python"
ENV_PIP="/home/yzs/miniconda3/envs/tau2verl/bin/pip"
TAU2_DIR="/root/siton-tmp/yzs/tau2-bench"

if [ ! -x "$ENV_PY" ]; then
  echo "ERROR: $ENV_PY not found. Run: conda create --clone verl -n tau2verl -y" >&2
  exit 1
fi

if [ ! -d "$TAU2_DIR" ]; then
  git clone https://github.com/sierra-research/tau2-bench.git "$TAU2_DIR"
fi

echo ">> pip install -e tau2-bench into tau2verl"
"$ENV_PIP" install -e "$TAU2_DIR"

echo ">> pip install -e this project (tau2_airline_verl)"
"$ENV_PIP" install -e "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo ">> pip check (look for litellm-induced downgrades of openai/httpx/pydantic)"
"$ENV_PIP" check || echo "WARN: pip check reported issues — review above."

echo ">> import smoke test"
"$ENV_PY" -c "import verl; from tau2.domains.airline.environment import get_environment, get_tasks; print('ok: verl + tau2 import')"
