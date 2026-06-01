#!/usr/bin/env bash
# Set up the tau2verl environment. This records the install path *verified on this
# machine* (8x A800-80GB, restricted outbound network). Neither verl nor tau2 source
# is modified — both are read-only git submodules under third_party/, installed editable.
#
# Network facts on this host (why the steps look unusual):
#   - PyPI works via the Tsinghua mirror (pip is preconfigured) — fast & stable.
#   - github.com / download.pytorch.org / readthedocs are UNRELIABLE via the proxy,
#     so torch comes from PyPI (NOT download.pytorch.org) and submodules were mirror-cloned.
#   - No nvcc on this host => flash-attn can't be compiled; we reuse a prebuilt wheel.
#
# Prereq (do NOT clone the verl env — make a fresh empty one):
#   conda create -n tau2verl python=3.12 -y
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # scripts/setup/ -> repo root
cd "$HERE"
[ -f .env ] && { set -a; . ./.env; set +a; }

ENV_NAME="${TAU2_ENV:-tau2verl}"
PY="$(conda run -n "$ENV_NAME" which python 2>/dev/null || true)"
[ -n "$PY" ] || { echo "ERROR: create the env first: conda create -n $ENV_NAME python=3.12 -y" >&2; exit 1; }
PIP="$(dirname "$PY")/pip"

echo ">> [1/8] ensure submodules present (third_party/{tau2-bench,verl})"
# NOT --recursive: verl's nested 'recipe' submodule isn't needed for GRPO/PPO and github is unreachable.
git submodule update --init third_party/tau2-bench third_party/verl || \
  echo "WARN: submodule update issue — if empty, mirror-clone via https://gh-proxy.com/https://github.com/..."

echo ">> [2/8] torch stack (from PyPI mirror — download.pytorch.org is unreachable here)"
"$PIP" install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

echo ">> [3/8] flash-attn 2.8.1 (prebuilt; no nvcc on this host so it can't compile)"
if "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "   flash_attn already present"
else
  SP="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
  VERL_SP=/ssd/home/zc/miniconda3/envs/verl/lib/python3.12/site-packages
  if [ -d "$VERL_SP/flash_attn" ]; then
    echo "   reusing prebuilt flash_attn from verl env (same py3.12/torch2.8/cu12 ABI)"
    cp -r "$VERL_SP"/flash_attn "$SP"/
    cp -r "$VERL_SP"/flash_attn-*.dist-info "$SP"/ 2>/dev/null || true
    cp "$VERL_SP"/flash_attn_*.so "$SP"/ 2>/dev/null || true
  else
    echo "   ERROR: no prebuilt flash_attn found. Install the matching wheel (no compile), e.g.:" >&2
    echo "     pip install https://gh-proxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" >&2
    exit 1
  fi
fi

echo ">> [4/8] vllm 0.11.0 (rollout backend)"
"$PIP" install vllm==0.11.0

echo ">> [5/8] verl (editable, from third_party submodule)"
"$PIP" install -e third_party/verl

echo ">> [6/8] pin numpy/transformers to the combo verified on this machine"
# verl 0.9-dev conservatively pins numpy<2, but numpy 2.2.6 is what actually runs with
# vllm 0.11 (cupy/opencv need >=2); transformers 4.56.1 is the verified version vllm accepts.
"$PIP" install "numpy==2.2.6" "transformers==4.56.1"

echo ">> [7/8] tau2-bench + this package (editable) + SOCKS support for litellm/httpx"
"$PIP" install -e third_party/tau2-bench
"$PIP" install -e .
"$PIP" install "httpx[socks]" socksio

echo ">> [8/8] smoke test"
"$PY" -c "import verl; from tau2.domains.airline.environment import get_environment, get_tasks; import tau2_airline_verl; print('ok: verl + tau2 + tau2_airline_verl')"
echo "DONE.  (pip check will warn 'verl requires numpy<2' — expected & harmless on this stack.)"
