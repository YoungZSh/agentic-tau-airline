#!/usr/bin/env bash
# Download Qwen3-8B for the policy model (stage 1+). ~16GB.
set -euo pipefail
DEST="${1:-/home/yzs/ckpts/Qwen3-8B}"
HF="/home/yzs/miniconda3/envs/tau2verl/bin/hf"
[ -x "$HF" ] || HF="/home/yzs/miniconda3/envs/verl/bin/hf"
"$HF" download Qwen/Qwen3-8B --local-dir "$DEST"
echo "downloaded -> $DEST"
