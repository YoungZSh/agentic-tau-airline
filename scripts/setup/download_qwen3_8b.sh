#!/usr/bin/env bash
# Download Qwen3-8B for the policy model (stage 1+). ~16GB.
# Dest + conda env from .env (QWEN3_8B_PATH / TAU2_ENV); first arg overrides dest.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
DEST="${1:-${QWEN3_8B_PATH:-$HERE/ckpts/Qwen3-8B}}"
ENV_NAME="${TAU2_ENV:-tau2verl}"
HF="$(conda run -n "$ENV_NAME" which hf 2>/dev/null || command -v hf)"
"$HF" download Qwen/Qwen3-8B --local-dir "$DEST"
echo "downloaded -> $DEST"
