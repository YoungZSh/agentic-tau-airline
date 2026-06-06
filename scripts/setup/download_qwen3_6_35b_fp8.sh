#!/usr/bin/env bash
# Download Qwen3.6-35B-A3B-FP8 for the user simulator (replaces gpt-5). ~37.5GB.
# Dest from arg or USER_SIM_PATH (.env); conda env from TAU2_ENV.
# Note: this repo is a multimodal (VL) checkpoint; we only use its text side.
#
# Uses aria2c (16 conns/file, resumable, retries) instead of the hf client:
# hf-mirror.com throttles/times-out the python client hard (saw it stall after
# 1/43 files); aria2c multi-connection is the mirror's officially-recommended,
# robust method. Already-complete files are skipped on resume.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
DEST="${1:-${USER_SIM_PATH:-$HERE/ckpts/Qwen3.6-35B-A3B-FP8}}"
ENV_NAME="${TAU2_ENV:-tau2verl}"
REPO="${USER_SIM_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"
REV="${USER_SIM_REV:-main}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# This box's HTTP proxy can't reach huggingface.co; hf-mirror.com is directly
# reachable only WITHOUT the proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

ARIA2="$(conda run -n "$ENV_NAME" which aria2c 2>/dev/null || command -v aria2c)"
mkdir -p "$DEST"
LIST="$DEST/.aria2-input.txt"

# Fetch the repo file list (with sizes) and emit an aria2c input file.
# Skip files already on disk at full size; delete wrong-size partials so aria2
# fetches them clean (no .aria2 control file exists for hf-client leftovers).
curl -s --max-time 60 "$ENDPOINT/api/models/$REPO?blobs=true" \
  | DEST="$DEST" python3 -c "
import sys, os, json
d = json.load(sys.stdin)
endpoint, repo, rev, dest = '$ENDPOINT', '$REPO', '$REV', os.environ['DEST']
kept = skipped = 0
for s in d.get('siblings', []):
    f = s['rfilename']; size = s.get('size')
    p = os.path.join(dest, f)
    if os.path.exists(p) and size and os.path.getsize(p) == size:
        skipped += 1; continue           # already complete -> leave it
    if os.path.exists(p) and not os.path.exists(p + '.aria2'):
        os.remove(p)                      # partial w/o aria2 ctrl -> redo clean
    print(f'{endpoint}/{repo}/resolve/{rev}/{f}')
    print(f'  out={f}')
    kept += 1
sys.stderr.write(f'queued {kept} files, skipped {skipped} already-complete\n')
" > "$LIST"
echo "file list -> $LIST ($(grep -c '^http' "$LIST") files to fetch)"

"$ARIA2" \
  --dir="$DEST" \
  --input-file="$LIST" \
  --continue=true \
  --max-connection-per-server=16 \
  --split=16 \
  --min-split-size=1M \
  --max-concurrent-downloads=5 \
  --max-tries=20 \
  --retry-wait=5 \
  --timeout=60 \
  --connect-timeout=30 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --console-log-level=warn \
  --summary-interval=15

echo "downloaded -> $DEST"
