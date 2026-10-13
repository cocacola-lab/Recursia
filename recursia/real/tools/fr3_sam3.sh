#!/usr/bin/env bash
# Standalone vision environment; does not connect to or move the robot.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3_ENV="${ASPIRE_SAM3_ENV:-$HOME/.local/share/aspire/sam3-venv}"
PYTHON="$SAM3_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing SAM3 environment: $SAM3_ENV" >&2
  exit 2
fi
cd "$ROOT"
case "${1:-start}" in
  download-modelscope)
    exec "$PYTHON" tools/vision/download_sam3_modelscope.py
    ;;
  login)
    exec "$SAM3_ENV/bin/hf" auth login
    ;;
  download)
    exec "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
try:
    path = snapshot_download(
        "facebook/sam3",
        revision="3c879f39826c281e95690f02c7821c4de09afae7",
        allow_patterns=["*.json", "*.txt", "model.safetensors", "LICENSE", "README.md"],
    )
except GatedRepoError:
    raise SystemExit(
        "SAM3 weights require access at https://huggingface.co/facebook/sam3. "
        "After access is granted, run: bash tools/fr3_sam3.sh login"
    )
print(path)
PY
    ;;
  start)
    if [[ -z "${SAM3_MODEL_DIR:-}" && -f "$HOME/.local/share/aspire/models/sam3/modelscope-manifest.json" ]]; then
      export SAM3_MODEL_DIR="$HOME/.local/share/aspire/models/sam3"
    fi
    exec "$PYTHON" tools/vision/serve_sam3.py --host 127.0.0.1 --port 6767 --preload
    ;;
  status)
    exec curl --fail --max-time 5 -sS http://127.0.0.1:6767/health
    ;;
  *)
    echo "Usage: bash tools/fr3_sam3.sh {login|download|download-modelscope|start|status}" >&2
    exit 2
    ;;
esac
