#!/usr/bin/env bash
# Install only the perception environment; no camera or robot connection.
set -euo pipefail
SAM3_ENV="${ASPIRE_SAM3_ENV:-$HOME/.local/share/aspire/sam3-venv}"
BASE_PYTHON="${ASPIRE_VISION_PYTHON:-$(command -v python3)}"
REAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -x "$SAM3_ENV/bin/python" ]] || "$BASE_PYTHON" -m venv "$SAM3_ENV"
"$SAM3_ENV/bin/python" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cu128
"$SAM3_ENV/bin/python" -m pip install -r "$REAL_ROOT/tools/vision/requirements-sam3.txt" \
  'numpy>=1.26,<2' pillow pyyaml 'opencv-python-headless<4.12' huggingface-hub modelscope
printf 'Installed SAM3 environment: %s\n' "$SAM3_ENV"
