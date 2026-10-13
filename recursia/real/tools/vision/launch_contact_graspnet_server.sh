#!/usr/bin/env bash
set -euo pipefail
REAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${ASPIRE_CONTACT_GRASPNET_DIR:-$HOME/.local/share/aspire/contact-graspnet}"
exec "$PREFIX/venv/bin/python" "$REAL_ROOT/tools/vision/serve_contact_graspnet.py" \
  --source "$PREFIX/source" "$@"
