#!/usr/bin/env bash
# Stop the perception servers started by start_perception_servers.sh (pid files only; never pkill -f).
set -uo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in sam3 graspnet pyroki; do
  f="$SIM_ROOT/.perception/$name.pid"
  [[ -f "$f" ]] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" 2>/dev/null || kill "$pid"; echo "stopped $name ($pid)"; fi
  rm -f "$f"
done
