#!/usr/bin/env bash
# Start the three perception servers used by every LIBERO episode: SAM3 (8114), Contact-GraspNet (8115), PyRoki IK (8116).
# Run it in a persistent terminal (tmux). Processes are tracked by pid files in .perception/ (stop with stop_perception_servers.sh).
#   bash scripts/start_perception_servers.sh [--gpu-sam3 0] [--gpu-graspnet 1] [--python .venv-libero/bin/python3]
set -euo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIM_ROOT"
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
GPU_SAM3=0; GPU_GRASPNET=1; PYTHON_BIN="${RECURSIA_PERCEPTION_PYTHON:-.venv-libero/bin/python3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-sam3) GPU_SAM3="$2"; shift 2 ;;
    --gpu-graspnet) GPU_GRASPNET="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: $PYTHON_BIN not found (create .venv-libero first; README § Setup)"; exit 1; }
bash scripts/apply_contact_graspnet_patch.sh
mkdir -p .perception
for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$p/health" || true)
  if [[ -n "$code" && "$code" != "000" ]]; then echo "ERROR: port $p is already in use (stop_perception_servers.sh, or another user's servers)"; exit 1; fi
done
start() {   # name gpu args...
  local name=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$PYTHON_BIN" -u "$@" > ".perception/$name.log" 2>&1 &
  echo $! > ".perception/$name.pid"; echo "started $name (pid $!, log .perception/$name.log)"
}
start sam3 "$GPU_SAM3" cap/serving/launch_sam3_server.py --device cuda --port 8114 --host 127.0.0.1
start graspnet "$GPU_GRASPNET" cap/serving/launch_contact_graspnet_server.py --port 8115 --host 127.0.0.1
start pyroki "" cap/serving/launch_pyroki_server.py --port 8116 --host 127.0.0.1 --robot panda_description --target-link panda_hand
echo "waiting for the models to load (up to 300 s)..."
for i in $(seq 1 60); do
  up=0
  for p in 8114 8115 8116; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$p/health" || true)
    [[ -n "$code" && "$code" != "000" ]] && up=$((up+1))
  done
  [[ $up == 3 ]] && { echo "PERCEPTION_OK (8114 SAM3, 8115 Contact-GraspNet, 8116 PyRoki; 404 on /health = up)"; exit 0; }
  sleep 5
done
echo "ERROR: not all servers came up; see .perception/*.log"; exit 1
