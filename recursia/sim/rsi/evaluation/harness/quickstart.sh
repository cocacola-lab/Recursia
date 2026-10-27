#!/usr/bin/env bash
# Quick start: LIBERO-Long-Pro evaluation of the frozen N20 stack (default: closed loop N20C, interim 100 episodes).
#   export ANTHROPIC_API_KEY=...            # held only by the loopback proxy process; workers never see it
#   bash rsi/evaluation/harness/quickstart.sh [closed|oneshot|rule] [interim|full]
# Env: GPUS="3 4 5 6 7" (simulation workers), OUT=<experiment dir>, CAP_USD=35 (VLM dollar cap).
# Steps: perception preflight -> setup (frozen copy + offline tests) -> proxy -> 1-episode smoke -> detached run.
set -euo pipefail
METHOD=${1:-closed}; SCALE=${2:-interim}
HERE=$(cd "$(dirname "$0")" && pwd)
export RECURSIA_SIM_ROOT=${RECURSIA_SIM_ROOT:-$(cd "$HERE/../../.." && pwd)}
cd "$RECURSIA_SIM_ROOT"
PY=.venv-libero/bin/python3; T="$HERE/rsi_eval.py"
declare -A ARM=([closed]=N20C [oneshot]=N20O [rule]=N20)
OUT=${OUT:-$RECURSIA_SIM_ROOT/outputs/rsi_zs_eval_${ARM[$METHOD]}_${SCALE}_$(date +%Y%m%d_%H%M)}
read -r -a GPU_LIST <<< "${GPUS:-3 4 5 6 7}"

for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$p/health" || true)
  [[ -n "$code" && "$code" != "000" ]] || { echo "perception server on port $p is down: start them first (QUICKSTART.md step 2)"; exit 2; }
done
[[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "export ANTHROPIC_API_KEY first (only the proxy process uses it)"; exit 2; }

"$PY" "$T" setup --out "$OUT" --method "$METHOD" --scale "$SCALE" --gpus "${GPU_LIST[@]}" --cap-usd "${CAP_USD:-35}"
"$PY" "$T" start-proxy --out "$OUT"
"$PY" "$T" smoke --out "$OUT"
( setsid nohup env -u ANTHROPIC_API_KEY "$PY" "$T" run --out "$OUT" > "$OUT/logs/run.out" 2>&1 & echo $! > "$OUT/logs/run.pid" )
cat <<EOF

RUNNING  $OUT  (${ARM[$METHOD]}, $SCALE)
  progress : wc -l < $OUT/attempts_log_${SCALE}_${ARM[$METHOD]}.csv    (header + one line per finished episode)
  log      : tail -f $OUT/logs/run.out
  stop     : kill \$(cat $OUT/logs/run.pid)        (never pkill -f)
When it finishes (STAGE_DONE in logs/run.out):
  $PY $T rerun-infra --out $OUT     # only if episodes hit proxy transport errors (predeclared, once)
  $PY $T summarize   --out $OUT     # results/summary.json + results/REPORT_auto.md
  $PY $T stop-proxy  --out $OUT
EOF
