#!/usr/bin/env bash
# Frozen source-only N7 stage. Usage: run_plan_90.sh <plan> <phase> [timeout].
set -euo pipefail
PLAN=$1; PHASE=$2; TO=${3:-1500}
ROOT=$RECURSIA_SIM_ROOT
R=${RSI_EXPERIMENT_DIR:?Set RSI_EXPERIMENT_DIR to the N7 continuation directory}
[[ "$R" == "$ROOT/outputs/rsi_libero90_N20C_closedloop_long100_20261016" ]]
LEDGER="$ROOT/outputs/rsi_libero90_20260924/logs/starts.txt"
LEDGER_LOCK="$ROOT/outputs/rsi_libero90_20260924/logs/starts.lock"
CAP=20000; GPUS=(3 4 5 6 7)   # shared start cap 12000 -> 14000 -> 20000 (user, 2026-10-07: "increase the ledger upperbound")
mkdir -p "$R/logs"
[[ ! -e "$R/logs/plan_${PHASE}.log" ]] || { echo "Existing phase log: $PHASE" >&2; exit 2; }
: > "$R/logs/plan_${PHASE}.log"
declare -A BUSY
count=0; fail=0
while read -r ARM SHORT SEED; do
  [[ -z "${ARM:-}" || "$ARM" == \#* ]] && continue
  while :; do
    for g in "${GPUS[@]}"; do
      pid=${BUSY[$g]:-}
      if [[ -z "$pid" ]]; then FREE=$g; break 2; fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || fail=1
        unset "BUSY[$g]"
        FREE=$g; break 2
      fi
    done
    sleep 5
  done
  until env -u ANTHROPIC_API_KEY "$ROOT/.venv-libero/bin/python3" "$R/code/vlm_probe.py"; do echo "VLM_WAIT $(date -Is) before $ARM $SHORT $SEED" >> "$R/logs/plan_${PHASE}.log"; sleep 60; done   # N15 infra amendment 2026-10-10: do not start an episode while the VLM provider is unreachable
  exec 8>>"$LEDGER_LOCK"; flock 8
  N=$(wc -l < "$LEDGER")
  if (( N >= CAP )); then
    flock -u 8
    echo "CAP REACHED ($N)" >> "$R/logs/plan_${PHASE}.log"
    exit 2
  fi
  ENTRY="$(date -Is) $PHASE $ARM $SHORT $SEED gpu$FREE"
  echo "$ENTRY" >> "$LEDGER"
  echo "$ENTRY" >> "$R/logs/starts.txt"
  flock -u 8
  bash "$R/code/run_attempt_orch.sh" "$ARM" "$SHORT" "$SEED" "$PHASE" "$FREE" "$TO" >> "$R/logs/plan_${PHASE}.log" 2>&1 &
  BUSY[$FREE]=$!
  count=$((count+1))
  sleep 3
done < "$PLAN"
for g in "${GPUS[@]}"; do
  pid=${BUSY[$g]:-}
  if [[ -n "$pid" ]]; then wait "$pid" || fail=1; fi
done
echo "PLAN DONE phase=$PHASE started=$count failures=$fail" >> "$R/logs/plan_${PHASE}.log"
[[ "$fail" == 0 ]]
