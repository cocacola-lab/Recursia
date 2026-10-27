#!/usr/bin/env bash
# Generic RSI stage runner (from the reference run_plan_90.sh). Usage: stage.sh <plan> <phase>
# Env (set by rsi.py run): RSI_ROUND, RSI_GPUS ("3 4 5"), RSI_START_CAP, RSI_TIMEOUT, RECURSIA_SIM_ROOT. Workers hold no credential.
set -euo pipefail
PLAN=$1; PHASE=$2
R=${RSI_ROUND:?}; ROOT=${RECURSIA_SIM_ROOT:?}; CAP=${RSI_START_CAP:?}; TO=${RSI_TIMEOUT:-1500}
read -r -a GPUS <<< "${RSI_GPUS:?}"
LEDGER="$R/logs/starts.txt"; LEDGER_LOCK="$R/logs/starts.lock"
mkdir -p "$R/logs"; touch "$LEDGER"
[[ ! -e "$R/logs/plan_${PHASE}.log" ]] || { echo "Existing phase log: $PHASE" >&2; exit 2; }
: > "$R/logs/plan_${PHASE}.log"
echo $$ > "$R/logs/stage_${PHASE}.pid"   # stop with: kill $(cat logs/stage_<phase>.pid)  (never pkill -f)
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
        unset "BUSY[$g]"; FREE=$g; break 2
      fi
    done
    sleep 5
  done
  # do not start an episode while the VLM provider is unreachable
  until env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u OPENROUTER_API_KEY "$ROOT/.venv-libero/bin/python3" "$R/code/vlm_probe.py"; do
    echo "VLM_WAIT $(date -Is) before $ARM $SHORT $SEED" >> "$R/logs/plan_${PHASE}.log"; sleep 60
  done
  exec 8>>"$LEDGER_LOCK"; flock 8
  N=$(wc -l < "$LEDGER")
  if (( N >= CAP )); then flock -u 8; echo "CAP REACHED ($N)" >> "$R/logs/plan_${PHASE}.log"; exit 2; fi
  echo "$(date -Is) $PHASE $ARM $SHORT $SEED gpu$FREE" >> "$LEDGER"
  flock -u 8
  bash "$R/code/attempt.sh" "$ARM" "$SHORT" "$SEED" "$PHASE" "$FREE" "$TO" >> "$R/logs/plan_${PHASE}.log" 2>&1 &
  BUSY[$FREE]=$!
  count=$((count+1))
  sleep 3
done < "$PLAN"
for g in "${GPUS[@]}"; do
  pid=${BUSY[$g]:-}
  if [[ -n "$pid" ]]; then wait "$pid" || fail=1; fi
done
echo "PLAN DONE phase=$PHASE started=$count failures=$fail" >> "$R/logs/plan_${PHASE}.log"
rm -f "$R/logs/stage_${PHASE}.pid"
[[ "$fail" == 0 ]]
