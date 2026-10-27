#!/bin/bash
# usage: run_attempt_90.sh <arm> <task_short> <seed> <phase> <gpu> [timeout_s]
# Suite and full task come from task_manifest.json[task_short]. Unique task-qualified IDs:
#   stage_<phase>/attempts/<arm>__<task_short>__s<seed>__<uid> ; logs/<phase>__<arm>__<task_short>__s<seed>__<uid>.log
set -u
# N20C: copy of run_attempt_90.sh (N20 interim) with the experiment path and runner iso_runner_orch.py; RSI_VLM_URL passes through.
ARM=$1; SHORT=$2; SEED=$3; PHASE=$4; GPU=$5; TO=${6:-1200}
ROOT="$RECURSIA_SIM_ROOT"; EXP="${RSI_EXPERIMENT_DIR:?Set RSI_EXPERIMENT_DIR}"
[[ "$EXP" == "$ROOT/outputs/rsi_libero90_N20C_closedloop_long100_20261016" ]]
read -r SUITE TASK < <("$ROOT/.venv-libero/bin/python3" -c "import json;t=json.load(open('$EXP/task_manifest.json'))['tasks']['$SHORT'];print(t['suite'],t['task'])") || { echo "unknown task_short $SHORT"; exit 3; }
CODE="$EXP/programs/${ARM}.py"; [[ -f "$CODE" ]] || { echo "missing $CODE"; exit 3; }
UID8=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)
ID="${ARM}__${SHORT}__s${SEED}__${UID8}"
export RECURSIA_SIM_ROOT="$ROOT" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$ROOT" JAX_PLATFORMS=cpu EXPERIMENT_DIR="$EXP"
OUTDIR="$EXP/stage_${PHASE}/attempts/$ID"; mkdir -p "$(dirname $OUTDIR)" "$EXP/logs"
[[ -e "$OUTDIR" ]] && { echo "attempt dir exists: $OUTDIR"; exit 4; }
LOG="$EXP/logs/${PHASE}__${ID}.log"; T0=$(date +%s)
setsid env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u OPENROUTER_API_KEY timeout -k 20 "$TO" "$ROOT/.venv-libero/bin/python3" "$EXP/code/iso_runner_orch.py" --arm "$ARM" --suite "$SUITE" --task "$TASK" --seed "$SEED" --code "$CODE" --outdir "$OUTDIR" --phase "$PHASE" --uid "$UID8" > "$LOG" 2>&1
RC=$?; T1=$(date +%s)
DONE=$(grep -h "^ATTEMPT_DONE" "$LOG" | tail -1 | cut -c14-)
TC=$(echo "$DONE" | grep -oP '"official_task_completed": \K\d' || echo "NA")
SRC=$(echo "$DONE" | grep -oP '"sandbox_rc": \K-?\d+' || echo "NA")
SHA=$(sha256sum "$CODE" | cut -c1-12)
CSV="$EXP/attempts_log_${PHASE}.csv"
exec 9>>"$EXP/logs/csv.lock"; flock 9
if [[ ! -f "$CSV" ]]; then echo "timestamp,phase,arm,suite,task,seed,state_idx,code_sha12,rc,task_completed,sandbox_rc,wall_s,gpu,log,attempt_dir" > "$CSV"; fi
echo "$(date -Is),$PHASE,$ARM,$SUITE,$TASK,$SEED,$(( (SEED-1)%50 )),$SHA,$RC,$TC,$SRC,$((T1-T0)),$GPU,$(basename $LOG),$ID" >> "$CSV"
flock -u 9
echo "RESULT phase=$PHASE arm=$ARM task=$SHORT suite=$SUITE seed=$SEED rc=$RC tc=$TC sandbox_rc=$SRC wall=$((T1-T0))s id=$ID"
exit $RC
