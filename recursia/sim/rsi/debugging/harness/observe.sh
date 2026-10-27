#!/bin/bash
# Observe-only frames (from the reference run_observe_90.sh). Usage: observe.sh <phase> "<seeds>" <task_short...>
# No program runs and no motion; output stage_observe_<phase>/<task>/seed_<s>/. Env from rsi.py: RSI_ROUND, RSI_GPUS, RECURSIA_SIM_ROOT.
set -u
PHASE=$1; SEEDS=$2; shift 2
ROOT=${RECURSIA_SIM_ROOT:?}; EXP=${RSI_ROUND:?}
read -r -a GPUS <<< "${RSI_GPUS:?}"; NG=${#GPUS[@]}
export RECURSIA_SIM_ROOT=$ROOT MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$ROOT" JAX_PLATFORMS=cpu EXPERIMENT_DIR=$EXP
OUT=$EXP/stage_observe_$PHASE; mkdir -p "$OUT" "$EXP/logs"
i=0
for T in "$@"; do for SEED in $SEEDS; do
  G=${GPUS[$((i % NG))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$G timeout -k 10 600 env -u ANTHROPIC_API_KEY "$ROOT/.venv-libero/bin/python3" "$EXP/code/observe_only_90.py" "$T" "$SEED" "$OUT" > "$EXP/logs/observe_${PHASE}__${T}__s${SEED}.log" 2>&1 ) &
  if (( i % NG == 0 )); then wait; fi
done; done; wait
N=$(grep -l "OBSERVE_DONE" "$EXP"/logs/observe_"${PHASE}"__*.log | wc -l)
echo "OBSERVE_STAGE_DONE phase=$PHASE done=$N planned=$i"
[[ "$N" == "$i" ]]
