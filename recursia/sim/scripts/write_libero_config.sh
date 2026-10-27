#!/usr/bin/env bash
# Write recursia/sim/.libero/config.yaml for the pinned LIBERO-PRO checkout. Upstream LIBERO reads $LIBERO_CONFIG_PATH/config.yaml
# at import time (and prompts interactively without it). The Recursia harnesses export LIBERO_CONFIG_PATH=<sim>/.libero for every
# worker, so ~/.libero is never touched; export it yourself for ad-hoc scripts.
set -euo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="$SIM_ROOT/cap/third_party/LIBERO-PRO/libero"
[[ -d "$L/libero/bddl_files" ]] || { echo "ERROR: LIBERO-PRO missing; run scripts/fetch_third_party.sh"; exit 1; }
mkdir -p "$SIM_ROOT/.libero"
{
  echo "benchmark_root: $L/libero"
  echo "bddl_files: $L/libero/bddl_files"
  echo "init_states: $L/libero/init_files"
  echo "datasets: $L/datasets"
  echo "assets: $L/libero/assets"
} > "$SIM_ROOT/.libero/config.yaml"
echo "wrote $SIM_ROOT/.libero/config.yaml   (export LIBERO_CONFIG_PATH=$SIM_ROOT/.libero)"
