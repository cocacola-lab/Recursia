# CLAUDE.md: agent guide for recursia/sim

You are working in the Recursia simulation workspace. [README.md](README.md) is the human guide; this file tells a coding agent exactly how to carry out the common requests. Work from this directory (`recursia/sim`), and export `RECURSIA_SIM_ROOT="$(pwd)"`.

## Rules (always)
- Never `git push`, and never commit unless the user asks.
- **Credentials:**
  - Never print, log, write or echo `ANTHROPIC_API_KEY` or any token.
  - Only the model proxy process may hold the key; every launcher strips it from simulation workers.
  - If the key is not in your environment, ask the user to exit, `export ANTHROPIC_API_KEY=...` in their shell, and restart Claude Code. Never ask them to paste it into the chat.
- Stop processes only through their pid files (`<run>/logs/*.pid`, `.perception/*.pid`). Never use `pkill -f`.
- Programs may use only the robot API in README § 3. Never read or use simulator ground truth or asset files.
- Never run LIBERO-Long-Pro tasks (`libero_10_*`) or seeds 1–50 inside an RSI debugging round.
- Long runs: start them detached (the scripts do this) and check progress by counting lines in `<run>/attempts_log_*.csv` at long intervals, not by tight polling.

## Request: "set up the simulation and run the N20C interim Long-Pro evaluation" (Quick Start)

1. **Prerequisites.**
   - Check `nvidia-smi` and pick free GPUs: 2 for perception (SAM3, Contact-GraspNet) and 1–5 for simulation workers.
   - Check that `uv`, `git`, `curl` and `tmux` exist.
   - Check that `ANTHROPIC_API_KEY` is set (test with `[ -n "$ANTHROPIC_API_KEY" ]` only).
2. **Third-party sources:** `bash scripts/fetch_third_party.sh`. Expect `THIRD_PARTY_OK`.
3. **Environments** (skip any that already exist):
   ```bash
   uv venv .venv-libero --python 3.12 && source .venv-libero/bin/activate
   uv sync --locked --active --extra libero --extra contactgraspnet --extra dev && deactivate
   uv venv .venv-vlmproxy --python 3.12 && uv pip install --python .venv-vlmproxy/bin/python3 "anthropic>=1.11"
   bash scripts/write_libero_config.sh
   ```
   Verify with `LIBERO_CONFIG_PATH=$PWD/.libero .venv-libero/bin/python3 -c "import libero, robosuite, sam3, contact_graspnet_pytorch, cap"`, then run `.venv-libero/bin/python3 -m pytest -q`, which must pass.
4. **SAM3 weights.** They are gated. If `.weights/sam3.pt` is missing and `SAM3_CHECKPOINT_PATH` is unset, check `.venv-libero/bin/hf auth whoami`. If that fails, ask the user to run `! .venv-libero/bin/hf auth login` (or to provide a checkpoint path), then continue.
5. **Perception servers.**
   - If ports 8114–8116 already answer (`curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8114/health` gives 404, not 000), reuse them.
   - Otherwise run `tmux new -d -s recursia-perception "bash scripts/start_perception_servers.sh --gpu-sam3 <g0> --gpu-graspnet <g1>; sleep infinity"` and wait until all three answer (up to 5 min; logs in `.perception/`).
6. **Evaluation:** `GPUS="<worker gpus>" bash rsi/evaluation/harness/quickstart.sh closed interim`. This sets up `outputs/rsi_zs_eval_N20C_interim_<date>/`, verifies the frozen stack, starts the proxy ($35 cap), runs a 1-episode smoke, and starts the 100 episodes detached. Report the output directory and the expected time: about 2.5 h with 5 workers, longer with fewer.
7. **When `logs/run.out` shows `STAGE_DONE`:**
   ```bash
   T=rsi/evaluation/harness/rsi_eval.py
   .venv-libero/bin/python3 $T rerun-infra --out <OUT>   # only re-runs episodes with proxy transport errors
   .venv-libero/bin/python3 $T summarize   --out <OUT>
   .venv-libero/bin/python3 $T stop-proxy  --out <OUT>
   ```
8. **Report** from `results/REPORT_auto.md`: the official total (reference 35/100; about 30–38 counts as a reproduction), POS/TASK with intervals, the total without long4, the paired comparison with the reference rows, integrity errors, and cost.
   - If the total is far off, inspect the paired rows and integrity errors before drawing conclusions.
   - Reference successful trajectories are in `rsi/evaluation/reference/trajectories/N20C_success34/` for comparison.

Other methods: `quickstart.sh rule interim` or `quickstart.sh oneshot interim`. For the full 1000 episodes, use `closed full` with `CAP_USD=200`.

## Request: "start an RSI debugging run"
Follow README § 6 and [rsi/debugging/INSTRUCTIONS.md](rsi/debugging/INSTRUCTIONS.md):
- the user writes `authorization.json` from `rsi/debugging/harness/authorization.template.json`;
- `rsi/debugging/harness/launch.sh` starts the outer-loop agent;
- offer the 3-episode harness smoke first (README § 6).

## Layout
- `cap/`: the LIBERO runtime.
- `rsi/debugging/` and `rsi/evaluation/`: the runbooks, harnesses and frozen references.
- `scripts/`: setup and perception servers.
- `tests/`: offline checks.
- `outputs/`: runs (git-ignored).
