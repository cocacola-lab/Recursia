# Quick Start: LIBERO-Long-Pro Evaluation of the N20 Shared Stack

This reproduces the **small interim LIBERO-Long-Pro evaluation** of the frozen N20 stack with the **closed-loop** method (arm N20C):
- 100 episodes: long0–long9 × {swap, task} × seeds 41–45;
- reference result: **35/100** (POS 12/50, TASK 23/50).

Expect about 2.5 h on 5 GPUs, and about $15–20 in model calls.

## 1. Simulation setup (once)

Follow [`recursia/sim/README.md` § 1 Setup](../../README.md#1-setup):
- `scripts/fetch_third_party.sh` (pinned LIBERO-PRO, robosuite fork, SAM3, Contact-GraspNet + compatibility patch);
- `uv sync --locked` into `.venv-libero`;
- `scripts/write_libero_config.sh`;
- `.venv-vlmproxy` for the model proxy;
- `hf auth login` for the gated SAM3 weights.

## 2. Perception servers (persistent tmux pane)

```bash
tmux new -s recursia-perception
cd "$RECURSIA_SIM_ROOT" && bash scripts/start_perception_servers.sh --gpu-sam3 0 --gpu-graspnet 1   # PERCEPTION_OK
```

## 3. Run (one command)

```bash
cd "$RECURSIA_SIM_ROOT"
export ANTHROPIC_API_KEY=...               # only the proxy process holds it
GPUS="3 4 5 6 7" bash rsi/evaluation/harness/quickstart.sh closed interim
```

The script runs five steps:

| Step | What it does |
|---|---|
| preflight | checks the perception servers and the key |
| `setup` | creates `outputs/rsi_zs_eval_N20C_interim_<date>/`: copies the frozen N20 stack, runner, interface, catalog, prompts and calibration; rebuilds `N20C.py` and verifies both hashes against the 35/100 run; writes the 100-row plan; runs the 70 offline tests |
| `start-proxy` | starts the loopback proxy on `127.0.0.1:8117` with this run's ledger and the dollar cap (default $35), and sends one provider probe |
| `smoke` | runs 1 episode (long2_swap s41), excluded from the count |
| `run` | starts the 100 episodes in a detached process and prints the commands to monitor and stop it |

## 4. Finish

```bash
T=rsi/evaluation/harness/rsi_eval.py; OUT=outputs/rsi_zs_eval_N20C_interim_<date>
.venv-libero/bin/python3 $T rerun-infra --out $OUT   # predeclared: rows with proxy transport errors run once more
.venv-libero/bin/python3 $T summarize   --out $OUT   # results/summary.json, results/REPORT_auto.md
.venv-libero/bin/python3 $T stop-proxy  --out $OUT
```

`REPORT_auto.md` contains:
- the official total;
- POS / TASK with Wilson 95% intervals;
- the per-task table and the total without long4;
- orchestration outcomes, timeouts and cost;
- paired sign tests against the reference rows: N20C 35, N18C 35, N18O 33, N18 rule-based 29, N21C 32 / 30.

For a full write-up, follow [report-template.md](report-template.md).

**Repeatability:**
- The arm program (`N20C.py`, `dbe25f14…`) is byte-identical to the reference run, and the runner is identical except for its import paths; `setup` checks both.
- Simulator states are fixed by the seeds.
- The LLM and Contact-GraspNet are not seed-controlled, so the total varies between runs. The same closed loop scored 35 twice (N18C and N20C), and N21C scored 32 and 30.
- Treat roughly 30–38 as a reproduction, and use the paired comparison to see which rows differ.

## Other methods and scales

```bash
bash rsi/evaluation/harness/quickstart.sh rule interim      # N20 rule-based (the stack's own run()); expected level ~29 (from N18)
bash rsi/evaluation/harness/quickstart.sh oneshot interim   # N20O one-shot orchestration; expected level ~33 (from N18O)
CAP_USD=200 bash rsi/evaluation/harness/quickstart.sh closed full   # 1000 episodes (seeds 1-50), ~21 h on 5 GPUs, ~$150
```

See [methods.md](methods.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `perception server on port … is down` | start step 2, and keep its tmux pane alive |
| `provider probe failed` | wrong or missing key, or no network route to the provider; see `<OUT>/logs/proxy.out` |
| `something already answers on 127.0.0.1:8117` | another proxy is running: stop it (`rsi_eval.py stop-proxy --out <its dir>`) or wait for that run |
| LIBERO asks for paths interactively | run `scripts/write_libero_config.sh` (the harness exports `LIBERO_CONFIG_PATH=<sim>/.libero`) |
| `VLM_WAIT` lines in `logs/plan_*.log` | provider outage; episodes are held back until the probe succeeds again |
| A burst of HTTP 400 on every request | provider-side outage (the proxy does not retry 400); afterwards, run `rerun-infra` |
| A total far below 30 | check `REPORT_auto.md` (paired rows, integrity errors) before trusting the number |

**Never put the key in a file or on a command line, and never use `pkill -f`.** Stop processes with the pid files under `<OUT>/logs/`.
