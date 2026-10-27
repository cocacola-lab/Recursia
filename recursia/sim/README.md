# Recursia Simulation: RSI Debugging on LIBERO-90, Zero-Shot Evaluation on LIBERO-Long-Pro

Recursia improves **one shared, task-agnostic Python stack** by recursive self-improvement (RSI) on the LIBERO-90 source tasks. It then evaluates the frozen stack, zero-shot, on the LIBERO-Long-Pro target tasks (`libero_10_swap` = POS, `libero_10_task` = TASK).

This directory is self-contained. It holds:
- the simulator runtime (`cap/`);
- the perception servers;
- both experiment runbooks with their harnesses (`rsi/`);
- the frozen reference stacks and results needed to reproduce the simulation numbers.

Third-party simulators and models are fetched at pinned commits.

| I want to… | Go to |
|---|---|
| Reproduce **N20C = 35/100** on the interim Long-Pro evaluation (100 episodes) | [§ 1 Setup](#1-setup) → [§ 2 Perception servers](#2-perception-servers) → [§ 4 Evaluation quick start](#4-evaluation-quick-start-n20c-35100) |
| Evaluate with **rule-based, one-shot or closed-loop** inference | [§ 5 Evaluation methods](#5-evaluation-methods-rule-based-one-shot-closed-loop) |
| Run a **new RSI debugging run** (inner loop + outer-loop strategy adaptation) | [§ 6 RSI debugging](#6-rsi-debugging) |

```text
recursia/sim/
├── cap/                      # LIBERO code-as-policy runtime: envs, robot API (integrations), utils, perception servers (serving)
│   └── third_party/          # fetched by scripts/fetch_third_party.sh (LIBERO-PRO, robosuite fork, SAM3, Contact-GraspNet)
├── env_configs/libero/       # LIBERO / Long-Pro environment config
├── scripts/                  # fetch_third_party.sh, write_libero_config.sh, start/stop_perception_servers.sh, libero/replay_trial.py
├── rsi/
│   ├── debugging/            # RSI runbook: outer-loop agent, inner-loop evaluator, proposer; harness/rsi.py; reference/ (A0, S1, K2, N20, ...)
│   ├── evaluation/           # evaluation runbook: three methods; harness/rsi_eval.py + quickstart.sh; reference/ (N20 stack, N20C code)
│   └── api-reference.md      # the robot API available to programs
├── tests/                    # workspace and LIBERO smoke tests
├── CLAUDE.md                 # agent guide: lets Claude Code run setup + evaluation from a one-sentence request
├── pyproject.toml            # uv project (extras: libero, contactgraspnet, dev)
└── third_party.lock          # pinned upstream repositories and commits
```

## 1. Setup

**Requirements:**
- Linux x86-64 with NVIDIA GPUs. Defaults: SAM3 on GPU 0, Contact-GraspNet on GPU 1, simulation workers on GPUs 3–7. Any free GPUs work.
- `git`, `curl`, `tmux`, and [uv](https://docs.astral.sh/uv/).
- About 15 GB of disk.
- A Hugging Face account with access to the gated [SAM3](https://huggingface.co/facebook/sam3) weights.
- For one-shot, closed-loop and RSI debugging: an `ANTHROPIC_API_KEY` with access to `claude-opus-5-5`.

```bash
git clone https://github.com/cocacola-lab/Recursia.git
cd Recursia/recursia/sim
export RECURSIA_SIM_ROOT="$(pwd)"

# 1. Pinned third-party sources (about 1.6 GB) + the Contact-GraspNet compatibility patch
bash scripts/fetch_third_party.sh                 # ends with THIRD_PARTY_OK

# 2. Simulation / perception environment (Python 3.12)
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --locked --active --extra libero --extra contactgraspnet --extra dev
deactivate

# 3. LIBERO path config, kept inside this workspace (the harnesses export LIBERO_CONFIG_PATH for every worker)
bash scripts/write_libero_config.sh
export LIBERO_CONFIG_PATH="$RECURSIA_SIM_ROOT/.libero"

# 4. Loopback model proxy environment (it alone holds the API key; rollout workers never see it)
uv venv .venv-vlmproxy --python 3.12
uv pip install --python .venv-vlmproxy/bin/python3 "anthropic>=1.11"

# 5. SAM3 weights are gated: either log in (weights download on first start) ...
.venv-libero/bin/hf auth login
#    ... or place an already downloaded checkpoint at .weights/sam3.pt (or set SAM3_CHECKPOINT_PATH)
```

**Verify:**

```bash
.venv-libero/bin/python3 -c "import libero, robosuite, sam3, contact_graspnet_pytorch, cap; print('env ok')"
.venv-libero/bin/python3 -m pytest -q          # offline: workspace + RSI harness tests
RECURSIA_INTEGRATION_REAL=1 MUJOCO_GL=egl .venv-libero/bin/python3 -m pytest -q tests/test_libero.py   # loads a LIBERO-10 task, 10 steps
```

**Credentials:** keep them in environment variables. Never put them in files, YAML, Markdown or command lines. Every launcher removes `ANTHROPIC_API_KEY` from the simulation workers; model calls go through the loopback proxy on `127.0.0.1:8117`, which enforces a dollar cap from its own ledger.

## 2. Perception servers

Every episode uses three local servers: SAM3 segmentation (8114), Contact-GraspNet (8115) and PyRoki IK (8116). Start them in a persistent terminal:

```bash
tmux new -s recursia-perception
cd "$RECURSIA_SIM_ROOT" && bash scripts/start_perception_servers.sh --gpu-sam3 0 --gpu-graspnet 1   # ends with PERCEPTION_OK
# detach: Ctrl-b d.  Logs: .perception/*.log.  Stop: bash scripts/stop_perception_servers.sh
```

On these servers, a `404` on `/health` means "up", and `000` means down. The ports are fixed. If another user's servers already hold them on a shared machine, reuse those servers instead of starting new ones.

## 3. The robot API and its boundary

Programs control the robot only through the reduced API: [rsi/api-reference.md](rsi/api-reference.md), source `cap/integrations/franka/libero_reduced*.py`.

**Allowed:**
- `get_observation()`, `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, `mask_to_world_points`, `get_oriented_bounding_box_from_3d_points`;
- `plan_grasp`, `select_top_down_grasp`, `solve_ik`, `move_to_joints`, `goto_pose`, `open_gripper` / `close_gripper`, `goto_home_joint_position`;
- the geometry utilities;
- `env.handle.task_language`;
- numpy and scipy.

**Forbidden (simulator ground truth; any use invalidates a result):**
- `env.handle.env` and anything reached through it (`.sim`, `sim.data.body_xpos`, `get_site_xpos`, `set_joint_qpos`, `qpos`, `body_name2id`, `joint_id2name`, `sim.forward()`);
- `parsed_problem`, `_eval_predicate`, `obj_body_id`, `_step_once`;
- reading `.bddl`, `.xml`, `.urdf` or MuJoCo model files.

Rule of thumb: if a real robot with a camera could do it, it is allowed. The harness records simulator truth **for diagnosis only**; programs never use it. The RSI build step rejects any patch containing these names.

**Security boundary:** language-model-generated Python runs with full import access. Simulator processes are not a security sandbox. Use an isolated host or container without credentials or sensitive mounts, restrict network access, and never connect generated code to physical hardware.

## 4. Evaluation quick start (N20C, 35/100)

This reproduces the **small interim LIBERO-Long-Pro evaluation**:
- 100 episodes: long0–long9 × {swap, task} × seeds 41–45;
- the frozen **N20** shared stack, evaluated with the **closed-loop** inference method (arm N20C);
- reference result: **35/100** (POS 12/50, TASK 23/50).

**With Claude Code (one sentence):** from this directory, with `ANTHROPIC_API_KEY` exported, run

```bash
claude "Set up the simulation and run the N20C interim Long-Pro evaluation, then report the success rate."
```

Claude Code then follows [CLAUDE.md](CLAUDE.md) through § 1, § 2 and the steps below.

**By hand,** after [§ 1](#1-setup) and [§ 2](#2-perception-servers):

```bash
cd "$RECURSIA_SIM_ROOT"
export ANTHROPIC_API_KEY=...        # used only by the proxy process
GPUS="3 4 5 6 7" bash rsi/evaluation/harness/quickstart.sh closed interim
```

The script runs these steps:

| Step | What it does |
|---|---|
| `setup` | creates `outputs/rsi_zs_eval_N20C_interim_<date>/` and copies the frozen stack and inference code; rebuilds the arm and verifies its program (`N20C.py`, `dbe25f14…`) and runner against the 35/100 run; writes the 100-row plan; runs 70 offline tests |
| `start-proxy` | starts the proxy, with a $35 cap by default, and sends one probe request |
| `smoke` | runs 1 episode, excluded from the count |
| `run` | starts the 100 episodes as a detached process and prints how to monitor and stop it |

It takes about 2.5 h on 5 GPUs and about $15–20.

When `logs/run.out` shows `STAGE_DONE`:

```bash
T=rsi/evaluation/harness/rsi_eval.py; OUT=outputs/rsi_zs_eval_N20C_interim_<date>
.venv-libero/bin/python3 $T rerun-infra --out $OUT   # predeclared: episodes hit by proxy transport errors run once more
.venv-libero/bin/python3 $T summarize   --out $OUT   # results/summary.json + results/REPORT_auto.md
.venv-libero/bin/python3 $T stop-proxy  --out $OUT
```

**What `REPORT_auto.md` contains:**
- the official total and POS / TASK with Wilson 95% intervals;
- per-task counts and the total without long4;
- orchestration outcomes, timeouts and cost;
- paired sign tests against the reference rows (N20C 35, N18C 35, N18O 33, N18 rule-based 29, N21C 32 / 30).

**Expected variation.** The program, runner and simulator states are fixed. The LLM and Contact-GraspNet are not seed-controlled, so treat about 30–38 as a reproduction (the same closed loop scored 35 twice; N21C scored 32 and 30), and use the paired comparison to see which rows differ.

More detail: [rsi/evaluation/QUICKSTART.md](rsi/evaluation/QUICKSTART.md). Reference successful trajectories (34 episodes, text records) are in [rsi/evaluation/reference/trajectories/](rsi/evaluation/reference/trajectories/README.md).

## 5. Evaluation methods: rule-based, one-shot, closed-loop

All three methods run the same frozen stack through the same runner. Only the **driver** changes. One-shot and closed-loop are inference methods; they never modify the learned stack.

| Method | Command | Arm | Driver | Reference (seeds 41–45) |
|---|---|---|---|---|
| Rule-based | `quickstart.sh rule` | N20 | the stack's own `run()`: rule-based language parse, each subgoal once | N18: 29/100 (POS 8, TASK 21) |
| One-shot | `quickstart.sh oneshot` | N20O | one Opus-written orchestration program per episode, interpreted over the frozen interface (never `exec`'d) | N18O: 33/100 (11 / 22) |
| Closed-loop | `quickstart.sh closed` | N20C | an Opus plan, then execute / retry / replace / stop after each subgoal, plus calibrated visual checks | **N20C: 35/100 (12 / 23)** |

```bash
GPUS="3 4 5 6 7" bash rsi/evaluation/harness/quickstart.sh rule    interim   # expected level ~29 (from N18)
GPUS="3 4 5 6 7" bash rsi/evaluation/harness/quickstart.sh oneshot interim   # expected level ~33 (from N18O)
CAP_USD=200 bash rsi/evaluation/harness/quickstart.sh closed full            # 1000 episodes (seeds 1-50), ~21 h, ~$150
```

On the N20 stack only closed-loop has a measured result. The N20 rule-based and one-shot levels above are taken from the N18 stack.

**Further reading:**
- the method details are in [rsi/evaluation/methods.md](rsi/evaluation/methods.md);
- the evaluation runbook for any other snapshot (including ones produced by § 6) is [rsi/evaluation/INSTRUCTIONS.md](rsi/evaluation/INSTRUCTIONS.md).

**Exposure note:** S1, the root of the N1–N21 lineage, was hand-developed on long4 seeds 1–50. Every report therefore also gives the total without long4. Only a stack produced by a clean RSI run (start program A0, § 6) gives a fully clean zero-shot number.

## 6. RSI debugging

An RSI run improves one shared program on the LIBERO-90 debug set. Every agent runs on Opus (`claude-opus-5-5`), and LIBERO-Long-Pro data is never visible.

**Roles:**
- **Inner loop:** an **evaluator** measures and gates candidates; one **proposer** per slot writes one patch, seeing development data only.
- **Outer-loop agent:** adapts the *improvement procedure itself* between iterations. It chooses the strategy `S_k`: target selection, the debug data roles, the regression check, the acceptance gate, pairing with the parent, and patch scope. `S_k` is frozen as hyperparameters for iteration `k` and persists until a later update.

**Start program:** by default, the clean **A0**, written from the original API only, with `interface: original`. The N1–N20 history is a reference, not a script: a new run takes its own path.

**Data roles:**

| Role | Seeds | Rows |
|---|---|---|
| Source development | 86–95 | 50 real + 30 synthetic |
| Source selection | 76–85 | 50 real + 30 synthetic |
| In-loop regression check | 11 tasks × 3 + 8 hidden synthetic × 2 | 49 |
| Long-Pro | 1–50 | never used in debugging |

**Start a run** (the only human action):

```bash
cd "$RECURSIA_SIM_ROOT"
cp rsi/debugging/harness/authorization.template.json ~/auth.json      # set K, GPUs, start/proposer/VLM budgets
export ANTHROPIC_API_KEY=...
CLAUDE_ARGS="<Claude Code permission flags for unattended shell use>" \
  bash rsi/debugging/harness/launch.sh outputs/rsi_zs_debug_<date> ~/auth.json
```

**What `launch.sh` does:**
- checks the perception servers;
- creates the round with `rsi.py init` (N0 = A0, offline tests);
- starts the proxy with the round's cap;
- starts the outer-loop agent in tmux (`tmux attach -t rsi-debug`).

From then on it is hands-off. Each iteration freezes `S_k` and runs the inner loop (`rsi.py plan / run / brief / build / gate / decide`); the evaluator then snapshots `programs/N<k>.py`. To stop the whole run: `touch outputs/rsi_zs_debug_<date>/STOP`.

**Smoke-test the harness first.** This runs 3 episodes on N0 and costs less than $1. Start the proxy as in `launch.sh` step 3 (the evaluation tool's `start-proxy` does the same), then:

```bash
R=outputs/rsi_zs_smoke; H=".venv-libero/bin/python3 rsi/debugging/harness/rsi.py --round $R"
$H init --authorization ~/auth.json
cp $R/iter_1/strategy_template.json $R/iter_1/strategy.json
# write $R/iter_1/targets.json (one group "A" of up to 5 libero_90 tasks; see rsi/debugging/strategy-hyperparameters.md)
# and $R/iter_1/STRATEGY_UPDATE.md (one line), then:
$H freeze --iter 1
$H plan --phase smoke1_a0 --kind smoke --rows s07_pick_up_the_alphabet_soup_an:86 s28_put_the_black_bowl_on_the_pl:86 s10_turn_on_the_stove:86
$H run --phase smoke1_a0
```

**Runbook files** (in `rsi/debugging/`):
- [INSTRUCTIONS.md](rsi/debugging/INSTRUCTIONS.md) and [SKILL.md](rsi/debugging/SKILL.md);
- the agent prompts: [outer loop](rsi/debugging/outer-loop-agent-prompt.md), [evaluator](rsi/debugging/main-agent-prompt.md), [proposer](rsi/debugging/proposer-prompt.md);
- [strategy-hyperparameters.md](rsi/debugging/strategy-hyperparameters.md) (`S_k` schema, triggers, invariants), [gates.md](rsi/debugging/gates.md), [data-roles.md](rsi/debugging/data-roles.md), [coordinator-rules.md](rsi/debugging/coordinator-rules.md);
- the harness reference: [harness/README.md](rsi/debugging/harness/README.md).

Evaluate the resulting snapshots with [§ 5](#5-evaluation-methods-rule-based-one-shot-closed-loop) and [rsi/evaluation/INSTRUCTIONS.md](rsi/evaluation/INSTRUCTIONS.md).

## 7. Notes

- **Provenance.** `cap/` and the reference code derive from ASPIRE (retained SPDX headers; see the repository `NOTICE`). The import path is `cap` with `recursia/sim` as the package root.
  - The evaluation runner is the 35/100 run's runner with only its import paths renamed; `rsi_eval.py setup` verifies this byte for byte.
  - Historical run directories named in the runbooks (`outputs/rsi_libero90_…`) are provenance labels. Their essential files are in `rsi/*/reference/`.
- **Seeds.**
  - Debugging uses development seeds 86–95 and selection seeds 76–85 on LIBERO-90.
  - Evaluation uses Long-Pro seeds 41–45 (interim) or 1–50 (full).
  - Never run Long-Pro inside a debugging round.
- **Stopping processes.** Every long-running process writes a pid file under its run's `logs/` (perception servers: `.perception/`). Stop it with `kill $(cat …pid)`, never `pkill -f`.
