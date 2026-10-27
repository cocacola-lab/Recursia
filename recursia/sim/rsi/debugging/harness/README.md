# Generic RSI Harness

This is one CLI for any RSI zero-shot transfer debugging run. It is driven by three files:

| File | Written by | When |
|---|---|---|
| `authorization.json` | the human | once |
| `iter_<k>/strategy.json` + `targets.json` | the outer-loop agent | before iteration `k`; frozen by `rsi.py freeze` |

Nothing in the harness names a past round, arm or seed range outside the invariant pools.

## Start a run

```bash
cd recursia/sim && export RECURSIA_SIM_ROOT="$(pwd)"
cp rsi/debugging/harness/authorization.template.json /tmp/auth.json   # edit budgets, GPUs, K
export ANTHROPIC_API_KEY=...        # only for the loopback VLM proxy window; workers and the agent window never get it
CLAUDE_ARGS="<your Claude Code permission flags>" \
  bash rsi/debugging/harness/launch.sh outputs/rsi_zs_debug_<date> /tmp/auth.json
```

`launch.sh` does four things:
1. checks the perception servers;
2. runs `rsi.py init`;
3. starts the VLM proxy, with the round's ledger and the authorized cap;
4. starts the outer-loop agent (Opus) in tmux.

To stop the whole run: `touch <round>/STOP`.

Use a `CLAUDE_ARGS` permission setting that lets the agent run shell commands unattended. The outer loop must not wait for a human.

## Commands (`rsi.py --round R <cmd>`)

| Command | Who | What it enforces |
|---|---|---|
| `init --authorization A` | launch | copies the source-only inputs; patches the runner to accept any `N<k>` / `C<k>I<j>[r<n>]` arm; builds `N0` from the start program (`A0` by default = clean start with `interface: original`; `S1`/`K2`/`N20` need `interface: rsi`); verifies synthetic hashes; refuses Long-Pro tasks |
| `freeze --iter k` | outer loop | schema and invariants (seed pools, roles, `timeout_rate_max` ≤ 2%, outcome mins ≥ 1, group sizes); the previous iteration is DONE; writes `strategy_sha256.txt`. Every later command re-verifies the hash |
| `plan --phase P --kind inventory\|baseline\|refresh\|candidate\|infra [--group G]` | evaluator | rows from `S_k` (data-roles.md); panel tasks in a group are swapped out; `baseline` only with `full_rerun`, `refresh` only with `row_store`; a candidate must be built on the current parent, with a matching `predeclared.json`, and with a passing coverage gate when `S_k` requires one; the row store must be complete |
| `observe --phase P --tasks …` | evaluator | observe-only frames on development seeds, when `S_k` enables them |
| `brief --phase P --group G` | evaluator | `proposals/P/BRIEF/attribution.md`: the parent's proposer-visible rows only |
| `proposer-call --phase P --kind first\|revision\|resume` | evaluator | the launch cap on proposer calls (a resume is free) |
| `build --phase P --group G` | evaluator | slot, revision and retry limits; patch scope; forbidden literals; unique anchors; no identical re-gate; the proposer's tests must pass |
| `coverage --phase P` | evaluator | the offline coverage gate (needs `la_parse` / `la_dispatch` in the program) |
| `run --phase P` | evaluator | prepared and frozen hashes; proxy health and cap; the start budget; then runs the stage (credential-free workers, provider probe), posthoc attribution, analysis and infra flags |
| `gate --phase P` | evaluator | loss caps per role (never netted); outcome OR mechanism; code-path exemption from `RUNTIME_MARKERS`; `gate.json` is write-once |
| `decide --phase P --decision accept\|reject --review FILE` | evaluator | a written review; a numeric failure can never be accepted; updates the current parent |
| `snapshot --iter k` | evaluator | `programs/N<k>.py` (marker only), `lineage.json`, `aggregates.json`, `DONE` |
| `aggregates --iter k` | evaluator | `iter_<k>/aggregates.json`: counts only, which is what the outer loop reads about evaluator-only roles |
| `status` | anyone | |

## Phase names
- `inventory<k>`
- `baseline<k>_<arm>_<G>`
- `refresh<k>_<arm>_<G>`
- `chunk<k>_iter<j>`, with arm `C<k>I<j>`
- `chunk<k>_iter<j>_rev<n>`, with arm `C<k>I<j>r<n>`
- `<phase>_infra1`: an infrastructure rerun of the flagged rows, done once

## Result layout (per phase)
- `results/P/analysis.json` and `attribution.md`: proposer-visible roles only.
- `results/P/evaluator_only/{rows.jsonl, selection.json}`: every row. Never shown to a proposer; the outer loop reads `aggregates.json` instead.
- `results/P/integrity.json`, `vlm_infra_episodes.json` and `gate.json`.

## Proposal contract (`proposals/P/patch.py`)
```python
MODULE = "place_primitive"                  # with one_module scope
TOUCHED_MODULES = ["place_primitive"]       # several allowed with primitive_across_modules (MODULES in code/apply_proposal_90.py)
FAILURE_CLASS = "place_descent:failed/premature_contact"
INSERTS = [(anchor, text), ...]             # text is inserted before anchor; each anchor must occur exactly once
PATCHES = [(old, new), ...]                 # each old string must occur exactly once
RUNTIME_MARKERS = ["PLACE_CLEARANCE_V2"]    # strings printed ONLY on the changed branches (needed for the code-path exemption)
```
Build rules on the new text:
- No object, scene or colour literal (`FORBIDDEN` in `code/apply_proposal_90.py`). Under the original interface the general spatial words `left` and `right` are allowed, because there is no side-word adapter.
- No simulator-truth API (`FORBIDDEN_API` in `rsi_lib.py`: `.sim`, `body_xpos`, `set_joint_qpos`, `handle.env`, `APIS`, `.bddl`, `.xml`, `mujoco`, …).

**Interface** (`authorization.json`):
- `original`: the runner leaves the reduced API untouched. The program sees only `ORIGINAL_API` plus `env.handle.task_language`.
- `rsi`: the N-era harness tools (motion interface, verifiers, binding service, task descriptors, VLM) are injected.

The runner records the interface in every episode manifest, and `analyze_phase` checks it.
The proposal directory must also hold:
- `DIAGNOSIS.md`;
- `offline_replay.py` + `.out`;
- at least one `test_*.py`. The tests run with `RSI_PROGRAM` = the built candidate.

`predeclared.json`, written by the evaluator before `plan --kind candidate`:
```json
{"arm": "C3I1", "parent": "N2", "group": "A", "mode": "outcome", "named_tasks": ["<the group's tasks>"],
 "failure_classes": ["grasp_descent"], "predicted": {"dev": "+3", "selection": "+1", "panels": "0 losses"}}
```

## Cost note
A `full_rerun` candidate stage runs both the parent and the candidate. Under `S_1` with a 5-task group, that is 2 × 209 = 418 episodes, about 8–9 h on 5 GPUs.

This cost is intentional: it is the "cost" trigger that historically moved the procedure to row-store pairing. The outer loop decides when to switch. A warm start (`start_strategy: warm_N17_N20`) begins with row-store pairing and per-group stages.

## Tests
`.venv-libero/bin/python3 -m pytest -q harness/tests` runs 37 offline tests. They cover:
- strategy validation and the invariants;
- row counts under `S_1`;
- loss accounting;
- the gate in both pairing modes and the mechanism gate;
- infra replacement and the strategy freeze;
- slots, the build rules and the runner patch;
- coverage on K2 and N20; the N0 marker on the S1, K2 and N20 start programs;
- `analyze_phase` and the infra flags on fabricated episodes;
- a full `init`.

`init` runs these tests itself.
