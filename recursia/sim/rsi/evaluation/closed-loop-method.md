# The Closed-Loop Method

Reference code:
- `reference/code/orch_stack.py`: interface v2, `RSIStack`;
- `reference/code/orch_check.py`: interpreter and static capability check;
- `reference/code/closed_loop.py`: the driver;
- `reference/prompts/`: the prompts;
- `reference/catalog/`: the interface catalog.

## 1. Interface v2 over the frozen stack (`orch_stack.RSIStack`)
- **Executed unchanged:** the snapshot's definitions run in the episode namespace. The interface never re-implements a skill, verifier, gate or threshold.
- **Clean-start lineage (A0, `interface: original`):** the snapshot has its own executors (A0: `pick_and_place`), so the wrappers and catalog are rebuilt around them. Episodes run with `RSI_INTERFACE=original`, exactly as in debugging.
- **Wrapped executors:** the complete subgoal executors, as present in the snapshot (`run_subgoal`, `run_open_articulated`, `run_close_articulated`, `run_actuate`, `run_rotary`). Each call gets the same pre-call handling as the baseline driver `run()`:
  - VB reference activation;
  - home between subgoals;
  - `others_xy`;
  - assembly prompts in plan order.
- **Capability check:** the snapshot's `la_dispatch`, used statically by `orch_check` and again at runtime before every call.
- **Identity references:** frozen at task start with the snapshot's `vb_resolve_references`, before any object moves.
- **Results:** a fixed schema, refusals included. It keeps "control completed", "enforced gate passed", "shadow verdict recorded" and "unknown" apart. Executor exceptions are recorded and re-raised, as in the baseline.
- **Shadow verifiers stay shadow.** A placement verifier that was not reliable is never used as a decision signal (N20's MF verifier was right in 11 of 23 "failed" verdicts).
- **Catalog:** `build_catalog.py` lists every contract-compatible entry. The LLM sees the catalog, never the program source.

## 2. The loop (typed JSON only)

1. **PLAN** (`cl_plan_v1.md`): an ordered list of subgoals from the frozen vocabulary.
   - One re-plan is allowed when the capability check refuses the plan **before any motion**. This is capability feedback, not outcome feedback.
2. **DECIDE** (`cl_decide_v1.md`): after every executed subgoal, the LLM chooses from the action list the harness computed as allowed.
   - It sees the instruction, steps used and remaining, the step cost estimate, and the structured results.
3. **VISUAL CHECK** (`cl_verify_v1.md`): the initial frame, the current frame and a relation sentence go in; `yes`, `no` or `unsure` comes out.
   - It runs after every object subgoal that reaches `control_completed`, and once before stopping.

| Action | Allowed when |
|---|---|
| `execute(sg)` | the subgoal is pending (first execution) |
| `retry(sg)` | the subgoal failed **and** the evidence shows nothing is held (orch_stack retry rule) **and** the step rule holds |
| `replace(sg)` | `control_completed` **and** the visual check says `no` **and** `REPLACE_ENABLED` (calibration gate passed) **and** the step rule holds |
| `stop` | always |

**Step rule:** `retry` and `replace` need `remaining steps ≥ STEP_COST × (1 + pending subgoals)`.
- `STEP_COST = 1109` is the p75 of `final_sim_step` over 426 successful single-goal LIBERO-90 development episodes. A recovery therefore never starves later subgoals.
- Recompute it from the snapshot's own LIBERO-90 development rows if the stack changes motion speed. Never compute it from Long-Pro.

## 3. Limits (per episode)

| Limit | Value |
|---|---|
| LLM calls | 24 |
| decisions | 10 |
| visual checks | 10 |
| replace per subgoal / total | 1 / 2 |
| invalid decisions tolerated | 1 |
| horizon | 4000 steps (the episode summary is written even when the horizon interrupts orchestration) |

## 4. No silent fallback
- If a call fails, a JSON is invalid twice, or a limit is reached, the episode ends with a typed outcome: `decision_error`, `binding_failed`, `capability_unsupported`, `episode_timeout_during_orchestration`, etc.
- It never switches to `run()` (`fallback_to_baseline: false` in every summary).

## 5. Building the arm
- `build_orch_arm.py` writes `programs/N<k>C.py`: `programs/N<k>.py` with only its final `run()` line replaced by the orchestration entry `orch_main(globals())`. The stack bytes above it are unchanged.
- It writes `code/iso_runner_orch.py` by exact-match patches to the snapshot's runner (each asserted once). The patches cover: the arm list, `VLM_ARMS`, and the asserts on the stack sha and the orchestration file hashes.
- Every episode start asserts the stack sha, the arm program sha, and the orchestration file hashes.
