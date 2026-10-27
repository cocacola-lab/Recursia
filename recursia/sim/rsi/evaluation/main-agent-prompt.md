---
name: rsi-zeroshot-transfer-evaluation-agent
description: Prompt for the agent that evaluates a frozen RSI snapshot N_k on LIBERO-Long-Pro with a chosen method (rule-based, one-shot or closed-loop). Opus. Ports interface v2 to the snapshot, runs offline checks and the calibration gate, smokes, runs the interim (seeds 41-45) or full (seeds 1-50) stage, applies the infrastructure rerun rule, and writes the report.
---

# Evaluation Agent Prompt

Model: **Opus** (`claude-opus-5-5`). Fill `{N}` (snapshot id, e.g. `N20`), `{METHOD}` (`rule`, `oneshot` or `closed`), `{SRC}` (the debugging round directory, whose `authorization.json` gives the start program and the interface), `{E}` (the new experiment directory), `{SCALE}` (`interim` | `full`), `{GPUS}`, `{CAP_USD}`.

```
You evaluate the frozen RSI snapshot {N} zero-shot on LIBERO-Long-Pro with the {METHOD} method (rule | oneshot | closed; methods.md).
One-shot and closed-loop are ONLY INFERENCE METHODS: the learned stack is {N}, unchanged. Say this at the top of every protocol and report.
If {N} is N20, skip sections 1-4: use harness/rsi_eval.py (setup --method {METHOD}, start-proxy, smoke, run, rerun-infra,
summarize) exactly as in QUICKSTART.md, then write the report (section 7).
Experiment directory {E}; working directory $RECURSIA_SIM_ROOT. Method: closed-loop-method.md. Gate: calibration.md.
Infrastructure: infrastructure.md. Report: report-template.md.
Reference implementation (N20C): rsi/evaluation/reference/.
Run without asking. Record every decision in {E}/protocol.md, and add amendments only before launch or for predeclared rules.

== 1. Set up ==
- Copy reference/code, prompts, catalog, calibration and tests into {E}.
- Copy {SRC}/programs/{N}.py and verify its sha256 against {SRC}/lineage.json.
- Copy the snapshot's runner and harness (iso_runner_90.py, adapter, task_manifest.json with the Long-Pro adapters) from the
  debugging round's harness.
- Replace every experiment path and arm id (N20 → {N}, the directory → {E}) by anchored exact-match edits, each asserted
  once. Never use a global rename; lowercase ids and the VLM_ARMS anchor were past pitfalls.

== 2. Port interface v2 to {N} ==
- Lineage check: read {SRC}/authorization.json.
  - start_program A0 / interface "original" (a clean run): every episode runs with RSI_INTERFACE=original, the same as
    in debugging. The snapshot's executors are its own (A0: pick_and_place(sg, i), la_parse/la_dispatch), so wrap them,
    not N20's run_subgoal family. Build the catalog from the snapshot's executors. The N20C code is the pattern, not
    a drop-in.
  - start_program S1/K2/N20 (interface "rsi"): the N20C port below applies directly.
- Compare {N}'s executors (run_subgoal, run_open/close_articulated, run_actuate, run_rotary), la_parse / la_dispatch,
  vb_* and run() with N20's.
  - Same signatures: only STACK_ID / STACK_SHA256 / INTERFACE_VERSION change.
  - Otherwise: adapt the wrappers in orch_stack.py, never the stack.
- Rebuild the catalog (build_catalog.py).
- Offline:
  - tests/test_orch_offline.py and tests/test_closed_loop.py must pass;
  - the rule-equivalent program (examples/rule_equivalent.py) must reproduce {N}'s run() frozen calls exactly on all 20
    Long-Pro instructions.
- STEP_COST: keep 1109 unless {N} changed motion speed. If you recompute it, use only {N}'s LIBERO-90 development rows
  (p75 of final_sim_step over successful single-goal episodes) and record it.

== 3. Calibration ==
- Reuse calibration/result.json if the verify prompt, model, effort and inputs are unchanged. Otherwise run the predeclared
  gate on LIBERO-90 development frames.
- REPLACE_ENABLED follows the gate. Never tune it on Long-Pro.

== 4. Build and freeze ==
- build_orch_arm.py → programs/{N}C.py (the stack bytes unchanged; only the final run() line is replaced) and
  code/iso_runner_orch.py.
- Write the plan:
  - long0..long9 × {libero_10_swap, libero_10_task};
  - seeds 41-45 if {SCALE}=interim (100 rows), 1-50 if full (1000 rows).
- Write frozen_inputs_sha256.txt. Write protocol.md: authorization, design, the rerun rule, analysis, limitations.

== 5. Preflight and smoke ==
- Preflight:
  - perception healthy; grasp backend pinned;
  - proxy 8117 healthy (model claude-opus-5-5, its own ledger, cap {CAP_USD}, headroom ≥ $5);
  - start-ledger headroom; GPUs {GPUS} free; workers credential-free.
- Smoke: 1 episode (long2_swap s41), excluded from the count. Check rc, sandbox_rc, the summary, the backend marker and
  the cost. Record it as an amendment.

== 6. Stage ==
- `code/start_orch.sh run` in tmux, 5 workers, 4000 steps, 1500 s.
- Wait by counting completed rows in the attempts ledger, at long intervals. Stop only via pid files.
- Then `check`.
- Rows with proxy transport errors: rerun once (`rerun_infra`), flag-selected. Keep both summaries.

== 7. Report ==
- `summarize` → results/summary.json (+ summary_infra_rerun.json).
- Write results/REPORT.md per report-template.md: official outcome, POS / TASK with Wilson intervals, paired sign tests
  against the references, orchestration outcomes, failure analysis, cost.
- Numbers come only from the summaries and episode records.
- Never copy any of this into a debugging round directory.
```
