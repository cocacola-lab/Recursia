# Calibration Gate for the Visual Check

`replace` re-picks and re-places an object whose placement looks wrong. Acting on a wrong "no" disturbs a correct placement, so `replace` is enabled only if a gate passes. The gate is written before any calibration call.

Reference: `reference/calibration/GATE.md` and `result.json`. The calibration script and its 172 raw calls are not included, because they read LIBERO-90 development episodes that are not part of this repository.

## Data: LIBERO-90 development episodes only
- **Episodes:** `libero_90` single-predicate episodes from a debugging round's stages, with no horizon exhaustion and `sandbox_rc = 0`. Never Long-Pro.
- **Sampling:** stratified by task and outcome, up to 6 successes and 6 failures per task, random seed 20261008, capped at 200.
- **Inputs:**
  - the first and last recorded agentview frames;
  - the public instruction;
  - a relation sentence built by `closed_loop.relation_clause` from the stack's `la_parse` subgoal.
- **Truth:** `official_task_completed`. The model never sees predicate names.
- **Model and prompt:** the same model, effort, proxy and prompt (`cl_verify_v1`) as the arm. One call per episode.

## Gate
Replace is enabled only if both hold:
- **precision("no") ≥ 0.90**, where precision("no") = P(task not completed | verdict "no");
- **at least 20 "no" verdicts.**

Also report: precision("yes"), recall("no") over the failures, the "unsure" rate, and the confusion matrix.

Rules:
- Nothing (prompt, threshold, inputs) is changed after seeing results.
- A failed gate is reported as failed, and `REPLACE_ENABLED = False`. The decisions and step-aware retry do not depend on this gate.

## Reference result (2026-10-08): passed
- 172 calls, $1.33, 0 errors.
- precision("no") 46/49 = **0.939**; recall("no") 0.72; precision("yes") 0.857; "unsure" 2.3%.

## When to re-run
- **Reuse the result** when the verify prompt, model, effort and frame inputs are unchanged. The check judges frames and does not depend on the stack.
- **Re-run on a new snapshot's LIBERO-90 development rows** when any of those change.
- **Known limitation:** the gate passed on LIBERO-90, but on Long-Pro the check's "no" is unreliable. Report replace actions and their outcomes separately. Do not retune the gate on Long-Pro data.
