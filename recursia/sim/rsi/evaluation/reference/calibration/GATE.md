# Predeclared gate: LLM visual relation check (written before any calibration call, 2026-10-08)

**Why this gate exists.** The closed-loop arm (N13C) may re-pick and re-place an object whose placement looks wrong. Before that, N13's own placement verifier (MF, shadow mode) was checked against official outcomes in the 100 N13O episodes. When it said "failed", the object really was misplaced only 11 of 23 times. Acting on it would disturb correct placements, so that verifier stays in shadow mode. The candidate replacement is the LLM's own judgment from camera frames. This gate decides whether that judgment may trigger a re-place.

**Calibration data.** LIBERO-90 development episodes from the N13 round (`rsi_libero90_N13_open_failures_20261007`, all stages). These are not the Long-Pro evaluation rows.
- Only `libero_90`, single-predicate episodes are used, with no horizon exhaustion and `sandbox_rc` 0.
- The sample is stratified by task and outcome: up to 3 successes and 3 failures per task, random seed 20261008, at most 160 episodes.
- **Inputs:**
  - the first and last recorded agentview frames (`diag/obs/obs_000`, last `obs_*`);
  - the public instruction;
  - a relation sentence built by `closed_loop.relation_clause` from the frozen N13 `la_parse` subgoal.
- Truth is `official_task_completed` (single predicate). The model never sees predicate names.

**Model and prompt.** claude-opus-5-5, effort low, through the loopback proxy (the same configuration the arm would use), prompt `closed_loop.VERIFY_PROMPT_VERSION`, one call per episode.

**Gate.** Re-place is enabled in N13C only if both of these hold:
- precision of "no" ≥ 0.90, i.e. P(task not completed | verdict "no");
- at least 20 "no" verdicts.

Also reported: precision of "yes", recall of "no" over the failures, the "unsure" rate, and the confusion matrix. If the gate fails, re-place stays disabled. The remaining closed-loop features (decisions after each subgoal, step-aware retry) do not depend on this gate.

No prompt or threshold is changed after seeing calibration results. A failed gate is reported as failed.

**Amendment (before any calibration call).** A preview of the sample, with no model calls, showed that 3 per task/outcome gives only 103 usable episodes, 37 of them failures. That is too few failures to reach 20 "no" verdicts. The sample is therefore enlarged to up to 6 per task/outcome, capped at 200, with the same seed. The threshold, prompt and inputs are unchanged.

## Result (2026-10-08, `calibration/result.json`; 172 calls, $1.33, 0 errors)
**The gate passed.**

| Measure | Value |
|---|---|
| precision("no") | **46/49 = 0.939** |
| "no" verdicts | 49 |
| recall("no") over the 64 failures | 0.72 |
| precision("yes") | 102/119 = 0.857 |
| "unsure" rate | 2.3% |

`closed_loop.REPLACE_ENABLED = True`. For comparison, N13's shadow MF verifier was right in 11 of 23 "failed" verdicts on the N13O Long-Pro episodes.
