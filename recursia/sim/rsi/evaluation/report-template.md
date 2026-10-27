# Report Template: `results/REPORT.md`

Example: [reference/REPORT_N20C.md](reference/REPORT_N20C.md).

```markdown
# N<k>C Long-Pro <interim 100 | full 1000>: closed-loop LLM orchestration over the frozen N<k> stack

> **N<k>C is only an inference method. The learned stack is N<k>** (`programs/N<k>.py`, `<sha12>…`), unchanged and runner-asserted.

**Run facts**
- the phase, start → end time, and GPUs;
- the rows: tasks × suites × seeds;
- integrity: errors, sandbox errors, stack exceptions;
- infrastructure: transport errors, probe waits, infrastructure reruns; the smoke (excluded).

## 1. Official outcome: **<n>/<N>**
- the POS (`libero_10_swap`) and TASK (`libero_10_task`) rates with Wilson 95% intervals; the total;
- a per-task table of swap / task successes out of the number of seeds;
- if there were infrastructure reruns: both the without-rerun and the with-rerun totals.
- Start program and interface of the debugging run, from its `authorization.json`. Only an A0 / `original` lineage is a clean zero-shot result. S1/K2/N-lineage snapshots carry long4 exposure (S1 was developed on long4, seeds 1–50), so also report the total **without long4**.

## 2. Paired comparisons (same rows)
- Gained / lost / exact two-sided sign test p for each reference on the same rows.
  - Interim references: N20C 35, N18C 35, N18 rule driver 29, N21C + GraspNet 32, N21C + AnyGrasp 30.
- Say what differs in each comparison (stack, driver, backend, time).

## 3. What the orchestration did
- the counts per episode outcome: `all_controls_completed`, `binding_failed`, `verification_failed`, `capability_unsupported`, `execution_failed`, `episode_timeout_during_orchestration`, `stack_exception`, `decision_error`;
- retries and replaces (attempted / succeeded); visual-check verdicts vs the official outcome.

## 4. Failure analysis
- per open task: the first failing subgoal and its reason, read from the structured results and stdout.
- keep the stack's capability gaps (refusals, unsupported relations) apart from execution and binding failures.

## 5. Cost and ledger
- orchestration $, stack VB $, total; the proxy ledger vs its cap;
- mean wall time per episode; start-ledger before → after.

**Limitations**
- states per cell; seed reuse; Scene5 exposure;
- stochasticity: GraspNet and the LLM are not seed-controlled; shared GPUs.
```

## Statistics
- **Wilson 95% interval** for each suite rate.
- **Exact two-sided sign test** on discordant pairs: `p = min(1, 2 · Σ_{i ≤ min(b, c)} C(b + c, i) / 2^{b+c})`. This is `interim100_cl.sign_p`.
- Every planned episode stays in the denominator. An infrastructure rerun replaces its original only in `summary_infra_rerun.json`.

## Boundary
- This report lives in the evaluation directory only.
- It is never copied into a debugging round directory, and the debugging agents never read it.
