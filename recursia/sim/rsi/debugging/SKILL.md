---
name: rsi-debugging
description: "Recursive self-improvement of one shared, task-agnostic LIBERO program on the LIBERO-90 debug set. An inner loop (evaluator + proposer) repairs the program under a frozen strategy S_k; an outer-loop agent adapts the strategy itself between iterations (target selection, debug data roles, regression check, gate, pairing, patch scope). All agents run on Opus; no agent sees Long-Pro data; no human intervenes after launch. Produces snapshots N1..NK for the evaluation runbook."
---

# RSI Zero-Shot Transfer: Debugging

## The idea in one paragraph
- **What is improved.** One Python program, the **stack**, executes every task. Tasks are data only. The stack is repaired one candidate patch at a time by an **inner loop**: a proposer writes a patch for a named failure class, and the evaluator measures it on frozen data roles and accepts or rejects it with a frozen gate.
- **What makes it RSI.** The **procedure that improves the stack is itself adapted**. Between iterations, an **outer-loop agent** reads the completed iteration's records and changes the strategy: which targets are repaired, which data the proposer sees and the gate uses, how regressions are checked, how candidates are gated, how they are paired with their parent, and how large a patch may be.
- **How the strategy persists.** Each change is justified by evidence from the previous iteration, written down, and frozen as the hyperparameters `S_{k+1}` of the next iteration. It persists until a later update changes it.

## Roles

| Role | Model | Scope | May see | May not see |
|---|---|---|---|---|
| **Outer-loop agent** | Opus | Between iterations: chooses `S_{k+1}` and the next targets, writes the strategy update; runs autonomously | all LIBERO-90 records of completed iterations (analysis, attribution, gate decisions, proposer reports, budgets) | **any Long-Pro data**: tasks, frames, results, diagnoses; selection/hidden trajectories are seen only as aggregate gate statistics |
| **Inner-loop evaluator** | Opus | Within iteration `k`: plans, freezes, runs stages, attributes, gates, accepts/rejects, snapshots; never edits the program | everything of iteration `k` on LIBERO-90 | Long-Pro data; it may not change `S_k` |
| **Proposer** | Opus, one call per slot | Writes one candidate patch | the program, development attribution and recordings, observe-only frames, API docs | selection and hidden data, synthetic definitions, simulator assets, Long-Pro data, gate thresholds' outcomes |
| **Human** | — | Authorizes the run once (compute, budgets, `K`, directory) | final reports | **may not intervene in the outer loop** or approve or modify `S_k`; may only stop the whole run |

## Setting

| Item | Value |
|---|---|
| Initial program | `N0` = **A0** ([reference/programs/A0.py](reference/programs/A0.py), sha256 `afc59057…`): a **clean start**. It is a minimal, task-agnostic pick-and-place driver written only from the original ASPIRE reduced API, without episodes or Long-Pro data. It parses 62 of the 90 LIBERO-90 instructions and reports the rest as unsupported. Optional, not clean: S1 (hand-built stack, developed on Long-Pro long4 seeds 1–50), K2 (S1 + 2 subset iterations, the N1–N21 start), N20 |
| Interface | `original` with A0: the program namespace holds only the reduced API functions and `env.handle.task_language`, with no harness tools and no task descriptors. RSI builds every helper inside the program. `rsi` (the N-era harness tools) is required for S1/K2/N20 |
| Reference harness | the harness of the round that produced N20 ([reference/code/](reference/code/); N20 = [reference/programs/N20.py](reference/programs/N20.py), `29b187df…`) |
| Debug set | the 90 `libero_90` tasks ([reference/ordering_90.txt](reference/ordering_90.txt)) and 18 source-only synthetic conjunction tasks ([reference/synthetic/](reference/synthetic/)) |
| Adapters | [reference/task_manifest.json](reference/task_manifest.json): data-only, source tasks only (Long-Pro adapters removed from this copy) |
| Source development (proposer-visible) | seeds **86–95** on the target tasks (5 tasks → **50**) + 10 synthetic development identities × 86–88 (**30**) |
| Source selection (evaluator-only) | seeds **76–85** on the target tasks (**50**) + 10 synthetic identities × 76–78 (**30**) |
| In-loop regression check (evaluator-only) | 11 ordinary source tasks × seeds 76/80/85 (**33**) + 8 hidden synthetic identities × seeds 76/80 (**16**) |
| Eval (never in this runbook) | Long-Pro seeds 1–50 |
| Proposer | one Opus subagent call per slot; at most 4 first-pass slots + 2 retries per iteration |
| Budgets (set at launch) | proposer calls, simulation starts, VLM dollars, 4000 steps / 1500 s per episode, GPUs |

## Pipeline

```text
launch (human, once) ─► outer loop: S_1 ─► inner loop N_1 ─► snapshot N1 ─► outer loop: S_1→S_2 ─► inner loop N_2 ─► … ─► NK
inner loop under S_k:  [inventory] → targets → baseline stage → for each slot: proposer → build → paired stage → gate → accept/reject → snapshot N_k
```

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Entry point |
| [outer-loop-agent-prompt.md](outer-loop-agent-prompt.md) | The outer-loop agent: strategy update between iterations, autonomy, information boundary |
| [main-agent-prompt.md](main-agent-prompt.md) | The inner-loop evaluator (coordinator) runbook for one iteration |
| [proposer-prompt.md](proposer-prompt.md) | The proposer subagent template |
| [strategy-hyperparameters.md](strategy-hyperparameters.md) | The `S_k` schema: dimensions, allowed values, update rule, history N1–N20 |
| [gates.md](gates.md) | The acceptance gates and their parameters |
| [data-roles.md](data-roles.md) | Seed partitions, panels, synthetic tasks, observe-only frames, inventory, row store |
| [coordinator-rules.md](coordinator-rules.md) | Rules learned from incidents in N1–N20 |
| [commit-convention.md](commit-convention.md) | Snapshots, hashes, lineage, records |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist |
| [harness/](harness/) | **The generic harness**: `rsi.py` CLI, `launch.sh`, authorization template, offline tests |
| [reference/](reference/) | Inputs the harness copies (A0, S1, K2, N20, manifest, ordering, synthetic tasks, runner and skill modules) |

Run the loop only through [harness/rsi.py](harness/README.md). `rsi.py init` copies the runner and its modules from `reference/code`. It also patches the runner's arm registry, so any `N<k>` / `C<k>I<j>` arm is accepted. Every round-specific value comes from `authorization.json` and the frozen `strategy.json` / `targets.json`.
