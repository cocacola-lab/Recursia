---
name: rsi-evaluation
description: "Zero-shot LIBERO-Long-Pro evaluation of a frozen RSI snapshot N_k with one of three inference methods (rule-based, one-shot, closed-loop; tool: harness/rsi_eval.py, quick start: QUICKSTART.md). The reference method is closed-loop: an Opus LLM plans subgoals from the snapshot's frozen interface catalog, decides after every subgoal (execute / retry / replace / stop) and runs calibrated visual checks; it never writes code and never changes a skill, gate or threshold. Interim = seeds 41-45 (100 episodes), full = seeds 1-50 (1000 episodes). Reference: N20C 35/100."
---

# RSI Zero-Shot Transfer: Evaluation

> **Quick start:** [QUICKSTART.md](QUICKSTART.md) reproduces the interim Long-Pro evaluation of the N20 stack with the closed loop (N20C, 35/100) in one command.
> **Methods:** [methods.md](methods.md) covers rule-based (`N20`), one-shot (`N20O`) and closed-loop (`N20C`), all run by [harness/rsi_eval.py](harness/rsi_eval.py).

## The idea
The debugging runbook produces a frozen, task-agnostic program `N_k`. Its own driver `run()` parses the instruction with rules and calls the subgoal executors once, in order.

The **closed-loop method** swaps only the driver:
- An LLM makes the plan and the recovery decisions, through a typed interface over `N_k`'s own executors.
- Every skill, binding module, gate, verifier and threshold stays `N_k`'s, byte for byte.
- The LLM returns JSON only. Any code-like plan goes through an interpreter (`orch_check`) and is never `exec`'d.
- The method never falls back to `run()`.

## Setting

| Item | Value |
|---|---|
| Stack | snapshot `N_k` (reference: N20, sha256 `29b187df…`) |
| Eval set | LIBERO-Long-Pro: long0–long9 × {`libero_10_swap` (POS), `libero_10_task` (TASK)} |
| Full | seeds **1–50** → 10 × 2 × 50 = **1000** episodes |
| Interim | seeds **41–45** → **100** episodes |
| Episode limits | 4000 sim steps, 1500 s wall time |
| Workers | 5 (one per GPU slot), credential-free |
| Model | Opus `claude-opus-5-5`, effort low, through the loopback proxy |
| Outcome | official task completion only; every planned episode stays in the denominator |

## Pipeline

```text
port interface to N_k ─► build catalog ─► offline tests + rule-equivalence ─► calibration gate (LIBERO-90 frames)
  ─► build arm N_kC (program = N_k + orchestration entry) ─► freeze hashes ─► smoke (1 ep, excluded)
  ─► stage (interim 100 | full 1000) ─► infrastructure reruns (flag-selected, once) ─► summarize ─► REPORT.md
```

## Reference results (interim, seeds 41–45, same 100 rows)

| Arm | Total | POS | TASK |
|---|---|---|---|
| **N20C** (closed loop, N20) | **35/100** | 12/50 | 23/50 |
| N18C (closed loop, N18) | 35/100 | 12/50 | 23/50 |
| N18O (one-shot, N18) | 33/100 | 11/50 | 22/50 |
| N18 rule driver | 29/100 | 8/50 | 21/50 |
| N21C + GraspNet | 32/100 | 11/50 | 21/50 |
| N21C + AnyGrasp | 30/100 | 10/50 | 20/50 |

Most of the closed loop's gain over the rule driver comes from **bounded retry**. On Long-Pro the visual check's "no" is weak.

## Files

| File | Purpose |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Simulation setup + one-command interim evaluation of N20C (35/100) |
| [methods.md](methods.md) | The three evaluation methods (rule-based, one-shot, closed-loop) and their reference results |
| [harness/rsi_eval.py](harness/rsi_eval.py), [harness/quickstart.sh](harness/quickstart.sh) | The evaluation tool: setup, proxy, check, smoke, run, rerun-infra, summarize |
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Entry point |
| [main-agent-prompt.md](main-agent-prompt.md) | The evaluation agent: port, check, smoke, stage, report |
| [closed-loop-method.md](closed-loop-method.md) | Interface v2, plan / decide / check, actions, limits, step rule |
| [calibration.md](calibration.md) | The predeclared visual-check gate that enables `replace` |
| [infrastructure.md](infrastructure.md) | Proxy, probe, ledger, GPUs, the infrastructure rerun rule, integrity checks |
| [report-template.md](report-template.md) | Required report sections and statistics |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist |
| [reference/](reference/) | The N20 stack, N20C code, prompts, catalog, calibration, tests, reference results and trajectories, report |

The reference code hard-codes the N20C experiment path, the N20 sha values and the arm id. [main-agent-prompt.md](main-agent-prompt.md) § 1 says how to re-point them.
