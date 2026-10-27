# Evaluation Methods: Rule-Based, One-Shot, Closed-Loop

All three methods evaluate the **same frozen stack** (the learned program `N_k`; reference N20, sha256 `29b187df…`) through the same runner, on the same rows. Only the **driver** differs: the part that turns the instruction into calls of the stack's subgoal executors.

Every skill, binding module, gate, verifier and threshold is the stack's own, byte for byte, and the runner asserts this at every episode start. One-shot and closed-loop are **inference methods only**; they never change the learned stack.

| | **Rule-based** | **One-shot** | **Closed-loop** |
|---|---|---|---|
| Arm (N20 stack) | `N20` | `N20O` | `N20C` |
| Driver | the stack's own `run()`: rule-based language parse (`la_parse`), each subgoal executed once, in order | one Opus call per episode writes an orchestration program over the frozen interface catalog; a restricted interpreter (`orch_check`) runs it; no rewrite | an Opus plan (typed JSON), then a decision after every subgoal (execute / retry / replace / stop), plus visual relation checks |
| LLM calls per episode | 0 (the stack's VB binding module may query the VLM) | 1 | ≤ 24 |
| Recovery | none beyond the stack's own retries | what the generated program encodes (bounded re-invocation allowed by the interface) | bounded `retry` (nothing held) and `replace` (the visual check says "no", gated by calibration); step-budget rule |
| Generated code | none | yes: interpreted, never `exec`'d | none |
| Program built by | `programs/N20.py` as is | `build_orch_arm.py`: N20 + `orch_main` entry (one-shot) | `build_orch_arm.py`: N20 + `orch_main` entry (closed loop) |
| Specification | `run()` in the stack | [closed-loop-method.md](closed-loop-method.md) § 1 (interface v2) + `prompts/orch_prompt_v2.md` | [closed-loop-method.md](closed-loop-method.md) |
| `rsi_eval.py setup --method` | `rule` | `oneshot` | `closed` |

## Reference results (interim rows: long0–long9 × swap/task × seeds 41–45)

| Method | Arm | Official | POS (swap) | TASK | Source |
|---|---|---|---|---|---|
| Rule-based | N18 | 29/100 | 8/50 | 21/50 | `rsi_libero90_N18_interim_long100_20261012` |
| One-shot | N18O | 33/100 | 11/50 | 22/50 | seeds 41–45 of `rsi_libero90_N18OC_orchestration_long200_20261012` (66/200 on 41–50) |
| **Closed-loop** | **N20C** | **35/100** | **12/50** | **23/50** | `rsi_libero90_N20C_closedloop_long100_20261016` |
| Closed-loop | N18C | 35/100 | 12/50 | 23/50 | seeds 41–45 of the N18OC run (70/200 on 41–50) |

The per-row outcomes are in [reference/refs/interim_refs.json](reference/refs/interim_refs.json). `rsi_eval.py summarize` uses them for paired sign tests on the shared rows.

**On the N20 stack only closed-loop has a measured result.** The N20 rule-driver interim run was set up but never run, and `N20O` has never run in simulation. Their expected levels come from N18: rule 29, one-shot 33.

**What the differences mean:**
- Over the rule driver, the gains come mostly from **bounded retry** of the same frozen executor after a failed first attempt.
- Closed-loop and one-shot are close: N18C vs N18O was 6 : 2 on 200 shared rows, p = 0.29.
- The visual check that enables `replace` passed its calibration on LIBERO-90 frames. On Long-Pro, however, its "no" verdicts are unreliable ([calibration.md](calibration.md)).

## Choosing
- **Rule-based:** the stack alone. It is the baseline every inference method is compared against, and the cheapest (no orchestration LLM).
- **One-shot:** tests whether an LLM can compose the frozen skills for a new instruction without feedback.
- **Closed-loop:** the best-performing reference method. Use it for headline numbers, and always report the rule-based baseline next to it.
- Run the methods on the **same rows** (same scale and seeds), so the paired comparison is valid.

## Exposure note (all methods on the N-lineage)
S1, the root of the N-lineage, was developed on long4 seeds 1–50. Report every Long-Pro total with and without long4: `summarize` prints both. Only a snapshot from a clean debugging run (start program A0) gives a fully clean zero-shot number.
