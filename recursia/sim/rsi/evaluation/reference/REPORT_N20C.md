# N20C Long-Pro evaluation, 100 episodes: closed-loop LLM orchestration over the frozen N20 stack

> **N20C is only an inference method. The learned stack is N20** (`programs/N20.py`, `29b187df…`, a snapshot of C19I3; N18 → C19I5 (N19) → C19I3),
> unchanged and runner-asserted at every episode start. The driver is the closed loop (an Opus plan, a decision after every subgoal,
> visual checks, the step-budget rule), identical to N18C's.

**Run facts**
- **Phase:** `long100_N20C`, run 2026-10-16 17:47 → 20:19 HKT, on 5 GPUs (3–7).
- **Rows:** 100 = 10 tasks × {swap, task} × seeds 41–45, the standard interim rows.
- **Integrity:** **complete, 0 integrity errors.**
- **Infrastructure:** 0 proxy transport errors, 0 probe waits, 0 runner or sandbox errors, so no infrastructure reruns were needed.
- **Smoke:** 1 episode beforehand, excluded.

## 1. Official outcome: **35/100**

| Suite | Successes | Wilson 95% |
|---|---|---|
| **POS (libero_10_swap)** | **12/50 = 24%** | [14–37] |
| **TASK (libero_10_task)** | **23/50 = 46%** | [33–60] |

**Per task**, on the same rows (seeds 41–45), with swap / task successes out of 5:

| Task | N18 rules | N18C | **N20C** |
|---|---|---|---|
| long0 (basket ×2) | 0 / 5 | 0 / 5 | 0 / 5 |
| long1 (basket ×2) | 2 / 2 | 2 / 3 | 2 / 3 |
| long2 (stove on + pot / pan) | 0 / 4 | 0 / 5 | 0 / 5 |
| long3 (drawer + close) | 0 / 0 | 0 / 0 | 0 / 0 |
| long4 (two mugs; Scene5, exposed) | 3* / 5 | 5 / 5 | 5 / 5 |
| long5 (caddy) | 0 / 0 | 0 / 0 | 0 / 0 |
| long6 (mug + pudding right) | 1 / 0 | 1 / 0 | 1 / 0 |
| long7 (basket ×2) | 2 / 5 | 4 / 5 | 4 / 5 |
| long8 (moka pots) | 0 / 0 | 0 / 0 | 0 / 0 |
| long9 (microwave) | 0 / 0 | 0 / 0 | 0 / 0 |
| **Total** | **29** (POS 8, TASK 21) | **35** (12, 23) | **35** (12, 23) |

\* N18's run lost long4_swap s41/s43 to VLM connection errors, by its own evaluator's note.

## 2. Comparisons (same rows, different stack and time)

| Comparison | Gained | Lost | Exact sign p |
|---|---:|---:|---|
| N20C vs N18 rule driver | 6 | 0 | 0.031 |
| **N20C vs N18C** | **0** | **0** | 1.0 |

**N20C reproduces N18C's outcome on every one of the 100 rows.**
- N20's additions (the `lpf_*` placement primitives inside `run_subgoal`) changed no Long-Pro outcome under the closed-loop driver.
- No N20 rule-driver result exists (the N20 interim run was stopped by the user), so the stack and driver effects on N20 cannot be separated. The 6 rows gained over N18's rule driver are the same 6 rows N18C gained: 4 from bounded retry, 2 from N18's infra-lost long4_swap rows.
- N18's `ValueError` on long1_swap s45 no longer occurs (no sandbox error in this run). The row still fails officially.

## 3. What the orchestration did

| Outcome | Episodes |
|---|---:|
| `all_controls_completed` | 40 |
| `binding_failed` | 20 |
| `verification_failed` | 17 |
| `capability_unsupported` (all 10 long9) | 10 |
| `execution_failed` | 8 |
| `episode_timeout_during_orchestration` | 5 |

- **Timeouts:** 5 (N18C: 3 on the same rows).
- **Re-places:** 4 (long0_task s45, long1_swap s43, long1_task s43, long8_task s42). None of them changed an outcome relative to N18C.

## 4. Cost, time, ledger
- **Cost:**

  | Item | Cost |
  |---|---:|
  | Orchestration | $12.81 |
  | N20 VB queries | $2.85 |
  | **Total for the 100 episodes** | **$15.66** |
  | Proxy ledger (also includes the smoke and probes) | $16.04 of the $35 cap |

- **Wall time:** mean 444 s per episode.
- **Start ledger:** 16744 → 16845, i.e. 1 smoke + 100 run starts (cap 20000).

## 5. Conclusion
On these Long-Pro rows the closed-loop inference method gives **35/100 on both N18 and N20** (POS 24%, TASK 46%).
- **The gain over N18's rule driver comes from the driver**, mainly bounded retries, not from N19/N20's stack changes.
- **The rows still at zero** (long3, long5, long8, long9, long0_swap, long6_task) are unchanged from N18 and need skill work.

**Limitations:**
- seeds 41–45 are heavily re-used;
- 5 states per cell;
- no N20 rule-driver baseline;
- Scene5 exposure;
- stochasticity is not seed-controlled.
