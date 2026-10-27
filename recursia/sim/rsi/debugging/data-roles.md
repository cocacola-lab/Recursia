# Data Roles

All data in this runbook comes from source tasks: the 90 `libero_90` tasks and 18 synthetic conjunctions built from them.
Every episode is one `(task, seed)` row. The seed selects init state `(seed − 1) mod 50`.

## Seed partitions (fixed for the whole run)

| Role | Tasks | Seeds | Rows | Who sees row details |
|---|---|---|---|---|
| **Source development** | the iteration's target tasks (≤ 5 per group) | 86–95 | 5 × 10 = **50** | proposer, evaluator, outer loop |
| Synthetic development | syn01–syn10 (`synthetic_dev`) | 86, 87, 88 | 10 × 3 = **30** | proposer, evaluator, outer loop |
| **Source selection** | the same target tasks | 76–85 | 5 × 10 = **50** | evaluator only; outer loop sees aggregates |
| Synthetic selection | syn01–syn10 | 76, 77, 78 | 10 × 3 = **30** | evaluator only; outer loop sees aggregates |
| **Ordinary regression** | 11 source tasks outside the targets: s01, s21, s28, s02, s39, s04, s17, s30, s14, s26, s10 | 76, 80, 85 | 11 × 3 = **33** | evaluator only; outer loop sees aggregates |
| **Hidden synthetic regression** | syn11–syn18 (`synthetic_regression`) | 76, 80 | 8 × 2 = **16** | evaluator only; outer loop sees aggregates |
| Guard panel (when `S_k` enables it) | previously accepted tasks outside the current group | 76, 80 | varies | evaluator only; outer loop sees aggregates |
| Inventory (when `S_k` asks for one) | all 90 `libero_90` tasks, parent only | 86, 87 | 180 | proposer, evaluator, outer loop (descriptive, never a gate) |
| **Eval** | Long-Pro (`libero_10_swap`, `libero_10_task`) | 1–50 | — | **nobody in this runbook** |

A target task that is also on a regression panel must be swapped out of the panel before the iteration starts. The evaluator records the swap in `targets.json`.

## Synthetic tasks

[reference/synthetic/](reference/synthetic/): revision `diverse_source_composition_v6_20260928`.
- 18 two-subgoal conjunctions built from source scenes, objects and verbs; 7 scenes; several relation-pair families.
- `registry.json` holds the BDDL and init sha256 per task.
- The role field fixes membership: 10 `synthetic_dev` and 8 `synthetic_regression` (hidden).
- The proposer never reads the synthetic BDDL or registry. Development syn tasks reach it only through attribution, like real tasks.

## Observe-only frames

A target that stops before perception (`relation_unsupported`, `language_unparsed`) gives the proposer nothing to replay.
- When `S_k.debug_data_roles.observe_only_frames` is on, the evaluator records the initial public observation of such targets on development seeds with `rsi.py observe` before dispatching the proposer.
- These records contain no outcome.

## Row store and refresh (when `S_k.pairing.mode = row_store`)

- **Row store:** an arm's row store is the union of its eligible stages. A `(task, seed)` row is never re-run for the same arm.
- **Candidate stages:** a candidate stage runs only the candidate's rows. Each row is paired with the parent's stored row with the same `(task, seed, state_idx)` (`harness/rsi_lib.py: row_store, missing_rows`).
- **Refresh stages:** after an acceptance, the new parent has no rows for the next group. A **refresh stage** (`refresh<k>_<arm>_<group>`) runs exactly the missing rows before the next candidate stage. `rsi.py plan --kind candidate` refuses when the parent's store is incomplete.
- **`full_rerun` mode:** a baseline stage (`baseline<k>_<arm>_<group>`) gives the proposer the parent's rows. Each candidate stage then runs both the parent and the candidate on all rows, and the pairs are within the stage.

## Simulator truth

- **Allowed, for diagnosis and attribution only:** `diag/events.jsonl`, `contact_truth.jsonl`, and the first-failure classifier (`posthoc_first_failure`).
- **Never inside the program, and never as a gate on its own:** the program uses the public API only (see the forbidden list in `recursia/sim/README.md`).
- **Robot contacts:** read `gripper_nonobject_contacts`, never the truncated `pairs` list.
