# Strategy `S_k`: the hyperparameters the outer loop adapts

The RSI strategy may change from one iteration to the next. Two changes matter most:
- **Verification gate:** changing it alters how candidate modifications are evaluated.
- **Debug data:** changing which data is used for debugging alters which failures guide the next repair.

These changes persist into later iterations, so the improvement procedure itself is adapted. Within one iteration, `S_k` is fixed:
- the outer-loop agent writes it before the iteration starts;
- it is hashed;
- the inner loop treats it as hyperparameters it cannot change.

## Dimensions, values seen in N1–N20, and their triggers

| Dimension | Values seen | Trigger in the history |
|---|---|---|
| Target selection | new 5-task chunks (N1–N9) > capability groups (N10–N14) > open-failure groups (N16) > 4-group rounds (N17–N20) | blocked tasks piling up; stale measurements |
| Debug data roles | dev seeds + independent selection seeds (N1–N10); + guard panel | acceptance overfit to dev; proposer had no data; cost |
| Regression check | LIBERO-90 panel (N1–N7) > ordinary + hidden synthetic + guard panels | acceptance overfit to dev; proposer had no data; cost |
| Gate | outcome-or-mechanism (N1–N16) > coverage_gate.json before staging, no identical re-gates, same-slot revisions (N17–N21) | gains cancelling losses; losses on unchanged code paths |
| Pairing | full re-run per candidate > paired against the parent's row store, with refresh stages after acceptance (N13+) | cost |
| Patch scope | one module (N1–N14) > primitives across modules (N15+) | measured reach limits |

## Schema: `iter_<k>/strategy.json`

This is `S_1` exactly as the harness writes it (`harness/rsi_lib.py: default_strategy_s1`). `rsi.py freeze` checks every field and the invariants below.

```json
{
 "iteration": 1,
 "parent": "N0",
 "target_selection": {
  "mode": "chunk",
  "n_groups": 1,
  "max_tasks_per_group": 5,
  "inventory": {
   "run": false,
   "seeds": [
    86,
    87
   ]
  },
  "carry_blocked": true
 },
 "debug_data_roles": {
  "proposer_visible": [
   "dev",
   "synthetic_dev",
   "observe_only_frames"
  ],
  "evaluator_only": [
   "selection",
   "synthetic_selection",
   "ordinary_regression",
   "synthetic_regression"
  ],
  "guard_panel": {
   "enabled": false,
   "tasks": "accepted_tasks_not_in_group"
  },
  "observe_only_frames": true,
  "proposer_vlm_usd": 0
 },
 "seeds": {
  "dev": [
   86,
   87,
   88,
   89,
   90,
   91,
   92,
   93,
   94,
   95
  ],
  "synthetic_dev": [
   86,
   87,
   88
  ],
  "selection": [
   76,
   77,
   78,
   79,
   80,
   81,
   82,
   83,
   84,
   85
  ],
  "synthetic_selection": [
   76,
   77,
   78
  ],
  "guard": [
   76,
   80
  ],
  "ordinary_regression": [
   76,
   80,
   85
  ],
  "synthetic_regression": [
   76,
   80
  ],
  "inventory": [
   86,
   87
  ]
 },
 "regression_check": {
  "ordinary_tasks": "panel_11",
  "hidden_synthetic": true
 },
 "gate": {
  "modes": [
   "outcome",
   "mechanism"
  ],
  "outcome": {
   "dev_net_min": 2,
   "selection_net_min": 1
  },
  "loss_caps": {
   "development": 5,
   "selection": 5,
   "synthetic_dev": 3,
   "synthetic_selection": 3,
   "guard": 1,
   "ordinary_regression": 1,
   "synthetic_regression": 1,
   "per_task": 1
  },
  "coverage_gate_before_staging": false,
  "runtime_code_path_exemption": false,
  "no_identical_regate": true,
  "same_slot_revision": {
   "enabled": false,
   "max": 1
  },
  "timeout_rate_max": 0.02
 },
 "pairing": {
  "mode": "full_rerun",
  "refresh_after_accept": false,
  "sequential_rebuild_on_current_parent": true
 },
 "patch_scope": {
  "mode": "one_module"
 },
 "slots": {
  "max_slots": 4,
  "max_retries_total": 2
 },
 "justification": "STRATEGY_UPDATE.md"
}
```

Allowed values:
- `target_selection.mode`: `chunk` | `capability_groups` | `open_failure_groups` | `k_group_round`.
- `seeds.<role>`: a subset of the role's pool (development 86–95, held-out 76–85).
- `regression_check.ordinary_tasks`: `"panel_11"` or an explicit list of `libero_90` tasks.
- `debug_data_roles.guard_panel.tasks`: `"accepted_tasks_not_in_group"` or an explicit list.
- `pairing.mode`: `full_rerun` | `row_store`.
- `patch_scope.mode`: `one_module` | `primitive_across_modules`.

The warm start is `warm_strategy_n17_n20`. Targets (`targets.json`) look like this:

```json
{"iteration": 2, "first_pass_order": ["A", "B"],
 "groups": {"A": {"name": "...", "tasks": ["s07_...", "s16_..."], "failure_classes": ["grasp_descent"],
                   "mechanism_allowed": true, "vb_changes_allowed": false, "pairs": []}, "B": {"...": "..."}}}
```

## Initial strategy `S_1`

These are the N1-style values, with the regression check that N7 introduced already in place. Running Long-Pro inside the loop (as N1–N6 did) is not allowed in any strategy.

| Dimension | `S_1` |
|---|---|
| Target selection | `chunk`: the next 5 tasks of `ordering_90.txt`, blocked tasks carried |
| Debug data roles | dev 86–95 + synthetic dev (proposer); selection 76–85 + synthetic selection (evaluator); no guard panel |
| Regression check | ordinary panel (11 tasks × 76/80/85) + hidden synthetic (syn11–18 × 76/80) |
| Gate | outcome (dev net ≥ 2, selection net ≥ 1) or mechanism; loss caps as in the schema; no coverage gate |
| Pairing | `full_rerun`: the parent and the candidate both run the stage's rows |
| Patch scope | `one_module` |

**Warm start (optional).** `authorization.json` may name the N17–N20 strategy instead:
- 4-group rounds;
- guard panel on;
- coverage gate before staging, no identical re-gates, same-slot revisions;
- row-store pairing with refresh;
- primitives across modules.

A warm start skips the procedure's own learning curve. Record which start was used.

## Triggers: the evidence that justifies a change

| Trigger | Evidence in the iteration-`k` records | Typical change |
|---|---|---|
| Blocked tasks piling up | ≥ 3 carried tasks whose first failure is the same class across ≥ 2 iterations | target selection → capability or open-failure groups |
| Stale measurements | parent rows older than the last acceptance on the target tasks; proposer diagnosis cites a parent behaviour that the current parent no longer shows | run an inventory; refresh stages |
| Acceptance overfit to dev | an accepted or passing candidate has dev net > 0 but selection net ≤ 0, or later loses on held-in tasks | add selection seeds, a guard panel, or hidden synthetic regression |
| Proposer had no data | a target stops before perception (`relation_unsupported`, `language_unparsed`) | observe-only frames; the coverage gate |
| Cost | starts per accepted candidate above the launch budget's pace | row-store pairing; smaller per-group stages; fewer seeds on panels |
| Gains cancelling losses | a candidate's group net is positive while a panel loses rows | per-role loss caps; ≤ 1 loss per task |
| Losses on unchanged code paths | a loss whose episode never calls a touched function | runtime code-path exemption; no identical re-gates |
| Measured reach limits | ≥ 2 rejected proposals cite the same limit outside the one allowed module | primitive patches across modules with declared `TOUCHED_MODULES` |

## Invariants: the outer loop may not change these

1. **Seed pools stay disjoint.** Development 86–95, selection 76–85, regression panels on 76/80/85 (non-target tasks only), eval 1–50. A strategy may use a subset of a pool, never another pool.
2. **Long-Pro is never used:** no task, seed, frame, or result, in any role, by any agent.
3. **Proposers never see** selection, guard, regression or hidden-synthetic rows, synthetic definitions, or simulator assets.
4. **The program stays shared and task-agnostic.** It uses the public API only; no object, scene, colour or seed literals; task facts come only from adapters.
5. **Integrity gate:** hashes frozen before every stage; every planned row resolved; timeouts ≤ 2%; no crash.
6. **Every acceptance passes an evaluator review** of genuine completions, on top of the numeric gate.
7. **Launch budgets** (starts, proposer calls, VLM dollars, GPUs) come from the human authorization and are not raised.
8. **The model** is Opus for every agent.
9. **The start program and the interface** (`authorization.json`) are fixed for the whole run. With A0, the interface is `original`.
