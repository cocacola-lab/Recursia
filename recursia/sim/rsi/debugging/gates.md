# Acceptance Gates

A candidate `C<k>I<j>` replaces the parent only if it passes every active gate. Which gates are active, and their thresholds, come from `S_k.gate` ([strategy-hyperparameters.md](strategy-hyperparameters.md)).

Implementation: [harness/rsi_lib.py](harness/rsi_lib.py), driven by `rsi.py`:
- `analyze_phase`: the integrity gate (timeout check);
- `coverage_gate`: the coverage gate;
- `compare` and `evaluate_gate`: loss caps in both pairing modes, outcome, mechanism, and the code-path exemption from `RUNTIME_MARKERS`;
- `apply_proposal` and `slot_check`: build-time rules.

## Order

```text
offline:  proposer tests pass ─► build checks (`rsi.py build`: scope, forbidden literals, unique anchors, no identical re-gate) ─► [coverage gate]
stage:    frozen hashes ─► run rows ─► integrity gate
paired:   loss caps per role ─► (outcome OR mechanism) ─► evaluator review ─► accept | reject
```

## 1. Integrity gate (always on)
- Every planned row is resolved exactly once.
- Policy, adapter and descriptor hashes match the frozen files.
- `state_idx = (seed − 1) mod 50`, `sandbox_rc = 0`, no crash traceback, recorder files present.
- Timeouts ≤ 2% of rows.

An ineligible stage is never gated. Fix the infrastructure, then rerun **only** the flagged rows as `<phase>_infra1`.

## 2. Coverage gate (`coverage_gate_before_staging`)
Runs offline, before any simulation. The parent's and the candidate's language parse and dispatch run on every `libero_90` instruction:
- the group's tasks must now dispatch;
- every other instruction's parse and dispatch output is unchanged;
- `LA_SUPPORTED` may gain only the group's predeclared `(relation, side)` pairs and lose nothing.

The result is written to `proposals/<phase>/coverage_gate.json`. A stage cannot start without `ok: true`.

## 3. Loss caps (always on; values from `S_k.gate.loss_caps`)
- **Pairing:** every candidate row is paired with the parent's row on the same `(task, seed, state_idx)`.
- **Loss:** a parent success that the candidate fails.
- **Caps:**
  - each role's loss count must stay within its cap;
  - no task may lose more than 1 row;
  - with a small per-group stage (2 seeds per role), every cap is 1.
- **Never netted:** gains never cancel losses. A candidate with +6 on development and 2 ordinary-regression losses is rejected.

## 4. Support: outcome OR mechanism
- **Outcome:** group net official gain ≥ `dev_net_min` (2) on development **and** ≥ `selection_net_min` (1) on selection.
- **Mechanism** (only for groups with `mechanism_allowed` and failure classes predeclared in `predeclared.json`):
  - the development first-failure count in those classes is > 0 for the parent and 0 for the candidate;
  - the selection count in those classes decreases;
  - selection net ≥ 0.

  The evaluator then verifies that the failing episodes moved to a **later physical stage**.

## 5. Runtime code-path exemption (`runtime_code_path_exemption`)
A loss on a row whose candidate episode **never executed** a changed branch is counted as variance, not as a loss.
- Evidence: none of the proposal's `RUNTIME_MARKERS` appears in that episode's stdout. Markers are strings printed only on the changed branches, and the evaluator checks this in review.
- No markers, or no stdout, means no exemption.
- Exempt losses still count against the group net. Each exemption is listed in `gate.json` and the acceptance record.

GraspNet candidates differ between identical programs. A 3/10 vs 4/10 change with identical binding and support records is variance, neither credit nor blame.

## 6. Re-gating rules (`no_identical_regate`, `same_slot_revision`)
- **No identical re-gate:** a rejected candidate is never staged again unchanged, to "try its luck".
- **Same-slot revision:**
  - when the proposer's offline forecast is below the gate, the evaluator may ask the same proposer once for a revision in the same slot;
  - the revision is a new candidate with new hashes;
  - it counts against `max_retries_total`.
- **Sequential rebuild:** candidates are staged one at a time. Each is rebuilt on the **current** parent, which includes every acceptance earlier in the iteration.

## 7. Evaluator review (always on)
Before writing `accepted`, the evaluator checks four things:
- the gained rows are genuine completions (the truth predicate holds at the end; there was no accidental push);
- the new first failures on previously passing rows are explained;
- the tests and hashes match the gated build;
- the proposer's claims agree with `analysis.json`.

The review can turn a numeric pass into a rejection, never a numeric failure into an acceptance.

## Records
Each candidate gets the following records:
- `results/<phase>/gate.json` (written once, never overwritten);
- `proposals/<phase>/predeclared.json`, written before the stage: parent, group, mode, named tasks, failure classes, and the predicted gains;
- an acceptance record in `acceptance.md`, with the decision, the exemptions and the review notes, written **before** the next proposer is dispatched.
