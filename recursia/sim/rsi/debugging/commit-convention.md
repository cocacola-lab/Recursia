# Snapshot and Record Convention

The inner-loop evaluator writes one snapshot per iteration, `N<k>`. The evaluation runbook consumes snapshots only through these files.

## Round layout

```text
{R}/
  authorization.json            # human, once: GPUs, K, budgets, optional warm-start strategy
  programs/N0.py                # start program (A0 by default, the clean start) with the marker 'arm_90': 'N0'
  programs/N<k>.py              # snapshot = accepted program, marker 'arm_90': 'N<k>'
  programs/C<k>I<j>.py          # candidates (rsi.py build)
  programs/hashes.txt           # sha256 of every program file
  lineage.json                  # N<k> -> parent, accepted candidates (C<k>I<j>), their gate.json paths
  iter_<k>/
    strategy.json  targets.json  STRATEGY_UPDATE.md  strategy_sha256.txt      # outer loop, frozen before the iteration
    evaluator_notes.md  RESULT.md  procedure_observations.md  DONE           # evaluator
  state.json                    # harness state: current parent, accepted, gated, slots, snapshots
  code/                         # harness + runner + modules (rsi.py init); hashed in prepared_inputs_sha256.txt
  proposals/<phase>/            # BRIEF/ (proposer inputs) + proposer files + build.json + predeclared.json + coverage_gate.json
  results/<phase>/              # analysis.json attribution.md integrity.json vlm_infra_episodes.json gate.json
  results/<phase>/evaluator_only/   # rows.jsonl selection.json (never shown to a proposer)
  iter_<k>/aggregates.json      # counts only: what the outer loop reads about evaluator-only roles
  acceptance.md                 # one record per candidate, in order
  frozen_sha256_<phase>.txt     # hashes frozen before each stage
  OUTER_LOOP_REPORT.md          # at the end
```

Phase names:
- baselines (`full_rerun` pairing): `baseline<k>_<arm>_<group>`;
- candidates: `chunk<k>_iter<j>`, arm `C<k>I<j>`; same-slot revisions: `chunk<k>_iter<j>_rev<n>`, arm `C<k>I<j>r<n>`;
- refresh stages: `refresh<k>_<arm>_<group>`;
- inventory: `inventory<k>`;
- infrastructure reruns: `<phase>_infra1`.

## Snapshot rules
- `rsi.py snapshot --iter k` writes `programs/N<k>.py`: byte-identical to the last accepted candidate (or to the parent if nothing was accepted), except that the program marker reads `'arm_90': 'N<k>'`. It records the sha256 in `hashes.txt` and `lineage.json`, then writes `aggregates.json` and `DONE`.
- A snapshot is never edited. A correction is a new snapshot.
- A git commit is optional and local only: `Snapshot RSI N<k>: <accepted arms or "none">`, then the tag `rsi-N<k>`. Never re-tag, and **never push**.

## Commit message (if committing)

```
Snapshot RSI N<k>: <C<k>I<j> ...|none>

Parent: N<k-1> (<sha12>)  ->  N<k> (<sha12>)
Strategy S_<k>: <dimensions changed vs S_<k-1>, or "unchanged">
Slots: <n> (<accepted> accepted, <rejected> rejected, <revisions> revisions)
Development / selection net (accepted): +<d> / +<s>
Starts this iteration: <n>; proposer calls: <n>; VLM $<x>

Refs: iter_<k>/RESULT.md, iter_<k>/STRATEGY_UPDATE.md
```

Report LIBERO-90 numbers only. Write no Long-Pro numbers in snapshot records.
