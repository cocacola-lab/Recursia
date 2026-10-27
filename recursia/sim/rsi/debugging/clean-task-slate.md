# Clean Task Slate: RSI Zero-Shot Transfer Debugging

Use this before (re)starting a run or an iteration.

1. **Round directory.** Use a new directory under `outputs/` (for example `outputs/rsi_zs_debug_<date>`). Never resume into a directory that holds another run's stages. Phase logs and gate files are write-once.
2. **Authorization.** Write `authorization.json` once: GPUs, `K`, start cap and reserve, proposer-call cap, VLM dollar cap, and the optional warm-start strategy. After launch, nobody edits it.
3. **Inputs.** `rsi.py init` (called by `harness/launch.sh`) copies them from `reference/`:
   - `N0` from the start program (A0 by default: the clean start, original interface);
   - the manifest, ordering and synthetic tasks (it verifies their hashes);
   - the code, with the runner patched.

   It then runs the offline harness tests. Do not copy files by hand.
4. **Services.** Perception 8114–8116 healthy. VLM proxy on 8117 with model `claude-opus-5-5` and its own ledger. GPUs free.
5. **Boundary check.** Search the round directory for `libero_10`, `long[0-9]` and `LIBERO-Long`. It must contain no Long-Pro task, seed, frame or result. Keep evaluation outputs in separate directories that the agents never read.
6. **Seeds.** Development 86–95, selection 76–85, regression 76/80/85 (non-target tasks), eval 1–50 (never here).
7. **Strategy.** `iter_1/strategy.json` = `iter_1/strategy_template.json` (`S_1` or the authorized warm start), frozen by `rsi.py freeze --iter 1`.
8. **Never move or re-tag** existing `rsi-N<k>` tags or snapshot files.
