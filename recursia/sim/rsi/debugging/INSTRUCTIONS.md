# RSI Zero-Shot Transfer: Debugging Instructions

This runbook improves one shared, task-agnostic robot program by recursive self-improvement (RSI) on the LIBERO-90 debug set.
Each RSI iteration `N_k` produces one frozen snapshot. LIBERO-Long-Pro is the transfer target, and this runbook never touches it:
evaluation is a separate runbook, [../evaluation/](../evaluation/).

1. Complete the setup in `recursia/sim/README.md`.
2. Export `RECURSIA_SIM_ROOT` (= `recursia/sim`), `MUJOCO_GL=egl`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, and start the perception
   servers (SAM3 8114, GraspNet 8115, PyRoki 8116) in a persistent tmux pane as in `recursia/sim/README.md` § Setup.
3. The human authorizes the whole run **once**: copy [harness/authorization.template.json](harness/authorization.template.json)
   and set GPUs, budgets (simulation starts, proposer calls, VLM dollars), the number of iterations `K`, the start program
   (`A0` with `interface: original`, the clean start; `S1`, `K2` or `N20` with `interface: rsi` only for comparisons) and the start strategy (`S_1`, or the N17–N20 warm start). After that, no human intervenes in the loop.
4. Launch: `bash harness/launch.sh outputs/rsi_zs_debug_<date> <authorization.json>` (see [harness/README.md](harness/README.md)).
   It creates the round with `rsi.py init`, starts the VLM proxy, and starts the **outer-loop agent** with
   [outer-loop-agent-prompt.md](outer-loop-agent-prompt.md) in tmux. Every agent in this runbook runs on Opus (`claude-opus-5-5`).
   To stop the whole run: `touch <round>/STOP`.
5. For each iteration `k = 1 .. K`:
   - The outer-loop agent fixes the strategy `S_k`, see [strategy-hyperparameters.md](strategy-hyperparameters.md).
   - The inner loop runs under `S_k`:
     - the inner-loop evaluator follows [main-agent-prompt.md](main-agent-prompt.md);
     - one proposer per slot follows [proposer-prompt.md](proposer-prompt.md);
     - candidates are gated by [gates.md](gates.md) on the data roles of [data-roles.md](data-roles.md).
   - The evaluator snapshots `N_k` ([commit-convention.md](commit-convention.md)).
   - The outer-loop agent writes the strategy update `S_k -> S_{k+1}`.
6. Hand the snapshots to [../evaluation/](../evaluation/).
7. Use [clean-task-slate.md](clean-task-slate.md) before any rerun.
