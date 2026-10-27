# RSI Zero-Shot Transfer: Evaluation Instructions

This runbook evaluates a frozen RSI snapshot `N_k` (from [../debugging/](../debugging/)) zero-shot on LIBERO-Long-Pro. There are three methods: rule-based (the stack's own driver), one-shot and closed-loop (LLM orchestration over the snapshot's frozen interfaces).
- One-shot and closed-loop are **only inference methods.** The learned stack is `N_k`, byte-identical and runner-asserted at every episode start.
- No result of this runbook ever flows back into the debugging loop.

> **Just want to reproduce N20C (35/100)?** Use [QUICKSTART.md](QUICKSTART.md): simulation setup plus one command.

0. Choose the evaluation method: **rule-based**, **one-shot** or **closed-loop** ([methods.md](methods.md)). Closed-loop is the reference method. Report rule-based next to any inference method.
1. Complete the setup in `recursia/sim/README.md`. Export `RECURSIA_SIM_ROOT`, `MUJOCO_GL=egl`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Start the perception servers (8114–8116).
2. The human authorizes:
   - the snapshot `N_k`;
   - the scale: **interim** = seeds 41–45, 100 episodes; **full** = seeds 1–50, 1000 episodes;
   - GPUs, the VLM dollar cap and the start budget.
3. Create the experiment directory:
   - **N20 stack:** `harness/rsi_eval.py setup --out outputs/rsi_zs_eval_<arm>_<scale>_<date> --method rule|oneshot|closed --scale interim|full`. Nothing has to be ported.
   - **Any other snapshot:** create `outputs/rsi_zs_eval_N<k><M>_<scale>_<date>` and port the reference as described in [main-agent-prompt.md](main-agent-prompt.md) § 1–2.
4. Launch the evaluation agent (Opus, `claude-opus-5-5`) with [main-agent-prompt.md](main-agent-prompt.md) in a persistent tmux session. Its steps:
   - port the interface to `N_k`;
   - run the offline tests and the rule-equivalence check;
   - check the calibration gate ([calibration.md](calibration.md));
   - run a 1-episode smoke;
   - launch the stage;
   - apply the infrastructure rule ([infrastructure.md](infrastructure.md));
   - write the report ([report-template.md](report-template.md)).
5. Use [clean-task-slate.md](clean-task-slate.md) before any rerun.

The reference run is N20C: interim 100 episodes, **35/100** (POS 12/50, TASK 23/50), in `outputs/rsi_libero90_N20C_closedloop_long100_20261016`.
