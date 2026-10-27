# Clean Task Slate: RSI Zero-Shot Transfer Evaluation

1. **Experiment directory.** Use a new directory per arm and scale (`outputs/rsi_zs_eval_N<k>C_<scale>_<date>`). Phase logs and attempt directories are write-once. Never reuse another run's directory.
2. **Snapshot.** Copy `programs/N<k>.py` from the debugging round. Verify its sha256 against that round's `lineage.json`.
3. **Plan.** Long-Pro long0–long9 × {swap, task} × seeds: 41–45 for interim, 1–50 for full.
4. **Services.** Perception 8114–8116 healthy; the grasp backend pinned; the proxy on 8117 with a fresh ledger and cap; GPUs free.
5. **Calibration.** Reuse the passed gate only if the verify prompt, model, effort and inputs are unchanged. Otherwise re-run it on LIBERO-90 development frames.
6. **Boundary.** Results stay in the evaluation directory. Never copy them, or their diagnoses, into a debugging round. Never use them to change a snapshot.
7. **Never push**, and never edit a snapshot.
