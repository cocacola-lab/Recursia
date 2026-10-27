# Infrastructure and Integrity

## Services
- **Perception:** SAM3 8114, Contact-GraspNet 8115, PyRoki 8116, all healthy before launch.
- **Grasp backend:** pin `GRASPNET_SERVICE_URL`. Write a per-episode backend marker and check it in the analysis. A different backend, e.g. AnyGrasp, is a separate arm; never mix backends within one run.
- **VLM proxy:** the retrying proxy `reference/code/vlm_proxy.py` on 127.0.0.1:8117, with this experiment's own ledger `logs/vlm_ledger.jsonl` and a hard dollar cap.
  - Start only with ≥ $5 headroom.
  - The proxy retries transport errors up to 4 attempts. It does **not** retry HTTP 400.
- **Credentials:** workers hold no credential. `start_orch.sh` unsets `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `OPENROUTER_API_KEY`; the proxy holds the key.

## Episode launcher (`run_plan_orch.sh`)
- **Slots:** 5 GPU slots, one episode per slot; the slot list comes from the authorization.
- **Per-episode provider probe** (`vlm_probe.py`): no episode starts while the provider is unreachable. Waits are logged as `VLM_WAIT`.
- **Limits:** each episode runs with 4000 steps and a 1500 s timeout. Unique attempt directories are used; an existing directory aborts the run.
- **Start ledger:** shared, with a cap and a reserve. Check headroom ≥ the planned episodes + reserve before launch.
- **Stopping:** stop runs with pid files (never `pkill -f`), and wait by counting rows in the attempts ledger.

## Integrity checks (`interim100_cl.py check | summarize`)
- every planned `(task, seed)` resolved exactly once;
- arm program sha, stack sha and orchestration file hashes match the frozen inputs;
- `state_idx = (seed − 1) mod 50`;
- `sandbox_rc = 0`, or a recorded stack exception (the stack's own exception counts as a failure, not as an infrastructure error);
- an episode summary exists, even after a horizon interrupt;
- the backend marker matches;
- the transport-error count per episode.

## Infrastructure rerun rule (predeclared)
- **Selection:** an episode with **any proxy transport error** is rerun **once**, as phase `rerun_infra_N<k>C`. It is selected by the flag only, never by outcome.
- **Reporting:** keep and report both summaries, `summary.json` and `summary_infra_rerun.json`.
  - Example: N21C + GraspNet scored 29 without and 32 with its four outage reruns.
- **Burst of 400s on the fixed probe:** a provider outage. Record its window in an amendment.

## Expected cost and time (from the N20C and N21C runs)

| Scale | Episodes | Wall time (5 workers) | VLM cost |
|---|---|---|---|
| Interim | 100 | ~2.5 h (mean ~370 s per episode) | ~$15 |
| Full | 1000 | ~21 h | ~$150 |

Set the dollar cap with margin (the N20C cap was $35 for 100 episodes).
