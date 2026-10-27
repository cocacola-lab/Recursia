# Reference Trajectories

`N20C_success34/` holds 34 successful LIBERO-Long-Pro episodes of the frozen N20 stack with closed-loop inference (N20C).

They come from the check run of 2026-10-27, which ran in this workspace on the 35 interim rows the reference N20C run had solved:
- 33 succeeded in the run itself;
- `long7_swap` s41 succeeded on its first diagnostic retry;
- `long1_task` s42 did not reproduce (0 of 14 attempts). The VLM's object-binding answer for "alphabet soup" changed since the reference run, so the robot picks the look-alike tomato-sauce can.

| File (per episode) | Content |
|---|---|
| `attempt_manifest.json` | identity (task, suite, seed, state index), program and runner hashes, official outcome, final goal predicates, orchestration summary, costs |
| `stdout.txt` | the stack's tagged log: binding, grasp, lift, place and subgoal lines, and orchestration calls |
| `summary.json`, `llm_calls.jsonl`, `stack_calls.jsonl` | closed-loop record: plan, decisions after every subgoal, visual checks, and every interface call |
| `vlm_calls.jsonl`, `binding_calls.jsonl` | the stack's VLM binding queries and answers, and SAM3 binding results |
| `grasp_candidates.jsonl`, `gate_decisions.jsonl` | grasp candidates and dependent-action gate decisions |
| `motion_calls.jsonl.gz`, `contact_truth.jsonl.gz`, `events.jsonl.gz` | motion interface calls; simulator contact truth (diagnosis only); the API-call event timeline |

`index.json` lists every episode with its outcome, steps, decisions and cost.

**Not included:**
- camera frames and depth arrays (about 0.5 GB per episode);
- the program copy (`N20C.py`; its hash is in every manifest).

Simulator truth is recorded by the runner for diagnosis only; the program never sees it.

**Export your own run** the same way:

```bash
.venv-libero/bin/python3 rsi/evaluation/harness/rsi_eval.py export-trajectories --out outputs/<run> --dest <dir>   # add --all for failures too
```
