# Rules Learned in N1–N20

Each rule was added after an incident in the original rounds. Both the inner-loop evaluator and the outer-loop agent follow them.

## Process
1. **Build only after the proposer's FINAL report.** Never build from interim files. (One stage was built from an interim `patch.py` and had to be killed and archived.)
2. **Resume, don't re-dispatch.** If a proposer subagent dies or the session restarts, resume the same agent with its context and count the call once.
3. **Predict before measuring.** Write `predeclared.json` (including predictions for parent stages) before every stage. Write the acceptance record before dispatching the next proposer.
4. **Rebuild sequentially on the current parent.** Two candidates prepared on the same old parent are never staged in parallel.
5. **Data and harness amendments happen only between iterations.** They are recorded, hashed, and keep the original file. Examples: the IK floor −0.035, the `min_opening` clamp 0.03, the retrying VLM proxy, the outage probe. Earlier rows are not reused across a harness amendment; refresh them.
6. **Strategy changes happen only in the outer loop, between iterations.** The evaluator logs friction in `evaluator_notes.md` and never works around `S_k`.

## Measurement
7. **Key identity checks on `task_short`**, never on a prefix of the full task name. (Checks were vacuous for several chunks until re-run.)
8. **Read `gripper_nonobject_contacts` for robot contacts.** `contact_truth` pairs are truncated to 40 entries, and a "no robot contact" conclusion drawn from them was wrong.
9. **Grasp-service nondeterminism is not a patch effect.** GraspNet candidates differ between identical programs.
10. **A step-horizon ending can be a crash.** Check `crashed_traceback` before calling a stage crash-free.
11. **Reported numbers come from `analysis.json` and truth files**, never from a proposer's or subagent's own claims.
12. **Count timeouts.** More than 2% `episode_timeout` rows make the stage ineligible.

## Infrastructure
13. **Do not start episodes during a provider outage.** The per-episode VLM probe holds new episodes back. Rows that recorded a proxy transport error are rerun once, selected by the flag only (`<phase>_infra1`), and both results are kept.
14. **The proxy does not retry HTTP 400.** A burst of 400s on a fixed probe request is a provider outage, not a request bug.
15. **Use pid files to stop processes, never `pkill -f <pattern>`.** The pattern can match the caller's own shell.
16. **Waits are count-based.** Count completed rows in the attempts ledger. Never grep a possibly empty log, which can trigger a false "done".
17. **Credentials never reach workers.** Launch workers with `env -u ANTHROPIC_API_KEY`; model calls go through the loopback proxy (127.0.0.1:8117). Keep keys out of logs and committed files.

## Boundary
18. **Never run the eval set inside the loop.** Long-Pro seeds 1–50 are locked. The original loop ran a Long-Pro smoke through N6. That overlap was retired, and it is not allowed in this runbook.
19. **Proposers see development data only.** Selection, guard, regression and hidden-synthetic details never appear in a proposer prompt, a revision request, or a file the proposer can read.
20. **The outer loop sees aggregates only for evaluator-only roles**, and nothing of Long-Pro. If it opens such a record by accident, it logs the incident in `boundary_incidents.md`.
21. **Clean-start isolation.** In a run started from A0, no agent reads earlier programs, outputs or learned skills (S1, K0–K2, N1–N21, external skill libraries). They were developed with Long-Pro exposure, and copying from them would bring it back in.
