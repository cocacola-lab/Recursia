You supervise a tabletop robot that runs a fixed plan of subgoals with frozen, version-fixed skills (RSI N20). Image 1 is the scene
at the start of the task; image 2 is the scene now. Choose the NEXT action from "allowed_actions" only.

Current state (JSON):
{{STATE}}

Meaning of the fields:
- status "control_completed": the skill finished its motions (e.g. released the object at the target). It is NOT proof of success.
- "check": a separate visual check of the subgoal's relation_sentence after completion ("yes" / "no" / "unsure" / null).
- other statuses are failures reported by the frozen skill (binding_failed = target not found or not confirmed, nothing moved;
  verification_failed = a safety or identity gate refused; execution_failed = a motion could not be done).
- steps_remaining: control steps left in the episode's fixed 4000-step budget; a subgoal attempt typically needs about
  step_cost_estimate_per_subgoal steps. The harness only offers retry / replace when the budget covers it and the remaining subgoals.

Actions:
- execute <id>: run a pending subgoal (the first time).
- retry <id>: run a failed subgoal once more (offered only when nothing is held and budget allows).
- replace <id>: pick up and place again an object whose completed placement the visual check judged wrong (offered only then).
- stop: end the episode.

Guidance: complete every subgoal the instruction needs; prefer executing pending subgoals in plan order; retry a failed subgoal when
the remaining subgoals still fit the budget or when later subgoals depend on it; replace only when the check says "no" and image 2
confirms the object is not where it should be; stop when all subgoals are completed (and checks are not "no"), or when nothing useful
remains. Your choice must be exactly one entry of allowed_actions.

Return JSON: {"action": "execute" | "retry" | "replace" | "stop", "subgoal": "<id>" or null, "reason": "<one sentence>"}.
