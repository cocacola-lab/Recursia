You write ONE task-orchestration program for a Franka robot in a tabletop scene. The attached image is the initial camera view
(agentview). The robot's skills, perception, motion control, verification and safety gates already exist in a frozen,
version-fixed stack (RSI N20). Your program only decides WHICH stack skills to call, in what order, with which arguments, and how
to react to their structured results. You cannot and must not implement motion, grasping, placement, perception or verification.

Instruction: "{{INSTRUCTION}}"

Vocabulary for this instruction (the only phrases the stack can bind; use them verbatim):
{{VOCABULARY}}

Capabilities of the frozen stack (relations, side words each relation implements, executors):
{{CAPABILITIES}}

Retrieved interface catalog entries (version-pinned to this stack):
{{CATALOG}}

The program is INTERPRETED by a restricted interpreter, not run as Python. Anything outside this subset is rejected and nothing runs:
1. Exactly one top-level `def run(context, stack):`. No imports, no helper functions, no classes, no other top-level code.
2. Statements: `name = value`, `if / elif / else`, `for x in range(N)` with an integer literal N <= 8, `for x in some_list` over a list
   you hold (<= 16 items), `break`, `continue`, `return`, `pass`.
   NOT allowed: while, try/except, with, raise, `+=`, comprehensions, lambda, global, del, slices.
3. Expressions: str / int / float / bool / None literals, lists, tuples, dicts with string-literal keys, indexing `value[key]`,
   `and / or / not`, comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `is`), `a if c else b`, and `+` between two strings,
   two lists, or two integers (convert other values with `str(x)` first; a failure reason may be None).
4. Calls: ONLY `stack.<method>(...)` for the catalog methods, `len(x)` and `str(x)`. There are NO attribute reads or method calls on
   values: write `r['status']`, never `r.get('status')`; lists cannot be appended to, so build a list literal such as `[a, b]`.
   Every stack result is a dict whose keys are always present (see the catalog), so indexing a documented key never fails.
5. Names must not start with `_`; `stack` may only appear as `stack.<method>(...)`; do not assign to `context` or `stack`.

Orchestration rules:
6. Declare every subgoal with `stack.subgoal(...)` using vocabulary phrases and implemented side words, then call
   `stack.plan([...])` exactly ONCE before any `stack.execute`. Literal arguments are capability-checked before any motion. If `plan`
   returns status 'unsupported', call `stack.finish('unsupported', ...)` and return: never substitute an unrelated capability (for
   example, an articulated part such as a microwave is never the container of an object relation).
7. Keep the instruction's order of actions unless its meaning requires another (an object goes into a drawer before the drawer is
   closed). One subgoal per object placement / per part action; "both X and Y" means two subgoals.
8. After each `stack.execute`, inspect `result['status']`:
   - 'control_completed' -> continue. This is control completion, not proof of success; never repeat a completed subgoal.
   - a failure -> you may call `stack.recover(result, 'retry_subgoal')` ONLY if 'retry_subgoal' is in
     `result['allowed_recovery_actions']` (at most once per subgoal). Each retry spends steps from the episode's fixed 4000-step
     budget, so weigh it against the remaining subgoals; otherwise continue with the remaining subgoals when they are still
     meaningful, or stop with `stack.finish`.
   - `stack.locate(phrase)` / `stack.observe()` give fresh evidence when a branch needs it.
9. End with `stack.finish(<short status>, <reason string>)`. Your declared status is recorded as a claim only.
10. Keep the program short and readable (well under 80 lines).

Return JSON with two fields: "plan_notes" (2-5 sentences: the subgoals you chose from the instruction and the image, and your
branching / recovery policy) and "program" (the complete source, no markdown fences).
