"""Deterministic translator: la_parse subgoal records -> an orchestration program that issues the same subgoals, in the same
order, with no branching or recovery (interpreter subset v2) (the baseline driver's behaviour expressed through the interface). Used by the offline
equivalence test; also an example of the program format."""

KEYS = ("relation", "object", "reference", "part", "side", "object_side", "instance")


def args_from_record(sg, td):
    inv = {}
    for w, c in (td.get("side_words") or {}).items():
        inv.setdefault(c, w)
    a = {"relation": sg["relation"], "object": sg.get("object"), "reference": sg.get("support"), "part": sg.get("part"),
         "side": inv.get(sg.get("side"), sg.get("side")) if sg.get("side") is not None else None,
         "object_side": inv.get(sg.get("object_side"), sg.get("object_side")) if sg.get("object_side") is not None else None,
         "instance": sg.get("instance")}
    return {k: v for k, v in a.items() if v is not None}


def program_for(subgoals, td):
    lines = ["def run(context, stack):"]
    names = []
    for i, sg in enumerate(subgoals):
        kw = ", ".join(f"{k}={v!r}" for k, v in args_from_record(sg, td).items())
        lines.append(f"    s{i} = stack.subgoal({kw})"); names.append(f"s{i}")
    lines += [f"    sgs = [{', '.join(names)}]",
              "    p = stack.plan(sgs)",
              "    if p['status'] != 'ok':",
              "        stack.finish('unsupported', str(p['unsupported']))",
              "        return",
              "    for sg in sgs:",
              "        stack.execute(sg)",
              "    stack.finish('done', 'rule-equivalent straight-line program')"]
    return "\n".join(lines) + "\n"
