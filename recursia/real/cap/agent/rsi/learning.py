"""Source findings -> reusable patterns -> merge, following sim's fix-loop.

The model sees only this method's source evidence and current skill documents.
Learning is read-only with respect to cameras/robot; Markdown is never executed.
"""
import json

from .observation import image_paths
from .prompts import API, public_task
from .promotion import documents
from .store import read_json

FINDINGS_SCHEMA = {"type": "object", "properties": {
    "analysis": {"type": "string"}, "findings": {"type": "string"}},
    "required": ["analysis", "findings"], "additionalProperties": False}

UPDATE_SCHEMA = {"type": "object", "properties": {
    "analysis": {"type": "string"}, "reason": {"type": "string"},
    "updates": {"type": "array", "items": {"type": "object", "properties": {
        "file": {"type": "string"}, "content": {"type": "string"}},
        "required": ["file", "content"], "additionalProperties": False}}},
    "required": ["analysis", "reason", "updates"], "additionalProperties": False}


def source_evidence(store, folder, execution):
    evidence = {"source_path": str(folder.relative_to(store.root)), "execution": execution, "turns": []}
    images = []
    for turn in sorted((folder / "execution" / "turns").glob("turn_*")):
        step = turn / "step.json"
        if not step.is_file():
            continue
        raw = read_json(step)
        item = {"evidence_path": str(step.relative_to(store.root)),
                "step": {key: value for key, value in raw.items() if key != "decision_prompt"}}
        response = turn / "response.json"
        if response.is_file():
            item["model_feedback"] = read_json(response)
        observation = turn / "observation.json"
        if observation.is_file():
            item["observation_path"] = str(observation.relative_to(store.root))
            paths = image_paths(read_json(observation))
            if all(path.is_file() for path in paths):
                images = paths
        evidence["turns"].append(item)
    return evidence, images


def evolve(store, task, folder, execution, request, promotion):
    """Two tool-less calls: discover evidence, then merge reusable skills."""
    evidence, images = source_evidence(store, folder, execution)
    prompt = ("Analyze this source execution and write findings as Markdown. "
              "Use actual executed code, stdout/stderr, tool results, observations and repairs; "
              "include failures and distinguish observed facts from hypotheses. FINISH is not proof of success. "
              "Identify the first failure, what was changed, its observed effect, and any transferable lesson. "
              "No new execution, robot commands, invented results or target evidence. Cite the supplied evidence paths. "
              "If there is no reusable lesson, say so.\n" + json.dumps(
                  {"task": public_task(task), "source_evidence": evidence}, ensure_ascii=False))
    finding = request(folder / "reflection", prompt, schema=FINDINGS_SCHEMA, images=images)
    if not finding["findings"].strip():
        raise ValueError("source reflection must contain findings")
    findings_path = folder / "findings.md"
    findings_path.write_text(finding["findings"])
    prompt = (API + "\nUpdate the reusable source skill library from these findings. "
              "Merge into an existing topic when it already covers the lesson; do not append duplicate patterns. "
              "For substantive patterns use Trigger, actual generalized Python code (typically 5-20 lines), "
              "Why it works, and source evidence/provenance. One-line lookups may use a table. "
              "Failures can teach reusable lessons, but do not present untested remedies as validated. "
              "Remove one-off object/scene coordinates; generalize real executed snippets using tool results or arguments. "
              "Return complete replacement Markdown contents for affected topic files (e.g. localize.md, grasp.md, "
              "transport.md, manipulation.md). Unmentioned files remain unchanged. "
              "The legacy knowledge/functions are reference only, not files to replace. "
              "Use only current source evidence, never target results. "
              "If nothing generalizable should change, return updates=[] and explain the no-op in reason. "
              "This is read-only authoring; no snippets will execute now.\n" + json.dumps(
                  {"current_skill_files": documents(store.root), "legacy_knowledge": store.knowledge,
                   "findings": finding["findings"], "findings_path": str(findings_path.relative_to(store.root)),
                   "source_evidence": evidence}, ensure_ascii=False))
    response = request(folder / "skill_update", prompt, schema=UPDATE_SCHEMA)
    if not response["reason"].strip():
        raise ValueError("skill update requires a reason, including for no-op")
    promotion.apply(response["updates"])
    return response["reason"], findings_path
