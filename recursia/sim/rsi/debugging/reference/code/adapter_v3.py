"""Frozen, data-only adapter from public runtime language. No task/scene/seed input."""
import copy
import json
from pathlib import Path
import re

REVISION = "runtime_language_v3_20260928"
DATA = json.loads(Path(__file__).with_name("adapter_lexicon_v3.json").read_text())
SUPPORTS = {"plate", "stove", "cabinet", "top of the cabinet", "basket", "tray", "caddy", "wine rack", "cabinet shelf", "shelf"}
PARTS = {"microwave", "drawer", "stove knob"}
# Generic nouns retain their literal semantics: do not infer a scene-specific subtype.
GENERIC_NOUNS = {"pan": ["pan"], "bottle": ["bottle"], "cup": ["cup"]}


def normalize(language):
    if not isinstance(language, str) or not language.strip():
        raise ValueError("A nonempty public runtime language string is required")
    return " ".join(language.lower().split()).strip(" .!?;,")


def descriptors_for(language):
    text = normalize(language)
    vocab = copy.deepcopy(DATA["VOCAB"])
    vocab.update(GENERIC_NOUNS)
    # Apply the same prefix rule to every object noun. Existing lexical entries
    # retain their frozen prompts; new entries inherit the base noun prompts.
    for phrase, prompts in list(vocab.items()):
        if phrase in SUPPORTS or phrase in PARTS or phrase.split()[0] in DATA["SIDE"]:
            continue
        for side in DATA["SIDE"]:
            vocab.setdefault(side + " " + phrase, list(prompts))
    d = {"objects": {}, "supports": {}, "articulated": {},
         "side_words": copy.deepcopy(DATA["SIDE"]),
         "language_templates": copy.deepcopy(DATA["TEMPLATES"]),
         "anaphora_templates": copy.deepcopy(DATA["ANAPHORA"]),
         "support_prompts": []}
    for phrase, prompts in vocab.items():
        # The shared two-instance template captures the singular of noun+s.
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"s?(?![a-z])", text):
            kind = "supports" if phrase in SUPPORTS else "articulated" if phrase in PARTS else "objects"
            d[kind][phrase] = list(prompts)
    if "stove" in d["supports"]:
        d["articulated"]["stove knob"] = list(vocab["stove knob"])
    d["support_prompts"] = next(iter(d["supports"].values()), [])
    # Legacy interface compatibility is selected by a linguistic construction,
    # identically for any instruction. No scene, task ID, arm, or suite branch.
    if re.search(r"on the (left|right) plate(?: and |$)", text):
        d.update(subgoal_regex=r"put the (.+?) on the (left|right) plate",
                 side_select={"left": "min", "right": "max"}, relation="on",
                 support_prompts=list(vocab["plate"]))
    elif re.search(r"to the (left|right) of the plate$", text):
        d.update(subgoal_regex=r"put the (.+?) to the (left|right) of the plate",
                 side_select={"left": "min", "right": "max"}, relation="beside",
                 support_prompts=list(vocab["plate"]))
    return d


def manifest_from(legacy):
    result = copy.deepcopy(legacy)
    result["written_before_any_simulation"] = False
    result["adapter_revision"] = REVISION
    result["note"] = ("Post-N6 measurement correction. Nominal descriptors are documentation only; "
                      "the runner regenerates descriptors from env.handle.task_language with the same "
                      "frozen rule for every task and arm. Historical target exposure remains.")
    for task in result["tasks"].values():
        task["descriptors"] = descriptors_for(task["language"])
        task["note"] = "Nominal language is a label, never the runtime goal."
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("legacy_manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(manifest_from(json.loads(args.legacy_manifest.read_text())), indent=1) + "\n")
