"""Generic RSI harness library (rsi-debugging). Pure functions + file I/O; no simulator imports.

Everything round-specific comes from three files written outside this module:
  {R}/authorization.json            human, once (budgets, GPUs, K, start program, optional warm-start strategy)
  {R}/iter_<k>/strategy.json        outer-loop agent, frozen per iteration (hyperparameters S_k)
  {R}/iter_<k>/targets.json         outer-loop agent, frozen per iteration (groups)
The CLI is rsi.py. Tests: tests/test_rsi_harness.py (no simulator).
"""
from __future__ import annotations

import copy
import csv
import glob
import hashlib
import importlib.util
import json
import py_compile
import re
from collections import Counter, defaultdict
from pathlib import Path

MODEL = "claude-opus-5-5"
HORIZON = 4000

# ----------------------------------------------------------------------------- invariants
# Seed pools are fixed for the whole run (strategy-hyperparameters.md, Invariants). A strategy may use a subset of a pool, never
# another pool. Long-Pro eval seeds 1-50 are in no pool.
DEV_POOL = tuple(range(86, 96))
HELD_POOL = tuple(range(76, 86))
SEED_POOLS = {
    "dev": DEV_POOL, "synthetic_dev": DEV_POOL, "inventory": DEV_POOL,
    "selection": HELD_POOL, "synthetic_selection": HELD_POOL,
    "guard": HELD_POOL, "ordinary_regression": HELD_POOL, "synthetic_regression": HELD_POOL,
}
ROLES = ("dev", "synthetic_dev", "selection", "synthetic_selection", "guard", "ordinary_regression", "synthetic_regression", "inventory")
PROPOSER_VISIBLE_ROLES = ("dev", "synthetic_dev", "inventory")
EVALUATOR_ONLY_ROLES = ("selection", "synthetic_selection", "guard", "ordinary_regression", "synthetic_regression")
GATED_ROLES = ("dev", "synthetic_dev", "selection", "synthetic_selection", "guard", "ordinary_regression", "synthetic_regression")
CAP_KEY = {"dev": "development", "synthetic_dev": "synthetic_dev", "selection": "selection", "synthetic_selection": "synthetic_selection",
           "guard": "guard", "ordinary_regression": "ordinary_regression", "synthetic_regression": "synthetic_regression"}
EVAL_SEEDS = tuple(range(1, 51))
SOURCE_SUITES = ("libero_90", "rsi_source_synthetic")
LONGPRO_PATTERN = re.compile(r"libero_10|^long\d|libero[-_ ]?long", re.I)
TIMEOUT_RATE_MAX = 0.02

PANEL_11 = ("s01_pick_up_the_cream_cheese_box", "s21_open_the_top_drawer_of_the_c", "s28_put_the_black_bowl_on_the_pl",
            "s02_put_the_white_mug_on_the_lef", "s39_put_the_yellow_and_white_mug", "s04_put_the_chocolate_pudding_to",
            "s17_pick_up_the_yellow_and_white", "s30_put_the_middle_black_bowl_on", "s14_put_the_black_bowl_at_the_ba",
            "s26_put_the_black_bowl_on_top_of", "s10_turn_on_the_stove")
TARGET_MODES = ("chunk", "capability_groups", "open_failure_groups", "k_group_round")
PAIRING_MODES = ("full_rerun", "row_store")
SCOPE_MODES = ("one_module", "primitive_across_modules")
GATE_MODES = ("outcome", "mechanism")

ARM_RE = re.compile(r"^(N\d+|C\d+I\d+(r\d+)?)$")
CAND_PHASE_RE = re.compile(r"^chunk(\d+)_iter(\d+)(_rev(\d+))?$")
MARK_ANCHOR = "'arm_s': 'S1',"            # present in S1 and every descendant (K0-K2, N1-N21)
MARK_RE = re.compile(r"'arm_90': '[A-Za-z0-9]+',")   # the arm marker the runner checks (one per program)
START_PROGRAMS = {"A0": "hashes_parent_A0.txt", "S1": "hashes_parent_S1.txt", "K2": "hashes_parent_K2.txt", "N20": "hashes.txt"}
# A0 = clean start (original ASPIRE API only; default). S1/K2/N20 were developed with Long-Pro exposure and need the harness tools.
INTERFACES = ("original", "rsi")
START_INTERFACES = {"A0": ("original", "rsi"), "S1": ("rsi",), "K2": ("rsi",), "N20": ("rsi",)}
ORIGINAL_API = ("get_observation", "segment_sam3_text_prompt", "segment_sam3_point_prompt", "plan_grasp", "plan_grasp_from_point_clouds",
                "get_oriented_bounding_box_from_3d_points", "solve_ik", "move_to_joints", "open_gripper", "close_gripper", "goto_pose",
                "goto_home_joint_position", "subsample_point_cloud", "filter_noise", "rotation_matrix_to_quaternion", "decompose_transform",
                "depth_to_point_cloud", "mask_to_world_points", "pixel_to_world_point", "transform_points", "interpolate_segment",
                "normalize_vector", "select_top_down_grasp", "env.handle.task_language")
SPATIAL_WORDS = ("left", "right")   # general relation words: allowed in patch text under the original interface (no side-word adapter)
FORBIDDEN_API = (r"\.sim\b", r"body_xpos", r"site_xpos", r"set_joint_qpos", r"body_name2id", r"joint_id2name", r"\bqpos\b", r"\.forward\(",
                 r"parsed_problem", r"_eval_predicate", r"obj_body_id", r"_step_once", r"handle\.env", r"\bAPIS\b", r"\.bddl\b", r"\.xml\b",
                 r"\.urdf\b", r"\bmujoco\b")   # simulator ground truth (recursia/sim/README.md); never in a patch


class HarnessError(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise HarnessError(msg)


# ----------------------------------------------------------------------------- io
def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, once=False):
    path = Path(path)
    require(not (once and path.exists()), f"write-once file exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=list) + "\n")


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(value, default=list) + "\n")


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def hash_manifest(root, files):
    root = Path(root)
    return "".join(f"{digest(p)}  {Path(p).relative_to(root)}\n" for p in files)


def verify_hash_manifest(root, manifest_path):
    root = Path(root); bad = []
    for line in Path(manifest_path).read_text().splitlines():
        sha, name = line.split(None, 1)
        p = root / name
        if not p.is_file() or digest(p) != sha:
            bad.append(name)
    require(not bad, f"frozen input mismatch ({manifest_path}): {bad[:5]}")


# ----------------------------------------------------------------------------- round paths and state
class Round:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def p(self, *parts):
        return self.root.joinpath(*parts)

    @property
    def auth(self):
        return read_json(self.p("authorization.json"))

    @property
    def manifest(self):
        return read_json(self.p("task_manifest.json"))["tasks"]

    def state(self):
        return read_json(self.p("state.json"))

    def save_state(self, st):
        write_json(self.p("state.json"), st)

    def iter_dir(self, k):
        return self.p(f"iter_{k}")

    def strategy(self, k):
        verify_strategy_frozen(self, k)
        return read_json(self.iter_dir(k) / "strategy.json")

    def targets(self, k):
        verify_strategy_frozen(self, k)
        d = self.iter_dir(k)
        if (d / "targets_sha256.txt").exists():
            verify_hash_manifest(self.root, d / "targets_sha256.txt")
        else:
            require("targets.json" in (d / "strategy_sha256.txt").read_text(), f"targets for iteration {k} are not frozen yet (inventory first, then: rsi.py freeze --iter {k} --targets)")
        return read_json(d / "targets.json")

    def program(self, arm):
        return self.p("programs", f"{arm}.py")

    def plan_roles(self, phase):
        return read_json(self.p(f"plan_{phase}_roles.json"))

    def results(self, phase):
        return self.p("results", phase)


def new_state(start_arm="N0"):
    return {"current": start_arm, "iteration": 0, "accepted": [], "gated": [], "slots": {}, "snapshots": {"0": start_arm}}


# ----------------------------------------------------------------------------- Long-Pro boundary
def assert_no_longpro_manifest(tasks):
    bad = [k for k, v in tasks.items() if v.get("suite") not in SOURCE_SUITES or LONGPRO_PATTERN.search(k) or LONGPRO_PATTERN.search(v.get("suite", ""))]
    require(not bad, f"Long-Pro or non-source tasks in the round manifest: {bad[:5]}")


def assert_rows_source_only(rows, tasks):
    for r in rows:
        t = tasks.get(r["task_short"])
        require(t is not None, f"unknown task {r['task_short']}")
        require(t["suite"] in SOURCE_SUITES, f"non-source task in a plan: {r['task_short']}")
        require(int(r["seed"]) not in EVAL_SEEDS, f"eval seed {r['seed']} in a plan: {r}")


# ----------------------------------------------------------------------------- strategy
def default_strategy_s1(parent="N0"):
    return {
        "iteration": 1, "parent": parent,
        "target_selection": {"mode": "chunk", "n_groups": 1, "max_tasks_per_group": 5,
                             "inventory": {"run": False, "seeds": [86, 87]}, "carry_blocked": True},
        "debug_data_roles": {"proposer_visible": ["dev", "synthetic_dev", "observe_only_frames"],
                             "evaluator_only": ["selection", "synthetic_selection", "ordinary_regression", "synthetic_regression"],
                             "guard_panel": {"enabled": False, "tasks": "accepted_tasks_not_in_group"},
                             "observe_only_frames": True, "proposer_vlm_usd": 0},
        "seeds": {"dev": list(range(86, 96)), "synthetic_dev": [86, 87, 88], "selection": list(range(76, 86)),
                  "synthetic_selection": [76, 77, 78], "guard": [76, 80], "ordinary_regression": [76, 80, 85],
                  "synthetic_regression": [76, 80], "inventory": [86, 87]},
        "regression_check": {"ordinary_tasks": "panel_11", "hidden_synthetic": True},
        "gate": {"modes": ["outcome", "mechanism"], "outcome": {"dev_net_min": 2, "selection_net_min": 1},
                 "loss_caps": {"development": 5, "selection": 5, "synthetic_dev": 3, "synthetic_selection": 3,
                               "guard": 1, "ordinary_regression": 1, "synthetic_regression": 1, "per_task": 1},
                 "coverage_gate_before_staging": False, "runtime_code_path_exemption": False,
                 "no_identical_regate": True, "same_slot_revision": {"enabled": False, "max": 1}, "timeout_rate_max": 0.02},
        "pairing": {"mode": "full_rerun", "refresh_after_accept": False, "sequential_rebuild_on_current_parent": True},
        "patch_scope": {"mode": "one_module"},
        "slots": {"max_slots": 4, "max_retries_total": 2},
        "justification": "STRATEGY_UPDATE.md",
    }


def warm_strategy_n17_n20(parent="N0"):
    s = default_strategy_s1(parent)
    s["target_selection"].update(mode="k_group_round", n_groups=4, max_tasks_per_group=5, inventory={"run": True, "seeds": [86, 87]})
    s["debug_data_roles"]["guard_panel"]["enabled"] = True
    s["debug_data_roles"]["evaluator_only"] = list(EVALUATOR_ONLY_ROLES)
    s["gate"].update(coverage_gate_before_staging=True, runtime_code_path_exemption=True, no_identical_regate=True,
                     same_slot_revision={"enabled": True, "max": 1})
    s["gate"]["loss_caps"].update(development=2, selection=2, synthetic_dev=1, synthetic_selection=1)   # ~5% of 50 / 30 rows (N13 rule)
    s["pairing"] = {"mode": "row_store", "refresh_after_accept": True, "sequential_rebuild_on_current_parent": True}
    s["patch_scope"] = {"mode": "primitive_across_modules"}
    s["slots"] = {"max_slots": 6, "max_retries_total": 2}
    return s


def synthetic_tasks(tasks, role):
    return sorted(k for k, v in tasks.items() if v.get("synthetic_role") == role)


def libero90_tasks(tasks):
    return sorted(k for k, v in tasks.items() if v["suite"] == "libero_90")


def ordinary_panel(strategy, tasks):
    spec = strategy["regression_check"]["ordinary_tasks"]
    panel = list(PANEL_11) if spec == "panel_11" else list(spec)
    return panel


def validate_strategy(s, tasks, k, parent):
    """Return a list of violations of the schema and the invariants (empty = valid)."""
    err = []

    def chk(ok, msg):
        if not ok:
            err.append(msg)
    try:
        chk(s.get("iteration") == k, f"iteration must be {k}")
        chk(s.get("parent") == parent, f"parent must be {parent}")
        ts = s["target_selection"]
        chk(ts["mode"] in TARGET_MODES, f"target_selection.mode not in {TARGET_MODES}")
        chk(isinstance(ts["n_groups"], int) and 1 <= ts["n_groups"] <= 8, "n_groups must be 1..8")
        chk(isinstance(ts["max_tasks_per_group"], int) and 1 <= ts["max_tasks_per_group"] <= 5, "max_tasks_per_group must be 1..5")
        inv = ts.get("inventory", {"run": False})
        chk(isinstance(inv.get("run"), bool), "inventory.run must be bool")
        dr = s["debug_data_roles"]
        pv, eo = set(dr["proposer_visible"]), set(dr["evaluator_only"])
        chk(pv <= set(PROPOSER_VISIBLE_ROLES) | {"observe_only_frames"}, f"proposer_visible may only hold {PROPOSER_VISIBLE_ROLES} + observe_only_frames")
        chk(eo <= set(EVALUATOR_ONLY_ROLES), f"evaluator_only may only hold {EVALUATOR_ONLY_ROLES}")
        chk("dev" in pv and "selection" in eo, "dev must be proposer-visible and selection evaluator-only")
        chk(not (pv & eo), "a role cannot be both proposer-visible and evaluator-only")
        chk(("guard" in eo) == bool(dr["guard_panel"]["enabled"]), "guard in evaluator_only iff guard_panel.enabled")
        chk(isinstance(dr.get("proposer_vlm_usd", 0), (int, float)) and dr.get("proposer_vlm_usd", 0) >= 0, "proposer_vlm_usd >= 0")
        for role, seeds in s["seeds"].items():
            chk(role in SEED_POOLS, f"unknown seed role {role}")
            if role in SEED_POOLS:
                chk(bool(seeds) and len(set(seeds)) == len(seeds) and set(seeds) <= set(SEED_POOLS[role]),
                    f"seeds.{role} must be a non-empty subset of its pool {SEED_POOLS[role][0]}-{SEED_POOLS[role][-1]}")
        for role in pv | eo:
            if role != "observe_only_frames":
                chk(role in s["seeds"], f"seeds.{role} missing for an enabled role")
        if inv.get("run"):
            chk("inventory" in s["seeds"], "seeds.inventory missing")
        rc = s["regression_check"]
        panel = ordinary_panel(s, tasks)
        chk(all(t in tasks and tasks[t]["suite"] == "libero_90" for t in panel), "ordinary panel must be libero_90 tasks")
        chk(("ordinary_regression" in eo) == bool(panel), "ordinary_regression role iff a non-empty ordinary panel")
        chk(("synthetic_regression" in eo) == bool(rc.get("hidden_synthetic")), "synthetic_regression role iff hidden_synthetic")
        g = s["gate"]
        chk(set(g["modes"]) <= set(GATE_MODES) and "outcome" in g["modes"], "gate.modes must include outcome, subset of outcome/mechanism")
        chk(isinstance(g["outcome"]["dev_net_min"], int) and g["outcome"]["dev_net_min"] >= 1, "dev_net_min >= 1")
        chk(isinstance(g["outcome"]["selection_net_min"], int) and g["outcome"]["selection_net_min"] >= 1, "selection_net_min >= 1")
        caps = g["loss_caps"]
        for role in GATED_ROLES:
            if role in pv | eo:
                chk(isinstance(caps.get(CAP_KEY[role]), int) and caps[CAP_KEY[role]] >= 0, f"loss_caps.{CAP_KEY[role]} missing")
        chk(isinstance(caps.get("per_task"), int) and caps["per_task"] >= 0, "loss_caps.per_task missing")
        chk(float(g.get("timeout_rate_max", 0.02)) <= TIMEOUT_RATE_MAX, "timeout_rate_max cannot exceed 0.02 (invariant)")
        for key in ("coverage_gate_before_staging", "runtime_code_path_exemption", "no_identical_regate"):
            chk(isinstance(g.get(key), bool), f"gate.{key} must be bool")
        chk(isinstance(g["same_slot_revision"]["enabled"], bool) and int(g["same_slot_revision"].get("max", 1)) in (0, 1, 2), "same_slot_revision")
        chk(s["pairing"]["mode"] in PAIRING_MODES, f"pairing.mode not in {PAIRING_MODES}")
        chk(s["patch_scope"]["mode"] in SCOPE_MODES, f"patch_scope.mode not in {SCOPE_MODES}")
        sl = s["slots"]
        chk(isinstance(sl["max_slots"], int) and 1 <= sl["max_slots"] <= 12, "max_slots 1..12")
        chk(isinstance(sl["max_retries_total"], int) and 0 <= sl["max_retries_total"] <= sl["max_slots"], "max_retries_total 0..max_slots")
    except (KeyError, TypeError) as e:
        err.append(f"schema: missing or malformed field {e!r}")
    return err


def validate_targets(t, s, tasks, k):
    err = []

    def chk(ok, msg):
        if not ok:
            err.append(msg)
    try:
        chk(t.get("iteration") == k, f"targets.iteration must be {k}")
        groups = t["groups"]
        chk(1 <= len(groups) <= s["target_selection"]["n_groups"], "number of groups exceeds n_groups")
        chk(sorted(t["first_pass_order"]) == sorted(groups), "first_pass_order must list every group once")
        seen = []
        for name, g in groups.items():
            chk(re.match(r"^[A-Za-z][A-Za-z0-9]*$", name) is not None, f"group name {name!r} must be alphanumeric")
            chk(1 <= len(g["tasks"]) <= s["target_selection"]["max_tasks_per_group"], f"group {name}: 1..max_tasks_per_group tasks")
            for x in g["tasks"]:
                chk(x in tasks and tasks[x]["suite"] == "libero_90", f"group {name}: {x} is not a libero_90 task")
            seen += g["tasks"]
            chk(isinstance(g.get("failure_classes"), list), f"group {name}: failure_classes list")
            chk(isinstance(g.get("mechanism_allowed"), bool), f"group {name}: mechanism_allowed bool")
            chk(isinstance(g.get("vb_changes_allowed"), bool), f"group {name}: vb_changes_allowed bool")
            chk(all(isinstance(p, list) and len(p) == 2 for p in g.get("pairs", [])), f"group {name}: pairs = [[relation, side], ...]")
            if g.get("mechanism_allowed"):
                chk("mechanism" in s["gate"]["modes"], f"group {name}: mechanism_allowed but the strategy gate has no mechanism mode")
        chk(len(seen) == len(set(seen)), "a task appears in two groups")
    except (KeyError, TypeError) as e:
        err.append(f"schema: missing or malformed field {e!r}")
    return err


def freeze_strategy(rnd: Round, k: int, targets_only=False):
    """Freeze S_k (and targets). Two-step form for inventory-first iterations: freeze strategy.json + STRATEGY_UPDATE.md without
    targets.json (only allowed when S_k runs an inventory), run the inventory, then freeze targets.json (targets_only=True)."""
    d = rnd.iter_dir(k)
    out = d / "strategy_sha256.txt"
    tasks = rnd.manifest
    if targets_only:
        verify_strategy_frozen(rnd, k)
        require(not (d / "targets_sha256.txt").exists() and "targets.json" not in out.read_text(), f"targets for iteration {k} already frozen")
        s, t = read_json(d / "strategy.json"), read_json(d / "targets.json")
        errs = validate_targets(t, s, tasks, k)
        require(not errs, "targets invalid:\n  " + "\n  ".join(errs))
        (d / "targets_sha256.txt").write_text(hash_manifest(rnd.root, [d / "targets.json"]))
        return s, t
    require(not out.exists(), f"strategy for iteration {k} is already frozen")
    st = rnd.state()
    parent = st["snapshots"].get(str(k - 1))
    require(parent is not None, f"iteration {k - 1} has no snapshot")
    require(k == 1 or (rnd.iter_dir(k - 1) / "DONE").exists(), f"iteration {k - 1} is not DONE")
    require((d / "STRATEGY_UPDATE.md").is_file(), "STRATEGY_UPDATE.md missing")
    s = read_json(d / "strategy.json")
    errs = validate_strategy(s, tasks, k, parent)
    has_targets = (d / "targets.json").exists()
    t = read_json(d / "targets.json") if has_targets else None
    if has_targets:
        errs += validate_targets(t, s, tasks, k)
    else:
        require(s.get("target_selection", {}).get("inventory", {}).get("run") is True, "targets.json missing (allowed only when S_k runs an inventory first)")
    require(not errs, "strategy/targets invalid:\n  " + "\n  ".join(errs))
    files = [d / "strategy.json", d / "STRATEGY_UPDATE.md"] + ([d / "targets.json"] if has_targets else [])
    out.write_text(hash_manifest(rnd.root, files))
    st["iteration"] = k; st["slots"].setdefault(str(k), [])
    rnd.save_state(st)
    return s, t


def verify_strategy_frozen(rnd: Round, k: int):
    f = rnd.iter_dir(k) / "strategy_sha256.txt"
    require(f.exists(), f"strategy for iteration {k} is not frozen (run: rsi.py freeze --iter {k})")
    verify_hash_manifest(rnd.root, f)


# ----------------------------------------------------------------------------- plans
def guard_tasks(strategy, state, group_tasks, panel):
    spec = strategy["debug_data_roles"]["guard_panel"]
    if not spec["enabled"]:
        return []
    if spec["tasks"] == "accepted_tasks_not_in_group":
        pool = []
        for a in state["accepted"]:
            pool += [x for x in a["tasks"] if x not in pool]
    else:
        pool = list(spec["tasks"])
    return [x for x in pool if x not in group_tasks and x not in panel]


def arm_rows(arm, strategy, targets, group, tasks, state):
    """The rows one arm runs for one group under strategy S_k (role-labelled)."""
    g = targets["groups"][group]; gt = list(g["tasks"]); sd = strategy["seeds"]
    roles = set(strategy["debug_data_roles"]["proposer_visible"]) | set(strategy["debug_data_roles"]["evaluator_only"])
    panel = [x for x in ordinary_panel(strategy, tasks) if x not in gt]
    spec = [(gt, "dev"), (gt, "selection")]
    if "synthetic_dev" in roles:
        spec.append((synthetic_tasks(tasks, "synthetic_dev"), "synthetic_dev"))
    if "synthetic_selection" in roles:
        spec.append((synthetic_tasks(tasks, "synthetic_dev"), "synthetic_selection"))
    if "guard" in roles:
        spec.append((guard_tasks(strategy, state, gt, panel), "guard"))
    if "ordinary_regression" in roles:
        spec.append((panel, "ordinary_regression"))
    if "synthetic_regression" in roles:
        spec.append((synthetic_tasks(tasks, "synthetic_regression"), "synthetic_regression"))
    rows = [dict(arm=arm, task_short=x, seed=int(s), role=role) for ts, role in spec for x in ts for s in sd[role]]
    keys = [(r["task_short"], r["seed"]) for r in rows]
    require(len(keys) == len(set(keys)), f"duplicate (task, seed) rows for arm {arm} group {group}")
    assert_rows_source_only(rows, tasks)
    return rows


def inventory_rows(arm, strategy, tasks):
    return [dict(arm=arm, task_short=x, seed=int(s), role="inventory") for x in libero90_tasks(tasks)
            for s in strategy["target_selection"]["inventory"]["seeds"]]


def write_plan(rnd: Round, phase, meta):
    path = rnd.p(f"plan_{phase}.txt")
    require(not path.exists() and not rnd.p(f"plan_{phase}_roles.json").exists(), f"plan {phase} exists")
    rows = meta["rows"]
    require(rows, f"plan {phase} has no rows")
    for a in sorted({r["arm"] for r in rows}):
        require(rnd.program(a).is_file(), f"missing program {a}")
    path.write_text("".join(f"{r['arm']} {r['task_short']} {r['seed']}\n" for r in rows))
    write_json(rnd.p(f"plan_{phase}_roles.json"), meta, once=True)
    return path


# ----------------------------------------------------------------------------- analysis
STDOUT_KEYS = ("EPISODE_TIMEOUT ", "GRASP_ABORT reason=")
TAG_LINE = re.compile(r"^[A-Z][A-Z0-9_]+ \{")   # any tagged JSON line the program prints (BINDING, GRASP_VERIFY, LIFT_MOTION, ...)
MAX_KEY_LINES = 60
PRE_MOTION_MAX_STEPS = 50   # an episode that stops before any motion ends within a few environment steps


def stdout_text(rnd: Round, episode_dir):
    return "\n".join(p.read_text(errors="replace") for p in sorted(rnd.p(episode_dir).glob("artifacts/*/stdout.txt")))


def key_lines(txt):
    return [l[:400] for l in txt.splitlines() if l.startswith(STDOUT_KEYS) or TAG_LINE.match(l)][:MAX_KEY_LINES]


def first_stop(txt):
    for line in txt.splitlines():
        head = line.split(" ", 1)
        if len(head) == 2 and head[0].endswith("_STOP") and head[0].isupper():
            try:
                js = json.loads(head[1])
            except json.JSONDecodeError:
                js = {}
            return dict(tag=head[0], status=js.get("status") or js.get("why"), relation=js.get("relation"), side=js.get("side"))
    return None


def summarize(rows):
    """{task: {seed: row}} -> per-task summary used by attribution, gates and the outer loop's aggregates."""
    out = {}
    for t, v in sorted(rows.items()):
        out[t] = {"official": f"{sum(x['official'] for x in v.values())}/{len(v)}",
                  "by_seed": {str(s): {k: x.get(k) for k in ("official", "first_failure", "state_idx", "stop")} for s, x in sorted(v.items())},
                  "first_failures": dict(Counter(x["first_failure"] for x in v.values() if not x["official"]))}
    return out


def analyze_phase(rnd: Round, phase, posthoc_csv=None):
    """Integrity + per-role summaries of one stage. Writes results/<phase>/ and returns the integrity dict."""
    plan = rnd.plan_roles(phase)
    wanted = {(r["arm"], r["task_short"], int(r["seed"])): r["role"] for r in plan["rows"]}
    require(len(wanted) == len(plan["rows"]), "duplicate plan rows")
    expected_sha = {a: digest(rnd.program(a)) for a in {r["arm"] for r in plan["rows"]}}
    posthoc_csv = Path(posthoc_csv or rnd.p("results/posthoc/episodes.csv"))
    entries = [x for x in csv.DictReader(posthoc_csv.open()) if x["phase"] == phase]
    errors, rows, seen, timeouts = [], [], set(), 0
    adapters = {n: digest(rnd.p("code", n)) for n in ("adapter_v3.py", "adapter_lexicon_v3.json")}
    for x in entries:
        if x.get("resolved") != "True" or not x.get("episode_dir"):
            errors.append(f"unresolved: {x.get('task')} {x.get('seed')}"); continue
        d = rnd.p(x["episode_dir"]); m = read_json(d / "attempt_manifest.json")
        key = (m["arm"], m["task_short"], int(m["seed"]))
        if key not in wanted or key in seen:
            errors.append(f"duplicate or unplanned: {key}"); continue
        seen.add(key)
        timeout = m.get("failure_class") == "episode_timeout"; timeouts += int(timeout)
        txt = stdout_text(rnd, x["episode_dir"]); stop = first_stop(txt)
        diag_ok = all((d / "diag" / n).is_file() and (d / "diag" / n).stat().st_size > 0 for n in ("events.jsonl", "recorder_meta.json"))
        moved = (d / "diag/samples.npz").is_file() and (d / "diag/samples.npz").stat().st_size > 0
        # zero-motion episodes are valid only as a recorded stop before any motion (e.g. language_unparsed)
        pre_motion_stop = (not moved and int(m.get("n_samples", 0)) == 0 and stop is not None
                           and int(m.get("final_sim_step", 10 ** 6)) <= PRE_MOTION_MAX_STEPS and not m["official_task_completed"])
        checks = {
            "policy": m["policy_sha256"] == expected_sha[m["arm"]] == digest(d / "policy.py"),
            "adapter": all(m["adapter_files_sha256"][n] == adapters[n] for n in adapters),
            "descriptor": m["descriptor_sha256"] == hashlib.sha256(json.dumps(m["task_descriptors"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "identity": m["phase"] == phase,
            "outcome": int(x.get("official") or 0) == int(m["official_task_completed"]),
            "state_index": m["state_idx"] == (int(m["seed"]) - 1) % m["n_init_states"],
            "runtime": m["sandbox_rc"] == 0 and x.get("crashed_traceback") == "False" and not m.get("recorder_errors"),
            "trajectory": diag_ok and (moved or pre_motion_stop),
            "stdout": len(list(d.glob("artifacts/*/stdout.txt"))) == 1,
            "horizon": not timeout or (m["truncated"] and m["final_sim_step"] == HORIZON and m["official_task_completed"] == 0),
            "interface": m.get("interface", "rsi") == plan.get("interface", "rsi"),
        }
        errors += [f"{label}: {key}" for label, ok in checks.items() if not ok]
        ff = "episode_timeout" if timeout else x.get("first_failure_truth_corrected", "")
        if not int(m["official_task_completed"]) and ff in ("", "none_logged") and stop is not None:
            ff = f"stop:{stop.get('status') or stop['tag'].lower()}"   # the program's own stop reason, e.g. stop:language_unparsed
        rows.append(dict(arm=m["arm"], task_short=m["task_short"], seed=int(m["seed"]), role=wanted[key],
                         official=int(m["official_task_completed"]), state_idx=m["state_idx"], first_failure=ff,
                         stop=stop, key_lines=key_lines(txt), episode_dir=x["episode_dir"], pre_motion_stop=pre_motion_stop))
    if seen != set(wanted):
        errors.append(f"missing planned episodes: {sorted(set(wanted) - seen)[:10]}")
    ledger = rnd.p(f"attempts_log_{phase}.csv")
    led = list(csv.DictReader(ledger.open())) if ledger.exists() else []
    if len(led) != len(wanted) or any(x["rc"] != "0" for x in led):
        errors.append("runner ledger count or return code mismatch")
    fz = rnd.p(f"frozen_sha256_{phase}.txt")
    try:
        verify_hash_manifest(rnd.root, fz)
    except (HarnessError, OSError) as e:
        errors.append(str(e))
    limit = float(plan.get("timeout_rate_max", TIMEOUT_RATE_MAX))
    if timeouts > limit * len(wanted):
        errors.append(f"timeout rate {timeouts}/{len(wanted)} exceeds {limit:.0%}")
    out = rnd.results(phase); out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        by[r["arm"]][r["role"]][r["task_short"]][r["seed"]] = r
    visible = {a: {role: summarize(by[a][role]) for role in PROPOSER_VISIBLE_ROLES if by[a][role]} for a in by}
    hidden = {a: {role: summarize(by[a][role]) for role in EVALUATOR_ONLY_ROLES if by[a][role]} for a in by}
    def total(block):
        vals = [x["official"] for v in block.values() for x in v.values()]
        return f"{sum(vals)}/{len(vals)}"
    totals = {a: {role: total(by[a][role]) for role in PROPOSER_VISIBLE_ROLES if by[a][role]} for a in by}
    write_json(out / "analysis.json", {"phase": phase, "kind": plan["kind"], "group": plan.get("group"), "proposer_visible": visible, "totals": totals})
    lines = [f"# Development attribution: {phase}", "", "Proposer-visible rows only (development, synthetic development, inventory).", ""]
    for r in sorted(rows, key=lambda r: (r["arm"], r["task_short"], r["seed"])):
        if r["role"] in PROPOSER_VISIBLE_ROLES:
            lines += [f"## {r['arm']} {r['task_short']} seed {r['seed']}: official={r['official']} ({r['role']})",
                      f"episode: {r['episode_dir']}", r["first_failure"]] + r["key_lines"] + [""]
    (out / "attribution.md").write_text("\n".join(lines) + "\n")
    eo = out / "evaluator_only"; eo.mkdir(exist_ok=True)
    write_json(eo / "selection.json", hidden)
    (eo / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    integrity = {"phase": phase, "planned": len(wanted), "resolved": len(seen), "timeout_count": timeouts,
                 "timeout_rate": timeouts / max(1, len(wanted)), "errors": errors, "eligible": not errors}
    write_json(out / "integrity.json", integrity)
    return integrity


def flag_infra(rnd: Round, phase):
    """Episodes whose VLM calls hit a transport error (proxy status=error). Selection by flag only, never by outcome."""
    plan = rnd.plan_roles(phase)
    role = {(r["arm"], r["task_short"], int(r["seed"])): r["role"] for r in plan["rows"]}
    bad, n_calls, n_err = [], 0, 0
    for f in glob.glob(str(rnd.p(f"stage_{phase}/attempts/*/*/*/seed_*/diag/vlm_calls.jsonl"))):
        parts = Path(f).parts; arm, task, seed = parts[-5], parts[-4], int(parts[-3].split("_")[1])
        recs = read_jsonl(f); errs = [r for r in recs if r.get("status") == "error"]
        n_calls += len(recs); n_err += len(errs)
        if errs:
            bad.append({"arm": arm, "task_short": task, "seed": seed, "role": role.get((arm, task, seed)), "errors": len(errs), "calls": len(recs)})
    out = {"phase": phase, "episodes": sorted(bad, key=lambda b: (b["arm"], b["task_short"], b["seed"])), "vlm_calls": n_calls, "vlm_errors": n_err}
    write_json(rnd.results(phase) / "vlm_infra_episodes.json", out)
    return out


# ----------------------------------------------------------------------------- effective rows and row store
def phase_rows(rnd: Round, phase):
    """Rows of an eligible phase, with infrastructure-flagged rows replaced by the phase's single infra rerun."""
    d = rnd.results(phase)
    require(read_json(d / "integrity.json")["eligible"] is True, f"phase {phase} is not eligible")
    rows = {(r["arm"], r["task_short"], r["seed"]): r for r in read_jsonl(d / "evaluator_only/rows.jsonl")}
    infra = d / "vlm_infra_episodes.json"
    flagged = [(b["arm"], b["task_short"], b["seed"]) for b in read_json(infra)["episodes"]] if infra.exists() else []
    if flagged:
        rd = rnd.results(phase + "_infra1")
        require(rd.exists(), f"phase {phase} has {len(flagged)} infrastructure-flagged rows: run the infra rerun first")
        require(read_json(rd / "integrity.json")["eligible"] is True, f"phase {phase}_infra1 is not eligible")
        rerun = {(r["arm"], r["task_short"], r["seed"]): r for r in read_jsonl(rd / "evaluator_only/rows.jsonl")}
        require(set(rerun) == set(flagged), f"{phase}_infra1 rows differ from the flagged rows")
        for k, r in rerun.items():
            rows[k] = dict(r, role=rows[k]["role"], infra_rerun=True)
    return rows


def all_phases(rnd: Round):
    """Analyzed phases (infra reruns excluded: they enter through their base phase), oldest first."""
    ds = [p for p in rnd.p("results").iterdir() if p.is_dir() and p.name != "posthoc" and not p.name.endswith("_infra1")
          and (p / "integrity.json").exists()]
    return [p.name for p in sorted(ds, key=lambda p: ((p / "integrity.json").stat().st_mtime, p.name))]


def row_store(rnd: Round, arm, strict=True):
    """Union of the arm's rows over every eligible phase.
    strict (row_store pairing): a (task, seed) row is never re-run for the same arm, so a second, different outcome is an error.
    non-strict (briefs; full_rerun pairing re-runs the parent on purpose): the latest phase wins."""
    store = {}
    for phase in all_phases(rnd):
        if not read_json(rnd.results(phase) / "integrity.json")["eligible"]:
            continue
        try:
            rows = phase_rows(rnd, phase)
        except HarnessError:
            continue   # pending infra rerun: not part of any store yet
        for (a, t, s), r in rows.items():
            if a != arm:
                continue
            if strict:
                require((t, s) not in store or store[(t, s)]["official"] == r["official"], f"row re-run for arm {arm}: {(t, s)}")
                store.setdefault((t, s), dict(r, phase=phase))
            else:
                store[(t, s)] = dict(r, phase=phase)
    return store


def missing_rows(rnd: Round, arm, rows):
    store = row_store(rnd, arm, strict=False)
    return [r for r in rows if (r["task_short"], r["seed"]) not in store]


# ----------------------------------------------------------------------------- gate
def compare(parent, cand, cap, per_task_cap, exempt=frozenset()):
    """parent/cand: {task: {seed: row}} with identical keys. Gains never cancel losses."""
    require(set(parent) == set(cand), "parent/candidate task sets differ")
    lost, gained, exempted, by_task = [], [], [], {}
    for t in parent:
        require(set(parent[t]) == set(cand[t]), f"parent/candidate seeds differ for {t}")
        for s in parent[t]:
            require(parent[t][s]["state_idx"] == cand[t][s]["state_idx"], f"state_idx mismatch {t} {s}")
        loss = [s for s in parent[t] if parent[t][s]["official"] and not cand[t][s]["official"]]
        gain = [s for s in parent[t] if not parent[t][s]["official"] and cand[t][s]["official"]]
        counted = [s for s in loss if (t, s) not in exempt]
        exempted += [(t, s) for s in loss if (t, s) in exempt]
        lost += [(t, s) for s in counted]; gained += [(t, s) for s in gain]
        by_task[t] = {"lost": counted, "lost_exempt": [s for s in loss if (t, s) in exempt], "gained": gain}
    ok = len(lost) <= cap and all(len(v["lost"]) <= per_task_cap for v in by_task.values())
    return {"ok": ok, "lost": lost, "gained": gained, "exempt": exempted, "by_task": by_task, "cap": cap, "per_task_cap": per_task_cap}


def blocks(rows, arm):
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["arm"] == arm:
            out[r["role"]][r["task_short"]][r["seed"]] = r
    return out


def class_count(block, tasks, classes):
    return sum(1 for t in tasks for x in block.get(t, {}).values() if not x["official"] and any(str(x["first_failure"]).startswith(c) for c in classes))


def has_marker(rnd: Round, row, markers):
    if not markers:
        return None
    txt = stdout_text(rnd, row["episode_dir"])
    return any(m in txt for m in markers) if txt else None


def evaluate_gate(rnd: Round, phase):
    plan = rnd.plan_roles(phase)
    require(plan["kind"] == "candidate", f"{phase} is not a candidate stage")
    k = plan["iteration"]; s = rnd.strategy(k); t = rnd.targets(k)
    decl = read_json(rnd.p("proposals", phase, "predeclared.json"))
    arm, parent, group = plan["arm"], plan["parent"], plan["group"]
    require(decl["arm"] == arm and decl["parent"] == parent and decl["group"] == group, "predeclared.json does not match the plan")
    g = t["groups"][group]; gt = g["tasks"]; gs = s["gate"]
    rows = phase_rows(rnd, phase)
    crow = [r for r in rows.values() if r["arm"] == arm]
    if s["pairing"]["mode"] == "full_rerun":
        prow = [r for r in rows.values() if r["arm"] == parent]
    else:
        store = row_store(rnd, parent)
        miss = [(r["task_short"], r["seed"]) for r in crow if (r["task_short"], r["seed"]) not in store]
        require(not miss, f"parent store incomplete for {len(miss)} rows: {miss[:5]}")
        prow = [dict(store[(r["task_short"], r["seed"])], role=r["role"], arm=parent) for r in crow]
    cb, pb = blocks(crow, arm), blocks(prow, parent)
    markers = []
    if gs["runtime_code_path_exemption"]:
        markers = list(read_json(rnd.p("proposals", phase, "build.json")).get("runtime_markers") or [])
    exempt, evidence = set(), {}
    if markers:
        for role in GATED_ROLES:
            for task, seeds in pb.get(role, {}).items():
                for sd, pr in seeds.items():
                    cr = cb[role][task][sd]
                    if pr["official"] and not cr["official"]:
                        hit = has_marker(rnd, cr, markers)
                        evidence[f"{task}@{sd}"] = {"role": role, "marker_in_episode": hit}
                        if hit is False:
                            exempt.add((task, sd))
    caps = gs["loss_caps"]; res = {}
    for role in GATED_ROLES:
        if role in cb or role in pb:
            res[role] = compare(pb.get(role, {}), cb.get(role, {}), caps[CAP_KEY[role]], caps["per_task"], frozenset(exempt))

    def net(role):
        r = res.get(role)
        return sum(len(r["by_task"][x]["gained"]) - len(r["by_task"][x]["lost"]) - len(r["by_task"][x]["lost_exempt"]) for x in gt if x in r["by_task"]) if r else 0
    dev_net, sel_net = net("dev"), net("selection")
    outcome_ok = dev_net >= gs["outcome"]["dev_net_min"] and sel_net >= gs["outcome"]["selection_net_min"]
    mech, mechanism_ok = None, False
    classes = decl.get("failure_classes") or []
    if "mechanism" in gs["modes"] and g["mechanism_allowed"] and decl.get("mode") == "mechanism" and classes:
        mech = {"classes": classes, "dev_before": class_count(pb["dev"], gt, classes), "dev_after": class_count(cb["dev"], gt, classes),
                "sel_before": class_count(pb["selection"], gt, classes), "sel_after": class_count(cb["selection"], gt, classes)}
        mechanism_ok = mech["dev_before"] > 0 and mech["dev_after"] == 0 and mech["sel_after"] < mech["sel_before"] and sel_net >= 0
    coverage_ok = True
    if gs["coverage_gate_before_staging"]:
        cg = rnd.p("proposals", phase, "coverage_gate.json")
        coverage_ok = cg.exists() and read_json(cg).get("ok") is True
    losses_ok = all(r["ok"] for r in res.values())
    report = {"phase": phase, "arm": arm, "parent": parent, "group": group, "tasks": gt, "pairing": s["pairing"]["mode"],
              "coverage_ok": coverage_ok, "losses_ok": losses_ok, "group_dev_net": dev_net, "group_selection_net": sel_net,
              "outcome_ok": outcome_ok, "mechanism": mech, "mechanism_ok": mechanism_ok,
              "runtime_markers": markers, "exemption_evidence": evidence, "roles": res,
              "accepted_by": "outcome" if outcome_ok else ("mechanism" if mechanism_ok else None),
              "passes_numeric_gates": bool(coverage_ok and losses_ok and (outcome_ok or mechanism_ok)),
              "requires_evaluator_review": "genuine completions; mechanism: movement to a later physical stage; new first failures on "
                                           "previously passing rows; tests and hashes; proposer claims vs analysis. Never feed "
                                           "evaluator-only details to a proposer."}
    write_json(rnd.results(phase) / "gate.json", report, once=True)
    return report


def gate_aggregates(report):
    """What the outer loop may read about evaluator-only roles: counts only."""
    return {role: {"gained": len(r["gained"]), "lost": len(r["lost"]), "exempt": len(r["exempt"]), "cap": r["cap"], "ok": r["ok"]}
            for role, r in report["roles"].items()}


# ----------------------------------------------------------------------------- coverage gate (offline)
def load_policy_ns(path):
    src = re.sub(r"^run\(\)\s*$", "", Path(path).read_text(), flags=re.M)
    ns = {"__name__": "policy_offline"}
    exec(compile(src, str(path), "exec"), ns)   # noqa: S102 - frozen program, definitions only (run() removed)
    return ns


def coverage(policy_path, tasks):
    ns = load_policy_ns(policy_path)
    if "la_parse" not in ns or "la_dispatch" not in ns:
        return None, None
    rows = {}
    for short, t in sorted(tasks.items()):
        if t["suite"] != "libero_90":
            continue
        td = t["descriptors"]; ns["task_descriptors"] = lambda td=td: copy.deepcopy(td)
        subgoals, mode = ns["la_parse"](td, t["language"])
        if not subgoals and td.get("subgoal_regex") and "la_parse_old_form" in ns:
            subgoals = ns["la_parse_old_form"](td, t["language"]); mode = "old_form_fallback" if subgoals else mode
        status = "dispatch_ok" if subgoals else "language_unparsed"
        for sg in subgoals:
            ok, why = ns["la_dispatch"](sg)
            if not ok:
                status = "language_unparsed" if why == "vocabulary" else "relation_unsupported"; break
        summ = ns.get("la_summary", repr)
        rows[short] = {"mode": mode, "status": status, "subgoals": [summ(x) for x in subgoals]}
    supported = {r: sorted(map(str, v)) for r, v in ns.get("LA_SUPPORTED", {}).items()}
    return rows, supported


def coverage_gate(rnd: Round, phase):
    plan_k, arm, parent, group = _cand_identity(rnd, phase)
    t = rnd.targets(plan_k); g = t["groups"][group]; tasks = rnd.manifest
    prow, psup = coverage(rnd.program(parent), tasks)
    crow, csup = coverage(rnd.program(arm), tasks)
    if crow is None:
        rep = {"group": group, "applicable": False, "ok": False, "why": "candidate has no la_parse/la_dispatch language adapter"}
    else:
        allowed = {(r, str(x)) for r, x in g.get("pairs", [])}
        pset = {(r, x) for r, v in (psup or {}).items() for x in v}; cset = {(r, x) for r, v in csup.items() for x in v}
        added, removed = cset - pset, pset - cset
        not_disp = [x for x in g["tasks"] if crow[x]["status"] != "dispatch_ok"]
        changed = [x for x in crow if x not in g["tasks"] and prow is not None and prow[x] != crow[x]]
        rep = {"group": group, "applicable": True, "parent_has_language_adapter": prow is not None,
               "added_pairs": sorted(map(list, added)), "removed_pairs": sorted(map(list, removed)),
               "unexpected_pairs": sorted(map(list, added - allowed)), "group_not_dispatching": not_disp,
               "other_instructions_changed": changed,
               "parent_dispatch_ok": sum(r["status"] == "dispatch_ok" for r in prow.values()) if prow else None,
               "candidate_dispatch_ok": sum(r["status"] == "dispatch_ok" for r in crow.values()), "n": len(crow)}
        rep["ok"] = not (removed or (added - allowed) or not_disp or changed)
    write_json(rnd.p("proposals", phase, "coverage_gate.json"), rep, once=True)
    return rep


# ----------------------------------------------------------------------------- build (proposal -> candidate program)
def load_apply_module(rnd: Round):
    spec = importlib.util.spec_from_file_location("apply_proposal_90", rnd.p("code/apply_proposal_90.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def set_marker(src, arm):
    require(len(MARK_RE.findall(src)) == 1, "program marker not found exactly once")
    return MARK_RE.sub(f"'arm_90': '{arm}',", src, count=1)


def add_marker(src, arm="N0"):
    """Start program -> N0: set its arm marker, or add one after the S1 anchor if it has none (S1, K2)."""
    if MARK_RE.search(src):
        return set_marker(src, arm)
    require(src.count(MARK_ANCHOR) == 1, "S1 marker anchor not found exactly once")
    return src.replace(MARK_ANCHOR, MARK_ANCHOR + f" 'arm_90': '{arm}',", 1)


def normalized_sha(src):
    return hashlib.sha256(MARK_RE.sub("'arm_90': '*',", src).encode()).hexdigest()


def apply_proposal(parent_src, proposal_path, scope_mode, forbidden, modules, arm, interface="rsi"):
    spec = importlib.util.spec_from_file_location("proposal", proposal_path)
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    touched = list(getattr(P, "TOUCHED_MODULES", [getattr(P, "MODULE", None)]))
    require(touched and all(m in modules for m in touched), f"TOUCHED_MODULES must be from {modules}")
    if scope_mode == "one_module":
        require(len(touched) == 1 and getattr(P, "MODULE", touched[0]) == touched[0], "patch scope one_module: exactly one module")
    require(isinstance(getattr(P, "FAILURE_CLASS", None), str) and P.FAILURE_CLASS, "FAILURE_CLASS missing")
    patches, inserts = list(getattr(P, "PATCHES", [])), list(getattr(P, "INSERTS", []))
    require(patches or inserts, "proposal has no PATCHES/INSERTS")
    new_text = "".join(n for _, n in patches) + "".join(x for _, x in inserts)
    words = [w for w in forbidden if not (interface == "original" and w in SPATIAL_WORDS)]
    bad = sorted({w for w in words if re.search(r"(?<![a-z_])" + w + r"(?![a-z_])", new_text.lower())})
    require(not bad, f"forbidden literal(s) in patch: {bad}")
    api = sorted({pat for pat in FORBIDDEN_API if re.search(pat, new_text)})
    require(not api, f"forbidden simulator-truth API in patch: {api}")
    require("arm_90" not in new_text, "a patch may not touch the program marker")
    prog = set_marker(parent_src, arm)
    for i, (anchor, text) in enumerate(inserts):
        require(prog.count(anchor) == 1, f"insert {i}: anchor found {prog.count(anchor)} times"); prog = prog.replace(anchor, text + anchor)
    for i, (old, new) in enumerate(patches):
        require(prog.count(old) == 1, f"patch {i}: anchor found {prog.count(old)} times"); prog = prog.replace(old, new)
    return prog, {"failure_class": P.FAILURE_CLASS, "touched_modules": touched, "runtime_markers": list(getattr(P, "RUNTIME_MARKERS", []))}


def cand_arm(phase):
    m = CAND_PHASE_RE.match(phase)
    require(m is not None, f"candidate phase must be chunk<k>_iter<j>[_rev<n>]: {phase}")
    k, j, rev = int(m.group(1)), int(m.group(2)), m.group(4)
    return k, j, (int(rev) if rev else 0), f"C{k}I{j}" + (f"r{rev}" if rev else "")


def _cand_identity(rnd, phase):
    b = read_json(rnd.p("proposals", phase, "build.json"))
    return b["iteration"], b["arm"], b["parent"], b["group"]


def slot_check(state, strategy, k, j, rev, group):
    slots = state["slots"].get(str(k), [])
    sl = strategy["slots"]; ssr = strategy["gate"]["same_slot_revision"]
    require(1 <= j <= sl["max_slots"], f"slot {j} exceeds max_slots {sl['max_slots']}")
    if rev:
        require(ssr["enabled"] and rev <= int(ssr.get("max", 1)), "same-slot revision not allowed by S_k")
        require(any(x["j"] == j and x["rev"] == rev - 1 for x in slots), f"revision {rev} needs slot {j} rev {rev - 1}")
        require(all(x["j"] != j or x["group"] == group for x in slots), "a revision keeps its slot's group")
    else:
        require(all(x["j"] != j for x in slots), f"slot {j} already used")
    retries = sum(1 for x in slots if x["retry"])
    is_retry = bool(rev) or any(x["group"] == group for x in slots)
    require(not is_retry or retries < sl["max_retries_total"], "max_retries_total reached")
    return is_retry


# ----------------------------------------------------------------------------- runner patch (generic arm registry)
RUNNER_PATCHES = [
    ("arms", re.compile(r"^ARMS90 = tuple\(.*$", re.M),
     'import re as _rsi_re\n'
     '_RSI_ARM = _rsi_re.compile(r"^(N\\d+|C\\d+I\\d+(r\\d+)?)$")   # generic harness: any snapshot N<k> or candidate C<k>I<j>[r<n>]\n\n\n'
     'class _RSIArmMap:\n'
     '    def __init__(self, f):\n        self.f = f\n\n'
     '    def __contains__(self, a):\n        return isinstance(a, str) and bool(_RSI_ARM.match(a))\n\n'
     '    def __getitem__(self, a):\n        if a not in self:\n            raise KeyError(a)\n        return self.f(a)\n\n'
     '    def get(self, a, default=None):\n        return self.f(a) if a in self else default\n\n\n'
     'def _rsi_arm(a):\n    if not _RSI_ARM.match(a):\n        raise ValueError(f"not an RSI arm: {a}")\n    return a\n\n\n'
     'ARMS90 = _RSIArmMap(lambda a: a)'),
    ("vlm_arms", re.compile(r"^VLM_ARMS = .*$", re.M),
     'VLM_ARMS = ARMS90   # generic harness: every arm may call query_vlm (loopback proxy); a program without a VB module never calls it'),
    ("arm_base", re.compile(r'^ARM_BASE = \{a: "MGG" for a in ARMS90\}$', re.M), 'ARM_BASE = _RSIArmMap(lambda a: "MGG")'),
    ("marker", re.compile(r"^PROGRAM_MARKER = \{a: f\"'arm_k': 'K2', 'arm_90': '\{a\}',\" for a in ARMS90\}$", re.M),
     "PROGRAM_MARKER = _RSIArmMap(lambda a: f\"'arm_90': '{a}',\")   # generic harness: start programs S1/K2/N20 differ before the arm marker"),
    ("argparse", re.compile(r'choices=list\(ARM_BASE\)'), 'type=_rsi_arm'),
]


RUNNER_PATCHES += [
    ("interface", re.compile(r"^        api\.functions = patched$", re.M),
     '        if os.environ.get("RSI_INTERFACE", "rsi") != "original":   # generic harness: original interface = reduced API only\n'
     '            api.functions = patched'),
    ("manifest_interface", re.compile(r'manifest\.update\(harness_revision="n8_clean_horizon_v1", '),
     'manifest.update(interface=os.environ.get("RSI_INTERFACE", "rsi"), harness_revision="n8_clean_horizon_v1", '),
]


def patch_runner(src):
    for name, pat, new in RUNNER_PATCHES:
        n = len(pat.findall(src))
        require(n == 1, f"runner patch {name}: anchor found {n} times")
        src = pat.sub(lambda _m, new=new: new, src, count=1)
    return "# Generic RSI harness: derived from the reference iso_runner_90.py by harness/rsi_lib.patch_runner (arm registry only).\n" + src
