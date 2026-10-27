# A0: clean RSI start program (RSI-zeroshot-transfer-debugging, written 2026-10-27).
#
# Provenance
#   - Written from the original ASPIRE reduced API only (the reduced skill-library API and its api-reference.md). No episode
#     was run while writing it, no constant is tuned to any outcome, and no long-horizon evaluation task, state or result was
#     used. It is not derived from S1, K0-K2 or N1-N21.
#   - It runs with the ORIGINAL interface (authorization "interface": "original"): only the functions the reduced skill-library
#     API binds into the program namespace, plus env.handle.task_language. No harness tools (no motion interface, verifiers,
#     binding service, task descriptors, VLM).
#   - Constants below are unit-geometry priors (centimetre clearances), documented where they are set; RSI may change them.
#
# Scope: one shared, task-agnostic driver. It parses "put/place/stack the A on|in the B" and "pick up the A and put it on|in
# the B" clauses from the instruction, binds A and B with SAM3 text prompts taken from the instruction itself, grasps A top-down
# across its narrow horizontal axis, carries it above B and releases it. Every other instruction form is reported as
# unsupported (LANG_PARSE / BINDING_STOP lines) instead of being guessed.
#
# Stdout tags (read by the posthoc attribution): LANG_PARSE, BINDING, BINDING_STOP, GRASP_MOTION, GRASP_VERIFY, LIFT_MOTION,
# PLACE_RESULT, SUBGOAL_STATUS, TASK_STOP. The program never raises: exceptions are reported as a TASK_STOP line.
import json
import re

import numpy as np
from scipy.spatial.transform import Rotation

PROGRAM_INFO = {'arm_90': 'A0', 'origin': 'aspire_original_api', 'interface': 'original'}

# ============================================================================ module: motion_preconditions
APPROACH_HEIGHT = 0.12       # m above the grasp / release point before descending (clears a held object of ~10 cm)
LIFT_CLEARANCE = 0.10        # m the held object's bottom is carried above the support top
DESCENT_STEPS = 2            # the descent is split into this many IK targets (move_to_joints interpolates in joint space)


def log(tag, **data):
    print(tag + " " + json.dumps(data, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)


def quat_top_down(yaw):
    """wxyz quaternion of a top-down hand (approach = world -z) rotated by yaw about world z."""
    r = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


def move_tcp(position, quat, label):
    """One IK target + blocking joint move. Returns True when IK succeeded (arrival is not verified in A0)."""
    try:
        joints = solve_ik(np.asarray(position, float), np.asarray(quat, float))
    except Exception as e:  # noqa: BLE001 - IK failure is a reported outcome, not a crash
        log("GRASP_MOTION", approach=label, descent="ik_failed", error=repr(e)[:160])
        return False
    move_to_joints(joints)
    return True


def move_line(start, end, quat, label, steps):
    for i in range(1, steps + 1):
        if not move_tcp(start + (end - start) * i / steps, quat, f"{label}_{i}"):
            return False
    return True


# ============================================================================ module: binding
MIN_MASK_PIXELS = 40         # smaller SAM3 masks are treated as noise
MIN_POINTS = 30              # fewer valid depth points cannot give a reliable geometry
SAME_INSTANCE_IOU = 0.5      # object and support masks overlapping more than this are the same instance


def observe():
    obs = get_observation()
    cam = obs["agentview"]
    return obs, cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def bind(prompt, view, exclude_mask=None):
    """Best SAM3 instance for a text prompt, as world points + simple geometry; None if not found."""
    _, rgb, depth, K, E = view
    try:
        segs = segment_sam3_text_prompt(rgb, prompt)
    except Exception as e:  # noqa: BLE001
        log("BINDING", status="segmentation_error", prompt=prompt, error=repr(e)[:160])
        return None
    best = None
    for s in sorted(segs or [], key=lambda d: -float(d.get("score", 0.0))):
        mask = np.asarray(s["mask"], bool)
        if mask.sum() < MIN_MASK_PIXELS or (exclude_mask is not None and iou(mask, exclude_mask) > SAME_INSTANCE_IOU):
            continue
        pts = np.asarray(mask_to_world_points(mask, depth, K, E), float)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < MIN_POINTS:
            continue
        best = {"prompt": prompt, "mask": mask, "points": pts, "score": float(s.get("score", 0.0)),
                "center": np.median(pts, axis=0), "top_z": float(np.percentile(pts[:, 2], 95)),
                "bottom_z": float(np.percentile(pts[:, 2], 5))}
        break
    log("BINDING", status="bound" if best else "not_found", support=None, rel_margin=None, prompt=prompt,
        score=best and best["score"], n_points=best and len(best["points"]))
    return best


# ============================================================================ module: grasp_geometry
GRASP_DEPTH = 0.03           # m below the object's top at which the fingertips close (half the object height if smaller)


def narrow_axis_yaw(points):
    """Yaw of a top-down hand whose finger-closing axis (hand y) lies along the object's narrow horizontal axis."""
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    minor = v[:, int(np.argmin(w))]
    phi = float(np.arctan2(minor[1], minor[0]))
    return phi + np.pi / 2       # hand y of Rz(yaw)Rx(pi) is (sin yaw, -cos yaw) = (cos phi, sin phi)


def grasp_pose(obj):
    height = max(obj["top_z"] - obj["bottom_z"], 0.0)
    z = obj["top_z"] - min(GRASP_DEPTH, 0.5 * height)
    p = np.array([obj["center"][0], obj["center"][1], z])
    return p, quat_top_down(narrow_axis_yaw(obj["points"]))


# ============================================================================ module: hold_contract
EMPTY_GRIPPER_OPENING = 0.02  # normalized opening after close_gripper (0 = fully closed); below it nothing is held (prior)


def gripper_opening():
    return float(np.asarray(get_observation()["robot_cartesian_pos"], float)[7])


# ============================================================================ module: place_primitive
RELEASE_MARGIN = {"on": 0.02, "in": 0.01}   # m between the held object's bottom and the support top at release


def release_point(obj, support, relation, grasp_z):
    hang = grasp_z - obj["bottom_z"]           # TCP height above the held object's bottom, fixed by the grasp
    z = support["top_z"] + RELEASE_MARGIN[relation] + hang
    return np.array([support["center"][0], support["center"][1], z])


# ============================================================================ module: verifier
# A0 has no placement verifier: success is decided by the simulator only. PLACE_RESULT reports what was executed.


# ============================================================================ module: manipulation_primitive
def pick_and_place(sg, index):
    goto_home_joint_position()
    view = observe()
    obj = bind(sg["object"], view)
    if obj is None:
        log("BINDING_STOP", status="object_not_found", prompt=sg["object"])
        return "binding_failed"
    support = bind(sg["support"], view, exclude_mask=obj["mask"])
    if support is None:
        log("BINDING_STOP", status="support_not_found", prompt=sg["support"])
        return "binding_failed"
    open_gripper()
    p, q = grasp_pose(obj)
    above = p + np.array([0.0, 0.0, APPROACH_HEIGHT])
    if not move_tcp(above, q, "grasp_above") or not move_line(above, p, q, "grasp_descent", DESCENT_STEPS):
        return "grasp_motion_failed"
    log("GRASP_MOTION", approach="top_down", descent="reached", target=p)
    close_gripper()
    opening = gripper_opening()
    held = opening >= EMPTY_GRIPPER_OPENING
    log("GRASP_VERIFY", status="verified" if held else "failed", detail="gripper_opening", reason=None, after=round(opening, 4))
    if not held:
        open_gripper()
        return "grasp_failed"
    hang = p[2] - obj["bottom_z"]
    carry_z = max(p[2] + APPROACH_HEIGHT, support["top_z"] + LIFT_CLEARANCE + hang)
    lift = np.array([p[0], p[1], carry_z])
    ok = move_line(p, lift, q, "lift", DESCENT_STEPS)
    opening = gripper_opening()
    log("LIFT_MOTION", status="reached" if ok else "ik_failed", hold="ok" if opening >= EMPTY_GRIPPER_OPENING else "failed")
    if not ok:
        return "lift_failed"
    r = release_point(obj, support, sg["relation"], p[2])
    over = np.array([r[0], r[1], carry_z])
    if not move_tcp(over, q, "transport") or not move_line(over, r, q, "place_descent", DESCENT_STEPS):
        log("PLACE_RESULT", status="motion_failed", reason="ik_failed")
        return "place_motion_failed"
    open_gripper()
    log("PLACE_RESULT", status="released", reason=None, target=r)
    move_tcp(r + np.array([0.0, 0.0, APPROACH_HEIGHT]), q, "retreat")
    return "released"


# ============================================================================ module: language_adapter
# Grammar only: verbs and relation words are English function words, objects are the instruction's own noun phrases.
PICK_VERBS = r"(?:pick up|pick|grab|take)"
PLACE_VERBS = r"(?:put|place|set|stack|move)"
ANY_VERB = r"(?:pick|grab|take|put|place|set|stack|move|open|close|turn|push|pull)"   # clause starts (split points only)
RELATIONS = {"on top of": "on", "onto": "on", "on": "on", "into": "in", "inside": "in", "in": "in"}
_REL = "|".join(sorted(map(re.escape, RELATIONS), key=len, reverse=True))
LA_SUPPORTED = {"on": ["-"], "in": ["-"]}


def _np(text):
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip())


def la_parse(td, language):
    """(subgoals, mode). td (task descriptors) is ignored: the original interface provides none."""
    text = re.sub(r"[.!]+$", "", str(language).strip().lower())
    clauses = [c.strip() for c in re.split(rf"\s*,?\s*\band\s+(?={ANY_VERB}\b)", text) if c.strip()]   # "and" inside a noun phrase stays
    subgoals, pending = [], None
    for c in clauses:
        m = re.match(rf"^{PICK_VERBS}\s+(?P<obj>.+)$", c)
        if m:
            pending = _np(m.group("obj")); continue
        m = re.match(rf"^{PLACE_VERBS}\s+(?P<obj>.+?)\s+(?P<rel>{_REL})\s+(?P<sup>.+)$", c)
        if m:
            obj = _np(m.group("obj"))
            if obj in ("it", "them") and pending:
                obj = pending
            subgoals.append({"verb": "place", "object": obj, "relation": RELATIONS[m.group("rel")], "side": "-", "support": _np(m.group("sup"))})
            pending = None; continue
        return [], "unparsed_clause"
    if pending is not None:
        return [], "pick_without_place"
    return subgoals, "grammar_v0"


def la_dispatch(sg):
    """(ok, why): can this subgoal be executed by the program?"""
    if sg.get("verb") != "place":
        return False, "vocabulary"
    if sg.get("relation") not in LA_SUPPORTED or sg.get("side", "-") not in LA_SUPPORTED[sg["relation"]]:
        return False, "relation"
    return True, "ok"


def la_summary(sg):
    return f"{sg.get('verb')}({sg.get('object')} {sg.get('relation')}/{sg.get('side')} {sg.get('support')})"


# ============================================================================ module: task_driver
def run():
    try:
        lang = str(env.handle.task_language)
        subgoals, mode = la_parse({}, lang)
        log("LANG_PARSE", mode=mode, language=lang, subgoals=[la_summary(s) for s in subgoals])
        if not subgoals:
            log("BINDING_STOP", status="language_unparsed", relation=None, side=None)
            return
        for i, sg in enumerate(subgoals):
            ok, why = la_dispatch(sg)
            if not ok:
                log("BINDING_STOP", status="relation_unsupported" if why == "relation" else "language_unparsed",
                    relation=sg.get("relation"), side=sg.get("side"))
                return
        for i, sg in enumerate(subgoals):
            status = pick_and_place(sg, i)
            log("SUBGOAL_STATUS", index=i, subgoal=la_summary(sg), status=status)
            if status != "released":
                return
        goto_home_joint_position()
    except Exception as e:  # noqa: BLE001 - reported, never raised (a traceback would invalidate the stage)
        log("TASK_STOP", status="exception", error=repr(e)[:300])


run()
