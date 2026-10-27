#!/usr/bin/env python3
"""Offline first-failure re-analysis of ASPIRE zero-shot rounds (read-only on source rounds).

For every launch row in <round>/attempts_log_<phase>.csv this script
  1. resolves the episode directory by the FULL task string + arm + seed recorded in the
     per-episode attempt_manifest.json (never by truncated names, never glob(...)[0]);
  2. rebuilds a chronological primitive timeline from task-qualified per-episode files only
     (diag/motion_calls.jsonl, diag/contact_calls.jsonl, diag/gate_decisions.jsonl,
     diag/events.jsonl, artifacts/*/stdout.txt) -- the shared logs/ directory is not used;
  3. attributes the episode to its FIRST failing primitive, labels every later failure as
     dependent (precondition / guard / gate refusal that follows an earlier failure) or
     independent, and summarises the retry chain (approach / goto_home non-arrivals);
  4. checks the grasp/hold self-reports against simulator truth (object z + gripper contact
     at close and after lift) from events.jsonl.

Outputs (only under --out): episodes.csv, attempts.csv, failures.csv, timelines/*.jsonl,
resolution.json, report.md, and a copy of this script.

Usage:
  python3 posthoc_first_failure.py --out OUTDIR ROUND_DIR [ROUND_DIR ...]
Stdlib only.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime

PRECONDITION_DETAILS = {"precondition_start_pose", "precondition_axis", "precondition_gripper"}
LIFT_TRUTH_DZ_M = 0.02      # object must rise >= 2 cm with the EEF to count as held after lift
NO_LIFT_DZ_M = 0.005        # < 0.5 cm rise: object never left the support
TAG_JSON = re.compile(r"^([A-Z][A-Z0-9_]+) (\{.*\})\s*$")
ATTEMPT_LINE = re.compile(r"^Attempt (\d+) \((.*)\): (\w+)")
ABORT_LINE = re.compile(r"^GRASP_ABORT reason=(.*)$")
STDOUT_TAGS = {"CANDIDATE_FILTER", "GRASP_VERIFY", "LIFT_MOTION", "E1_GUARD", "PLACE_RESULT",
               "RELEASE_VERIFY", "MF_SHADOW_PRE", "MF_SHADOW_POST", "V1_SHADOW", "GRASP_MOTION",
               "MF_GRASP_PLAN", "MF_GRASP_ACTUAL", "BINDING", "BINDING_STOP", "APPROACH_PLAN"}


# ----------------------------------------------------------------------------- io helpers
def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append({"_unparsed": line[:200]})
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


# ----------------------------------------------------------------------------- resolution
def resolve_episode(round_dir, row):
    """Return (episode_dir, manifest, problem). Exact full-task match only."""
    base = os.path.join(round_dir, f"stage_{row['phase']}", "attempts", row["attempt_dir"], row["arm"])
    if not os.path.isdir(base):
        return None, None, f"missing attempt dir {base}"
    matches = []
    for mpath in sorted(glob.glob(os.path.join(base, "*", "seed_*", "attempt_manifest.json"))):   # seed dirs are zero-padded (seed_01); match numerically below
        try:
            m = json.load(open(mpath))
        except Exception as e:  # noqa: BLE001
            continue
        if m.get("task") == row["task"] and m.get("arm") == row["arm"] and str(m.get("seed")).lstrip("0") == str(row["seed"]).lstrip("0"):
            matches.append((os.path.dirname(mpath), m))
    if len(matches) == 1:
        return matches[0][0], matches[0][1], None
    if not matches:
        return None, None, f"no manifest under {base} matches full task {row['task']!r}"
    return None, None, f"{len(matches)} manifests match (ambiguous): {[d for d, _ in matches]}"


# ----------------------------------------------------------------------------- stdout
def parse_stdout(path):
    evs = []
    if not path or not os.path.exists(path):
        return evs, False
    crashed = False
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if "jaxls" in line:
            continue
        if line.startswith("Traceback"):
            crashed = True
        m = TAG_JSON.match(line)
        if m and (m.group(1) in STDOUT_TAGS or m.group(1) in ("CONTACT_MOVE", "HOME_MOTION")):
            try:
                evs.append({"tag": m.group(1), "data": json.loads(m.group(2))})
            except json.JSONDecodeError:
                evs.append({"tag": m.group(1), "data": {"_unparsed": m.group(2)[:200]}})
            continue
        m = ATTEMPT_LINE.match(line)
        if m:
            evs.append({"tag": "ATTEMPT_END", "data": {"n": int(m.group(1)), "desc": m.group(2), "result": m.group(3)}})
            continue
        m = ABORT_LINE.match(line)
        if m:
            evs.append({"tag": "GRASP_ABORT", "data": {"reason": m.group(1)}})
            continue
        if line.startswith("All grasp attempts failed"):
            evs.append({"tag": "ALL_ATTEMPTS_FAILED", "data": {}})
    return evs, crashed


# ----------------------------------------------------------------------------- timeline
def motion_event(r, step):
    status = r.get("status")
    ok = status == "reached"
    ev = {"src": "motion", "kind": "home" if r.get("call") == "goto_home" else "move", "label": r.get("label"),
          "call": r.get("call"), "status": status, "detail": r.get("detail"), "ok": ok, "sim_step": step,
          "steps_used": r.get("steps_used"), "steps_planned": r.get("steps_planned"), "budget": r.get("budget"),
          "plan_exceeds_budget": r.get("plan_exceeds_budget"), "reason": (r.get("reason") or "")[:200]}
    tj, fj = r.get("target_joints"), r.get("final_joints")
    if tj and fj and len(tj) == len(fj):
        errs = [abs(a - b) for a, b in zip(fj, tj)]
        k = max(range(len(errs)), key=lambda i: errs[i])
        ev.update({"joint_err_max_rad": round(errs[k], 4), "joint_err_argmax": k + 1})
    sj = r.get("start_joints")
    if tj and sj and len(tj) == len(sj):
        d = [abs(a - b) for a, b in zip(sj, tj)]
        k = max(range(len(d)), key=lambda i: d[i])
        ev.update({"joint_travel_max_rad": round(d[k], 4), "joint_travel_argmax": k + 1})
    return ev


def build_timeline(ep_dir, stdout_path):
    diag = os.path.join(ep_dir, "diag")
    notes = []
    motions = [r for r in read_jsonl(os.path.join(diag, "motion_calls.jsonl"))
               if r.get("call") in ("move_to_pose", "goto_home", "move_to_joints") and r.get("detail") != "plan_only"]
    contacts = [r for r in read_jsonl(os.path.join(diag, "contact_calls.jsonl")) if r.get("status") is not None]
    gates = read_jsonl(os.path.join(diag, "gate_decisions.jsonl"))
    events = read_jsonl(os.path.join(diag, "events.jsonl"))
    home_steps = [e.get("sim_step") for e in events if e.get("kind") == "after:goto_home_joint_position"]
    call_steps = {k: [e.get("sim_step") for e in events if e.get("kind") == f"call:{k}"]
                  for k in ("verify_grasp", "verify_hold", "verify_release")}

    tl = []
    last_step, hi = 0, 0
    for r in motions:
        step = r.get("sim_step")
        if r.get("call") == "goto_home":
            if hi < len(home_steps):
                step = home_steps[hi]
            else:
                notes.append("goto_home without events.jsonl step; placed after previous motion")
                step = last_step + 0.5
            hi += 1
        if step is None:
            step = last_step + 0.5
        last_step = step
        tl.append(motion_event(r, step))
    if hi == 0 and home_steps:
        notes.append(f"home status not recorded (legacy goto_home, {len(home_steps)} calls)")
    elif hi != len(home_steps):
        notes.append(f"goto_home count mismatch motion_calls={hi} events={len(home_steps)}")
    for r in contacts:
        tl.append({"src": "contact", "kind": "contact_move", "label": r.get("label"), "status": r.get("status"),
                   "detail": r.get("detail"), "ok": bool(r.get("success")), "sim_step": r.get("sim_step"),
                   "steps_used": r.get("steps_used"), "budget": r.get("budget"),
                   "reason": (r.get("reason") or "")[:200]})
    for r in gates:
        if "allow" not in r:      # "executed" bookkeeping records, not decisions
            continue
        tl.append({"src": "gate", "kind": "gate", "label": r.get("action"), "status": "allow" if r.get("allow") else "refuse",
                   "detail": r.get("status"), "ok": bool(r.get("allow")), "sim_step": r.get("sim_step"),
                   "reason": str(r.get("why"))[:200]})

    # stdout-only events: anchor on CONTACT_MOVE (k-th <-> contacts[k]) and HOME_MOTION (k-th <-> home k-th)
    sevs, crashed = parse_stdout(stdout_path)
    ci = hj = 0
    floor = 0
    n_cm_stdout = sum(1 for e in sevs if e["tag"] == "CONTACT_MOVE")
    if n_cm_stdout != len(contacts):
        notes.append(f"CONTACT_MOVE stdout={n_cm_stdout} vs contact_calls={len(contacts)}; stdout anchoring approximate")
    home_tl = [e for e in tl if e["kind"] == "home"]
    tag_call = {"GRASP_VERIFY": "verify_grasp", "LIFT_MOTION": "verify_hold", "RELEASE_VERIFY": "verify_release"}
    tag_idx = Counter()
    order = 0
    for e in sevs:
        tag, d = e["tag"], e["data"]
        if tag == "CONTACT_MOVE":
            if ci < len(contacts) and contacts[ci].get("sim_step") is not None:
                floor = contacts[ci]["sim_step"]
            ci += 1
            continue
        if tag == "HOME_MOTION":
            if hj < len(home_tl):
                floor = home_tl[hj]["sim_step"]
            hj += 1
            continue
        if tag in tag_call:
            steps = call_steps[tag_call[tag]]
            k = tag_idx[tag]
            tag_idx[tag] += 1
            if k < len(steps) and steps[k] is not None:
                floor = steps[k]
            else:
                notes.append(f"{tag} #{k+1} has no events.jsonl call step; anchored on previous event")
        order += 1
        ev = {"src": "stdout", "kind": tag.lower(), "sim_step": floor + 0.001 * order, "anchor_step": floor}
        if tag == "CANDIDATE_FILTER":
            ev.update(ok=bool(d.get("ok")), status="ok" if d.get("ok") else "rejected", detail=str(d.get("why"))[:160])
        elif tag == "GRASP_VERIFY":
            ev.update(ok=d.get("status") == "verified", status=d.get("status"), detail=d.get("detail"),
                      reason=str(d.get("reason"))[:160], opening_after=d.get("after"))
        elif tag == "LIFT_MOTION":
            ev.update(ok=d.get("status") == "reached" and d.get("hold") != "failed", status=d.get("status"),
                      detail=f"hold={d.get('hold')}", hold=d.get("hold"))
        elif tag == "E1_GUARD":
            ev.update(ok=bool(d.get("allow")), status="allow" if d.get("allow") else "refuse",
                      label=d.get("action"), detail=str(d.get("why"))[:160])
        elif tag == "PLACE_RESULT":
            ev.update(ok=d.get("status") == "released", status=d.get("status"), detail=d.get("reason"),
                      shadow_post=d.get("shadow_post"))
        elif tag == "RELEASE_VERIFY":
            ev.update(ok=d.get("status") == "verified", status=d.get("status"), detail=str(d.get("reason"))[:160])
        elif tag == "GRASP_MOTION":
            ev.update(ok=True, info=True, status=d.get("approach"), detail=f"descent={d.get('descent')}")
        elif tag in ("MF_SHADOW_PRE", "MF_SHADOW_POST", "V1_SHADOW"):
            ev.update(ok=True, info=True, status=d.get("status"), detail=str(d.get("reason"))[:160])
        elif tag in ("MF_GRASP_PLAN", "MF_GRASP_ACTUAL"):
            eng = d.get("engagement") or {}
            ev.update(ok=True, info=True, status=d.get("status") or d.get("mode"), mode=d.get("mode"), rule=d.get("rule"),
                      target_z=d.get("target_z"), pad_below_top_m=eng.get("pad_below_top_m"))
        elif tag == "BINDING":
            ev.update(ok=d.get("status") == "bound", status=d.get("status"), label="binding",
                      detail=f"support={d.get('support')} margin={d.get('rel_margin')}")
        elif tag == "BINDING_STOP":
            ev.update(ok=True, info=True, status="binding_stop", detail=str(d.get("status")))
        elif tag == "APPROACH_PLAN":
            ev.update(ok=True, info=True, status=f"chosen={d.get('chosen')}", detail=f"planned={d.get('planned')} budget={d.get('budget')}")
        elif tag == "ATTEMPT_END":
            ev.update(ok=True, info=True, status=d.get("result"), detail=d.get("desc"), attempt_n=d.get("n"))
        elif tag == "GRASP_ABORT":
            ev.update(ok=True, info=True, status="abort", detail=str(d.get("reason"))[:200])
        elif tag == "ALL_ATTEMPTS_FAILED":
            ev.update(ok=True, info=True, status="all_failed")
        else:
            continue
        tl.append(ev)

    tl.sort(key=lambda e: (e["sim_step"] if e.get("sim_step") is not None else 1e12))
    # attempt index: attempts are closed by ATTEMPT_END lines
    att = 1
    for e in tl:
        e["attempt"] = att
        if e["kind"] == "attempt_end":
            att = (e.get("attempt_n") or att) + 1
    return tl, events, crashed, notes


# ----------------------------------------------------------------------------- truth
def goal_object(manifest):
    preds = manifest.get("final_official_predicates") or ""
    if isinstance(preds, list):
        for p in preds:
            pr = p.get("predicate") if isinstance(p, dict) else p
            if isinstance(pr, (list, tuple)) and len(pr) >= 2:
                return pr[1]
        return None
    m = re.search(r"\['\w+', '([^']+)'", str(preds))
    return m.group(1) if m else None


def _dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def descent_binding(events, obj):
    """Nearest object (xy) to the EEF at the end of each contact_move, i.e. what the grasp descent was aimed at."""
    out = []
    for e in events:
        if e.get("kind") != "after:contact_move":
            continue
        eef = e.get("eef_pos_world"); objs = e.get("objects") or {}
        if not eef or obj not in objs:
            continue
        near = sorted((_dist(v["pos_world"][:2], eef[:2]), k) for k, v in objs.items())
        out.append({"step": e.get("sim_step"), "nearest": near[0][1], "nearest_xy_m": round(near[0][0], 4),
                    "goal_xy_m": round(_dist(objs[obj]["pos_world"][:2], eef[:2]), 4)})
    return out


def hold_truth(events, obj):
    """Per close_gripper: which object is nearest the EEF at close, goal-object z at close / after the following lift,
    gripper contact, and which objects (any, not only the goal) rose with the EEF during the lift."""
    out = []
    if not obj:
        return out
    closes = [i for i, e in enumerate(events) if e.get("kind") == "after:close_gripper"]
    for ci in closes:
        c = events[ci]
        objs0 = c.get("objects") or {}
        o = objs0.get(obj)
        if not o:
            continue
        z0 = o["pos_world"][2]
        eef = c.get("eef_pos_world")
        near = sorted((_dist(v["pos_world"], eef), k) for k, v in objs0.items()) if eef else []
        rec = {"close_step": c.get("sim_step"), "contact_at_close": (c.get("gripper_contact") or {}).get(obj),
               "nearest_obj_at_close": near[0][1] if near else None,
               "nearest_dist_m": round(near[0][0], 4) if near else None,
               "goal_dist_at_close_m": round(_dist(o["pos_world"], eef), 4) if eef else None,
               "gripper_qpos_close": c.get("gripper_qpos"),
               "lift_step": None, "dz_lift_m": None, "contact_after_lift": None, "lifted_objects": "", "truth": "no_lift_executed"}
        for e in events[ci + 1:]:
            if e.get("kind") == "after:close_gripper" or e.get("kind") == "after:open_gripper":
                break
            if e.get("kind") == "after:move_to_pose" and str((e.get("extra") or {}).get("label", "")).endswith(":lift"):
                objs1 = e.get("objects") or {}
                lifted = [k for k, v in objs1.items() if k in objs0 and v["pos_world"][2] - objs0[k]["pos_world"][2] >= LIFT_TRUTH_DZ_M]
                rec["lifted_objects"] = ";".join(lifted)
                o2 = objs1.get(obj)
                if o2:
                    dz = o2["pos_world"][2] - z0
                    con = (e.get("gripper_contact") or {}).get(obj)
                    rec.update(lift_step=e.get("sim_step"), dz_lift_m=round(dz, 4), contact_after_lift=con)
                    if dz >= LIFT_TRUTH_DZ_M and con:
                        rec["truth"] = "held"
                    elif dz < NO_LIFT_DZ_M and any(k != obj for k in lifted):
                        rec["truth"] = "wrong_object_held"
                    elif dz < NO_LIFT_DZ_M:
                        rec["truth"] = "never_lifted"
                    else:
                        rec["truth"] = "partial_or_slipping"
                break
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- attribution
def fail_label(e):
    lab = e.get("label") or ""
    lab = lab.split(":")[-1] if ":" in lab else lab
    if e["kind"] == "candidate_filter":
        return "candidate_filter:rejected"
    if e["kind"] == "grasp_verify":
        return f"grasp_verify:{e.get('detail') or e.get('status')}"
    if e["kind"] == "lift_motion":
        return f"lift:{e.get('status')}/{e.get('detail')}"
    if e["kind"] == "home":
        return f"goto_home:{e.get('status')}"
    if e["kind"] == "binding":
        return f"binding:{e.get('status')}"
    if e["kind"] in ("e1_guard", "gate"):
        return f"{e['kind']}_refuse:{lab}"
    return f"{lab or e['kind']}:{e.get('status')}/{e.get('detail')}"


def is_dependent(e):
    return (e.get("detail") in PRECONDITION_DETAILS) or e["kind"] in ("gate", "e1_guard")


def attribute(tl, official, place_released):
    fails = [e for e in tl if not e.get("ok") and not e.get("info")]
    seen_fail = False
    last_nonarrival = None
    for e in tl:
        if e.get("info"):
            continue
        if not e.get("ok"):
            e["is_failure"] = True
            e["dependent"] = bool(seen_fail and is_dependent(e))
            if e["dependent"] and last_nonarrival is not None:
                e["caused_by"] = fail_label(last_nonarrival)
            seen_fail = True
            if e["kind"] in ("move", "home"):
                last_nonarrival = e
        elif e["kind"] in ("move", "home"):
            last_nonarrival = None
    first = fails[0] if fails else None
    last = fails[-1] if fails else None
    if not official and place_released:
        # release was reached but the official relation is false
        synthetic = {"kind": "placement", "label": "placement", "status": "released_relation_false", "detail": "official=0",
                     "ok": False, "sim_step": tl[-1]["sim_step"] + 1 if tl else 0, "is_failure": True, "dependent": False}
        if first is None:
            first = synthetic
        last = synthetic
    return first, last, fails


# ----------------------------------------------------------------------------- main
def analyse_round(round_dir, out_dir, rows_out, att_out, fail_out, res_log):
    rname = os.path.basename(os.path.normpath(round_dir))
    orig = {}
    oc = os.path.join(round_dir, "results", "results.csv")
    if os.path.exists(oc):
        for r in csv.DictReader(open(oc)):
            orig.setdefault((r.get("phase"), r.get("arm"), r.get("task"), str(r.get("seed"))), []).append(r)  # task = task_short
    launch_rows = []
    for p in sorted(glob.glob(os.path.join(round_dir, "attempts_log_*.csv"))):
        launch_rows += list(csv.DictReader(open(p)))
    dir_tasks = defaultdict(set)
    for r in launch_rows:
        dir_tasks[(r["phase"], r["attempt_dir"], r["arm"])].add(r["task"])
    for row in launch_rows:
        key = f"{rname}|{row['phase']}|{row['arm']}|{row['task']}|s{row['seed']}|{row['attempt_dir']}"
        collided = len(dir_tasks[(row["phase"], row["attempt_dir"], row["arm"])]) > 1
        ep_dir, man, problem = resolve_episode(round_dir, row)
        base = {"round": rname, "phase": row["phase"], "arm": row["arm"], "task": row["task"], "seed": row["seed"],
                "state_idx": row.get("state_idx"), "attempt_dir": row["attempt_dir"], "attempt_dir_collision": collided,
                "csv_task_completed": row.get("task_completed")}
        if problem:
            res_log.append({"key": key, "status": "unresolved", "why": problem})
            rows_out.append({**base, "resolved": False, "resolve_problem": problem})
            continue
        res_log.append({"key": key, "status": "ok", "episode_dir": os.path.relpath(ep_dir, round_dir)})
        stdouts = sorted(glob.glob(os.path.join(ep_dir, "artifacts", "*", "stdout.txt")))
        tl, events, crashed, notes = build_timeline(ep_dir, stdouts[0] if len(stdouts) == 1 else None)
        if len(stdouts) != 1:
            notes.append(f"{len(stdouts)} stdout.txt files; stdout events skipped")
        official = str(man.get("official_task_completed")) in ("1", "True", "true")
        place_released = any(e["kind"] == "place_result" and e.get("status") == "released" for e in tl)
        first, last, fails = attribute(tl, official, place_released)
        obj = goal_object(man)
        holds = hold_truth(events, obj)
        binds = [b for b in descent_binding(events, obj)]
        # grasp descents only: contact_moves before the first place_descent-like event are grasp descents; use the first one
        b0 = binds[0] if binds else None
        verifies = [e for e in tl if e["kind"] == "grasp_verify"]
        lifts = [e for e in tl if e["kind"] == "lift_motion"]
        hold_self = [l.get("hold") for l in lifts]
        # pair k-th grasp_verify with k-th close (both one per executed close)
        gv_fp = gv_fn = wrong_obj = 0
        for k, h in enumerate(holds):
            v = verifies[k] if k < len(verifies) else None
            h["grasp_verify"] = v.get("status") if v else None
            if v and v.get("status") == "verified" and h["truth"] in ("never_lifted", "partial_or_slipping"):
                gv_fp += 1
            if h["truth"] == "wrong_object_held":
                wrong_obj += 1
            if v and v.get("status") != "verified" and h.get("contact_at_close"):
                gv_fn += 1
        hold_fp = sum(1 for k, h in enumerate(holds) if h["truth"] != "held" and h["dz_lift_m"] is not None
                      and k < len(hold_self) and hold_self[k] == "verified")
        homes = [e for e in tl if e["kind"] == "home"]
        home_unknown = any(n.startswith("home status not recorded") for n in notes)
        # truth-corrected root: a self-reported grasp/hold success that truth contradicts, before the first self-reported failure
        corrected = fail_label(first) if first else ("" if official else "none_logged")
        first_step = first.get("sim_step") if first else float("inf")
        for k, h in enumerate(holds):
            if h["close_step"] is None or h["close_step"] >= first_step:
                continue
            if h["truth"] == "wrong_object_held":
                corrected = f"wrong_object_held:{h['lifted_objects']}"
                break
            if h.get("grasp_verify") == "verified" and h["truth"] in ("never_lifted", "partial_or_slipping"):
                corrected = f"grasp_verify_false_positive:{h['truth']}"
                break
            if h["truth"] in ("never_lifted", "partial_or_slipping") and k < len(hold_self) and hold_self[k] == "verified":
                corrected = f"hold_verify_false_positive:{h['truth']}"
                break
        approaches = [e for e in tl if e["kind"] == "move" and str(e.get("label", "")).endswith(":approach")]
        n_att = max([e.get("attempt", 1) for e in tl] or [1])
        o_rows = orig.get((row["phase"], row["arm"], man.get("task_short"), str(row["seed"])), []) or \
            orig.get((row["phase"], row["arm"], row["task"], str(row["seed"])), [])
        o = o_rows[0] if len(o_rows) == 1 else {}
        dep_fails = [f for f in fails if f.get("dependent")]
        rec = {**base, "resolved": True, "episode_dir": os.path.relpath(ep_dir, round_dir),
               "attempt_id": man.get("attempt_id"), "official": int(official),
               "official_mismatch_vs_csv": str(int(official)) != str(row.get("task_completed")),
               "crashed_traceback": crashed, "sandbox_rc": man.get("sandbox_rc"), "goal_object": obj,
               "n_attempts": n_att, "n_failure_events": len(fails), "n_dependent_failures": len(dep_fails),
               "first_failure": fail_label(first) if first else ("" if official else "none_logged"),
               "first_failure_truth_corrected": corrected,
               "first_failure_attempt": first.get("attempt") if first else "",
               "first_failure_step": first.get("sim_step") if first else "",
               "last_failure": fail_label(last) if last else "",
               "last_failure_dependent": bool(last.get("dependent")) if last else "",
               "last_failure_caused_by": last.get("caused_by", "") if last else "",
               "recovered_after_failure": bool(official and fails),
               "approach_calls": len(approaches), "approach_nonarrival": sum(1 for a in approaches if not a["ok"]),
               "approach_plan_exceeds_budget": sum(1 for a in approaches if a.get("plan_exceeds_budget")),
               "approach_ik_failed": sum(1 for a in approaches if a.get("detail") == "ik_failed"),
               "home_status_recorded": not home_unknown,
               "home_calls": len(homes), "home_nonarrival": sum(1 for h in homes if not h["ok"]),
               "home_nonarrival_joint_argmax": ";".join(str(h.get("joint_err_argmax")) for h in homes if not h["ok"]),
               "home_nonarrival_joint_err_max": max([h.get("joint_err_max_rad", 0) for h in homes if not h["ok"]] or [""], key=lambda x: x if x != "" else -1),
               "home_nonarrival_travel_max": ";".join(f"j{h.get('joint_travel_argmax')}={h.get('joint_travel_max_rad')}" for h in homes if not h["ok"]),
               "n_closes_truth": len(holds), "truth_per_close": ";".join(h["truth"] for h in holds),
               "grasp_verify_per_close": ";".join(str(h.get("grasp_verify")) for h in holds),
               "wrong_object_held": wrong_obj,
               "first_descent_nearest_obj": b0["nearest"] if b0 else "",
               "first_descent_nearest_xy_m": b0["nearest_xy_m"] if b0 else "",
               "first_descent_goal_xy_m": b0["goal_xy_m"] if b0 else "",
               "first_descent_on_goal": (b0["nearest"] == obj) if b0 else "",
               "nearest_obj_per_close": ";".join(f"{h['nearest_obj_at_close']}@{h['nearest_dist_m']}" for h in holds),
               "grasp_verify_false_pos": gv_fp, "grasp_verify_false_neg": gv_fn, "hold_verify_false_pos": hold_fp,
               "orig_failure_phase": o.get("failure_phase", ""), "orig_row_ambiguous": len(o_rows) > 1,
               "orig_row_suspect_collision": collided,
               "notes": " | ".join(notes)}
        rows_out.append(rec)
        for k, h in enumerate(holds):
            att_out.append({"round": rname, "phase": row["phase"], "arm": row["arm"], "task": row["task"], "seed": row["seed"],
                            "close_idx": k + 1, **h, "lift_self_hold": hold_self[k] if k < len(hold_self) else None})
        for f in fails + ([last] if last is not None and last not in fails else []):
            fail_out.append({"round": rname, "phase": row["phase"], "arm": row["arm"], "task": row["task"], "seed": row["seed"],
                             "attempt": f.get("attempt", ""), "sim_step": f.get("sim_step"), "failure": fail_label(f),
                             "dependent": f.get("dependent"), "caused_by": f.get("caused_by", ""),
                             "is_first": f is first, "reason": f.get("reason", "")})
        tdir = os.path.join(out_dir, "timelines")
        os.makedirs(tdir, exist_ok=True)
        tname = safe(f"{rname}__{row['phase']}__{row['arm']}__{row['task']}__s{row['seed']}__{man.get('attempt_id')}") + ".jsonl"
        with open(os.path.join(tdir, tname), "w") as fh:
            for e in tl:
                fh.write(json.dumps(e, default=str) + "\n")


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def write_report(out_dir, rows, res_log, rounds):
    ok = [r for r in rows if r.get("resolved")]
    lines = ["# Offline first-failure re-analysis", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by `code/posthoc_first_failure.py` (read-only on sources).", "",
             "Rounds: " + ", ".join(f"`{os.path.basename(os.path.normpath(r))}`" for r in rounds), "",
             "Method: episodes resolved by exact full task + arm + seed from each attempt_manifest.json; timeline from "
             "task-qualified diag/*.jsonl + per-episode stdout.txt (shared logs/ not used). First failure = first non-success "
             "primitive in sim-step order. A later failure is *dependent* if it is a precondition/guard/gate refusal after an "
             "earlier failure. Hold truth: object must rise >= 2 cm with gripper contact after lift.", "",
             f"Resolution: {sum(1 for x in res_log if x['status']=='ok')} resolved, "
             f"{sum(1 for x in res_log if x['status']!='ok')} unresolved (see resolution.json).", ""]
    unresolved = [x for x in res_log if x["status"] != "ok"]
    if unresolved:
        lines += ["Unresolved:", ""] + [f"- {x['key']}: {x['why']}" for x in unresolved] + [""]
    by_round = defaultdict(list)
    for r in ok:
        by_round[r["round"]].append(r)
    for rn, rr in by_round.items():
        lines += [f"## {rn}", ""]
        grp = defaultdict(list)
        for r in rr:
            grp[(r["phase"], r["arm"], r["task"])].append(r)
        tab = []
        for (ph, arm, task), g in sorted(grp.items()):
            ff = Counter(r["first_failure_truth_corrected"] for r in g if not r["official"])
            bind_cell = f"{sum(1 for r in g if r['first_descent_on_goal'] is True)}/{sum(1 for r in g if r['first_descent_on_goal'] != '')}"
            home_cell = sum(r["home_nonarrival"] for r in g) if all(r["home_status_recorded"] for r in g) else "n/r"
            tab.append([ph, arm, task.replace("LIVING_ROOM_", "")[:60], f"{sum(r['official'] for r in g)}/{len(g)}",
                        "; ".join(f"{k} x{v}" for k, v in ff.most_common()) or "-",
                        sum(1 for r in g if not r["official"] and r["last_failure_dependent"] is True),
                        home_cell, sum(r["approach_nonarrival"] for r in g),
                        sum(r["grasp_verify_false_pos"] for r in g), sum(r["wrong_object_held"] for r in g),
                        sum(1 for r in g for x in str(r["nearest_obj_per_close"]).split(";") if x and not x.startswith(str(r["goal_object"]) + "@")),
                        bind_cell])
        lines += [md_table(["phase", "arm", "task", "official", "first failure, truth-corrected (failed eps)", "last failure dependent",
                            "home non-arrival", "approach non-arrival", "grasp-verify FP", "wrong object held",
                            "closes nearest a non-goal object", "1st descent aimed at goal"], tab), ""]
        failed = [r for r in rr if not r["official"]]
        joined = [r for r in failed if r["orig_failure_phase"]]
        changed = [r for r in joined if not r["orig_failure_phase"].startswith(r["first_failure"])]
        lines += [f"Original `failure_phase` vs first failure: {len(changed)} of {len(joined)} joined failed episodes differ "
                  f"({len(failed) - len(joined)} not joinable).", ""]
        if changed:
            lines += [md_table(["arm", "task", "seed", "original failure_phase", "first failure", "last failure (dependent?)"],
                               [[r["arm"], r["task"].replace("LIVING_ROOM_", "")[:45], r["seed"], r["orig_failure_phase"][:60],
                                 r["first_failure"], f"{r['last_failure']} ({r['last_failure_dependent']})"] for r in changed]), ""]
        coll = [r for r in rr if r["attempt_dir_collision"]]
        if coll:
            lines += ["Attempt-dir collisions (original results.csv rows for these are suspect; values here come from the "
                      "task-qualified episode dir):", ""] + \
                     [f"- {r['arm']} {r['task']} s{r['seed']} -> `{r['episode_dir']}` first={r['first_failure']}" for r in coll] + [""]
    # cross-round aggregates
    allf = [r for r in ok if not r["official"]]
    lines += ["## Cross-round aggregates (failed episodes)", "",
              f"- failed episodes: {len(allf)}; last recorded failure is a dependent refusal: "
              f"{sum(1 for r in allf if r['last_failure_dependent'] is True)}",
              f"- goto_home non-arrivals (rounds with recorded status): {sum(r['home_nonarrival'] for r in ok)} of "
              f"{sum(r['home_calls'] for r in ok if r['home_status_recorded'])} calls; "
              f"dominant joint: {Counter(j for r in ok for j in str(r['home_nonarrival_joint_argmax']).split(';') if j).most_common(3)}",
              f"- approach non-arrivals: {sum(r['approach_nonarrival'] for r in ok)} of {sum(r['approach_calls'] for r in ok)} "
              f"(plan_exceeds_budget {sum(r['approach_plan_exceeds_budget'] for r in ok)}, ik_failed {sum(r['approach_ik_failed'] for r in ok)})",
              f"- grasp_verify false positives vs truth: {sum(r['grasp_verify_false_pos'] for r in ok)} of "
              f"{sum(r['n_closes_truth'] for r in ok)} closes; false negatives: {sum(r['grasp_verify_false_neg'] for r in ok)}",
              f"- hold_verify false positives: {sum(r['hold_verify_false_pos'] for r in ok)}",
              f"- wrong object held after lift (truth): {sum(r['wrong_object_held'] for r in ok)} closes", "",
              "First-failure histogram, truth-corrected (failed episodes, all rounds):", ""]
    raw = Counter(r["first_failure"] for r in allf)
    lines += [f"- {k}: {v}" for k, v in Counter(r["first_failure_truth_corrected"] for r in allf).most_common()]
    lines += ["", "Self-reported first failure: " + "; ".join(f"{k} x{v}" for k, v in raw.most_common()), "",
              "n/r = goto_home status not recorded in that round (legacy home call); home counts cover G-interface rounds only."]
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("rounds", nargs="+")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for r in a.rounds:
        if os.path.abspath(r).startswith(out):
            sys.exit("output dir must not be inside a source round")
    os.makedirs(os.path.join(out, "code"), exist_ok=True)
    dst = os.path.join(out, "code", os.path.basename(__file__))
    if not os.path.exists(dst) or not os.path.samefile(__file__, dst):
        shutil.copy2(__file__, dst)
    rows, atts, fails, res = [], [], [], []
    for r in a.rounds:
        analyse_round(r, out, rows, atts, fails, res)
    write_csv(os.path.join(out, "episodes.csv"), rows)
    write_csv(os.path.join(out, "attempts.csv"), atts)
    write_csv(os.path.join(out, "failures.csv"), fails)
    json.dump({"generated": datetime.now().isoformat(timespec="seconds"),
               "script_sha256": sha256(__file__), "rounds": [os.path.abspath(r) for r in a.rounds], "episodes": res},
              open(os.path.join(out, "resolution.json"), "w"), indent=1)
    write_report(out, rows, res, a.rounds)
    print(f"episodes={len(rows)} resolved={sum(1 for x in res if x['status']=='ok')} -> {out}")


if __name__ == "__main__":
    main()
