"""Isolated diagnostic recorder v2 (copied from round 20260915_058594; this round adds move_to_pose to MOTION_FNS).

Constraints (unchanged from the parent round's recorder):
- The policy file is never modified. Wrapped API functions return the original result unchanged.
- Ground truth (MuJoCo poses, contacts, official predicates) is read ONLY inside runner-side wrappers
  and the `handle.step` observer, and written to `diag/` for offline scoring. Nothing is returned to
  the policy. No physics steps are added.
- A one-way hook `diag_mark(json_str)` is exposed to the policy namespace. It stores the label in
  the recorder and returns None. It carries information *out* of the policy only.

Generalization: tracked objects = all bodies in the task env's `obj_body_id`; goal pairs come from
`parsed_problem["goal_state"]` (`on <obj> <target>`); when the target is a region/site the pair has
no body position and only the predicate is recorded.
Frames: world = MuJoCo world (wxyz quats). Offsets in the gripper0_eef body frame.
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

EEF_BODY = "gripper0_eef"
BASE_BODY = "robot0_base"
MOTION_FNS = {"move_to_joints", "goto_pose", "goto_home_joint_position", "open_gripper", "close_gripper", "move_to_pose"}
OBS_FN = "get_observation"
PERCEPTION_FNS = {"segment_sam3_text_prompt", "segment_sam3_point_prompt", "plan_grasp", "plan_grasp_from_point_clouds", "point_prompt_molmo"}
FN_CODES = {None: 0, "move_to_joints": 1, "goto_pose": 2, "goto_home_joint_position": 3, "open_gripper": 4, "close_gripper": 5, "move_to_pose": 6}


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _tolist(a):
    return np.asarray(a, dtype=float).round(6).tolist()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return {"unserializable": repr(o)}


class DiagRecorder2:
    def __init__(self, code_env, out_dir: Path):
        self.code_env = code_env
        self.low = code_env.low_level_env
        self.handle = self.low.handle
        self.task_env = self.handle.env.env  # OffScreenRenderEnv.env -> task env
        self.sim = self.task_env.sim
        self.out_dir = Path(out_dir)
        self.obs_dir = self.out_dir / "obs"
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.samples: list[list[float]] = []
        self.sample_fields: list[str] = []
        self.event_idx = 0
        self.obs_idx = 0
        self.errors: list[str] = []
        self.marks: list[dict] = []
        self.perception_calls = 0
        self._current_api_fn: str | None = None
        self._current_mark: dict | None = None
        self._mark_code = 0
        self._probe_static()

    # ------------------------------------------------------------------ probes
    def _probe_static(self):
        m = self.sim.model
        self.eef_bid = m.body_name2id(EEF_BODY)
        self.base_bid = m.body_name2id(BASE_BODY)
        self.obj_bid = dict(self.task_env.obj_body_id)
        self.goal_state = [list(map(str, s)) for s in self.task_env.parsed_problem["goal_state"]]
        # goal pairs: (predicate, object, target, target_is_body)
        self.goal_pairs = []
        for st in self.goal_state:
            if len(st) >= 3:
                self.goal_pairs.append({"pred": st[0], "obj": st[1], "target": st[2], "target_is_body": st[2] in self.obj_bid})
            else:
                self.goal_pairs.append({"pred": st[0], "obj": st[1] if len(st) > 1 else None, "target": None, "target_is_body": False})
        self.goal_objects = sorted({p["obj"] for p in self.goal_pairs if p["obj"] in self.obj_bid})
        self.goal_targets = sorted({p["target"] for p in self.goal_pairs if p["target_is_body"]})
        try:
            g = self.task_env.robots[0].gripper
            self.finger_geoms = list(g.important_geoms["left_finger"]) + list(g.important_geoms["right_finger"])
        except Exception as e:  # noqa: BLE001
            self.finger_geoms = None
            self.errors.append(f"finger geoms unavailable: {e!r}")
        try:
            self.gripper_qpos_addr = [m.get_joint_qpos_addr(j) for j in self.task_env.robots[0].gripper.joints]
        except Exception as e:  # noqa: BLE001
            self.gripper_qpos_addr = None
            self.errors.append(f"gripper joint qpos unavailable: {e!r}")
        self.static = {
            "eef_body": EEF_BODY, "base_body": BASE_BODY,
            "base_pos_world": _tolist(self.sim.data.xpos[self.base_bid]), "base_quat_wxyz": _tolist(self.sim.data.xquat[self.base_bid]),
            "object_body_ids": self.obj_bid, "goal_state": self.goal_state, "goal_pairs": self.goal_pairs,
            "goal_objects": self.goal_objects, "goal_targets": self.goal_targets,
            "finger_geoms": self.finger_geoms, "task_language": getattr(self.handle, "task_language", None),
            "frame_note": "world = MuJoCo world frame; offsets in gripper0_eef body frame; public base frame = world - base_pos_world",
        }

    # -------------------------------------------------------------- snapshots
    def _eef(self):
        d = self.sim.data
        return d.xpos[self.eef_bid].copy(), d.xquat[self.eef_bid].copy()

    def _obj(self, name):
        d = self.sim.data
        b = self.obj_bid[name]
        return d.xpos[b].copy(), d.xquat[b].copy()

    def _contact_obj_obj(self, a, b):
        try:
            return bool(self.task_env.object_states_dict[a].check_contact(self.task_env.object_states_dict[b]))
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def _contact_gripper_obj(self, name):
        if self.finger_geoms is None:
            return {"error": "finger geoms unavailable"}
        try:
            obj = self.task_env.get_object(name)
            return bool(self.task_env.check_contact(self.finger_geoms, obj))
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def _gripper_qpos(self):
        if self.gripper_qpos_addr is None:
            return {"error": "unavailable"}
        return _tolist([self.sim.data.qpos[a] for a in self.gripper_qpos_addr])

    def _predicates(self):
        out = []
        for st in self.task_env.parsed_problem["goal_state"]:
            try:
                out.append({"predicate": list(map(str, st)), "value": bool(self.task_env._eval_predicate(st))})
            except Exception as e:  # noqa: BLE001
                out.append({"predicate": list(map(str, st)), "error": repr(e)})
        try:
            full = bool(self.task_env._check_success())
        except Exception as e:  # noqa: BLE001
            full = {"error": repr(e)}
        return out, full

    def snapshot(self, kind: str, api_fn: str | None = None, extra: dict | None = None):
        eef_p, eef_q = self._eef()
        R_eef = _quat_wxyz_to_R(eef_q)
        objs = {}
        for n in self.obj_bid:
            p, q = self._obj(n)
            rel_world = p - eef_p
            objs[n] = {"pos_world": _tolist(p), "quat_wxyz": _tolist(q), "offset_from_eef_world": _tolist(rel_world),
                       "offset_in_eef_frame": _tolist(R_eef.T @ rel_world), "dist_to_eef": round(float(np.linalg.norm(rel_world)), 6)}
        pairs = {}
        for gp in self.goal_pairs:
            if gp["obj"] in objs and gp["target_is_body"]:
                mp = np.array(objs[gp["obj"]]["pos_world"]); pp = np.array(objs[gp["target"]]["pos_world"])
                pairs[f"{gp['obj']}|{gp['target']}"] = {
                    "xy_dist": round(float(np.linalg.norm(mp[:2] - pp[:2])), 6),
                    "obj_z_minus_target_z": round(float(mp[2] - pp[2]), 6),
                    "contact_target_obj": self._contact_obj_obj(gp["target"], gp["obj"]),
                }
        held = {n: self._contact_gripper_obj(n) for n in self.goal_objects}
        preds, full = self._predicates()
        ev = {
            "event_idx": self.event_idx, "kind": kind, "api_fn": api_fn,
            "sim_step": int(getattr(self.low, "_sim_step_count", -1)), "wall_time": time.time(),
            "mark": self._current_mark,
            "eef_pos_world": _tolist(eef_p), "eef_quat_wxyz": _tolist(eef_q),
            "gripper_qpos": self._gripper_qpos(), "gripper_cmd_fraction": float(getattr(self.low, "_gripper_fraction", float("nan"))),
            "objects": objs, "subgoal_geometry": pairs, "gripper_contact": held,
            "official_predicates": preds, "official_full_success": full,
        }
        if extra:
            ev["extra"] = extra
        self.events.append(ev)
        self.event_idx += 1
        return ev

    # ------------------------------------------------------ per-physics-step
    def install_step_sampler(self):
        orig = self.handle.step
        rec = self

        def sampled_step(action):
            r = orig(action)
            rec._sample()
            return r

        self.handle.step = sampled_step
        self._orig_step = orig
        f = ["sim_step", "eef_x", "eef_y", "eef_z", "grip_q0", "grip_q1"]
        for o in self.goal_objects:
            f += [f"obj:{o}:x", f"obj:{o}:y", f"obj:{o}:z", f"obj:{o}:in_gripper"]
        for t in self.goal_targets:
            f += [f"tgt:{t}:x", f"tgt:{t}:y", f"tgt:{t}:z"]
        for i, gp in enumerate(self.goal_pairs):
            f += [f"pair{i}:contact", f"pair{i}:pred"]
        f += ["api_fn_code", "mark_code"]
        self.sample_fields = f

    def _sample(self):
        try:
            eef_p, _ = self._eef()
            gq = self._gripper_qpos()
            if isinstance(gq, dict):
                gq = [float("nan"), float("nan")]
            row = [int(getattr(self.low, "_sim_step_count", -1)), *eef_p, *gq]
            for o in self.goal_objects:
                p, _ = self._obj(o); c = self._contact_gripper_obj(o)
                row += [*p, float(c) if isinstance(c, bool) else float("nan")]
            for t in self.goal_targets:
                p, _ = self._obj(t); row += [*p]
            preds, _ = self._predicates()
            for i, gp in enumerate(self.goal_pairs):
                c = self._contact_obj_obj(gp["target"], gp["obj"]) if gp["target_is_body"] else None
                pv = preds[i].get("value", None)
                row += [float(c) if isinstance(c, bool) else float("nan"), float(pv) if isinstance(pv, bool) else float("nan")]
            row += [FN_CODES.get(self._current_api_fn, 9), self._mark_code]
            self.samples.append([float(x) for x in row])
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"sample error at sim_step {getattr(self.low, '_sim_step_count', -1)}: {e!r}")

    # ------------------------------------------------------------ API wrapping
    def install_api_wrappers(self):
        rec = self
        for api in self.code_env._apis.values():
            orig_functions = api.functions

            def patched_functions(_orig=orig_functions):
                fns = _orig()
                out = {name: rec._wrap(name, fn) for name, fn in fns.items()}
                out["diag_mark"] = rec.mark  # one-way annotation hook (returns None)
                return out

            api.functions = patched_functions

    def mark(self, payload):
        """One-way hook: the policy labels attempts/subgoals/segments. Returns None (no info flows back)."""
        try:
            d = json.loads(payload) if isinstance(payload, str) else dict(payload)
        except Exception:  # noqa: BLE001
            d = {"raw": str(payload)}
        d["_sim_step"] = int(getattr(self.low, "_sim_step_count", -1)); d["_event_idx"] = self.event_idx
        self.marks.append(d)
        self._current_mark = d
        self._mark_code = len(self.marks)
        self.events.append({"event_idx": self.event_idx, "kind": "mark", "api_fn": "diag_mark", "sim_step": d["_sim_step"], "mark": d})
        self.event_idx += 1
        return None

    def _save_public_obs(self, obs, idx):
        d = self.obs_dir / f"obs_{idx:03d}"
        d.mkdir(exist_ok=True)
        try:
            cam = obs["agentview"]
            rgb = np.asarray(cam["images"]["rgb"]); depth = np.asarray(cam["images"]["depth"])
            from PIL import Image
            Image.fromarray(rgb.astype(np.uint8)).save(d / "rgb.jpg", quality=95)
            np.save(d / "rgb.npy", rgb); np.save(d / "depth.npy", depth)
            np.save(d / "K.npy", np.asarray(cam["intrinsics"])); np.save(d / "E.npy", np.asarray(cam["pose_mat"]))
            np.save(d / "robot_joint_pos.npy", np.asarray(obs.get("robot_joint_pos", [])))
            np.save(d / "robot_cartesian_pos.npy", np.asarray(obs.get("robot_cartesian_pos", [])))
            return str(d)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"obs save error idx {idx}: {e!r}")
            return {"error": repr(e)}

    def _wrap(self, name, fn):
        rec = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            extra = rec._summarize_args(name, args, kwargs)
            if name in MOTION_FNS:
                rec.snapshot(f"before:{name}", name, extra)
                rec._current_api_fn = name
                try:
                    return fn(*args, **kwargs)
                finally:
                    rec._current_api_fn = None
                    rec.snapshot(f"after:{name}", name, extra)
            elif name == OBS_FN:
                result = fn(*args, **kwargs)
                idx = rec.obs_idx; rec.obs_idx += 1
                path = rec._save_public_obs(result, idx)
                rec.snapshot(f"obs:{idx}", name, {"obs_path": path})
                return result
            else:
                if name in PERCEPTION_FNS:
                    rec.perception_calls += 1
                rec.events.append({"event_idx": rec.event_idx, "kind": f"call:{name}", "api_fn": name,
                                   "sim_step": int(getattr(rec.low, "_sim_step_count", -1)), "mark": rec._current_mark, "extra": extra})
                rec.event_idx += 1
                return fn(*args, **kwargs)

        return wrapper

    @staticmethod
    def _summarize_args(name, args, kwargs):
        out = {}
        try:
            if name == "move_to_joints" and args:
                out["joints"] = _tolist(args[0])
            elif name in ("solve_ik", "goto_pose", "move_to_pose") and len(args) >= 2:
                out["position"] = _tolist(args[0]); out["quaternion_wxyz"] = _tolist(args[1])
                if name == "move_to_pose":
                    out["mode"] = kwargs.get("mode", args[2] if len(args) > 2 else "free"); out["label"] = kwargs.get("label", "")
            elif name == "segment_sam3_text_prompt" and len(args) >= 2:
                out["text_prompt"] = str(args[1])
        except Exception as e:  # noqa: BLE001
            out["summarize_error"] = repr(e)
        return out

    # ------------------------------------------------------------------- save
    def save(self, manifest_extra: dict | None = None):
        (self.out_dir / "events.jsonl").write_text("\n".join(json.dumps(e, default=_json_default) for e in self.events) + "\n")
        if self.samples:
            np.savez_compressed(self.out_dir / "samples.npz", data=np.array(self.samples, dtype=float), fields=np.array(self.sample_fields))
        (self.out_dir / "marks.json").write_text(json.dumps(self.marks, indent=1, default=_json_default))
        meta = {"static": self.static, "errors": self.errors, "n_events": len(self.events), "n_samples": len(self.samples),
                "n_obs": self.obs_idx, "n_marks": len(self.marks), "perception_calls": self.perception_calls}
        if manifest_extra:
            meta.update(manifest_extra)
        (self.out_dir / "recorder_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return meta
