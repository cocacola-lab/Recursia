"""Six-tool RGB-D FR3 adapter and evaluator-only relation checks."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cap.saved_scripts.fr3_placement.runtime import PlacementRuntime, PhysicalFault
from cap.saved_scripts.fr3_placement.perception import describe_object
from cap.saved_scripts.fr3_runtime.geometry import points_from_mask
from cap.agent.fr3_console import emit_progress
from .prompts import TOOLS
from .observation import capture_observation, array_observation
from .graspnet import ContactGraspNetClient, select_grasp
from .evaluation import is_cup_plate_task, placement_prompt, validate_placement_response

class RecoverableCollision(RuntimeError):
    """A stopped motion is feedback; controller recovery is still required."""
    hardware_fault = False


def collision_fault(state):
    """Classify only observed collision/reflex errors, never arbitrary faults."""
    codes = set()
    for key in ("current_errors", "last_motion_errors"):
        value = state.get(key, [])
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return False
        if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) for v in value):
            return False
        codes.update(value)
    allowed = {"joint_reflex", "cartesian_reflex", "self_collision_avoidance_violation"}
    if codes:
        return codes <= allowed
    # Some reads expose empty error lists while the reflex snapshot already
    # contains collision flags. Require both pieces of physical evidence.
    if str(state.get("robot_mode")) != "RobotMode.Reflex":
        return False
    for key in ("joint_collision", "cartesian_collision"):
        values = np.asarray(state.get(key, []), dtype=float)
        if np.all(np.isfinite(values)) and np.any(values > 0):
            return True
    return False

def describe_volume(capture, mask, cfg, name):
    """Visible RGB-D envelope, not simulator dimensions or a top-plane-only fit."""
    points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              mask, stride=2, distortion=capture.distortion_coeffs)
    if len(points) < 20:
        raise ValueError(f"insufficient valid depth for {name}")
    lo, hi = np.quantile(points, [0.02, 0.98], axis=0)
    kept = points[np.all((points >= lo) & (points <= hi), axis=1)]
    if len(kept) < 20:
        raise ValueError(f"insufficient robust depth for {name}")
    rectangle = cv2.minAreaRect(kept[:, :2].astype(np.float32))
    corners_xy = cv2.boxPoints(rectangle).astype(float)
    edges = np.roll(corners_xy, -1, axis=0) - corners_xy
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 0.005:
        raise ValueError(f"degenerate visible geometry for {name}")
    x = np.r_[edges[np.argmax(lengths)] / max(lengths), 0.0]
    if x[0] < 0:
        x = -x
    z = np.array([0., 0., 1.])
    center_xy = corners_xy.mean(axis=0)
    center = np.r_[center_xy, (lo[2] + hi[2]) / 2]
    return {"name": name, "position_base_m": center.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat().tolist(),
            "extent_m": [float(max(lengths)), float(min(lengths)), float(hi[2] - lo[2])],
            "corners_base_m": np.column_stack([corners_xy, np.full(4, lo[2])]).tolist(),
            "bottom_position_base_m": [*center_xy, float(lo[2])], "top_z_m": float(hi[2]),
            "geometry_source": "visible RGB-D envelope; occluded geometry is not measured",
            "valid_depth_points": len(kept)}

def inside_xy(target, points, margin=0.005):
    polygon = np.ascontiguousarray(np.asarray(target["corners_base_m"], dtype=np.float32)[:, :2])
    return all(cv2.pointPolygonTest(polygon, tuple(map(float, p[:2])), True) >= -margin for p in points)

class FiveToolRuntime(PlacementRuntime):
    # Keep this class name for compatibility with existing callers/fixtures.
    def __init__(self, cfg, run_dir, *, task, vision_generator=None, vision_timeout_s=300,
                 vision_call_limit=None, graspnet=None, **kwargs):
        super().__init__(cfg, run_dir, **kwargs)
        self.task = task
        self.graspnet = graspnet or ContactGraspNetClient(cfg.graspnet_url)
        self.descriptors = {d["name"]: d for d in task["objects"]}
        self.masks = {}
        self.vision_generator = vision_generator
        self.vision_timeout_s = vision_timeout_s
        self.vision_call_limit = vision_call_limit
        self.vision_model_calls = 0
        self._evaluation_observation = None
        self.collision_faults = []
        self.collision_pending = False

    def capture_feedback(self):
        """Fresh dual views for the action model, with no visual-description call."""
        self._stop_video()
        folder = self.run_dir / "observations" / f"feedback_{time.time_ns()}"
        packet = capture_observation(self.cfg, folder, snapshot=self.snapshot if self.dry_run else None,
                                     camera_fn=self.camera_fn)
        # Evaluate the same fresh views shown to the action model, without
        # another camera capture or any policy/skill text in the judge call.
        self._evaluation_observation = packet
        return packet

    def get_observation(self):
        """Capture front/wrist RGB-D data without invoking a model."""
        self._stop_video()  # Release the front camera before capturing either view.
        folder = self.run_dir / "observations" / f"dual_{time.time_ns()}"
        packet = capture_observation(self.cfg, folder, snapshot=self.snapshot if self.dry_run else None,
                                     camera_fn=self.camera_fn)
        return array_observation(folder, packet)

    def capture_scene(self):
        snapshot = self.snapshot
        if snapshot is not None and (Path(snapshot) / "front" / "meta.json").is_file():
            self.snapshot = Path(snapshot) / "front"
        try:
            result = super().capture_scene()
        finally:
            self.snapshot = snapshot
        self.masks.clear()
        return result

    def get_robot_state(self):
        # Evaluator/internal hardware state, not proof of object attachment.
        return self._ensure_arm().state()

    def _ensure_arm(self):
        try:
            arm = super()._ensure_arm()
        except PhysicalFault as exc:
            state = self.arm.state()
            if not collision_fault(state):
                raise
            self._record_collision(state, str(exc))
            raise RecoverableCollision(
                "FR3 collision reflex is still active; controller Recovery is required before any new motion. "
                "The agent may observe and regenerate; no failed motion was replayed. "
                f"current_errors={state.get('current_errors')}") from exc
        if self.collision_pending:
            self.collision_pending = False
            self._event(tool="collision_recovery_observed", state=arm.state(),
                        note="controller errors cleared; no automatic recovery or motion")
        # Attachment verification belongs to the policy/evaluator, not an
        # inferred task state inside a generic motion primitive.
        arm.monitor_grasp_state = False
        return arm

    def _record_collision(self, state, error):
        self.hardware_fault = False
        self.collision_pending = bool(state.get("has_errors"))
        record = {"error": error, "state": state, "pending_controller_recovery": self.collision_pending}
        self.collision_faults.append(record)
        self._write("collision_faults.json", self.collision_faults)
        self._write("robot_fault.json", state)
        self._event(tool="recoverable_collision", **record)

    def _diagnostic(self, record):
        super()._diagnostic(record)
        if record.get("event") != "failed":
            return
        state = record.get("failure_state", {})
        error = record.get("error", "")
        force_stop = "external force change " in error and " exceeded " in error
        if ((state.get("has_errors") and collision_fault(state)) or
                (force_stop and state.get("has_errors") is False)):
            self._record_collision(state, error)

    def _segment(self, prompt, side=None):
        candidates = sorted(self.sam3.segment_all(self.capture.rgb, prompt), key=lambda x: x[1], reverse=True)
        unique = []
        for mask, score in candidates:
            if not np.any(mask):
                continue
            if any(np.count_nonzero(mask & old) / max(1, np.count_nonzero(mask | old)) > 0.7 for old, _ in unique):
                continue
            unique.append((mask, score))
        if not unique:
            raise ValueError(f"SAM3 did not find {prompt}")
        if side:
            if len(unique) < 2:
                raise ValueError("left/right plate selection requires two distinct observed plates")
            # Image-left/right is the real-task language convention, shared by all methods.
            unique = sorted(unique[:2], key=lambda pair: np.where(pair[0])[1].mean())
            return unique[0 if side == "left" else -1]
        return unique[0]

    def _detect(self, name):
        # Free-form descriptions remain available for learned SAM3 prompt fixes.
        descriptor = self.descriptors.get(name, {"name": name, "prompt": name})
        mask, score = self._segment(descriptor.get("prompt", name), descriptor.get("side"))
        obj = describe_volume(self.capture, mask, self.cfg, name)
        self.masks[name] = mask
        obj.update(score=float(score), snapshot=str(self.capture_folder))
        self.objects[name] = obj
        self._write(f"observations/object_{time.time_ns()}.json", obj)
        return obj

    def get_object_pose(self, name):
        self.capture_scene()
        obj = self._detect(name)
        return tuple(np.array(obj[k]) for k in ("position_base_m", "quaternion_xyzw", "extent_m"))

    def sample_grasp_pose(self, name):
        self.capture_scene()
        descriptor = self.descriptors.get(name, {"prompt": name})
        mask, _ = self._segment(descriptor.get("prompt", name), descriptor.get("side"))
        self.masks[name] = mask
        candidates = self.graspnet.predict(self.capture, mask)
        artifact = self.run_dir / "observations" / f"contact_graspnet_{time.time_ns()}.npz"
        np.savez_compressed(artifact, **candidates)
        # The network uses an 80 mm Hand model; sensing must not claim FCI or
        # depend on the current open/closed/holding state. Actual limits are
        # checked by the hardware primitive when it is commanded.
        maximum = .08
        diagnostic = {
            "name": name, "snapshot": str(self.capture_folder), "model": "contact_graspnet_pytorch",
            "candidate_count": len(candidates["scores"]), "model_max_width_m": maximum,
            "widths_m": candidates["widths"].tolist(), "scores": candidates["scores"].tolist()}
        diagnostic_path = artifact.with_suffix(".json").relative_to(self.run_dir)
        try:
            transform, width, index = select_grasp(candidates["grasps"], candidates["scores"],
                candidates["widths"], self.cfg.camera_to_base, self.cfg.graspnet_to_ee,
                max_width_m=maximum)
        except ValueError as exc:
            self._write(diagnostic_path, {**diagnostic, "error": str(exc)})
            raise
        position = transform[:3, 3] + self.cfg.grasp_ee_offset_from_target_base_m
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self._write(diagnostic_path, {**diagnostic,
            "selected_index": index, "score": float(candidates["scores"][index]),
            "width_m": width, "T_base_ee": transform.tolist(),
            "T_camera_grasp": candidates["grasps"][index].tolist(),
            "T_grasp_ee": self.cfg.graspnet_to_ee.tolist()})
        return position, quaternion

    def goto_pose(self, position, quaternion):
        # Motion does not invoke cameras, segmentation or task-stage inference.
        before = len(self.collision_faults)
        try:
            result = super().goto_pose(position, quaternion, scene_preview=False)
        except RecoverableCollision:
            raise
        except Exception as exc:
            if len(self.collision_faults) > before and not self.hardware_fault:
                raise RecoverableCollision(
                    "FR3 collision stopped the current motion; observe the actual scene and revise subsequent code. "
                    f"pending_controller_recovery={self.collision_pending}; "
                    "the failed target was not reached and will not be replayed automatically. "
                    f"Original error: {exc}") from exc
            raise
        return {"success": bool(result["success"])}

    def close_gripper(self):
        # ASPIRE's no-argument primitive closes the hand independently of
        # perception. Allow contact anywhere inside the measured opening.
        arm = self._ensure_arm()
        self._start_video()
        self.goal_evidence = None
        if self.dry_run:
            arm.width, arm.held, ok = 0., True, True
        else:
            self.physical_motion_commanded = True
            ok = bool(arm.gripper.grasp(0., self.cfg.grasp_speed_m_s,
                                       self.cfg.grasp_force_n, self.cfg.grasp_epsilon_m,
                                       float(arm.gripper.max_width)))
        return {"success": ok}

    def open_gripper(self):
        result = super().open_gripper()
        return {"success": bool(result["success"])}

    def namespace(self):
        def wrap(name):
            def call(*args):
                if self.hardware_fault and name != "get_observation":
                    raise PhysicalFault("episode finished after hardware fault; only get_observation is allowed")
                started = time.monotonic()
                emit_progress(self.progress_path, tool=name, status="started", detail={
                    "prompt": args[0] if name in {"get_object_pose", "sample_grasp_pose"} and args else None,
                    "position": list(args[0]) if name == "goto_pose" and args else None})
                try:
                    value = getattr(self, name)(*args)
                except RecoverableCollision as exc:
                    record = {"tool": name, "ok": False, "recoverable_collision": True,
                              "pending_controller_recovery": self.collision_pending,
                              "error": f"{type(exc).__name__}: {exc}"}
                    self.trace.append(record)
                    self._event(**record)
                    emit_progress(self.progress_path, tool=name, status="error", error=record["error"])
                    raise
                except PhysicalFault as exc:
                    self.hardware_fault = True
                    record = {"tool": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self.trace.append(record)
                    self._event(**record)
                    emit_progress(self.progress_path, tool=name, status="error", error=record["error"])
                    raise
                except Exception as exc:
                    record = {"tool": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self.trace.append(record)
                    self._event(**record)
                    emit_progress(self.progress_path, tool=name, status="error", error=record["error"])
                    raise
                if name == "get_observation":
                    packet = {"views": {key: {k: v for k, v in view.items() if k != "images"}
                                        for key, view in value["views"].items()},
                              "capture_mode": value["capture_mode"]}
                else:
                    packet = _json_value(value)
                self.trace.append({"tool": name, "ok": True, "result": packet})
                self._event(tool=name, status="done", result=packet)
                emit_progress(self.progress_path, tool=name, status="done", elapsed_s=time.monotonic()-started,
                              detail=packet if isinstance(packet, dict) else {})
                return value
            return call
        return {name: wrap(name) for name in TOOLS}

    def _evaluate_visual_placement(self):
        observation = self._evaluation_observation or self.capture_feedback()
        self._evaluation_observation = None  # Direct repeated evaluations also need fresh views.
        folder = Path(observation["views"]["front"]["rgb_path"]).parent.parent
        result = {"success": False, "status": "UNSURE", "method": "vlm_reward",
                  "evaluator_version": "cup_on_plate_vlm.v1",
                  "dry_run": self.dry_run, "snapshot": str(folder),
                  "verification_source": "independent VLM; current front/wrist images"}
        (folder / "placement_verification_prompt.txt").write_text(placement_prompt(self.task))
        try:
            if self.vision_generator is None:
                raise RuntimeError("visual placement evaluation requires the campaign's visual model")
            if self.vision_call_limit is not None and self.vision_model_calls >= self.vision_call_limit:
                raise RuntimeError("observation model-call budget exhausted")
            self.vision_model_calls += 1
            self._write("vision_model_calls.json", {"calls": self.vision_model_calls})
            verdict = validate_placement_response(self.vision_generator.evaluate_placement(
                observation, self.task, timeout_s=self.vision_timeout_s), self.task)
            outcomes = [{**goal, **score, "success": score["status"] == "SUCCESS"}
                        for goal, score in zip(self.task["goals"], verdict["goals"])]
            status = ("SUCCESS" if all(g["success"] for g in outcomes) else
                      "FAILURE" if any(g["status"] == "FAILURE" for g in outcomes) else "UNSURE")
            result.update(status=status, success=status == "SUCCESS" and not self.dry_run,
                          goals=outcomes, summary=verdict["summary"])
        except Exception as exc:
            result.update(error=f"{type(exc).__name__}: {exc}",
                          goals=[{**goal, "success": False, "status": "UNSURE"} for goal in self.task["goals"]])
        self._write(folder.resolve().relative_to(self.run_dir.resolve()) / "placement_verification.json", result)
        self._write("verification.json", result)
        return result

    def evaluate_goal(self):
        """Evaluator-only fresh perception. Never enters policy namespace or target feedback."""
        if is_cup_plate_task(self.task):
            return self._evaluate_visual_placement()
        state = self.get_robot_state()
        self.capture_scene()
        objects = {d["name"]: self._detect(d["name"]) for d in self.task["objects"]}
        outcomes = []
        # Distinct named objects cannot all be credited from the same SAM3 mask.
        names = list(objects)
        ambiguous = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
                     if np.count_nonzero(self.masks[a] & self.masks[b]) /
                     max(1, np.count_nonzero(self.masks[a] | self.masks[b])) > 0.9]
        support = None
        if any(g["relation"] in {"in", "right_of"} for g in self.task["goals"]):
            mask, score = self._segment(self.cfg.board_prompt)
            support = describe_object(self.capture, mask, self.cfg, name="support surface", score=score)
        camera_right = self.cfg.camera_to_base[:3, 0].copy()
        camera_right[2] = 0
        camera_right /= np.linalg.norm(camera_right)
        for goal in self.task["goals"]:
            obj, target = objects[goal["object"]], objects[goal["target"]]
            center, target_center = np.array(obj["position_base_m"]), np.array(target["position_base_m"])
            bottom = obj["bottom_position_base_m"][2]
            if goal["relation"] == "on":
                passed = inside_xy(target, obj["corners_base_m"]) and abs(bottom-target["top_z_m"]) <= 0.025
            elif goal["relation"] == "in":
                support_height = float((np.asarray(obj["bottom_position_base_m"])-
                                        np.asarray(support["position_base_m"])) @ np.asarray(support["normal_base"]))
                passed = (inside_xy(target, obj["corners_base_m"]) and bottom < target["top_z_m"] - 0.005
                          and support_height >= -0.025)
            elif goal["relation"] == "right_of":
                delta = center - target_center
                along = float(delta @ camera_right)
                across = np.linalg.norm(delta[:2] - along*camera_right[:2])
                support_height = float((np.asarray(obj["bottom_position_base_m"])-
                                        np.asarray(support["position_base_m"])) @ np.asarray(support["normal_base"]))
                passed = (along > (target["extent_m"][1]+obj["extent_m"][1])/2 - 0.005
                          and along < 0.30 and across < 0.15 and abs(support_height) <= 0.025)
            else:
                raise ValueError("unsupported goal relation")
            outcomes.append({**goal, "success": bool(passed)})
        released = not state["hand_is_grasped"] and state["hand_width_m"] >= 0.065
        result = {"success": bool(all(g["success"] for g in outcomes) and released and not self.dry_run and not ambiguous),
                  "released": released, "goals": outcomes, "snapshot": str(self.capture_folder),
                  "ambiguous_object_masks": ambiguous,
                  "dry_run": self.dry_run, "verification_source": "fresh SAM3/RGB-D visible geometry"}
        self._write("verification.json", result)
        return result

def _json_value(value):
    import json
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))
