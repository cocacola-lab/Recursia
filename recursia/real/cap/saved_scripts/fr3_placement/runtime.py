"""ASPIRE-style primitive tools. This module does not sequence a task."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cap.agent.fr3_console import emit_progress
from cap.saved_scripts.fr3_runtime.camera import Capture, ColorVideoRecorder, capture_realsense
from cap.saved_scripts.fr3_runtime.hardware import active_fci_clients
from cap.saved_scripts.fr3_runtime.kinematics import preview_phase, solve_joint_pose
from cap.saved_scripts.fr3_runtime.robot import Fr3WipeRobot
from cap.saved_scripts.fr3_runtime.vision import Sam3Client

from .perception import describe_object, grasp_candidate, placement_geometry


class PhysicalFault(RuntimeError):
    """Robot/controller faults require recovery, rather than an agent retry."""


class DryArm:
    """Command preview only: never creates Franky or an FCI connection."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = np.eye(4)
        self.pose[:3, :3] = Rotation.from_quat([1, 0, 0, 0]).as_matrix()
        self.pose[:3, 3] = [0.45, 0, 0.25]
        self.width, self.held = 0.076, False

    def state(self):
        return {"q": [0, -0.5, 0, -1.5, 0, 2, 0], "dq": [0] * 7,
                "ee_pose_base": self.pose.tolist(), "hand_width_m": self.width, "hand_max_width_m": .08,
                "hand_is_grasped": self.held, "has_errors": False,
                "current_errors": "[]", "dry_run": True}

    def move_pose(self, step):
        from cap.saved_scripts.fr3_runtime.robot import MotionResult
        before = self.pose[:3, 3].tolist()
        self.pose[:3, 3] = step["ee_xyz_base_m"]
        self.pose[:3, :3] = Rotation.from_quat(step["ee_quat_xyzw"]).as_matrix()
        return MotionResult(step["stage"], before, self.pose[:3, 3].tolist(), 0, 0)

    def stop(self):
        pass


class PlacementRuntime:
    def __init__(self, cfg, run_dir: Path, *, dry_run=False, snapshot: Path | None = None,
                 progress_path: Path | None = None, arm_factory=None, camera_fn=None,
                 sam3=None, recorder_factory=None):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run, self.snapshot = dry_run, snapshot
        self.progress_path = progress_path
        self.arm_factory = arm_factory or self._connect_arm
        self.camera_fn = camera_fn or capture_realsense
        self.sam3 = sam3 or Sam3Client(cfg.segmentation_url)
        self.recorder_factory = recorder_factory or ColorVideoRecorder
        self.arm = self.video = self.capture = None
        self.objects: dict[str, dict] = {}
        self.held = None
        self.physical_motion_commanded = self.hardware_fault = False
        self.goal_evidence = None
        self.capture_count = self.video_count = self.move_count = 0
        self.trace: list[dict] = []
        previous = self.run_dir / "held_object.json"
        if previous.exists() and not dry_run:
            self.held = json.loads(previous.read_text())

    def _write(self, path, value):
        target = self.run_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")

    def _event(self, **record):
        record["host_monotonic_s"] = time.monotonic()
        with (self.run_dir / "trace.jsonl").open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _connect_arm(self, cfg):
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
            raise PermissionError("OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required")
        clients = active_fci_clients()
        if clients:
            raise RuntimeError(f"another FCI client is running: {clients}")
        import franky
        hand = franky.Gripper(cfg.robot_ip)
        robot = franky.Robot(cfg.robot_ip, realtime_config=franky.RealtimeConfig.Enforce,
                             relative_dynamics_factor=franky.RelativeDynamicsFactor(*cfg.relative_dynamics))
        return Fr3WipeRobot(cfg, robot, hand, franky)

    def _ensure_arm(self):
        if self.arm is None:
            self.arm = DryArm(self.cfg) if self.dry_run else self.arm_factory(self.cfg)
            if not self.dry_run:
                self.arm.monitor = lambda state: (
                    self.video.check() if self.video else None,
                    self._event(tool="robot_state", state=state))
                self.arm.diagnostics = self._diagnostic
        state = self.arm.state()
        if state.get("has_errors"):
            self.hardware_fault = True
            raise PhysicalFault(f"FR3 recovery required: {state.get('current_errors')}")
        return self.arm

    def _diagnostic(self, record):
        self._event(tool="motion_diagnostics", diagnostics=record)
        if record.get("event") == "completed" and record.get("state"):
            state = {**record["state"], "held_object": self.held}
            self._write("last_robot_state.json", state)
        if record.get("event") == "failed":
            state = record.get("failure_state", {})
            if state.get("has_errors"):
                self.hardware_fault = True
                self._write("robot_fault.json", state)

    def _start_video(self):
        if not self.dry_run and self.video is None:
            self.video_count += 1
            self.video = self.recorder_factory(
                self.cfg.external_serial, self.run_dir / "videos" / f"motion_{self.video_count:03d}.mp4")
            try:
                self.video.start()
            except BaseException:
                self.video.stop()
                self.video = None
                raise

    def _stop_video(self):
        if self.video is not None:
            self.video.stop()
            self.video = None

    def capture_scene(self):
        # Release the recording stream before opening aligned RGB-D. One owner
        # at a time, and no parent-process camera connection.
        self._stop_video()
        self.capture = (Capture.load(self.snapshot) if self.snapshot is not None
                        else self.camera_fn(self.cfg.external_serial))
        self.objects.clear()
        self.capture_count += 1
        folder = self.run_dir / "observations" / f"capture_{time.time_ns()}"
        self.capture.save(folder)
        self.capture_folder = folder
        return {"snapshot": str(folder), "rgb": str(folder / "rgb.png"),
                "capture_time": self.capture.captured_at, "frame": "FR3 base"}

    def get_camera_image(self):
        return self.capture_scene()

    def detect_object(self, object_name: str, *, refresh: bool = False):
        if self.capture is None or refresh:
            self.capture_scene()
        name = object_name.strip().lower()
        board = name in {"whiteboard", self.cfg.board_prompt.lower()}
        canonical = "whiteboard" if board else ("whiteboard eraser" if name in {
            "eraser", "whiteboard eraser", self.cfg.eraser_prompt.lower()} else name)
        if canonical in self.objects:
            return self.objects[canonical]
        prompt = self.cfg.board_prompt if board else (self.cfg.eraser_prompt if canonical == "whiteboard eraser" else object_name)
        candidates = sorted(self.sam3.segment_all(self.capture.rgb, prompt), key=lambda x: x[1], reverse=True)
        if not candidates:
            raise ValueError(f"SAM3 did not find {object_name}")
        mask, score = candidates[0]
        obj = describe_object(self.capture, mask, self.cfg, name=canonical, score=score)
        obj["snapshot"] = str(self.capture_folder)
        self.objects[canonical] = obj
        self._write(f"observations/detection_{time.time_ns()}.json", obj)
        overlay = self.capture.rgb.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(self.capture_folder / f"{canonical.replace(' ', '_')}_mask.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return obj

    def get_object_pose(self, object_name: str, return_bbox_extent: bool = False):
        obj = self.detect_object(object_name)
        return (np.array(obj["position_base_m"]), np.array(obj["quaternion_xyzw"]),
                np.array(obj["extent_m"]) if return_bbox_extent else None)

    def sample_grasp_pose(self, object_name: str = "whiteboard eraser"):
        obj = self.detect_object(object_name)
        state = self.get_robot_state()
        current = Rotation.from_matrix(np.array(state["ee_pose_base"])[:3, :3]).as_quat()
        return grasp_candidate(obj, current, self.cfg.grasp_ee_offset_from_target_base_m)

    def get_robot_state(self):
        state = self._ensure_arm().state()
        if state["hand_is_grasped"] and self.held is None:
            # A new task may start with the eraser already held. Expose the
            # station's provisional offset honestly, without reopening it.
            self.held = {"name": "whiteboard eraser",
                         "bottom_in_ee_m": self.cfg.fallback_bottom_in_ee_m.tolist(),
                         "offset_source": "provisional simulation offset; task starts with held eraser",
                         "grasp_width_m": state["hand_width_m"]}
            self._write("held_object.json", self.held)
        if not state["hand_is_grasped"] and self.held:
            self.held = None
            self._write("held_object.json", None)
        state["held_object"] = self.held
        self._write("last_robot_state.json", state)
        return state

    def get_held_object(self):
        return self.get_robot_state().get("held_object")

    def get_task_info(self):
        return {"success": bool(self.goal_evidence and self.goal_evidence.get("success")),
                "verification": self.goal_evidence, "dry_run": self.dry_run}

    def goto_pose(self, position, quaternion_xyzw, *, label="agent_move", scene_preview=True):
        position, quat = np.asarray(position, dtype=float), np.asarray(quaternion_xyzw, dtype=float)
        if position.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quat)):
            raise ValueError("goto_pose needs finite XYZ and XYZW")
        if abs(np.linalg.norm(quat) - 1) > 0.01:
            raise ValueError("quaternion must be unit length (XYZW)")
        if np.any(position < self.cfg.workspace_min_m) or np.any(position > self.cfg.workspace_max_m):
            raise ValueError("EE target outside station workspace")
        arm = self._ensure_arm()
        step = {"stage": str(label), "ee_xyz_base_m": position.tolist(), "ee_quat_xyzw": quat.tolist()}
        if not self.dry_run:
            board = self.objects.get("whiteboard") if scene_preview else None
            if scene_preview and board is None:
                raise ValueError("detect the whiteboard before moving so the arm/board preview has observed geometry")
            if self.cfg.pose_control == "ik_joint":
                preview = solve_joint_pose(self.cfg, board, arm.state(), step)
                step["joint_target_q"] = preview["target_q"]
            elif scene_preview:
                preview = preview_phase(self.cfg, {"plan_id": "agent_target", "board": board,
                                                   "staging": [step], "grasp": []}, "grasp", arm.state())
            else:
                preview = {"motor_command": "one CartesianWaypointMotion", "scene_preview": False}
            self._write(f"kinematics/move_{time.time_ns()}.json", preview)
        self._start_video()
        self.goal_evidence = None
        if not self.dry_run:
            self.physical_motion_commanded = True
        self.move_count += 1
        self._event(tool="goto_pose", status="commanded", target=step, dry_run=self.dry_run)
        try:
            result = arm.move_pose(step)
        except BaseException as exc:
            if self.hardware_fault:
                raise PhysicalFault(f"FR3 motion fault; see robot_fault.json: {exc}") from exc
            raise
        return {"success": True, "dry_run": self.dry_run, **asdict(result)}

    def open_gripper(self):
        arm = self._ensure_arm()
        self._start_video()
        self.goal_evidence = None
        if self.dry_run:
            arm.width, arm.held, ok = 0.076, False, True
        else:
            self.physical_motion_commanded = True
            # Releasing a held object is the purpose of this primitive.
            ok = bool(arm.gripper.move(float(arm.gripper.max_width), self.cfg.grasp_speed_m_s))
        if ok:
            self.held = None
            self._write("held_object.json", None)
        state = self.get_robot_state()
        return {"success": ok, "dry_run": self.dry_run, **state}

    def close_gripper(self, width_m: float, *, object_name="whiteboard eraser"):
        width_m = float(width_m)
        if not np.isfinite(width_m) or width_m < 0:
            raise ValueError("width_m must be finite and nonnegative")
        arm = self._ensure_arm()
        maximum = .08 if self.dry_run else float(arm.gripper.max_width)
        if not np.isfinite(maximum) or maximum <= 0 or width_m > maximum:
            raise ValueError(f"width_m must fit the measured hand opening (0..{maximum:.6f} m)")
        self._start_video()
        self.goal_evidence = None
        if self.dry_run:
            arm.width, arm.held, ok = width_m, True, True
        else:
            self.physical_motion_commanded = True
            ok = bool(arm.gripper.grasp(width_m, self.cfg.grasp_speed_m_s,
                                       self.cfg.grasp_force_n, self.cfg.grasp_epsilon_m,
                                       self.cfg.grasp_epsilon_m))
        state = arm.state()
        success = bool(ok and state["hand_is_grasped"])
        if success:
            obj = self.objects.get(object_name) or self.objects.get("whiteboard eraser")
            pose = np.asarray(state["ee_pose_base"])
            bottom = obj.get("bottom_position_base_m") if obj else None
            offset = (pose[:3, :3].T @ (np.asarray(bottom) - pose[:3, 3]) if bottom is not None
                      else self.cfg.fallback_bottom_in_ee_m)
            self.held = {"name": object_name, "bottom_in_ee_m": offset.tolist(),
                         "offset_source": "RGB-D support estimate at grasp" if bottom is not None
                         else "provisional simulation offset", "grasp_width_m": state["hand_width_m"]}
            self._write("held_object.json", self.held)
        # False is feedback: the agent can observe/reposition/retry explicitly.
        return {"success": success, "dry_run": self.dry_run, **state, "held_object": self.held,
                "epsilon_inner_m": self.cfg.grasp_epsilon_m, "epsilon_outer_m": self.cfg.grasp_epsilon_m}

    def verify_placement(self):
        state = self.get_robot_state()
        self.capture_scene()
        board = self.detect_object("whiteboard")
        eraser = self.detect_object("whiteboard eraser")
        evidence = placement_geometry(board, eraser)
        released = not state["hand_is_grasped"] and state["hand_width_m"] >= 0.065
        result = {"success": bool(evidence["on_board"] and released and not self.dry_run),
                  "released": released, "dry_run": self.dry_run, **evidence,
                  "snapshot": str(self.capture_folder), "board": board, "eraser": eraser}
        self.goal_evidence = result
        self._write("verification.json", result)
        return result

    def namespace(self):
        names = ("capture_scene", "get_camera_image", "detect_object", "get_object_pose",
                 "sample_grasp_pose", "get_robot_state", "get_held_object", "goto_pose",
                 "open_gripper", "close_gripper", "verify_placement", "get_task_info")
        def wrap(name):
            fn = getattr(self, name)
            def call(*args, **kwargs):
                started = time.monotonic()
                detail = {"label": kwargs.get("label", "agent_move"),
                          "position": list(args[0]) if name == "goto_pose" and args else None}
                if name in {"detect_object", "get_object_pose"} and args:
                    detail["prompt"] = args[0]
                emit_progress(self.progress_path, tool=name, status="started", detail=detail)
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    record = {"tool": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self.trace.append(record)
                    self._event(**record)
                    emit_progress(self.progress_path, tool=name, status="error", error=record["error"])
                    raise
                packet = json.loads(json.dumps(result, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))
                self.trace.append({"tool": name, "ok": True, "result": packet})
                self._event(tool=name, status="done", result=packet)
                if isinstance(result, dict):
                    detail.update({k: v for k, v in result.items() if k in {
                        "success", "after_xyz_m", "hand_is_grasped", "hand_width_m", "on_board"}})
                emit_progress(self.progress_path, tool=name, status="done", detail=detail,
                              elapsed_s=time.monotonic() - started)
                return result
            return call
        return {name: wrap(name) for name in names}

    def close(self):
        try:
            if self.arm is not None:
                self.arm.stop()
        finally:
            self._stop_video()
