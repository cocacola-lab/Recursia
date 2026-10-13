"""Guarded FR3/Franka Hand executor for whiteboard plans.

This module is import-safe: the FCI connection is opened only by ``connect``.
The task has no generic ``exec`` or simulator access surface.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import WhiteboardConfig
from .hardware import active_fci_clients


# FR3 physical joint limits and the 0.052 rad guard margin used by
# other FCI control bridges. Cartesian motions need a measured-state
# guard too, because this task uses franky directly instead of the ROS mux.
FR3_Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
JOINT_MARGIN_RAD = 0.052


def _process_usage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {name: getattr(usage, name) for name in (
        "ru_utime", "ru_stime", "ru_nvcsw", "ru_nivcsw", "ru_minflt", "ru_majflt")}


def _usage_delta(before: dict[str, float | int]) -> dict[str, float | int]:
    after = _process_usage()
    return {name: after[name] - value for name, value in before.items()}


def _host_fault_snapshot() -> dict[str, Any]:
    """Read host counters only after motion has stopped."""
    result: dict[str, Any] = {"network_interfaces": {}, "realtime_threads": []}
    try:
        for interface in Path("/sys/class/net").iterdir():
            if interface.name == "lo":
                continue
            counters = {}
            for name in ("rx_errors", "rx_dropped", "tx_errors", "tx_dropped"):
                counters[name] = int((interface / "statistics" / name).read_text())
            result["network_interfaces"][interface.name] = counters
    except (OSError, ValueError) as exc:
        result["network_read_error"] = f"{type(exc).__name__}: {exc}"
    try:
        tasks = list(Path("/proc/self/task").iterdir())
        result["thread_count"] = len(tasks)
        for task in tasks:
            tid = int(task.name)
            try:
                policy = os.sched_getscheduler(tid)
                if policy not in (os.SCHED_FIFO, os.SCHED_RR):
                    continue
                result["realtime_threads"].append({
                    "tid": tid,
                    "name": (task / "comm").read_text().strip(),
                    "policy": "FIFO" if policy == os.SCHED_FIFO else "RR",
                    "priority": os.sched_getparam(tid).sched_priority,
                    "schedstat": (task / "schedstat").read_text().split(),
                })
            except (OSError, ValueError):
                continue
    except OSError as exc:
        result["thread_read_error"] = f"{type(exc).__name__}: {exc}"
    return result


@dataclass(frozen=True)
class MotionResult:
    stage: str
    before_xyz_m: list[float]
    after_xyz_m: list[float]
    peak_delta_force_n: float
    elapsed_s: float
    final_normal_force_n: float | None = None


class _FrankyCommandTrace:
    """Keep the first Franky commands in memory; never write from a callback."""

    def __init__(self, max_samples: int = 256):
        self.max_samples = max_samples
        self.samples: list[dict[str, Any]] = []
        self.callback_count = 0
        self.callback_error: str | None = None

    def __call__(self, state: Any, period: Any, relative_time: Any,
                 absolute_time: Any, command: Any) -> None:
        # Franky dispatches Python callbacks outside its real-time thread.
        # Bound conversion and memory use so diagnostics cannot grow with a run.
        self.callback_count += 1
        if len(self.samples) >= self.max_samples or self.callback_error:
            return
        try:
            self.samples.append({
                "period_s": float(period.to_sec()),
                "relative_time_s": float(relative_time.to_sec()),
                "absolute_time_s": float(absolute_time.to_sec()),
                "robot_time_s": float(state.time.to_sec()),
                # libfranka's column-major 4x4 Cartesian pose, before FCI.
                "franky_pose_col_major": [float(v) for v in command.O_T_EE],
                "robot_commanded_xyz_m": np.asarray(
                    state.O_T_EE_c.translation, dtype=float).ravel().tolist(),
                "control_command_success_rate": float(state.control_command_success_rate),
            })
        except Exception as exc:
            self.callback_error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, Any]:
        return {"source": "franky.CartesianWaypointMotion.register_callback",
                "callback_count": self.callback_count,
                "max_samples": self.max_samples,
                "samples": list(self.samples),
                "callback_error": self.callback_error,
                "note": "Python callback runs off the real-time thread; "
                        "sample arrival can lag FCI and is not proof a packet arrived at the robot"}


class Fr3WipeRobot:
    def __init__(self, cfg: WhiteboardConfig, robot: Any, gripper: Any,
                 franky_module: Any):
        self.cfg = cfg
        self.robot = robot
        self.gripper = gripper
        self.franky = franky_module
        self._contact_force_reference: np.ndarray | None = None
        self.monitor: Callable[[dict[str, Any]], Any] | None = None
        self.diagnostics: Callable[[dict[str, Any]], Any] | None = None

    def _diagnostic(self, **record: Any) -> None:
        # Diagnostic output must never replace a motion error or prevent stop.
        if self.diagnostics:
            try:
                self.diagnostics(record)
            except Exception:
                pass

    @classmethod
    def connect(cls, cfg: WhiteboardConfig) -> Fr3WipeRobot:
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
            raise PermissionError("OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required")
        clients = active_fci_clients()
        if clients:
            raise RuntimeError(f"another FCI bridge is running (PIDs {[x['pid'] for x in clients]}); "
                               "stop its stack before direct FR3 control")
        import franky

        # Connect to the independent hand port before claiming the sole FCI
        # arm session, so an unconfigured Franka Hand fails without an arm lock.
        gripper = franky.Gripper(cfg.robot_ip)
        robot = franky.Robot(
            cfg.robot_ip,
            # Use one quarter of the original arm velocity limit; keep acceleration and jerk factors.
            relative_dynamics_factor=franky.RelativeDynamicsFactor(0.05, 0.05, 0.04),
            realtime_config=franky.RealtimeConfig.Enforce,
        )
        return cls(cfg, robot, gripper, franky)

    @staticmethod
    def _check_joint_margin(state: dict[str, Any]) -> None:
        q = np.asarray(state["q"], dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise RuntimeError("invalid FR3 joint state")
        if np.any(q <= FR3_Q_MIN + JOINT_MARGIN_RAD) or np.any(q >= FR3_Q_MAX - JOINT_MARGIN_RAD):
            indices = np.flatnonzero((q <= FR3_Q_MIN + JOINT_MARGIN_RAD)
                                     | (q >= FR3_Q_MAX - JOINT_MARGIN_RAD)) + 1
            raise RuntimeError(f"FR3 joint margin reached: joints {indices.tolist()}")

    def state(self) -> dict[str, Any]:
        state_started = time.monotonic()
        arm = self.robot.state
        arm_read_end = time.monotonic()
        hand = self.gripper.state
        hand_read_end = time.monotonic()
        pose = np.asarray(arm.O_T_EE.matrix, dtype=np.float64)
        result = {"q": np.asarray(arm.q, dtype=np.float64).tolist(),
                "dq": np.asarray(arm.dq, dtype=np.float64).tolist(),
                "ee_pose_base": pose.tolist(),
                "external_force_base_n": np.asarray(arm.O_F_ext_hat_K, dtype=np.float64).ravel()[:3].tolist(),
                "hand_width_m": float(hand.width),
                "hand_max_width_m": float(hand.max_width),
                "hand_is_grasped": bool(hand.is_grasped),
                "current_errors": str(getattr(arm, "current_errors", "unavailable")),
                "last_motion_errors": str(getattr(arm, "last_motion_errors", "unavailable")),
                "has_errors": bool(self.robot.has_errors)}
        # Read the same cached arm snapshot; no extra FCI session or RT callback.
        for name in ("q_d", "dq_d", "ddq_d", "tau_J", "tau_J_d",
                     "tau_ext_hat_filtered", "joint_contact", "joint_collision",
                     "cartesian_contact", "cartesian_collision"):
            value = getattr(arm, name, None)
            result[name] = None if value is None else np.asarray(value).ravel().tolist()
        for name in ("O_T_EE_c", "O_T_EE_d"):
            value = getattr(arm, name, None)
            result[name] = None if value is None else np.asarray(value.matrix).tolist()
        for name in ("O_dP_EE_c", "O_ddP_EE_c"):
            value = getattr(arm, name, None)
            result[name] = (None if value is None else {
                "linear": np.asarray(value.linear).ravel().tolist(),
                "angular": np.asarray(value.angular).ravel().tolist(),
            })
        for name in ("elbow", "elbow_c", "elbow_d"):
            value = getattr(arm, name, None)
            result[name] = (None if value is None else {
                "joint_3_pos": float(value.joint_3_pos),
                "joint_4_flip": str(value.joint_4_flip),
            })
        for name in ("delbow_c", "ddelbow_c"):
            value = getattr(arm, name, None)
            result[name] = None if value is None else float(value)
        result["robot_mode"] = str(getattr(arm, "robot_mode", "unavailable"))
        result["control_command_success_rate"] = getattr(arm, "control_command_success_rate", None)
        stamp = getattr(arm, "time", None)
        result["robot_time_s"] = stamp.to_sec() if stamp is not None else None
        result["host_monotonic_s"] = time.monotonic()
        result["arm_snapshot_host_monotonic_s"] = arm_read_end
        result["gripper_snapshot_host_monotonic_s"] = hand_read_end
        result["state_access_timing_s"] = {
            "arm": arm_read_end - state_started,
            "gripper": hand_read_end - arm_read_end,
            "serialization": result["host_monotonic_s"] - hand_read_end,
        }
        if result["q_d"] is not None:
            result["joint_tracking_error_rad"] = (
                np.asarray(result["q_d"]) - np.asarray(arm.q).ravel()).tolist()
        return result

    def _ready(self, *, allow_grasped: bool = True) -> dict[str, Any]:
        state = self.state()
        self._check_finite_state(state)
        if state["has_errors"]:
            raise RuntimeError("FR3 reports an error; resolve in Desk")
        self._check_joint_margin(state)
        if max(abs(v) for v in state["dq"]) > 0.02:
            raise RuntimeError("FR3 is still moving")
        if not allow_grasped and state["hand_is_grasped"]:
            raise RuntimeError("the hand already holds an object")
        return state

    @staticmethod
    def _check_finite_state(state: dict[str, Any]) -> None:
        for name, shape in (("dq", (7,)), ("external_force_base_n", (3,)),
                            ("ee_pose_base", (4, 4))):
            value = np.asarray(state[name], dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise RuntimeError(f"invalid FR3 {name} state")

    def _run_motion(self, motion: Any, *, stage: str, timeout_s: float,
                    force_limit_n: float,
                    force_reference: np.ndarray | None = None,
                    contact_normal: np.ndarray | None = None,
                    motion_parameters: dict[str, Any] | None = None) -> MotionResult:
        before = self._ready()
        before_xyz = np.asarray(before["ee_pose_base"])[:3, 3]
        baseline = (np.asarray(before["external_force_base_n"])
                    if force_reference is None else force_reference)
        # Franka's estimated external wrench drifts with arm configuration.
        # Track that slow drift during non-contact free-space motion while
        # preserving a fixed reference for contact/probe force measurement.
        adaptive_baseline = force_reference is None and contact_normal is None
        peak = 0.0
        started = time.monotonic()
        usage_before = _process_usage()
        current = before
        previous_sample = started
        last_timing: dict[str, float | None] = {}
        command_trace = None
        # The 1 kHz callback is useful for a targeted trace, but may itself
        # add host load. Keep normal trials comparable to pre-trace runs.
        if (os.environ.get("ASPIRE_FRANKY_COMMAND_TRACE") == "1"
                and stage in ("pregrasp", "grasp")
                and hasattr(motion, "register_callback")):
            command_trace = _FrankyCommandTrace()
            try:
                motion.register_callback(command_trace)
            except Exception as exc:
                self._diagnostic(event="franky_trace_unavailable", stage=stage,
                                 error=f"{type(exc).__name__}: {exc}")
                command_trace = None
        factor = getattr(self.robot, "relative_dynamics_factor", None)
        motion_factor = getattr(motion, "relative_dynamics_factor", None)
        waypoint_factors = [getattr(waypoint, "relative_dynamics_factor", None)
                            for waypoint in getattr(motion, "waypoints", [])]
        factor_values = lambda item: {key: getattr(item, key, None)
                                      for key in ("velocity", "acceleration", "jerk")}
        self._diagnostic(event="start", stage=stage, motion_type=type(motion).__name__,
                         timeout_s=timeout_s, force_limit_n=force_limit_n,
                         adaptive_force_baseline=adaptive_baseline,
                         global_dynamics=factor_values(factor),
                         motion_dynamics=factor_values(motion_factor),
                         waypoint_dynamics=[factor_values(item) for item in waypoint_factors],
                         parameters=motion_parameters, state=before,
                         sampling_note="Python polling; Franky callback trace is emitted separately",
                         franky_command_trace_enabled=command_trace is not None)
        try:
            if self.monitor:
                op_started = time.monotonic()
                try:
                    self.monitor(before)
                finally:
                    last_timing["initial_monitor_s"] = time.monotonic() - op_started
            op_started = time.monotonic()
            try:
                self.robot.move(motion, asynchronous=True)
            finally:
                last_timing["move_submit_s"] = time.monotonic() - op_started
            while True:
                op_started = time.monotonic()
                try:
                    finished = self.robot.poll_motion()
                finally:
                    last_timing["poll_motion_s"] = time.monotonic() - op_started
                op_started = time.monotonic()
                try:
                    current = self.state()
                finally:
                    last_timing["state_read_s"] = time.monotonic() - op_started
                now = time.monotonic()
                current["motion_stage"] = stage
                current["motion_elapsed_s"] = now - started
                current["host_sample_interval_s"] = now - previous_sample
                current["host_call_timing_s"] = dict(last_timing)
                current["motion_finished"] = bool(finished)
                current["force_reference_base_n"] = baseline.tolist()
                current["force_delta_n"] = float(np.linalg.norm(
                    np.asarray(current["external_force_base_n"]) - baseline))
                previous_sample = now
                self._check_finite_state(current)
                self._check_joint_margin(current)
                if (getattr(self, "monitor_grasp_state", True)
                        and before["hand_is_grasped"] and not current["hand_is_grasped"]):
                    raise RuntimeError("eraser lost during motion")
                if self.monitor:
                    op_started = time.monotonic()
                    try:
                        self.monitor(current)
                    finally:
                        last_timing["monitor_s"] = time.monotonic() - op_started
                force = np.asarray(current["external_force_base_n"])
                peak = max(peak, float(np.linalg.norm(force - baseline)))
                if peak > force_limit_n:
                    raise RuntimeError(f"external force change {peak:.1f} N exceeded {force_limit_n:.1f} N")
                if adaptive_baseline:
                    baseline = baseline * 0.95 + force * 0.05
                if current["has_errors"]:
                    raise RuntimeError(
                        f"FR3 reported an error during {stage}: "
                        f"current_errors={current.get('current_errors', 'unavailable')}; "
                        f"last_motion_errors={current.get('last_motion_errors', 'unavailable')}")
                if time.monotonic() - started > timeout_s:
                    raise TimeoutError(f"{stage} exceeded {timeout_s:.1f} seconds")
                if finished:
                    break
                op_started = time.monotonic()
                time.sleep(0.02)
                last_timing["sleep_s"] = time.monotonic() - op_started
            if not self.robot.join_motion(timeout=1.0):
                raise TimeoutError(f"{stage} motion did not join")
            after = self._ready()
            if (getattr(self, "monitor_grasp_state", True)
                    and before["hand_is_grasped"] and not after["hand_is_grasped"]):
                raise RuntimeError("eraser lost at motion completion")
            delta_force = np.asarray(after["external_force_base_n"]) - baseline
            peak = max(peak, float(np.linalg.norm(delta_force)))
            if peak > force_limit_n:
                raise RuntimeError(f"external force change {peak:.1f} N exceeded {force_limit_n:.1f} N")
            normal_force = None if contact_normal is None else float(delta_force @ contact_normal)
            if normal_force is not None and normal_force < self.cfg.min_contact_force_n:
                raise RuntimeError(f"no confirmed board contact: normal force {normal_force:.2f} N "
                                   f"below {self.cfg.min_contact_force_n:.2f} N")
        except BaseException as exc:
            # poll_motion itself can throw before the usual sample is logged.
            try:
                failure_state = self.state()
            except Exception as read_error:
                failure_state = {"read_error": str(read_error)}
            try:
                self.robot.stop()
            finally:
                self._diagnostic(event="failed", stage=stage,
                                 error=f"{type(exc).__name__}: {exc}",
                                 elapsed_s=time.monotonic() - started,
                                 last_sample=current, failure_state=failure_state,
                                 peak_delta_force_n=peak,
                                 host_call_timing_s=dict(last_timing),
                                 process_usage_delta=_usage_delta(usage_before),
                                 host_fault_snapshot=_host_fault_snapshot())
            raise
        finally:
            if command_trace is not None:
                self._diagnostic(event="franky_command_trace", stage=stage,
                                 trace=command_trace.snapshot())
        after_xyz = np.asarray(after["ee_pose_base"])[:3, 3]
        self._diagnostic(event="completed", stage=stage, state=after,
                         elapsed_s=time.monotonic() - started, peak_delta_force_n=peak,
                         host_call_timing_s=dict(last_timing),
                         process_usage_delta=_usage_delta(usage_before))
        return MotionResult(stage, before_xyz.tolist(), after_xyz.tolist(), peak,
                            time.monotonic() - started, normal_force)

    def move_pose(self, step: dict[str, Any], *, contact: bool = False,
                  probe: bool = False) -> MotionResult:
        cfg = self.cfg
        xyz = np.asarray(step["ee_xyz_base_m"], dtype=np.float64)
        quat = np.asarray(step["ee_quat_xyzw"], dtype=np.float64)
        if xyz.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)):
            raise ValueError("invalid EE pose")
        if abs(np.linalg.norm(quat) - 1.0) > 0.01:
            raise ValueError("EE quaternion is not unit length")
        if np.any(xyz < cfg.workspace_min_m) or np.any(xyz > cfg.workspace_max_m):
            raise ValueError("EE target outside configured workspace")
        before = self._ready()
        from scipy.spatial.transform import Rotation

        target = np.eye(4)
        target[:3, :3] = Rotation.from_quat(quat).as_matrix()
        target[:3, 3] = xyz
        if contact:
            normal = np.asarray(step.get("board_normal_base"), dtype=float)
            if normal.shape != (3,) or not np.all(np.isfinite(normal)) or abs(np.linalg.norm(normal) - 1) > 0.01:
                raise ValueError("contact step requires a measured unit board normal")
            if self._contact_force_reference is None:
                # Keep a fixed force reference while the pad contacts the board.
                self._contact_force_reference = np.asarray(
                    before["external_force_base_n"], dtype=np.float64)
            limit = cfg.force_limit_n
        else:
            grasp_descent = step["stage"] == "grasp"
            # Respect the selected stage's configured threshold; no hidden 12 N cap.
            limit = cfg.grasp_force_limit_n if grasp_descent else cfg.force_limit_n
        joint_target = step.get("joint_target_q")
        if getattr(cfg, "pose_control", "cartesian") == "ik_joint":
            q = np.asarray(joint_target, dtype=float)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise ValueError("ik_joint requires a previewed finite seven-joint target")
            self._check_joint_margin({"q": q})
            motion = self.franky.JointMotion(q.tolist())
        else:
            waypoint = self.franky.CartesianWaypoint(
                self.franky.Affine(target), self.franky.ReferenceType.Absolute,
            )
            motion = self.franky.CartesianWaypointMotion([waypoint])
        result = self._run_motion(motion, stage=str(step["stage"]),
                                  timeout_s=25.0,
                                  force_limit_n=limit,
                                  force_reference=self._contact_force_reference if contact else None,
                                  contact_normal=normal if contact else None,
                                  motion_parameters={"target": step, "contact": contact,
                                      "minimum_time_ms": None,
                                      "impedance_duration_ms": None})
        error = np.linalg.norm(np.asarray(result.after_xyz_m) - xyz)
        if error > (0.012 if contact else 0.008):
            raise RuntimeError(f"EE position error after {step['stage']} is {error:.3f} m")
        final_rotation = Rotation.from_matrix(np.asarray(self.state()["ee_pose_base"])[:3, :3])
        orientation_error = float((final_rotation.inv() * Rotation.from_quat(quat)).magnitude())
        if orientation_error > np.deg2rad(5.0):
            raise RuntimeError(f"EE orientation error after {step['stage']} is "
                               f"{np.rad2deg(orientation_error):.1f} degrees")
        return result

    def move_joints(self, target_q: np.ndarray, *, stage: str = "joint_move") -> dict[str, Any]:
        """Run one slow, force-monitored joint motion to a planned endpoint."""
        target = np.asarray(target_q, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("joint target must contain seven finite values")
        if np.any(target <= FR3_Q_MIN + JOINT_MARGIN_RAD) or np.any(
                target >= FR3_Q_MAX - JOINT_MARGIN_RAD):
            raise ValueError("joint target is outside the guarded FR3 limits")
        motion = self.franky.JointMotion(target.tolist())
        result = self._run_motion(motion, stage=stage, timeout_s=60.0,
                                  force_limit_n=min(12.0, self.cfg.force_limit_n * 1.5),
                                  motion_parameters={"target_q": target.tolist()})
        after = self.state()
        error = float(np.max(np.abs(np.asarray(after["q"], dtype=float) - target)))
        if error > 0.01:
            raise RuntimeError(f"joint error after {stage} is {error:.4f} rad")
        return {"motion": result.__dict__, "target_q": target.tolist(),
                "after_q": after["q"], "max_joint_error_rad": error}

    def open_hand(self) -> dict[str, Any]:
        self._ready(allow_grasped=False)
        width = float(self.gripper.max_width)
        if not self.gripper.move(width, self.cfg.grasp_speed_m_s):
            raise RuntimeError("Franka Hand open command failed")
        return self.state()

    def grasp(self) -> dict[str, Any]:
        self._ready(allow_grasped=False)
        # The grasp width is estimated from RGB-D; allow 10 mm on each side.
        epsilon_inner_m = epsilon_outer_m = 0.010
        confirmed = bool(self.gripper.grasp(
            self.cfg.grasp_width_m, self.cfg.grasp_speed_m_s,
            self.cfg.grasp_force_n, epsilon_inner_m, epsilon_outer_m))
        state = self.state()
        self._diagnostic(event="grasp_result", stage="grasp_hand",
                         status="done" if confirmed else "failed",
                         target_width_m=self.cfg.grasp_width_m,
                         epsilon_inner_m=epsilon_inner_m, epsilon_outer_m=epsilon_outer_m,
                         measured_width_m=state["hand_width_m"],
                         hand_is_grasped=state["hand_is_grasped"],
                         command_success=confirmed)
        if not confirmed:
            raise RuntimeError(
                "Franka Hand did not confirm an eraser grasp: "
                f"target={self.cfg.grasp_width_m:.4f} m, "
                f"measured={state['hand_width_m']:.4f} m, tolerance=+/-0.010 m, "
                f"is_grasped={state['hand_is_grasped']}")
        if not state["hand_is_grasped"]:
            raise RuntimeError("Franka Hand state does not confirm a held eraser")
        return state

    def stop(self) -> None:
        if self.robot.is_in_control:
            self.robot.stop()

    def __enter__(self) -> Fr3WipeRobot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
