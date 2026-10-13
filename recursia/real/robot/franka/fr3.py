"""Small FR3 control surface using franky's jerk-limited joint motions.

This module has no YAM, camera, or leader-arm dependency. Constructing a client
connects to FCI and reads state; it never moves or recovers the robot on start.
Only one process may own the FCI arm connection at a time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# FR3 datasheet limits. Keep a margin
# inside the mechanical stops even for a deliberately small diagnostic move.
FR3_Q_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_Q_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
LIMIT_MARGIN_RAD = 0.052
MAX_STEP_RAD = 0.03
STILL_RAD_S = 0.02
PROBE_VELOCITY_FACTOR = 0.005
PROBE_ACCELERATION_FACTOR = 0.003
PROBE_JERK_FACTOR = 0.002


@dataclass(frozen=True)
class Fr3State:
    q: np.ndarray
    dq: np.ndarray
    ee: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {"q": self.q.tolist(), "dq": self.dq.tolist(), "ee": self.ee.tolist()}


class Fr3Client:
    """Own one FCI connection and expose state plus one bounded joint step."""

    def __init__(self, robot_ip: str, *, enforce_realtime: bool = True,
                 robot: Any | None = None, franky_module: Any | None = None):
        self._robot = robot
        self._franky = franky_module
        self._enforce_realtime = enforce_realtime
        self._closed = False
        if self._robot is None:
            if self._franky is None:
                import franky

                self._franky = franky
            self._robot = self._franky.Robot(
                robot_ip,
                relative_dynamics_factor=self._franky.RelativeDynamicsFactor(
                    PROBE_VELOCITY_FACTOR, PROBE_ACCELERATION_FACTOR, PROBE_JERK_FACTOR),
                realtime_config=(self._franky.RealtimeConfig.Enforce if enforce_realtime
                                 else self._franky.RealtimeConfig.Ignore),
            )

    def read_state(self) -> Fr3State:
        if self._closed:
            raise RuntimeError("FR3 connection is closed")
        state = self._robot.state
        q = np.asarray(state.q, dtype=np.float64).reshape(-1).copy()
        dq = np.asarray(state.dq, dtype=np.float64).reshape(-1).copy()
        ee = np.asarray(state.O_T_EE.matrix, dtype=np.float64).reshape(4, 4).copy()
        if q.shape != (7,) or dq.shape != (7,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise RuntimeError("FR3 returned invalid joint state")
        if not np.all(np.isfinite(ee)):
            raise RuntimeError("FR3 returned invalid end-effector pose")
        return Fr3State(q=q, dq=dq, ee=ee)

    def move_joint_step(self, joint: int, delta_rad: float) -> Fr3State:
        """Blocking one-joint move, at most 0.03 rad from a stationary pose."""
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
            raise PermissionError("Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 to enable physical motion")
        if not self._enforce_realtime:
            raise RuntimeError("physical motion requires enforced real-time scheduling")
        if self._franky is None:
            raise RuntimeError("franky motion API is unavailable")
        if not isinstance(joint, int) or not 1 <= joint <= 7:
            raise ValueError("joint must be an integer from 1 to 7")
        if not np.isfinite(delta_rad) or abs(delta_rad) > MAX_STEP_RAD or delta_rad == 0:
            raise ValueError(f"delta_rad must be nonzero and within ±{MAX_STEP_RAD} rad")
        state = self.read_state()
        if np.max(np.abs(state.dq)) > STILL_RAD_S:
            raise RuntimeError("FR3 is moving; wait for it to stop")
        if bool(self._robot.has_errors):
            raise RuntimeError("FR3 reports errors; resolve them in Desk before moving")
        target = state.q.copy()
        target[joint - 1] += delta_rad
        if np.any(target < FR3_Q_MIN + LIMIT_MARGIN_RAD) or np.any(target > FR3_Q_MAX - LIMIT_MARGIN_RAD):
            raise ValueError("target is too close to an FR3 joint limit")
        self._robot.move(self._franky.JointMotion(target.tolist()))
        return self.read_state()

    def stop(self) -> None:
        if not self._closed and self._robot.is_in_control:
            self._robot.stop()

    def close(self) -> None:
        if not self._closed:
            self.stop()
            self._closed = True

    def __enter__(self) -> Fr3Client:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
