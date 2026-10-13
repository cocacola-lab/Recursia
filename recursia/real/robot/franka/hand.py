"""Bounded Franka Hand exercise via the independent gripper port (1338)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_CYCLE_TRAVEL_M = 0.04
DEFAULT_SPEED_M_S = 0.03
SLOW_SPEED_M_S = 0.01


@dataclass(frozen=True)
class HandState:
    width_m: float
    max_width_m: float
    is_grasped: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {"width_m": self.width_m, "max_width_m": self.max_width_m,
                "is_grasped": self.is_grasped}


class Fr3Hand:
    """One connection to the Franka Hand; no arm FCI connection is opened."""

    def __init__(self, robot_ip: str, *, gripper: Any | None = None):
        if gripper is None:
            import franky

            gripper = franky.Gripper(robot_ip)
        self._gripper = gripper

    def read_state(self) -> HandState:
        state = self._gripper.state
        width = float(state.width)
        max_width = float(self._gripper.max_width)
        if not np.isfinite(width) or not np.isfinite(max_width) or max_width <= 0:
            raise RuntimeError("Franka Hand returned invalid width")
        return HandState(width, max_width, bool(state.is_grasped))

    def close_then_open(self, travel_m: float = 0.02, *,
                        speed_m_s: float = DEFAULT_SPEED_M_S,
                        hold_s: float = 0.0) -> tuple[HandState, HandState]:
        """Close by a bounded distance, optionally pause, then restore width."""
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
            raise PermissionError("Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 for hand motion")
        if not np.isfinite(travel_m) or not 0 < travel_m <= MAX_CYCLE_TRAVEL_M:
            raise ValueError(f"travel_m must be within (0, {MAX_CYCLE_TRAVEL_M}]")
        if not np.isfinite(speed_m_s) or not 0 < speed_m_s <= DEFAULT_SPEED_M_S:
            raise ValueError(f"speed_m_s must be within (0, {DEFAULT_SPEED_M_S}]")
        if not np.isfinite(hold_s) or not 0 <= hold_s <= 10:
            raise ValueError("hold_s must be within [0, 10]")
        before = self.read_state()
        if before.is_grasped:
            raise RuntimeError("Hand reports an object grasped; refusing the cycle")
        if before.width_m < travel_m + 0.005:
            raise RuntimeError("Hand is not open far enough for the requested cycle")
        close_width = before.width_m - travel_m
        try:
            if not self._gripper.move(close_width, speed_m_s):
                raise RuntimeError("Franka Hand close move was rejected")
            closed = self.read_state()
            if abs(closed.width_m - close_width) > 0.005:
                raise RuntimeError("Franka Hand did not reach the requested close width")
            if hold_s:
                time.sleep(hold_s)
        finally:
            # Reopening is the exit path even if the close command fails.
            reopened_ok = self._gripper.move(before.width_m, speed_m_s)
        opened = self.read_state()
        if not reopened_ok or abs(opened.width_m - before.width_m) > 0.005:
            raise RuntimeError("Franka Hand did not return to its original width")
        return closed, opened
