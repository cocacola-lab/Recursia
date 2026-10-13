"""FR3 forward kinematics and damped-least-squares inverse kinematics.

The bundled arm model resolves the flange at ``attachment_site``. A measured
joint configuration and end-effector pose calibrate the rigid flange-to-TCP
transform. A static Franka Hand TCP is also provided for offline use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Franka Hand default TCP (Desk default): +103.4 mm along flange z, -45 deg.
DEFAULT_HAND_TCP_XYZ = [0.0, 0.0, 0.1034]
DEFAULT_HAND_TCP_RPY = [0.0, 0.0, -0.7853981633974483]


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "model" / "fr3.xml"


def tcp_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()
    tcp[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return tcp


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    pos_err_m: float
    rot_err_rad: float
    iters: int


class ArmKinematics:
    def __init__(self, xml_path: str | Path | None = None,
                 site: str = "attachment_site", limit_margin_rad: float = 0.02):
        import mujoco  # deferred: replay path never needs mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path or default_model_path()))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 7:
            raise ValueError(f"expected 7-dof arm model, got nq={self.model.nq}")
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self._site_id < 0:
            raise ValueError(f"site {site!r} not in model")
        rng = self.model.jnt_range.copy()
        self.q_min = rng[:, 0] + limit_margin_rad
        self.q_max = rng[:, 1] - limit_margin_rad

    def fk(self, q: np.ndarray, tcp: np.ndarray | None = None) -> np.ndarray:
        """4x4 pose of the flange site (or flange @ tcp) in the base frame."""
        mj = self._mujoco
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)
        mj.mj_kinematics(self.model, self.data)
        out = np.eye(4)
        out[:3, :3] = self.data.site_xmat[self._site_id].reshape(3, 3)
        out[:3, 3] = self.data.site_xpos[self._site_id]
        return out if tcp is None else out @ tcp

    def calibrate_tcp(self, q: np.ndarray, ee: np.ndarray) -> np.ndarray:
        """T_tcp such that fk(q) @ T_tcp == ee (one rigid constant)."""
        flange = self.fk(q)
        inv = np.eye(4)
        r = flange[:3, :3]
        inv[:3, :3] = r.T
        inv[:3, 3] = -r.T @ flange[:3, 3]
        return inv @ np.asarray(ee, dtype=np.float64).reshape(4, 4)

    def _site_jacobian(self) -> np.ndarray:
        """6x7 world-frame jacobian [linear; angular] at the site (assumes
        self.data holds the kinematics of the q of interest)."""
        mj = self._mujoco
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_comPos(self.model, self.data)  # jacobians need subtree COMs
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id)
        return np.vstack([jacp, jacr])

    def solve_ik(self, q0: np.ndarray, target: np.ndarray,
                 tcp: np.ndarray | None = None, damping: float = 0.05,
                 max_iters: int = 8, tol_pos_m: float = 1e-3,
                 tol_rot_rad: float = 5e-3,
                 max_dq_per_iter: float = 0.2) -> IKResult:
        """DLS IK for the TCP toward `target` (4x4 base frame), seeded at q0."""
        # Solve at the flange: site_target = target @ tcp^-1.
        target = np.asarray(target, dtype=np.float64).reshape(4, 4)
        if tcp is not None:
            tcp_inv = np.eye(4)
            r = tcp[:3, :3]
            tcp_inv[:3, :3] = r.T
            tcp_inv[:3, 3] = -r.T @ tcp[:3, 3]
            target = target @ tcp_inv

        q = np.clip(np.asarray(q0, dtype=np.float64).copy(), self.q_min, self.q_max)
        lam2 = float(damping) ** 2
        pos_err = rot_err = float("inf")
        it = 0
        for it in range(1, max_iters + 1):
            cur = self.fk(q)
            e_pos = target[:3, 3] - cur[:3, 3]
            e_rot = Rotation.from_matrix(
                target[:3, :3] @ cur[:3, :3].T
            ).as_rotvec()
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            if pos_err < tol_pos_m and rot_err < tol_rot_rad:
                return IKResult(q, True, pos_err, rot_err, it)
            jac = self._site_jacobian()  # fk() above refreshed self.data
            err = np.concatenate([e_pos, e_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(6), err)
            step = float(np.max(np.abs(dq)))
            if step > max_dq_per_iter:
                dq *= max_dq_per_iter / step
            q = np.clip(q + dq, self.q_min, self.q_max)
        cur = self.fk(q)
        pos_err = float(np.linalg.norm(target[:3, 3] - cur[:3, 3]))
        rot_err = float(np.linalg.norm(
            Rotation.from_matrix(target[:3, :3] @ cur[:3, :3].T).as_rotvec()
        ))
        converged = pos_err < tol_pos_m and rot_err < tol_rot_rad
        return IKResult(q, converged, pos_err, rot_err, it)
