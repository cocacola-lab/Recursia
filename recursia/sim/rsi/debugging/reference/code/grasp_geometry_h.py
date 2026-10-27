"""H palm rule: body-pinch depth bounded by the MEASURED hand-collision clearance (round recheck_palm_20260922).

Evidence (outputs/zeroshot_binding_interface_20260922_qsSbSJ, ketchup, 15/15 first descents): perception fine (axis error
0.5 cm), but every descent is deflected 3-4 cm at TCP z = top - 0.0335 with the only robot/object contact being
ketchup vs the HAND body (finger pads never touch). Module B / G2 place the fingertips finger_len - PALM_CLEAR = 4.9 cm below
the top, assuming the palm is finger_len = 0.054 above the fingertips; in the MuJoCo model the hand collision box bottom is
0.0400 above the fingertip collision bottom and 0.0335 ABOVE the planner TCP (tcp_offsets.py, measured at the home pose).
So on any body taller than ~4.5 cm the planned grasp is unreachable: the hand hits the top first.

Rule (body_pinch only; rim_pinch / infeasible untouched; gripper constants only, no object/task/scene/seed word):
  TCP z >= top - HAND_CLEAR_ABOVE_TCP + PALM_CLEAR + CONTACT_OVERSHOOT
  i.e. even after the contact primitive's 1 cm overshoot the hand collision box stays PALM_CLEAR above the perceived top.
Applied on top of G2 (which may deepen short bodies): z = z_palm_H when z_palm_H - z_G2 > ACTIVATION_MIN, else z_G2.
ACTIVATION_MIN = PERCEPTION z_noise (4 mm): a raise smaller than the perception noise is not acted on, so short-body plans
(where the two bounds nearly coincide) stay byte-identical to G2 and the rule cannot flip on noise.
Constants:
  HAND_CLEAR_ABOVE_TCP 0.0335  measured (hand_collision bottom - TCP) at the home pose
  PALM_CLEAR           0.005   unchanged (grasp_geometry_g2.PALM_CLEAR)
  CONTACT_OVERSHOOT    0.010   the caller's overshoot_after (program CONTACT_OVERSHOOT)
  ACTIVATION_MIN       0.004   = grasp_geometry.PERCEPTION["z_noise"]
"""
from __future__ import annotations

import numpy as np

import grasp_geometry_g2 as G2
from grasp_geometry import GRIPPER, PERCEPTION

HAND_CLEAR_ABOVE_TCP = 0.0335
PALM_CLEAR = G2.PALM_CLEAR
CONTACT_OVERSHOOT = G2.CONTACT_OVERSHOOT
ACTIVATION_MIN = PERCEPTION["z_noise"]


def palm_floor_z(prof):
    return float(prof["top"] - HAND_CLEAR_ABOVE_TCP + PALM_CLEAR + CONTACT_OVERSHOOT)


def plan_grasp(pts, cam_xy=None, yaws=(0, 90)):
    plan = G2.plan_grasp(pts, cam_xy, yaws)
    if plan.get("mode") != "body_pinch" or plan.get("profile") is None or plan.get("target") is None:
        return plan
    prof = plan["profile"]; z_in = float(plan["target"][2]); z_p = palm_floor_z(prof)
    plan["z_before_palm_rule"] = z_in; plan["z_palm_H"] = z_p
    if z_p - z_in <= ACTIVATION_MIN:
        plan["palm_rule"] = "inactive"; return plan
    z_ft = z_p - GRIPPER["tcp_to_fingertip"]
    tgt = np.asarray(plan["target"], float).copy(); tgt[2] = z_p
    plan.update({"target": tgt, "rule": "H_palm_clearance", "palm_rule": "active",
                 "hang_est": float(z_p - prof["base_est"]), "hang_interval": [float(z_p - prof["base_hi"]), float(z_p - prof["base_lo"])],
                 "pad_engaged_m": float(min(prof["top"], z_ft + GRIPPER["pad_engage"]) - max(prof["base_est"], z_ft)),
                 "expected": "reached", "predicted_contact_tcp_z": None,
                 "contact_range": [float(prof["base_lo"] + GRIPPER["tcp_to_fingertip"]), float(prof["base_hi"] + GRIPPER["tcp_to_fingertip"])]})
    return plan
