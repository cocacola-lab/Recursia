"""G2: ONE shared body-pinch geometry change on top of the frozen module B (round zeroshot_general_grasp_contract_20260918_OxvXZB).

Evidence (parent round, offline truth vs online labels; results/pudding_failure_analysis.json):
  * E2 body-pinch descents end within the contact primitive's 1 cm arrival tolerance ABOVE the planned TCP height (pos_err
    0.96 cm on the failed cell). Module B places the fingertips 1.0 cm above the perceived-base upper bound
    (`base_hi + 0.010 + tcp_to_fingertip`), so for a ~2.5 cm tall body the nominal pad/body overlap is ~1.4 cm and the ACTUAL
    overlap after the arrival tolerance was 0.46 cm on the failed cell (fingertip 0.5 cm below the true top) versus
    0.85 / 1.1 cm on the two successful cells; the failed close pinched the top edge (opening 0.134 = 1.1 cm "width" for a
    4 cm body; the object tilted 4 mm up and slipped out during the lift, truth: no gripper contact after the lift).
  * The MC reference (E0) descends to a fixed low height, meets the support inside the contact window and closes with full
    pad engagement (opening 0.57 = 4.6 cm, consistent with the 4 cm body): 3/3 on the same states.
Rule (body_pinch only; rim_pinch and infeasible untouched; no object/task/scene/seed word):
  fingertips aimed DEPTH_CLEAR above the perceived-base upper bound instead of 1 cm above it; the palm-clearance bound for
  tall bodies is unchanged (max with z_palm). The contact-seeking descent is unchanged: if the support is met first, the
  contact is inside its window (contact_tol_before 3 cm) -> contact_established -> close, exactly the MC reference behaviour.
  The overshoot end (target - CONTACT_OVERSHOOT) must stay inside the public IK box, so the height is floored at
  IK_FLOOR + CONTACT_OVERSHOOT (a candidate below it would be refused by the primitive without motion).
Effect: nominal pad/body overlap grows by 0.8 cm; the arrival tolerance can no longer consume the whole engagement of a short body.
"""
from __future__ import annotations

import numpy as np

import grasp_geometry as GG
from grasp_geometry import GRIPPER

DEPTH_CLEAR = 0.002        # m, fingertips above the perceived-base upper bound (was 0.010 in module B)
PALM_CLEAR = 0.005         # m, unchanged palm clearance (fusion_geometry.PALM_CLEAR)
IK_FLOOR = 0.005           # m, public IK box floor (motion_interface_r.IK_WORKSPACE_LO[2])
CONTACT_OVERSHOOT = 0.01   # m, MC caller overshoot (program header CONTACT_OVERSHOOT)


def body_pinch_depth_z(prof):
    z_palm = float(prof["top"] - (GRIPPER["finger_len"] - PALM_CLEAR))
    z_deep = float(prof["base_hi"] + DEPTH_CLEAR + GRIPPER["tcp_to_fingertip"])
    return max(z_palm, z_deep, IK_FLOOR + CONTACT_OVERSHOOT + 1e-4)


def plan_grasp(pts, cam_xy=None, yaws=(0, 90)):
    plan = GG.plan_grasp(pts, cam_xy, yaws); plan["rule"] = "module_B"
    if plan.get("mode") != "body_pinch" or plan.get("profile") is None or plan.get("target") is None:
        return plan
    prof = plan["profile"]; z_b = float(plan["target"][2]); z_new = body_pinch_depth_z(prof)
    plan["z_module_B"] = z_b
    if z_new >= z_b - 1e-6:                 # module B is already at least as deep (palm clearance dominates): keep it
        plan["rule"] = "module_B_kept"; return plan
    z_ft = z_new - GRIPPER["tcp_to_fingertip"]
    tgt = np.asarray(plan["target"], float).copy(); tgt[2] = z_new
    plan.update({"target": tgt, "rule": "G2_body_pinch_depth", "hang_est": float(z_new - prof["base_est"]), "hang_interval": [float(z_new - prof["base_hi"]), float(z_new - prof["base_lo"])],
                 "pad_engaged_m": float(min(prof["top"], z_ft + GRIPPER["pad_engage"]) - max(prof["base_est"], z_ft)), "expected": "reached",
                 "predicted_contact_tcp_z": None, "contact_range": [float(prof["base_lo"] + GRIPPER["tcp_to_fingertip"]), float(prof["base_hi"] + GRIPPER["tcp_to_fingertip"])]})
    return plan
