"""Grasp-width consistency check (round width_gate_20260922): closed opening vs the object's width at the pinch band.

Evidence (all rounds, 63 body-pinch closes, width_gate_data.csv): every `verified` close whose object was never lifted closed to
1.1-1.8 cm on a body >= 2.7 cm wide (ratio measured/perceived 0.23-0.30: the pads pinched a top edge or air next to the body);
every truth-held close measured >= 0.55 x the mid-band width and >= 0.98 x the width at the pinch band. verify_grasp alone
(opening >= 0.06 = 5 mm) accepts the edge pinches.

Rule (public data only: the object's visible points, the actual TCP height after the descent, the closing axis of the actual
gripper orientation, and the public opening):
  band  = points with z in [z_tcp - tcp_to_fingertip, z_tcp - tcp_to_fingertip + pad_engage]   (where the pads are)
  w_ref = 3-97 percentile extent of the band projected on the closing axis (xy), if the band has >= MIN_BAND points
  measured = opening_after * max_opening
  verified  if measured >= RATIO_MIN * w_ref          (the pads closed on something as wide as the body there)
  failed    if measured <  RATIO_MIN * w_ref          (edge pinch / air)
  failed    if the band is empty (< MIN_BAND points) and its lower edge z_ft >= top - ABOVE_TOL, where top = 98th
            percentile z of the visible points: the pads are above the body -> edge pinch / air (the recorded pudding
            false positives: z_ft 0.050-0.060 vs top 0.046)
  uncertain if the band is empty but below the visible top (occluded lower part): no verdict, caller keeps the
            opening-only verifier.
RATIO_MIN = 0.5 and ABOVE_TOL = 0.005 were CHOSEN from the recorded closes above (dev data); fresh-state simulation is the
held-out check.
"""
from __future__ import annotations

import numpy as np

from grasp_geometry import GRIPPER

RATIO_MIN = 0.5
MIN_BAND = 12
ABOVE_TOL = 0.005


def closing_axis_xy(quat_wxyz):
    """Pad closing axis = tool y axis of the gripper orientation, projected on the table plane."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    # second column of the rotation matrix (tool y in world)
    ty = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    d = ty[:2]; n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else None


def width_at_pinch(pts, z_tcp, closing_dir):
    pts = np.asarray(pts, float)
    z_ft = float(z_tcp) - GRIPPER["tcp_to_fingertip"]
    band = pts[(pts[:, 2] >= z_ft) & (pts[:, 2] <= z_ft + GRIPPER["pad_engage"])]
    if len(band) < MIN_BAND or closing_dir is None:
        return None, int(len(band))
    proj = band[:, :2] @ np.asarray(closing_dir, float)
    return float(np.percentile(proj, 97) - np.percentile(proj, 3)), int(len(band))


def verify_width(pts, z_tcp, quat_wxyz, opening_after):
    try:
        a = float(opening_after)
    except (TypeError, ValueError):
        return {"status": "uncertain", "reason": "opening not observable"}
    d = closing_axis_xy(quat_wxyz)
    w_ref, n = width_at_pinch(pts, z_tcp, d)
    measured = a * GRIPPER["max_opening"]
    if w_ref is None:
        top = float(np.percentile(np.asarray(pts, float)[:, 2], 98)) if len(pts) else None
        z_ft = float(z_tcp) - GRIPPER["tcp_to_fingertip"]
        if top is not None and d is not None and z_ft >= top - ABOVE_TOL:
            return {"status": "failed", "detail": "pads_above_body", "reason": f"pinch band empty ({n} pts) and fingertips at {z_ft:.3f} >= visible top {top:.3f} - {ABOVE_TOL}: pads above the body",
                    "width_measured_m": round(measured, 4), "n_band": n, "top_m": round(top, 4), "z_ft_m": round(z_ft, 4)}
        return {"status": "uncertain", "reason": f"pinch band has {n} points (< {MIN_BAND}) below the visible top", "width_measured_m": round(measured, 4), "n_band": n, "top_m": None if top is None else round(top, 4)}
    ratio = measured / w_ref if w_ref > 1e-6 else None
    ok = ratio is not None and ratio >= RATIO_MIN
    return {"status": "verified" if ok else "failed", "detail": None if ok else "width_inconsistent",
            "reason": f"closed to {measured*100:.1f} cm vs {w_ref*100:.1f} cm body width at the pinch band (ratio {ratio if ratio is None else round(ratio, 2)})",
            "width_ref_m": round(w_ref, 4), "width_measured_m": round(measured, 4), "ratio": None if ratio is None else round(ratio, 3), "n_band": n}
