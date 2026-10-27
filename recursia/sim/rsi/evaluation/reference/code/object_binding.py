"""Task-agnostic object binding from several text prompts (round binding_interface_20260922).

Replaces the legacy rule "first prompt that returns a mask, highest-score mask" (localize_object), which bound the
bare-noun prompt's top mask even at very low SAM3 confidence (transfer round: butter -> cream cheese, tomato sauce ->
ketchup, 16/24 episodes).

Rule (public data only: RGB-D, K, E, the task's own prompt list; no object names, scene, seed or truth):
  1. query SAM3 with EVERY prompt; drop masks with < MIN_POINTS valid depth points;
  2. group masks from all prompts into instances by mask IoU >= IOU_SAME (greedy, highest score first);
  3. instance support = mean, over RESPONDING prompts (>= 1 kept mask), of the instance's best score under that prompt
     (0 if the instance is absent under a responding prompt). Amendment 2026-09-22 (v2): v1 averaged over ALL prompts, so a
     prompt SAM3 does not recognise at all counted as 0 and pushed an unambiguous object below MIN_SUPPORT (pudding: only
     "brown box" responds, 10/10 not_found offline). Chosen after seeing those source-task rows -> they are dev data now;
     validation is on fresh initial states (results/fresh_states/).
  4. bound      if top support >= MIN_SUPPORT and (top - second) / top >= MIN_REL_MARGIN
     ambiguous  if top support >= MIN_SUPPORT but the margin is smaller  (fail-closed: caller must not grasp)
     not_found  otherwise.
Constants:
  IOU_SAME = 0.5         masks of one object under different prompts overlap almost fully (probe: pixel counts within 1%)
  MIN_POINTS = 10        same as the legacy localize_object
  MIN_SUPPORT = 0.30     mean best-score across prompts; below this no prompt is confident about any instance
  MIN_MASK_SCORE = 0.05  efficiency only: SAM3 returns ~200 candidate masks per prompt, almost all near 0; a mask below
                         0.05 can move an instance's support (mean over >=1 prompts) by < 0.05/n_prompts, far below MIN_SUPPORT
  MAX_MASKS_PER_PROMPT = 20  efficiency only (highest-scoring first)
  MIN_REL_MARGIN = 0.05  CHOSEN AFTER inspecting the transfer-round SAM3 probe (tomato-sauce margins 0.04-0.08 relative
                         to alphabet soup, butter-vs-cream-cheese 0.01-0.05). It is therefore NOT validated by that data;
                         the source-task rounds and fresh states are the held-out check.
"""
from __future__ import annotations

import numpy as np

IOU_SAME = 0.5
MIN_POINTS = 10
MIN_SUPPORT = 0.30
MIN_REL_MARGIN = 0.05
MIN_MASK_SCORE = 0.05
MAX_MASKS_PER_PROMPT = 20


def deproject(mask, depth, K, E):
    """Identical arithmetic to mask_to_world_points (camera -> public frame via the camera-to-world pose E)."""
    ys, xs = np.where(np.asarray(mask) > 0)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    z = depth[ys, xs]
    v = z > 0
    ys, xs, z = ys[v], xs[v], z[v]
    if len(z) == 0:
        return np.empty((0, 3))
    pc = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], axis=-1)
    return (np.asarray(E) @ pc.T).T[:, :3]


def _bbox(m):
    ys, xs = np.where(m)
    return (ys.min(), ys.max(), xs.min(), xs.max()) if len(ys) else None


def mask_iou(a, b, ba=None, bb=None):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    ba = ba if ba is not None else _bbox(a); bb = bb if bb is not None else _bbox(b)
    if ba is None or bb is None or ba[1] < bb[0] or bb[1] < ba[0] or ba[3] < bb[2] or bb[3] < ba[2]:
        return 0.0
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def group_instances(per_prompt, n_prompts):
    """per_prompt: list over prompts of lists of {"mask", "score"}. Returns instances sorted by support (desc).
    Support averages over responding prompts only (prompts with at least one kept mask)."""
    flat = [(float(m["score"]), pi, m) for pi, ms in enumerate(per_prompt) for m in ms]
    flat.sort(key=lambda t: -t[0])
    inst = []
    for score, pi, m in flat:
        mk = np.asarray(m["mask"], bool); bm = _bbox(mk)
        best, best_iou = None, 0.0
        for I in inst:
            iou = mask_iou(mk, I["mask"], bm, I["bbox"])
            if iou > best_iou:
                best, best_iou = I, iou
        if best is not None and best_iou >= IOU_SAME:
            best["scores"][pi] = max(best["scores"][pi], score)
            best["members"].append((pi, score))
        else:
            sc = [0.0] * n_prompts; sc[pi] = score
            inst.append({"mask": mk, "bbox": bm, "scores": sc, "members": [(pi, score)], "rep_score": score})
    responding = [pi for pi in range(n_prompts) if per_prompt[pi]]
    for I in inst:
        I["support"] = float(np.mean([I["scores"][pi] for pi in responding])) if responding else 0.0
    inst.sort(key=lambda I: -I["support"])
    return inst


def decide(inst):
    if not inst or inst[0]["support"] < MIN_SUPPORT:
        return "not_found", None
    top = inst[0]["support"]; second = inst[1]["support"] if len(inst) > 1 else 0.0
    rel = (top - second) / top if top > 0 else 0.0
    return ("bound" if rel >= MIN_REL_MARGIN else "ambiguous"), rel


def bind_object(seg_fn, rgb, depth, K, E, prompts):
    """Returns {"status": bound|ambiguous|not_found, "center", "pts", "mask", "support", "rel_margin", "instances": [...]}.
    `center` is the OBB-free median of the instance points (callers that need the legacy OBB centre recompute it)."""
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    per_prompt, n_calls = [], 0
    for pr in prompts:
        res = seg_fn(rgb, pr) or []
        n_calls += 1
        keep = []
        res = sorted((m for m in res if float(m["score"]) >= MIN_MASK_SCORE), key=lambda m: -float(m["score"]))[:MAX_MASKS_PER_PROMPT]
        for m in res:
            pts = deproject(np.asarray(m["mask"]).astype(np.uint8), depth, K, E)
            if len(pts) >= MIN_POINTS:
                keep.append({"mask": np.asarray(m["mask"], bool), "score": float(m["score"])})
        per_prompt.append(keep)
    inst = group_instances(per_prompt, len(prompts))
    status, rel = decide(inst)
    n_resp = sum(1 for k in per_prompt if k)
    summary = [{"support": round(I["support"], 4), "scores": [round(s, 4) for s in I["scores"]], "n_px": int(I["mask"].sum())} for I in inst[:5]]
    out = {"status": status, "support": inst[0]["support"] if inst else 0.0, "rel_margin": rel, "prompts": prompts,
           "sam3_calls": n_calls, "n_responding_prompts": n_resp, "rule_version": "v2_responding_mean", "instances": summary, "center": None, "pts": None, "mask": None}
    if inst:
        pts = deproject(inst[0]["mask"].astype(np.uint8), depth, K, E)
        out.update(mask=inst[0]["mask"], pts=pts, center=np.median(pts, axis=0) if len(pts) else None)
        out["_instances_full"] = inst          # for offline evaluation only
    return out


def legacy_localize(seg_fn, rgb, depth, K, E, prompts):
    """The legacy rule, for side-by-side evaluation: first prompt with a usable top mask wins."""
    if isinstance(prompts, str):
        prompts = [prompts]
    for pr in prompts:
        res = seg_fn(rgb, pr) or []
        if not res:
            continue
        best = max(res, key=lambda d: d["score"])
        pts = deproject(np.asarray(best["mask"]).astype(np.uint8), depth, K, E)
        if len(pts) < MIN_POINTS:
            continue
        return {"status": "bound", "mask": np.asarray(best["mask"], bool), "pts": pts, "center": np.median(pts, axis=0),
                "score": float(best["score"]), "prompt": pr}
    return {"status": "not_found", "mask": None, "pts": None, "center": None}
