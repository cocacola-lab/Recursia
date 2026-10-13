#!/usr/bin/env python3
"""Real Contact-GraspNet inference, with widths preserved for the Franka Hand."""
import argparse
import base64
import io
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()
estimator = None
lock = threading.Lock()
metadata = {}


def encode(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode(value):
    array = np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("point cloud must be a finite Nx3 array in optical-camera metres")
    if not 20 <= len(array) <= 1_000_000:
        raise ValueError("point cloud must contain 20 to 1000000 points")
    return array.astype(np.float32)


class Request(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    forward_passes: int = Field(default=1, ge=1, le=5)


@app.get("/health")
def health():
    return {"model_loaded": estimator is not None, "mock": False,
            "model": "contact_graspnet_pytorch", "frame": "camera_optical", **metadata}


@app.post("/plan_point_clouds")
def plan(request: Request):
    if estimator is None:
        raise HTTPException(503, "model not loaded")
    try:
        scene, segment = decode(request.pc_full_base64), decode(request.pc_segment_base64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    started = time.monotonic()
    # GPU model is shared; never run competing forward passes in server threads.
    with lock, torch.inference_mode():
        grasps, scores, contacts, openings = estimator.predict_scene_grasps(
            scene, pc_segments={1: segment}, local_regions=True, filter_grasps=True,
            forward_passes=request.forward_passes)
    g = np.asarray(grasps.get(1, [])).reshape(-1, 4, 4)
    s = np.asarray(scores.get(1, [])).reshape(-1)
    c = np.asarray(contacts.get(1, [])).reshape(-1, 3)
    w = np.asarray(openings.get(1, [])).reshape(-1)
    # The network adds extra_opening for pregrasp clearance. Hand.grasp needs
    # the predicted contact width; passing clearance can report a false grasp.
    w = np.maximum(w - float(estimator._contact_grasp_cfg['TEST']['extra_opening']), 0)
    return {"model": "contact_graspnet_pytorch", "frame": "camera_optical",
            "elapsed_s": time.monotonic() - started,
            **{key + "_base64": encode(value) for key, value in
               (("grasps", g), ("scores", s), ("widths", w), ("contact_pts", c))}}


def main():
    global estimator, metadata
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir(source)  # Upstream meshes/control points use repository-relative paths.
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "Pointnet_Pointnet2_pytorch"))
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    folder = source / "checkpoints/contact_graspnet"
    config = yaml.safe_load((folder / "config.yaml").read_text())
    config["OPTIMIZER"]["batch_size"] = 1
    torch.set_num_threads(4)
    estimator = GraspEstimator(config)
    CheckpointIO(checkpoint_dir=str(folder / "checkpoints"), model=estimator.model).load("model.pt")
    estimator.model.eval()
    metadata = {"device": str(estimator.device), "checkpoint": str(folder / "checkpoints/model.pt"),
                "width_convention": "predicted contact width, without pregrasp extra_opening"}
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
