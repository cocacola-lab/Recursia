#!/usr/bin/env python3
"""Saved RGB-D -> SAM3 mask -> Contact-GraspNet -> base TCP; no hardware access."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cap.agent.rsi.graspnet import ContactGraspNetClient, select_grasp
from cap.saved_scripts.fr3_placement.config import load_config
from cap.saved_scripts.fr3_runtime.camera import Capture
from cap.saved_scripts.fr3_runtime.vision import Sam3Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/rsi_fr3_station.example.yaml")
    parser.add_argument("--prompt", default="mug")
    parser.add_argument("--mask", type=Path, help="optional mask from exactly this snapshot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg, capture = load_config(args.config), Capture.load(args.snapshot)
    output = args.output or ROOT / "logs" / f"contact_graspnet_check_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.mask:
        mask = np.load(args.mask, allow_pickle=False).astype(bool)
    else:
        candidates = Sam3Client(cfg.segmentation_url).segment_all(capture.rgb, args.prompt)
        if not candidates:
            raise ValueError(f"SAM3 did not find {args.prompt}")
        mask = max(candidates, key=lambda value: value[1])[0]
    np.save(output / "mask.npy", mask)
    predicted = ContactGraspNetClient(cfg.graspnet_url).predict(capture, mask)
    np.savez_compressed(output / "candidates.npz", **predicted)
    transform, width, index = select_grasp(predicted["grasps"], predicted["scores"], predicted["widths"],
                                         cfg.camera_to_base, cfg.graspnet_to_ee)
    result = {"physical_motion_commanded": False, "hardware_accessed": False,
              "model": "contact_graspnet_pytorch", "snapshot": str(args.snapshot.resolve()),
              "prompt": args.prompt, "mask_pixels": int(np.count_nonzero(mask)),
              "candidate_count": len(predicted["scores"]), "selected_index": index,
              "score": float(predicted["scores"][index]), "width_m": width,
              "T_camera_grasp": predicted["grasps"][index].tolist(),
              "T_base_ee": transform.tolist(),
              "quaternion_xyzw": Rotation.from_matrix(transform[:3,:3]).as_quat().tolist(),
              "elapsed_s": time.monotonic()-started}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
