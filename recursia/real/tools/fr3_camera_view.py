#!/usr/bin/env python3
"""Live, read-only preview of the configured FR3 external RealSense camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyrealsense2 as rs  # noqa: E402

from cap.saved_scripts.fr3_placement.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/rsi_fr3_station.local.yaml")
    args = parser.parse_args()
    serial = load_config(args.config).external_serial

    pipeline = rs.pipeline()
    stream = rs.config()
    stream.enable_device(serial)
    stream.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(stream)
    print(f"Front camera {serial}: press q or Esc to close")
    try:
        while True:
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                continue
            cv2.imshow("FR3 front camera", np.asanyarray(frame.get_data()))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
