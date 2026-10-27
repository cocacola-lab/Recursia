#!/usr/bin/env python3
"""N15 infrastructure probe (2026-10-10): one tiny real VLM request through the loopback proxy; exit 0 if answered ok within 60 s.
Used by run_plan_90.sh to wait (not start an episode) while the provider is unreachable. Tag 'infra:probe' in the VLM ledger."""
import base64, io, json, sys, urllib.request
import numpy as np
from PIL import Image
im = Image.fromarray(np.full((16, 16, 3), 128, np.uint8)); b = io.BytesIO(); im.save(b, 'PNG')
body = json.dumps({"tag": "infra:probe", "prompt": "Is this image grey? Answer yes or no.", "images": [base64.b64encode(b.getvalue()).decode()],
                   "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}).encode()
try:
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8117/v1/vlm", data=body, headers={"Content-Type": "application/json"}), timeout=60) as r:
        sys.exit(0 if json.loads(r.read().decode()).get("status") == "ok" else 1)
except Exception:  # noqa: BLE001
    sys.exit(1)
