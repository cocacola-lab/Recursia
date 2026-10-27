"""Offline helpers: load the frozen N20 definitions (programs/N20.py minus its final `run()` call) into a namespace with stub
robot APIs, so the interface layer can be exercised without a simulator. Used by tests/ and build_catalog.py only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

from adapter_v3 import descriptors_for  # noqa: E402
import orch_stack as OS  # noqa: E402

N12_PATH = EXP / "programs" / "N20.py"   # name kept for the tests; the frozen stack is N20
TRAILER = b"\n\nrun()\n"


def stack_source():
    raw = N12_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OS.STACK_SHA256, "programs/N20.py is not the frozen N20 stack"
    assert raw.endswith(TRAILER)
    return raw[: -len("run()\n")].decode()


class _Handle:
    def __init__(self, lang):
        self.task_language = lang


class _Env:
    def __init__(self, lang):
        self.handle = _Handle(lang)


def load_namespace(lang, api=None):
    """Frozen N20 definitions executed unchanged; `api` supplies the runner-injected functions (stubs offline)."""
    ns = {"__name__": "n12_offline", "env": _Env(lang)}
    td = descriptors_for(lang)
    ns["task_descriptors"] = lambda: json.loads(json.dumps(td))
    ns.update(api or {})
    exec(compile(stack_source(), str(N12_PATH), "exec"), ns)   # noqa: S102 - the frozen stack itself
    return ns, td


def runtime_languages():
    import glob
    out = {}
    for f in sorted(glob.glob(str(EXP.parent / "rsi_libero90_N18_interim_long100_20261012/stage_interim_N18_long100/attempts/*/N18/*/seed_41/attempt_manifest.json"))):
        m = json.loads(Path(f).read_text()); out[m["task_short"]] = m["task_language"]
    return out
