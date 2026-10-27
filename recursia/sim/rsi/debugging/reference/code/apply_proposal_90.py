"""RSI harness for the LIBERO-90 round (fixed; the proposer may not edit this file). Builds programs/<out>.py from programs/<parent>.py
by applying a proposal's exact-anchor patches, then runs the proposal's unit test. Usage:
  apply_proposal_90.py --init                          builds programs/N0.py from parent_programs/K2.py (hash-checked) + marker
  apply_proposal_90.py --parent <arm> --out <arm> --proposal proposals/<dir>/patch.py
A proposal defines MODULE, FAILURE_CLASS, PATCHES = [(old, new), ...] (each anchor exactly once), optional INSERTS = [(anchor, text), ...].
No object / part / scene / colour / seed literal may appear in new text (asserted): task facts come from task_descriptors() data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import py_compile
import subprocess
import sys
from pathlib import Path

EXP = Path(os.environ.get("EXPERIMENT_DIR", Path(__file__).resolve().parents[1]))
PARENT_SRC = EXP / "parent_programs" / "K2.py"
PARENT_HASHES = EXP.parent / "rsi_subset_20260923" / "programs" / "hashes.txt"
FORBIDDEN = ("mug", "cup", "pudding", "plate", "white", "yellow", "brown", "chocolate", "scene", "seed", "libero", "tray", "butter", "ketchup",
             "tomato", "sauce", "bottle", "cream", "cheese", "soup", "left", "right", "basket", "bowl", "drawer", "cabinet", "microwave", "stove",
             "moka", "book", "caddy", "shelf", "wine", "rack", "alphabet", "salad", "dressing", "milk", "juice", "orange", "frying", "pan", "black",
             "red", "gray", "compartment", "knob", "burner", "kitchen", "study", "living")
MODULES = ("motion_preconditions", "grasp_geometry", "hold_contract", "place_primitive", "verifier", "binding", "language_adapter",
           "manipulation_primitive", "task_driver")
MARK_OLD = "'arm_s': 'S1', 'arm_k': 'K2',"


def marker(arm):
    return MARK_OLD + f" 'arm_90': '{arm}',"


def sub1(src, old, new, what):
    n = src.count(old); assert n == 1, f"{what}: anchor found {n} times"; return src.replace(old, new)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", action="store_true"); ap.add_argument("--snapshot", help="write programs/N<c>.py = --parent program with the marker changed (byte-identical otherwise)"); ap.add_argument("--parent"); ap.add_argument("--out"); ap.add_argument("--proposal"); a = ap.parse_args()
    out = EXP / "programs"; out.mkdir(exist_ok=True)
    if a.snapshot:
        assert a.parent; src = (out / f"{a.parent}.py").read_text(); name = f"N{int(a.snapshot)}"
        prog = sub1(src, marker(a.parent), marker(name), "marker"); notes = f"{name} = snapshot of {a.parent} (marker only)"
    elif a.init:
        src = PARENT_SRC.read_text(); sha = hashlib.sha256(src.encode()).hexdigest()
        line = [l for l in PARENT_HASHES.read_text().splitlines() if "programs/K2.py" in l]
        assert line and line[0].split()[0] == sha, "parent K2 hash mismatch"
        prog = sub1(src, MARK_OLD, marker("N0"), "marker"); name = "N0"; notes = "N0 = subset-round K2 + marker"
    else:
        assert a.parent and a.out and a.proposal
        src = (out / f"{a.parent}.py").read_text()
        spec = importlib.util.spec_from_file_location("proposal", a.proposal); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
        assert P.MODULE in MODULES, f"MODULE must be one of {MODULES}"; assert isinstance(P.FAILURE_CLASS, str) and P.FAILURE_CLASS
        new_text = "".join(n for _, n in P.PATCHES) + "".join(t for _, t in getattr(P, "INSERTS", []))
        import re as _re
        bad = sorted({w for w in FORBIDDEN if _re.search(r"(?<![a-z_])" + w + r"(?![a-z_])", new_text.lower())})
        assert not bad, f"forbidden literal(s) in patch: {bad}"
        prog = sub1(src, marker(a.parent), marker(a.out), "marker")
        for i, (anchor, text) in enumerate(getattr(P, "INSERTS", [])):
            prog = sub1(prog, anchor, text + anchor, f"insert {i}")
        for i, (old, new) in enumerate(P.PATCHES):
            prog = sub1(prog, old, new, f"patch {i}")
        name = a.out; notes = f"{a.out} = {a.parent} + {Path(a.proposal).parent.name} ({P.MODULE}; {P.FAILURE_CLASS})"
    p = out / f"{name}.py"; p.write_text(prog); py_compile.compile(str(p), doraise=True)
    h = hashlib.sha256(prog.encode()).hexdigest()
    with open(out / "hashes.txt", "a") as fh:
        fh.write(f"{h}  programs/{name}.py  # {notes}\n")
    print(f"{h}  programs/{name}.py  # {notes}")
    if a.proposal:
        tests = sorted(Path(a.proposal).parent.glob("test_*.py"))
        if tests:
            env = dict(os.environ, EXPERIMENT_DIR=str(EXP), RSI_PROGRAM=str(p))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", *map(str, tests)], env=env, capture_output=True, text=True); print(r.stdout[-600:], r.stderr[-300:]); assert r.returncode == 0, "proposal unit tests failed"
        else:
            print("WARNING: no unit test file in", Path(a.proposal).parent)


if __name__ == "__main__":
    main()
