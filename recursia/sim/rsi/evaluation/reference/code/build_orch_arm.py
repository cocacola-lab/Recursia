"""Build the two N20 inference arms from frozen inputs (auditable, idempotent):
  programs/N20C.py = programs/N20.py with its final line `run()` replaced by the arm's orchestration entry
  (N20C: closed loop, closed_loop.orch_closed_loop); stack bytes unchanged
  code/iso_runner_orch.py = code/iso_runner_90.py + the patch list below (exact-match replacements, each asserted once)
Run: .venv-libero/bin/python3 code/build_orch_arm.py"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
N13_SHA = "29b187dfaffb4e1bc3a9e8d866637c552453a36fbc2f00691c78dc065f2d7cee"   # name kept; = programs/N20.py
RUNNER_SHA = "9784f490bcf9b6d1a017f82f5543e51c9c40125c0c6246cd16c988a4ac6b4817"   # Recursia port of the N20 interim iso_runner_90.py (original 082faf420c99…; import paths renamed only)   # N20 interim iso_runner_90.py
ARMS = ("N20C",)
TRAILER = {a: f"\n\n# ===== {a} orchestration entry (arm {a}; replaces the baseline driver call `run()`; every definition above is the frozen N20 stack) =====\n"
              f"orch_main(globals())   # {a}_ORCH_ENTRY\n" for a in ARMS}

PATCHES = [
    ("header",
     '"""Isolated runner (round zeroshot_bodypinch_transfer',
     '# N20C: derived from code/iso_runner_90.py by code/build_orch_arm.py (LLM orchestration INFERENCE methods over the frozen N20 stack: N20C closed loop; arm N20 unchanged).\n"""Isolated runner (round zeroshot_bodypinch_transfer'),
    ("arms",
     'ARMS90 = tuple(["N%d" % i for i in range(21)]',
     'ARMS90 = ("N20C",) + tuple(["N%d" % i for i in range(21)]'),
    ("vlm_arms",
     'VLM_ARMS = ("N20",)',
     'VLM_ARMS = ("N20", "N20C")   # N20C: the frozen stack keeps its VB binding module (query_vlm); orchestration generation uses query_llm'),
    ("vlm_url",
     'VLM_URL = "http://127.0.0.1:8117/v1/vlm"',
     'VLM_URL = os.environ.get("RSI_VLM_URL", "http://127.0.0.1:8117/v1/vlm")   # N20C: the launcher sets the experiment\'s own loopback proxy\n'
     'ORCH_ARMS = ("N20C",)\n'
     'ORCH_MARKERS = {a: "orch_main(globals())   # " + a + "_ORCH_ENTRY" for a in ORCH_ARMS}\n'
     'ORCH_FILES = ("orch_stack.py", "orch_check.py", "closed_loop.py")'),
    ("marker",
     'PROGRAM_MARKER = {a: f"\'arm_k\': \'K2\', \'arm_90\': \'{a}\'," for a in ARMS90}',
     'PROGRAM_MARKER = {a: f"\'arm_k\': \'K2\', \'arm_90\': \'{a}\'," for a in ARMS90}\n'
     'PROGRAM_MARKER.update(ORCH_MARKERS)   # N20C: the program is the N20 stack (its N20 marker) + the arm\'s orchestration entry'),
    ("query_llm",
     '    def goto_pose_m2(position, quaternion_wxyz, z_approach=0.0):',
     '    llm_stats = {"calls": 0, "ok": 0, "abstain_or_error": 0, "cost_usd": 0.0, "latency_s": 0.0, "usage": []}\n\n'
     '    def query_llm(images, prompt, schema, tag="", max_tokens=6000):\n'
     '        """N13C: the one orchestration-generation call through the same loopback proxy (no credential here). Returns the\n'
     '        proxy response dict (status, data, usage, cost_usd, id) plus latency; never raises."""\n'
     '        import base64, io, urllib.request\n'
     '        from PIL import Image\n'
     '        llm_stats["calls"] += 1; t0 = time.time(); enc = []\n'
     '        for im in images:\n'
     '            buf = io.BytesIO(); Image.fromarray(np.asarray(im, dtype=np.uint8)).save(buf, "PNG"); enc.append(base64.b64encode(buf.getvalue()).decode())\n'
     '        try:\n'
     '            body = json.dumps({"tag": str(tag), "prompt": str(prompt), "images": enc, "schema": schema, "max_tokens": int(max_tokens)}).encode()\n'
     '            with urllib.request.urlopen(urllib.request.Request(VLM_URL, data=body, headers={"Content-Type": "application/json"}), timeout=600) as r:\n'
     '                resp = json.loads(r.read().decode())\n'
     '        except Exception as e:  # noqa: BLE001 - transport failure = generation error, recorded\n'
     '            resp = {"status": "error", "detail": repr(e)[:300]}\n'
     '        resp["latency_s"] = round(time.time() - t0, 2)\n'
     '        llm_stats["cost_usd"] += float(resp.get("cost_usd") or 0.0); llm_stats["latency_s"] += resp["latency_s"]; llm_stats["usage"].append(resp.get("usage"))\n'
     '        llm_stats["ok" if resp.get("status") == "ok" else "abstain_or_error"] += 1\n'
     '        with open(diag_dir / "orch_llm_calls.jsonl", "a") as fh:\n'
     '            fh.write(json.dumps({k: resp.get(k) for k in ("status", "detail", "usage", "cost_usd", "id", "latency_s")} | {"tag": tag, "sim_step": int(low._sim_step_count), "max_tokens": int(max_tokens)}, default=str) + "\\n")\n'
     '        return resp\n\n'
     '    def goto_pose_m2(position, quaternion_wxyz, z_approach=0.0):'),
    ("inject",
     '            if exp_arm in VLM_ARMS:\n                fns["query_vlm"] = query_vlm\n',
     '            if exp_arm in VLM_ARMS:\n                fns["query_vlm"] = query_vlm\n'
     '            if exp_arm in ORCH_ARMS:   # N20C: the orchestration entry, bound to this episode\'s diag dir and the model calls\n'
     '                import orch_stack as OS\n'
     '                _hashes = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ORCH_FILES}\n'
     '                import closed_loop as CL\n'
     '                _cat = HERE.parent / "catalog" / "rsi_n20_interface_catalog_v2.json"\n'
     '                if exp_arm == "N20C":   # closed loop: typed JSON plan / decisions / visual checks; no generated code\n'
     '                    fns["orch_main"] = lambda ns: CL.orch_closed_loop(ns, diag_dir=diag_dir, query_llm=query_llm, catalog_path=_cat, prompt_dir=HERE.parent / "prompts", code_hashes=_hashes)\n'
     '                else:   # (one-shot arm; not used in this experiment)\n'
     '                    fns["orch_main"] = lambda ns: OS.orch_main(ns, diag_dir=diag_dir, query_llm=query_llm, catalog_path=_cat, prompt_path=HERE.parent / "prompts" / "orch_prompt_v2.md", code_hashes=_hashes)\n'),
    ("stats_attr",
     '    install_interface.vlm_stats = vlm_stats\n',
     '    install_interface.vlm_stats = vlm_stats\n    install_interface.llm_stats = llm_stats\n'),
    ("prefix_check",
     '    attempt_id = f"{a.arm}:{short}:s{a.seed:02d}:{a.uid}"\n',
     '    if a.arm in ORCH_ARMS:   # N20C: the program must be the frozen N20 stack byte-for-byte + the arm\'s orchestration entry\n'
     '        _src = code_bytes.decode()\n'
     '        assert _src.endswith(ORCH_MARKERS[a.arm] + "\\n"), "program must end with its arm\'s orchestration entry"\n'
     '        _stack = _src[: _src.index("\\n\\n# ===== " + a.arm + " orchestration entry")] + "\\n\\nrun()\\n"\n'
     '        assert hashlib.sha256(_stack.encode()).hexdigest() == "' + N13_SHA + '", "stack bytes differ from frozen N20"\n'
     '    attempt_id = f"{a.arm}:{short}:s{a.seed:02d}:{a.uid}"\n'),
    ("manifest",
     '    manifest.update(harness_revision="n8_clean_horizon_v1", failure_class=info_step.get("failure_class"), horizon_exhausted=horizon_guard.exhausted)\n',
     '    manifest.update(harness_revision="n8_clean_horizon_v1", failure_class=info_step.get("failure_class"), horizon_exhausted=horizon_guard.exhausted)\n'
     '    if a.arm in ORCH_ARMS:   # N20C: orchestration record (generation, check, stop) next to the official outcome; never a success measure\n'
     '        _os = out / "diag" / "orch" / "summary.json"\n'
     '        manifest.update(orch_summary=json.loads(_os.read_text()) if _os.is_file() else {"outcome": "orch_summary_missing"}, orch_llm_stats=dict(install_interface.llm_stats),\n'
     '                        orch_files_sha256={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ORCH_FILES})\n'),
]


def build():
    n20 = (EXP / "programs" / "N20.py").read_bytes()
    assert hashlib.sha256(n20).hexdigest() == N13_SHA and n20.endswith(b"\n\nrun()\n")
    for a in ARMS:
        (EXP / "programs" / f"{a}.py").write_text(n20[: -len(b"\n\nrun()\n")].decode() + TRAILER[a])
    src = (HERE / "iso_runner_90.py").read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == RUNNER_SHA
    for name, old, new in PATCHES:
        assert src.count(old) == 1, f"patch {name}: anchor found {src.count(old)} times"
        src = src.replace(old, new)
    (HERE / "iso_runner_orch.py").write_text(src)
    return {p: hashlib.sha256((EXP / p).read_bytes()).hexdigest() for p in ("programs/N20C.py", "code/iso_runner_orch.py")}


if __name__ == "__main__":
    print(build())
