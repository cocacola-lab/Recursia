#!/usr/bin/env python3
"""RSI zero-shot transfer evaluation on LIBERO-Long-Pro: one tool for the three inference methods over a frozen stack.

  rule     arm N20   the stack's own driver run() (rule-based language parse, each subgoal executed once)
  oneshot  arm N20O  one Opus-generated orchestration program per episode, interpreted over interface v2 (no rewrite)
  closed   arm N20C  closed loop: Opus plan, a decision after every subgoal (execute / retry / replace / stop), visual checks

Every method runs the same frozen N20 stack (programs/N20.py, sha256 29b187df...) through the same runner (iso_runner_orch.py);
only the driver differs. For `closed`, setup verifies that the arm program and the runner are byte-identical to the run that
scored 35/100 (outputs/rsi_libero90_N20C_closedloop_long100_20261016).

  setup        --out DIR --method rule|oneshot|closed [--scale interim|full|subset] [--subset-ref N20C] [--gpus 3 4 5 6 7] [--cap-usd 35]
               subset = the interim rows on which a reference run succeeded (refs/interim_refs.json), e.g. a correctness check
               that the evaluation reproduces the reference's successes
  start-proxy  --out DIR      loopback VLM proxy with DIR's ledger and cap (run from a shell that has ANTHROPIC_API_KEY)
  stop-proxy   --out DIR
  check        --out DIR      frozen inputs + plan + services
  smoke        --out DIR      1 episode (long2_swap s41), excluded from the count
  run          --out DIR      the planned episodes (interim 100 | full 1000)
  rerun-infra  --out DIR      predeclared: episodes with a proxy transport error are rerun once (flag-selected)
  summarize    --out DIR      results/summary.json (+ summary_infra_rerun.json) and results/REPORT_auto.md
  export-trajectories --out DIR --dest PATH [--all]   compact, text-only episode records (successes only unless --all):
               manifest, stdout, orchestration summary/calls, VLM calls, motion + contact truth (gz); no frames or arrays
  retry        --out DIR --attempt N --rows TASK:SEED ...   diagnostic re-runs of planned rows (phase retry<N>_<arm>);
               reported in results/retry<N>.json, never merged into the official summary (selection by outcome biases it)
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
from math import comb, sqrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUNBOOK = HERE.parent
REF = RUNBOOK / "reference"
OLD_ROOT = "$RECURSIA_SIM_ROOT"
OLD_EXP = OLD_ROOT + "/outputs/rsi_libero90_N20C_closedloop_long100_20261016"
STACK_SHA = "29b187dfaffb4e1bc3a9e8d866637c552453a36fbc2f00691c78dc065f2d7cee"
N20C_SHA = "dbe25f1450857b0b2c583e07d42620b98fb07da6c56d189bb66376b950511ec1"
RUNNER_SHA = "c7517569e1996316f92523b570f9c139ed8a8c9070b25e7708c3dfc4bdd71b78"   # iso_runner_orch.py of the 35/100 run (original 3d1d1378…), import paths renamed for Recursia
METHODS = {"rule": "N20", "oneshot": "N20O", "closed": "N20C"}
ORCH_FILES = ("orch_stack.py", "orch_check.py", "closed_loop.py")
SEEDS = {"interim": tuple(range(41, 46)), "full": tuple(range(1, 51)), "subset": tuple(range(41, 46))}
TASKS = tuple(f"long{i}_{x}" for i in range(10) for x in ("swap", "task"))
STALE = ("start_orch.sh", "start_long100.sh", "interim100.py", "interim100_cl.py", "run_plan_90.sh", "run_attempt_90.sh", "run_observe_90.sh",
         "recheck_identity_90.py", "run_eval.py", "summarize_eval.py", "amend_manifest_long8.py", "gen_manifest_90.py")
PROXY = "http://127.0.0.1:8117"


def die(msg):
    print("ERROR:", msg, file=sys.stderr); raise SystemExit(2)


def need(ok, msg):
    if not ok:
        die(msg)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sim_root():
    if os.environ.get("RECURSIA_SIM_ROOT"):
        return Path(os.environ["RECURSIA_SIM_ROOT"]).resolve()
    for d in HERE.parents:
        if (d / ".venv-libero").is_dir() and (d / "cap").is_dir():
            return d
    die("set RECURSIA_SIM_ROOT to recursia/sim")


def py():
    return str(sim_root() / ".venv-libero/bin/python3")


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending"); tmp.write_text(json.dumps(data, indent=1, default=str) + "\n"); tmp.replace(path)


def sub1(text, old, new, what):
    n = text.count(old)
    need(n == 1, f"{what}: anchor found {n} times")
    return text.replace(old, new)


def edit(path, pairs):
    t = Path(path).read_text()
    for what, old, new in pairs:
        t = sub1(t, old, new, f"{Path(path).name}:{what}")
    Path(path).write_text(t)


# ----------------------------------------------------------------------------- setup
def phases(cfg):
    a = cfg["arm"]
    return {"main": f"{cfg['scale']}_{a}", "smoke": f"smoke_{a}", "rerun": f"rerun_infra_{a}"}


def plan_rows(arm, seeds):
    return [(arm, t, s) for t in TASKS for s in seeds]


def cmd_setup(a):
    out = Path(a.out).resolve(); root = sim_root()
    need(not out.exists(), f"{out} exists; use a new directory")
    arm = METHODS[a.method]
    out.mkdir(parents=True)
    for d in ("code", "prompts", "catalog", "calibration", "examples", "tests", "programs"):
        shutil.copytree(REF / d, out / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("task_manifest.json", "runtime_languages.json"):
        shutil.copy2(REF / f, out / f)
    (out / "logs").mkdir(); (out / "results").mkdir()
    for f in STALE:
        (out / "code" / f).unlink(missing_ok=True)
    need(sha(out / "programs/N20.py") == STACK_SHA, "programs/N20.py is not the frozen N20 stack")
    # re-point the launch scripts (anchored edits; the runner itself is never edited here)
    gpus = " ".join(map(str, a.gpus))
    edit(out / "code/run_plan_orch.sh", [
        ("root", f"ROOT={OLD_ROOT}\n", 'ROOT="${RECURSIA_SIM_ROOT:?}"\n'),
        ("exp", f'[[ "$R" == "$ROOT/outputs/rsi_libero90_N20C_closedloop_long100_20261016" ]]', f'[[ "$R" == "{out}" ]]'),
        ("ledger", 'LEDGER="$ROOT/outputs/rsi_libero90_20260924/logs/starts.txt"', 'LEDGER="$R/logs/start_ledger.txt"; touch "$LEDGER"'),
        ("lock", 'LEDGER_LOCK="$ROOT/outputs/rsi_libero90_20260924/logs/starts.lock"', 'LEDGER_LOCK="$R/logs/start_ledger.lock"'),
        ("cap", 'CAP=20000; GPUS=(3 4 5 6 7)', f'CAP={a.start_cap}; GPUS=({gpus})'),
        ("mkdir", 'mkdir -p "$R/logs"\n', ''),
    ])
    t = (out / "code/run_plan_orch.sh").read_text()
    (out / "code/run_plan_orch.sh").write_text(sub1(t, 'set -euo pipefail\n', 'set -euo pipefail\nmkdir -p "${RSI_EXPERIMENT_DIR:?}/logs"\n', "mkdir-first"))
    edit(out / "code/run_attempt_orch.sh", [
        ("root", f'ROOT="{OLD_ROOT}"; EXP=', 'ROOT="${RECURSIA_SIM_ROOT:?}"; EXP='),
        ("exp", f'[[ "$EXP" == "$ROOT/outputs/rsi_libero90_N20C_closedloop_long100_20261016" ]]', f'[[ "$EXP" == "{out}" ]]'),
    ])
    edit(out / "code/orch_offline.py", [("languages",
        '    for f in sorted(glob.glob(str(EXP.parent / "rsi_libero90_N18_interim_long100_20261012/stage_interim_N18_long100/attempts/*/N18/*/seed_41/attempt_manifest.json"))):\n'
        '        m = json.loads(Path(f).read_text()); out[m["task_short"]] = m["task_language"]\n',
        '    if (EXP / "runtime_languages.json").is_file():   # evaluation runbook: the 20 recorded Long-Pro instructions ship with the reference\n'
        '        return dict(json.loads((EXP / "runtime_languages.json").read_text())["languages"])\n'
        '    for f in sorted(glob.glob(str(EXP.parent / "rsi_libero90_N18_interim_long100_20261012/stage_interim_N18_long100/attempts/*/N18/*/seed_41/attempt_manifest.json"))):\n'
        '        m = json.loads(Path(f).read_text()); out[m["task_short"]] = m["task_language"]\n')])
    if a.method == "oneshot":   # register the one-shot arm N20O next to N20C (same interface v2, same one-shot entry orch_stack.orch_main)
        edit(out / "code/build_orch_arm.py", [
            ("arms_tuple", '\nARMS = ("N20C",)\n', '\nARMS = ("N20C", "N20O")\n'),
            ("runner_arms", '\'ARMS90 = ("N20C",) + tuple(["N%d" % i for i in range(21)]\'', '\'ARMS90 = ("N20C", "N20O") + tuple(["N%d" % i for i in range(21)]\''),
            ("vlm_arms", '\'VLM_ARMS = ("N20", "N20C")', '\'VLM_ARMS = ("N20", "N20C", "N20O")'),
            ("orch_arms", "'ORCH_ARMS = (\"N20C\",)\\n'", "'ORCH_ARMS = (\"N20C\", \"N20O\")\\n'"),
        ])
    r = subprocess.run([py(), "code/build_orch_arm.py"], cwd=out, capture_output=True, text=True)
    need(r.returncode == 0, "build_orch_arm failed:\n" + r.stderr[-1500:])
    program = out / "programs" / f"{arm}.py"
    if a.method == "closed":
        need(sha(program) == N20C_SHA, "N20C.py differs from the 35/100 run")
        need(sha(out / "code/iso_runner_orch.py") == RUNNER_SHA, "iso_runner_orch.py differs from the 35/100 run")
    seeds = SEEDS[a.scale]
    rows = plan_rows(arm, seeds)
    if a.scale == "subset":
        refs = json.loads((REF / "refs/interim_refs.json").read_text())["refs"]
        need(a.subset_ref in refs, f"--subset-ref must be one of {sorted(refs)}")
        ok = {k for k, v in refs[a.subset_ref]["rows"].items() if v == 1}
        rows = [r for r in rows if f"{r[1]}:{r[2]}" in ok]
        need(len(rows) == len(ok) > 0, "subset rows do not match the reference")
    else:
        need(a.subset_ref is None, "--subset-ref only with --scale subset")
    cfg = {"created": utc(), "method": a.method, "arm": arm, "scale": a.scale, "seeds": list(seeds), "planned": len(rows), "subset_ref": a.subset_ref,
           "gpus": a.gpus, "cap_usd": a.cap_usd, "start_cap": a.start_cap, "stack_sha256": STACK_SHA, "program_sha256": sha(program),
           "runner_sha256": sha(out / "code/iso_runner_orch.py"), "identical_to_n20c_run": a.method == "closed",
           "model": "claude-opus-5-5", "effort": "low", "limits": {"steps": 4000, "timeout_s": 1500}, "removed_stale_launchers": list(STALE)}
    cfg["phases"] = phases(cfg)
    ph = cfg["phases"]
    (out / f"plan_{ph['main']}.txt").write_text("".join(f"{x} {t} {s}\n" for x, t, s in rows))
    (out / f"plan_{ph['smoke']}.txt").write_text(f"{arm} long2_swap 41\n")
    save(out / "eval_config.json", cfg)
    files = sorted(p for d in ("code", "prompts", "catalog", "programs") for p in (out / d).rglob("*") if p.is_file() and "__pycache__" not in str(p))
    files += [out / "calibration/result.json", out / "task_manifest.json", out / "eval_config.json", out / f"plan_{ph['main']}.txt", out / f"plan_{ph['smoke']}.txt"]
    (out / "frozen_inputs_sha256.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files))
    real = out.parent / "rsi_libero90_N18_interim_long100_20261012"   # the test resolves it next to the experiment
    sel = [] if real.is_dir() else ["-k", "not real_n12_logs"]   # that test replays recorded N18 logs, absent outside the original machine
    r = subprocess.run([py(), "-m", "pytest", "-q", "tests", *sel], cwd=out, capture_output=True, text=True)
    (out / "logs/offline_tests.txt").write_text(r.stdout + r.stderr)
    need(r.returncode == 0, "offline tests failed; see logs/offline_tests.txt")
    print(r.stdout.strip().splitlines()[-1])
    print(json.dumps({k: cfg[k] for k in ("method", "arm", "scale", "planned", "program_sha256", "identical_to_n20c_run")}, indent=1))
    print(f"SETUP_DONE {out}")


# ----------------------------------------------------------------------------- services
def load(out):
    out = Path(out).resolve()
    need((out / "eval_config.json").is_file(), f"{out} is not set up (rsi_eval.py setup)")
    return out, json.loads((out / "eval_config.json").read_text())


def health():
    try:
        r = subprocess.run(["curl", "-s", "-m", "5", PROXY + "/health"], capture_output=True, text=True)
        return json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def cmd_start_proxy(a):
    out, cfg = load(a.out)
    need(not health(), "something already answers on 127.0.0.1:8117; stop it first (this run needs its own ledger and cap)")
    need(bool(os.environ.get("ANTHROPIC_API_KEY")), "export ANTHROPIC_API_KEY in this shell (only the proxy process holds it)")
    log = open(out / "logs/proxy.out", "a")
    p = subprocess.Popen([str(sim_root() / ".venv-vlmproxy/bin/python3"), str(out / "code/vlm_proxy.py"), "--port", "8117",
                          "--cap-usd", str(cfg["cap_usd"]), "--ledger", str(out / "logs/vlm_ledger.jsonl")],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=sim_root())
    (out / "logs/proxy.pid").write_text(f"{p.pid}\n")
    for _ in range(30):
        time.sleep(1)
        if health().get("status") == "ok":
            break
    h = health(); need(h.get("status") == "ok", "proxy did not come up; see logs/proxy.out")
    r = subprocess.run([py(), str(out / "code/vlm_probe.py")], env=worker_env(out), cwd=out)
    need(r.returncode == 0, "provider probe failed; see logs/proxy.out")
    print("PROXY_OK", json.dumps(h))


def cmd_stop_proxy(a):
    out, _ = load(a.out); pid = out / "logs/proxy.pid"
    need(pid.is_file(), "no proxy pid file")
    try:
        os.kill(int(pid.read_text()), 15)
    except ProcessLookupError:
        pass
    pid.unlink(); print("PROXY_STOPPED")


def worker_env(out):
    root = sim_root()
    env = dict(os.environ, RSI_EXPERIMENT_DIR=str(out), EXPERIMENT_DIR=str(out), RECURSIA_SIM_ROOT=str(root), MUJOCO_GL="egl",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root),
               RSI_VLM_URL=PROXY + "/v1/vlm", LIBERO_CONFIG_PATH=os.environ.get("LIBERO_CONFIG_PATH", str(root / ".libero")))
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"):
        env.pop(k, None)
    return env


def freeze_check(out):
    bad = []
    for line in (out / "frozen_inputs_sha256.txt").read_text().splitlines():
        h, name = line.split(None, 1)
        if not (out / name).is_file() or sha(out / name) != h:
            bad.append(name)
    need(not bad, f"frozen input mismatch: {bad[:5]}")


def planned(out, cfg, phase):
    rows = [l.split() for l in (out / f"plan_{phase}.txt").read_text().splitlines() if l.strip()]
    return {(x, t, int(s)) for x, t, s in rows}


def preflight(out, cfg, n):
    freeze_check(out)
    need(sha(out / "programs/N20.py") == STACK_SHA and sha(out / "programs" / f"{cfg['arm']}.py") == cfg["program_sha256"], "program hash")
    for p in (8114, 8115, 8116):
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "3", f"http://127.0.0.1:{p}/health"], capture_output=True, text=True)
        need(r.stdout not in ("", "000"), f"perception server on port {p} is down")
    h = health()
    need(h.get("status") == "ok" and h.get("model") == cfg["model"], f"VLM proxy not healthy on 8117 (rsi_eval.py start-proxy): {h}")
    need(h.get("cap_usd", 0) - h.get("spent_usd", 1e9) >= 5.0, f"less than $5 VLM headroom: {h}")
    led = out / "logs/start_ledger.txt"; used = sum(1 for _ in led.open()) if led.exists() else 0
    need(used + n <= cfg["start_cap"], f"start budget: {used} used + {n} > {cfg['start_cap']}")
    return h


def run_plan(out, phase):
    return subprocess.run(["bash", str(out / "code/run_plan_orch.sh"), str(out / f"plan_{phase}.txt"), phase, "1500"], cwd=out, env=worker_env(out)).returncode


def cmd_check(a):
    out, cfg = load(a.out); preflight(out, cfg, 0)
    print(json.dumps({"frozen": "ok", "planned": len(planned(out, cfg, cfg["phases"]["main"])), "proxy": health()}, indent=1))


def cmd_smoke(a):
    out, cfg = load(a.out); ph = cfg["phases"]["smoke"]; preflight(out, cfg, 1)
    need(not (out / f"stage_{ph}").exists(), "smoke already ran")
    rc = run_plan(out, ph)
    recs = [rec for _, rec in load_phase(out, cfg, ph)]
    print("SMOKE_DONE rc", rc, json.dumps([{k: r.get(k) for k in ("task_short", "seed", "official", "orch_outcome", "errors", "wall_s")} for r in recs], indent=1))


def cmd_run(a):
    out, cfg = load(a.out); ph = cfg["phases"]["main"]
    h = preflight(out, cfg, cfg["planned"])
    need(not (out / f"attempts_log_{ph}.csv").exists() and not (out / f"stage_{ph}").exists(), "the main phase already ran")
    with (out / "run_once.lock").open("x") as f:
        f.write(utc() + "\n")
    save(out / "state.json", {"status": "running", "phase": ph, "started_at": utc(), "pid": os.getpid(), "planned": cfg["planned"], "proxy_at_start": h})
    rc = run_plan(out, ph)
    s = summarize(out, cfg)
    save(out / "state.json", {"status": s["status"], "phase": ph, "finished_at": utc(), "runner_plan_rc": rc, "official_successes": s["official_successes"],
                              "infra_rows": s["infra"]["episodes_with_transport_errors"]})
    print("STAGE_DONE", s["status"], s["official_successes"], "/", cfg["planned"], flush=True)


def cmd_rerun_infra(a):
    out, cfg = load(a.out); ph = cfg["phases"]
    s = summarize(out, cfg); rows = s["infra"]["episodes_with_transport_errors"]
    if not rows:
        print("no infrastructure-flagged rows"); return
    need(not (out / f"stage_{ph['rerun']}").exists(), "the infra rerun already ran (once only)")
    (out / f"plan_{ph['rerun']}.txt").write_text("".join(f"{x[0]} {x[1]} {x[2]}\n" for x in rows))
    preflight(out, cfg, len(rows)); rc = run_plan(out, ph["rerun"])
    s = summarize(out, cfg, with_reruns=True)
    print("RERUN_DONE rc", rc, s["status"], s["official_successes"])


def cmd_retry(a):
    out, cfg = load(a.out); arm = cfg["arm"]; ph = f"retry{a.attempt}_{arm}"
    exp = planned(out, cfg, cfg["phases"]["main"])
    rows = []
    for spec in a.rows:
        t, sd = spec.rsplit(":", 1); key = (arm, t, int(sd))
        need(key in exp, f"{spec} is not a planned row of this run")
        rows.append(key)
    need(not (out / f"stage_{ph}").exists(), f"attempt {a.attempt} already ran")
    (out / f"plan_{ph}.txt").write_text("".join(f"{x} {t} {sd}\n" for x, t, sd in rows))
    preflight(out, cfg, len(rows)); rc = run_plan(out, ph)
    recs = [rec for _, rec in load_phase(out, cfg, ph)]
    res = {"phase": ph, "attempt": a.attempt, "runner_rc": rc, "note": "diagnostic re-run; not part of the official result",
           "rows": [{k: r.get(k) for k in ("task_short", "seed", "official", "orch_outcome", "transport_errors", "errors", "wall_s", "episode_dir")} for r in recs]}
    save(out / f"results/retry{a.attempt}.json", res)
    print("RETRY_DONE", json.dumps([(r["task_short"], r["seed"], r["official"], r.get("orch_outcome"), r["errors"]) for r in recs]))


TRAJ_PLAIN = ("diag/orch/summary.json", "diag/orch/llm_calls.jsonl", "diag/orch/stack_calls.jsonl", "diag/vlm_calls.jsonl",
              "diag/binding_calls.jsonl", "diag/grasp_candidates.jsonl", "diag/gate_decisions.jsonl")
TRAJ_GZ = ("diag/motion_calls.jsonl", "diag/contact_truth.jsonl", "diag/events.jsonl")


def scrub(text, out):
    for prefix in (str(out) + "/", str(sim_root()) + "/", str(sim_root().parent) + "/"):
        text = text.replace(prefix, "")
    return text


def cmd_export_trajectories(a):
    import gzip
    out, cfg = load(a.out); dest = Path(a.dest).resolve()
    need(not dest.exists(), f"{dest} exists")
    phases = [cfg["phases"]["main"], cfg["phases"]["rerun"]] + sorted(p.name[len("stage_"):] for p in out.glob(f"stage_retry*_{cfg['arm']}"))
    rows = []
    for ph in phases:
        for key, rec in load_phase(out, cfg, ph):
            if rec["errors"] or not (a.all or rec["official"]):
                continue
            rows.append((key, ph, rec))
    seen = set(); index = []
    for (arm, task, seed), ph, rec in rows:
        if (task, seed) in seen and not a.all:
            continue   # one successful episode per row (the first in phase order)
        seen.add((task, seed))
        src = out / rec["episode_dir"]; name = f"{task}_s{seed}" + ("" if not a.all else f"_{ph}")
        d = dest / name; d.mkdir(parents=True)
        m = json.loads((src / "attempt_manifest.json").read_text())
        (d / "attempt_manifest.json").write_text(scrub(json.dumps(m, indent=1, default=str), out) + "\n")
        so = sorted(src.glob("artifacts/*/stdout.txt"))
        if so:
            (d / "stdout.txt").write_text(scrub(so[0].read_text(errors="replace"), out))
        for rel in TRAJ_PLAIN:
            if (src / rel).is_file():
                (d / Path(rel).name).write_text(scrub((src / rel).read_text(errors="replace"), out))
        for rel in TRAJ_GZ:
            if (src / rel).is_file():
                with gzip.open(d / (Path(rel).name + ".gz"), "wt") as fh:
                    fh.write(scrub((src / rel).read_text(errors="replace"), out))
        o = m.get("orch_summary") or {}
        index.append({"episode": name, "task": task, "suite": rec["suite"], "seed": seed, "state_idx": rec.get("state_idx"), "phase": ph,
                      "official": rec["official"], "final_sim_step": rec.get("final_sim_step"), "wall_s": rec["wall_s"],
                      "orch_outcome": rec.get("orch_outcome"), "decisions": [(x.get("action"), x.get("subgoal")) for x in o.get("decisions") or []],
                      "orch_cost_usd": rec.get("orch_cost_usd"), "stack_vb_cost_usd": rec.get("vlm_cost_usd")})
    save(dest / "index.json", {"arm": cfg["arm"], "method": cfg["method"], "program_sha256": cfg["program_sha256"], "stack_sha256": STACK_SHA,
                               "runner_sha256": cfg["runner_sha256"], "source_run": out.name, "n": len(index),
                               "contents": "per episode: attempt_manifest.json, stdout.txt, orchestration summary/llm/stack calls, VLM calls, "
                                           "binding/grasp/gate records; motion, contact truth and API-call events gzipped. Simulator truth is "
                                           "recorded for diagnosis only; the program never sees it.",
                               "episodes": sorted(index, key=lambda r: (r["task"], r["seed"]))})
    print(f"EXPORTED {len(index)} episodes -> {dest}")


# ----------------------------------------------------------------------------- analysis
def transport_errors(d):
    n = 0
    for f in (d / "diag/orch_llm_calls.jsonl", d / "diag/vlm_calls.jsonl"):
        if f.is_file():
            n += sum(1 for l in f.read_text().splitlines() if l.strip() and json.loads(l).get("status") == "error")
    return n


def episode_record(out, cfg, row, phase, lookup):
    sys.path.insert(0, str(out / "code"))
    from adapter_v3 import descriptors_for, REVISION
    arm = row["arm"]; short = lookup.get((row["suite"], row["task"]), ""); key = (arm, short, int(row["seed"]))
    d = out / f"stage_{phase}/attempts" / row["attempt_dir"] / arm / short / f"seed_{int(row['seed']):02d}"
    rec = {"arm": arm, "task_short": short, "suite": row["suite"], "seed": int(row["seed"]), "official": 0, "resolved": False, "runner_rc": int(row["rc"]),
           "wall_s": int(row["wall_s"]), "episode_dir": str(d.relative_to(out)), "phase": phase, "errors": []}

    def chk(ok, why):
        if not ok:
            rec["errors"].append(why)
    chk(not rec["runner_rc"], "runner_error")
    try:
        m = json.loads((d / "attempt_manifest.json").read_text())
        rec.update(official=int(m["official_task_completed"]), resolved=True, sandbox_rc=int(m["sandbox_rc"]), state_idx=m["state_idx"],
                   final_sim_step=m["final_sim_step"], horizon_exhausted=m.get("horizon_exhausted", False), failure_class=m.get("failure_class"),
                   vlm_cost_usd=round(float((m.get("vlm_stats") or {}).get("cost_usd") or 0), 5), transport_errors=transport_errors(d))
        chk((m["arm"], m["task_short"], m["seed"]) == key, "manifest_key_mismatch")
        chk(m["phase"] == phase and m["suite"] == row["suite"] and m["task"] == row["task"], "manifest_identity")
        chk(m["policy_sha256"] == cfg["program_sha256"] == sha(d / "policy.py"), "policy_hash")
        chk(m["adapter_revision"] == REVISION and m["task_descriptors"] == descriptors_for(m["task_language"]), "adapter")
        chk(m["n_init_states"] > 0 and m["state_idx"] == (m["seed"] - 1) % m["n_init_states"], "state_index")
        chk(str(m["official_task_completed"]) == row["task_completed"], "official_mismatch")
        chk(m["sandbox_rc"] == 0 and row["sandbox_rc"] == "0", "sandbox_error")
        chk(not m["recorder_errors"], "recorder_error")
        if m.get("horizon_exhausted"):
            chk(m["truncated"] and m["final_sim_step"] == 4000 and m["official_task_completed"] == 0, "invalid_expected_timeout")
        else:
            chk(not m["truncated"], "unclassified_truncation")
        if arm in ("N20C", "N20O"):
            o = m.get("orch_summary") or {}
            chk(o.get("outcome") not in (None, "orch_summary_missing"), "orch_summary_missing")
            chk((o.get("stack") or {}).get("sha256") == STACK_SHA and o.get("fallback_to_baseline") is False, "orch_identity")
            chk(m.get("orch_files_sha256") == {n: sha(out / "code" / n) for n in ORCH_FILES}, "orch_files_hash")
            n_calls = int((m.get("orch_llm_stats") or {}).get("calls", -1))
            if arm == "N20O":
                chk(o.get("arm_mode") == "llm_orchestration" and n_calls == 1, "oneshot_mode_or_calls")
                rec.update(orch_cost_usd=(o.get("generation") or {}).get("cost_usd"))
            else:
                chk(o.get("arm_mode") == "closed_loop_llm_orchestration" and 1 <= n_calls <= 24 and n_calls == len(o.get("llm_calls") or []), "closed_mode_or_calls")
                rec.update(orch_cost_usd=o.get("cost_usd"), orch_retries=o.get("retries"), orch_replaced=o.get("replaced"))
            rec.update(orch_outcome=o.get("outcome"), orch_llm_calls=n_calls)
    except Exception as exc:  # noqa: BLE001 - recorded as an integrity error
        rec["errors"].append(type(exc).__name__ + ": " + str(exc)[:200])
    return key, rec


def load_phase(out, cfg, phase):
    p = out / f"attempts_log_{phase}.csv"
    if not p.exists():
        return []
    manifest = json.loads((out / "task_manifest.json").read_text())
    lookup = {(v["suite"], v["task"]): k for k, v in manifest["tasks"].items()}
    return [episode_record(out, cfg, r, phase, lookup) for r in csv.DictReader(p.open())]


def wilson(k, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def summarize(out, cfg, with_reruns=False):
    freeze_check(out); ph = cfg["phases"]; exp = planned(out, cfg, ph["main"])
    seen, errors = {}, []
    for key, rec in load_phase(out, cfg, ph["main"]):
        if key not in exp or key in seen:
            errors.append({"type": "unexpected_or_duplicate_row", "key": key}); continue
        seen[key] = rec
    reruns = {}
    if with_reruns:
        for key, rec in load_phase(out, cfg, ph["rerun"]):
            need(key in seen and key not in reruns and seen[key].get("transport_errors"), f"rerun row not infra-flagged: {key}")
            reruns[key] = dict(rec, replaced_original={k: seen[key].get(k) for k in ("official", "episode_dir", "transport_errors")})
    eps = {k: reruns.get(k, v) for k, v in seen.items()}
    errors += [{"key": k, "errors": r["errors"]} for k, r in eps.items() if r["errors"]]
    missing = sorted(exp - set(eps))
    if missing:
        errors.append({"type": "missing_rows", "n": len(missing), "rows": missing[:20]})
    E = list(eps.values()); n_plan = cfg["planned"]
    n_suite = {x: sum(1 for _, t, _ in exp if t.endswith("_" + x)) for x in ("swap", "task")}   # from the plan (subset plans are uneven)
    n_no4 = sum(1 for _, t, _ in exp if not t.startswith("long4_"))
    suites = {x: sum(r["official"] for r in E if r["suite"] == "libero_10_" + x) for x in ("swap", "task")}
    total = sum(r["official"] for r in E)
    no4 = sum(r["official"] for r in E if not r["task_short"].startswith("long4_"))
    s = {"status": "complete" if not errors and not missing else "complete_with_integrity_errors", "timestamp": utc(), "method": cfg["method"],
         "arm": cfg["arm"], "note": "inference method only; the learned stack is N20 (unchanged)", "scale": cfg["scale"], "seeds": cfg["seeds"],
         "with_infra_reruns": with_reruns, "official_successes": total, "planned": n_plan, "resolved": sum(r["resolved"] for r in E),
         "by_suite": {"POS (libero_10_swap)": {"successes": suites["swap"], "of": n_suite["swap"], "wilson95_pct": wilson(suites["swap"], n_suite["swap"])},
                      "TASK (libero_10_task)": {"successes": suites["task"], "of": n_suite["task"], "wilson95_pct": wilson(suites["task"], n_suite["task"])}},
         "without_long4": {"successes": no4, "of": n_no4,
                           "note": "long4 (LIVING_ROOM_SCENE5 two mugs) was used to develop S1, the root of the N-lineage, on seeds 1-50"},
         "per_task": {f"long{i}": {x: sum(r["official"] for r in E if r["task_short"] == f"long{i}_{x}") for x in ("swap", "task")} for i in range(10)},
         "timeouts": sum(bool(r.get("horizon_exhausted")) for r in E), "orch_outcomes": dict(Counter(str(r.get("orch_outcome")) for r in E)),
         "cost_usd": {"orchestration": round(sum(float(r.get("orch_cost_usd") or 0) for r in E), 3), "stack_vb": round(sum(float(r.get("vlm_cost_usd") or 0) for r in E), 3)},
         "wall_s_mean": round(sum(r["wall_s"] for r in E) / max(1, len(E)), 1),
         "infra": {"episodes_with_transport_errors": sorted([list(k) for k, r in seen.items() if r.get("transport_errors")]), "reruns": sorted([list(k) for k in reruns])},
         "errors": errors, "paired": {}}
    D = {f"{k[1]}:{k[2]}": r["official"] for k, r in eps.items()}
    refs = json.loads((REF / "refs/interim_refs.json").read_text())["refs"]
    for name, ref in refs.items():
        common = [k for k in ref["rows"] if k in D]
        if not common:
            continue
        b = sum(1 for k in common if D[k] > ref["rows"][k]); c = sum(1 for k in common if D[k] < ref["rows"][k])
        s["paired"][name] = {"rows": len(common), "this": sum(D[k] for k in common), name: sum(ref["rows"][k] for k in common), "gained": b, "lost": c,
                             "sign_p": round(sign_p(b, c), 4), "method": ref["method"], "stack": ref["stack"]}
    s["episodes"] = sorted(E, key=lambda r: (r["task_short"], r["seed"]))
    save(out / ("results/summary_infra_rerun.json" if with_reruns else "results/summary.json"), s)
    write_report(out, cfg, s)
    return s


def write_report(out, cfg, s):
    scope = cfg["scale"] + (f": the rows {cfg['subset_ref']} solved" if cfg.get("subset_ref") else "")
    L = [f"# {cfg['arm']} ({cfg['method']}) on LIBERO-Long-Pro, {scope} ({cfg['planned']} episodes)", "",
         f"> Inference method only; the learned stack is N20 (`{STACK_SHA[:12]}…`), unchanged.", "",
         f"**Official outcome: {s['official_successes']}/{s['planned']}** (status `{s['status']}`, infra reruns: {s['with_infra_reruns']})", "",
         "| Suite | Successes | Wilson 95% |", "|---|---|---|"]
    for k, v in s["by_suite"].items():
        L.append(f"| {k} | {v['successes']}/{v['of']} | [{v['wilson95_pct'][0]}–{v['wilson95_pct'][1]}] |")
    L += ["", f"Without long4: {s['without_long4']['successes']}/{s['without_long4']['of']} ({s['without_long4']['note']}).", "",
          "| Task | swap | task |", "|---|---|---|"] + [f"| {t} | {v['swap']} | {v['task']} |" for t, v in s["per_task"].items()]
    if s["paired"]:
        L += ["", "| Reference (same rows) | rows | this | ref | gained | lost | sign p |", "|---|---|---|---|---|---|---|"]
        L += [f"| {n} ({v['method']}, {v['stack']}) | {v['rows']} | {v['this']} | {v[n]} | {v['gained']} | {v['lost']} | {v['sign_p']} |" for n, v in s["paired"].items()]
    L += ["", f"Timeouts: {s['timeouts']}. Orchestration outcomes: {s['orch_outcomes']}. Cost: {s['cost_usd']}. Mean wall time {s['wall_s_mean']} s.",
          f"Integrity errors: {len(s['errors'])}. Infra-flagged episodes: {len(s['infra']['episodes_with_transport_errors'])}."]
    (out / "results/REPORT_auto.md").write_text("\n".join(L) + "\n")


def cmd_summarize(a):
    out, cfg = load(a.out)
    s = summarize(out, cfg, with_reruns=(out / f"attempts_log_{cfg['phases']['rerun']}.csv").exists())
    print((out / "results/REPORT_auto.md").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("setup"); q.add_argument("--out", required=True); q.add_argument("--method", required=True, choices=sorted(METHODS))
    q.add_argument("--scale", default="interim", choices=sorted(SEEDS)); q.add_argument("--gpus", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    q.add_argument("--cap-usd", type=float, default=35.0); q.add_argument("--start-cap", type=int, default=None)
    q.add_argument("--subset-ref", default=None, help="with --scale subset: reference whose successful rows form the plan")
    for name in ("start-proxy", "stop-proxy", "check", "smoke", "run", "rerun-infra", "summarize"):
        q = sp.add_parser(name); q.add_argument("--out", required=True)
    q = sp.add_parser("export-trajectories"); q.add_argument("--out", required=True); q.add_argument("--dest", required=True)
    q.add_argument("--all", action="store_true", help="every clean episode (all phases), not only one success per row")
    q = sp.add_parser("retry"); q.add_argument("--out", required=True); q.add_argument("--attempt", type=int, required=True)
    q.add_argument("--rows", nargs="+", required=True)
    a = p.parse_args()
    if a.cmd == "setup" and a.start_cap is None:
        a.start_cap = len(TASKS) * len(SEEDS[a.scale]) * 2 + 10   # main run + one full infra rerun + smoke, at most
    globals()["cmd_" + a.cmd.replace("-", "_")](a)


if __name__ == "__main__":
    main()
