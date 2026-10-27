#!/usr/bin/env python3
"""Generic RSI harness CLI (rsi-debugging). Run from anywhere; every command takes --round R.

  init      --authorization A.json            create the round: copy reference inputs, patch the runner, build N0, write state
  freeze    --iter k                          validate + hash iter_k/strategy.json, targets.json, STRATEGY_UPDATE.md (outer loop)
  plan      --phase P --kind K [--group G]    K = inventory | baseline | refresh | candidate | infra | smoke (--rows task:seed ...)
  brief     --phase P --group G               proposer inputs: parent's proposer-visible attribution for the group
  proposer-call --phase P --kind first|revision|resume   count a proposer call against the launch cap (resume is free)
  build     --phase P --group G               candidate program from the CURRENT parent + proposals/P/patch.py (+ tests)
  coverage  --phase P                         offline coverage gate -> proposals/P/coverage_gate.json
  observe   --phase P --tasks T [T ...]       observe-only frames on development seeds -> stage_observe_P/
  run       --phase P                         freeze hashes, run the stage, attribute, analyze, flag infra rows
  analyze   --phase P                         re-analyze a finished stage (no episodes)
  gate      --phase P                         paired gate -> results/P/gate.json (exit 2 = numeric rejection)
  decide    --phase P --decision accept|reject --review FILE
  snapshot  --iter k                          programs/N<k>.py, lineage.json, iter_k/DONE
  aggregates --iter k                         iter_k/aggregates.json: what the outer loop may read about evaluator-only roles
  status
Phase names: inventory<k> | baseline<k>_<arm>_<G> | refresh<k>_<arm>_<G> | chunk<k>_iter<j>[_rev<n>] | <phase>_infra1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsi_lib as L  # noqa: E402

RUNBOOK = HERE.parent if (HERE.parent / "reference").is_dir() else None
CODE_EXCLUDE = {"n19.py", "n19_groups.json", "n13_groups.json", "start_stage_n19.sh", "capability_gate_n19.py", "coverage_gate_n19.py",
                "code_path_exemption_runtime.py", "code_path_exemption_runtime_v3.py", "regression_gate_v6.py", "run_plan_90.sh",
                "run_attempt_90.sh", "run_observe_90.sh", "recheck_identity_90.py", "run_eval.py", "vlm_episode_infra.py", "vlm_integrity.py",
                "analyze_90.py", "analyze_90_v6.py", "build_runner_90.py", "review_zero_motion.py", "add_guards.py", "slip_offline.py",
                "prerelease_data.py", "prerelease_vec.py", "verify_horizon_integration.py", "tests", "evaluator_amendments", "__pycache__"}
HARNESS_FILES = ("rsi.py", "rsi_lib.py", "stage.sh", "attempt.sh", "observe.sh", "vendor/posthoc_first_failure.py")
AUTH_REQUIRED = ("K", "gpus", "start_cap", "start_reserve", "proposer_call_cap", "vlm_usd_cap", "episode_limits", "start_program", "start_strategy")


def sim_root():
    if os.environ.get("RECURSIA_SIM_ROOT"):
        return Path(os.environ["RECURSIA_SIM_ROOT"]).resolve()
    for d in HERE.parents:
        if (d / ".venv-libero").is_dir() and (d / "cap").is_dir():
            return d
    die("set RECURSIA_SIM_ROOT to recursia/sim")


def py():
    return str(sim_root() / ".venv-libero/bin/python3")


def die(msg, code=1):
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(code)


# ----------------------------------------------------------------------------- init
def validate_auth(a):
    miss = [k for k in AUTH_REQUIRED if k not in a]
    L.require(not miss, f"authorization.json missing {miss}")
    L.require(a["start_program"] in L.START_PROGRAMS, f"start_program must be one of {sorted(L.START_PROGRAMS)} (A0 = clean start)")
    L.require(a.get("interface") in L.INTERFACES, f"interface must be one of {L.INTERFACES}")
    L.require(a["interface"] in L.START_INTERFACES[a["start_program"]], f"start program {a['start_program']} needs interface {L.START_INTERFACES[a['start_program']]}")
    L.require(a["start_strategy"] in ("S_1", "warm_N17_N20"), "start_strategy must be S_1 or warm_N17_N20")
    L.require(isinstance(a["K"], int) and a["K"] >= 1, "K >= 1")
    L.require(a["gpus"] and all(isinstance(g, int) for g in a["gpus"]), "gpus = [int, ...]")
    L.require(a["episode_limits"].get("steps") == L.HORIZON and 60 <= a["episode_limits"].get("timeout_s", 0) <= 3600, "episode_limits")


def cmd_init(args):
    R = Path(args.round).resolve()
    L.require(not R.exists() or not any(R.iterdir()), f"round directory not empty: {R}")
    ref = Path(args.reference).resolve() if args.reference else (RUNBOOK / "reference")
    auth = json.loads(Path(args.authorization).read_text()); validate_auth(auth)
    R.mkdir(parents=True, exist_ok=True)
    for sub in ("programs", "results", "proposals", "logs"):
        (R / sub).mkdir()
    shutil.copy2(args.authorization, R / "authorization.json")
    # inputs
    for name in ("task_manifest.json", "ordering_90.txt"):
        shutil.copy2(ref / name, R / name)
    shutil.copytree(ref / "synthetic", R / "synthetic")
    tasks = json.loads((R / "task_manifest.json").read_text())["tasks"]
    L.assert_no_longpro_manifest(tasks)
    for t in L.synthetic_tasks(tasks, "synthetic_dev") + L.synthetic_tasks(tasks, "synthetic_regression"):
        v = tasks[t]
        for suffix, key in ((".bddl", "bddl_sha256"), (".pruned_init", "init_sha256")):
            L.require(L.digest(R / "synthetic/tasks" / (v["task"] + suffix)) == v[key], f"synthetic file hash mismatch: {t}{suffix}")
    # harness code: reference code minus round-specific files, generic runner, harness files
    code = R / "code"; code.mkdir()
    for p in sorted((ref / "code").iterdir()):
        if p.name in CODE_EXCLUDE:
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, code / p.name)
    runner = code / "iso_runner_90.py"
    runner.write_text(L.patch_runner(runner.read_text()))
    for f in HARNESS_FILES:
        shutil.copy2(HERE / f, code / Path(f).name)
    # N0
    src_prog = ref / "programs" / f"{auth['start_program']}.py"
    hashes = (ref / "programs" / L.START_PROGRAMS[auth["start_program"]]).read_text()
    sha = L.digest(src_prog)
    L.require(sha in hashes, f"start program {auth['start_program']} hash not in the reference hash file")
    (R / "programs/N0.py").write_text(L.add_marker(src_prog.read_text(), "N0"))
    with open(R / "programs/hashes.txt", "a") as fh:
        fh.write(f"{L.digest(R / 'programs/N0.py')}  programs/N0.py  # N0 = {auth['start_program']} ({sha[:12]}) + marker\n")
    # iteration 1 strategy template (the outer loop edits targets and may edit the strategy before freezing)
    s1 = L.default_strategy_s1() if auth["start_strategy"] == "S_1" else L.warm_strategy_n17_n20()
    L.write_json(R / "iter_1/strategy_template.json", s1)
    # frozen inputs
    files = sorted(p for p in code.rglob("*") if p.is_file() and "__pycache__" not in str(p))
    files += sorted(p for p in (R / "synthetic").rglob("*") if p.is_file())
    files += [R / "task_manifest.json", R / "ordering_90.txt", R / "authorization.json", R / "programs/N0.py"]
    (R / "prepared_inputs_sha256.txt").write_text(L.hash_manifest(R, files))
    (R / "logs/starts.txt").touch()
    L.Round(R).save_state(L.new_state("N0"))
    if (HERE / "tests").is_dir() and not os.environ.get("RSI_SKIP_INIT_TESTS"):
        r = subprocess.run([py(), "-m", "pytest", "-q", str(HERE / "tests")], capture_output=True, text=True)
        print(r.stdout[-400:])
        L.require(r.returncode == 0, "harness tests failed")
    print(f"INIT_DONE {R}  N0={L.digest(R / 'programs/N0.py')[:12]}  strategy template: iter_1/strategy_template.json")


# ----------------------------------------------------------------------------- helpers
def phase_iter(phase):
    m = re.match(r"^(?:inventory|baseline|refresh|chunk|smoke)(\d+)", phase)
    L.require(m is not None, f"cannot read the iteration from phase {phase}")
    return int(m.group(1))


def proxy_health(auth):
    url = auth.get("proxy_url", "http://127.0.0.1:8117") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            h = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        L.require(False, f"VLM proxy unreachable: {e!r}")
    L.require(h.get("status") == "ok" and h.get("model") == L.MODEL, f"proxy health/model mismatch: {h}")
    L.require(h.get("cap_usd", 0) <= auth["vlm_usd_cap"] + 1e-9, "proxy cap exceeds the authorized VLM cap")
    L.require(h.get("cap_usd", 0) - h.get("spent_usd", 1e9) >= 5.0, f"less than $5 VLM headroom: {h}")
    return h


def starts_used(rnd):
    return sum(1 for _ in rnd.p("logs/starts.txt").open())


def budget_check(rnd, n):
    a = rnd.auth; used = starts_used(rnd)
    L.require(used + n + a["start_reserve"] <= a["start_cap"], f"budget stop: {used} used + {n} planned + reserve {a['start_reserve']} > cap {a['start_cap']}")


# ----------------------------------------------------------------------------- freeze / plan / brief
def cmd_freeze(args):
    rnd = L.Round(args.round)
    s, t = L.freeze_strategy(rnd, args.iter, targets_only=args.targets)
    print(f"FROZEN iter_{args.iter}: pairing={s['pairing']['mode']} groups={list(t['groups']) if t else 'pending (inventory first)'}")


def cmd_plan(args):
    rnd = L.Round(args.round); st = rnd.state(); phase = args.phase; tasks = rnd.manifest
    if args.kind == "infra":
        L.require(phase.endswith("_infra1"), "infra phase must be <base>_infra1")
        base = phase[: -len("_infra1")]; bplan = rnd.plan_roles(base)
        flags = L.read_json(rnd.results(base) / "vlm_infra_episodes.json")["episodes"]
        want = {(b["arm"], b["task_short"], b["seed"]) for b in flags}
        rows = [r for r in bplan["rows"] if (r["arm"], r["task_short"], int(r["seed"])) in want]
        L.require(len(rows) == len(want) > 0, "no infrastructure-flagged rows")
        meta = dict(bplan, phase=phase, kind="infra", base_phase=base, rows=rows)
        print(L.write_plan(rnd, phase, meta), len(rows)); return
    k = phase_iter(phase)
    L.require(k == st["iteration"], f"phase iteration {k} != current iteration {st['iteration']}")
    s = rnd.strategy(k)
    t = rnd.targets(k) if args.kind not in ("inventory", "smoke") else None
    common = dict(phase=phase, kind=args.kind, iteration=k, timeout_rate_max=s["gate"]["timeout_rate_max"], group=args.group,
                  interface=rnd.auth["interface"])
    cur = st["current"]
    if args.kind == "inventory":
        L.require(phase == f"inventory{k}" and s["target_selection"]["inventory"]["run"], "inventory not enabled by S_k or wrong phase name")
        rows = L.missing_rows(rnd, cur, L.inventory_rows(cur, s, tasks))
        meta = dict(common, arm=cur, parent=cur, rows=rows)
    elif args.kind == "smoke":
        L.require(re.match(rf"^smoke{k}_\w+$", phase) is not None and args.rows, "phase must be smoke<k>_<name>, with --rows task:seed ...")
        rows = []
        for spec in args.rows:
            task, seed = spec.rsplit(":", 1)
            L.require(int(seed) in L.DEV_POOL, f"smoke rows use development seeds 86-95 only: {spec}")
            rows.append(dict(arm=cur, task_short=task, seed=int(seed), role="dev"))
        L.assert_rows_source_only(rows, tasks)
        rows = L.missing_rows(rnd, cur, rows)
        meta = dict(common, arm=cur, parent=cur, rows=rows)
    elif args.kind == "baseline":
        L.require(s["pairing"]["mode"] == "full_rerun", "baseline stages belong to full_rerun pairing; use refresh with row_store")
        L.require(args.group in t["groups"] and phase == f"baseline{k}_{cur}_{args.group}", f"phase must be baseline{k}_{cur}_<group>")
        meta = dict(common, arm=cur, parent=cur, rows=L.arm_rows(cur, s, t, args.group, tasks, st))
    elif args.kind == "refresh":
        L.require(s["pairing"]["mode"] == "row_store", "refresh stages belong to row_store pairing")
        L.require(args.group in t["groups"] and phase == f"refresh{k}_{cur}_{args.group}", f"phase must be refresh{k}_{cur}_<group>")
        rows = L.missing_rows(rnd, cur, L.arm_rows(cur, s, t, args.group, tasks, st))
        L.require(rows, "nothing to refresh")
        meta = dict(common, arm=cur, parent=cur, rows=rows)
    elif args.kind == "candidate":
        kk, j, rev, arm = L.cand_arm(phase); L.require(kk == k, "candidate phase iteration mismatch")
        b = L.read_json(rnd.p("proposals", phase, "build.json"))
        decl = L.read_json(rnd.p("proposals", phase, "predeclared.json"))
        L.require(b["arm"] == arm and b["parent"] == cur, f"candidate must be built on the current parent {cur} (sequential rebuild)")
        L.require(decl["arm"] == arm and decl["parent"] == cur and decl["group"] == b["group"], "predeclared.json mismatch")
        L.require(decl["mode"] in s["gate"]["modes"], "predeclared mode not allowed by S_k")
        L.require(decl["mode"] == "outcome" or t["groups"][b["group"]]["mechanism_allowed"], "mechanism mode not allowed for this group")
        L.require(decl["named_tasks"] == t["groups"][b["group"]]["tasks"], "predeclared named_tasks must equal the group's tasks")
        if s["gate"]["coverage_gate_before_staging"]:
            cg = rnd.p("proposals", phase, "coverage_gate.json")
            L.require(cg.exists() and L.read_json(cg).get("ok") is True, "coverage gate missing or failed: the candidate cannot be staged")
        crow = L.arm_rows(arm, s, t, b["group"], tasks, st)
        if s["pairing"]["mode"] == "full_rerun":
            rows = L.arm_rows(cur, s, t, b["group"], tasks, st) + crow
        else:
            miss = L.missing_rows(rnd, cur, L.arm_rows(cur, s, t, b["group"], tasks, st))
            L.require(not miss, f"parent row store incomplete ({len(miss)} rows): run refresh{k}_{cur}_{b['group']} first")
            rows = crow
        meta = dict(common, group=b["group"], arm=arm, parent=cur, rows=rows)
    else:
        die(f"unknown kind {args.kind}")
    print(L.write_plan(rnd, phase, meta), len(meta["rows"]))


def cmd_brief(args):
    rnd = L.Round(args.round); st = rnd.state(); k = phase_iter(args.phase)
    s, t = rnd.strategy(k), rnd.targets(k); g = t["groups"][args.group]; cur = st["current"]
    visible = set(s["debug_data_roles"]["proposer_visible"]) & set(L.PROPOSER_VISIBLE_ROLES)
    syn = set(L.synthetic_tasks(rnd.manifest, "synthetic_dev")) if "synthetic_dev" in visible else set()
    store = L.row_store(rnd, cur, strict=False)
    rows = [r for (task, _), r in sorted(store.items()) if r["role"] in visible and (task in g["tasks"] or task in syn)]
    d = rnd.p("proposals", args.phase, "BRIEF"); d.mkdir(parents=True, exist_ok=False)
    lines = [f"# Proposer brief: {args.phase} (parent {cur}, group {args.group})", "",
             "Proposer-visible rows only (development seeds; synthetic development). Episode directories are relative to the round.", ""]
    for r in rows:
        lines += [f"## {r['task_short']} seed {r['seed']}: official={r['official']} ({r['role']}, phase {r['phase']})",
                  f"episode: {r['episode_dir']}", str(r["first_failure"])] + r["key_lines"] + [""]
    (d / "attribution.md").write_text("\n".join(lines) + "\n")
    obs = rnd.p(f"stage_observe_{args.phase}")
    L.write_json(d / "brief.json", {"phase": args.phase, "parent": cur, "group": args.group, "tasks": g["tasks"],
                                    "failure_classes": g["failure_classes"], "mechanism_allowed": g["mechanism_allowed"],
                                    "vb_changes_allowed": g["vb_changes_allowed"], "patch_scope": s["patch_scope"]["mode"],
                                    "interface": rnd.auth["interface"],
                                    "available_functions": list(L.ORIGINAL_API) if rnd.auth["interface"] == "original" else "reduced API + harness tools (see the parent program)",
                                    "proposer_vlm_usd": s["debug_data_roles"].get("proposer_vlm_usd", 0),
                                    "observe_only_frames": str(obs) if obs.exists() else None, "n_rows": len(rows)})
    print(d / "attribution.md", len(rows))


def cmd_proposer_call(args):
    rnd = L.Round(args.round); log = rnd.p("logs/proposer_calls.jsonl")
    calls = [c for c in L.read_jsonl(log) if c["kind"] != "resume"]
    if args.kind != "resume":
        L.require(len(calls) < rnd.auth["proposer_call_cap"], f"proposer call cap {rnd.auth['proposer_call_cap']} reached")
    L.append_jsonl(log, {"t": time.time(), "phase": args.phase, "kind": args.kind})
    print(f"proposer calls counted: {len(calls) + (args.kind != 'resume')}/{rnd.auth['proposer_call_cap']}")


# ----------------------------------------------------------------------------- build / coverage / observe
def cmd_build(args):
    rnd = L.Round(args.round); st = rnd.state(); phase = args.phase
    k, j, rev, arm = L.cand_arm(phase)
    L.require(k == st["iteration"], "candidate iteration != current iteration")
    s, t = rnd.strategy(k), rnd.targets(k)
    L.require(args.group in t["groups"], f"unknown group {args.group}")
    pdir = rnd.p("proposals", phase); patch = pdir / "patch.py"
    L.require(patch.is_file() and (pdir / "DIAGNOSIS.md").is_file(), "proposal needs DIAGNOSIS.md and patch.py (FINAL report first)")
    L.require(not rnd.program(arm).exists() and not (pdir / "build.json").exists(), f"{arm} already built")
    is_retry = L.slot_check(st, s, k, j, rev, args.group)
    ap = L.load_apply_module(rnd)
    parent = st["current"]
    prog, info = L.apply_proposal(rnd.program(parent).read_text(), patch, s["patch_scope"]["mode"], ap.FORBIDDEN, ap.MODULES, arm,
                                  interface=rnd.auth["interface"])
    nsha = L.normalized_sha(prog)
    if s["gate"]["no_identical_regate"]:
        L.require(all(g["program_norm_sha"] != nsha for g in st["gated"]), "identical to an already gated candidate (no identical re-gate)")
        L.require(nsha != L.normalized_sha(rnd.program(parent).read_text()), "candidate is identical to its parent")
    if s["gate"]["runtime_code_path_exemption"] and not info["runtime_markers"]:
        print("WARNING: no RUNTIME_MARKERS declared; the code-path exemption cannot apply to this candidate")
    out = rnd.program(arm); out.write_text(prog)
    import py_compile
    py_compile.compile(str(out), doraise=True)
    tests = sorted(pdir.glob("test_*.py"))
    L.require(tests, "proposal has no test_*.py")
    env = dict(os.environ, EXPERIMENT_DIR=str(rnd.root), RSI_PROGRAM=str(out))
    r = subprocess.run([py(), "-m", "pytest", "-q", *map(str, tests)], env=env, capture_output=True, text=True, cwd=pdir)
    print(r.stdout[-800:], r.stderr[-300:])
    if r.returncode:
        out.unlink(); die("proposal tests failed; candidate removed")
    sha = L.digest(out)
    with open(rnd.p("programs/hashes.txt"), "a") as fh:
        fh.write(f"{sha}  programs/{arm}.py  # {arm} = {parent} + proposals/{phase} ({','.join(info['touched_modules'])}; {info['failure_class']})\n")
    L.write_json(pdir / "build.json", dict(info, iteration=k, slot=j, rev=rev, arm=arm, parent=parent, group=args.group, program_sha256=sha,
                                           program_norm_sha=nsha, retry=is_retry, patch_sha256=L.digest(patch)), once=True)
    st["slots"].setdefault(str(k), []).append({"j": j, "rev": rev, "arm": arm, "group": args.group, "retry": is_retry, "phase": phase})
    rnd.save_state(st)
    print(f"BUILT {arm} = {parent} + {phase}  sha {sha[:12]}  retry={is_retry}")


def cmd_coverage(args):
    rnd = L.Round(args.round)
    rep = L.coverage_gate(rnd, args.phase)
    print(json.dumps({k: rep.get(k) for k in ("ok", "applicable", "group_not_dispatching", "other_instructions_changed", "unexpected_pairs", "removed_pairs")}, indent=1))
    if not rep["ok"]:
        raise SystemExit(2)


def stage_env(rnd):
    a = rnd.auth
    env = dict(os.environ, RSI_ROUND=str(rnd.root), RSI_GPUS=" ".join(map(str, a["gpus"])), RSI_START_CAP=str(a["start_cap"]),
               RSI_TIMEOUT=str(a["episode_limits"]["timeout_s"]), RECURSIA_SIM_ROOT=str(sim_root()), EXPERIMENT_DIR=str(rnd.root),
               RSI_EXPERIMENT_DIR=str(rnd.root), MUJOCO_GL="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", JAX_PLATFORMS="cpu",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(sim_root()), RSI_INTERFACE=a["interface"],
               LIBERO_CONFIG_PATH=os.environ.get("LIBERO_CONFIG_PATH", str(sim_root() / ".libero")))
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"):
        env.pop(key, None)
    return env


def cmd_observe(args):
    rnd = L.Round(args.round); k = phase_iter(args.phase); s = rnd.strategy(k)
    L.require(s["debug_data_roles"]["observe_only_frames"], "observe-only frames are not enabled by S_k")
    tasks = rnd.manifest
    for x in args.tasks:
        L.require(x in tasks and tasks[x]["suite"] in L.SOURCE_SUITES, f"not a source task: {x}")
    seeds = s["seeds"]["dev"]
    L.require(not rnd.p(f"stage_observe_{args.phase}").exists(), "observe stage exists")
    r = subprocess.run(["bash", str(rnd.p("code/observe.sh")), args.phase, " ".join(map(str, seeds)), *args.tasks], env=stage_env(rnd))
    raise SystemExit(r.returncode)


# ----------------------------------------------------------------------------- run
def cmd_run(args):
    rnd = L.Round(args.round); phase = args.phase
    L.verify_hash_manifest(rnd.root, rnd.p("prepared_inputs_sha256.txt"))
    meta = rnd.plan_roles(phase); plan = rnd.p(f"plan_{phase}.txt")
    k = meta["iteration"]; L.verify_strategy_frozen(rnd, k)
    L.require(not rnd.p(f"stage_{phase}").exists() and not rnd.p(f"frozen_sha256_{phase}.txt").exists(), f"stage {phase} exists")
    L.require(plan.read_text() == "".join(f"{r['arm']} {r['task_short']} {r['seed']}\n" for r in meta["rows"]), "plan text != roles")
    L.assert_rows_source_only(meta["rows"], rnd.manifest)
    st = rnd.state()
    if meta["kind"] in ("inventory", "baseline", "refresh", "smoke"):
        L.require(meta["arm"] == st["current"], "parent stages run only for the current parent")
    if meta["kind"] == "candidate":
        L.require(meta["parent"] == st["current"], "the parent changed since planning: rebuild on the current parent")
    proxy_health(rnd.auth)
    budget_check(rnd, len(meta["rows"]))
    files = sorted(p for p in rnd.p("code").rglob("*") if p.is_file() and "__pycache__" not in str(p))
    files += sorted(p for p in rnd.p("synthetic").rglob("*") if p.is_file())
    files += [rnd.p("task_manifest.json"), rnd.p("ordering_90.txt"), plan, rnd.p(f"plan_{phase}_roles.json")]
    files += [rnd.program(a) for a in sorted({r["arm"] for r in meta["rows"]})]
    files += [rnd.iter_dir(k) / n for n in ("strategy.json", "targets.json") if (rnd.iter_dir(k) / n).exists()]
    if meta["kind"] == "candidate":
        files += sorted(p for p in rnd.p("proposals", phase).rglob("*") if p.is_file() and "__pycache__" not in str(p) and "/work/" not in str(p))
    rnd.p(f"frozen_sha256_{phase}.txt").write_text(L.hash_manifest(rnd.root, files))
    env = stage_env(rnd)
    result = subprocess.run(["bash", str(rnd.p("code/stage.sh")), str(plan), phase], env=env, cwd=rnd.root)
    post = subprocess.run([py(), str(rnd.p("code/posthoc_first_failure.py")), "--out", str(rnd.p("results/posthoc")), str(rnd.root)], cwd=rnd.root)
    if post.returncode:
        raise SystemExit(post.returncode)
    integ = L.analyze_phase(rnd, phase)
    infra = L.flag_infra(rnd, phase)
    print(json.dumps({"integrity": {k2: integ[k2] for k2 in ("planned", "resolved", "timeout_count", "eligible")}, "errors": integ["errors"][:10],
                      "infra_flagged": len(infra["episodes"])}, indent=1))
    if infra["episodes"] and not phase.endswith("_infra1"):
        print(f"INFRA: {len(infra['episodes'])} rows flagged; run: rsi.py plan --phase {phase}_infra1 --kind infra && rsi.py run --phase {phase}_infra1")
    if result.returncode or not integ["eligible"]:
        raise SystemExit(2)
    print("STAGE_DONE", phase, flush=True)


# ----------------------------------------------------------------------------- gate / decide / snapshot
def cmd_analyze(args):
    """Re-analysis of a finished stage (no episode is re-run), e.g. after a harness fix; frozen inputs are still verified."""
    rnd = L.Round(args.round)
    integ = L.analyze_phase(rnd, args.phase)
    L.flag_infra(rnd, args.phase)
    print(json.dumps({k: integ[k] for k in ("planned", "resolved", "timeout_count", "eligible", "errors")}, indent=1))
    if not integ["eligible"]:
        raise SystemExit(2)


def cmd_gate(args):
    rnd = L.Round(args.round)
    rep = L.evaluate_gate(rnd, args.phase)
    st = rnd.state(); b = L.read_json(rnd.p("proposals", args.phase, "build.json"))
    st["gated"].append({"arm": rep["arm"], "phase": args.phase, "program_norm_sha": b["program_norm_sha"], "passes": rep["passes_numeric_gates"], "decision": None})
    rnd.save_state(st)
    print(json.dumps({k: rep[k] for k in ("passes_numeric_gates", "accepted_by", "coverage_ok", "losses_ok", "outcome_ok", "mechanism_ok",
                                          "group_dev_net", "group_selection_net")}, indent=1))
    print(json.dumps(L.gate_aggregates(rep), indent=1))
    if not rep["passes_numeric_gates"]:
        raise SystemExit(2)


def cmd_decide(args):
    rnd = L.Round(args.round); st = rnd.state(); phase = args.phase
    rep = L.read_json(rnd.results(phase) / "gate.json")
    review = Path(args.review).read_text().strip()
    L.require(len(review) >= 40, "the evaluator review must be written (genuine completions, new first failures, hashes)")
    entry = next(g for g in st["gated"] if g["phase"] == phase)
    L.require(entry["decision"] is None, "already decided")
    if args.decision == "accept":
        L.require(rep["passes_numeric_gates"], "a numeric failure can never be accepted")
        L.require(rep["parent"] == st["current"], "the parent changed since gating")
        st["current"] = rep["arm"]
        st["accepted"].append({"arm": rep["arm"], "phase": phase, "group": rep["group"], "tasks": rep["tasks"], "iteration": phase_iter(phase)})
    entry["decision"] = args.decision
    rnd.save_state(st)
    agg = L.gate_aggregates(rep)
    with open(rnd.p("acceptance.md"), "a") as fh:
        fh.write(f"\n## {phase} -> {rep['arm']} (parent {rep['parent']}, group {rep['group']}): **{args.decision.upper()}**\n"
                 f"- numeric: passes={rep['passes_numeric_gates']} by={rep['accepted_by']} dev_net={rep['group_dev_net']} "
                 f"selection_net={rep['group_selection_net']} coverage_ok={rep['coverage_ok']} losses_ok={rep['losses_ok']}\n"
                 f"- per-role counts: {json.dumps(agg)}\n- exempt rows: {sum(len(r['exempt']) for r in rep['roles'].values())}\n"
                 f"- review: {review}\n")
    print(f"DECIDED {phase}: {args.decision}; current parent {st['current']}")


def cmd_snapshot(args):
    rnd = L.Round(args.round); st = rnd.state(); k = args.iter
    L.require(st["iteration"] == k, "snapshot iteration != current iteration")
    d = rnd.iter_dir(k)
    for n in ("RESULT.md", "procedure_observations.md"):
        L.require((d / n).is_file(), f"iter_{k}/{n} missing")
    name = f"N{k}"; L.require(not rnd.program(name).exists(), f"{name} exists")
    src = rnd.program(st["current"]).read_text()
    prog = L.set_marker(src, name)
    rnd.program(name).write_text(prog); sha = L.digest(rnd.program(name))
    with open(rnd.p("programs/hashes.txt"), "a") as fh:
        fh.write(f"{sha}  programs/{name}.py  # {name} = snapshot of {st['current']} (marker only)\n")
    lin = L.read_json(rnd.p("lineage.json")) if rnd.p("lineage.json").exists() else {"snapshots": {}}
    lin["snapshots"][name] = {"sha256": sha, "parent_snapshot": st["snapshots"][str(k - 1)], "source_arm": st["current"],
                              "accepted": [a for a in st["accepted"] if a["iteration"] == k],
                              "strategy_sha256": (d / "strategy_sha256.txt").read_text()}
    L.write_json(rnd.p("lineage.json"), lin)
    st["snapshots"][str(k)] = name; st["current"] = name
    rnd.save_state(st)
    cmd_aggregates(argparse.Namespace(round=args.round, iter=k))
    (d / "DONE").write_text(f"{name} {sha}\n")
    print(f"SNAPSHOT {name} = {lin['snapshots'][name]['source_arm']}  sha {sha[:12]}")


def cmd_aggregates(args):
    rnd = L.Round(args.round); k = args.iter; st = rnd.state()
    out = {"iteration": k, "candidates": [], "stages": {}, "budgets": {}}
    for g in st["gated"]:
        if phase_iter(g["phase"]) != k:
            continue
        rep = L.read_json(rnd.results(g["phase"]) / "gate.json")
        out["candidates"].append({"phase": g["phase"], "arm": g["arm"], "group": rep["group"], "decision": g["decision"],
                                  "passes_numeric_gates": rep["passes_numeric_gates"], "accepted_by": rep["accepted_by"],
                                  "group_dev_net": rep["group_dev_net"], "group_selection_net": rep["group_selection_net"],
                                  "coverage_ok": rep["coverage_ok"], "per_role": L.gate_aggregates(rep),
                                  "mechanism": rep["mechanism"] and {x: rep["mechanism"][x] for x in ("classes", "dev_before", "dev_after", "sel_before", "sel_after")}})
    for p in rnd.p("results").iterdir():
        if p.is_dir() and (p / "integrity.json").exists() and re.match(rf"^(inventory|baseline|refresh|chunk|smoke){k}(?!\d)", p.name):
            i = L.read_json(p / "integrity.json")
            out["stages"][p.name] = {x: i[x] for x in ("planned", "resolved", "timeout_count", "eligible")}
    out["budgets"] = {"starts_used": starts_used(rnd), "start_cap": rnd.auth["start_cap"],
                      "proposer_calls": sum(1 for c in L.read_jsonl(rnd.p("logs/proposer_calls.jsonl")) if c["kind"] != "resume"),
                      "proposer_call_cap": rnd.auth["proposer_call_cap"]}
    L.write_json(rnd.iter_dir(k) / "aggregates.json", out)
    print(rnd.iter_dir(k) / "aggregates.json")


def cmd_status(args):
    rnd = L.Round(args.round); st = rnd.state()
    print(json.dumps({"current": st["current"], "iteration": st["iteration"], "snapshots": st["snapshots"],
                      "accepted": [a["arm"] for a in st["accepted"]], "starts_used": starts_used(rnd), "start_cap": rnd.auth["start_cap"]}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", required=True)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("init"); q.add_argument("--authorization", required=True); q.add_argument("--reference")
    q = sp.add_parser("freeze"); q.add_argument("--iter", type=int, required=True); q.add_argument("--targets", action="store_true", help="second step: freeze targets.json after the inventory")
    q = sp.add_parser("plan"); q.add_argument("--phase", required=True); q.add_argument("--group")
    q.add_argument("--kind", required=True, choices=("inventory", "baseline", "refresh", "candidate", "infra", "smoke"))
    q.add_argument("--rows", nargs="*", help="smoke only: task_short:seed (development seeds)")
    q = sp.add_parser("brief"); q.add_argument("--phase", required=True); q.add_argument("--group", required=True)
    q = sp.add_parser("proposer-call"); q.add_argument("--phase", required=True); q.add_argument("--kind", required=True, choices=("first", "revision", "resume"))
    q = sp.add_parser("build"); q.add_argument("--phase", required=True); q.add_argument("--group", required=True)
    q = sp.add_parser("coverage"); q.add_argument("--phase", required=True)
    q = sp.add_parser("observe"); q.add_argument("--phase", required=True); q.add_argument("--tasks", nargs="+", required=True)
    for name in ("run", "gate", "analyze"):
        q = sp.add_parser(name); q.add_argument("--phase", required=True)
    q = sp.add_parser("decide"); q.add_argument("--phase", required=True); q.add_argument("--decision", required=True, choices=("accept", "reject"))
    q.add_argument("--review", required=True)
    for name in ("snapshot", "aggregates"):
        q = sp.add_parser(name); q.add_argument("--iter", type=int, required=True)
    sp.add_parser("status")
    a = p.parse_args()
    try:
        globals()["cmd_" + a.cmd.replace("-", "_")](a)
    except L.HarnessError as e:
        die(str(e))


if __name__ == "__main__":
    main()
