"""Offline checks of the recursia/sim workspace: self-contained imports, pinned inputs, runner identity, links, third-party pins."""
import hashlib
import re
import subprocess
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parents[1]
EVAL = SIM / "rsi/evaluation"
DEBUG = SIM / "rsi/debugging"
TEXT = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt", ".toml"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def text_files():
    for root in ("cap", "scripts", "rsi", "env_configs", "tests"):
        for p in (SIM / root).rglob("*"):
            if p.is_file() and p.suffix in TEXT and "third_party" not in p.parts and "__pycache__" not in p.parts:
                yield p


def test_workspace_is_independent_of_the_aspire_checkout():
    import getpass
    bad = []
    pats = ["aspire.sim", "ASPIRE_ROOT", "/data/", "/home/", ".claude/", str(SIM.parents[2])]
    user = getpass.getuser()
    if len(user) >= 4:
        pats.append(user)   # the local account name must never be committed
    for p in text_files():
        t = p.read_text(errors="replace")
        for pat in pats:
            if pat in t and p.name != "test_workspace.py":
                bad.append((p.relative_to(SIM).as_posix(), pat))
    assert not bad, bad[:10]


@pytest.mark.parametrize("rel,digest", [
    ("rsi/evaluation/reference/programs/N20.py", "29b187dfaffb4e1bc3a9e8d866637c552453a36fbc2f00691c78dc065f2d7cee"),
    ("rsi/debugging/reference/programs/N20.py", "29b187dfaffb4e1bc3a9e8d866637c552453a36fbc2f00691c78dc065f2d7cee"),
    ("rsi/debugging/reference/programs/A0.py", "afc59057ad17120bacf868968b6f98e394097e752b36b263e0292d89e6a6327b"),
    ("rsi/debugging/reference/programs/K2.py", "1898f9bc3c076e822718b42a1c15c15ab88a9ac487ff953d5b4de87a633bc166"),
    ("rsi/debugging/reference/programs/S1.py", "39bd30bf9a65862b9eca5843547ba3899861704da2c821f90df8542b255f7695"),
])
def test_frozen_programs_are_byte_identical(rel, digest):
    assert sha(SIM / rel) == digest


def test_evaluation_runner_rebuilds_to_the_35_of_100_runner():
    code = EVAL / "reference/code"
    ns = {"__file__": str(code / "build_orch_arm.py"), "__name__": "check"}
    exec(compile((code / "build_orch_arm.py").read_text(), "build_orch_arm.py", "exec"), ns)
    src = (code / "iso_runner_90.py").read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == ns["RUNNER_SHA"]
    for name, old, new in ns["PATCHES"]:
        assert src.count(old) == 1, name
        src = src.replace(old, new)
    built = hashlib.sha256(src.encode()).hexdigest()
    tool = (EVAL / "harness/rsi_eval.py").read_text()
    assert built == sha(code / "iso_runner_orch.py") and f'RUNNER_SHA = "{built}"' in tool
    n20c = (code.parent / "programs/N20.py").read_bytes()[: -len(b"\n\nrun()\n")].decode() + ns["TRAILER"]["N20C"]
    assert hashlib.sha256(n20c.encode()).hexdigest() == "dbe25f1450857b0b2c583e07d42620b98fb07da6c56d189bb66376b950511ec1"


def test_reference_results_reproduce_the_reported_totals():
    import json
    refs = json.loads((EVAL / "reference/refs/interim_refs.json").read_text())["refs"]
    totals = {k: sum(v["rows"].values()) for k, v in refs.items()}
    assert totals == {"N20C": 35, "N18C": 35, "N18O": 33, "N18": 29, "N21C_graspnet": 32, "N21C_anygrasp": 30}
    assert all(len(v["rows"]) == 100 for v in refs.values())


def test_markdown_links_resolve():
    bad = []
    for md in [SIM / "README.md", *SIM.joinpath("rsi").rglob("*.md")]:
        for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]*)?\)", md.read_text()):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).exists():
                bad.append((md.relative_to(SIM).as_posix(), target))
    assert not bad, bad


def test_third_party_lock_and_checkouts():
    rows = [l.split("\t") for l in (SIM / "third_party.lock").read_text().splitlines() if l and not l.startswith("#")]
    assert {r[0] for r in rows} == {"cap/third_party/LIBERO-PRO", "cap/third_party/libero_dependencies/robosuite",
                                    "cap/third_party/sam3", "cap/third_party/contact_graspnet_pytorch"}
    for path, _url, commit in rows:
        if not (SIM / path / ".git").exists():
            pytest.skip("third-party sources not fetched (scripts/fetch_third_party.sh)")
        head = subprocess.run(["git", "-C", str(SIM / path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == commit, path


def test_launchers_find_the_sim_root():
    for rel, line in (("rsi/debugging/harness/launch.sh", 'ROOT=${RECURSIA_SIM_ROOT:-$(cd "$RUNBOOK/../.." && pwd)}'),
                      ("rsi/evaluation/harness/quickstart.sh", 'export RECURSIA_SIM_ROOT=${RECURSIA_SIM_ROOT:-$(cd "$HERE/../../.." && pwd)}')):
        assert line in (SIM / rel).read_text(), rel
    assert "parents[2]" in (SIM / "scripts/libero/replay_trial.py").read_text()
