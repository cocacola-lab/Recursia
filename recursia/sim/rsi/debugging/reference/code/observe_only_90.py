"""Evaluator-only observe pass (round rsi_libero90_20260924, amendment 2026-09-25): record the public observation (agentview rgb/depth/K/E,
robot joint/cartesian) of a dev state of a task whose episodes stop before perception (relation_unsupported), so the proposer has recorded
frames. No program runs; no motion; the same env construction as iso_runner_90.py; the same obs file layout as DiagRecorder2._save_public_obs.
Ground truth (calibration_truth.json) is written for attribution only, exactly as the runner does. usage: observe_only_90.py <task_short> <seed> <outroot>"""
from __future__ import annotations
import copy, json, os, sys, time
from pathlib import Path
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"; os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
RECURSIA_SIM_ROOT = Path(os.environ["RECURSIA_SIM_ROOT"]).resolve(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECURSIA_SIM_ROOT)); sys.path.insert(0, str(RECURSIA_SIM_ROOT / "scripts" / "libero")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "frozen_parent"))
import yaml
from cap.envs.configs.instantiate import instantiate
import replay_trial as rt
import diag_recorder2 as DR
from diag_recorder2 import DiagRecorder2
from iso_runner_90 import calibration_truth, MANIFEST

short, seed, outroot = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
t = MANIFEST["tasks"][short]; suite, task = t["suite"], t["task"]
out = outroot / short / f"seed_{seed}"; (out / "diag").mkdir(parents=True, exist_ok=True)
task_id = rt._find_task_id(suite, task)
config = copy.deepcopy(yaml.safe_load((RECURSIA_SIM_ROOT / "env_configs/libero/franka_libero_libero10_traced.yaml").read_text()))
config["env"]["cfg"]["low_level"] = {"_target_": "cap.envs.simulators.libero.FrankaLiberoEnv", "suite_name": suite, "task_id": task_id,
                                      "privileged": config.get("env", {}).get("cfg", {}).get("privileged", False), "max_steps": 4000, "seed": None, "enable_render": False, "viser_debug": False}
config["trials"] = 1; config["output_dir"] = str(out / "artifacts")
t0 = time.time(); env = instantiate(config["env"]); obs, info = env.reset(seed=seed)
n_states = len(env.low_level_env.handle.init_states); state_idx = (seed - 1) % n_states
(out / "diag" / "calibration_truth.json").write_text(json.dumps(calibration_truth(env), indent=1))
(out / "diag" / "task_descriptors.json").write_text(json.dumps(t["descriptors"], indent=1))
rec = DiagRecorder2(env, out / "diag")
src = "reset"
if not (isinstance(obs, dict) and "agentview" in obs):
    src = None
    for api in env._apis.values():
        fns = api.functions()
        if DR.OBS_FN in fns:
            obs = fns[DR.OBS_FN](); src = f"api:{DR.OBS_FN}"; break
path = rec._save_public_obs(obs, 0)
(out / "observe_manifest.json").write_text(json.dumps({"task_short": short, "suite": suite, "task": task, "seed": seed, "state_idx": state_idx, "n_init_states": n_states,
    "task_language": getattr(env.low_level_env.handle, "task_language", None), "obs_source": src, "obs_path": path, "errors": rec.errors, "setup_s": round(time.time() - t0, 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
print("OBSERVE_DONE", short, seed, src, path, rec.errors)
