# Recursia: Real-Robot Experiments

Real-world recursive self-improvement experiments on Franka FR3 with Franka Hand.

This repository contains the physical RSI runner, six perception/control tools,
SAM3 and Contact-GraspNet service adapters, source/target task definitions, and
offline tests. The robot runtime and its kinematics model are self-contained.

## Methods

| Display name | CLI identifier | Persistent source knowledge |
| --- | --- | --- |
| Recursia | `recursia` | A shared Python module, updated through paired source-task validation |
| Skill Document Only (SD-only) | `sd-only` | One natural-language skill document |
| ASPIRE | `aspire` | Topic skill documents and promoted Python functions |

Each method has a separate campaign workspace. Targets use frozen source
knowledge; corrections within a target episode do not update that knowledge.

The six tools are `get_observation`, `get_object_pose`, `sample_grasp_pose`,
`goto_pose`, `open_gripper`, and `close_gripper`. The observation tool returns
front/wrist RGB-D data. Object geometry uses SAM3; grasp proposals use
Contact-GraspNet.

All commands below run from `Recursia/recursia/real/` with the `recursia`
Conda environment active. The setup sequence is: install dependencies and
weights, configure the station, check services and cameras, develop source
skills, and evaluate targets.

## Station requirements

- One Franka FR3 with Franka Hand, FCI enabled, Ethernet connectivity, and a
  robot system version compatible with the installed
  [franky-control SDK](https://github.com/timschneider42/franky).
- Front and wrist RealSense RGB-D cameras, each connected over USB 3. The
  capture profile is aligned color/depth at 640 x 480 and 30 FPS.
- Linux x86_64 with Conda and the robot SDK's real-time setup. The station
  self-check requires a hard real-time priority limit of at least 80
  (`ulimit -Hr`).
- An NVIDIA GPU and a driver compatible with PyTorch 2.8.0 (the SAM3
  installer selects CUDA 12.8); the two perception models run in separate environments.
- An authenticated `codex` executable on `PATH` for the default provider.
  The runner also supports the `claude` CLI selected during campaign creation.
- SAM3 weights and the Contact-GraspNet checkpoint, plus measured camera
  calibration and workspace bounds for your station.

## Installation

Use Linux x86_64 with Conda installed. Create and activate the `recursia`
environment from a fresh checkout:

```bash
git clone https://github.com/cocacola-lab/Recursia.git
cd Recursia/recursia/real
conda env create -f environment.yml
conda activate recursia
```

This installs Python 3.11 and the editable runner, with the robot dependencies
pinned to the versions used on the experimental station: NumPy 1.26.4,
SciPy 1.11.4, OpenCV Contrib 4.10.0.84, PyYAML 6.0.3, MuJoCo 3.10.0,
RealSense 2.58.2.10647, and `franky-control` 2.0.0. The latter provides the
`franky` Python module and its libfranka runtime. Pytest 7.4.4 is included for
offline checks. The FR3 DLS IK solver, MJCF model, and all referenced meshes
are included under `robot/franka/`; no external robot project or ROS workspace
is needed. A physical FR3 must have FCI enabled and a compatible robot system
version; follow the robot SDK's deployment requirements for real-time control.

Use `conda activate recursia` in each new terminal before running the commands
below. Perception runs in separate GPU environments created by the installers.
The coding-agent CLI is installed separately from the Python environment.

Install SAM3 and download its weights. For the Hugging Face route, first obtain
access to [facebook/sam3](https://huggingface.co/facebook/sam3), then log in
with that account:

```bash
bash tools/vision/install_sam3.sh
bash tools/fr3_sam3.sh login
bash tools/fr3_sam3.sh download
```

The installer creates `~/.local/share/aspire/sam3-venv/`; the Hugging Face
weights use its normal cache. The available ModelScope download route is
`bash tools/fr3_sam3.sh download-modelscope`. To use an existing local model,
set `SAM3_MODEL_DIR` to its directory before starting services.

Install Contact-GraspNet in its own environment:

```bash
ASPIRE_ROBOT_PYTHON="$(command -v python)" \
  bash tools/vision/install_contact_graspnet.sh
```

The Contact-GraspNet installer fetches a pinned external implementation and
applies the included compatibility patch. Perception checkpoints and their
licenses are managed by those external installations. The installer stores the
source, checkpoint, and environment under
`~/.local/share/aspire/contact-graspnet/`. The inherited directory names and
`ASPIRE_*` environment-variable names are retained by these scripts.

Verify the robot-side packages without accessing hardware:

```bash
python -m pip check
python -c "import franky, pyrealsense2, mujoco, cv2; print('Robot dependencies ready')"
command -v codex
```

## Local configuration

```bash
cp config/hardware.example.yaml config/hardware.local.yaml
cp config/safety.example.yaml config/safety.local.yaml
cp config/rsi_fr3_station.example.yaml config/rsi_fr3_station.local.yaml
cp config/fr3_station.example.yaml config/fr3_station.local.yaml
```

Set the robot IP and both camera serial numbers in `hardware.local.yaml`.
Set the measured workspace bounds in `safety.local.yaml`. Supply the matching
front-camera serial and measured camera-to-base transform in
`fr3_station.local.yaml`. The manipulation configuration references these local
files and specifies the motion dynamics; no teleoperation configuration is needed.
The example deliberately has a null transform and cannot be used for motion.
The transform is `T_base_camera_optical`: it maps points from the front camera's
color optical frame into the robot base frame, with translations in meters.
The wrist camera supplies observation images; this configuration's object
geometry uses the front camera.

Check the completed configuration without opening cameras or connecting to FCI:

```bash
python -c "from cap.saved_scripts.fr3_placement.config import load_config; load_config('config/rsi_fr3_station.local.yaml'); print('Station configuration valid')"
```

This checks configuration consistency, not calibration accuracy. Use measured
extrinsics for your station. `python tools/fr3_calibrate_camera.py --help`
describes the included marker-based refinement helper; it requires an initial
camera transform and is not a replacement for initial calibration.

The runtime prompt code is included. Independent protocol/prompt documents,
other papers, pre-trained stacks, accumulated skills, and experiment logs are excluded.

## Service operation

With the station configured, start SAM3 and Contact-GraspNet and run the
read-only robot, hand, and camera checks:

```bash
python tools/fr3_services.py start --hardware-config config/hardware.local.yaml
python tools/fr3_services.py status
```

`start` launches missing perception services, waits for readiness, and checks
FR3/Hand state, both cameras, and real-time limits. It does not issue arm or
hand motion, unlock the robot, or perform error recovery. Services expose
`/health` at `http://127.0.0.1:6767` (SAM3) and
`http://127.0.0.1:8115` (Contact-GraspNet). A successful combined report has
`ok: true`; `status` checks perception health only.

Useful lifecycle commands:

```bash
# Repeat hardware and perception checks without starting services.
python tools/fr3_services.py check --hardware-config config/hardware.local.yaml

# Stop only service processes recorded as owned by this launcher.
python tools/fr3_services.py stop
```

Service logs and self-check reports are written to `logs/fr3_services/`.
For manual foreground operation, run each service in its own terminal:

```bash
bash tools/fr3_sam3.sh start
bash tools/vision/launch_contact_graspnet_server.sh --host 127.0.0.1 --port 8115
```

The camera viewer is available before a trial:

```bash
python tools/fr3_camera_view.py --config config/rsi_fr3_station.local.yaml
```

Close the viewer before checks or execution so the runner can acquire the
camera. The front and wrist cameras are opened by the runner during trials.

## Run

Use an interactive terminal for manual reset and scoring. The default provider
is Codex with GPT-6 Astra, medium reasoning effort, and Fast mode requests.
Model-provider credentials stay with the authenticated CLI.

Create a campaign without connecting to hardware:

```bash
python run_agent.py rsi init --campaign logs/my_campaign
```

Initialization creates empty method workspaces and records the model/provider.
This release uses `recursia`, `sd-only`, and `aspire` as method identifiers.
Create a fresh campaign for the renamed methods; historical experiment
archives are left unchanged.
An existing campaign is resumed with `source` or `eval`; do not initialize over
its directory. To select another provider or model at creation, use the
`--provider` and `--model` options shown by `python run_agent.py rsi init --help`.

### Source development

Recursia runs three source chunks: S1/S2, S3/S4, and S5/S6. Each command runs
the current module on the two tasks, generates up to two candidate updates,
validates candidates on the same pair, and completes update selection. The
operator resets the scene and presses Enter before each trial, scores each
trial with `s`/`f`, and chooses `1` to continue or `0` to discard and retry the
current task. Input `0` retries that task, not the entire chunk. Repeating the
command resumes unfinished chunk state.

```bash
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 python run_agent.py rsi source \
  --campaign logs/my_campaign --method recursia --chunk 1 \
  --config config/rsi_fr3_station.local.yaml
# Repeat for --chunk 2, then --chunk 3.
```

SD-only executes one source task per command and updates its document after
manual scoring. Replace S1 with S2 through S6 for a full pass:

```bash
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 python run_agent.py rsi source \
  --campaign logs/my_campaign --method sd-only --task S1 \
  --config config/rsi_fr3_station.local.yaml
```
### Target evaluation

After source development, prepare the target scene and evaluate a target
independently, replacing T1 with T2 or T3 as needed:

```bash
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 python run_agent.py rsi eval \
  --campaign logs/my_campaign --method recursia --task T1 \
  --config config/rsi_fr3_station.local.yaml
```

Target execution pins source knowledge for the trial and does not write back
learned changes. Target trials permit at most 20 code blocks; source limits are
configured at initialization. Model and tool calls are logged, but this inherited
runner does not preserve exact input/output token usage.

### Progress and outputs

```bash
python run_agent.py rsi status --campaign logs/my_campaign --method recursia
```

Method data is isolated under `logs/my_campaign/recursia/`,
`logs/my_campaign/sd-only/`, and `logs/my_campaign/aspire/`. Recursia persists
its module in `stack.py`; SD-only persists its document in `skills.md`.
Trial folders retain generated code, observations, execution traces, results,
and video segments. Target evaluations keep their own frozen knowledge
snapshot. Reset and rollback utilities are listed below.

## Tasks

| Task | Goal |
| --- | --- |
| S1 | Mug in basket |
| S2 | Carton of milk in basket |
| S3 | White mug on left plate |
| S4 | Blue mug on right plate |
| S5 | Plastic bowl on plate |
| S6 | Mug in plastic bowl |
| T1 | Mug and carton of milk in the same basket |
| T2 | Blue mug on left plate; white mug on right plate |
| T3 | First bowl on plate, then mug in relocated bowl |

These are LIBERO-inspired physical adaptations. Left/right is defined in the
front-camera image. Trial completion and a model's FINISH message are not
interchangeable with task success.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Coding-agent CLI not found | Put the authenticated CLI on `PATH`; for Codex, `ASPIRE_CODEX_EXECUTABLE` can select an existing executable. |
| Invalid or missing station transform | Fill in `fr3_station.local.yaml`; its serial must match the front camera in `hardware.local.yaml`. |
| SAM3 missing weights or access denied | Complete the model-access and download steps, or set `SAM3_MODEL_DIR` to valid local weights. |
| Contact-GraspNet installation incomplete | Check the installer output and the checkpoint under its `source/checkpoints/contact_graspnet/` directory. |
| Service loading times out | Inspect `logs/fr3_services/`; `--timeout-s` changes the readiness wait. A timed-out loading process may still be running. |
| Camera frame timeout or USB error | Close other camera clients, verify serials and USB 3 links, then repeat `check`. |
| Robot or hand state is not ready | Inspect the self-check's robot mode/error fields, FCI connection, and competing controllers. |
| Real-time priority check fails | Check `ulimit -Hr` and complete the robot SDK's host setup; the hard limit must be at least 80. |

## Utilities and tests

- `python tools/fr3_camera_view.py --config config/rsi_fr3_station.local.yaml`
  displays the front camera.
- `python tools/fr3_return_pose.py restore --config config/rsi_fr3_station.local.yaml --delay-s 3`
  returns to the default joint pose after the delay; motion requires the same
  `OPENFORGE_ALLOW_PHYSICAL_MOTION=1` opt-in. Use `--pose` for a local bookmark.
- `python tools/rsi_rollback_agent.py --help` and
  `python tools/rsi_clear_agent.py --help` describe knowledge rollback/reset.
- `python tools/fr3_calibrate_camera.py --help` lists marker-based calibration
  helpers. Collection uses your local manipulation config and an initial
  calibration; it is not an automatic robot-motion calibration routine.

The offline suite uses the bundled arm model and requires no camera, service,
or robot connection:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q
```

The IK test loads the bundled model. Source and
target observations, scoring, and videos are written under `logs/`, which is
excluded from Git. Videos are recorded in segments around action calls;
observation capture interrupts recording to release the camera.

## Attribution

ASPIRE copyright notices are preserved. See [LICENSE](../../LICENSE), [NOTICE](../../NOTICE),
and [third-party licenses](../../LICENSES/THIRD_PARTY_LICENSES.md). External perception and robot SDK packages, and the bundled model assets,
retain their own terms. The public paper is available at `../../assets/Recursia.pdf`. This repository
contains no independent protocol/prompt documents and does not claim to reproduce the full
ASPIRE simulation benchmark.
