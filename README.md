# Recursia: Recursive Self-Improvement for Robot Manipulation

[Project Page](https://cocacola-lab.github.io/Recursia-website/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2610.24464) &ensp;|&ensp; [Real-Robot Quick Start](recursia/real/README.md) &ensp;|&ensp; [Simulation](recursia/sim/README.md)

Recursia improves a shared executable Python stack from source-task feedback,
then reuses the frozen stack on new task compositions. Improvements persist as
programs rather than model-weight updates.

This repository provides experiments on a Franka FR3 with a Franka
Hand, including a Skill Document Only (SD-only) baseline, and simulation
experiments under `recursia/sim/`: RSI on LIBERO-90 and zero-shot evaluation on
LIBERO-Long-Pro with rule-based, one-shot and closed-loop inference.

## Quick Start with Real-robot Setup

Create the environment for the Franka FR3 workspace:

```bash
git clone https://github.com/cocacola-lab/Recursia.git
cd Recursia/recursia/real
conda env create -f environment.yml
conda activate recursia
```

Follow the [real-robot setup guide](recursia/real/README.md) to configure your
station and install/start SAM3 and Contact-GraspNet. Once ready, run the first
source chunk:

```bash
python run_agent.py rsi init --campaign logs/my_campaign
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 python run_agent.py rsi source \
  --campaign logs/my_campaign --method recursia --chunk 1 \
  --config config/rsi_fr3_station.local.yaml
```

See [Run](recursia/real/README.md#run) for the remaining chunks, SD-only baseline,
and target evaluation.

## Quick Start with Simulation Setup

Reproduce the small interim LIBERO-Long-Pro evaluation (100 episodes) of the frozen N20 shared stack with closed-loop inference (reference: **35/100**). Clone the repository, export your key, and give Claude Code one sentence:

```bash
git clone https://github.com/cocacola-lab/Recursia.git && cd Recursia/recursia/sim
export ANTHROPIC_API_KEY=...      # used only by the local model proxy
claude "Set up the simulation and run the N20C interim Long-Pro evaluation, then report the success rate."
```

Claude Code follows [`recursia/sim/CLAUDE.md`](recursia/sim/CLAUDE.md):
- fetches the pinned simulators and models, builds the environments and runs the tests;
- starts the perception servers and the evaluation;
- reports the result from `results/REPORT_auto.md`.

You need Linux with NVIDIA GPUs, `uv`, and access to the gated [SAM3](https://huggingface.co/facebook/sam3) weights. The run takes about 2.5 h on 5 GPUs and costs about $15–20.

<details>
<summary>Manual commands</summary>

```bash
cd Recursia/recursia/sim
bash scripts/fetch_third_party.sh
uv venv .venv-libero --python 3.12 && source .venv-libero/bin/activate
uv sync --locked --active --extra libero --extra contactgraspnet --extra dev && deactivate
bash scripts/write_libero_config.sh
uv venv .venv-vlmproxy --python 3.12 && uv pip install --python .venv-vlmproxy/bin/python3 "anthropic>=1.11"
.venv-libero/bin/hf auth login
tmux new -d -s recursia-perception "bash scripts/start_perception_servers.sh --gpu-sam3 0 --gpu-graspnet 1; sleep infinity"
GPUS="3 4 5 6 7" bash rsi/evaluation/harness/quickstart.sh closed interim
```
</details>

LLM and grasp sampling are not seed-controlled, so expect roughly 35–42.

The same tool runs the other two inference methods: `quickstart.sh rule` (the stack's own rule-based driver) and `quickstart.sh oneshot`. [`recursia/sim/README.md`](recursia/sim/README.md) covers the full setup, all three evaluation methods, and new RSI debugging runs.

## Demonstrations

Selected successful Recursia target executions on Franka FR3. Videos play at
original speed and concatenate recorded action segments; unrecorded perception
and model-wait intervals are omitted. Click a preview to open the full MP4.
Animated previews below run at 4x speed.

<table>
  <thead>
    <tr><th>Task</th><th>Composition</th><th>Video</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>T1: Shared basket</td>
      <td>Place both the mug and the carton of milk in the basket.</td>
      <td><a href="assets/media/T1.mp4"><img src="assets/media/T1-preview.gif" alt="T1 physical robot demonstration preview" width="280"></a><br><a href="assets/media/T1.mp4">Watch T1</a></td>
    </tr>
    <tr>
      <td>T2: Reassigned plates</td>
      <td>Place the blue mug on the left plate and the white mug on the right plate.</td>
      <td><a href="assets/media/T2.mp4"><img src="assets/media/T2-preview.gif" alt="T2 physical robot demonstration preview" width="280"></a><br><a href="assets/media/T2.mp4">Watch T2</a></td>
    </tr>
    <tr>
      <td>T3: Bowl--mug dependency</td>
      <td>First place the bowl on the plate, then place the mug in the relocated bowl.</td>
      <td><a href="assets/media/T3.mp4"><img src="assets/media/T3-preview.gif" alt="T3 physical robot demonstration preview" width="280"></a><br><a href="assets/media/T3.mp4">Watch T3</a></td>
    </tr>
  </tbody>
</table>

## Repository layout

```text
Recursia/
├── recursia/
│   ├── real/             # Physical RSI runner, tools, configs and tests
│   │   ├── cap/
│   │   ├── config/
│   │   ├── patches/
│   │   ├── robot/
│   │   ├── tools/
│   │   ├── environment.yml
│   │   ├── pyproject.toml
│   │   └── run_agent.py
│   └── sim/              # Simulation: LIBERO runtime (cap/), RSI debugging and evaluation (rsi/), scripts, tests
├── assets/media/         # Selected target demonstration videos
├── LICENSES/
├── LICENSE
├── NOTICE
└── README.md
```

Logs, raw experiment recordings, model checkpoints, local station configurations, papers, and
independent experiment protocol documents are excluded from this repository.
Runtime prompt code required by the agents is included.

## Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md). Run the offline tests from `recursia/real/` when
changing the physical runner, and `.venv-libero/bin/python3 -m pytest -q` from `recursia/sim/` when changing the simulation
workspace. Keep simulation dependencies and coordinates
separate from real-robot configuration.

## Contact

For questions about Recursia, please contact
[wjhan@bjtu.edu.cn](mailto:wjhan@bjtu.edu.cn) or
[chenyangzhi@bjtu.edu.cn](mailto:chenyangzhi@bjtu.edu.cn).

## License

Inherited ASPIRE copyright notices are retained. Project material is available
under the [Apache License 2.0](LICENSE), subject to component-specific terms.
See [NOTICE](NOTICE) and the [license inventory](LICENSES/README.md).

## Citation

If you find Recursia useful in your research, please cite:

```bibtex
@misc{chen2026learningcomposerecursiveselfimprovement,
  title         = {Learning to Compose: Recursive Self-Improvement of Shared Executable Stacks for Zero-Shot Long-Horizon Robot Manipulation},
  author        = {Yangzhi Chen and Feiran Liu and Zishuo Wang and Zijian Jin and Xin Zhu and Lecheng Ruan and Wenjuan Han},
  year          = {2026},
  eprint        = {2610.24464},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.24464}
}
```
