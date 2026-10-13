# Contributing

Use `recursia/real/` as the working directory for the physical runner. Describe the
behavior change and relevant validation in your pull request.

Run the offline checks before submitting a runtime change:

```bash
cd recursia/real
conda activate recursia
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q cap robot tools
```

The IK test uses the bundled FR3 model and MuJoCo installed by `environment.yml`.
Tests do not require a live robot.

Preserve source/target isolation and upstream copyright notices. Keep local
configuration, credentials, weights, logs, raw recordings, papers, and independent
protocol documents out of commits. Use English for documentation and messages.

Simulation contributions belong under `recursia/sim/` and should include their own setup
and execution instructions.

Selected demonstration videos belong in `assets/media/`; keep complete runtime
recordings under ignored log directories.
