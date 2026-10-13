#!/usr/bin/env bash
# Public PyTorch implementation pinned to ASPIRE's simulation SDK revision.
set -euo pipefail
REAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${ASPIRE_CONTACT_GRASPNET_DIR:-$HOME/.local/share/aspire/contact-graspnet}"
PYTHON="${ASPIRE_ROBOT_PYTHON:-$(command -v python3)}"
REVISION=2d71da4e50a04aa353352d1cae99f20f7022145b
REUSE_PYTHON="${ASPIRE_TORCH_PYTHON:-}"
mkdir -p "$PREFIX"
if [[ ! -d "$PREFIX/source/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/elchun/contact_graspnet_pytorch.git "$PREFIX/source"
  # Exclude unrelated PointNet training checkpoints and large test RGB-D archives.
  git -C "$PREFIX/source" sparse-checkout set --no-cone '/*' '!/test_data/' '!/Pointnet_Pointnet2_pytorch/log/'
fi
if [[ "$(git -C "$PREFIX/source" rev-parse HEAD)" != "$REVISION" ]]; then
  git -C "$PREFIX/source" fetch origin "$REVISION"
fi
git -C "$PREFIX/source" checkout --detach "$REVISION"
PATCH="$REAL_ROOT/patches/contact_graspnet_pytorch-compat.patch"
if git -C "$PREFIX/source" apply --unidiff-zero --check "$PATCH"; then
  git -C "$PREFIX/source" apply --unidiff-zero "$PATCH"
elif ! git -C "$PREFIX/source" apply --unidiff-zero --reverse --check "$PATCH"; then
  echo 'Contact-GraspNet source differs from the expected compatibility patch.' >&2
  exit 1
fi
[[ -x "$PREFIX/venv/bin/python" ]] || "$PYTHON" -m venv "$PREFIX/venv"
if [[ -n "$REUSE_PYTHON" ]]; then
  # Optional disk-saving reuse of immutable Python dependencies, never SDK/license files.
  SHARED_SITE="$("$REUSE_PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parents[1])')"
  "$PREFIX/venv/bin/python" - "$SHARED_SITE" <<'PY'
import pathlib, site, sys
path = pathlib.Path(site.getsitepackages()[0]) / 'aspire_shared_dependencies.pth'
path.write_text(sys.argv[1] + '\n')
PY
else
  "$PREFIX/venv/bin/python" -m pip install 'torch==2.8.0' 'numpy<2'
fi
"$PREFIX/venv/bin/python" -m pip install 'numpy<2' scipy pyyaml trimesh tqdm pillow \
  'opencv-python-headless<4.12' fastapi uvicorn 'pyrender==0.1.45' 'pyglet<2'
"$PREFIX/venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$PREFIX/source"
"$PREFIX/venv/bin/python" - "$PREFIX/source" <<'PY'
from pathlib import Path
import hashlib, json, subprocess, sys
root = Path(sys.argv[1])
model = root / 'checkpoints/contact_graspnet/checkpoints/model.pt'
if not model.is_file() or model.stat().st_size < 1000000:
    raise RuntimeError('Pretrained checkpoint is missing or is a Git LFS pointer')
manifest = {'source': 'https://github.com/elchun/contact_graspnet_pytorch',
            'revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
(root.parent / 'installation.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
echo "Installed. Start: bash $REAL_ROOT/tools/vision/launch_contact_graspnet_server.sh"
