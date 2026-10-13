"""Clear one initialized method's outputs and restore its empty learning state."""
import shutil
from pathlib import Path

from .store import METHODS, MethodStore, initial_knowledge, initial_state, write_json


def clear_method(campaign, method, *, dry_run=False):
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    campaign = Path(campaign).expanduser().resolve(strict=True)
    root = campaign / method
    # Never follow a group-directory link into another group or external directory.
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("method workspace must be a real directory directly inside the campaign")
    for name in ("state.json", ".lock"):
        if (root / name).is_symlink():
            raise ValueError(f"method {name} must not be a symlink")
    store = MethodStore(campaign, method)
    if store.meta.get("schema") != "aspire.rsi.campaign.v1" or store.state.get("method") != method:
        raise ValueError("workspace is not an initialized RSI method")
    with store.locked():
        entries = sorted((p for p in root.iterdir() if p.name != ".lock"), key=lambda p: p.name)
        removed = [p.name for p in entries]
        if not dry_run:
            for path in entries:
                if path.is_symlink() or not path.is_dir():
                    path.unlink()
                else:
                    shutil.rmtree(path)
            # Preserve .lock itself so concurrent commands still contend on the same inode.
            store.knowledge_path.write_text(initial_knowledge(method))
            write_json(store.state_path, initial_state(method))
        return {"method": method, "workspace": str(root), "dry_run": dry_run,
                "removed_entries": removed, "reset": not dry_run}
