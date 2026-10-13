"""Download and SHA256-check the public ModelScope SAM3 snapshot."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = "https://modelscope.cn/api/v1/models/facebook/sam3"
REVISION = "96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
DEST = Path(os.environ.get("SAM3_MODEL_DIR", str(Path.home() / ".local/share/aspire/models/sam3")))
WEIGHT_HASH = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE + "/repo/files?" + urlencode({"Revision": REVISION, "Recursive": "true"}), timeout=30) as response:
        listing = json.load(response)
    if listing.get("Code") != 200:
        raise RuntimeError("ModelScope file listing failed")
    files = [f for f in listing["Data"]["Files"]
             if f["Type"] == "blob" and f["Path"] not in {"sam3.pt", ".gitattributes"}]
    if next(f["Sha256"] for f in files if f["Path"] == "model.safetensors") != WEIGHT_HASH:
        raise RuntimeError("Unexpected SAM3 weight identity")

    def fetch(f):
        name = f["Path"]
        if Path(name).name != name:
            raise ValueError("Unexpected nested model file")
        target = DEST / name
        if target.is_file() and digest(target) == f["Sha256"]:
            print(f"Verified cached {name}", flush=True)
            return
        partial = target.with_name(target.name + ".partial")
        url = BASE + "/repo?" + urlencode({"Revision": REVISION, "FilePath": name})
        print(f"Downloading {name} ({f['Size']} bytes)", flush=True)
        with urlopen(url, timeout=90) as response, partial.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        if partial.stat().st_size != f["Size"] or digest(partial) != f["Sha256"]:
            raise RuntimeError(f"Size/SHA256 mismatch: {name}")
        partial.replace(target)
        print(f"Verified {name}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    (DEST / "modelscope-manifest.json").write_text(json.dumps({
        "repository": "facebook/sam3", "source": "https://modelscope.cn/models/facebook/sam3",
        "revision": REVISION, "files": files,
    }, indent=2) + "\n")
    print(f"Complete: {DEST}", flush=True)


if __name__ == "__main__":
    main()
