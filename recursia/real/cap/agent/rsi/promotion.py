# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Method-local adaptation of sim/scripts/libero/record_skill_promotion.py.

Keep before/after snapshots, exact patches and an append-only promotion ledger.
The real runner supplies source evidence and model-authored Markdown updates;
this module never executes generated snippets or accesses the robot.
"""
import difflib
import json
import os
import re
import time
import tempfile
from datetime import datetime, timezone

from .store import digest, write_json, read_json

SKILL_NAME = re.compile(r"[A-Za-z0-9_-]+\.md\Z")


def documents(root):
    folder = root / "skills"
    if folder.is_symlink():
        raise ValueError("skill directory must belong to this method")
    if not folder.exists():
        return {}
    result = {}
    for path in sorted(folder.glob("*.md")):
        if path.is_symlink() or not path.is_file() or not SKILL_NAME.fullmatch(path.name):
            raise ValueError("invalid method-local skill document")
        result[path.name] = path.read_text()
    return result


def library_files(store):
    if store.knowledge_path.is_symlink() or (store.root / "skill_library.json").is_symlink():
        raise ValueError("skill files must belong to this method")
    result = {store.knowledge_path.name: store.knowledge}
    result.update({"skills/" + name: content for name, content in documents(store.root).items()})
    functions = store.root / "skill_library.json"
    if functions.is_file():
        result[functions.name] = functions.read_text()
    return result


def hashes(files):
    return {name: digest(content) for name, content in files.items()}


class Promotion:
    def __init__(self, store, task_id, source_folder):
        if store.method != "aspire" or store.state["frozen"]:
            raise ValueError("skill promotion requires unfrozen ASPIRE source learning")
        self.store = store
        self.folder = store.root / "skill_promotions" / str(time.time_ns())
        self.before = library_files(store)
        self.folder.mkdir(parents=True)
        for name, content in self.before.items():
            path = self.folder / "before" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.begin = {"schema": "aspire.rsi.skill_promotion.v1", "promotion_id": self.folder.name,
                      "method": store.method, "task_id": task_id,
                      "source_path": str(source_folder.relative_to(store.root)),
                      "began_at": datetime.now(timezone.utc).isoformat(),
                      "skill_sha256_before": hashes(self.before)}
        write_json(self.folder / "begin.json", self.begin)

    def apply(self, updates):
        """Only complete Markdown replacements within this method's skills/."""
        if self.store.state["frozen"]:
            raise ValueError("cannot update frozen skills")
        current = documents(self.store.root)
        seen = set()
        for update in updates:
            if (not isinstance(update, dict) or set(update) != {"file", "content"}
                    or not isinstance(update["file"], str) or not SKILL_NAME.fullmatch(update["file"])
                    or not isinstance(update["content"], str) or not update["content"].strip()):
                raise ValueError("skill update requires a local .md filename and nonempty content")
            if update["file"] in seen:
                raise ValueError("duplicate skill update filename")
            seen.add(update["file"])
        folder = self.store.root / "skills"
        folder.mkdir(exist_ok=True)
        written = []
        try:
            for update in updates:
                path = folder / update["file"]
                with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False) as stream:
                    stream.write(update["content"])
                    temporary = stream.name
                try:
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                written.append(update["file"])
        except Exception:
            for name in written:
                path = folder / name
                if name in current:
                    path.write_text(current[name])
                else:
                    path.unlink()
            raise

    def finish(self, *, reason, status="completed", findings_path=None):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("promotion requires an update or no-op reason")
        after = library_files(self.store)
        before_hashes, after_hashes = hashes(self.before), hashes(after)
        changes = [name for name in sorted(set(self.before) | set(after))
                   if before_hashes.get(name) != after_hashes.get(name)]
        chunks = []
        for name in changes:
            chunks.extend(difflib.unified_diff(
                self.before.get(name, "").splitlines(keepends=True),
                after.get(name, "").splitlines(keepends=True),
                fromfile="a/" + name if name in self.before else "/dev/null",
                tofile="b/" + name if name in after else "/dev/null"))
        patch = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                        for line in chunks)
        (self.folder / "changes.patch").write_text(patch)
        for name, content in after.items():
            path = self.folder / "after" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        record = {**self.begin, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "status": status, "reason": reason, "no_op": not changes,
                  "changed_skill_files": changes, "skill_sha256_after": after_hashes,
                  "patch_sha256": digest(patch), "findings_path": str(findings_path.relative_to(self.store.root))
                  if findings_path else None,
                  "findings_sha256": digest(findings_path.read_text()) if findings_path else None}
        write_json(self.folder / "record.json", record)
        ledger = self.store.root / "skill_promotions.jsonl"
        with ledger.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.verify()
        return record

    def verify(self):
        record = read_json(self.folder / "record.json")
        for stage in ("before", "after"):
            for name, checksum in record[f"skill_sha256_{stage}"].items():
                if digest((self.folder / stage / name).read_text()) != checksum:
                    raise ValueError("skill promotion snapshot hash changed")
        if digest((self.folder / "changes.patch").read_text()) != record["patch_sha256"]:
            raise ValueError("skill promotion patch hash changed")
        if hashes(library_files(self.store)) != record["skill_sha256_after"]:
            raise ValueError("skill promotion does not match the saved library")
        if record["findings_path"] and digest((self.store.root / record["findings_path"]).read_text()) != record["findings_sha256"]:
            raise ValueError("skill promotion findings hash changed")
