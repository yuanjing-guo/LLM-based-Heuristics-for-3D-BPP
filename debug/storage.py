# debug/storage.py
import os
import json
import re
from datetime import datetime
from typing import Dict, Optional


_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def validate_archive_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Empty name.")
    if not _NAME_RE.match(name):
        raise ValueError("Invalid name. Use only letters, numbers, underscore: [A-Za-z0-9_]")
    return name


def archive_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "heuristics", "llm_archives")


def manifest_path() -> str:
    return os.path.join(archive_dir(), "_manifest.json")


def write_archive(name: str, code: str, meta: Optional[Dict] = None, overwrite: bool = False) -> str:
    """
    Write a standard heuristic plugin to heuristics/llm_archives/<name>.py
    The output is directly runnable in non-debug mode (importable BaseHeuristic subclass).
    """
    name = validate_archive_name(name)

    if not code or "def llm_policy" not in code:
        raise ValueError("Code is empty or does not contain def llm_policy(...)")

    out_dir = archive_dir()
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{name}.py")
    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"Archive '{name}' already exists at {out_path}")

    class_name = "ArchivedHeuristic"

    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

class {class_name}(BaseHeuristic):
    name = "{name}"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template)

    # update manifest (best-effort)
    mpath = manifest_path()
    record = {
        "name": name,
        "file": os.path.relpath(out_path, start=os.path.join(os.path.dirname(__file__), "..")),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta:
        record["meta"] = meta

    try:
        if os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"archives": []}
        data["archives"].append(record)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        # don't fail save if manifest fails
        pass

    return out_path
