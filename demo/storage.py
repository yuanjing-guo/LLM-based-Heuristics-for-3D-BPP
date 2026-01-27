# demo/storage.py
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any


def write_archive(
    name: str,
    code: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
    out_dir: str = "heuristics/llm_archives_demo",
) -> str:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-", "."))
    if not safe:
        raise ValueError("Invalid archive name.")

    os.makedirs(out_dir, exist_ok=True)
    py_path = os.path.join(out_dir, f"{safe}.py")
    json_path = os.path.join(out_dir, f"{safe}.json")

    if (not overwrite) and os.path.exists(py_path):
        raise FileExistsError(f"{py_path} already exists (use a different name).")

    # minimal module wrapper so it can be imported later if needed
    module = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

class ArchivedHeuristic(BaseHeuristic):
    name = "{safe}"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
"""

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(module)

    meta_out = meta or {}
    meta_out["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    return py_path
