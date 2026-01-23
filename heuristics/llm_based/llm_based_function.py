import os
from typing import Optional

from heuristics.llm_based.llm_interface import call_llm

def _extract_code(text: str) -> str:
    if "```" not in text:
        body = text.rstrip()
    else:
        parts = text.split("```")
        if len(parts) < 3:
            body = text.rstrip()
        else:
            body = parts[1].rstrip()
            if body.lower().startswith("python"):
                body = body.split("\n", 1)[1] if "\n" in body else ""

    cleaned_lines = []
    for line in body.splitlines():
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).rstrip()

#生成给 LLM 的提示词（prompt）
def build_prompt(previous_code: Optional[str], feedback: Optional[str]) -> str:
    #basic rules (To-be-adjusted)
    base = (
        "Write ONLY the body of a Python function named llm_policy(heur, obs).\n"
        "Return a tuple (box_slot, rot_id, x, y).\n"
        "Use only these helpers from heur: get_slot_props(obs, slot_i), "
        "slot_is_empty(obs, slot_i), props_to_size_bins(props), rotate_size_bins(size, rot_id).\n"
        "obs only has 'buffer' and 'pallet_obs_density'.\n"
        "Do NOT import anything. Do NOT define classes.\n"
        "Indent each line by 4 spaces.\n"
        "Return only the function body (no def line).\n"
        "Use helper methods as heur.get_slot_props(...), heur.slot_is_empty(...), "
        "heur.props_to_size_bins(...), heur.rotate_size_bins(...).\n"
        "obs['buffer'] is a numpy array; do NOT use it in boolean context. "
        "Use len(obs['buffer']) or obs['buffer'].size instead.\n"
        "pallet_obs_density is 3D (X,Y,H); use heur.X, heur.Y, heur.H for sizes.\n"
        "Before using props_to_size_bins, ensure the slot is not empty "
        "(heur.slot_is_empty(obs, slot_i) == False).\n"
        "If size_bins is empty or has any non‑positive values, skip that slot.\n"
        "Do NOT access self.feasibility attributes; only call "
        "heur.feasibility.is_within_pallet(...) and heur.feasibility.is_feasible(...).\n"
        "slot_is_empty/get_slot_props only accept slot index (int), not (x,y,z).\n"
        "Never use numpy arrays in boolean context (no `if size_bins` or `if obs['buffer']`). "
        "Use size_bins.size > 0 or np.all(size_bins > 0).\n"
        "Function body MUST NOT contain any import statements.\n"
        "Call heur.feasibility.is_within_pallet(x, y, dx, dy) with 4 scalars. Never pass a tuple.\n"
        "Call heur.feasibility.is_feasible(pallet_obs, x, y, dx, dy, dz, z) with 7 arguments. Do NOT pass box_slot/rot_id.\n"

    )


    if not previous_code:
        return base + "\nStart from a simple deterministic policy."

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\nUser feedback:\n"
        + (feedback or "")
        + "\nRevise the code accordingly."
    )

#生成 heuristic
def generate_heuristic(api_url: str, api_key: str, model: str,
                       previous_code: Optional[str], feedback: Optional[str]) -> Optional[str]:
    prompt = build_prompt(previous_code, feedback)
    content = call_llm(api_url, api_key, model, prompt)
    if not content:
        print("[LLM] content is empty or None")
        return None
    print("[LLM] content head:", content[:200])
    code = _extract_code(content)
    print("[LLM] code head:", code[:200])
    print("[LLM] code length:", len(code))
    return code

#把生成的代码写成一个 Python 文件
def write_heuristic(path: str, code: str) -> None:
    # 去掉首尾空行
    lines = [ln.rstrip("\n") for ln in code.strip("\n").splitlines()]

    # 计算最小缩进（忽略空行）
    min_indent = None
    for ln in lines:
        if ln.strip() == "":
            continue
        leading = len(ln) - len(ln.lstrip(" "))
        if min_indent is None or leading < min_indent:
            min_indent = leading
    if min_indent is None:
        min_indent = 0

    # 先去掉最小缩进，再统一加4空格
    normalized = []
    for ln in lines:
        if ln.strip() == "":
            normalized.append("")
        else:
            normalized.append(" " * 4 + ln[min_indent:])

    indented = "\n".join(normalized)

    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

def llm_policy(heur: BaseHeuristic, obs: dict):
{indented}

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        box_slot, rot_id, x, y = llm_policy(self, obs)
        return self.encode_action_logits(box_slot, rot_id, x, y)
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)


