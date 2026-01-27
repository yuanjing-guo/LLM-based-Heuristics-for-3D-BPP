import json
from typing import Optional, Dict, Any

from heuristics.llm_based.llm_interface import call_llm


def _extract_code(text: str) -> str:
    if "```" not in text:
        body = text.rstrip()
    else:
        parts = text.split("```")
        body = parts[1].rstrip() if len(parts) >= 3 else text.rstrip()
        if body.lower().startswith("python"):
            body = body.split("\n", 1)[1] if "\n" in body else ""

    cleaned = []
    for line in body.splitlines():
        s = line.lstrip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        cleaned.append(line)
    body = "\n".join(cleaned).rstrip()

    start = body.find("def llm_policy")
    return body[start:].rstrip() if start >= 0 else body.rstrip()


def _load_run_context(path: str = "runs_demo/latest/run_context.json") -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _format_ctx(ctx: Optional[Dict[str, Any]], caps: Dict[str, str]) -> str:
    if not ctx:
        return (
            "[RUN CONTEXT]\n"
            "No run_context.json found.\n"
            "- obs['pallet_obs_density'] shape (X,Y,H)\n"
            "- obs['buffer'] 1D length N*n_properties\n"
            "- return (box_slot, rot_id, x, y)\n"
            f"[CAPS] buffer={caps.get('buffer')} unstack={caps.get('unstack')}\n"
        )

    lines = []
    lines.append("[RUN CONTEXT]")
    lines.append(f"physics_mode={str(ctx.get('physics_mode','rigid'))}")
    lines.append(f"expose_physics_obs={bool(ctx.get('expose_physics_obs', True))}")
    lines.append(f"pallet_discrete=(X,Y,H)=({ctx.get('X')},{ctx.get('Y')},{ctx.get('H')})")
    lines.append(f"n_properties={ctx.get('n_properties')}")
    lines.append(f"[CAPS] buffer={caps.get('buffer')} unstack={caps.get('unstack')}")
    return "\n".join(lines) + "\n"


def build_prompt(
    *,
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[list],
    caps: Dict[str, str],
    run_context_path: str = "runs_demo/latest/run_context.json",
) -> str:
    ctx = _load_run_context(run_context_path)
    ctx_block = _format_ctx(ctx, caps)

    buffer_mode = (caps.get("buffer") or "full").strip().lower()
    unstack_mode = (caps.get("unstack") or "off").strip().lower()

    buffer_rule = (
        "- BUFFER MODE first: you MUST always return box_slot=0 (slot_i fixed to 0).\n"
        if buffer_mode == "first"
        else "- BUFFER MODE full: you may choose any slot_i in [0..n_slots-1].\n"
    )
    unstack_rule = (
        "- UNSTACK is ON (stage-1): you may add comments, but you STILL MUST return a placement tuple.\n"
        if unstack_mode == "on"
        else "- UNSTACK is OFF: ignore unstacking.\n"
    )

    base = (
        ctx_block
        + "\n"
        + "Write ONLY Python code for: def llm_policy(heur, obs): ...\n"
        + "Return (box_slot, rot_id, x, y) as integers.\n"
        + "\n"
        + "ABSOLUTE RULES:\n"
        + "- Do NOT import anything (no import/from).\n"
        + "- Do NOT define classes.\n"
        + "- You may use np.* (np is already available).\n"
        + "- NEVER use numpy arrays in boolean context.\n"
        + "\n"
        + "You MUST use ONLY these helpers:\n"
        + "  heur.slot_is_empty(obs, slot_i)\n"
        + "  heur.get_slot_props(obs, slot_i)\n"
        + "  heur.props_to_size_bins(props)\n"
        + "  heur.rotate_size_bins(size_bins, rot_id)\n"
        + "\n"
        + "Feasibility MUST be handled ONLY via:\n"
        + "  heur.feasibility.is_within_pallet(x, y, dx, dy)\n"
        + "  heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z)\n"
        + "\n"
        + "CAPABILITY RULES:\n"
        + buffer_rule
        + unstack_rule
        + "\n"
        + "REQUIRED NAIVE STRATEGY:\n"
        + "- Scan slot_i increasing, rot_id=0..5, x,y increasing.\n"
        + "- Return the FIRST feasible placement.\n"
        + "- If none found, return (0,0,0,0).\n"
        + "\n"
        + "CRITICAL FEASIBILITY (MUST):\n"
        + "1) pallet = obs['pallet_obs_density']\n"
        + "2) For candidate (x,y,dx,dy):\n"
        + "   place_area = pallet[x:x+dx, y:y+dy, :]\n"
        + "   non_zero = np.any(place_area > 0, axis=(0,1))\n"
        + "   z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0\n"
        + "3) If z + dz > heur.H: continue\n"
        + "4) If not heur.feasibility.is_within_pallet(x, y, dx, dy): continue\n"
        + "5) If not heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z): continue\n"
        + "\n"
        + "Loop bounds to avoid out-of-range:\n"
        + "  for x in range(heur.X - dx + 1):\n"
        + "    for y in range(heur.Y - dy + 1):\n"
        + "\n"
        + "Slot count MUST be:\n"
        + "  n_slots = int(obs['buffer'].size // heur.n_properties)\n"
    )

    history_block = ""
    if feedback_history:
        history_block = "HARD CONSTRAINTS FROM HISTORY:\n- " + "\n- ".join(feedback_history) + "\n"
        base = history_block + "\n" + base

    if not previous_code:
        return base + "\nStart from scratch with the naive approach.\n"

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\n\nHuman feedback:\n"
        + (feedback or "")
        + "\n\nRevise while keeping all rules.\n"
    )


def generate_heuristic(
    api_url: str,
    api_key: str,
    model: str,
    *,
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[list],
    caps: Dict[str, str],
) -> Optional[str]:
    prompt = build_prompt(
        previous_code=previous_code,
        feedback=feedback,
        feedback_history=feedback_history,
        caps=caps,
    )
    content = call_llm(api_url, api_key, model, prompt)
    if not content:
        print("[LLM] empty response")
        return None
    print("[LLM] content head:", content[:200])
    code = _extract_code(content)
    print("[LLM] code head:", code[:200])
    print("[LLM] code length:", len(code))
    return code


def write_heuristic(path: str, code: str) -> None:
    import os

    # ✅ 运行期再加一层“buffer first clamp”，但只 clamp box_slot，不丢 x/y/rot
    clamp_block = r"""
def _clamp_action_if_needed(heur, obs, action):
    try:
        box_slot, rot_id, x, y = action
        # buffer-first: force box_slot=0
        if getattr(heur, "_caps", {}).get("buffer", "full") == "first":
            box_slot = 0
        return (int(box_slot), int(rot_id), int(x), int(y))
    except Exception:
        return (0, 0, 0, 0)
""".strip()

    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

{clamp_block}

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_demo_generated"

    def __init__(self):
        super().__init__()
        # 注意：这里为了让 clamp 能拿到 caps，我们在 wrapper 里会把 _caps 注入到 impl 上（见下面说明）
        # 如果没注入，也不会崩，只是不会 clamp。

    def __call__(self, obs):
        action = llm_policy(self, obs)
        action = _clamp_action_if_needed(self, obs, action)
        box_slot, rot_id, x, y = action
        return self.encode_action_logits(box_slot, rot_id, x, y)
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
