# heuristics/llm_based/llm_based_function.py
import os
import json
from typing import Optional

from heuristics.llm_based.llm_interface import call_llm
from heuristics.llm_based.llm_interface_ollama import call_llm_ollama


# ---------------------------
# Code extraction (sanitize LLM output)
# ---------------------------

def _extract_code(text: str) -> str:
    """
    Extract only:
      def llm_policy(heur, obs):
        ...
    Remove any import/from lines that the LLM might output.
    """
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
    body = "\n".join(cleaned_lines).rstrip()

    start = body.find("def llm_policy")
    if start >= 0:
        return body[start:].rstrip()
    return body.rstrip()


# ---------------------------
# Run context injection
# ---------------------------

def _load_run_context(path: str = "runs/latest/run_context.json") -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _format_run_context_for_prompt(ctx: Optional[dict]) -> str:
    """
    Facts only: schema + mode. This helps the LLM understand the mode without "cheating" via templates.
    """
    if not ctx:
        return (
            "[RUN CONTEXT]\n"
            "No run_context.json found. Assume minimum schema:\n"
            "- obs['pallet_obs_density'] is a 3D numpy array (X,Y,H)\n"
            "- obs['buffer'] is a 1D numpy array of length (N*n_properties)\n"
            "- action is a tuple (box_slot, rot_id, x, y)\n"
            "- rot_id in [0..5]\n"
        )

    physics_mode = str(ctx.get("physics_mode", "rigid")).strip().lower()
    expose_physics_obs = bool(ctx.get("expose_physics_obs", True))
    X = ctx.get("X")
    Y = ctx.get("Y")
    H = ctx.get("H")
    n_props = ctx.get("n_properties")

    obs_schema = ctx.get("obs_schema", {})
    action_schema = ctx.get("action_schema", {})

    lines = []
    lines.append("[RUN CONTEXT]")
    lines.append(f"physics_mode={physics_mode}")
    lines.append(f"expose_physics_obs={expose_physics_obs}")
    if X is not None and Y is not None and H is not None:
        lines.append(f"pallet_discrete=(X,Y,H)=({X},{Y},{H})")
    if n_props is not None:
        lines.append(f"n_properties={n_props}")

    if isinstance(obs_schema, dict) and obs_schema:
        lines.append("obs_schema:")
        for k, v in obs_schema.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: shape={v.get('shape', None)} dtype={v.get('dtype', None)}")
            else:
                lines.append(f"- {k}: {v}")

    if isinstance(action_schema, dict) and action_schema:
        lines.append("action_schema:")
        for k, v in action_schema.items():
            lines.append(f"- {k}: {v}")

    if (physics_mode != "soft") or (not expose_physics_obs):
        lines.append(
            "MODE RULE: You MUST NOT use physics-aware fields such as "
            "obs['buffer_physics'] or obs['pallet_obs_softness']."
        )
    else:
        lines.append(
            "MODE RULE: Soft mode with physics obs enabled. You MAY use physics-aware fields "
            "ONLY if they are present in obs (check keys first)."
        )

    return "\n".join(lines) + "\n"


# ---------------------------
# Prompt (minimal, but MUST use feasibility correctly)
# ---------------------------

def build_prompt(
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[list],
    run_context: Optional[dict] = None,
) -> str:
    """
    Goal:
      - Start very naive (so it can improve via feedback),
      - BUT never "naive to the point of ignoring feasibility".
      - MUST compute z exactly like env.py does, then call feasibility methods to filter invalid actions.
    """
    ctx_block = _format_run_context_for_prompt(run_context)

    base = (
        ctx_block
        + "\n"
        + "Write Python code that defines ONLY a function named llm_policy(heur, obs).\n"
        + "Return a tuple (box_slot, rot_id, x, y) using integers.\n"
        + "\n"
        + "ABSOLUTE RULES (must follow):\n"
        + "- Do NOT import anything inside llm_policy (no 'import', no 'from').\n"
        + "- Do NOT define classes.\n"
        + "- Return ONLY the function code for llm_policy (including the def line). Nothing else.\n"
        + "- You may use np.* WITHOUT importing numpy (np is available in the module).\n"
        + "- NEVER use numpy arrays in boolean context (no `if arr:`). Use `.size`, `len(...)`, `np.any`, `np.all`.\n"
        + "\n"
        + "You MUST use ONLY these helper methods from heur:\n"
        + "  * heur.slot_is_empty(obs, slot_i)\n"
        + "  * heur.get_slot_props(obs, slot_i)\n"
        + "  * heur.props_to_size_bins(props)\n"
        + "  * heur.rotate_size_bins(size_bins, rot_id)\n"
        + "\n"
        + "Feasibility MUST be handled ONLY via these methods:\n"
        + "  * heur.feasibility.is_within_pallet(x, y, dx, dy)\n"
        + "  * heur.feasibility.is_feasible(pallet_obs, x, y, dx, dy, dz, z)\n"
        + "- is_within_pallet takes 4 scalars; never pass tuples.\n"
        + "- is_feasible takes exactly 7 args: (pallet_obs, x, y, dx, dy, dz, z).\n"
        + "\n"
        + "MODE RULES:\n"
        + "- If physics_mode != soft OR expose_physics_obs == False: DO NOT use physics-aware fields.\n"
        + "- If soft mode AND expose_physics_obs == True: you MAY use physics-aware fields only if present.\n"
        + "\n"
        + "NAIVE STRATEGY (required):\n"
        + "- Implement the simplest correct strategy: scan slots from 0..n_slots-1, scan rotations 0..5,\n"
        + "  then scan x,y in deterministic order and return the FIRST feasible action.\n"
        + "- It does NOT need to be smart.\n"
        + "- Correctness and strict feasibility filtering come first.\n"
        + "\n"
        + "Required implementation details (MUST follow exactly):\n"
        + "- Use heur.X, heur.Y, heur.H for pallet bounds.\n"
        + "- Determine number of visible slots ONLY like this:\n"
        + "    n_slots = int(obs['buffer'].size // heur.n_properties)\n"
        + "  then loop: for slot_i in range(n_slots):\n"
        + "- rot_id must be in range(6).\n"
        + "- If heur.slot_is_empty(obs, slot_i) -> continue.\n"

        #==================================================
        # Interface Safety Constraints (DO NOT TUNE)
        # Reason: prevent illegal use of slot_is_empty
        #==================================================
        + "- DO NOT negate slot_is_empty. Never write `not heur.slot_is_empty(...)`.\n"
        + "- The only allowed pattern is:\n"
        + "    if heur.slot_is_empty(obs, slot_i):\n"
        + "        continue\n"
        #==================================================
        # End
        #==================================================
        
        + "- size_bins = heur.props_to_size_bins(props). If size_bins.size==0 or any element<=0 -> continue.\n"
        + "\n"
        + "CRITICAL: z MUST be computed exactly like env.py computes it (do NOT guess z):\n"
        + "- pallet = obs['pallet_obs_density']\n"
        + "- For candidate (x,y,dx,dy):\n"
        + "    place_area = pallet[x:x+dx, y:y+dy, :]\n"
        + "    non_zero = np.any(place_area > 0, axis=(0,1))\n"
        + "    z = int(np.max(np.nonzero(non_zero)) + 1) if np.any(non_zero) else 0\n"
        + "- If z + dz > heur.H: continue (skip). This avoids height_oob.\n"
        + "- THEN call heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z). If False -> continue.\n"
        + "\n"
        + "Loop bounds (avoid out-of-range slices):\n"
        + "- When scanning x,y you MUST use:\n"
        + "    for x in range(heur.X - dx + 1):\n"
        + "        for y in range(heur.Y - dy + 1):\n"
        + "\n"
        + "If no feasible action is found, return (0, 0, 0, 0).\n"
    )

    history_block = ""
    if feedback_history:
        history_block = "HARD CONSTRAINTS FROM HISTORY (must satisfy):\n- " + "\n- ".join(feedback_history)

    if history_block:
        base = history_block + "\nYou MUST keep all prior constraints.\n\n" + base

    if not previous_code:
        return base + "\nStart from scratch with the naive approach.\n"

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\n\nHuman feedback for this revision:\n"
        + (feedback or "")
        + "\n\nRevise the code accordingly while keeping all rules.\n"
    )


# ---------------------------
# Generate
# ---------------------------

def generate_heuristic(
    api_url: str,
    api_key: str,
    model: str,
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[list],
    run_context_path: str = "runs/latest/run_context.json",
) -> Optional[str]:
    run_ctx = _load_run_context(run_context_path)
    prompt = build_prompt(previous_code, feedback, feedback_history, run_context=run_ctx)

    # Backend routing:
    # - If api_url points to Ollama generate endpoint, use Ollama caller
    # - Otherwise use the existing DeepSeek(OpenAI-style) caller
    if "localhost:11434" in api_url or api_url.rstrip("/").endswith("/api/generate"):
        content = call_llm_ollama(api_url, model, prompt)
    else:
        content = call_llm(api_url, api_key, model, prompt)

    if not content:
        print("[LLM] content is empty or None")
        return None
    print("[LLM] content head:", content[:200])

    code = _extract_code(content)

    print("[LLM] code head:", code[:200])
    print("[LLM] code length:", len(code))
    return code


# ---------------------------
# Write generated heuristic (temp module)
# ---------------------------

def write_heuristic(path: str, code: str) -> None:
    """
    Generated module will have numpy available as np.
    LLM policy must NOT import numpy inside llm_policy.
    """
    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

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
