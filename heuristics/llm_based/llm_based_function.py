# heuristics/llm_based/llm_based_function.py
import os
import json
from typing import Optional, List

from heuristics.llm_based.llm_interface import call_llm


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
    Facts only: schema + mode.
    """
    if not ctx:
        return (
            "[RUN CONTEXT]\n"
            "No run_context.json found. Assume minimum schema:\n"
            "- obs['pallet_obs_density'] is a 3D numpy array (X,Y,H)\n"
            "- obs['buffer'] is a 1D numpy array of length (N*n_properties)\n"
            "- action interface uses 5 integers: (op,a1,a2,a3,a4)\n"
            "  op=0 PLACE: (0, slot, rot_id, x, y)\n"
            "  op=1 REMOVE: (1, pallet_index, 0, 0, 0)\n"
            "- rot_id in [0..5]\n"
            "- heur.X heur.Y heur.H heur.n_properties are valid\n"
            "IMPORTANT: The returned 5-int action is passed DIRECTLY to env.step (NO logits encoding).\n"
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

    # SUPER IMPORTANT: for Mixed env, action is raw int32[5]
    lines.append("IMPORTANT: The returned 5-int action is passed DIRECTLY to env.step (NO logits encoding).")

    # MODE RULE remains
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
# Prompt builder
# ---------------------------

def build_prompt(
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[List[str]],
    run_context: Optional[dict] = None,
) -> str:
    """
    Mixed env aligned prompt:
      - MUST return 5 ints as RAW action (env.step consumes directly).
      - MUST compute z exactly env-style and enforce z+dz<=H.
      - MUST enforce strict voxel non-overlap.
      - MUST cast dx,dy,dz to int before any range/slicing.
      - REMOVE a1 MUST be pallet_index (footprint index), NEVER a z/voxel coordinate.
      - In soft+physics_obs mode (and keys exist), code MUST include hard/soft reasoning capability.
    """
    ctx_block = _format_run_context_for_prompt(run_context)

    base = (
        ctx_block
        + "\n"
        + "You must output ONLY Python code that defines ONLY ONE function:\n"
        + "    def llm_policy(heur, obs):\n"
        + "Return EXACTLY 5 integers: (op, a1, a2, a3, a4).\n"
        + "\n"
        + "=== ENV INTERFACE (THIS IS GROUND TRUTH, DO NOT INVENT) ===\n"
        + "Action is passed DIRECTLY to env.step as int32[5]. NO logits encoding.\n"
        + "- PLACE:  op=0 -> [0, slot, rot_id, x, y]\n"
        + "- REMOVE: op=1 -> [1, pallet_index, 0, 0, 0]\n"
        + "\n"
        + "Observation keys ALWAYS PRESENT in Mixed env:\n"
        + "- obs['pallet_obs_density']: float32 array shape (X,Y,H)\n"
        + "- obs['pallet_obs_softness']: float32 array shape (X,Y,H)  (filled only when expose_physics_obs=True)\n"
        + "- obs['buffer']: float32 array shape (N_visible * heur.n_properties)\n"
        + "- obs['buffer_physics']: float32 array shape (N_visible * 2) where stride=2:\n"
        + "    softness = buffer_physics[2*slot + 0]   (float, in [0,1], >0.5 means SOFT)\n"
        + "    mu       = buffer_physics[2*slot + 1]\n"
        + "- obs['front_ids']: int32 array shape (N_visible). front_ids[slot] is the true box_id in env.\n"
        + "- obs['pallet_count']: int\n"
        + "- obs['pallet_footprints']: Python list length pallet_count.\n"
        + "    each footprint fp = (x,y,z,dx,dy,dz) in DISCRETE bins.\n"
        + "- obs['pallet_ids']: Python list length pallet_count. pallet_ids[idx] is box_id aligned with footprints[idx].\n"
        + "- obs['removable_mask']: int8 array shape (pallet_count,), aligned with pallet_ids/footprints.\n"
        + "    removable_mask[idx] > 0 means idx can be removed now.\n"
        + "\n"
        + "ABSOLUTE RULES (must follow):\n"
        + "- Do NOT import anything inside llm_policy (no 'import', no 'from').\n"
        + "- Do NOT define classes.\n"
        + "- Do NOT define helper functions outside llm_policy.\n"
        + "- You may use np.* WITHOUT importing numpy (np is available).\n"
        + "- NEVER use numpy arrays in boolean context (no `if arr:`). Use np.any/np.all/len/size.\n"
        + "- Always cast returned values to int.\n"
        + "\n"
        + "DX/DY/DZ TYPE RULE (MUST ALWAYS FOLLOW):\n"
        + "- dx,dy,dz MUST be Python int before using range() or slicing.\n"
        + "- Always do: dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]\n"
        + "\n"
        + "HEIGHT COMPUTATION (MUST MATCH env.get_target_position_strict EXACTLY):\n"
        + "- pallet = obs['pallet_obs_density']\n"
        + "- place_area = pallet[x:x+dx, y:y+dy, :]\n"
        + "- non_zero_mask = np.any(place_area > 0, axis=(0,1))\n"
        + "- z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0\n"
        + "- If z + dz > heur.H: reject candidate (continue). NEVER return an action that can height_oob.\n"
        + "\n"
        + "STRICT NON-OVERLAP (MUST ALWAYS ENFORCE):\n"
        + "- eps = 1e-6\n"
        + "- blk = pallet[x:x+dx, y:y+dy, z:z+dz]\n"
        + "- If blk.size==0: continue\n"
        + "- If np.any(blk > eps): continue\n"
        + "\n"
        + "Feasibility API (EXACT signature):\n"
        + "- heur.feasibility.is_within_pallet(x, y, dx, dy)  # 4 scalars\n"
        + "- heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z)  # 7 args\n"
        + "\n"
        + "SLOT ITERATION (EXACT):\n"
        + "- n_slots = int(obs['buffer'].size // heur.n_properties)\n"
        + "- for slot_i in range(n_slots):\n"
        + "\n"
        + "=== REMOVE CONTRACT (FIXED, DO NOT GUESS) ===\n"
        + "- REMOVE uses pallet_index which is an INDEX into obs['pallet_ids'] / obs['pallet_footprints'].\n"
        + "- You MUST NEVER use z/top_z/voxel coordinate as pallet_index.\n"
        + "- You MUST NEVER try to infer pallet_index from pallet_obs_density columns.\n"
        + "- You MUST ONLY choose indices where removable_mask[idx] > 0.\n"
        + "- If pallet_count==0 or removable_mask empty or no removable idx: DO NOT REMOVE.\n"
        + "\n"
        + "=== PHYSICS HARD/SOFT (FIXED FIELD SEMANTICS) ===\n"
        + "You MUST implement the capability to reason about hard/soft, using EXACT fields:\n"
        + "- Slot softness:\n"
        + "    softness = float(obs['buffer_physics'][2*slot_i + 0]) if in bounds else 0.0\n"
        + "    box_is_soft = (softness > 0.5)\n"
        + "    box_is_hard = not box_is_soft\n"
        + "- Support softness at placement (only meaningful when z>0):\n"
        + "    support = obs['pallet_obs_softness'][x:x+dx, y:y+dy, z-1]\n"
        + "    support_is_soft = (np.max(support) > 0.5)   # use max, NOT mean\n"
        + "\n"
        + "=== REQUIRED BEHAVIOR FOR THIS REVISION (you MUST implement) ===\n"
        + "Goal: consider REMOVE actions to avoid putting HARD boxes on SOFT support.\n"
        + "Algorithm (deterministic, no randomness):\n"
        + "1) PASS-1 (safe place):\n"
        + "   scan slot->rot->x->y and return the FIRST feasible PLACE that satisfies:\n"
        + "   - height safe\n"
        + "   - non-overlap\n"
        + "   - feasible()\n"
        + "   - if box_is_hard and z>0 and support_is_soft: REJECT this candidate in PASS-1\n"
        + "2) PASS-2 (remove-then-place attempt):\n"
        + "   If PASS-1 finds no action AND there exists removable boxes (removable_mask has >0):\n"
        + "   choose ONE removable pallet_index and return REMOVE.\n"
        + "   Selection rule MUST be deterministic and simple:\n"
        + "   - Prefer the LAST removable index (highest index) to mimic top-removal.\n"
        + "3) PASS-3 (relax rule):\n"
        + "   If PASS-1 fails AND no removable exists, you MAY place hard-on-soft (ignore rule)\n"
        + "   using the same first-feasible scan. (This matches: 'if not possible, ignore')\n"
        + "\n"
        + "=== ANTI-REPEAT RULE (MUST IMPLEMENT) ===\n"
        + "To prevent repeated actions, llm_policy MUST remember recent actions using function attributes.\n"
        + "- Maintain llm_policy._recent = list of last 8 actions (tuples of 5 ints).\n"
        + "- Before returning an action, if it equals the most recent action, skip and search next candidate.\n"
        + "- For REMOVE, also skip if same remove index was returned last step.\n"
        + "- If no non-repeating action exists, return (0,0,0,0,0).\n"
        + "\n"
        + "IMPORTANT about (0,0,0,0,0):\n"
        + "- This is interpreted as PLACE(slot=0,rot=0,x=0,y=0) by env.\n"
        + "- Only return it as a LAST RESORT when absolutely no other action exists.\n"
        + "\n"
        + "You MUST use ONLY these helper methods from heur:\n"
        + "  * heur.slot_is_empty(obs, slot_i)\n"
        + "  * heur.get_slot_props(obs, slot_i)\n"
        + "  * heur.props_to_size_bins(props)\n"
        + "  * heur.rotate_size_bins(size_bins, rot_id)\n"
    )


    history_block = ""
    if feedback_history:
        items = [str(x).strip() for x in feedback_history if str(x).strip()]
        if items:
            history_block = "HARD CONSTRAINTS FROM HISTORY (must satisfy, never remove):\n- " + "\n- ".join(items)

    if history_block:
        base = history_block + "\n\nYou MUST keep all prior constraints.\n\n" + base

    if not previous_code:
        return base + "\nStart from scratch with the default strategy.\n"

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\n\nHuman feedback for this revision (natural language). "
          "Only implement advanced behavior if explicitly requested here:\n"
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
    feedback_history: Optional[List[str]],
    run_context_path: str = "runs/latest/run_context.json",
) -> Optional[str]:
    run_ctx = _load_run_context(run_context_path)
    prompt = build_prompt(previous_code, feedback, feedback_history, run_context=run_ctx)

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
    Mixed env expects RAW int32[5] actions.

    llm_policy returns 5 integers:
      - PLACE: (0, slot, rot_id, x, y)  -> returned as np.int32[5]
      - REMOVE: (1, pallet_index, 0, 0, 0) -> returned as np.int32[5]
    """
    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __init__(self):
        super().__init__()

    def __call__(self, obs):
        out = llm_policy(self, obs)

        # robust unpack
        if isinstance(out, (list, tuple, np.ndarray)):
            out = list(out)
        else:
            out = [out]

        # pad / truncate to 5
        while len(out) < 5:
            out.append(0)
        if len(out) > 5:
            out = out[:5]

        op, a1, a2, a3, a4 = [int(x) for x in out]

        # IMPORTANT: Mixed env consumes raw int32[5] directly (no logits)
        return np.array([op, a1, a2, a3, a4], dtype=np.int32)
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
