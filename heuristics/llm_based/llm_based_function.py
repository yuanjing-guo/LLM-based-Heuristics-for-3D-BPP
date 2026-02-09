# heuristics/llm_based/llm_based_function.py
import os
import json
import re
from typing import Optional, List, Tuple

from heuristics.llm_based.llm_interface import call_llm


# ============================================================
# Code extraction (sanitize LLM output)
# ============================================================

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


# ============================================================
# Run context injection
# ============================================================

def _load_run_context(path: str = "runs/latest/run_context.json") -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _format_run_context_for_prompt(ctx: Optional[dict]) -> str:
    """
    Keep it short. Only tell LLM what can change per run.
    """
    if not ctx:
        return (
            "[RUN CONTEXT]\n"
            "- obs['buffer'] is 1D flat (N*heur.n_properties)\n"
            "- Action is raw int32[5]: PLACE(0,slot,rot,x,y) / REMOVE(1,idx,0,0,0)\n"
            "- Use ONLY heur.X heur.Y heur.H heur.n_properties\n"
            "- Physics keys may exist but can be all-zeros.\n"
        )

    physics_mode = str(ctx.get("physics_mode", "rigid")).strip().lower()
    expose_physics_obs = bool(ctx.get("expose_physics_obs", True))

    lines = []
    lines.append("[RUN CONTEXT]")
    lines.append(f"physics_mode={physics_mode}")
    lines.append(f"expose_physics_obs={expose_physics_obs}")
    lines.append("- obs['buffer'] is 1D flat (N*heur.n_properties)")
    lines.append("- Use ONLY heur.X heur.Y heur.H heur.n_properties")
    lines.append("- Physics keys may exist but can be all-zeros.")
    return "\n".join(lines) + "\n"


# ============================================================
# Static validation (stop wrong APIs / hallucinated schema)
# ============================================================

def _find_line_col(code: str, idx: int) -> Tuple[int, int]:
    before = code[:idx]
    line = before.count("\n") + 1
    col = idx - (before.rfind("\n") + 1)
    return line, col


def _validate_generated_code(code: str) -> List[str]:
    """
    Returns a list of error strings. Empty => OK.
    Hard-fail common LLM mistakes that crash runtime or cause REMOVE loops.
    """
    errors: List[str] = []

    # 1) Must define llm_policy
    if "def llm_policy" not in code:
        errors.append("Missing function definition: def llm_policy(heur, obs):")
        return errors

    # 2) Ban imports
    for m in re.finditer(r"^\s*(import|from)\s+", code, flags=re.MULTILINE):
        line, col = _find_line_col(code, m.start())
        errors.append(f"BANNED: import/from inside generated code at L{line}:{col}")

    # 3) Ban helper defs other than llm_policy
    for m in re.finditer(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code, flags=re.MULTILINE):
        fn = m.group(1)
        if fn != "llm_policy":
            line, col = _find_line_col(code, m.start())
            errors.append(f"BANNED: extra function def '{fn}' at L{line}:{col}. Only llm_policy is allowed.")

    # 4) Ban 2D indexing of buffer
    banned_buffer_patterns = [
        r"\bbuffer\s*\[\s*[^]\n]*,\s*[^]\n]*\]",
        r"obs\s*\[\s*['\"]buffer['\"]\s*\]\s*\[\s*[^]\n]*,\s*[^]\n]*\]",
    ]
    for pat in banned_buffer_patterns:
        for m in re.finditer(pat, code):
            line, col = _find_line_col(code, m.start())
            snippet = code[m.start():min(len(code), m.start() + 80)].replace("\n", " ")
            errors.append(
                f"BANNED: 2D-like buffer indexing at L{line}:{col}: '{snippet}...' "
                f"obs['buffer'] is 1D. Use heur.get_slot_props(obs, slot)."
            )

    # 5) Require use of heur.get_slot_props
    if "heur.get_slot_props" not in code:
        errors.append("Missing required API usage: heur.get_slot_props(obs, slot). Do NOT index obs['buffer'] manually.")

    # 6) CRASH-CRITICAL: is_feasible signature misuse
    for m in re.finditer(r"heur\.feasibility\.is_feasible\s*\(", code):
        i0 = m.end()
        depth = 1
        j = i0
        while j < len(code) and depth > 0:
            if code[j] == "(":
                depth += 1
            elif code[j] == ")":
                depth -= 1
            j += 1
        call_txt = code[m.start():j]

        if re.search(r"is_feasible\s*\(\s*obs\b", call_txt):
            line, col = _find_line_col(code, m.start())
            errors.append(
                f"WRONG is_feasible() call at L{line}:{col}: first arg is 'obs'. "
                f"Must be pallet array: is_feasible(pallet, x, y, dx, dy, dz, z)."
            )
            continue

        inner = call_txt[call_txt.find("(") + 1: call_txt.rfind(")")]
        depth2 = 0
        commas = 0
        for ch in inner:
            if ch == "(":
                depth2 += 1
            elif ch == ")":
                depth2 -= 1
            elif ch == "," and depth2 == 0:
                commas += 1
        arg_count = commas + 1 if inner.strip() else 0
        if arg_count != 7:
            line, col = _find_line_col(code, m.start())
            errors.append(
                f"WRONG is_feasible() arg count at L{line}:{col}: got {arg_count}, expected 7. "
                f"Must be is_feasible(pallet, x, y, dx, dy, dz, z)."
            )

    # 7) Ban invented obs schema keys (common hallucination)
    invented_keys = [
        "physics_obs",
        "obs['physics_obs']",
        'obs["physics_obs"]',
        "obs['support']",
        'obs["support"]',
    ]
    for k in invented_keys:
        if k in code:
            errors.append(f"BANNED: invented/unknown obs key usage: {k}")

    # 8) REMOVE contract: ban invented pallet_index mapping (x*Y+y etc.)
    banned_remove_math = [
        r"x\s*\*\s*heur\.Y",
        r"x\s*\*\s*self\.Y",
        r"x\s*\*\s*Y",
        r"pallet_index\s*=\s*.*\*.*\+",
    ]
    for pat in banned_remove_math:
        for m in re.finditer(pat, code):
            line, col = _find_line_col(code, m.start())
            snippet = code[m.start():min(len(code), m.start() + 80)].replace("\n", " ")
            errors.append(f"BANNED: invented pallet_index mapping at L{line}:{col}: '{snippet}...'")

    # 9) Ban REMOVE(0) as fallback/no-op (causes infinite remove loops)
    for m in re.finditer(r"return\s*\(\s*1\s*,\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)", code):
        line, col = _find_line_col(code, m.start())
        errors.append(f"BANNED: REMOVE(0) used as fallback/no-op at L{line}:{col}")

    return errors


# ============================================================
# Prompt builder (SIMPLIFIED & SECTIONED)
# ============================================================

def build_prompt(
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[List[str]],
    run_context: Optional[dict] = None,
) -> str:
    ctx_block = _format_run_context_for_prompt(run_context)

    # ============================================================
    # 1) INTERFACE (NON-NEGOTIABLE)
    # ============================================================
    interface_block = (
        "=== 1) INTERFACE (NON-NEGOTIABLE) ===\n"
        "You are given (heur, obs). You must return ONE action of 5 ints.\n"
        "The returned 5-int action is passed DIRECTLY to env.step (NO logits encoding).\n"
        "\n"
        "ACTIONS:\n"
        "- PLACE:  (0, slot, rot_id, x, y)\n"
        "- REMOVE: (1, pallet_index, 0, 0, 0)\n"
        "\n"
        "OBS (guaranteed keys):\n"
        "- pallet = obs['pallet_obs_density']  # float32 array shape (X,Y,H)\n"
        "- obs['buffer'] is 1D flat length N*heur.n_properties  (NOT 2D)\n"
        "- obs['front_ids']  # length N\n"
        "- obs['pallet_count']  # int\n"
        "- obs['pallet_footprints']  # list length pallet_count; fp=(x,y,z,dx,dy,dz)\n"
        "- obs['pallet_ids']  # list length pallet_count\n"
        "- obs['removable_mask']  # int8 array length pallet_count\n"
        "\n"
        "PHYSICS (optional keys; may be all zeros):\n"
        "- obs.get('buffer_physics')  # 1D length N*2, softness = buffer_physics[2*slot + 0]\n"
        "- obs.get('pallet_obs_softness')  # float32 array shape (X,Y,H)\n"
        "\n"
        "RULES:\n"
        "- NEVER invent obs keys (e.g., do NOT use obs['physics_obs']).\n"
        "- NEVER index obs['buffer'] as 2D (no buffer[slot,:]). Use heur.get_slot_props(obs, slot).\n"
        "- Use ONLY heur.X, heur.Y, heur.H, heur.n_properties.\n"
    )

    # ============================================================
    # 2) ALLOWED CODE PATTERNS (MUST FOLLOW EXACTLY)
    # ============================================================
    allowed_block = (
        "=== 2) ALLOWED CODE PATTERNS (MUST FOLLOW EXACTLY) ===\n"
        "Output ONLY ONE function:\n"
        "  def llm_policy(heur, obs):\n"
        "No imports. No classes. No extra helper defs using 'def'.\n"
        "You may use inline variables, inline loops, and small lambdas.\n"
        "\n"
        "READ SLOT PROPS (ONLY THIS WAY):\n"
        "  props = heur.get_slot_props(obs, slot)\n"
        "  size_bins = heur.props_to_size_bins(props)\n"
        "  dx, dy, dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]\n"
        "\n"
        "LEGAL PLACEMENT CHECK ORDER (MUST):\n"
        "  1) compute dx,dy,dz\n"
        "  2) loop x,y over range(heur.X-dx+1), range(heur.Y-dy+1)\n"
        "  3) compute z by EXACT height rule:\n"
        "     place_area = pallet[x:x+dx, y:y+dy, :]\n"
        "     non_zero_mask = np.any(place_area > 0, axis=(0,1))\n"
        "     z = int(np.max(np.nonzero(non_zero_mask)[0]) + 1) if np.any(non_zero_mask) else 0\n"
        "     if z + dz > heur.H: continue\n"
        "  4) strict non-overlap:\n"
        "     blk = pallet[x:x+dx, y:y+dy, z:z+dz]\n"
        "     if blk.size==0: continue\n"
        "     if np.any(blk > 1e-6): continue\n"
        "  5) within + feasible:\n"
        "     if not heur.feasibility.is_within_pallet(x,y,dx,dy): continue\n"
        "     if not heur.feasibility.is_feasible(pallet, x,y,dx,dy,dz,z): continue\n"
        "\n"
        "IMPORTANT:\n"
        "- Do NOT call is_feasible(..., 0) as a pre-filter unless you computed z and confirmed z==0.\n"
        "- The last argument of is_feasible MUST be the variable z, not a literal.\n"
        "\n"
        "PHYSICS INFORMATIVE:\n"
        "  physics_informative = ('buffer_physics' in obs) and ('pallet_obs_softness' in obs) and \\\n"
        "                        (np.any(obs['buffer_physics']!=0) or np.any(obs['pallet_obs_softness']!=0))\n"
        "\n"
        "HARD/SOFT (only if physics_informative):\n"
        "  softness = float(obs['buffer_physics'][2*slot + 0])  (if index in bounds)\n"
        "  SOFT if softness > 0.5 else HARD\n"
        "If physics_informative is False: treat all boxes as HARD.\n"
    )

    # ============================================================
    # 3) STRATEGY (YOUR GOAL IMPLEMENTED AS STATE MACHINE)
    # ============================================================
    strategy_block = (
        "=== 3) STRATEGY (IMPLEMENT EXACTLY, AS A PRIORITIZED STATE MACHINE) ===\n"
        "Your ONLY goal:\n"
        "  Final bottom layer (z==0) should become ALL HARD.\n"
        "Upper layers (z>0): NO hard/soft constraints; place anything legal.\n"
        "\n"
        "You MUST implement THREE PHASES (A then B then C). Execute the first applicable phase each step.\n"
        "\n"
        "PHASE A: BASE-HARDIFY (replacement)\n"
        "Condition:\n"
        "- physics_informative is True\n"
        "- buffer contains at least one HARD non-empty slot\n"
        "- pallet bottom contains at least one SOFT footprint (a footprint with fz==0 whose region in pallet_obs_softness\n"
        "  inside that footprint has mean > 0.5)\n"
        "Action:\n"
        "A1) If a bottom-soft footprint index i is directly removable (removable_mask[i]>0), REMOVE it.\n"
        "    After removing it, store memory:\n"
        "      llm_policy._pending_base = (fx, fy, fdx, fdy, ttl=10)\n"
        "A2) Else (bottom soft not removable): REMOVE a removable blocker ABOVE it (fz>0) that overlaps its XY.\n"
        "    Prefer: higher top height (fz+fdz), then larger overlap. This uncovers the bottom soft.\n"
        "\n"
        "PHASE B: FILL PENDING HOLE WITH HARD (immediate replacement)\n"
        "Condition:\n"
        "- llm_policy._pending_base exists\n"
        "Action:\n"
        "- Try to PLACE a HARD box at z==0 that overlaps the pending region.\n"
        "- ONLY accept placements where computed z==0.\n"
        "- ONLY consider HARD boxes.\n"
        "- When ttl expires, clear pending.\n"
        "\n"
        "PHASE C: NORMAL PLACEMENT\n"
        "If not doing A or B, place normally:\n"
        "- On z==0: prefer HARD first; allow SOFT only if NO HARD-bottom legal placement exists now.\n"
        "- On z>0: place anything legal.\n"
        "\n"
        "REMOVE CONTRACT (critical):\n"
        "- REMOVE pallet_index MUST be an index into obs['pallet_footprints']/obs['pallet_ids'].\n"
        "- Only remove if obs['removable_mask'][i] > 0.\n"
        "- NEVER compute pallet_index from x,y (no x*Y+y).\n"
        "\n"
        "ANTI-OSCILLATION (required but simple):\n"
        "- Keep llm_policy._recent: last 8 actions; reject candidates that appear in last 4.\n"
        "- Keep llm_policy._rm_cd cooldown dict: after REMOVE(i) set cooldown[i]=3; decrement each step.\n"
        "- Only do general REMOVE (outside Phase A) when you failed to find any PLACE for 2 consecutive steps.\n"
        "\n"
        "LAST RESORT:\n"
        "- If absolutely no legal PLACE and no legal REMOVE: return PLACE(0,0,0,0,0).\n"
        "- NEVER return REMOVE(1,0,0,0,0) as fallback.\n"
    )

    # ============================================================
    # 4) OUTPUT FORMAT (STRICT)
    # ============================================================
    output_block = (
        "=== 4) OUTPUT FORMAT (STRICT) ===\n"
        "Output ONLY Python code that defines exactly ONE function:\n"
        "  def llm_policy(heur, obs):\n"
        "Return exactly 5 integers: (op,a1,a2,a3,a4)\n"
        "\n"
        "BANNED OUTPUT:\n"
        "- any 'def' other than llm_policy\n"
        "- any import/from\n"
        "- any class\n"
        "- buffer[slot, :] or obs['buffer'][slot, :]\n"
        "- invented obs keys like obs['physics_obs']\n"
        "- is_feasible(obs, ...) or is_feasible with wrong arg count\n"
        "- pallet_index = x*Y+y or any grid-math mapping\n"
        "- return (1,0,0,0,0) as fallback\n"
    )

    base = (
        ctx_block
        + "\n"
        + interface_block
        + "\n"
        + allowed_block
        + "\n"
        + strategy_block
        + "\n"
        + output_block
    )

    if feedback_history:
        items = [str(x).strip() for x in feedback_history if str(x).strip()]
        if items:
            base = (
                "HARD CONSTRAINTS FROM HISTORY (must satisfy):\n- "
                + "\n- ".join(items)
                + "\n\n"
                + base
            )

    if not previous_code:
        return base + "\nStart from scratch.\n"

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\n\nHuman feedback for this revision:\n"
        + (feedback or "")
        + "\n\nRevise the code accordingly while keeping ALL rules.\n"
    )

    base = (
        ctx_block
        + "\n"
        + interface_block
        + "\n"
        + allowed_block
        + "\n"
        + strategy_block
        + "\n"
        + output_block
    )

    if feedback_history:
        items = [str(x).strip() for x in feedback_history if str(x).strip()]
        if items:
            base = (
                "HARD CONSTRAINTS FROM HISTORY (must satisfy):\n- "
                + "\n- ".join(items)
                + "\n\n"
                + base
            )

    if not previous_code:
        return base + "\nStart from scratch.\n"

    return (
        base
        + "\nPrevious version:\n"
        + previous_code
        + "\n\nHuman feedback for this revision:\n"
        + (feedback or "")
        + "\n\nRevise the code accordingly while keeping ALL rules.\n"
    )


# ============================================================
# Generate
# ============================================================

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

    code = _extract_code(content)

    # errs = _validate_generated_code(code)
    # if errs:
    #     print("[LLM][VALIDATION FAILED] refusing to write heuristic:")
    #     for e in errs[:60]:
    #         print(" -", e)
    #     return None

    return code


# ============================================================
# Write generated heuristic (temp module)
# ============================================================

def write_heuristic(path: str, code: str) -> None:
    """
    Mixed env expects RAW int32[5] actions.
    """
    template = f"""import numpy as np
from heuristics.base import BaseHeuristic

{code}

class GeneratedHeuristic(BaseHeuristic):
    name = "llm_generated"

    def __call__(self, obs):
        out = llm_policy(self, obs)
        if isinstance(out, (list, tuple, np.ndarray)):
            out = list(out)
        else:
            out = [out]

        while len(out) < 5:
            out.append(0)
        out = out[:5]

        return np.array([int(v) for v in out], dtype=np.int32)
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(template)
