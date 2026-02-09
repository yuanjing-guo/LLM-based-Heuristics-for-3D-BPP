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
    if not ctx:
        return (
            "[RUN CONTEXT]\n"
            "No run_context.json found. Assume minimum schema:\n"
            "- obs['pallet_obs_density']: float32 (X,Y,H)\n"
            "- obs['buffer']: float32 1D (N*heur.n_properties)  <-- IMPORTANT: 1D FLAT\n"
            "- Action is raw int32[5]:\n"
            "  PLACE:  (0, slot, rot_id, x, y)\n"
            "  REMOVE: (1, pallet_index, 0, 0, 0)\n"
            "- heur.X heur.Y heur.H heur.n_properties are valid\n"
            "IMPORTANT: Returned action is passed DIRECTLY to env.step (NO logits).\n"
            "NOTE: Physics fields may exist but be all-zeros.\n"
        )

    physics_mode = str(ctx.get("physics_mode", "rigid")).strip().lower()
    expose_physics_obs = bool(ctx.get("expose_physics_obs", True))
    X = ctx.get("X")
    Y = ctx.get("Y")
    H = ctx.get("H")
    n_props = ctx.get("n_properties")

    lines = []
    lines.append("[RUN CONTEXT]")
    lines.append(f"physics_mode={physics_mode}")
    lines.append(f"expose_physics_obs={expose_physics_obs}")
    if X is not None and Y is not None and H is not None:
        lines.append(f"grid_size_XYH=({X},{Y},{H})  # informational only")
    if n_props is not None:
        lines.append(f"n_properties={n_props}")

    lines.append("IMPORTANT: obs['buffer'] is a 1D flat array, NOT 2D.")
    lines.append("IMPORTANT: You MUST use heur.get_slot_props(obs, slot) to read a slot.")
    lines.append("IMPORTANT: Use ONLY heur.X, heur.Y, heur.H for pallet dimensions.")
    lines.append("IMPORTANT: Do NOT invent new heur attributes.")
    lines.append("NOTE: Physics fields may be present but uninformative (all zeros).")
    return "\n".join(lines) + "\n"


# ============================================================
# Static validation (stop wrong APIs / buffer indexing)
# ============================================================

def _find_line_col(code: str, idx: int) -> Tuple[int, int]:
    before = code[:idx]
    line = before.count("\n") + 1
    col = idx - (before.rfind("\n") + 1)
    return line, col


def _validate_generated_code(code: str) -> List[str]:
    """
    Returns a list of error strings. Empty => OK.
    Hard-fail common LLM mistakes that crash runtime.
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

    # 3) Ban helper defs other than llm_policy (LLM loves inventing get_box_props)
    for m in re.finditer(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code, flags=re.MULTILINE):
        fn = m.group(1)
        if fn != "llm_policy":
            line, col = _find_line_col(code, m.start())
            errors.append(f"BANNED: extra function def '{fn}' at L{line}:{col}. Only llm_policy is allowed.")

    # 4) Ban 2D indexing of buffer: buffer[slot, :] or obs['buffer'][slot, :]
    #    (any comma inside [] right after buffer/obs['buffer'])
    banned_patterns = [
        r"\bbuffer\s*\[\s*[^]\n]*,\s*[^]\n]*\]",
        r"obs\s*\[\s*['\"]buffer['\"]\s*\]\s*\[\s*[^]\n]*,\s*[^]\n]*\]",
    ]
    for pat in banned_patterns:
        for m in re.finditer(pat, code):
            line, col = _find_line_col(code, m.start())
            snippet = code[m.start():min(len(code), m.start() + 80)].replace("\n", " ")
            errors.append(
                f"BANNED: 2D-like buffer indexing at L{line}:{col}: '{snippet}...' "
                f"obs['buffer'] is 1D. Use heur.get_slot_props(obs, slot)."
            )

    # 5) Require use of heur.get_slot_props somewhere (otherwise likely cheating/buggy)
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

    return errors


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(
    previous_code: Optional[str],
    feedback: Optional[str],
    feedback_history: Optional[List[str]],
    run_context: Optional[dict] = None,
) -> str:
    ctx_block = _format_run_context_for_prompt(run_context)

    base = (
        ctx_block
        + "\n"
        + "=== PROBLEM CONTEXT (P-CEoH) ===\n"
        + "You write a heuristic policy for a mixed palletization + unstacking env.\n"
        + "Each step: llm_policy(heur, obs) is called ONCE, you return ONE action, executed immediately.\n"
        + "\n"
        + "=== OUTPUT CONTRACT ===\n"
        + "Output ONLY Python code that defines EXACTLY ONE function:\n"
        + "    def llm_policy(heur, obs):\n"
        + "Return EXACTLY 5 integers: (op,a1,a2,a3,a4)\n"
        + "\n"
        + "=== ACTIONS ===\n"
        + "PLACE:  op=0 -> (0, slot, rot_id, x, y)\n"
        + "REMOVE: op=1 -> (1, pallet_index, 0, 0, 0)\n"
        + "\n"
        + "=== OBS + BUFFER FACTS ===\n"
        + "- pallet = obs['pallet_obs_density']  # (X,Y,H)\n"
        + "- obs['buffer'] is a 1D FLAT array of length N*heur.n_properties (NOT 2D)\n"
        + "- You MUST read slot properties ONLY by: heur.get_slot_props(obs, slot)\n"
        + "- NEVER reshape buffer, NEVER do buffer[slot,:], NEVER write helper get_box_props.\n"
        + "\n"
        + "=== ABSOLUTE RULES ===\n"
        + "- NO imports. NO classes. NO extra top-level helpers.\n"
        + "- Only llm_policy is allowed.\n"
        + "- Use ONLY heur.X, heur.Y, heur.H, heur.n_properties.\n"
        + "- Always cast to int.\n"
        + "- Never use numpy arrays directly in if; use np.any/np.all/size.\n"
        + "\n"
        + "=== BANNED PATTERNS (NEVER OUTPUT) ===\n"
        + "- def get_box_props(...), def get_props(...), any extra def other than llm_policy\n"
        + "- buffer[slot, :], obs['buffer'][slot, :], any comma-indexing of buffer\n"
        + "- heur.feasibility.is_feasible(obs, ...)\n"
        + "- heur.feasibility.is_feasible(obs, slot, rot_id, x, y)\n"
        + "- any is_feasible call with arg count != 7\n"
        + "\n"
        + "=== DX/DY/DZ CRASH RULE (MUST) ===\n"
        + "Allowed pattern ONLY:\n"
        + "  props = heur.get_slot_props(obs, slot)\n"
        + "  size_bins = heur.props_to_size_bins(props)\n"
        + "  dx,dy,dz = [int(v) for v in heur.rotate_size_bins(size_bins, rot_id)]\n"
        + "\n"
        + "=== HEIGHT COMPUTE (MUST MATCH ENV) ===\n"
        + "place_area = pallet[x:x+dx, y:y+dy, :]\n"
        + "non_zero_mask = np.any(place_area > 0, axis=(0,1))\n"
        + "z = int(np.max(np.nonzero(non_zero_mask)) + 1) if np.any(non_zero_mask) else 0\n"
        + "if z+dz > heur.H: continue\n"
        + "\n"
        + "=== STRICT NON-OVERLAP (MUST) ===\n"
        + "blk = pallet[x:x+dx, y:y+dy, z:z+dz]\n"
        + "if blk.size==0: continue\n"
        + "if np.any(blk > 1e-6): continue\n"
        + "\n"
        + "=== FEASIBILITY API (ONLY THIS) ===\n"
        + "  heur.feasibility.is_within_pallet(x, y, dx, dy)\n"
        + "  heur.feasibility.is_feasible(pallet, x, y, dx, dy, dz, z)\n"
        + "\n"
        + "=== REQUIRED STRATEGY (SUMMARY) ===\n"
        + "- physics_informative only if physics keys exist AND not all zeros.\n"
        + "- Bottom layer z==0: HARD first, SOFT only if no HARD bottom feasible.\n"
        + "- Prohibit placing hard/heavy onto soft support; must do TARGETED REMOVE of that soft support box.\n"
        + "- Anti-oscillation: track recent actions; avoid repeating removes; limit remove bursts.\n"
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

    errs = _validate_generated_code(code)
    if errs:
        print("[LLM][VALIDATION FAILED] refusing to write heuristic:")
        for e in errs[:40]:
            print(" -", e)
        return None

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
