# demo/context.py
import json
import os
from typing import Any, Dict


def _space_to_schema(space) -> Dict[str, Any]:
    # gymnasium spaces typically have shape / dtype
    out = {}
    if hasattr(space, "shape"):
        out["shape"] = tuple(space.shape) if space.shape is not None else None
    if hasattr(space, "dtype"):
        out["dtype"] = str(space.dtype)
    # Dict spaces:
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        out = {k: _space_to_schema(v) for k, v in space.spaces.items()}
    return out


def collect_run_context(env, *, seed: int, max_steps: int, soft: bool, expose_physics_obs: bool) -> Dict[str, Any]:
    # try to get pallet discrete size from wrapper properties
    X = int(env.pallet_size_discrete[0])
    Y = int(env.pallet_size_discrete[1])
    H = int(env.env.max_pallet_height)
    n_props = int(env.env.n_properties)
    N = int(env.N_visible_boxes)
    n_phy = int(getattr(env.env, "n_physics_properties", 0))

    ctx = {
        "seed": int(seed),
        "max_steps": int(max_steps),
        "physics_mode": "soft" if soft else "rigid",
        "expose_physics_obs": bool(expose_physics_obs),
        "X": X,
        "Y": Y,
        "H": H,
        "N_visible_boxes": N,
        "n_properties": n_props,
        "n_physics_properties": n_phy,
        "obs_schema": _space_to_schema(env.observation_space),
        "action_schema": _space_to_schema(env.action_space),
    }
    return ctx


def write_latest_run_context(ctx: Dict[str, Any], root: str = "runs_demo/latest") -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "run_context.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)
    return path
