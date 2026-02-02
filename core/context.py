# core/context.py
import json
import os
from typing import Any, Dict, Optional


def _obs_schema_from_obs(obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}
    for k, v in (obs or {}).items():
        if hasattr(v, "shape"):
            shape = tuple(int(x) for x in v.shape)
        else:
            shape = None
        dtype = str(getattr(v, "dtype", None))
        schema[k] = {"shape": shape, "dtype": dtype}
    return schema


def collect_run_context(
    env_wrapper,
    *,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    soft: Optional[bool] = None,
    expose_physics_obs: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    env_wrapper: BoxPlanningEnvWrapper
    We intentionally collect a compact "truth source" context for prompt + future UI.
    """
    env = getattr(env_wrapper, "env", None)  # BoxPlanning
    if env is None:
        raise ValueError("env_wrapper.env not found")

    # Basic dims
    X = int(env.pallet_size_discrete[0])
    Y = int(env.pallet_size_discrete[1])
    H = int(env.max_pallet_height)
    N = int(env.N_visible_boxes)

    ctx: Dict[str, Any] = {
        "physics_mode": str(getattr(env, "physics_mode", "rigid")),
        "expose_physics_obs": bool(getattr(env, "expose_physics_obs", True)),
        "X": X,
        "Y": Y,
        "H": H,
        "N_visible_boxes": N,
        "bin_size": float(getattr(env, "bin_size", 0.0)),
        "n_properties": int(getattr(env, "n_properties", 0)),
        "n_physics_properties": int(getattr(env, "n_physics_properties", 0)),
        "rotations": 6,
        "orders": {
            "0": [0, 1, 2],
            "1": [0, 2, 1],
            "2": [1, 0, 2],
            "3": [1, 2, 0],
            "4": [2, 0, 1],
            "5": [2, 1, 0],
        },
        "action_schema": {
                # what LLM policy is required to output
                "llm_policy_return": "(op, a1, a2, a3, a4)  # 5 ints",
                "discrete_action_semantics": {
                    "0": "PLACE -> (0, slot, rot_id, x, y)",
                    "1": "REMOVE -> (1, pallet_index, 0, 0, 0)"
                },

                # how the environment / wrapper currently expects actions
                # keep this if you still use encode_action_logits for PLACE
                "place_logits_layout": "slot_logits[0:N] + rot_logits[N:N+6] + x_logits[N+6:N+6+X] + y_logits[N+6+X:N+6+X+Y]",
                "supports_remove_action": True,  # set False if env cannot execute REMOVE yet

                "constraints": [
                    "XY out-of-bounds raises RuntimeError (should be avoided by heuristic).",
                    "Z height overflow terminates episode gracefully (height_oob)."
                ]
        },

        "termination_reasons": {
            "0": "continue",
            "2": "unstable",
            "3": "success (all boxes placed)",
            "4": "invalid_slot (box_slot >= remaining)",
            "5": "height_oob",
        },
    }

    # Add run meta (optional)
    if seed is not None:
        ctx["seed"] = int(seed)
    if max_steps is not None:
        ctx["max_steps"] = int(max_steps)
    if soft is not None:
        ctx["arg_soft"] = bool(soft)
    if expose_physics_obs is not None:
        ctx["arg_expose_physics_obs"] = bool(expose_physics_obs)

    # TaskConfig snapshot (best-effort)
    try:
        from helpers.task_config import TaskConfig  # local import to avoid circulars

        # Keep it compact: only key fields you care about for LLM/UI
        ctx["task_config"] = {
            "buffer_size": int(getattr(TaskConfig, "buffer_size", 0)),
            "bin_size": float(getattr(TaskConfig, "bin_size", 0.0)),
            "pallet": {
                "size": list(getattr(TaskConfig.pallet, "size", [])),
                "max_pallet_height": int(getattr(TaskConfig.pallet, "max_pallet_height", 0)),
            },
            "box": {
                "n_properties": int(getattr(TaskConfig.box, "n_properties", 0)),
                "n_type": int(getattr(TaskConfig.box, "n_type", 0)),
                "type_dict_keys": sorted([int(k) for k in getattr(TaskConfig.box, "type_dict", {}).keys()]),
            },
        }
    except Exception:
        ctx["task_config"] = None

    # Obs schema (best-effort): prefer current obs if present
    obs = getattr(env, "obs", None)
    if isinstance(obs, dict) and obs:
        ctx["obs_schema"] = _obs_schema_from_obs(obs)
    else:
        # Fallback to declared keys
        ctx["obs_schema"] = {
            "pallet_obs_density": {"shape": (X, Y, H), "dtype": "float32"},
            "buffer": {"shape": (N * int(ctx["n_properties"]),), "dtype": "float32"},
            "pallet_obs_softness": {"shape": (X, Y, H), "dtype": "float32"},
            "buffer_physics": {"shape": (N * int(ctx["n_physics_properties"]),), "dtype": "float32"},
        }

    return ctx


def write_latest_run_context(ctx: Dict[str, Any], out_path: str = "runs/latest/run_context.json") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)
    return out_path
