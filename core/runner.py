# core/runner.py
from typing import Optional, Dict, Any, Tuple

from envs.mixed_env import MixedBoxPlanningEnvWrapper


def run_episode(
    heuristic,
    max_steps: int = 200,
    seed: int = 0,
    save_video: bool = True,
    soft: bool = False,
    expose_physics_obs: bool = True,
    video_dir: str = "video",
    run_context_meta: Optional[Dict[str, Any]] = None,   # NEW
) -> Tuple[float, int, str]:
    physics_mode = "soft" if soft else "rigid"

    video_path: Optional[str] = None
    if save_video:
        video_path = f"{video_dir}/{heuristic.name}__{physics_mode}.mp4"

    env = MixedBoxPlanningEnvWrapper(
        save_video_path=video_path,
        physics_mode=("soft" if soft else None),
        expose_physics_obs=expose_physics_obs,
    )

    # -------- NEW: write runs/latest/run_context.json --------
    try:
        from core.context import collect_run_context, write_latest_run_context

        meta = run_context_meta or {}
        meta.setdefault("llm_policy_interface", "5-int action (op,a1,a2,a3,a4)")
        meta.setdefault("supports_remove_action", True)
        ctx = collect_run_context(
            env,
            seed=meta.get("seed", seed),
            max_steps=meta.get("max_steps", max_steps),
            soft=meta.get("soft", soft),
            expose_physics_obs=meta.get("expose_physics_obs", expose_physics_obs),
        )
        ctx["llm_policy_interface"] = meta["llm_policy_interface"]
        ctx["supports_remove_action"] = bool(meta["supports_remove_action"])
        outp = write_latest_run_context(ctx)
        print(f"[Context] wrote: {outp}")
    except Exception as e:
        print(f"[Context] write failed: {e}")

    obs, _ = env.reset(seed=seed)

    if hasattr(heuristic, "reset"):
        heuristic.reset()

    step = 0
    done = False
    last_util = 0.0
    last_info: Dict[str, Any] = {}
    last_status = "running"
    boxes_on_pallet = 0

    while (not done) and (step < max_steps):
        action = heuristic(obs)
        obs, reward, done, trunc, info = env.step(action)
        step += 1

        info = info or {}
        last_info = info
        last_util = float(info.get("util", info.get("util_current", last_util)))
        boxes_on_pallet = int(info.get("pallet_count", boxes_on_pallet))

        term = str(info.get("termination_reason", ""))
        if trunc:
            last_status = term or "truncated"
        elif done:
            last_status = term or "done"
        else:
            last_status = term or "running"

        # close video
    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()

    # explicit robosuite / mujoco cleanup
    try:
        env.env.close()
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass

    del env

    if (not done) and step >= max_steps:
        last_status = last_status if last_status != "running" else "max_steps"

    if not last_status:
        last_status = "unknown"

    return float(last_util), int(boxes_on_pallet), str(last_status)


def format_run_banner(
    heuristic_name: str,
    seed: int,
    max_steps: int,
    soft: bool,
    no_physics_obs: bool,
    no_video: bool,
) -> str:
    mode_str = "soft" if soft else "rigid"
    return (
        "[Run] Heuristic={} | seed={} | max_steps={} | physics_mode={} | physics_obs={} | video={}".format(
            heuristic_name,
            seed,
            max_steps,
            mode_str,
            "off" if no_physics_obs else "on",
            "off" if no_video else "on",
        )
    )
