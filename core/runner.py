# core/runner.py
from typing import Optional, Dict, Any

from envs.env import BoxPlanningEnvWrapper


def run_episode(
    heuristic,
    max_steps: int = 200,
    seed: int = 0,
    save_video: bool = True,
    soft: bool = False,
    expose_physics_obs: bool = True,
    video_dir: str = "video",
    run_context_meta: Optional[Dict[str, Any]] = None,   # NEW
) -> float:
    physics_mode = "soft" if soft else "rigid"

    video_path: Optional[str] = None
    if save_video:
        video_path = f"{video_dir}/{heuristic.name}__{physics_mode}.mp4"

    env = BoxPlanningEnvWrapper(
        save_video_path=video_path,
        physics_mode=("soft" if soft else None),
        expose_physics_obs=expose_physics_obs,
    )

    # -------- NEW: write runs/latest/run_context.json --------
    try:
        from core.context import collect_run_context, write_latest_run_context

        meta = run_context_meta or {}
        ctx = collect_run_context(
            env,
            seed=meta.get("seed", seed),
            max_steps=meta.get("max_steps", max_steps),
            soft=meta.get("soft", soft),
            expose_physics_obs=meta.get("expose_physics_obs", expose_physics_obs),
        )
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

    while (not done) and (step < max_steps):
        action = heuristic(obs)
        obs, reward, done, trunc, info = env.step(action)
        step += 1
        last_util = float(info.get("util", info.get("util_current", 0.0)))

    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()
    del env

    return float(last_util)


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
