from typing import Optional, Dict, Any
from pathlib import Path

from envs.env import BoxPlanningEnvWrapper

def _write_latest_run_context(ctx: Dict[str, Any]) -> str:
    import json
    out_dir = Path("runs_demo") / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / "run_context.json"
    outp.write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    return str(outp)

def _collect_run_context(env: BoxPlanningEnvWrapper, *, seed: int, max_steps: int, soft: bool, expose_physics_obs: bool, meta: Dict[str, Any]) -> Dict[str, Any]:
    # 你 env 里通常这些属性都能拿到；拿不到就保守写 None
    X = getattr(env, "X", None)
    Y = getattr(env, "Y", None)
    H = getattr(env, "H", None)

    # 尽量从 env.env 或 wrapper 中拿离散尺寸（按你项目实际）
    inner = getattr(env, "env", None)
    if inner is not None:
        X = X if X is not None else getattr(inner, "X", None)
        Y = Y if Y is not None else getattr(inner, "Y", None)
        H = H if H is not None else getattr(inner, "H", None)

    ctx = {
        "seed": seed,
        "max_steps": max_steps,
        "physics_mode": "soft" if soft else "rigid",
        "expose_physics_obs": bool(expose_physics_obs),
        "X": X, "Y": Y, "H": H,
        # meta 里放 caps 等
        **(meta or {}),
    }
    return ctx

def run_episode_demo(
    heuristic,
    max_steps: int = 200,
    seed: int = 0,
    save_video: bool = True,
    soft: bool = False,
    expose_physics_obs: bool = True,
    video_dir: str = "video_demo",
    run_context_meta: Optional[Dict[str, Any]] = None,
) -> float:
    physics_mode = "soft" if soft else "rigid"

    video_path = None
    if save_video:
        from pathlib import Path
        Path(video_dir).mkdir(exist_ok=True)
        video_path = f"{video_dir}/{heuristic.name}__{physics_mode}__seed{seed}.mp4"

    env = BoxPlanningEnvWrapper(
        save_video_path=video_path,
        physics_mode=("soft" if soft else None),
        expose_physics_obs=expose_physics_obs,
    )

    # 写 demo 的 run_context（给 prompt 读）
    try:
        meta = run_context_meta or {}
        ctx = _collect_run_context(
            env,
            seed=meta.get("seed", seed),
            max_steps=meta.get("max_steps", max_steps),
            soft=meta.get("soft", soft),
            expose_physics_obs=meta.get("expose_physics_obs", expose_physics_obs),
            meta=meta,
        )
        outp = _write_latest_run_context(ctx)
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
