# evaluate.py
import argparse
import os
import secrets
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

from env import BoxPlanningEnvWrapper, compute_utilization_volume
from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding
from heuristics.llm_entry import LLMBasedHeuristic
from heuristics.empty_maximal_spaces import EmptyMaximalSpace
from heuristics.extreme_point import ExtremePoint


# ------------------------------------------------------------
# Heuristic registry
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    "largest_volume_lowest_z": LargestVolumeLowestZ,
    "floor_building": FloorBuilding,
    "llm_based": LLMBasedHeuristic,
    "empty_maximal_space": EmptyMaximalSpace,
    "extreme_point": ExtremePoint,
}


# ------------------------------------------------------------
# Run one episode (no video by default)
# ------------------------------------------------------------
def run_episode_once(
    heuristic,
    max_steps: int,
    seed: int,
    save_video: bool = False,
    physics_mode: str = "rigid",          # default rigid; only "soft" enables soft-contact mode
    expose_physics_obs: bool = True,
) -> Tuple[float, int, str]:
    """
    Returns:
      util_final: float
      n_boxes:    int   (committed stable boxes)
      term:       str   (e.g., "success", "unstable", "height_oob", "invalid_action", "cutoff", "exception")
    """
    video_path = f"video/{heuristic.name}.mp4" if save_video else None

    env = BoxPlanningEnvWrapper(
        save_video_path=video_path,
        physics_mode=physics_mode,
        expose_physics_obs=expose_physics_obs,
    )

    obs, _ = env.reset(seed=seed)
    if hasattr(heuristic, "reset"):
        heuristic.reset()

    done = False
    util_final: Optional[float] = None
    term: str = "cutoff"

    step = 0
    try:
        while (not done) and (step < max_steps):
            action = heuristic(obs)
            obs, reward, done, trunc, info = env.step(action)
            step += 1

            if done:
                # New env convention: util only at terminal (success/unstable/height_oob/etc)
                if "util" in info and info["util"] is not None:
                    util_final = float(info["util"])

                # termination_reason might be int OR string depending on your env version
                tr = info.get("termination_reason", None)
                if isinstance(tr, str):
                    term = tr
                elif isinstance(tr, (int, np.integer)):
                    # back-compat mapping (if you still use ints somewhere)
                    # 3: success, 2: unstable, 4: invalid, else unknown
                    term = {3: "success", 2: "unstable", 4: "invalid_action"}.get(int(tr), f"term_{int(tr)}")
                else:
                    # if env didn't provide it, still mark as done
                    term = "done"

        # cutoff: compute util from current occupancy
        if util_final is None:
            metrics = compute_utilization_volume(env.env.obs["pallet_obs_density"])
            util_final = float(metrics["util"])
            if done and term == "cutoff":
                term = "done_no_util"

    except RuntimeError as e:
        # In case some strict errors still raise (e.g., OutOfBounds boundary) in some branch
        # We still want a usable metric.
        metrics = compute_utilization_volume(env.env.obs["pallet_obs_density"])
        util_final = float(metrics["util"])
        msg = str(e)
        if "OutOfBoundsZ" in msg:
            term = "height_oob_exception"
        elif "OutOfBounds" in msg:
            term = "boundary_oob_exception"
        else:
            term = "runtime_error"
    except Exception:
        metrics = compute_utilization_volume(env.env.obs["pallet_obs_density"])
        util_final = float(metrics["util"])
        term = "exception"

    # committed stable boxes
    n_boxes = int(len(env.env.boxes_on_pallet_id))

    # Graceful cleanup (reduces EGL warnings)
    try:
        if hasattr(env.env, "writer") and env.env.writer is not None:
            env.env.writer.close()
    except Exception:
        pass

    del env
    return float(util_final), n_boxes, str(term)


# ------------------------------------------------------------
# Stats helpers
# ------------------------------------------------------------
def mean_var(xs: List[float]) -> Tuple[float, float]:
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.var(ddof=0))  # population variance


# ------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--heuristics",
        type=str,
        default="all",
        help="Comma-separated heuristic names (e.g., floor_building,extreme_point) or 'all'",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Number of rounds (each round uses a new random seed; all heuristics share that seed)",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_video", action="store_true", help="Save a video for each episode (slow, large)")
    parser.add_argument("--report_dir", type=str, default="report")

    # -------- Runtime env switches --------
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Enable soft-contact physics mode. If not set, physics_mode='rigid'.",
    )
    parser.add_argument(
        "--no_physics_obs",
        action="store_true",
        help="Disable physics-aware observations (fields exist but filled with zeros).",
    )

    args = parser.parse_args()

    physics_mode = "soft" if args.soft else "rigid"
    expose_physics_obs = (not args.no_physics_obs)

    # Parse heuristic list
    if args.heuristics.strip().lower() == "all":
        heuristic_names = list(HEURISTIC_REGISTRY.keys())
    else:
        heuristic_names = [s.strip() for s in args.heuristics.split(",") if s.strip()]
        unknown = [h for h in heuristic_names if h not in HEURISTIC_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown heuristic(s): {unknown}. Available: {list(HEURISTIC_REGISTRY.keys())}")

    # Init heuristic instances once
    heuristics = [HEURISTIC_REGISTRY[name]() for name in heuristic_names]

    # Prepare report file
    os.makedirs(args.report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(args.report_dir, f"eval__{ts}.txt")

    # Storage: per heuristic -> lists over rounds
    per_h: Dict[str, Dict[str, List]] = {}
    for h in heuristics:
        per_h[h.name] = {"util": [], "n_boxes": [], "term": []}

    seeds: List[int] = []

    # Evaluate
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Batch Evaluation Report ===\n")
        f.write(f"Time: {ts}\n")
        f.write(f"Heuristics: {', '.join([h.name for h in heuristics])}\n")
        f.write(f"Rounds: {args.rounds}\n")
        f.write(f"Max steps per episode: {args.max_steps}\n")
        f.write(f"Save video: {bool(args.save_video)}\n")
        f.write(f"physics_mode: {physics_mode}\n")
        f.write(f"physics_obs: {'on' if expose_physics_obs else 'off'}\n")
        f.write("\n")

        for r in range(args.rounds):
            seed = secrets.randbelow(2**31 - 1)
            seeds.append(int(seed))

            f.write(f"--- Round {r+1}/{args.rounds} | seed={seed} ---\n")
            f.flush()

            for h in heuristics:
                util, n_boxes, term = run_episode_once(
                    h,
                    max_steps=args.max_steps,
                    seed=seed,
                    save_video=args.save_video,
                    physics_mode=physics_mode,
                    expose_physics_obs=expose_physics_obs,
                )
                per_h[h.name]["util"].append(util)
                per_h[h.name]["n_boxes"].append(n_boxes)
                per_h[h.name]["term"].append(term)

                f.write(f"{h.name:24s}  util={util:.6f}  n_boxes={n_boxes:3d}  term={term}\n")
                f.flush()

            f.write("\n")

        # Summary
        f.write("\n=== Summary (mean / variance over rounds) ===\n")
        for h in heuristics:
            name = h.name
            util_mean, util_var = mean_var(per_h[name]["util"])
            box_mean, box_var = mean_var([float(x) for x in per_h[name]["n_boxes"]])

            f.write(f"\n[{name}]\n")
            f.write(f"  util:    mean={util_mean:.6f}  var={util_var:.6f}\n")
            f.write(f"  n_boxes: mean={box_mean:.6f}  var={box_var:.6f}\n")

            terms = per_h[name]["term"]
            counts: Dict[str, int] = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
            f.write(f"  term_counts: {dict(sorted(counts.items(), key=lambda kv: kv[0]))}\n")

        f.write("\n=== Seeds (one per round) ===\n")
        f.write(", ".join(str(s) for s in seeds) + "\n")

    print(f"[Done] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
