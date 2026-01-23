# evaluate.py
import argparse
import os
import time
import secrets
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from env import BoxPlanningEnvWrapper, compute_utilization_volume
from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding
from heuristics.llm_entry import LLMBasedHeuristic
from heuristics.empty_maximal_spaces import EMSOnline
from heuristics.extreme_point import ExtremePointPhysicsAware


# ------------------------------------------------------------
# Heuristic registry
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    "largest_volume_lowest_z": LargestVolumeLowestZ,
    "floor_building": FloorBuilding,
    "llm_based": LLMBasedHeuristic,
    "empty_maximal_space": EMSOnline,
    "extreme_point": ExtremePointPhysicsAware,
}


# ------------------------------------------------------------
# Run one episode (no video by default)
# ------------------------------------------------------------
def run_episode_once(heuristic, max_steps: int, seed: int, save_video: bool = False) -> Tuple[float, int, int]:
    """
    Returns:
      util_final: float
      n_boxes:    int  (committed stable boxes)
      term_reason:int  (0 ongoing, 2 unstable, 3 success, 4 invalid action; if cutoff -> -1)
    """
    video_path = f"video/{heuristic.name}.mp4" if save_video else None
    env = BoxPlanningEnvWrapper(save_video_path=video_path)

    obs, _ = env.reset(seed=seed)
    if hasattr(heuristic, "reset"):
        heuristic.reset()

    done = False
    term_reason = -1
    util_final = None

    step = 0
    while (not done) and (step < max_steps):
        action = heuristic(obs)
        obs, reward, done, trunc, info = env.step(action)
        step += 1

        # util is only present at terminal in your updated env
        if done:
            term_reason = int(info.get("termination_reason", -1))
            util_final = info.get("util", None)

    # If episode ended by cutoff (max_steps), compute util from current occupancy
    if util_final is None:
        metrics = compute_utilization_volume(env.env.obs["pallet_obs_density"])
        util_final = float(metrics["util"])
        term_reason = -1  # cutoff

    # box count = committed stable boxes
    n_boxes = int(len(env.env.boxes_on_pallet_id))

    # Graceful cleanup (reduces EGL warnings)
    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()
    del env

    return float(util_final), n_boxes, term_reason


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
    parser.add_argument("--rounds", type=int, required=True, help="Number of rounds (each round uses a new random seed)")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--save_video", action="store_true", help="Save a video for each episode (slow, large)")
    parser.add_argument("--report_dir", type=str, default="report")
    args = parser.parse_args()

    # Parse heuristic list
    if args.heuristics.strip().lower() == "all":
        heuristic_names = list(HEURISTIC_REGISTRY.keys())
    else:
        heuristic_names = [s.strip() for s in args.heuristics.split(",") if s.strip()]
        unknown = [h for h in heuristic_names if h not in HEURISTIC_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown heuristic(s): {unknown}. Available: {list(HEURISTIC_REGISTRY.keys())}")

    # Init heuristic instances once (so LLM-based can keep internal state if needed)
    heuristics = []
    for name in heuristic_names:
        heuristics.append(HEURISTIC_REGISTRY[name]())

    # Prepare report file
    os.makedirs(args.report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(args.report_dir, f"eval__{ts}.txt")

    # Storage: per heuristic -> lists over rounds
    per_h: Dict[str, Dict[str, List]] = {}
    for h in heuristics:
        per_h[h.name] = {
            "util": [],
            "n_boxes": [],
            "term_reason": [],
        }

    seeds: List[int] = []

    # Evaluate
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Batch Evaluation Report ===\n")
        f.write(f"Time: {ts}\n")
        f.write(f"Heuristics: {', '.join([h.name for h in heuristics])}\n")
        f.write(f"Rounds: {args.rounds}\n")
        f.write(f"Max steps per episode: {args.max_steps}\n")
        f.write(f"Save video: {bool(args.save_video)}\n")
        f.write("\n")

        for r in range(args.rounds):
            # Generate a fresh random seed for this round
            seed = secrets.randbelow(2**31 - 1)  # safe int seed for gym/np
            seeds.append(int(seed))

            f.write(f"--- Round {r+1}/{args.rounds} | seed={seed} ---\n")
            f.flush()

            # Run all heuristics with the same seed
            for h in heuristics:
                util, n_boxes, term_reason = run_episode_once(
                    h, max_steps=args.max_steps, seed=seed, save_video=args.save_video
                )
                per_h[h.name]["util"].append(util)
                per_h[h.name]["n_boxes"].append(n_boxes)
                per_h[h.name]["term_reason"].append(term_reason)

                f.write(
                    f"{h.name:24s}  util={util:.6f}  n_boxes={n_boxes:3d}  term={term_reason}\n"
                )
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

            # Optional: termination reason counts
            terms = per_h[name]["term_reason"]
            counts = {k: terms.count(k) for k in sorted(set(terms))}
            f.write(f"  term_reason_counts: {counts}\n")

        # Seeds at end
        f.write("\n=== Seeds (one per round) ===\n")
        f.write(", ".join(str(s) for s in seeds) + "\n")

    print(f"[Done] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
