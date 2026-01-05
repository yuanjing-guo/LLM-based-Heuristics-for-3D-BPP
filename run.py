# run.py
import argparse

from env import BoxPlanningEnvWrapper
from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding


# ------------------------------------------------------------
# Heuristic registry
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    "largest_volume_lowest_z": LargestVolumeLowestZ,
    "floor_building": FloorBuilding,
}


# ------------------------------------------------------------
# Run one episode
# ------------------------------------------------------------
def run_episode(heuristic, max_steps: int = 200, seed: int = 0) -> float:
    env = BoxPlanningEnvWrapper(save_video_path=f"video/{heuristic.name}.mp4")

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

        util = float(info.get("util_current", 0.0))
        last_util = util

        print(f"step={step:02d}")
        #     f"V_boxes={info.get('V_boxes_bins3', 0):.0f}, "
        #     f"V_env_hm={info.get('V_env_hm_bins3', 0):.0f}, "
        #     f"hmax={info.get('hmax_bins', 0):.0f}, "
        #     f"footprint={info.get('footprint_bins2', 0):.0f}, "
        #     f"placed={len(env.env.boxes_on_pallet_id)}, "
        #     f"term={info.get('termination_reason', -1)}"
        # )

    # print(f"[{heuristic.name}] Finished. final_util={last_util:.4f}")

    # Graceful cleanup (reduces EGL warnings)
    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()
    del env

    return float(last_util)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heuristic",
        type=str,
        required=True,
        choices=HEURISTIC_REGISTRY.keys(),
        help="Which heuristic to run",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    heuristic_cls = HEURISTIC_REGISTRY[args.heuristic]
    heuristic = heuristic_cls()

    print(f"[Run] Heuristic = {heuristic.name}")
    run_episode(heuristic, max_steps=args.max_steps, seed=args.seed)
