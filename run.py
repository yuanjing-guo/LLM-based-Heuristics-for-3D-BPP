import argparse
import numpy as np

from env import BoxPlanningEnvWrapper

from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding
from helpers.task_config import TaskConfig
# from heuristics.llm_based import LLMHeuristic


# ------------------------------------------------------------
# Heuristic registry
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    "largest_volume_lowest_z": LargestVolumeLowestZ,
    "floor_building": FloorBuilding
    # "llm_v1": LLMHeuristic,
}


# ------------------------------------------------------------
# Run one episode
# ------------------------------------------------------------
def run_episode(heuristic, max_steps=200, seed=0):
    env = BoxPlanningEnvWrapper(
        save_video_path=f"video/{heuristic.name}.mp4"
    )

    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        action = heuristic(obs)
        obs, reward, done, trunc, info = env.step(action)

        total_reward += reward
        step += 1

        print(
            f"step={step:02d}, reward={reward:.3f}, "
            f"placed={len(env.env.boxes_on_pallet_id)}, "
            f"term={info['termination_reason']}"
        )

    print(f"[{heuristic.name}] Finished. total_reward={total_reward:.3f}")

    # Graceful cleanup (reduces EGL warnings)
    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()
    del env

    return total_reward


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
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Build env once to query dimensions
    # --------------------------------------------------------
    dummy_env = BoxPlanningEnvWrapper()
    heuristic_cls = HEURISTIC_REGISTRY[args.heuristic]

    heuristic = heuristic_cls(
        N_visible_boxes=dummy_env.N_visible_boxes,
        pallet_size_discrete=dummy_env.pallet_size_discrete,
        max_pallet_height=dummy_env.env.max_pallet_height,
        bin_size=TaskConfig.bin_size,
    )

    print(f"[Run] Heuristic = {heuristic.name}")
    run_episode(
        heuristic,
        max_steps=args.max_steps,
        seed=args.seed,
    )
