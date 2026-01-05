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

    total_reward = 0.0
    step = 0
    done = False

    while (not done) and (step < max_steps):
        action = heuristic(obs)
        obs, reward, done, trunc, info = env.step(action)

        total_reward += float(reward)
        step += 1

        print(
            f"step={step:02d}, reward={reward:.3f}, "
            f"placed={len(env.env.boxes_on_pallet_id)}, "
            f"term={info.get('termination_reason', -1)}"
        )

    print(f"[{heuristic.name}] Finished. total_reward={total_reward:.3f}")

    # Graceful cleanup (reduces EGL warnings)
    if hasattr(env.env, "writer") and env.env.writer is not None:
        env.env.writer.close()
    del env

    return float(total_reward)


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

    # IMPORTANT:
    # This assumes your heuristics now read TaskConfig via BaseHeuristic,
    # so they can be constructed with no dimension args:
    #
    # class FloorBuilding(BaseHeuristic):
    #     name = "floor_building"
    #     def __init__(self):
    #         super().__init__()
    #         ...
    #
    heuristic = heuristic_cls()

    print(f"[Run] Heuristic = {heuristic.name}")
    run_episode(heuristic, max_steps=args.max_steps, seed=args.seed)
