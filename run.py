# run.py
import argparse

from env import BoxPlanningEnvWrapper
from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding
from heuristics.empty_maximal_spaces import EmptyMaximalSpace
from heuristics.extreme_point import ExtremePoint
from heuristics.corner_point import CornerPoint
from heuristics.first_fit import FirstFit
from heuristics.best_fit import BestFit
from heuristics.llm_entry import LLMBasedHeuristic
from heuristics.floor_building_buffer import FloorBuildingBuffer
from heuristics.floor_building_buffer_rule_physics import FloorBuildingBufferRulePhysics





# ------------------------------------------------------------
# Heuristic registry
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    "largest_volume_lowest_z": LargestVolumeLowestZ,
    "floor_building": FloorBuilding,
    "empty_maximal_space": EmptyMaximalSpace,
    "extreme_point": ExtremePoint,
    "corner_point": CornerPoint,
    "first_fit": FirstFit,
    "best_fit": BestFit,
    "llm_based": LLMBasedHeuristic,
    "floor_building_buffer":FloorBuildingBuffer,
    "floor_building_buffer_rule_physics": FloorBuildingBufferRulePhysics
}


# ------------------------------------------------------------
# Run one episode
# ------------------------------------------------------------
def run_episode(
    heuristic,
    max_steps: int = 200,
    seed: int = 0,
    save_video: bool = True,         # DEFAULT: save video
    soft: bool = False,      # DEFAULT: rigid; only True => soft
    expose_physics_obs: bool = True, # False -> physics obs fields filled with zeros
) -> float:
    physics_mode = "soft" if soft else "rigid"

    video_path = (
        f"video/{heuristic.name}__{physics_mode}.mp4"
        if save_video
        else None
    )

    env = BoxPlanningEnvWrapper(
        save_video_path=video_path,
        physics_mode=("soft" if soft else None),  # None => rigid default (per env policy)
        expose_physics_obs=expose_physics_obs,
    )

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
        # terminal util is stored in info["util"] when done
        last_util = float(info.get("util", info.get("util_current", 0.0)))

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
        choices=list(HEURISTIC_REGISTRY.keys()),
        help="Which heuristic to run",
    )
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    # -------- Runtime env switches --------
    # NEW: only a boolean switch; default is rigid
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Enable soft-contact physics. If not set, rigid physics is used by default.",
    )
    parser.add_argument(
        "--no_physics_obs",
        action="store_true",
        help="Disable physics-aware observations (fields exist but filled with zeros).",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable video saving (default: video is saved).",
    )

    args = parser.parse_args()

    heuristic_cls = HEURISTIC_REGISTRY[args.heuristic]
    heuristic = heuristic_cls()

    # --------------------------------------------------------
    # Interactive loop (only relevant for LLM-based heuristic)
    # --------------------------------------------------------
    while True:
        mode_str = "soft" if args.soft else "rigid"
        print(
            "[Run] Heuristic={} | seed={} | max_steps={} | physics_mode={} | physics_obs={} | video={}".format(
                heuristic.name,
                args.seed,
                args.max_steps,
                mode_str,
                "off" if args.no_physics_obs else "on",
                "off" if args.no_video else "on",
            )
        )

        run_episode(
            heuristic,
            max_steps=args.max_steps,
            seed=args.seed,
            save_video=(not args.no_video),
            soft=args.soft,
            expose_physics_obs=(not args.no_physics_obs),
        )

        # Only LLM-based heuristic is expected to have regenerate()
        if not hasattr(heuristic, "regenerate"):
            break

        feedback = input("输入反馈（直接回车退出）：").strip()
        if not feedback:
            break

        heuristic.regenerate(feedback)
