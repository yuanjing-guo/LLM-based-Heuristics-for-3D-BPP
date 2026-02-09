# run.py
import argparse

from heuristics.largest_volume_lowest_z import LargestVolumeLowestZ
from heuristics.floor_building import FloorBuilding
from heuristics.empty_maximal_spaces import EmptyMaximalSpace
from heuristics.extreme_point import ExtremePoint
from heuristics.corner_point import CornerPoint
from heuristics.first_fit import FirstFit
from heuristics.best_fit import BestFit

from heuristics.llm_archives.before_training_soft import BeforeTrainingSoft  
from heuristics.llm_archives.before_training import BeforeTraining  
from heuristics.llm_archives.expert_guidance_2 import ExpertGuidance2  
from heuristics.llm_archives.expert_guidance_3 import ExpertGuidance3
from heuristics.llm_archives.expert_guidance import ExpertGuidance

from heuristics.llm_archives.expert_guidance_soft import ExpertGuidanceSoft
from heuristics.llm_archives.my_best_heuristic_no_physics import MyBestHeuristic
from heuristics.llm_archives.worker_guidance_simple_soft import WorkerGuidanceSimpleSoft
from heuristics.llm_archives.worker_guidance_specific_soft import WorkerGuidanceSpecificSoft
from heuristics.llm_archives.worker_guidance import WorkerGuidance 



from heuristics.llm_entry import LLMBasedHeuristic
from heuristics.floor_building_buffer import FloorBuildingBuffer
from heuristics.floor_building_buffer_rule_physics import FloorBuildingBufferRulePhysics
from core.registry import build_registry
from core.runner import run_episode, format_run_banner


# ------------------------------------------------------------
# Handcrafted Heuristic registry (static)
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
    "floor_building_buffer": FloorBuildingBuffer,
    "floor_building_buffer_rule_physics": FloorBuildingBufferRulePhysics,
    "before_training_soft": BeforeTrainingSoft,
    "before_training": BeforeTraining,
    "expert_guidance2": ExpertGuidance2,
    "expert_guidance3": ExpertGuidance3,
    "expert_guidance": ExpertGuidance,
    "expert_guidance_soft": ExpertGuidanceSoft,
    "my_best_heuristic_no_physics": MyBestHeuristic,
    "worker_guidance": WorkerGuidance,
    "worker_guidance_simple_soft": WorkerGuidanceSimpleSoft,
    "worker_guidance_specific_soft": WorkerGuidanceSpecificSoft,

    
 


    

}


def parse_args(all_heuristics):
    parser = argparse.ArgumentParser()

    # Normal run mode (non-debug): require --heuristic
    parser.add_argument(
        "--heuristic",
        type=str,
        choices=all_heuristics,
        help="Which heuristic to run (handcrafted or archived). Required when --debug is not set.",
    )

    # Debug mode (explicit)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable interactive debug loop. Only in this mode we ask for human feedback.",
    )
    parser.add_argument(
        "--debug_start",
        type=str,
        default=None,
        help=(
            "Start point for debug mode. Use '0' to start from llm_based, "
            "or specify any heuristic name from registry (including llm_archives). "
            "Required when --debug is set."
        ),
    )

    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    # -------- Runtime env switches --------
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

    return parser.parse_args()


if __name__ == "__main__":
    # Build runtime registry (handcrafted + llm_archives)
    registry = build_registry(HEURISTIC_REGISTRY, include_archives=True)
    all_heuristics = sorted(registry.keys())

    args = parse_args(all_heuristics)

    if args.debug:
        # Debug loop is isolated in debug/cli.py (keeps run.py clean)
        from debug.cli import run_debug_loop
        run_debug_loop(args, registry, HEURISTIC_REGISTRY)

    else:
        if not args.heuristic:
            raise SystemExit("ERROR: --heuristic is required when not using --debug.")

        heuristic = registry[args.heuristic]()

        print(
            format_run_banner(
                heuristic_name=heuristic.name,
                seed=args.seed,
                max_steps=args.max_steps,
                soft=args.soft,
                no_physics_obs=args.no_physics_obs,
                no_video=args.no_video,
            )
        )

        util = run_episode(
            heuristic,
            max_steps=args.max_steps,
            seed=args.seed,
            save_video=(not args.no_video),
            soft=args.soft,
            expose_physics_obs=(not args.no_physics_obs),
            video_dir="video",
            run_context_meta={
            "seed": args.seed,
            "max_steps": args.max_steps,
            "soft": args.soft,
            "expose_physics_obs": (not args.no_physics_obs),
            },
        )
        print(f"[Result] util={util:.4f}")
