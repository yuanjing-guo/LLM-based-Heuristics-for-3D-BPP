# run.py
import argparse


# from heuristics.llm_entry import LLMBasedHeuristic
from heuristics.dummy_mixed import DummyMixed
from core.registry import build_registry
from core.runner import run_episode, format_run_banner
from heuristics.dummy_mixed_remove import DummyMixedRemove
from heuristics.test_height_debug import TestHeightDebug
from heuristics.llm_archives import worked1


# ------------------------------------------------------------
# Handcrafted Heuristic registry (static)
# ------------------------------------------------------------
HEURISTIC_REGISTRY = {
    #llm_based": LLMBasedHeuristic,
    "dummy_mixed": DummyMixed,
    "dummy_mixed_remove": DummyMixedRemove,
    "test_height_debug": TestHeightDebug,
    "worked1": worked1,
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

    parser.add_argument("--max_steps", type=int, default=100)
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

        util, n_boxes, term = run_episode(
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
        print(f"[Result] util={util:.4f} status={term} boxes={n_boxes}")
