# debug/cli.py
import importlib
from typing import Dict, Type

from core.runner import run_episode, format_run_banner
from core.registry import build_registry
from debug.storage import write_archive


def _print_help():
    print(
        "\n[Debug Commands]\n"
        "  help                 Show this help\n"
        "  eval                 Run one episode with current heuristic\n"
        "  gen <feedback>       Regenerate LLM heuristic with feedback\n"
        "  save <name>          Save current LLM policy as heuristics/llm_archives/<name>.py\n"
        "  start <name|0>       Switch starting heuristic (0 means llm_based)\n"
        "  list                 List all available heuristics\n"
        "  quit                 Exit debug\n"
    )


def _resolve_start(name: str) -> str:
    name = (name or "").strip()
    if name == "0":
        return "llm_based"
    return name


def run_debug_loop(args, registry: Dict[str, Type], handcrafted_registry: Dict[str, Type]):
    """
    Command-based debug loop.
    - registry: runtime registry (handcrafted + archives)
    - handcrafted_registry: original handcrafted dict (used to rebuild registry after save)
    """
    start = _resolve_start(args.debug_start)
    if not start:
        raise SystemExit("ERROR: --debug_start is required when --debug is set.")
    if start not in registry:
        raise SystemExit(f"ERROR: debug_start '{start}' not found. Available: {sorted(registry.keys())}")

    current_name = start
    heuristic = registry[current_name]()

    _print_help()

    while True:
        cmdline = input(f"[debug:{current_name}]> ").strip()
        if not cmdline:
            continue

        parts = cmdline.split(" ", 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("help", "h", "?"):
            _print_help()
            continue

        if cmd in ("quit", "q", "exit"):
            break

        if cmd == "list":
            print("\n".join(sorted(registry.keys())))
            continue

        if cmd == "start":
            target = _resolve_start(rest)
            if not target:
                print("Usage: start <name|0>")
                continue
            if target not in registry:
                print(f"Unknown heuristic '{target}'. Use 'list' to see options.")
                continue
            current_name = target
            heuristic = registry[current_name]()
            print(f"[Debug] switched to: {current_name}")
            continue

        if cmd == "eval":
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
            )
            print(f"[Result] util={util:.4f} status={term} boxes={n_boxes}")
            continue

        if cmd == "gen":
            if not hasattr(heuristic, "regenerate"):
                print("[Debug] current heuristic has no regenerate(). Use start 0 to switch to llm_based.")
                continue
            if not rest:
                print("Usage: gen <feedback>")
                continue
            heuristic.regenerate(rest)
            print("[Debug] regenerated.")
            continue

        if cmd == "save":
            if not rest:
                print("Usage: save <name>")
                continue

            # Need LLM code to save
            code = None
            if hasattr(heuristic, "get_current_code"):
                code = heuristic.get_current_code()
            elif hasattr(heuristic, "current_code"):
                code = getattr(heuristic, "current_code")

            if not code:
                print("[Debug] No current_code found. Only LLM-based heuristic can be saved.")
                continue

            meta = {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "physics_mode": "soft" if args.soft else "rigid",
                "physics_obs": "off" if args.no_physics_obs else "on",
            }
            try:
                out_path = write_archive(rest, code, meta=meta, overwrite=False)
            except Exception as e:
                print(f"[Save] failed: {e}")
                continue

            print(f"[Save] archived to: {out_path}")

            # Refresh registry so newly saved heuristic is immediately available
            importlib.invalidate_caches()
            registry = build_registry(handcrafted_registry, include_archives=True)
            print("[Debug] registry refreshed. You can now 'start <name>' or 'list'.")
            continue

        print(f"Unknown command: {cmd}. Type 'help'.")
