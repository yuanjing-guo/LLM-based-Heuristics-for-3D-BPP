# demo/cli.py
import importlib

from demo.llm_entry_demo import LLMBasedHeuristicDemo
from demo.storage import write_archive


def _print_help():
    print(
        "\n[Demo Commands]\n"
        "  help                         Show this help\n"
        "  eval                         Run one episode with current policy\n"
        "  gen <feedback>               Regenerate LLM policy with feedback\n"
        "  save <name>                  Save current LLM policy to heuristics/llm_archives_demo/<name>.py\n"
        "  caps                         Show current capability switches\n"
        "  buffer <first|full>          Buffer capability (first => can ONLY pick box_slot=0)\n"
        "  unstack <on|off>             Unstack capability (stage-1: prompt only)\n"
        "  reset                        Reset feedback history + restart from naive\n"
        "  quit                         Exit demo\n"
    )


def run_demo_loop(args):
    """
    Strongly-isolated demo loop.
    - Only one heuristic: LLM-based (demo).
    - Physics is controlled by CLI args (--soft, --no_physics_obs).
    - Buffer/unstack are controlled inside demo loop (caps).
    """
    heuristic = LLMBasedHeuristicDemo(
        soft=args.soft,
        expose_physics_obs=(not args.no_physics_obs),
    )

    _print_help()

    while True:
        cmdline = input("[demo]> ").strip()
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

        if cmd == "caps":
            print("[Caps]", heuristic.get_capabilities())
            continue

        if cmd == "buffer":
            mode = rest.lower().strip()
            if mode not in ("first", "full"):
                print("Usage: buffer <first|full>")
                continue
            heuristic.set_capability("buffer", mode)
            print(f"[Caps] buffer={mode}")
            continue

        if cmd == "unstack":
            mode = rest.lower().strip()
            if mode not in ("on", "off"):
                print("Usage: unstack <on|off>")
                continue
            heuristic.set_capability("unstack", mode)
            print(f"[Caps] unstack={mode}")
            continue

        if cmd == "reset":
            heuristic.reset_to_naive()
            print("[Demo] reset to naive + cleared feedback history.")
            continue

        if cmd == "gen":
            if not rest:
                print("Usage: gen <feedback>")
                continue
            heuristic.regenerate(rest)
            print("[Demo] regenerated.")
            continue

        if cmd == "eval":
            util = heuristic.eval_once(
                seed=args.seed,
                max_steps=args.max_steps,
                save_video=(not args.no_video),
            )
            print(f"[Result] util={util:.4f}")
            continue

        if cmd == "save":
            if not rest:
                print("Usage: save <name>")
                continue
            code = heuristic.get_current_code()
            if not code:
                print("[Demo] No current code to save.")
                continue

            meta = {
                "seed": args.seed,
                "max_steps": args.max_steps,
                "physics_mode": "soft" if args.soft else "rigid",
                "physics_obs": "off" if args.no_physics_obs else "on",
                "caps": heuristic.get_capabilities(),
            }
            out_path = write_archive(rest, code, meta=meta, overwrite=False)
            print(f"[Save] archived to: {out_path}")
            importlib.invalidate_caches()
            continue

        print(f"Unknown command: {cmd}. Type 'help'.")
