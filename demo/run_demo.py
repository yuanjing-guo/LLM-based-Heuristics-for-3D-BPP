# demo/run_demo.py
import argparse

from demo.cli import run_demo_loop


def parse_args():
    p = argparse.ArgumentParser()

    # core eval params
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=200)

    # physics is ONLY via CLI (parallel capability)
    p.add_argument("--soft", action="store_true", help="Enable soft-contact physics")
    p.add_argument("--no_physics_obs", action="store_true", help="Disable physics-aware obs filling")

    # demo always saves video by default (you said you need video every time)
    p.add_argument("--no_video", action="store_true", help="Disable video saving (default: save)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo_loop(args)
