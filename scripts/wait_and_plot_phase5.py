#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Phase5 full-scale completion, then generate plots.")
    parser.add_argument("--runs_json", type=str, required=True, help="Path to run mapping JSON")
    parser.add_argument("--poll_seconds", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--max_wait_hours", type=float, default=120.0, help="Max wait time in hours")
    parser.add_argument("--output_dir", type=str, default="docs/figures")
    parser.add_argument("--title_prefix", type=str, default="NYUv2 V2 Full-scale Phase5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_json = Path(args.runs_json)
    deadline = time.time() + args.max_wait_hours * 3600.0

    print(f"[wait_and_plot] Waiting for: {runs_json}")
    while not runs_json.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timeout waiting for {runs_json}")
        time.sleep(args.poll_seconds)

    print(f"[wait_and_plot] Found: {runs_json}")
    cmd = [
        "python",
        "-m",
        "scripts.compare_v2_methods",
        "--runs_json",
        str(runs_json),
        "--output_dir",
        args.output_dir,
        "--title_prefix",
        args.title_prefix,
    ]
    print("[wait_and_plot] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[wait_and_plot] Done.")


if __name__ == "__main__":
    main()
