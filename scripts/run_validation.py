"""Reproducible CPU, quick T4, or full T4 validation. Stops on command failure."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args, env=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "quick", "full"))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    run(python, "scripts/guard.py")
    run(python, "-m", "compileall", "-q", "solution", "scripts")
    if args.mode == "cpu":
        run(python, "-m", "pytest", "solution", "-q")
    else:
        # Do not start a CUDA context in this orchestration process.
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--id=0",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True)
        if "T4" not in gpu.stdout:
            raise SystemExit("The official benchmark requires GPU 0 to be a T4")
        run(python, "scripts/verify_patches.py")
    run(python, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
        env=dict(os.environ, SOUP_APPLY_SOLUTION="1"))
    if args.mode == "cpu":
        print("CPU VALIDATION PASSED; GPU execution and benchmark remain unverified")
        return
    run(python, "task/prepare_data.py")
    profile = ["--profile"] if args.profile else []
    if args.mode == "quick":
        run(python, "harness/run_task.py", "--only", "solution", "--out", "runs_try", *profile)
        print("Quick results: runs_try/result.json. Base comparisons require full mode.")
        return
    run(python, "harness/run_task.py", "--repeats", "2", *profile)
    run(python, "harness/check_rules.py")
    run(python, "harness/score.py", "runs/result.json")
    run(python, "scripts/report.py")


if __name__ == "__main__":
    main()
