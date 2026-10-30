"""Print actual harness evidence, rejecting stale checks. No synthetic results."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    result_path = args.out / "result.json"
    checks_path = args.out / "checks.json"
    if not result_path.exists() or not checks_path.exists():
        raise SystemExit("Missing T4 results: run the harness and check_rules.py first")
    result = json.loads(result_path.read_bytes())
    checks = json.loads(checks_path.read_bytes())
    spec = importlib.util.spec_from_file_location("challenge_rules", ROOT / "harness/check_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    if checks.get("result_sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise SystemExit("Stale checks: result.json changed")
    if checks.get("solution_sha256") != rules.solution_sha256(ROOT / "solution"):
        raise SystemExit("Stale checks: solution changed")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    mode = "quick" if result.get("task", {}).get("only") else "full"
    print(f"commit: {commit}   mode: {mode}")
    ratios = result.get("summary", {}).get("ratios", {})
    print(f"s: {ratios.get('speedup', 'unavailable')}   "
          f"m: {ratios.get('memory_ratio', 'unavailable')}   "
          f"eval delta %: {ratios.get('quality_delta_pct', 'unavailable')}")
    for check in checks.get("checks", []):
        print(f"{'PASS' if check['ok'] else 'FAIL'} {check['check']}: {check['detail']}")
    own = rules.source_kernels(ROOT / "solution")[0]
    for run in result.get("runs", {}).get("solution", []):
        metrics = run.get("metrics", {})
        launched = {k: v for k, v in metrics.get("triton_launches", {}).items() if k in own}
        elapsed = metrics.get("timed_loop_s", 0)
        ms = sum(v for k, v in metrics.get("own_kernel_ms", {}).items() if k in own)
        share = 100 * ms / (elapsed * 1000) if elapsed else None
        print(f"{run.get('run')}: own launches={launched}; share %={share}")
    warnings = []
    for log in args.out.glob("*/train.log"):
        warnings.extend(f"{log.name}: {line}" for line in log.read_text(errors="replace").splitlines()
                        if "not applied" in line)
    print('"not applied" in logs: ' + (" | ".join(warnings) if warnings else "none"))


if __name__ == "__main__":
    main()
