"""Check the protected ZIP files byte-for-byte and use the official kernel scanner."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = json.loads((ROOT / "scripts/starter_hashes.json").read_text())
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("harness", "tests", "task", "templates")
              for p in (ROOT / folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if actual != expected:
        raise SystemExit("GUARD FAIL: protected starter files changed")
    spec = importlib.util.spec_from_file_location("challenge_rules", ROOT / "harness/check_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    own, _, lookalikes, banned, broken = rules.source_kernels(ROOT / "solution")
    if not own or lookalikes or banned or broken:
        raise SystemExit(f"GUARD FAIL: {lookalikes=} {banned=} {broken=}")
    print("GUARD OK; original kernels: " + ", ".join(sorted(own)))


if __name__ == "__main__":
    main()
