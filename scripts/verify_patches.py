"""T4 preflight: require installed patches and run the real GPU comparisons."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import solution
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; run this preflight on the Kaggle T4")
    solution.apply()
    if not all(solution.APPLIED.values()):
        raise SystemExit(f"Patches missing: {solution.APPLIED}")
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    norm = qwen.Qwen2RMSNorm(1536).cuda().requires_grad_(False)
    x = torch.randn(2, 17, 1536, device="cuda", requires_grad=True)
    norm(x).sum().backward()
    q = torch.randn(2, 12, 17, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 2, 17, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    c = torch.randn(2, 17, 128, device="cuda")
    s = torch.randn_like(c)
    oq, ok = qwen.apply_rotary_pos_emb(q, k, c, s)
    (oq.sum() + ok.sum()).backward()
    torch.cuda.synchronize()
    subprocess.run([sys.executable, "-m", "pytest", "solution", "-q"], cwd=ROOT, check=True)
    print("PATCH PREFLIGHT OK", solution.APPLIED)


if __name__ == "__main__":
    main()
