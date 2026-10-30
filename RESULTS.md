# Results

## Measured locally

Environment: Windows, Python 3.12.14, torch 2.11.0+cpu, Soup 0.77.0.

- Protected starter files and original-kernel source scan: passed.
- Syntax compilation: passed.
- Solution tests: 7 passed, 21 skipped (CUDA/Triton-dependent coverage).
- Soup regressions with `SOUP_APPLY_SOLUTION=1`: 79 passed.
- Notebook code cells: syntax-checked, outputs empty.

Local Triton is unavailable. The CPU results verify derivative algebra, import
compatibility and source classification. They do not verify compiled kernels,
GPU-patched regressions, training-time kernel share or adapter equivalence.

## Official T4 comparison

No official T4 run has completed for this implementation.

- Base / solution wall time: pending.
- Base / solution peak memory in MiB: pending.
- Base / solution held-out adapter loss: pending.
- Speedup `s = base time / solution time`: pending.
- Memory ratio `m = base peak / solution peak`: pending.
- Quality delta: pending.
- Original-kernel training share: pending.
- All ten checks: not established.

The starter's estimates are not this solution's measured results and are not used
as its baseline. Run `python scripts/run_validation.py full` on the T4, retain
`runs/result.json` and `runs/checks.json`, and copy their measured values here and
into SUBMISSION. `scripts/report.py` rejects stale result/source hashes.

## Limits and next iteration

RMSNorm requires frozen scale weights and supported contiguous input. RoPE
supports the guarded Qwen2 layout and non-differentiable positional coefficients.
Other calls use the original methods. Higher-order gradients are unsupported.
Fused cross-entropy, NF4 changes and LoRA fusion are not implemented. The current
original-kernel share could be insufficient. Select the next optimization from
an actual T4 profile, then rerun the correctness, quality and performance gates.
