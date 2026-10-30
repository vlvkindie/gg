# Demo recording script (2 minutes 40 seconds)

This is a recording plan, not a claim that a benchmark has completed.
Record the repository and actual T4 output when available.

## 0:00–0:20 — Fixed task

Show README and `task/task.yaml`. Explain that this is Qwen2.5-1.5B-Instruct
SFT with NF4 and the unchanged LoRA task on a Kaggle T4. The aim is less time
or memory without changing the task, logged outputs or adapter quality.

## 0:20–0:50 — Repository and scope

Show `main` and its root folders, then `solution/`. Open `SOURCES.md` and
`solution/__init__.py`. Explain narrow class/function patches and fallback paths.
Identify the shipped SwiGLU as an example excluded from original-kernel share.

## 0:50–1:20 — RMSNorm

Show forward and backward in `solution/rmsnorm.py`. Explain one reduction per
row, FP32 accumulation, preserved intermediate casts and the frozen-weight guard.
Show the PyTorch reference tests rather than claiming bitwise training equivalence.

## 1:20–1:45 — RoPE

Show `solution/rope.py`. Explain that Q and K share a launch, grouped-query heads
may differ, and backward uses the transposed two-coordinate Jacobian. Show
broadcast-coefficient and strided-input cases in the reference tests.

## 1:45–2:20 — Real validation

Show the Kaggle notebook and actual `scripts/report.py` output. Read measured
base/solution wall time, peak memory, quality delta, kernel share and check status.
If T4 execution has not happened, state that these are pending and that this is
an implementation walkthrough, not a final results demo. Never substitute local
CPU tests or starter estimates for measured GPU results.

## 2:20–2:40 — Limits and reproduction

Show the validated SHA and saved public notebook link when available. State any
failed checks. Mention unsupported cases, pending GPU validation where applicable,
and the next optimization chosen from the actual profile. End with repository,
notebook and sources links. Keep the complete recording below three minutes.
