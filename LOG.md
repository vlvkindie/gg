# Iteration log

## 2026-10-30 — first candidate

model: Codex

Baseline: user-supplied `soup-kernel-challenge-main.zip`. Destination:
`vlvkindie/gg`, candidate branch `codex/soup-first-kernels`.

Diagnosis: the destination contained only a README; the ZIP contains the stock
starter, including only the example SwiGLU implementation. There was no measured
or previously optimized v0 to validate.

Source reconnaissance: read the starter README, task configuration, common
settings, harness training instrumentation, rule checks and regression fixtures;
inspected the exact Transformers 5.16.1, TRL 0.29.1 and Soup 0.77.0 wheel sources.
Confirmed normalization casts, broadcast rotary positions, and the independent
TRL/Soup metrics and rewind paths. No NF4 or trainer patch is attempted.

Changes: original RMSNorm forward/backward and joint Q/K RoPE forward/backward;
idempotent patches, explicit patch status, CPU import tolerance, original example
attribution, reference tests, guard, GPU preflight, compact report, configured
Kaggle notebook, and English documentation. Protected starter directories remain
byte-identical to the ZIP.

Expected effect: fewer intermediate normalization tensors and fewer rotary
launches; the example retains its original memory benefit. Actual speed, memory,
quality, and original-kernel share require measurement. No numeric benefit claimed.

Validation:
- G0: GUARD OK. Official scanner recognizes three original kernel functions;
  no banned imports, syntax failures, or example lookalikes.
- G1: compileall passed; solution tests: **7 passed, 21 skipped**. CPU tests check
  the derivative algebra, import tolerance, and official kernel classification.
- CPU Soup regressions: **79 passed** with `SOUP_APPLY_SOLUTION=1`.
  Triton is absent locally, so patch registration reports unavailable kernels.
  This establishes CPU compatibility, not GPU-patched regression correctness.
- Notebook code cells compile; no saved outputs or benchmark numbers fabricated.
- Protected-file hash verification and `git diff --check` passed.
- Report refuses to emit results without harness files; GPU preflight rejects CPU.

Environment: Windows, Python 3.12.14, torch 2.11.0+cpu, Soup 0.77.0,
pytest 9.2.0; NumPy 2.3.5 and pandas 3.0.1 from the bundled runtime, pyarrow 26.0.0
in the local test environment. The target remains the starter's Linux/Kaggle
environment and exact package versions. No packages are installed during a
measured training run.

G3/G4/G5: pending T4 execution. s: unmeasured; m: unmeasured; eval delta:
unmeasured; own share: unmeasured. Do not merge this candidate into main or mark
it valid until the GPU checks pass. Run `kaggle_gg.ipynb` and return the output of
`python scripts/report.py`; profile if kernel share is insufficient.

## 2026-10-30 — repository completion requested by the user

model: Codex

The user explicitly requested a complete repository on the default branch and
a separate unpacked `solution/` delivery. This supersedes the earlier decision
to keep repository setup only in a draft pull request. Publishing implementation
to main does not establish GPU validity or final competition admission.

Repository setup and kernel placement are published as separate commits.
Added pinned dependencies, LF policy, CPU CI, a CPU/quick/full runner, complete
architecture and requirements documentation, results status, expanded submission
form, an editable presentation draft and a timed demo script. The configured
notebook now clones main. Kernel code and protected starter files are unchanged.

Verified the new CPU runner end-to-end: guard and compilation passed, solution
tests 7 passed / 21 skipped, and Soup regressions 79 passed. The unpacked solution
contains the six Python files from the implementation. T4 execution, a public
saved Kaggle version, actual measurements, final results recording, team details
and platform submission remain external requirements with explicit pending status.
