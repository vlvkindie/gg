# Soup: original RMSNorm and rotary embedding kernels

This project targets the fixed Soup Kernel Challenge workload on a Kaggle T4.
It imports the supplied starter kit into https://github.com/vlvkindie/gg and adds
three original Triton kernel functions. **T4 correctness, training-time share,
speed, memory, and adapter quality are not measured yet. This is a candidate,
not a validated competition submission.**

## What changed

- `solution/rmsnorm.py`: row-wise normalization with FP32 reductions, forward
  and backward, frozen normalization weights, and the original dtype promotion.
  It removes intermediate normalization tensors and saves one inverse RMS per row.
- `solution/rope.py`: one launch for both query and key rotations, and one for
  their backward pass. Supports grouped-query heads, transposed tensors, and
  broadcast positional coefficients. Products round before addition.
- `solution/__init__.py`: lazy, repeatable patch application and an explicit
  `APPLIED` registry. Missing dependencies are reported to stderr. Unsupported
  shapes, devices, dtypes, and trainable normalization/position weights use
  the original functions.
- The starter's `example_*` SwiGLU kernels remain attributed examples; they are
  excluded from the original-kernel share. Their patch is now idempotent.

Kernel equations and idea sources are in `SOURCES.md`. No trainer or loss patch
is added. The original loss, entropy, accuracy, token counts, and rewind recording
therefore retain their original computation paths. Their numerical agreement
after training must still be established by the harness.

## Run on Kaggle

Import the root-level `kaggle_gg.ipynb` into Kaggle. Enable Internet and T4 x2.
For this private repository, enable a read-only `GITHUB_TOKEN` Kaggle secret.
The notebook selects the candidate branch, installs the starter's exact versions,
runs guard, GPU preflight, all Soup tests, then two base/solution repetitions,
rule checks, scoring, and a compact report. No changes to the protected template
are needed. Every validation command stops on failure.

Alternatively, in the specified Linux/CUDA environment:

```bash
bash scripts/gates.sh
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q -p no:cacheprovider
python scripts/verify_patches.py
python task/prepare_data.py
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
python scripts/report.py
```

A quick diagnostic run uses `python harness/run_task.py --only solution --out
runs_try --profile`. It cannot establish base-versus-solution validity.

## Results and limits

Local validation: guard and compilation passed, solution tests report **7 passed,
21 skipped**, and the supplied Soup regressions report **79 passed** with
`SOUP_APPLY_SOLUTION=1`. Triton is unavailable locally: these CPU results confirm
algebra/import compatibility, not compiled-kernel or GPU-patched correctness.
Details are recorded in `LOG.md`. No benchmark numbers are inferred from the starter's published
measurements. The original-kernel share may be below the required threshold even
if correctness tests pass; profiling on the T4 decides the next iteration.

Fused cross-entropy and NF4 optimization remain future work. This first iteration
keeps the loss and quantization paths intact to establish a trustworthy baseline.
No speedup or memory reduction is promised. The local GTX 1070 is not the required
benchmark hardware, and Triton GPU execution is not available in the local Windows
test environment. Kernels do not implement higher-order gradients.

The protected starter directories are checked byte-for-byte against a manifest.
The official harness remains the final authority. After all checks pass, pin
the notebook to the validated commit and save a public Kaggle version. The Kaggle
URL, team details, video, presentation, and final measurements are pending in
`SUBMISSION.md`; none has been fabricated.

License: Apache-2.0; see `LICENSE`, `NOTICE`, and `SOURCES.md`.
