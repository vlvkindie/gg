# gg: Soup Kernel Challenge

This project targets the fixed Soup Kernel Challenge workload on a Kaggle T4.
It imports the supplied starter kit into https://github.com/vlvkindie/gg and adds
three original Triton kernel functions. **T4 correctness, training-time share,
speed, memory, and adapter quality are not measured yet. This is a candidate,
not a validated competition submission.**

## Fixed task

Qwen2.5-1.5B-Instruct, Soup 0.77.0, SFT on 256 public Kazakh instruction rows,
batch 4, max length 512, one epoch (64 steps), LoRA r=16 and alpha=32, 4-bit NF4.
The fixed task is in `task/task.yaml`. Only Kaggle T4 measurements count.

## Repository layout

- `solution/`: original kernels, guarded patches and their reference tests.
- `harness/`: unchanged benchmark, rule checks, adapter evaluation and scoring.
- `task/`: unchanged configuration and public-data preparation.
- `tests/`: unchanged Soup regression tests.
- `templates/`: unchanged original notebook and submission template.
- `scripts/`: guard, CPU/GPU validation, preflight and report commands.
- `docs/`: architecture, complete requirements checklist, presentation and demo script.
- `.github/workflows/cpu.yml`: CPU validation on push and pull requests.

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
The notebook clones `main`, installs the starter's exact versions,
runs guard, GPU preflight, all Soup tests, then two base/solution repetitions,
rule checks, scoring, and a compact report. No changes to the protected template
are needed. Every validation command stops on failure.

Clone this repository and install the pinned dependencies before training:

```bash
git clone https://github.com/vlvkindie/gg.git
cd gg
python -m pip install -r requirements-train.txt
python scripts/run_validation.py full
```

Use `quick --profile` for a diagnostic solution-only run. Full mode runs both
sides twice. The validation runner stops on failure and keeps GPU work in child
processes. In the specified Linux/CUDA environment, the individual commands are:

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

## CPU development

```bash
python -m venv .venv
# Activate .venv using your shell.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python scripts/run_validation.py cpu
```

CPU mode verifies import tolerance and algebra while skipping CUDA comparisons.
GitHub Actions runs this same command. It does not establish GPU correctness.

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

## Submission materials

- [Architecture](docs/ARCHITECTURE.md)
- [Every requirement and its evidence/status](docs/REQUIREMENTS.md)
- [Presentation, draft pending T4 results](docs/presentation.pptx)
- [Demo recording script, 2:40](docs/DEMO_SCRIPT.md)
- [Root submission form](SUBMISSION.md)
- [Original starter instructions](README.starter.md)

Public saved Kaggle notebook URL: **pending**. A local notebook file is not a
public saved Kaggle version. Final video URL, team details and measured results
are pending. The repository is private; jury access must be arranged through
the platform or an explicit publication decision before submission.

All ten official checks are required. The original kernels must launch and
account for at least 5% of timed training; the shipped example is excluded. Loss
curve gaps must meet the first-step 1% and mean 3% limits, held-out loss must be
within +0.5%, and time/memory must stay within the 10% regression limits. The
complete checklist also covers exact task/output agreement and clean measurement.
No all-checks pass or final competition admission is claimed yet.
