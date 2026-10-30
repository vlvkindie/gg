# Architecture

## Entry point and ownership

The unchanged harness imports `solution` and calls `solution.apply()` before Soup
constructs the model. All training patches and original kernels live inside
`solution/`, because the jury copies only that directory into a clean starter.
Scripts, notebooks and documentation support local reproduction and submission.

`solution.apply()` lazily loads three patches. `APPLIED` records installation
status. Repeated calls retain the same wrappers. Missing libraries produce an
explicit stderr message; GPU preflight fails when required patches are absent.

## Normalization

`rmsnorm.py` replaces `Qwen2RMSNorm.forward` only for supported CUDA tensors and
frozen scale weights. For each row, `r = (mean(x*x) + eps)^(-1/2)` and
`y = weight * cast_input_dtype(x*r)`. The forward kernel saves the original input
and one inverse RMS per row. Backward computes
`dx = r*(g - x*r*r*mean(g*x))`, with `g = cast_input_dtype(dy*weight)`.
Reductions use FP32. Output dtype follows PyTorch promotion. Other calls use
the original method, including trainable scale weights and CPU tensors.

## Rotary positions

`rope.py` replaces `apply_rotary_pos_emb` for the supported Qwen2 convention.
The grid covers batch/sequence positions and query plus key heads. It handles
different Q/K head counts, transposed input strides, and coefficients broadcast
across a batch. For a half-pair `(a,b)`, the output is
`(a*c0 - b*s0, b*c1 + a*s1)`. Backward applies the transposed Jacobian.
Products round separately before addition; compiler FP fusion is disabled.
Unsupported arguments, layouts or differentiable coefficients retain the original.

## Example and preserved behavior

`swiglu.py` retains the supplied `example_*` kernels and attribution. They do not
count toward original-kernel share. Its wrapper is idempotent. There is no patch
of the trainer, loss, quantizer, optimizer, batch formation or task configuration.
Numerical equivalence throughout training still requires the official T4 checks.

## Verification and outputs

`scripts/guard.py` checks protected-file SHA256 hashes and invokes the official
source scanner. Reference tests compare values and input gradients with PyTorch.
`scripts/run_validation.py` provides CPU, quick and full modes. Full mode runs
two base/solution repetitions and official checks. `scripts/report.py` validates
result and source hashes before printing measured evidence. Original kernel
launches, share, metrics and quality come from the unchanged harness.
