# Sources and derivation

- The supplied `soup-kernel-challenge-main.zip` is the starter kit. Its LICENSE,
  NOTICE, harness, task, tests, and templates are preserved. The example SwiGLU
  kernels are retained with their `example_*` names and are not claimed as original.
- Qwen2 operation contracts: Transformers **v5.16.1**,
  https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py
  (`Qwen2RMSNorm.forward`, `apply_rotary_pos_emb`, and `rotate_half`). These define
  the reference behavior, including intermediate casts; no third-party kernel
  implementation was used.
- RMSNorm idea: Zhang and Sennrich, *Root Mean Square Layer Normalization*,
  https://arxiv.org/abs/1910.07467. Implemented directly from
  `y = w*x/sqrt(mean(x*x)+eps)` and its derivative. The gradient of the scale
  is unnecessary only when the scale is frozen; other calls use the original.
- RoPE idea: Su et al., *RoFormer: Enhanced Transformer with Rotary Position
  Embedding*, https://arxiv.org/abs/2104.09864. Implemented from a two-coordinate
  rotation and its transposed Jacobian. Separate coefficient halves are supported.

The kernels in `rmsnorm.py` and `rope.py` were written for this solution from these
equations. No Liger, Unsloth, or cross-entropy kernel source was copied or recreated.
