"""Pairwise rotary embedding, derived from the rotation equations (SOURCES.md).

For each half-pair (a,b), output = (a*c0-b*s0, b*c1+a*s1).
Backward applies the transpose: (ga*c0+gb*s1, gb*c1-ga*s0).
Products round separately, as in the unfused PyTorch expression; no FMA.
Q and K share a launch, including grouped-query attention with different head counts.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rope_pairs(Q, K, C, S, OQ, OK,
                    HQ: tl.constexpr, HK: tl.constexpr, SEQ: tl.constexpr, D: tl.constexpr,
                    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr,
                    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
                    CB: tl.constexpr, CS: tl.constexpr,
                    BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    batch, pos = token // SEQ, token % SEQ
    j = tl.arange(0, BLOCK)
    mask = j < D // 2
    if head < HQ:
        source = Q + batch * QB + head * QH + pos * QS
        target = OQ + ((batch * HQ + head) * SEQ + pos) * D
    else:
        source = K + batch * KB + (head - HQ) * KH + pos * KS
        target = OK + ((batch * HK + head - HQ) * SEQ + pos) * D
    a = tl.load(source + j, mask, 0).to(tl.float32)
    b = tl.load(source + j + D // 2, mask, 0).to(tl.float32)
    index = batch * CB + pos * CS + j
    c0 = tl.load(C + index, mask, 0).to(tl.float32)
    c1 = tl.load(C + index + D // 2, mask, 0).to(tl.float32)
    s0 = tl.load(S + index, mask, 0).to(tl.float32)
    s1 = tl.load(S + index + D // 2, mask, 0).to(tl.float32)
    if BACKWARD:
        ac = (a * c0).to(target.dtype.element_ty).to(tl.float32)
        bs = (b * s1).to(target.dtype.element_ty).to(tl.float32)
        bc = (b * c1).to(target.dtype.element_ty).to(tl.float32)
        ass = (a * s0).to(target.dtype.element_ty).to(tl.float32)
        first, second = ac + bs, bc - ass
    else:
        ac = (a * c0).to(target.dtype.element_ty).to(tl.float32)
        bs = (-b * s0).to(target.dtype.element_ty).to(tl.float32)
        bc = (b * c1).to(target.dtype.element_ty).to(tl.float32)
        ass = (a * s1).to(target.dtype.element_ty).to(tl.float32)
        first, second = ac + bs, bc + ass
    tl.store(target + j, first, mask)
    tl.store(target + j + D // 2, second, mask)


def supported(q, k, cos, sin, unsqueeze_dim=1):
    tensors = (q, k, cos, sin)
    if not (unsqueeze_dim == 1 and all(t.is_cuda for t in tensors)
            and all(t.device == q.device for t in tensors)
            and all(t.dtype in (torch.float16, torch.float32) for t in tensors)
            and q.ndim == k.ndim == 4 and cos.ndim == sin.ndim == 3
            and q.dtype == k.dtype and cos.dtype == sin.dtype
            and not cos.requires_grad and not sin.requires_grad):
        return False
    b, h, seq, d = q.shape
    return (0 < d <= 512 and d % 2 == 0 and b * h * seq > 0
            and 0 < k.shape[1] <= 64 and h <= 64
            and k.shape[0] == b and k.shape[2:] == (seq, d)
            and cos.shape == sin.shape and cos.shape[0] in (1, b)
            and cos.shape[1:] == (seq, d)
            and all(t.stride(-1) == 1 for t in tensors)
            and cos.stride() == sin.stride())


def launch(q, k, cos, sin, out_q, out_k, backward=False):
    b, h, seq, d = q.shape
    soup_rope_pairs[(b * seq, h + k.shape[1])](
        q, k, cos, sin, out_q, out_k, h, k.shape[1], seq, d,
        *q.stride()[:3], *k.stride()[:3],
        (0 if cos.shape[0] == 1 else cos.stride(0)), cos.stride(1),
        backward, triton.next_power_of_2(d // 2), num_warps=4, enable_fp_fusion=False)


class Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        dtype = torch.promote_types(q.dtype, cos.dtype)
        oq = torch.empty(q.shape, device=q.device, dtype=dtype)
        ok = torch.empty(k.shape, device=k.device, dtype=dtype)
        launch(q, k, cos, sin, oq, ok)
        ctx.save_for_backward(cos, sin)
        ctx.input_dtype = q.dtype
        return oq, ok

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        dq = torch.empty(gq.shape, device=gq.device, dtype=ctx.input_dtype)
        dk = torch.empty(gk.shape, device=gk.device, dtype=ctx.input_dtype)
        launch(gq.contiguous(), gk.contiguous(), cos, sin, dq, dk, backward=True)
        return dq, dk, None, None


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_rope", False):
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return Rotary.apply(q, k, cos, sin)

    apply_rotary_pos_emb._soup_rope = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
    return True
