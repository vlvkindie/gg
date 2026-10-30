"""RMS normalization derived from its definition; idea and source in SOURCES.md.

r = (mean(x**2) + eps)**(-1/2), y = w * cast_x(x*r).
For frozen w: dx = r*(g - x*r**2*mean(g*x)), g = cast_x(dy*w).
The cast before multiplication preserves Qwen2's intermediate rounding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def soup_rmsnorm_forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    normalized = (x * inverse).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * N + col, normalized * w, col < N)
    tl.store(R + row, inverse)


@triton.jit
def soup_rmsnorm_backward(DY, X, W, R, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    dy = tl.load(DY + row * N + col, col < N, 0).to(tl.float32)
    g = (dy * w).to(X.dtype.element_ty).to(tl.float32)
    inverse = tl.load(R + row)
    projection = tl.sum(g * x, 0) / N
    dx = inverse * (g - x * inverse * inverse * projection)
    tl.store(DX + row * N + col, dx, col < N)


def supported(x, weight):
    return (x.is_cuda and weight.is_cuda and x.device == weight.device
            and x.dtype in (torch.float16, torch.float32)
            and weight.dtype in (torch.float16, torch.float32)
            and x.ndim >= 2 and 0 < x.shape[-1] <= 8192 and x.numel() > 0
            and x.is_contiguous() and weight.is_contiguous()
            and weight.shape == (x.shape[-1],) and not weight.requires_grad)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        n = x.shape[-1]
        out = torch.empty(x.shape, device=x.device, dtype=torch.promote_types(x.dtype, weight.dtype))
        inverse = torch.empty(x.numel() // n, device=x.device, dtype=torch.float32)
        soup_rmsnorm_forward[(inverse.numel(),)](
            x, weight, out, inverse, n, eps, triton.next_power_of_2(n),
            num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inverse)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, inverse = ctx.saved_tensors
        dx = torch.empty_like(x)
        soup_rmsnorm_backward[(inverse.numel(),)](
            grad.contiguous(), x, weight, inverse, dx, x.shape[-1],
            triton.next_power_of_2(x.shape[-1]), num_warps=4, enable_fp_fusion=False)
        return dx, None, None


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_rmsnorm", False):
        return True
    original = cls.forward

    def forward(self, hidden_states):
        if not supported(hidden_states, self.weight):
            return original(self, hidden_states)
        return RMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)

    forward._soup_rmsnorm = True
    cls.forward = forward
    return True
