"""Forward/backward comparisons with the pinned Qwen2 equations."""

import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", [73, 1536])
def test_rmsnorm(dtype, width):
    from solution.rmsnorm import RMSNorm

    torch.manual_seed(7)
    x = torch.randn(2, 17, width, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(width, device="cuda", dtype=torch.float32)
    grad = torch.randn_like(x, dtype=torch.float32)
    z = x.float()
    reference = weight * (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(dtype)
    expected_dx, = torch.autograd.grad(reference, x, grad)
    actual = RMSNorm.apply(x, weight, 1e-6)
    actual_dx, = torch.autograd.grad(actual, x, grad)
    tolerance = 3e-3 if dtype == torch.float16 else 3e-5
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_dx, expected_dx, atol=tolerance, rtol=tolerance)


def rotate(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm_derivative_algebra_on_cpu(dtype):
    # Check the derived Jacobian against autograd even when GPU tests cannot run.
    torch.manual_seed(11)
    x = torch.randn(7, 73, dtype=dtype, requires_grad=True)
    w = torch.randn(73)
    r = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    y = w * (x.float() * r).to(dtype)
    dy = torch.randn_like(y)
    expected, = torch.autograd.grad(y, x, dy)
    g = (dy * w).to(dtype).float()
    actual = (r * (g - x.float() * r.square() * (g * x.float()).mean(-1, keepdim=True))).to(dtype)
    torch.testing.assert_close(actual, expected, atol=3e-3 if dtype == torch.float16 else 2e-6,
                               rtol=3e-3 if dtype == torch.float16 else 2e-5)


@pytest.mark.parametrize("dtype,coeff_dtype", [(torch.float16, torch.float16),
                                              (torch.float16, torch.float32),
                                              (torch.float32, torch.float32)])
def test_rotary_transposed_jacobian_on_cpu(dtype, coeff_dtype):
    torch.manual_seed(12)
    x = torch.randn(2, 3, 4, 16, dtype=dtype, requires_grad=True)
    c = torch.randn(1, 1, 4, 16, dtype=coeff_dtype)
    s = torch.randn_like(c)
    y = x * c + rotate(x) * s
    dy = torch.randn_like(y)
    expected, = torch.autograd.grad(y, x, dy)
    ga, gb = dy.chunk(2, -1)
    c0, c1 = c.chunk(2, -1)
    s0, s1 = s.chunk(2, -1)
    da = (ga.float() * c0.float()).to(dtype) + (gb.float() * s1.float()).to(dtype)
    db = (gb.float() * c1.float()).to(dtype) - (ga.float() * s0.float()).to(dtype)
    torch.testing.assert_close(torch.cat((da, db), -1), expected, atol=0, rtol=0)


@gpu
@pytest.mark.parametrize("dtype,coeff_dtype", [(torch.float16, torch.float16),
                                              (torch.float16, torch.float32),
                                              (torch.float32, torch.float32)])
@pytest.mark.parametrize("only_q", [False, True])
@pytest.mark.parametrize("coeff_batch", [1, 2])
def test_rope(dtype, coeff_dtype, only_q, coeff_batch):
    from solution.rope import Rotary

    torch.manual_seed(9)
    # Deliberately noncontiguous heads, GQA, and independent half coefficients.
    q = torch.randn(2, 19, 12, 128, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(2, 19, 2, 128, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    cos = torch.randn(coeff_batch, 19, 128, device="cuda", dtype=coeff_dtype)
    sin = torch.randn_like(cos)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    reference = q * c + rotate(q) * s, k * c + rotate(k) * s
    actual = Rotary.apply(q, k, cos, sin)
    grads = tuple(torch.randn_like(y) for y in reference)
    chosen = slice(0, 1) if only_q else slice(None)
    expected_dx = torch.autograd.grad(reference[chosen], (q, k), grads[chosen],
                                      allow_unused=True)
    actual_dx = torch.autograd.grad(actual[chosen], (q, k), grads[chosen], allow_unused=True)
    tolerance = 4e-3 if dtype == torch.float16 else 2e-5
    for got, expected in zip(actual, reference):
        assert got.dtype == expected.dtype
        torch.testing.assert_close(got, expected, atol=tolerance, rtol=tolerance)
    for i, (got, expected) in enumerate(zip(actual_dx, expected_dx)):
        if expected is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            torch.testing.assert_close(got, expected, atol=tolerance, rtol=tolerance)


def test_apply_is_repeatable_without_gpu():
    import solution

    solution.apply()
    first = dict(solution.APPLIED)
    solution.apply()
    assert solution.APPLIED == first
    assert set(first) == {"rmsnorm", "rope", "example_mlp"}


def test_official_scanner_recognizes_original_kernels():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("challenge_rules", root / "harness/check_rules.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    own, copies, lookalikes, banned, broken = rules.source_kernels(root / "solution")
    assert set(own) == {"soup_rmsnorm_forward", "soup_rmsnorm_backward", "soup_rope_pairs"}
    assert not lookalikes and not banned and not broken
    assert "example_swiglu_fwd_kernel" in copies


def test_cpu_fallback_if_triton_is_available():
    pytest.importorskip("triton")
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    import solution

    solution.apply()
    norm = qwen.Qwen2RMSNorm(16)
    x = torch.randn(2, 3, 16, requires_grad=True)
    expected = norm.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    torch.testing.assert_close(norm(x), expected)
    norm(x).sum().backward()
    assert norm.weight.grad is not None and x.grad is not None
    q = torch.randn(2, 4, 3, 16, requires_grad=True)
    c, s = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    got, _ = qwen.apply_rotary_pos_emb(q, q, c, s)
    torch.testing.assert_close(got, q * c[:, None] + rotate(q) * s[:, None])
