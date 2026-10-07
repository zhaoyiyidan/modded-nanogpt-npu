"""Exact-oriented Triton/official hybrid for logits cap and FP32 cast."""
import os

os.environ.setdefault(
    "TRITON_CACHE_DIR", f"/tmp/triton_cache_record009_{os.environ.get('RANK', '0')}"
)

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 32768


@triton.jit
def _divide_bf16_kernel(x_ptr, out_ptr, n_elements: tl.constexpr,
                        BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = (x / 30.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _multiply_cast_kernel(x_ptr, out_ptr, n_elements: tl.constexpr,
                          BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = (x * 30.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


@triton.jit
def _grad_cast_multiply_kernel(x_ptr, out_ptr, n_elements: tl.constexpr,
                               BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.bfloat16)
    out = (x * 30.0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


class TritonHybridCapCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        scaled = torch.empty_like(x)
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        _divide_bf16_kernel[grid](x, scaled, x.numel(), BLOCK=BLOCK_SIZE)
        t = torch.tanh(scaled)
        out = torch.empty_like(x, dtype=torch.float32)
        _multiply_cast_kernel[grid](t, out, x.numel(), BLOCK=BLOCK_SIZE)
        ctx.save_for_backward(t)
        return out

    @staticmethod
    def backward(ctx, grad):
        (t,) = ctx.saved_tensors
        grid = (triton.cdiv(t.numel(), BLOCK_SIZE),)
        grad_t = torch.empty_like(t)
        _grad_cast_multiply_kernel[grid](
            grad, grad_t, t.numel(), BLOCK=BLOCK_SIZE
        )
        grad_scaled = torch.ops.aten.tanh_backward(grad_t, t)
        grad_x = torch.empty_like(t)
        _divide_bf16_kernel[grid](
            grad_scaled, grad_x, t.numel(), BLOCK=BLOCK_SIZE
        )
        return grad_x


class TritonInplaceHybridCapCast(torch.autograd.Function):
    """Memory-reduced variant: reuse the BF16 logits buffer for scaled/tanh."""

    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        _divide_bf16_kernel[grid](x, x, x.numel(), BLOCK=BLOCK_SIZE)
        torch.tanh_(x)
        out = torch.empty_like(x, dtype=torch.float32)
        _multiply_cast_kernel[grid](x, out, x.numel(), BLOCK=BLOCK_SIZE)
        ctx.save_for_backward(x)
        return out, x

    @staticmethod
    def backward(ctx, grad, _grad_saved):
        (t,) = ctx.saved_tensors
        grid = (triton.cdiv(t.numel(), BLOCK_SIZE),)
        grad_t = torch.empty_like(t)
        _grad_cast_multiply_kernel[grid](
            grad, grad_t, t.numel(), BLOCK=BLOCK_SIZE
        )
        grad_scaled = torch.ops.aten.tanh_backward(grad_t, t)
        grad_x = torch.empty_like(t)
        _divide_bf16_kernel[grid](
            grad_scaled, grad_x, t.numel(), BLOCK=BLOCK_SIZE
        )
        return grad_x
