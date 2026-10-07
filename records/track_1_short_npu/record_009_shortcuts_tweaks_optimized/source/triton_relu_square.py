"""Triton-Ascend fused ReLU-square forward/backward kernels."""
import os

os.environ.setdefault(
    "TRITON_CACHE_DIR", f"/tmp/triton_cache_record009_{os.environ.get('RANK', '0')}"
)

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 8192


@triton.jit
def _relu_square_kernel(x_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.maximum(x, 0.0)
    tl.store(x_ptr + offsets, x * x, mask=mask)


@triton.jit
def _backward_from_root_kernel(grad_ptr, root_ptr, out_ptr,
                               n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask)
    root = tl.load(root_ptr + offsets, mask=mask)
    scale = (root * 2.0).to(tl.bfloat16)
    out = (grad * scale).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _fused_backward_kernel(grad_ptr, y_ptr, out_ptr,
                           n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    root = tl.sqrt(y).to(tl.bfloat16)
    scale = (root * 2.0).to(tl.bfloat16)
    out = (grad * scale).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


class TritonHybridReluSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        _relu_square_kernel[grid](x, x.numel(), BLOCK=BLOCK_SIZE)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        root = torch.sqrt(y)
        out = torch.empty_like(grad)
        grid = (triton.cdiv(grad.numel(), BLOCK_SIZE),)
        _backward_from_root_kernel[grid](
            grad, root, out, grad.numel(), BLOCK=BLOCK_SIZE
        )
        return out


class TritonFusedReluSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        _relu_square_kernel[grid](x, x.numel(), BLOCK=BLOCK_SIZE)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        out = torch.empty_like(grad)
        grid = (triton.cdiv(grad.numel(), BLOCK_SIZE),)
        _fused_backward_kernel[grid](
            grad, y, out, grad.numel(), BLOCK=BLOCK_SIZE
        )
        return out
