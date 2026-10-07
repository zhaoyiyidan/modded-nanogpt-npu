"""Triton-Ascend shortcut affine forward with native PyTorch backward."""
import os

os.environ.setdefault(
    "TRITON_CACHE_DIR", f"/tmp/triton_cache_record009_{os.environ.get('RANK', '0')}"
)

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 8192


@triton.jit
def _shortcut_affine_kernel(x_ptr, x0_ptr, lambdas_ptr, out_ptr,
                            n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x0 = tl.load(x0_ptr + offsets, mask=mask)
    lambda0 = tl.load(lambdas_ptr)
    lambda1 = tl.load(lambdas_ptr + 1)
    out = x * lambda0 + x0 * lambda1
    tl.store(out_ptr + offsets, out, mask=mask)


class TritonShortcutAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0, lambdas):
        out = torch.empty_like(x)
        grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
        _shortcut_affine_kernel[grid](
            x, x0, lambdas, out, x.numel(), BLOCK=BLOCK_SIZE
        )
        ctx.save_for_backward(x, x0, lambdas)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, x0, lambdas = ctx.saved_tensors
        # Preserve the baseline expressions and reductions for parameter grads.
        grad_x = grad * lambdas[0]
        grad_x0 = grad * lambdas[1]
        grad_lambdas = torch.stack(((grad * x).sum(), (grad * x0).sum()))
        return grad_x, grad_x0, grad_lambdas
