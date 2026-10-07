"""Softcap with exact-oriented forward and chunk-launched fused backward."""
import os

os.environ.setdefault(
    "TRITON_CACHE_DIR", f"/tmp/triton_cap_cancel_chunked_{os.environ.get('RANK', '0')}"
)

import torch
import triton
import triton.language as tl
from triton_cap_hybrid32k import (
    BLOCK_SIZE as FORWARD_BLOCK,
    _divide_bf16_kernel,
    _multiply_cast_kernel,
)


BACKWARD_BLOCK = 16384
MAX_PROGRAMS = 65535


@triton.jit
def _cancelled_backward_chunk(grad_ptr, tanh_ptr, out_ptr, base,
                              n_elements: tl.constexpr,
                              BLOCK: tl.constexpr):
    offsets = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.bfloat16)
    t = tl.load(tanh_ptr + offsets, mask=mask)
    t2 = (t * t).to(tl.bfloat16)
    factor = (1.0 - t2).to(tl.bfloat16)
    out = (grad * factor).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


class TritonCancelledChunked16KNoMaterializeCapCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        ctx.mark_dirty(x)
        grid = (triton.cdiv(x.numel(), FORWARD_BLOCK),)
        _divide_bf16_kernel[grid](x, x, x.numel(), BLOCK=FORWARD_BLOCK)
        torch.tanh_(x)
        out = torch.empty_like(x, dtype=torch.float32)
        _multiply_cast_kernel[grid](x, out, x.numel(), BLOCK=FORWARD_BLOCK)
        ctx.save_for_backward(x)
        return out, x

    @staticmethod
    def backward(ctx, grad_out, _grad_saved):
        (t,) = ctx.saved_tensors
        grad_x = torch.empty_like(t)
        chunk_elements = MAX_PROGRAMS * BACKWARD_BLOCK
        for base in range(0, t.numel(), chunk_elements):
            programs = min(
                MAX_PROGRAMS,
                triton.cdiv(t.numel() - base, BACKWARD_BLOCK),
            )
            _cancelled_backward_chunk[(programs,)](
                grad_out, t, grad_x, base, t.numel(), BLOCK=BACKWARD_BLOCK
            )
        return grad_x
