import torch
import triton
import triton.language as tl


@triton.jit
def _softcap_forward(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask, 0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-x / 7.5))
        tl.store(Y + offsets, 30.0 * sigmoid, mask)


@triton.jit
def _softcap_backward(X, G, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + offsets, mask, 0.0).to(tl.float32)
        g = tl.load(G + offsets, mask, 0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-x / 7.5))
        dx = g * (4.0 * sigmoid * (1.0 - sigmoid))
        tl.store(DX + offsets, dx, mask)


class _SoftcapFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.empty_like(x, dtype=torch.float32)
        _softcap_forward[(48,)](x, out, x.numel(), 8192)
        return out

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        _softcap_backward[(48,)](x, grad.contiguous(), dx, x.numel(), 8192)
        return dx


def softcap_logits(x):
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return 30 * torch.sigmoid(x.float() / 7.5)
    return _SoftcapFused.apply(x)
