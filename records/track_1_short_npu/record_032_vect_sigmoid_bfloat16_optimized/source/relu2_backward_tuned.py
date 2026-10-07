import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _backward(R, G, DX, N: tl.constexpr, BLOCK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        r = tl.load(R + offsets, offsets < N, 0)
        g = tl.load(G + offsets, offsets < N, 0)
        twice = (2.0 * r.to(tl.float32)).to(r.dtype).to(tl.float32)
        dx = (g.to(tl.float32) * twice).to(r.dtype)
        dx = tl.where(r.to(tl.float32) <= 0.0, 0.0, dx)
        tl.store(DX + offsets, dx, offsets < N)


class _ReluSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        relu = F.relu(x)
        ctx.save_for_backward(relu)
        return relu.square()

    @staticmethod
    def backward(ctx, grad):
        (relu,) = ctx.saved_tensors
        if relu.dtype == torch.float32:
            relu = relu.pow(1)
        dx = torch.empty_like(relu)
        block = 4096 if relu.dtype == torch.float32 else 8192
        _backward[(48,)](relu, grad.contiguous(), dx, relu.numel(), block)
        return dx


def relu_square(x):
    if x.dtype not in (torch.bfloat16, torch.float32) or not x.is_contiguous():
        return F.relu(x).square()
    return _ReluSquare.apply(x)
