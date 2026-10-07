import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
# FlexAttention, CUDA FP8 scaled_mm, Triton, and torch.compile are CUDA/GPU-specific.
# The NPU port below uses torch_npu.npu_fusion_attention and pure PyTorch bf16 fallbacks.

# -----------------------------------------------------------------------------
# NPU replacements for GPU-only Record 27 operators

# The upstream record uses CUDA FP8 scaled_mm for the lm_head and two Triton
# symmetric-matmul kernels inside Newton-Schulz. torch_npu does not expose those
# exact kernels. The port keeps the same algebra with bf16/torch matmul kernels;
# this is a semantic fallback documented in source_audit.md.

def ns_line_1(A: torch.Tensor, out: torch.Tensor):
    """Compute C = A @ A.mT using NPU-supported torch matmul."""
    torch.matmul(A, A.mT, out=out)
    return out

def ns_line_2(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """Compute C = alpha * A @ A + beta * A for symmetric A."""
    if A.ndim > 2:
        torch.baddbmm(A, A, A, beta=beta, alpha=alpha, out=out)
    else:
        torch.addmm(A, A, A, beta=beta, alpha=alpha, out=out)
    return out

def newton_schulz_npu(G: torch.Tensor):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Allocate buffers. Shapes match the Triton implementation exactly.
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    for _ in range(5):
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = newton_schulz_npu(grad)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_gather_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

_RMS_GAMMA_CACHE: dict[tuple[str, torch.dtype, int], Tensor] = {}

def norm(x: Tensor):
    key = (str(x.device), x.dtype, x.size(-1))
    gamma = _RMS_GAMMA_CACHE.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        _RMS_GAMMA_CACHE[key] = gamma
    return torch_npu.npu_rms_norm(x, gamma, torch.finfo(x.dtype).eps)[0]

def npu_linear_nd(x: Tensor, weight: Tensor):
    output_shape = (*x.shape[:-1], weight.size(0))
    return torch_npu.npu_linear(x.view(-1, x.size(-1)), weight).view(output_shape)

@torch.compile(backend="npu", dynamic=False)
def compiled_train_softcap(logits: Tensor, scale: float):
    return 30 * torch.sigmoid(logits / scale)

@torch.compile(backend="npu", dynamic=False)
def compiled_relu_squared(x: Tensor):
    x = F.relu(x)
    return x * x

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        # NPU fallback: torch_npu does not expose CUDA _scaled_mm FP8 training.
        return npu_linear_nd(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        r1 = torch.cat((cos, cos), dim=-1).type_as(x_BTHD)
        r2 = -torch.cat((sin, sin), dim=-1).type_as(x_BTHD)
        return torch_npu.npu_rotary_mul(x_BTHD, r1, r2, "half")

class CausalSelfAttention(nn.Module):
    _causal_mask_cache: dict[tuple[str, torch.dtype], Tensor] = {}

    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        assert hdim == dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        # Preserve Record 27 merged QKVO parameterization.
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        with torch.no_grad():
            self.qkvo_w[:3].uniform_(-bound, bound)
            self.qkvo_w[3].zero_()
        self.rotary = Rotary(head_dim, max_seq_len)
        self.qk_norm_gamma = nn.Buffer(torch.ones(head_dim, dtype=torch.bfloat16), persistent=False)
        self.attn_scale = 0.12

    @classmethod
    def compressed_causal_mask(cls, device: torch.device, dtype: torch.dtype = torch.bool) -> Tensor:
        # npu_fusion_attention sparse modes use a compressed 2048x2048 causal mask.
        key = (str(device), dtype)
        if key not in cls._causal_mask_cache:
            cls._causal_mask_cache[key] = torch.triu(
                torch.ones((2048, 2048), device=device, dtype=torch.bool), diagonal=1
            ).to(dtype)
        return cls._causal_mask_cache[key]

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, actual_seq_qlen: list[int], window_num_blocks: int):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for NPU TND attention"
        qkv = npu_linear_nd(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim)
        qk = torch_npu.npu_rms_norm(
            qkv[..., :2 * self.num_heads, :], self.qk_norm_gamma, torch.finfo(x.dtype).eps
        )[0]
        q, k = self.rotary(qk).chunk(2, dim=-2)
        v = qkv[..., 2 * self.num_heads:, :]
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v

        # Record 27's FlexAttention mask is document-causal with block-wise long/short
        # sliding windows. Splitting TND attention by document preserves document
        # boundaries. sparse_mode=4 supplies the causal sliding band; the window is
        # token-granular rather than FlexAttention's block-granular window.
        window_tokens = max(int(window_num_blocks) * 128, 1)
        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.attn_scale,
            atten_mask=self.compressed_causal_mask(x.device),
            sparse_mode=4,
            pre_tockens=max(window_tokens - 1, 0),
            next_tockens=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = npu_linear_nd(y, self.qkvo_w[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make both matrices have the same shape because optimizer sorts params by shape
        # 2 matrices x 12 layers = 24 total, which is divisible by 8 GPU world size
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = npu_linear_nd(x, self.c_fc.T.type_as(x))
        x = compiled_relu_squared(x)
        x = npu_linear_nd(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, actual_seq_qlen: list[int], window_num_blocks: int):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, actual_seq_qlen, window_num_blocks)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        use_fp8 = False  # CUDA FP8 scaled_mm is not available on NPU.
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
            torch.ones(pad),
        ]))
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, actual_seq_qlen: list[int], sliding_window_num_blocks: int):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_blocks = int(sliding_window_num_blocks)
        # NPU sparse_mode=4 backward rejects pre_tockens=0 in the unpadded
        # varlen path, so the NPU fallback clamps the half-window to one block.
        short_blocks = max(long_blocks // 2, 1)
        block_windows = [long_blocks, short_blocks, short_blocks, short_blocks, long_blocks, short_blocks, short_blocks, long_blocks, short_blocks, short_blocks, short_blocks, long_blocks]
        assert len(block_windows) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None])

        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            block = self.blocks[i]
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i], actual_seq_qlen, block_windows[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits = self.lm_head(x)
            logits = compiled_train_softcap(logits, 7.5 * x.size(-1)**0.5)
            return torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), target_seq, reduction="sum"
            )[0]

        # Validation only needs the scalar mean loss. Chunking preserves the
        # result while avoiding a full seq_len x vocab logits allocation on NPU.
        x_2d = x.view(-1, x.size(-1))
        total_loss = 0.0
        chunk_size = 4096
        for start in range(0, x_2d.size(0), chunk_size):
            end = min(start + chunk_size, x_2d.size(0))
            logits = self.lm_head(x_2d[start:end]).float()
            logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
            total_loss += F.cross_entropy(logits, target_seq[start:end], reduction="sum").item()
        return torch.tensor(total_loss / target_seq.numel(), dtype=torch.float32, device=x.device)

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=False) # pin_memory is CUDA-specific
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, seq_len: int, token_window: int):
    boundary_mask = tokens[pos : pos + token_window] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= seq_len:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase token_window if necessary

def distributed_data_generator(filename_pattern: str, seq_len: int, grad_accum_steps: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_size = seq_len * world_size
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        token_window = grad_accum_steps * (2 * batch_size if align_to_bos else batch_size) # provide buffer to handle samples up to length seq_len
        if pos + token_window + 1 >= len(tokens):
            tokens = _load_data_shard(next(file_iter))
            pos = 0
        for _ in range(grad_accum_steps):
            if align_to_bos:
                batch_starts, tokens_consumed = find_batch_starts(tokens, pos, seq_len, token_window)
                start_idx = batch_starts[rank]
            else:
                tokens_consumed = batch_size
                start_idx = pos + rank * seq_len
            buf = tokens[start_idx:][:seq_len + 1]
            # Discover document boundaries while the source buffer is still on
            # CPU.  This is exactly the former model-side expression, but
            # avoids a per-step NPU nonzero followed by blocking D2H + tolist.
            bos = torch.nonzero(buf[:-1] == 50256, as_tuple=False).flatten()
            boundaries = bos[bos > 0].tolist() if bos.numel() else []
            if not boundaries or boundaries[-1] != seq_len:
                boundaries.append(seq_len)
            assert len(boundaries) <= 1000, f"too many document segments for npu_fusion_attention: {len(boundaries)}"
            actual_seq_qlen = [int(x) for x in boundaries]
            inputs = buf[:-1].to(device="npu", dtype=torch.int32, non_blocking=True) # no sync on host side;
            targets = buf[1:].to(device="npu", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
            pos += tokens_consumed
            token_window -= tokens_consumed
            yield inputs, targets, actual_seq_qlen

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # The GPU source uses 8 ranks with 48K train and 256K validation tokens per
    # rank. The benchmark requires 16 ranks, so the NPU port halves per-rank
    # lengths to preserve the global train/validation token counts.
    train_seq_len = 24*1024
    val_seq_len = 2*64*1024
    # optimization
    num_iterations = 1750 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size == 16, "record_027 NPU benchmark must run with NPROC=16"
grad_accum_steps = 1
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group(backend="hccl")
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    run_output_dir = os.environ.get("RUN_OUTPUT_DIR", os.getcwd())
    log_dir = os.path.join(run_output_dir, "logs", str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, "train.log")
    with open(os.path.join(log_dir, "train_gpt.py"), "w") as f:
        f.write(code)
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")
print0(f"Running torch_npu {getattr(torch_npu, '__version__', 'unknown')}")
def npu_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(npu_smi())
print0("="*100)

model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(args.train_seq_len, args.val_seq_len)).npu()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return max(window_size // 128, 1)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792.
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

# torch.compile is skipped on NPU.

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_seq_len, grad_accum_steps, align_to_bos=True)
for _ in range(warmup_steps):
    inputs, targets, actual_seq_qlen = next(train_loader)
    model(inputs, targets, actual_seq_qlen, get_window_size_blocks(1)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_seq_len, grad_accum_steps, align_to_bos=True)
training_time_ms = 0

# Optional, rank-local profiling hook. It is disabled by default so benchmark
# dynamics and timing are unchanged. The profiling ablation enables it only on
# rank 0 and records a short stable-step window after the normal warmup/reset.
profile_ctx = None
if rank == 0 and os.environ.get("RECORD027_PROFILE", "0") == "1":
    from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    profile_dir = os.environ["RECORD027_PROFILE_DIR"]
    os.makedirs(profile_dir, exist_ok=True)
    profile_ctx = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler(
            profile_dir, worker_name="rank0", analyse_flag=True, async_mode=False
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    profile_ctx.start()
# start the clock
torch.npu.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_seq_len, grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, actual_seq_qlen = next(val_loader)
                val_loss += model(inputs, targets, actual_seq_qlen, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, os.path.join(log_dir, f"state_step{step:06d}.pt"))
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for _ in range(grad_accum_steps):
        inputs, targets, actual_seq_qlen = next(train_loader)
        model(inputs, targets, actual_seq_qlen, get_window_size_blocks(step)).backward()
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    if profile_ctx is not None:
        profile_ctx.step()

if profile_ctx is not None:
    profile_ctx.stop()

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
