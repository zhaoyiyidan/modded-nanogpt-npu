import os
import sys

os.environ.setdefault("TORCH_NPU_COMPILE_CACHE_DIR", f"/tmp/npu_compile_cache_{os.environ.get('LOCAL_RANK', '0')}")

with open(sys.argv[0]) as f:
    code = f.read()
import copy
import glob
import math
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Pure PyTorch Newton-Schulz (replaces Triton kernels from GPU source)
# Coefficients identical to Record 32 GPU source: a=3.4445, b=-4.7750, c=2.0315

def newton_schulz(G: Tensor):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = X.contiguous()

    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(backend="npu", dynamic=False)
def compiled_adam_shard_update(
    p: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    grad: Tensor,
    step_size: Tensor,
    beta1: float,
    beta2: float,
    eps: float,
):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)
    update = exp_avg.div(denom).mul_(step_size)
    p.add_(other=update, alpha=-1.0)

# -----------------------------------------------------------------------------
# Muon optimizer (Record 32 version - NOT NorMuon)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
            num_params = len(params)
            padded_num_params = (num_params + world_size - 1) // world_size * world_size
            grads_to_stack = [p.grad for p in params]
            if padded_num_params > num_params:
                padding_grad = torch.zeros_like(params[0].grad)
                grads_to_stack.extend([padding_grad] * (padded_num_params - num_params))
            stacked_grads = torch.stack(grads_to_stack)
            chunk_size = padded_num_params // world_size
            grad_chunk = torch.empty(
                (chunk_size, *params[0].grad.shape),
                dtype=stacked_grads.dtype, device=stacked_grads.device,
            )
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()
            group_infos.append({
                "params": params, "grad_chunk": grad_chunk,
                "reduce_future": reduce_future, "chunk_size": chunk_size,
                "padded_num_params": padded_num_params,
            })

        all_gather_infos = []
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()
            params = info["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = info["chunk_size"]
            start_idx = rank * chunk_size

            p_example = params[0]
            eff_lr_val = (
                group["lr"]
                * max(1, p_example.size(-2) / p_example.size(-1)) ** 0.5
                * getattr(p_example, "lr_mul", 1.0)
            )
            eff_weight_decay_val = (
                group["lr"] * group["weight_decay"]
                * getattr(p_example, "wd_mul", 1.0)
            )

            updated_param_chunk = torch.empty(
                (chunk_size, *params[0].shape),
                dtype=params[0].dtype, device=params[0].device,
            )
            update_grads_for_zeropower = []

            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):
                    updated_param_chunk[i].zero_()
                    update_grads_for_zeropower.append(torch.zeros_like(params[0].grad))
                    continue
                p = params[param_idx]
                grad = grad_chunk[i]
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                update_grad = grad.lerp(momentum_buffer, group["momentum"])
                update_grads_for_zeropower.append(update_grad)
                updated_param_chunk[i].copy_(p)
                updated_param_chunk[i].mul_(1 - eff_weight_decay_val)

            batched_update_grads = torch.stack(update_grads_for_zeropower)
            original_shape = batched_update_grads.shape
            if batched_update_grads.ndim > 3:
                assert batched_update_grads.ndim == 4
                batch = original_shape[0] * original_shape[1]
                d1, d2 = original_shape[2], original_shape[3]
                batched = batched_update_grads.view(batch, d1, d2)
                v_chunk = newton_schulz(batched)
                v_chunk = v_chunk.view(original_shape)
            else:
                v_chunk = newton_schulz(batched_update_grads)

            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):
                    continue
                updated_param_chunk[i].add_(v_chunk[i], alpha=-eff_lr_val)

            stacked_params = torch.empty(
                (info["padded_num_params"], *params[0].shape),
                dtype=params[0].dtype, device=params[0].device,
            )
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_param_chunk, async_op=True
            ).get_future()
            all_gather_infos.append({
                "gather_future": gather_future,
                "stacked_params": stacked_params,
                "orig_params": params,
            })

        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]
            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures = []
        all_gather_futures = []
        grad_slices = []
        for group in self.param_groups:
            params = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                )
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            params = group["params"]
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group["lr"] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                if not state:
                    state["step"] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state["exp_avg"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                    state["exp_avg_sq"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                step_size = lr * (torch.sqrt(bias2) / bias1)
                compiled_adam_shard_update(
                    p_slice, exp_avg, exp_avg_sq, g_slice, step_size,
                    beta1, beta2, eps,
                )
                idx += 1
                all_gather_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions

# Only routes with exact bf16 output/input-gradient/weight-gradient equality
# and >=2% wins at both 8192 and 24576-token probe shapes are enabled.
def verified_linear(x: Tensor, weight: Tensor):
    if (
        x.dtype == torch.bfloat16
        and x.shape[-1] == 768
        and weight.shape[0] in (2304, 3072)
        and weight.shape[1] == 768
    ):
        return torch_npu.npu_linear(
            x.reshape(-1, x.shape[-1]), weight
        ).reshape(*x.shape[:-1], weight.shape[0])
    return F.linear(x, weight)

_rms_norm_gamma_cache = {}

def norm(x: Tensor):
    key = (x.size(-1), x.device.index, x.dtype)
    gamma = _rms_norm_gamma_cache.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        _rms_norm_gamma_cache[key] = gamma
    return torch_npu.npu_rms_norm(x, gamma, epsilon=torch.finfo(x.dtype).eps)[0]

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = False

    def reset_parameters(self):
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = cos[None, :x_BTHD.size(-3), None, :], sin[None, :x_BTHD.size(-3), None, :]
    return torch_npu.npu_rotary_mul(x_BTHD, cos, sin)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    rotary_cos: torch.Tensor
    rotary_sin: torch.Tensor
    attn_scale: float
    actual_seq_qlen: list
    max_doc_len: int

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, head_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        assert hdim == dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        with torch.no_grad():
            self.qkvo_w[:3].uniform_(-bound, bound)
            self.qkvo_w[3].zero_()
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.detach().zero_()

    _shared_mask_cache: dict = {}

    @classmethod
    def _get_window_causal_mask(cls, size, window, device):
        key = (size, window)
        if key not in cls._shared_mask_cache:
            row = torch.arange(size, device=device).unsqueeze(1)
            col = torch.arange(size, device=device).unsqueeze(0)
            cls._shared_mask_cache[key] = (col > row) | (col < row - window + 1)
        return cls._shared_mask_cache[key]

    @classmethod
    def clear_mask_cache(cls, keep_sizes=None):
        if keep_sizes is None:
            cls._shared_mask_cache.clear()
        else:
            cls._shared_mask_cache = {k: v for k, v in cls._shared_mask_cache.items() if k[0] in keep_sizes}

    @staticmethod
    def _extract_actual_seqlens(seqlens, total_tokens):
        cum = seqlens[1:]
        real_mask = torch.ones(cum.numel(), dtype=torch.bool, device=cum.device)
        real_mask[1:] = cum[1:] > cum[:-1]
        actual_cum = cum[real_mask]
        actual_cum = actual_cum[actual_cum > 0]
        result = actual_cum.tolist()
        if not result or result[-1] != total_tokens:
            result.append(total_tokens)
        return result

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)
        assert B == 1
        assert T % 16 == 0
        rotary_cos, rotary_sin = attn_args.rotary_cos, attn_args.rotary_sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = (
            verified_linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = norm(q), norm(k)
        q, k = rotary(q, rotary_cos, rotary_sin), rotary(k, rotary_cos, rotary_sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        actual_seq_qlen = attn_args.actual_seq_qlen
        attn_mask = self._get_window_causal_mask(2048, 2048, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=attn_scale,
            atten_mask=attn_mask,
            sparse_mode=4,
            pre_tockens=bm_size - 1,
            next_tockens=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        x = verified_linear(x, self.c_fc.T.type_as(x))
        x = relu_square(x)
        x = F.linear(x, self.c_proj.type_as(x))
        return x

from relu2_backward_tuned import relu_square

from softcap_fused_tuned import softcap_logits
from softcap_fused_bf16_output_candidate import softcap_logits as softcap_logits_bf16

class Block(nn.Module):
    def __init__(self, dim, head_dim, num_heads, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx != 7 else None
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim, max_seq_len):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            -1.5 * torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
            torch.ones(pad),
        ]))
        self.max_seq_len = max_seq_len
        self.setup_yarn(head_dim)
        for param in self.embed.parameters():
            param.lr_mul = 75.0
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def setup_yarn(self, head_dim):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.outer(t, angular_freq)
        rotary_cos = theta.cos().to(torch.bfloat16)
        rotary_sin = theta.sin().to(torch.bfloat16)
        self.rotary_cos = nn.Buffer(torch.cat((rotary_cos, rotary_cos), dim=-1), persistent=False)
        self.rotary_sin = nn.Buffer(-torch.cat((rotary_sin, rotary_sin), dim=-1), persistent=False)
        self.angular_freq = angular_freq
        windows = list(dict.fromkeys(list(args.ws_schedule) + [args.ws_validate]))
        scale_factors = [0.2 * math.log(curr / prev) + 1 for prev, curr in zip(windows[:-1], windows[1:])]
        attn_scales = list(accumulate([0.1] + scale_factors, lambda acc, factor: acc * factor))
        self.attn_scales = dict(zip(windows, attn_scales))

    def apply_yarn(self, old_window, new_window, alpha=1, beta=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        rotary_cos = theta.cos()
        rotary_sin = theta.sin()
        self.rotary_cos.copy_(torch.cat((rotary_cos, rotary_cos), dim=-1))
        self.rotary_sin.copy_(-torch.cat((rotary_sin, rotary_sin), dim=-1))

    def forward(self, input_seq, target_seq, seqlens, ws):
        assert input_seq.ndim == 1
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = ws * args.block_size, (ws // 2) * args.block_size
        bm_sizes = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm,
                     short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        actual_seq_qlen = CausalSelfAttention._extract_actual_seqlens(seqlens, input_seq.numel())
        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))

        x = x0 = norm(self.embed(input_seq)[None]).to(torch.bfloat16)

        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks):3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks):5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2
        for i in range(len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
                bm_size=bm_sizes[i], rotary_cos=self.rotary_cos,
                rotary_sin=self.rotary_sin, attn_scale=self.attn_scales[ws],
                actual_seq_qlen=actual_seq_qlen, max_doc_len=max_doc_len,
            )
            if i >= n:
                gate = torch.sigmoid(skip_weights[i - n])
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        if self.training:
            # Experimental full-run candidate: bf16 softcap math and bf16 output.
            logits = softcap_logits_bf16(self.lm_head(x))
            loss = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), target_seq,
                reduction="sum",
            )[0].reshape(())
        else:
            chunk_size = 4096
            x_2d = x.view(-1, x.size(-1))
            total_loss = 0.0
            num_tokens = x_2d.size(0)
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                logits_chunk = softcap_logits(self.lm_head(x_2d[start:end]))
                total_loss += F.cross_entropy(logits_chunk, target_seq[start:end], reduction="sum").item()
            loss = torch.tensor(total_loss / num_tokens, dtype=torch.float32, device=x.device)
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

BOS_ID = 50256

class BOSFinder:
    def __init__(self, tokens, world_size=1):
        self.size = tokens.numel()
        self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size

    def next_batch(self, num_tokens_local, max_seq_len):
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]
        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(
                    self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                    cur + max_seq_len,
                    cur + num_tokens_local - cur_len + 1,
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

def distributed_data_generator(filename_pattern, num_tokens, max_seq_len, grad_accum_steps=1, align_to_bos=True):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    finder = BOSFinder(tokens, world_size=world_size) if align_to_bos else None
    pos = 0
    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)
        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                tokens = _load_data_shard(next(file_iter))
                finder = BOSFinder(tokens, world_size=world_size)
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local:pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local,)
            _targets = buf[1:].view(num_tokens_local,)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="npu", dtype=torch.int32, non_blocking=True),
            _targets.to(device="npu", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="npu", dtype=torch.int32, non_blocking=True),
        )
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_batch_size: int = 2048 * 24 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    num_iterations: int = 1670
    cooldown_frac: float = 0.5
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 125
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
grad_accum_steps = max(1, 8 // world_size)
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group(backend="hccl")
dist.barrier()
master_process = rank == 0

logfile = None
run_dir = None
if master_process:
    run_id = args.run_id
    run_dir = f"logs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    logfile = f"{run_dir}/train.log"
    src_basename = os.path.basename(sys.argv[0])
    with open(f"{run_dir}/{src_basename}", "w") as f:
        f.write(code)
    with open(f"{run_dir}/command.txt", "w") as f:
        f.write(f"cwd: {os.getcwd()}\n")
        f.write(f"argv: {' '.join(sys.argv)}\n")
        f.write(f"world_size: {world_size}\n")
        for k in ("TORCHELASTIC_RUN_ID", "MASTER_ADDR", "MASTER_PORT", "DATA_PATH"):
            v = os.environ.get(k)
            if v is not None:
                f.write(f"env {k}={v}\n")
    print(run_dir)

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")

def npu_smi():
    import subprocess
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(npu_smi())
print0("=" * 100)

model = GPT(
    vocab_size=50257, num_layers=12, num_heads=6, head_dim=128, model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size),
).npu()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.0,
)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step):
    x = step / args.num_iterations
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step):
    if step == args.num_iterations:
        return args.ws_validate
    x = step / (1 + args.num_iterations)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx]

########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = dict(
    model=copy.deepcopy(model.state_dict()),
    optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
)
train_loader = distributed_data_generator(
    args.train_files, args.train_batch_size, args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps,
)
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    ws = args.ws_schedule[step % len(args.ws_schedule)]
    model(inputs, targets, cum_seqlens, ws).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

CausalSelfAttention.clear_mask_cache()
torch.npu.empty_cache()

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(
    args.train_files, args.train_batch_size, args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps,
)
training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations
ws = get_ws(0)
for step in range(train_steps + 1):
    last_step = step == train_steps
    new_ws = get_ws(step)
    if new_ws != ws:
        model.apply_yarn(ws, new_ws)
        ws = new_ws

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        torch.npu.empty_cache()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(
            args.val_files, args.val_batch_size, -1,
            grad_accum_steps=grad_accum_steps, align_to_bos=False,
        )
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws)
        val_loss /= val_steps
        del val_loader
        CausalSelfAttention.clear_mask_cache(keep_sizes={args.train_max_seq_len})
        torch.npu.empty_cache()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms", console=True)
        model.train()
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, f"{run_dir}/state_step{step:06d}.pt")
        break

    for _ in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        model(inputs, targets, cum_seqlens, ws).backward()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
