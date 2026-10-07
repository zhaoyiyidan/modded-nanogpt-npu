import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

# Stage 2 cumulative winner: use the default NPU caching allocator.
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Polar Express Sign Method (pure PyTorch replaces Triton XXT/ba_plus_cAA kernels)
# https://arxiv.org/pdf/2505.16932
# Coefficients identical to GPU Record 41 source.

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def polar_express(G: Tensor):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()
    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# NorMuon optimizer (Record 41 key trick)

class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, custom_sizing=True, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def generate_standard_param_groups(self, params):
        params = list(params)
        param_groups = []
        attn_subset = [p for p in params if p.label == 'attn']
        non_attn_subset = [p for p in params if p.label != 'attn']
        param_groups.append(dict(params=attn_subset))
        sizes = {p.shape for p in non_attn_subset}
        for size in sizes:
            group_params = [p for p in non_attn_subset if p.shape == size]
            param_groups.append(dict(params=group_params))
        return param_groups

    def generate_custom_param_groups(self, params):
        label_ranks = {'smear_gate': 1, 'attn_gate': 2, 'attn': 3, 'mlp': 4}
        params = list(params)
        params.sort(key=lambda x: label_ranks.get(x.label))
        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            group_params = params[idx:idx + size]
            param_groups.append(dict(params=group_params))
            idx += size
        return param_groups

    def communication_groups(self, world_size):
        """Build transient communication groups without changing state_dict."""
        groups = [(group, list(group["params"])) for group in self.param_groups if group["params"]]
        if world_size != 16:
            return groups
        attn_index = next((i for i, (_, ps) in enumerate(groups)
                           if len(ps) == 10 and all(getattr(p, "label", None) == "attn" for p in ps)), None)
        if attn_index is None:
            return groups
        attn_group, attn_params = groups[attn_index]
        matrix_index = next((i for i, (_, ps) in enumerate(groups)
                             if i != attn_index and len(ps) == 22
                             and all(tuple(p.shape) == tuple(attn_params[0].shape) for p in ps)), None)
        if matrix_index is None:
            return groups
        matrix_group, matrix_params = groups[matrix_index]
        keys = ("lr", "momentum", "beta2", "weight_decay")
        if any(attn_group[key] != matrix_group[key] for key in keys):
            return groups
        merged = (attn_group, attn_params + matrix_params)
        return [merged] + [entry for i, entry in enumerate(groups) if i not in (attn_index, matrix_index)]

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group_infos = []
        for group, params in self.communication_groups(world_size):
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
            group_infos.append(dict(
                group=group, params=params, grad_chunk=grad_chunk, reduce_future=reduce_future,
                chunk_size=chunk_size, padded_num_params=padded_num_params,
            ))

        all_gather_infos = []
        for info in group_infos:
            group = info["group"]
            info["reduce_future"].wait()
            params = info["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = info["chunk_size"]
            start_idx = rank * chunk_size
            p_example = params[start_idx] if start_idx < len(params) else params[0]
            eff_lr_val = (
                group["lr"]
                * max(1, p_example.size(-2) / p_example.size(-1)) ** 0.5
                * getattr(p_example, "lr_mul", 1.0)
            )
            eff_weight_decay_val = (
                group["lr"] * group["weight_decay"] * getattr(p_example, "wd_mul", 1.0)
            )
            updated_param_chunk = torch.empty(
                (chunk_size, *p_example.shape), dtype=p_example.dtype, device=p_example.device,
            )
            update_grads_for_zeropower = []
            second_momentum_buffer = None

            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):
                    updated_param_chunk[i].zero_()
                    update_grads_for_zeropower.append(torch.zeros_like(p_example.grad))
                    if i == 0:
                        second_momentum_buffer = (
                            torch.zeros((chunk_size, *p_example.shape[:-1], 1), dtype=torch.float32, device=p_example.device)
                            if p_example.size(-2) >= p_example.size(-1) else
                            torch.zeros((chunk_size, *p_example.shape[:-3], 1, p_example.shape[-1]), device=p_example.device, dtype=torch.float32)
                        )
                    continue
                param = params[param_idx]
                grad = grad_chunk[i]
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    if i == 0:
                        state["second_momentum_buffer"] = (
                            torch.zeros((chunk_size, *grad.shape[:-1], 1), dtype=torch.float32, device=grad.device)
                            if param.size(-2) >= param.size(-1) else
                            torch.zeros((chunk_size, *grad.shape[:-3], 1, grad.shape[-1]), dtype=torch.float32, device=grad.device)
                        )
                momentum_buffer = state["momentum_buffer"]
                if i == 0:
                    second_momentum_buffer = state["second_momentum_buffer"]
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                update_grad = grad.lerp(momentum_buffer, group["momentum"])
                update_grads_for_zeropower.append(update_grad)
                updated_param_chunk[i].copy_(param)
                updated_param_chunk[i].mul_(1 - eff_weight_decay_val)

            batched_update_grads = torch.stack(update_grads_for_zeropower)
            original_shape = batched_update_grads.shape
            param_idx = start_idx if start_idx < len(params) else 0
            if getattr(params[param_idx], 'label', None) == 'attn':
                batch = 4 * original_shape[0]
                d1 = original_shape[1]
                d2 = original_shape[2] // 4
                batched = batched_update_grads.view(batch, d1, d2)
                v_chunk = polar_express(batched)
                v_chunk = v_chunk.view(original_shape)
            else:
                v_chunk = polar_express(batched_update_grads)

            vnorm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = (torch.mean(v_chunk * v_chunk, dim=-1, keepdim=True)
                      if v_chunk.size(-2) >= v_chunk.size(-1)
                      else torch.mean(v_chunk * v_chunk, dim=-2, keepdim=True))
            second_momentum_buffer.lerp_(v_mean.float(), 1 - group["beta2"])
            step_size = 1.0 / second_momentum_buffer.sqrt().clamp_min(1e-10)
            v_chunk.mul_(step_size)
            vnorm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(vnorm / vnorm_new.clamp_min(1e-10))

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
            all_gather_infos.append(dict(
                gather_future=gather_future, stacked_params=stacked_params, orig_params=params,
            ))

        for info in all_gather_infos:
            info["gather_future"].wait()
            unstacked_params = torch.unbind(info["stacked_params"])
            for i, p in enumerate(info["orig_params"]):
                p.copy_(unstacked_params[i], non_blocking=True)

# -----------------------------------------------------------------------------
# DistAdam (Record 41: no CautiousWD, weight_decay=0.0)
# Simplified for 16-NPU: uses all_reduce instead of reduce_scatter to avoid
# param-size divisibility requirements.

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
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
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.bfloat16)
                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    param.mul_(1 - eff_weight_decay)
                torch_npu.npu_apply_adam(
                    beta1 ** t, beta2 ** t, lr, beta1, beta2, eps, grad,
                    False, False, out=(param, exp_avg, exp_avg_sq)
                )

# -----------------------------------------------------------------------------
# Model definitions

_RMS_GAMMA_CACHE = {}


def norm(x: Tensor):
    key = (x.device.index, x.dtype, x.size(-1))
    gamma = _RMS_GAMMA_CACHE.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        _RMS_GAMMA_CACHE[key] = gamma
    return torch_npu.npu_rms_norm(
        x, gamma, epsilon=1.1920928955078125e-7
    )[0]


@torch.compile(backend="npu", dynamic=False)
def compiled_softcap(x: Tensor):
    return 30 * torch.sigmoid(x / 7.5)


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8

    def reset_parameters(self):
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4, dtype=torch.float32, device=device)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window, new_window, alpha=1, beta=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = cos[None, :x_BTHD.size(-3), None, :], sin[None, :x_BTHD.size(-3), None, :]
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    actual_seq_qlen: list
    max_doc_len: int
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

class CausalSelfAttention(nn.Module):
    _shared_mask_cache: dict = {}

    def __init__(self, dim, head_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        self.qkvo_w.label = 'attn'
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = 'attn_gate'
        self.attn_gate.weight.detach().zero_()
        self.qk_norm_gamma = nn.Buffer(
            torch.ones(head_dim, dtype=torch.bfloat16), persistent=False
        )

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
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        actual_seq_qlen = attn_args.actual_seq_qlen
        max_doc_len = attn_args.max_doc_len
        attn_scale, bm_size = attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        qk = torch.cat((q, k), dim=-2)
        qk = torch_npu.npu_rms_norm(
            qk, self.qk_norm_gamma, epsilon=1.1920928955078125e-7
        )[0]
        cos_half = cos[None, :T, None, :]
        sin_half = sin[None, :T, None, :]
        cos_full = torch.cat((cos_half, cos_half), dim=-1)
        sin_full = torch.cat((-sin_half, -sin_half), dim=-1)
        qk = torch_npu.npu_rotary_mul(qk, cos_full, sin_full, rotary_mode="half")
        q, k = qk.chunk(2, dim=-2)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        attn_mask = self._get_window_causal_mask(max_doc_len, bm_size, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=attn_scale,
            atten_mask=attn_mask,
            pre_tockens=bm_size,
            next_tockens=0,
            sparse_mode=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y

@torch.compile(backend="npu", dynamic=False)
def compiled_mlp(x: Tensor, c_fc: Tensor, c_proj: Tensor):
    x = F.linear(x, c_fc.T)
    x = F.relu(x)
    x = x * x
    return F.linear(x, c_proj)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_proj.lr_mul = 2.
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        return compiled_mlp(x, self.c_fc, self.c_proj)

class Block(nn.Module):
    def __init__(self, dim, head_dim, num_heads, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# Main GPT model

def next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim, max_seq_len):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.detach().zero_()
        self.smear_gate.weight.label = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.detach().zero_()
        num_layers_val = num_layers
        pad = (-num_layers_val * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat([
                -1.5 * torch.ones(num_layers_val),
                *[torch.tensor([1.0, 0.0]) for _ in range(num_layers_val)],
                *[torch.tensor([0.5, 0.5]) for _ in range(num_layers_val)],
                torch.zeros(1),
                0.5 * torch.ones(1),
                torch.ones(pad),
            ])
        )
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim

    def forward(self, input_seq, target_seq, actual_seq_qlen, ws_short, ws_long):
        assert input_seq.ndim == 1
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq)
        smear_lambda = self.scalars[5 * len(self.blocks)]
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks):3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks):5 * len(self.blocks)].view(-1, 2)
        backout_lambda = self.scalars[5 * len(self.blocks) + 1]

        n = len(self.blocks) // 2
        x_backout = None
        backout_layer = 8
        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))

        for i in range(1, len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], actual_seq_qlen=actual_seq_qlen,
                max_doc_len=max_doc_len,
                bm_size=bm_sizes[i], cos=self.yarn.cos, sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            if i >= n and i < 11:
                gate = torch.sigmoid(skip_weights[i - n])
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        if self.training:
            logits = self.lm_head(x)
            logits = compiled_softcap(logits)
            loss = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), target_seq, reduction="sum"
            )[0]
        else:
            chunk_size = 4096
            x_2d = x.view(-1, x.size(-1))
            total_loss = 0.0
            num_tokens = x_2d.size(0)
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                logits_chunk = self.lm_head(x_2d[start:end])
                logits_chunk = 30 * torch.sigmoid(logits_chunk / 7.5)
                logits_chunk = logits_chunk.float()
                total_loss += F.cross_entropy(logits_chunk, target_seq[start:end], reduction="sum").item()
            loss = torch.tensor(total_loss / num_tokens, dtype=torch.float32, device=x.device)
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader (adapted from GPU source, pin_memory removed for NPU)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

BOS_ID = 50256

class BOSFinder:
    def __init__(self, tokens, world_size=1, quickload=False):
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local, max_seq_len):
        if self.quickload and self.batch_iter == 5:
            self.get()
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
                    cur + num_tokens_local - cur_len + 1
                )
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends

class DataPreloader:
    def __init__(self, file_iter, world_size=1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def _distributed_cpu_data_generator(filename_pattern, num_tokens, max_seq_len, grad_accum_steps=1, align_to_bos=True):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0
    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)
        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                tokens, finder = preloader.get()
                preloader.start()
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
        actual_seq_qlen = [int(v) for v in cum_lengths.tolist() if int(v) > 0]
        if not actual_seq_qlen or actual_seq_qlen[-1] != num_tokens_local:
            actual_seq_qlen.append(num_tokens_local)
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        new_params = yield (_inputs, _targets, actual_seq_qlen)
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len


class NpuStreamPrefetcher:
    """Keep one batch's H2D copies queued ahead of the compute stream."""

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.npu.Stream()
        self.next_batch = None
        self._preload()

    def _preload(self):
        inputs, targets, actual_seq_qlen = next(self.loader)
        with torch.npu.stream(self.stream):
            inputs = inputs.to(device="npu", non_blocking=True)
            targets = targets.to(device="npu", non_blocking=True)
        self.next_batch = (inputs, targets, actual_seq_qlen)

    def __iter__(self):
        return self

    def __next__(self):
        current_stream = torch.npu.current_stream()
        current_stream.wait_stream(self.stream)
        inputs, targets, actual_seq_qlen = self.next_batch
        inputs.record_stream(current_stream)
        targets.record_stream(current_stream)
        self._preload()
        return inputs, targets, actual_seq_qlen


def distributed_data_generator(filename_pattern, num_tokens, max_seq_len, grad_accum_steps=1, align_to_bos=True):
    cpu_loader = _distributed_cpu_data_generator(
        filename_pattern,
        num_tokens,
        max_seq_len,
        grad_accum_steps=grad_accum_steps,
        align_to_bos=align_to_bos,
    )
    return NpuStreamPrefetcher(cpu_loader)

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_batch_size: int = 2048 * 16 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    num_scheduled_iterations: int = 2275
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.45
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13
    ws_validate_post_yarn_ext: int = 20

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
master_process = (rank == 0)

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
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)
).npu()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.06, momentum=0.95, weight_decay=0.0, beta2=0.95)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step):
    x = min(0.9999, step / args.num_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step):
    if step == args.num_iterations:
        return args.ws_validate // 2, args.ws_validate
    x = min(step / (1 + args.num_scheduled_iterations), 0.9999)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_muon_momentum(step, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

def step_optimizers(step, optimizers, model):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    momentum = get_muon_momentum(step)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum
    if step % 2 == 0:
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)

########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
model.train()
model.yarn.reset()
ws_long = args.ws_schedule[0]

for step in range(warmup_steps):
    inputs, targets, actual_seq_qlen = next(train_loader)
    new_ws_long = args.ws_schedule[step % len(args.ws_schedule)]
    if new_ws_long > ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    elif new_ws_long < ws_long:
        model.yarn.reset()
        ws_long = new_ws_long
    model(inputs, targets, actual_seq_qlen, ws_long // 2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

model.yarn.reset()
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

CausalSelfAttention.clear_mask_cache()
torch.npu.empty_cache()

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations
ws_short, ws_long = get_ws(0)
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        torch.npu.empty_cache()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, actual_seq_qlen = next(val_loader)
                val_loss += model(inputs, targets, actual_seq_qlen, ws_short, ws_long)
        val_loss /= val_steps
        del val_loader
        CausalSelfAttention.clear_mask_cache(keep_sizes={args.train_max_seq_len})
        torch.npu.empty_cache()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, f"{run_dir}/state_step{step:06d}.pt")
        break

    for _ in range(grad_accum_steps):
        inputs, targets, actual_seq_qlen = next(train_loader)
        model(inputs, targets, actual_seq_qlen, ws_short, ws_long).backward()
    step_optimizers(step, optimizers, model)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
