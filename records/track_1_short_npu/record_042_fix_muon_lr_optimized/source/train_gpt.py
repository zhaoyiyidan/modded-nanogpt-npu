import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import copy
import glob
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCH_NPU_COMPILE_CACHE_DIR", f"/tmp/npu_compile_cache_{os.environ.get('LOCAL_RANK', '0')}")
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


def forward_weight(param: Tensor, x: Tensor):
    shadow = getattr(param, "_bf16_shadow", None)
    return shadow if shadow is not None else param.type_as(x)

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
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        params = list(params)
        for param in params:
            if param.dtype == torch.float32:
                param._bf16_shadow = param.detach().bfloat16().requires_grad_(True)
        if custom_sizing and dist.get_world_size() in (8, 16):
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()
            for param in group["params"]:
                shadow = getattr(param, "_bf16_shadow", None)
                if shadow is not None:
                    shadow.copy_(param.bfloat16())
                    shadow.grad = None

    @torch.no_grad()
    def materialize_master_params(self):
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            if not params or getattr(params[0], "_bf16_shadow", None) is None:
                continue
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size
            start_idx = rank * chunk_size
            num_params = min(chunk_size, max(0, len(params) - start_idx))
            local = torch.zeros(
                (chunk_size, *params[0].shape), dtype=params[0].dtype,
                device=params[0].device,
            )
            for i in range(num_params):
                local[i].copy_(params[start_idx + i])
            stacked = torch.empty(
                (padded_num_params, *params[0].shape), dtype=params[0].dtype,
                device=params[0].device,
            )
            dist.all_gather_into_tensor(stacked, local)
            for param, value in zip(params, torch.unbind(stacked)):
                param.copy_(value)

    def generate_standard_param_groups(self, params):
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)
        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
        return param_groups

    def generate_custom_param_groups(self, params):
        module_group_order = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        params_list = list(params)
        params_list.sort(key=lambda x: module_group_order.index(x.label))
        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size
        return param_groups

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size
            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype, device=params[0].device
            )
            for i, p in enumerate(params):
                shadow = getattr(p, "_bf16_shadow", None)
                grad = shadow.grad.float() if shadow is not None else p.grad
                stacked_grads[i].copy_(grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params):].zero_()
            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()
            group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))

        all_gather_infos = []
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()
            params = group["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size
            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0
            num_params = min(chunk_size, max(0, len(params) - start_idx))

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape
            if params[module_idx].label == 'attn':
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr" not in group:
                group["param_lr"] = (
                    max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                    * ref_param.new_tensor(
                        [getattr(param, "lr_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                    ).view(-1, 1, 1)
                )
                group["param_wd"] = ref_param.new_tensor(
                    [getattr(param, "wd_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                ).view(-1, 1, 1)

            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["weight_decay"] * group["param_wd"]

            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads)

            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
            second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

            v_chunk = v_chunk.view(grad_shape)

            updated_params = torch.empty_like(grad_chunk)
            param_chunk = torch.stack(params[module_idx:module_idx + num_params]) if num_params > 0 else torch.zeros_like(v_chunk)
            param_chunk.mul_(1 - eff_wd)
            param_chunk.add_(-eff_lr * v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            has_shadow = getattr(ref_param, "_bf16_shadow", None) is not None
            if has_shadow:
                for i, p in enumerate(params[module_idx:module_idx + num_params]):
                    p.copy_(param_chunk[i])
                gathered_chunk = updated_params.bfloat16()
            else:
                gathered_chunk = updated_params

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=gathered_chunk.dtype, device=gathered_chunk.device,
            )
            gather_future = dist.all_gather_into_tensor(
                stacked_params, gathered_chunk, async_op=True
            ).get_future()
            all_gather_infos.append({
                "gather_future": gather_future,
                "stacked_params": stacked_params,
                "orig_params": params,
                "has_shadow": has_shadow,
            })

        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]
            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                target = p._bf16_shadow if info["has_shadow"] else p
                target.copy_(unstacked_params[i], non_blocking=True)
                if info["has_shadow"]:
                    target.grad = None

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p[0].device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        reduce_scatter_futures = []
        all_gather_futures = []
        grad_slices = []
        for group in self.param_groups:
            params = group["params"]
            for param in params:
                grad = param.grad
                rank_size = grad.shape[0] // self.world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for param in params:
                reduce_scatter_futures[idx].wait()
                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                g_slice = grad_slices[idx]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

def norm(x: Tensor):
    return torch_npu.npu_rms_norm(
        x, rms_norm_weights[x.size(-1)], epsilon=1.1920928955078125e-7
    )[0]

@torch.compile(backend="npu", dynamic=False)
def compiled_softcap(logits: Tensor):
    return 30 * torch.sigmoid(logits / 7.5)


@torch.compile(backend="npu", dynamic=False)
def compiled_relu_square(x: Tensor):
    x = F.relu(x)
    return x * x


def compiled_training_loss(logits: Tensor, target_seq: Tensor):
    logits = compiled_softcap(logits)
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(
        logits.view(-1, logits.size(-1)), target_seq, reduction="sum"
    )
    return loss.sum()

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
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

def rotary(x_BTHD, cos, sin):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = cos[None, :x_BTHD.size(-3), None, :], sin[None, :x_BTHD.size(-3), None, :]
    r1 = torch.cat((cos, cos), dim=-1)
    r2 = -torch.cat((sin, sin), dim=-1)
    return torch_npu.npu_rotary_mul(x_BTHD, r1, r2)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    actual_seq_qlen: list[int]
    max_doc_len: int

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

    def forward(self, x, attn_args):
        B, T = x.size(0), x.size(1)
        assert B == 1
        assert T % 16 == 0
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        qkvo_w = forward_weight(self.qkvo_w, x).view(4, self.hdim, self.dim)
        q, k, v = F.linear(x, qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        actual_seq_qlen = attn_args.actual_seq_qlen
        max_doc_len = attn_args.max_doc_len
        attn_mask = self._get_window_causal_mask(max_doc_len, bm_size, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=attn_scale,
            atten_mask=attn_mask,
            sparse_mode=0,
            pre_tockens=bm_size,
            next_tockens=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, qkvo_w[3])
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_fc.lr_mul = 2.
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x):
        x = F.linear(x, forward_weight(self.c_fc, x).T)
        x = compiled_relu_square(x)
        x = F.linear(x, forward_weight(self.c_proj, x))
        return x

class Block(nn.Module):
    def __init__(self, dim, head_dim, num_heads, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x, x0, lambdas, attn_args):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

def next_multiple_of_n(v, *, n):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim, max_seq_len):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat([
                -1.5 * torch.ones(num_layers),
                *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
                *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
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

    def forward(self, input_seq, target_seq, seqlens, ws_short, ws_long):
        assert input_seq.ndim == 1
        num_layers = len(self.blocks)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [None, ve[1], ve[2]] + [None] * (num_layers - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == num_layers

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == num_layers

        x = self.embed(input_seq)
        skip_weights = self.scalars[:num_layers // 2]
        lambdas = self.scalars[1 * num_layers: 3 * num_layers].view(-1, 2)
        sa_lambdas = self.scalars[3 * num_layers: 5 * num_layers].view(-1, 2)
        smear_lambda = self.scalars[5 * num_layers]
        backout_lambda = self.scalars[5 * num_layers + 1]

        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        actual_seq_qlen = CausalSelfAttention._extract_actual_seqlens(seqlens, input_seq.numel())
        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))

        skip_connections = []
        n = num_layers // 2
        x_backout = None
        backout_layer = 8
        for i in range(1, num_layers):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
                bm_size=bm_sizes[i], cos=self.yarn.cos, sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale,
                actual_seq_qlen=actual_seq_qlen, max_doc_len=max_doc_len
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
            loss = compiled_training_loss(logits, target_seq)
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

def _load_data_shard(file):
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
                    raise StopIteration("Insufficient BOS; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len, cur + num_tokens_local - cur_len + 1)
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
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local,)
            _targets = buf[1:].view(num_tokens_local,)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        new_params = yield (
            _inputs.to(device="npu", non_blocking=True),
            _targets.to(device="npu", non_blocking=True),
            _cum_lengths.to(device="npu", non_blocking=True)
        )
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_batch_size: int = 2048 * 16 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    num_scheduled_iterations: int = 2245
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.50
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13
    ws_validate_post_yarn_ext: int = 20

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
grad_accum_steps = max(1, 8 // world_size)
print(f"[rank {rank}] Starting init, world_size={world_size}, grad_accum={grad_accum_steps}", flush=True)
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
rms_norm_weights = {
    dim: torch.ones(dim, dtype=torch.bfloat16, device=device)
    for dim in (128, 768)
}
print(f"[rank {rank}] Device set to {device}, calling init_process_group", flush=True)
dist.init_process_group(backend="hccl")
print(f"[rank {rank}] init_process_group done, calling barrier", flush=True)
dist.barrier()
print(f"[rank {rank}] barrier done", flush=True)
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
        f.write(f"cwd: {os.getcwd()}\nargv: {' '.join(sys.argv)}\nworld_size: {world_size}\n")
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

print(f"[rank {rank}] Creating model...", flush=True)
model = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)
).npu()
print(f"[rank {rank}] Model created, max_seq_len={max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size)}", flush=True)
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
    lr=0.008, betas=(0.65, 0.95), eps=1e-8, weight_decay=0.0,
)
optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.03, momentum=0.95, beta2=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step):
    x = min(0.9999, step / args.num_scheduled_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step):
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
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
ws_schedule = list(args.ws_schedule) + [args.ws_final]
ws_long = ws_schedule[0]
model.train()
model.yarn.reset()

for ws_idx in range(len(ws_schedule)):
    if ws_idx > 0:
        new_ws_long = ws_schedule[ws_idx]
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    for step in range(warmup_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        model(inputs, targets, cum_seqlens, ws_long // 2, ws_long).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

model.yarn.reset()
model.load_state_dict(initial_state["model"])
optimizer2.reset()
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

CausalSelfAttention.clear_mask_cache()
torch.npu.empty_cache()

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
import gc
gc.collect()

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
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
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
        if args.save_checkpoint:
            optimizer2.materialize_master_params()
            if master_process:
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                torch.save(log, f"{run_dir}/state_step{step:06d}.pt")
        break

    for _ in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        model(inputs, targets, cum_seqlens, ws_short, ws_long).backward()
    step_optimizers(step, optimizers, model)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
