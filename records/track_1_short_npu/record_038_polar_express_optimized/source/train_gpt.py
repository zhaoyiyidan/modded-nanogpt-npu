import os
import sys

if os.environ.get("NPU_PROFILE_LEVEL"):
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
    os.environ.setdefault("CPU_AFFINITY_CONF", "1")

os.environ.setdefault(
    "TORCH_NPU_COMPILE_CACHE_DIR",
    f"/tmp/record038_npu_compile_cache_{os.environ.get('LOCAL_RANK', '0')}",
)

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

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Polar Express Sign Method (pure PyTorch, replaces Triton XXT/ba_plus_cAA kernels)
# https://arxiv.org/pdf/2505.16932
# Coefficients identical to GPU version.

coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def polar_express(G: torch.Tensor):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        ws = dist.get_world_size() if dist.is_initialized() else 1
        if custom_sizing and ws == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)
        self.should_sync = False
        self._group_for_param = {
            p: group_idx for group_idx, group in enumerate(self.param_groups) for p in group["params"]
        }
        self._ready_params = [set() for _ in self.param_groups]
        self._reduce_infos = {}
        self._reduce_hooks = [
            p.register_post_accumulate_grad_hook(self._launch_group_sync)
            for group in self.param_groups for p in group["params"]
        ]

    @torch.no_grad()
    def _launch_group_sync(self, p):
        if not self.should_sync:
            return
        group_idx = self._group_for_param[p]
        ready = self._ready_params[group_idx]
        ready.add(p)
        group = self.param_groups[group_idx]
        params = group["params"]
        if len(ready) != len(params):
            return
        world_size = dist.get_world_size()
        num_params = len(params)
        padded_num_params = (num_params + world_size - 1) // world_size * world_size
        grads_to_stack = [item.grad for item in params]
        if padded_num_params > num_params:
            padding_grad = torch.zeros_like(params[0].grad)
            grads_to_stack.extend([padding_grad] * (padded_num_params - num_params))
        stacked_grads = torch.stack(grads_to_stack)
        chunk_size = padded_num_params // world_size
        grad_chunk = torch.empty(
            (chunk_size, *params[0].grad.shape), dtype=stacked_grads.dtype, device=stacked_grads.device
        )
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        self._reduce_infos[group_idx] = {
            "params": params, "grad_chunk": grad_chunk, "reduce_future": future,
            "chunk_size": chunk_size, "padded_num_params": padded_num_params,
        }

    def generate_standard_param_groups(self, params):
        params = list(params)
        param_groups = []
        attn_subset = [p for p in params if p.module == 'attn']
        non_attn_subset = [p for p in params if p.module != 'attn']
        param_groups.append(dict(params=attn_subset))
        sizes = {p.shape for p in non_attn_subset}
        for size in sizes:
            group_params = [p for p in non_attn_subset if p.shape == size]
            # R66: 22 MLP matrices pad to 32 parameters for the 16-rank
            # reduce-scatter. Splitting them 6+16 preserves that exact padded
            # payload (16+16) while allowing the later-layer group, whose
            # gradients become ready first in backward, to launch early.
            if size == torch.Size((768, 3072)) and len(group_params) == 22:
                param_groups.append(dict(params=group_params[6:]))
                param_groups.append(dict(params=group_params[:6]))
            else:
                param_groups.append(dict(params=group_params))
        return param_groups

    def generate_custom_param_groups(self, params):
        module_ranks = {
            'smear_gate': 1,
            'attn_gate': 2,
            'attn': 3,
            'mlp': 4,
        }
        params = list(params)
        params.sort(key=lambda x: module_ranks.get(x.module))
        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            group_params = params[idx:idx + size]
            param_groups.append(dict(params=group_params))
            idx += size
        return param_groups

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        group_infos = []
        for group_idx, group in enumerate(self.param_groups):
            if group["params"]:
                if group_idx not in self._reduce_infos:
                    raise RuntimeError("Muon gradient synchronization was not launched")
                group_infos.append(self._reduce_infos[group_idx])

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
                group["lr"] * group["weight_decay"] * getattr(p_example, "wd_mul", 1.0)
            )
            updated_param_chunk = torch.empty(
                (chunk_size, *p_example.shape), dtype=p_example.dtype, device=p_example.device,
            )
            update_grads_for_zeropower = []
            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):
                    updated_param_chunk[i].zero_()
                    update_grads_for_zeropower.append(torch.zeros_like(p_example.grad))
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
            module_idx = start_idx if start_idx < len(params) else 0
            if getattr(params[module_idx], 'module', 'none') == 'attn':
                for p in params[module_idx:module_idx + chunk_size]:
                    assert getattr(params[module_idx], 'module', 'none') == 'attn'
                batch = 4 * original_shape[0]
                d1 = original_shape[1]
                d2 = original_shape[2] // 4
                batched = batched_update_grads.view(batch, d1, d2)
                v_chunk = polar_express(batched)
                v_chunk = v_chunk.view(original_shape)
            else:
                v_chunk = polar_express(batched_update_grads)

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
        self._reduce_infos.clear()
        for ready in self._ready_params:
            ready.clear()
        self.should_sync = False


class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        self.world_size = dist.get_world_size()
        self.should_sync = False
        self._reduce_scatter_futures = {}
        self._all_gather_futures = []
        self._reduce_scatter_hooks = [
            p.register_post_accumulate_grad_hook(self._launch_grad_sync)
            for group in self.param_groups for p in group["params"]
        ]

    @torch.no_grad()
    def _launch_grad_sync(self, p):
        if not self.should_sync:
            return
        grad = p.grad
        rank_size = grad.shape[0] // self.world_size
        grad_slice = torch.empty_like(grad[:rank_size])
        future = dist.reduce_scatter_tensor(
            grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        self._reduce_scatter_futures[p] = (future, grad_slice)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = self.world_size
        all_gather_futures = []

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                p = params[base]
                if p not in self._reduce_scatter_futures:
                    raise RuntimeError("DistAdam gradient synchronization was not launched")
                reduce_future, g_slice = self._reduce_scatter_futures[p]
                reduce_future.wait()
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                if not state:
                    # The counter is control metadata. Keeping it on the host
                    # avoids two device pow operations, a scalar sqrt, and the
                    # associated AI_CPU casts for every parameter, every step.
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                    state["exp_avg_sq"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (math.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        # Keep the exact collective sequence and payload, but defer the host-side
        # wait so these gathers can overlap the independent Muon shard update.
        self._all_gather_futures = all_gather_futures
        self._reduce_scatter_futures.clear()
        self.should_sync = False

    @torch.no_grad()
    def finish_sync(self):
        if self._all_gather_futures:
            torch.futures.collect_all(self._all_gather_futures).wait()
            self._all_gather_futures.clear()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

_rms_gamma_cache = {}


def norm(x: Tensor):
    key = (x.size(-1), x.dtype, x.device)
    gamma = _rms_gamma_cache.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        _rms_gamma_cache[key] = gamma
    return torch_npu.npu_rms_norm(x, gamma, epsilon=torch.finfo(x.dtype).eps)[0]


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = False  # FP8 not available on NPU

    def reset_parameters(self) -> None:
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
        rotary_cos = theta.cos().to(torch.bfloat16)
        rotary_sin = theta.sin().to(torch.bfloat16)
        self.cos = nn.Buffer(torch.cat((rotary_cos, rotary_cos), dim=-1), persistent=False)
        self.sin = nn.Buffer(-torch.cat((rotary_sin, rotary_sin), dim=-1), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        rotary_cos = theta.cos()
        rotary_sin = theta.sin()
        self.cos.copy_(torch.cat((rotary_cos, rotary_cos), dim=-1))
        self.sin.copy_(-torch.cat((rotary_sin, rotary_sin), dim=-1))
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    return torch_npu.npu_rotary_mul(x_BTHD, cos, sin)


@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    actual_seq_qlen: list


class CausalSelfAttention(nn.Module):
    _shared_mask_cache: dict = {}

    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        self.qkvo_w.module = 'attn'
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.module = 'attn_gate'
        self.attn_gate.weight.detach().zero_()

    @classmethod
    def _get_window_causal_mask(cls, size: int, window: int, dev: torch.device) -> Tensor:
        key = (size, window)
        if key not in cls._shared_mask_cache:
            row = torch.arange(size, device=dev).unsqueeze(1)
            col = torch.arange(size, device=dev).unsqueeze(0)
            cls._shared_mask_cache[key] = (col > row) | (col < row - window + 1)
        return cls._shared_mask_cache[key]

    @classmethod
    def clear_mask_cache(cls, keep_sizes=None):
        if keep_sizes is None:
            cls._shared_mask_cache.clear()
        else:
            cls._shared_mask_cache = {k: v for k, v in cls._shared_mask_cache.items() if k[0] in keep_sizes}

    @staticmethod
    def _extract_actual_seqlens(seqlens: Tensor, total_tokens: int):
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
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(
            B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        actual_seq_qlen = attn_args.actual_seq_qlen
        # sparse_mode=4 consumes a compressed causal mask; the band width is
        # expressed exactly by pre_tockens/next_tockens.
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
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.c_fc.module = 'mlp'
        self.c_proj.module = 'mlp'
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        x = torch.matmul(x, self.c_fc.type_as(x))
        x = relu_square(x)
        x = torch.matmul(x, self.c_proj.T.type_as(x))
        return x


from relu2_backward_tuned_r2 import relu_square


@torch.compile(backend="npu", dynamic=False)
def softcap_logits_train(logits: Tensor):
    """R56: retain BF16 through softcap and fused CE as explicitly requested."""
    return 30 * torch.sigmoid(logits / 7.5)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
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
# The main model

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.detach().zero_()
        self.smear_gate.weight.module = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=False)
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        pad = (-num_layers * 6) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat([
                -1.5 * torch.ones(num_layers),
                *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
                *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
                torch.zeros(num_layers),
                torch.ones(pad),
            ])
        )
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
        actual_seq_qlen = (
            seqlens if isinstance(seqlens, list)
            else CausalSelfAttention._extract_actual_seqlens(seqlens, input_seq.numel())
        )
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
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        n = len(self.blocks) // 2
        for i in range(1, len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
                bm_size=bm_sizes[i], cos=self.yarn.cos, sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale,
                actual_seq_qlen=actual_seq_qlen,
            )
            if i >= n and i < 11:
                gate = torch.sigmoid(skip_weights[i - n])
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
        x = norm(x)
        if self.training:
            x_2d = x.reshape(-1, x.size(-1))
            logits = softcap_logits_train(torch_npu.npu_linear(x_2d, self.lm_head.weight.type_as(x)))
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
                logits_chunk = torch_npu.npu_linear(x_2d[start:end], self.lm_head.weight.type_as(x_2d))
                logits_chunk = 30 * torch.sigmoid(logits_chunk / 7.5)
                total_loss += F.cross_entropy(logits_chunk, target_seq[start:end], reduction="sum").item()
            loss = x.new_tensor(total_loss / num_tokens)
        return loss


# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
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

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
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
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    def __init__(self, file_iter, world_size: int = 1):
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


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
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
        actual_seq_qlen = CausalSelfAttention._extract_actual_seqlens(
            _cum_lengths, num_tokens_local
        )
        new_params = yield (
            _inputs.to(device="npu", dtype=torch.int32, non_blocking=True),
            _targets.to(device="npu", dtype=torch.int64, non_blocking=True),
            actual_seq_qlen,
        )
        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps


_data_copy_stream = None


def prefetch_to_npu_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int,
                                   grad_accum_steps: int = 1, align_to_bos: bool = True):
    """Preserve source order while overlapping the next batch H2D with this batch's compute."""
    source = distributed_data_generator(
        filename_pattern, num_tokens, max_seq_len,
        grad_accum_steps=grad_accum_steps, align_to_bos=align_to_bos,
    )
    global _data_copy_stream
    if _data_copy_stream is None:
        _data_copy_stream = torch.npu.Stream()
    copy_stream = _data_copy_stream
    with torch.npu.stream(copy_stream):
        prefetched = next(source)
    while True:
        compute_stream = torch.npu.current_stream()
        compute_stream.wait_stream(copy_stream)
        current = prefetched
        for value in current:
            if isinstance(value, Tensor):
                value.record_stream(compute_stream)
        with torch.npu.stream(copy_stream):
            prefetched = next(source)
        yield current


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
    num_iterations: int = 1630
    iteration_extension = 40
    cooldown_frac: int = 0.5
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 125
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13
    ws_long_validate: int = 20

args = Hyperparameters()

# Proportional Stage-1 short run. Defaults remain byte-for-byte equivalent to
# the original full task; only the dedicated ablation runner sets this flag.
if os.environ.get("NPU_TUNING_SHORT") == "1":
    args.num_iterations = 163
    args.iteration_extension = 4
    args.val_loss_every = 167

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

prof = None
profile_level = os.environ.get("NPU_PROFILE_LEVEL", "").upper()
if profile_level:
    profile_dir = os.environ["NPU_PROFILE_DIR"]
    rank_profile_dir = os.path.join(profile_dir, f"rank_{rank}")
    os.makedirs(rank_profile_dir, exist_ok=True)
    activities = [torch_npu.profiler.ProfilerActivity.NPU]
    profile_kwargs = {}
    if profile_level == "L1":
        activities.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
        profile_kwargs.update(
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            ),
        )
    elif profile_level != "L0":
        raise ValueError(f"unsupported NPU_PROFILE_LEVEL={profile_level!r}")
    prof = torch_npu.profiler.profile(
        activities=activities,
        schedule=torch_npu.profiler.schedule(wait=52, warmup=1, active=2, repeat=3),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            rank_profile_dir, worker_name=f"rank_{rank}"
        ),
        **profile_kwargs,
    )

logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")

def npu_smi():
    import subprocess
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

print0(npu_smi())
print0("=" * 100)

model: nn.Module = GPT(
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
    lr=0.008, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.0,
)
optimizer2 = Muon(hidden_matrix_params + gate_params, lr=0.06, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step: int):
    x = min(0.9999, step / args.num_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    if step == args.num_iterations + args.iteration_extension:
        return args.ws_validate // 2, args.ws_validate
    x = min(step / (1 + args.num_iterations), 0.9999)
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]


########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
train_loader = prefetch_to_npu_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len,
                                              grad_accum_steps=grad_accum_steps)
ws_long = args.ws_schedule[0]
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    new_ws_long = args.ws_schedule[step % len(args.ws_schedule)]
    if new_ws_long > ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    elif new_ws_long < ws_long:
        model.yarn.reset()
        ws_long = new_ws_long
    optimizer1.should_sync = True
    optimizer2.should_sync = True
    model(inputs, targets, cum_seqlens, ws_long // 2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    optimizer1.finish_sync()
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

train_loader = prefetch_to_npu_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len,
                                              grad_accum_steps=grad_accum_steps)
next_train_batch = next(train_loader)
training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations + args.iteration_extension
ws_short, ws_long = get_ws(0)
if prof is not None:
    prof.start()
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_long_validate
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        torch.npu.empty_cache()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1,
                                                grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = torch.zeros((), device=device, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
        val_loss /= val_steps
        del val_loader
        CausalSelfAttention.clear_mask_cache(keep_sizes={args.train_max_seq_len})
        torch.npu.empty_cache()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
            console=True)
        model.train()
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(),
                       optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    for micro_step in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next_train_batch
        next_train_batch = next(train_loader)
        if micro_step + 1 == grad_accum_steps:
            optimizer1.should_sync = True
            optimizer2.should_sync = True
        model(inputs, targets, cum_seqlens, ws_short, ws_long).backward()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        opt.step()
    optimizer1.finish_sync()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
        console=True)
    if prof is not None:
        prof.step()

if prof is not None:
    prof.stop()

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
