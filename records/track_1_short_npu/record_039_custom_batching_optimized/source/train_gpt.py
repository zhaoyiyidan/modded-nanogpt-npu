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

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Polar Express Sign Method (pure PyTorch, replaces Triton XXT/ba_plus_cAA kernels)
# Coefficients identical to GPU Record 39.

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


# -----------------------------------------------------------------------------
# Muon optimizer (Record 39 GPU version, adapted for NPU: no Triton, no torch.compile)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        if custom_sizing and dist.get_world_size() in (8, 16):
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)
        self._comm_buffers = {}

    def _get_comm_buffers(self, group, padded_num_params, chunk_size):
        key = id(group)
        buffers = self._comm_buffers.get(key)
        if buffers is None:
            example = group["params"][0]
            kwargs = dict(dtype=example.dtype, device=example.device)
            buffers = {
                "padding_grad": torch.zeros_like(example),
                "stacked_grads": torch.empty((padded_num_params, *example.shape), **kwargs),
                "grad_chunk": torch.empty((chunk_size, *example.shape), **kwargs),
                "stacked_params": torch.empty((padded_num_params, *example.shape), **kwargs),
                "padding_param": torch.zeros((chunk_size, *example.shape), **kwargs),
            }
            buffers["stacked_params"].zero_()
            for i, param in enumerate(group["params"]):
                buffers["stacked_params"][i].copy_(param)
                param.set_(buffers["stacked_params"][i])
            self._comm_buffers[key] = buffers
        return buffers

    def generate_standard_param_groups(self, params):
        params = list(params)
        param_groups = []
        attn_subset = [p for p in params if p.module == 'attn']
        non_attn_subset = [p for p in params if p.module != 'attn']
        param_groups.append(dict(params=attn_subset))
        sizes = {p.shape for p in non_attn_subset}
        for size in sizes:
            group_params = [p for p in non_attn_subset if p.shape == size]
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
        world_size = dist.get_world_size()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
            num_params = len(params)
            padded_num_params = (num_params + world_size - 1) // world_size * world_size
            chunk_size = padded_num_params // world_size
            buffers = self._get_comm_buffers(group, padded_num_params, chunk_size)
            grads_to_stack = [p.grad for p in params]
            if padded_num_params > num_params:
                padding_grad = buffers["padding_grad"]
                grads_to_stack.extend([padding_grad] * (padded_num_params - num_params))
            stacked_grads = buffers["stacked_grads"]
            torch.stack(grads_to_stack, out=stacked_grads)
            grad_chunk = buffers["grad_chunk"]
            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()
            group_infos.append({
                "params": params, "grad_chunk": grad_chunk,
                "reduce_future": reduce_future, "chunk_size": chunk_size,
                "padded_num_params": padded_num_params, "buffers": buffers,
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
                group["lr"]
                * group["weight_decay"]
                * getattr(p_example, "wd_mul", 1.0)
            )

            update_grads_for_zeropower = []

            for i in range(chunk_size):
                param_idx = start_idx + i
                if param_idx >= len(params):
                    update_grads_for_zeropower.append(info["buffers"]["padding_grad"])
                    continue
                p = params[param_idx]
                grad = grad_chunk[i]
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                update_grad = grad.lerp_(momentum_buffer, group["momentum"])
                update_grads_for_zeropower.append(update_grad)
                p.mul_(1 - eff_weight_decay_val)

            if chunk_size == 1:
                batched_update_grads = update_grads_for_zeropower[0].unsqueeze(0)
            else:
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
                params[param_idx].add_(v_chunk[i], alpha=-eff_lr_val)

            stacked_params = info["buffers"]["stacked_params"]
            if chunk_size == 1 and start_idx < len(params):
                gather_input = params[start_idx].unsqueeze(0)
                owner_idx = start_idx
            elif chunk_size == 1:
                gather_input = info["buffers"]["padding_param"]
                owner_idx = None
            else:
                gather_input = info["buffers"]["padding_param"]
                gather_input.zero_()
                valid = min(chunk_size, len(params) - start_idx)
                for i in range(max(0, valid)):
                    gather_input[i].copy_(params[start_idx + i])
                owner_idx = None
            gather_future = dist.all_gather_into_tensor(
                stacked_params, gather_input, async_op=True
            ).get_future()
            all_gather_infos.append({
                "gather_future": gather_future,
            })

        for info in all_gather_infos:
            info["gather_future"].wait()

# -----------------------------------------------------------------------------
# DistAdam optimizer (Record 39 GPU version, adapted for NPU: no torch.compile)

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        self._launch_on_grad_ready = False
        self._early_reductions = {}
        self._grad_slice_buffers = {}
        self._grad_ready_handles = [
            p.register_post_accumulate_grad_hook(self._make_grad_ready_hook(p))
            for group in self.param_groups for p in group["params"]
        ]

    def _make_grad_ready_hook(self, p):
        def launch(_):
            if not self._launch_on_grad_ready:
                return
            world_size = dist.get_world_size()
            rank_size = p.grad.shape[0] // world_size
            grad_slice = self._grad_slice_buffers.get(p)
            if grad_slice is None:
                grad_slice = torch.empty_like(p.grad[:rank_size])
                self._grad_slice_buffers[p] = grad_slice
            future = dist.reduce_scatter_tensor(
                grad_slice, p.grad, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()
            self._early_reductions[p] = (grad_slice, future)
        return launch

    def prepare_backward(self, launch_on_grad_ready: bool):
        if self._early_reductions:
            raise RuntimeError("unconsumed DistAdam early reductions")
        self._launch_on_grad_ready = launch_on_grad_ready

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures = []
        all_gather_futures = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                p = params[base_i]
                early = self._early_reductions.pop(p, None)
                if early is None:
                    grad = p.grad
                    rank_size = grad.shape[0] // world_size
                    grad_slice = self._grad_slice_buffers.get(p)
                    if grad_slice is None:
                        grad_slice = torch.empty_like(grad[:rank_size])
                        self._grad_slice_buffers[p] = grad_slice
                    future = dist.reduce_scatter_tensor(
                        grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future()
                else:
                    grad_slice, future = early
                reduce_scatter_futures.append(future)
                grad_slices.append(grad_slice)

        self._launch_on_grad_ready = False

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            group_step = None
            group_bias_scale = None
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                    state["exp_avg_sq"] = torch.zeros(p_slice.shape, dtype=torch.bfloat16, device=p_slice.device)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group_step is None:
                    loaded_step = state["step"]
                    if torch.is_tensor(loaded_step):
                        loaded_step = int(loaded_step.item())
                    group_step = int(loaded_step) + 1
                    bias1 = 1.0 - beta1 ** group_step
                    bias2 = 1.0 - beta2 ** group_step
                    group_bias_scale = math.sqrt(bias2) / bias1
                state["step"] = group_step
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg.div(denom).mul_(lr * group_bias_scale)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model (Record 39 architecture, NPU attention)

_norm_gamma_cache = {}


def norm(x: Tensor):
    key = (x.size(-1), x.device.index, x.dtype)
    gamma = _norm_gamma_cache.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        _norm_gamma_cache[key] = gamma
    return torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)[0]


def softcap(logits: Tensor):
    return 30 * torch.sigmoid(logits / 7.5)


compiled_softcap = torch.compile(softcap, backend="npu", dynamic=False)


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

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
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
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
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


compiled_rotary = torch.compile(rotary, backend="npu", dynamic=False)


@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    actual_seq_qlen: list[int]
    max_doc_len: int
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim
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

    _shared_mask_cache: dict = {}

    @classmethod
    def _get_window_causal_mask(cls, size: int, window: int, device: torch.device) -> Tensor:
        key = (size, window)
        if key not in cls._shared_mask_cache:
            row = torch.arange(size, device=device).unsqueeze(1)
            col = torch.arange(size, device=device).unsqueeze(0)
            cls._shared_mask_cache[key] = (col > row) | (col < row - window + 1)
        return cls._shared_mask_cache[key]

    @classmethod
    def clear_mask_cache(cls, keep_sizes: set = None):
        if keep_sizes is None:
            cls._shared_mask_cache.clear()
        else:
            cls._shared_mask_cache = {k: v for k, v in cls._shared_mask_cache.items() if k[0] in keep_sizes}

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)
        assert B == 1
        assert T % 16 == 0
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        actual_seq_qlen = attn_args.actual_seq_qlen
        max_doc_len = attn_args.max_doc_len
        attn_scale, bm_size = attn_args.attn_scale, attn_args.bm_size

        qkvo_w = self.qkvo_w.view(4, self.hdim, self.dim).type_as(x)
        q, k, v = F.linear(x, qkvo_w[:3].flatten(end_dim=1)).view(
            B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = compiled_rotary(q, cos, sin), compiled_rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        attn_mask = self._get_window_causal_mask(args.train_max_seq_len, args.train_max_seq_len, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=attn_scale,
            atten_mask=attn_mask,
            pre_tockens=bm_size - 1,
            next_tockens=0,
            sparse_mode=4,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, qkvo_w[3])
        return y


os.environ.setdefault(
    "TORCH_NPU_COMPILE_CACHE_DIR",
    f"/tmp/record039_mlp_compile_{os.environ.get('LOCAL_RANK', '0')}",
)


def activation_projection(x: Tensor, weight: Tensor):
    x = F.relu(x)
    x = x * x
    return F.linear(x, weight.type_as(x))


compiled_activation_projection = torch.compile(
    activation_projection, backend="npu", dynamic=False
)


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
        return compiled_activation_projection(x, self.c_proj)


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
# The main model (Record 39 architecture: 12 layers, U-net skip connections)

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int,
                 max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.detach().zero_()
        self.smear_gate.weight.module = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        use_fp8 = False
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim ** 0.5) / 448, w_s=2 ** -9,
                                    grad_s=1 / 448)
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

    def forward(self, input_seq: Tensor, target_seq: Tensor, actual_seq_qlen: list[int], ws_short: int, ws_long: int):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm,
                    short_bm, long_bm]
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

        x = norm(x)
        if self.training:
            logits = self.lm_head(x)
            logits = compiled_softcap(logits.view(-1, logits.size(-1)))
            loss, _, _, _ = torch_npu.npu_cross_entropy_loss(
                logits, target_seq, reduction="sum"
            )
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
            loss = x.new_tensor(total_loss / num_tokens)
        return loss


# -----------------------------------------------------------------------------
# Distributed data loader (Record 39 version, NPU device)

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
                    raise StopIteration(f"Insufficient BOS; hit tail of shard.")
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


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1,
                               align_to_bos: bool = True):
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

        actual_seq_qlen = cum_lengths[cum_lengths > 0].tolist()
        if not actual_seq_qlen or actual_seq_qlen[-1] != num_tokens_local:
            actual_seq_qlen.append(num_tokens_local)

        device_buf = buf.to(device="npu", dtype=torch.int32, non_blocking=True)
        new_params = yield (
            device_buf[:-1],
            device_buf[1:].to(dtype=torch.int64),
            actual_seq_qlen
        )

        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps

# -----------------------------------------------------------------------------
# int main (Record 39 hyperparameters and training loop)

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_batch_size: int = 2048 * 16 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    num_iterations: int = 2380
    iteration_extension: int = 40
    cooldown_frac: float = 0.4
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_validate: int = 13
    ws_long_validate: int = 20
    momentum_cd_steps: int = 50


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

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if
                        p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.7, 0.95),
    eps=1e-8,
    weight_decay=0.0,
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


def update_optimizer_params(step, optimizer1, optimizer2):
    for group in optimizer1.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)
    if step < 300:
        frac = step / 300
        momentum = 0.85 + frac * (0.95 - 0.85)
        for group in optimizer2.param_groups:
            group["momentum"] = momentum
    momentum_cd_start = args.num_iterations + args.iteration_extension - args.momentum_cd_steps
    if step > momentum_cd_start:
        frac = (step - momentum_cd_start) / args.momentum_cd_steps
        momentum = 0.95 - frac * (0.95 - 0.85)
        for group in optimizer2.param_groups:
            group["momentum"] = momentum


# No torch.compile on NPU

########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len,
                                          grad_accum_steps=grad_accum_steps)
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
    optimizer1.prepare_backward(True)
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

import gc
gc.collect()

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len,
                                          grad_accum_steps=grad_accum_steps)
training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations + args.iteration_extension
ws_short, ws_long = get_ws(0)
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
                inputs, targets, actual_seq_qlen = next(val_loader)
                val_loss += model(inputs, targets, actual_seq_qlen, ws_short, ws_long)
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
        inputs, targets, actual_seq_qlen = next(train_loader)
        optimizer1.prepare_backward(
            step % 2 == 1 and micro_step == grad_accum_steps - 1
        )
        model(inputs, targets, actual_seq_qlen, ws_short, ws_long).backward()
    update_optimizer_params(step, optimizer1, optimizer2)
    if step % 2 == 0:
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)
    else:
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / (step + 1):.2f}ms",
        console=True)

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()
