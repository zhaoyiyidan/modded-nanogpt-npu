import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = f"/tmp/r049_stage1_v1_compile_{os.environ.get('RANK', '0')}"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/r049_stage1_v1_inductor_{os.environ.get('RANK', '0')}"
import torch
import torch_npu
import softcap_ce_ext
import unit_rmsnorm_dx_ext
torch.empty(1, device="npu", requires_grad=True).backward()  # prevents a bug on some systems

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Polar Express Sign Method (pure PyTorch, replaces Triton XXT/ba_plus_cAA kernels)
# https://arxiv.org/pdf/2505.16932
# Coefficients and logic identical to GPU version — only Triton kernels replaced with torch matmul.

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def polar_express(G: Tensor, split_baddbmm: bool = False):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()

    for a, b, c in polar_express_coeffs:
        A = X @ X.mT                       # XXT
        B = b * A + c * A @ A              # ba_plus_cAA
        X = a * X + B @ X                  # aX + BX

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# Helpers for NorMuon (no @torch.compile on NPU)

def cautious_wd_and_update_inplace(p, v, wd_tensor, lr_tensor):
    """Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors."""
    mask = ((v * p) >= 0).to(p.dtype)
    wd_factor = wd_tensor.to(p.dtype)
    lr_factor = lr_tensor.to(p.dtype)
    p.copy_(p - (p * mask * wd_factor * lr_factor) - (v * lr_factor))


def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


# -----------------------------------------------------------------------------
# NorMuon optimizer (adapted for NPU: pure PyTorch polar_express replaces Triton version)

class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95,
                 beta2=0.95, custom_sizing=True, pack_large_groups=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        elif pack_large_groups and self.world_size == 16:
            param_groups = self.generate_packed_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)
        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
        return param_groups

    def generate_packed_param_groups(self, params):
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)
        param_groups = []
        # Attention and MLP matrices have the same [3072, 768] shape.  Keep
        # attention first so the 10/22 split lands on a rank boundary when
        # world_size=16 (two matrices per rank), avoiding all padding for the
        # dominant communication payload.
        attention_params = groups.pop('attn', [])
        mlp_params = groups.pop('mlp', [])
        large_params = attention_params + mlp_params
        if large_params:
            assert all(p.shape == large_params[0].shape for p in large_params)
            chunk_size = (len(large_params) + self.world_size - 1) // self.world_size
            assert len(attention_params) % chunk_size == 0
            param_groups.append(dict(params=large_params, chunk_size=chunk_size))
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
                stacked_grads[i].copy_(p.grad, non_blocking=True)
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
            if num_params:
                assert len({p.label for p in params[module_idx:module_idx + num_params]}) == 1

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape
            if params[module_idx].label == 'attn':
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2])

            ref_param = params[module_idx]
            param_shape = ref_param.shape
            is_gate = ref_param.label in ['smear_gate', 'attn_gate']

            if "second_momentum_buffer" not in group:
                if is_gate:
                    group["second_momentum_buffer"] = torch.zeros_like(updated_grads[..., :, :1])
                else:
                    group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1])
                        if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
                    )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr_cpu" not in group:
                lr_mults = []
                wd_mults = []
                for p in params:
                    shape = p.shape
                    if len(shape) >= 2:
                        shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                    else:
                        shape_mult = 1.0
                    lr_mults.append(shape_mult * getattr(p, "lr_mul", 1.0))
                    wd_mults.append(getattr(p, "wd_mul", 1.0))
                group["param_lr_cpu"] = torch.tensor(lr_mults, dtype=torch.float32, device="cpu")
                group["param_wd_cpu"] = torch.tensor(wd_mults, dtype=torch.float32, device="cpu")

            eff_lr_all = group["param_lr_cpu"] * group["lr"]
            eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]
            eff_lr_cpu = eff_lr_all[module_idx:module_idx + num_params]
            eff_wd_cpu = eff_wd_all[module_idx:module_idx + num_params]

            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads, split_baddbmm=(ref_param.label == 'mlp'))

            red_dim = -1 if (is_gate or param_shape[-2] >= param_shape[-1]) else -2

            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum_buffer, group["beta2"], red_dim
            )
            v_chunk = v_chunk.view(grad_shape)

            updated_params = torch.empty_like(grad_chunk)
            if num_params > 0:
                param_chunk = torch.stack(params[module_idx:module_idx + num_params])
                for local_idx in range(num_params):
                    cautious_wd_and_update_inplace(
                        param_chunk[local_idx],
                        v_chunk[local_idx],
                        eff_wd_cpu[local_idx],
                        eff_lr_cpu[local_idx],
                    )
            else:
                param_chunk = torch.zeros_like(v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype, device=updated_params.device,
            )
            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
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
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        label_order = ['lm_head', 'scalars', 'value_embed', 'embed']
        params_by_label = defaultdict(list)
        for p in params:
            params_by_label[getattr(p, 'label', None)].append(p)
        param_groups = []
        for label in label_order:
            if label in params_by_label:
                param_groups.append(dict(params=params_by_label[label]))
        if None in params_by_label:
            param_groups.append(dict(params=params_by_label[None]))
        super().__init__(param_groups, defaults)
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p[0].device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)

        self.should_sync = False
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                hook = param.register_post_accumulate_grad_hook(self._sync_gradient)
                self._reduce_scatter_hooks.append(hook)

    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return
        grad = param.grad
        rank_size = grad.shape[0] // self.world_size
        grad_slice = torch.empty_like(grad[:rank_size])
        self._reduce_scatter_futures[param] = (
            dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future(),
            grad_slice
        )

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for param in group['params']:
                if param not in self._reduce_scatter_futures:
                    continue
                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()
                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                if wd != 0:
                    mask = ((update * p_slice) >= 0).to(update.dtype)
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    update.addcmul_(p_slice, mask, value=eff_weight_decay * lr)
                p_slice.add_(other=update, alpha=-1.0)
                all_gather_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())
        self._reduce_scatter_futures.clear()
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class UnitWeightRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor):
        out, rstd = torch_npu.npu_rms_norm(
            x, weight, epsilon=torch.finfo(x.dtype).eps
        )
        ctx.save_for_backward(x, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, rstd = ctx.saved_tensors
        return unit_rmsnorm_dx_ext.backward(grad_out, x, rstd), None


def norm(x: Tensor):
    weights = getattr(norm, "weights", None)
    if weights is None:
        weights = {}
        norm.weights = weights
    key = (x.size(-1), x.device, x.dtype)
    weight = weights.get(key)
    if weight is None:
        weight = x.new_ones((x.size(-1),))
        weights[key] = weight
    if x.size(-1) == 768:
        return UnitWeightRmsNorm.apply(x, weight)
    return torch_npu.npu_rms_norm(x, weight, epsilon=torch.finfo(x.dtype).eps)[0]

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
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

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
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


@torch.compile(backend="npu", dynamic=False, fullgraph=True)
def compiled_rotary_pair(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Compile only the shared elementwise rotary pair."""
    return rotary(q, cos, sin), rotary(k, cos, sin)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: list[int]
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float
    key_shift: bool


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim))
        self.qkvo_w.label = 'attn'
        with torch.no_grad():
            self.qkvo_w[:self.dim * 3].uniform_(-bound, bound)
            self.qkvo_w[self.dim * 3:].zero_()
        self.attn_gate = CastedLinear(12, num_heads)
        self.attn_gate.weight.label = 'attn_gate'

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
    def clear_mask_cache(cls, keep_sizes: set[int] | None = None):
        if keep_sizes is None:
            cls._shared_mask_cache.clear()
        else:
            cls._shared_mask_cache = {k: v for k, v in cls._shared_mask_cache.items() if k[0] in keep_sizes}

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas, key_shift = attn_args.ve, attn_args.sa_lambdas, attn_args.key_shift
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = compiled_rotary_pair(q, k, cos, sin)
        if key_shift:
            k0, k1, k2, k3 = k.chunk(4, dim=-1)
            k1 = torch.cat((k1[:, :1], k1[:, :-1]), dim=1)
            k3 = torch.cat((k3[:, :1], k3[:, :-1]), dim=1)
            k = torch.cat((k0, k1, k2, k3), dim=-1)
        if ve is not None:
            v = v + ve.view_as(v)

        actual_seq_qlen = seqlens
        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))
        attn_mask = self._get_window_causal_mask(max_doc_len, bm_size, x.device)

        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=attn_scale,
            atten_mask=attn_mask,
            sparse_mode=0,
            pre_tockens=bm_size - 1,
            next_tockens=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))
        return y
@torch.compile(backend="npu", dynamic=False, fullgraph=True)
def compiled_mlp_full_scope(x, c_fc, c_proj):
    """Full linear -> ReLU -> h*h -> linear MLP scope."""
    h = F.linear(x, c_fc.type_as(x))
    h = F.relu(h)
    h = h * h
    return F.linear(h, c_proj.T.type_as(h))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(hdim, dim))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim))
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_proj.lr_mul = 2.
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        return compiled_mlp_full_scope(x, self.c_fc, self.c_proj)

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx != 6 else None
        if self.attn is not None:
            self.attn.layer_idx = layer_idx
        self.mlp = MLP(dim)
        self.mlp.layer_idx = layer_idx

    def forward(self, x: Tensor, attn_args: AttnArgs):
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed.weight.label = 'embed'
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = 'smear_gate'
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
        for ve in self.value_embeds:
            ve.weight.label = 'value_embed'
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        use_fp8 = False  # FP8 not available on NPU
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        self.lm_head.weight.label = 'lm_head'
        pad = (-num_layers * 4 - 3) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat([
                1.1 * torch.ones(num_layers),
                0 * torch.ones(num_layers),
                *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],
                torch.zeros(1),
                0.5*torch.ones(1),
                -1.5 * torch.ones(1),
                torch.ones(pad),
            ])
        )
        self.scalars.label = 'scalars'
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
        assert input_seq.ndim == 1
        skip_connections = []
        skip_in = [3]
        skip_out = [6]
        x_backout = None
        backout_layer = 7

        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.scalars[1 * self.num_layers: 2 * self.num_layers]
        sa_lambdas = self.scalars[2 * self.num_layers: 4 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers+1]
        skip_lambda = self.scalars[4 * self.num_layers+2]

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == self.num_layers
        key_shift = [b==long_bm for b in bm_sizes]

        x = self.embed(input_seq)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[1], ve[2]] + [None] * (self.num_layers - 5) + [ve[0], ve[1], ve[2]]
        assert len(ve) == self.num_layers

        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        for i in range(self.num_layers):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
                bm_size=bm_sizes[i], cos=self.yarn.cos, sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale, key_shift=key_shift[i]
            )
            if i in skip_out:
                gate = torch.sigmoid(skip_lambda)
                x = x + gate * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            x = self.blocks[i](x, attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        if self.training:
            loss = compiled_training_output_loss(x, self.lm_head.weight, target_seq)
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
# Fuse softcap, single-pass online cross entropy, and their backward while
# preserving the exact summed-loss contract used by the accepted baseline.
class FusedSoftcapCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_logits, target):
        sigmoid, row_max, row_logsumexp, per_token_loss = softcap_ce_ext.forward(
            raw_logits, target
        )
        ctx.save_for_backward(sigmoid, target, row_max, row_logsumexp)
        return per_token_loss[:, 0].sum()

    @staticmethod
    def backward(ctx, grad_loss):
        sigmoid, target, row_max, row_logsumexp = ctx.saved_tensors
        grad_logits = softcap_ce_ext.backward(
            sigmoid,
            target,
            row_max,
            row_logsumexp,
            grad_loss.reshape(1).float(),
        )
        return grad_logits, None


@torch.compile(backend="npu", dynamic=False)
def compiled_training_output_logits(x, weight):
    return F.linear(x, weight)


def compiled_training_output_loss(x, weight, target):
    raw_logits = compiled_training_output_logits(x, weight)
    return FusedSoftcapCrossEntropy.apply(
        raw_logits.view(-1, raw_logits.size(-1)), target
    )

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)  # pin_memory removed for NPU
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

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        cum = _cum_lengths[1:]
        real_mask = torch.ones(cum.numel(), dtype=torch.bool)
        real_mask[1:] = cum[1:] > cum[:-1]
        actual_seq_lengths = cum[real_mask]
        actual_seq_lengths = actual_seq_lengths[actual_seq_lengths > 0].tolist()
        if not actual_seq_lengths or actual_seq_lengths[-1] != num_tokens_local:
            actual_seq_lengths.append(num_tokens_local)

        new_params = yield (
            _inputs.to(device="npu", non_blocking=True),
            _targets.to(device="npu", non_blocking=True),
            actual_seq_lengths
        )

        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_bs_schedule: tuple = (8 * 2048 * 8, 16 * 2048 * 8, 24 * 2048 * 8)
    train_bs_extension: int = 24 * 2048 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    num_scheduled_iterations: int = 2070
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.55
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    metrics_every: int = 50
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
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group(backend="hccl")
dist.barrier()
master_process = (rank == 0)

logfile = None
run_dir = None
metrics_file = None
if master_process:
    run_id = args.run_id
    run_dir = f"logs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    logfile = f"{run_dir}/train.log"
    metrics_file = f"{run_dir}/metrics.jsonl"
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

import json

class MetricsLogger:
    def __init__(self, path: str | None, num_layers: int, master: bool):
        self.path = path
        self.master = master
        self.num_layers = num_layers
        self.enabled = False
        self.mlp_rms = [None] * num_layers
        self._pending_grad_norms = [None] * num_layers

    def mlp_hook(self, module, inputs, output):
        if not self.enabled:
            return
        rms = output.detach().float().pow(2).mean().sqrt()
        self.mlp_rms[module.layer_idx] = rms

    def attach(self, model):
        for blk in model.blocks:
            blk.mlp.register_forward_hook(self.mlp_hook)

    def stash_grad_norms(self, model):
        # Preserve the exact global-gradient metric while replacing one
        # collective per parameter with one collective per dtype.
        buckets = defaultdict(list)
        for i, blk in enumerate(model.blocks):
            for p in blk.parameters():
                if p.grad is None:
                    continue
                buckets[p.grad.dtype].append((i, p.grad.detach()))

        packed = []
        for dtype_entries in buckets.values():
            flat = torch.cat([g.reshape(-1) for _, g in dtype_entries])
            handle = dist.all_reduce(flat, op=dist.ReduceOp.AVG, async_op=True)
            packed.append((dtype_entries, flat, handle))

        layer_sq = torch.zeros(self.num_layers, dtype=torch.float32, device=device)
        layer_seen = [False] * self.num_layers
        for dtype_entries, flat, handle in packed:
            handle.wait()
            offset = 0
            for layer_idx, grad in dtype_entries:
                count = grad.numel()
                reduced_grad = flat.narrow(0, offset, count)
                layer_sq[layer_idx].add_(reduced_grad.float().square().sum())
                layer_seen[layer_idx] = True
                offset += count
        self._pending_grad_norms = [
            layer_sq[i].sqrt() if layer_seen[i] else None
            for i in range(self.num_layers)
        ]

    def emit(self, step: int, train_time_ms: float):
        if not self.master or self.path is None:
            return

        def to_list(xs):
            return [None if x is None else float(x.item()) for x in xs]

        record = dict(
            step=step,
            train_time_ms=train_time_ms,
            mlp_rms=to_list(self.mlp_rms),
            grad_norm=to_list(self._pending_grad_norms),
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.mlp_rms = [None] * self.num_layers
        self._pending_grad_norms = [None] * self.num_layers

metrics_logger = None

print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")

def npu_smi():
    import subprocess
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(npu_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)
).npu()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

metrics_logger = MetricsLogger(metrics_file, num_layers=model.num_layers, master=master_process)
if master_process:
    metrics_logger.attach(model)

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

optimizer1 = DistAdam(
    embed_params + scalar_params + head_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(
    hidden_matrix_params + gate_params,
    lr=0.023,
    momentum=0.95,
    beta2=0.95,
    weight_decay=1.2,
    # Preserve the original optimizer state_dict group layout whenever
    # checkpoints are requested.  The benchmark does not save checkpoints.
    pack_large_groups=not args.save_checkpoint,
)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step: int):
    if step > args.num_scheduled_iterations:
        return 0.1
    lr_max = 1.0
    x = step / args.num_scheduled_iterations
    if x > 1/3:
       lr_max = 1.51
    if x > 2/3:
        lr_max = 1.93
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = lr_max * w + (1 - w) * 0.1
        return lr
    return lr_max

def get_ws(step: int):
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
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

def step_optimizers(step: int, optimizers, model):
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
        optimizers[0].should_sync = False

# torch.compile skipped on NPU

########################################
#            Warmup kernels            #
########################################

warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
train_loader = distributed_data_generator(args.train_files, args.train_bs_schedule[0], args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
ws_schedule = list(args.ws_schedule) + [args.ws_final]
bs_schedule = list(args.train_bs_schedule) + [args.train_bs_extension]
ws_long = ws_schedule[0]
model.train()
model.yarn.reset()
assert len(ws_schedule) == len(bs_schedule), "This warmup assumes len(ws_schedule) == len(bs_schedule)"

for idx in range(len(ws_schedule)):
    send_args = None
    if idx != 0:
        new_ws_long = ws_schedule[idx]
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
        send_args = (bs_schedule[idx], args.train_max_seq_len, grad_accum_steps)
    for step in range(warmup_steps):
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        if step % 2 == 1:
            optimizers[0].should_sync = True
        model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
        if step % 2 == 0:
            optimizers[1].step()
            optimizers[1].zero_grad(set_to_none=True)
        else:
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            optimizers[0].should_sync = False

model.zero_grad(set_to_none=True)
optimizers[0].should_sync = False
model.eval()

val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
val_loss = 0
with torch.no_grad():
    for step in range(val_steps):
        inputs, targets, cum_seqlens = next(val_loader)
        ws_idx = step % len(ws_schedule)
        if ws_idx == 0:
            model.yarn.reset()
            ws_long = ws_schedule[0]
        else:
            new_ws_long = ws_schedule[ws_idx]
            model.yarn.apply(ws_long, new_ws_long)
            ws_long = new_ws_long
        val_loss += model(inputs, targets, cum_seqlens, ws_long // 2, ws_long)

del val_loader, val_loss
model.train()
model.yarn.reset()
optimizer2.reset()
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

CausalSelfAttention.clear_mask_cache()
torch.npu.empty_cache()

########################################
#        Training and validation       #
########################################

step_batch_size = args.train_bs_schedule[0]
train_loader = distributed_data_generator(args.train_files, step_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)

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
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, f"{run_dir}/state_step{step:06d}.pt")
        break

    new_step_batch_size = get_bs(step)
    send_args = (new_step_batch_size, args.train_max_seq_len, grad_accum_steps) if new_step_batch_size != step_batch_size else None
    step_batch_size = new_step_batch_size
    sample_metrics = args.metrics_every > 0 and (step % args.metrics_every == 0)
    metrics_logger.enabled = sample_metrics and master_process
    for idx in range(grad_accum_steps):
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync = True
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (model(inputs, targets, cum_seqlens, ws_short, ws_long) / grad_accum_steps).backward()
    metrics_logger.enabled = False
    if sample_metrics:
        metrics_logger.stash_grad_norms(model)
    step_optimizers(step, optimizers, model)
    if sample_metrics and master_process:
        metrics_logger.emit(step + 1, training_time_ms + 1000 * (time.perf_counter() - t0))

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)
dist.destroy_process_group()

if master_process and run_dir is not None:
    import subprocess
    analyzer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_metrics.py")
    if os.path.exists(analyzer):
        try:
            subprocess.run([sys.executable, analyzer, run_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"analyze_metrics.py failed: {e}")
