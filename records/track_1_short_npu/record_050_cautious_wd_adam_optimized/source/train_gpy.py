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
os.environ["ACL_OP_COMPILER_CACHE_MODE"] = "enable"
os.environ["ACL_OP_COMPILER_CACHE_DIR"] = "/tmp/npu_op_cache"
os.environ["TASK_QUEUE_ENABLE"] = "2"
os.environ["CPU_AFFINITY_CONF"] = "1"
import torch
import torch_npu
torch.npu.config.allow_internal_format = True
torch.empty(1, device="npu", requires_grad=True).backward()  # prevents a bug on some systems

_FUSED_SOFTCAP_CE_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "custom_op", "fused_softcap_ce", "libcodex_fused_softcap_ce_ops.so",
)
torch.ops.load_library(_FUSED_SOFTCAP_CE_LIB)

_SOFTCAP_SCALE = 0.13333333333333333
_SOFTCAP_CAP = 30.0


class FusedSoftcapCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        loss, log_prob, sigmoid_saved = torch.ops.codex_ops.fused_softcap_ce_fwd(
            logits, target, _SOFTCAP_SCALE, _SOFTCAP_CAP
        )
        ctx.save_for_backward(sigmoid_saved, log_prob, target)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        sigmoid_saved, log_prob, target = ctx.saved_tensors
        grad_logits = torch.ops.codex_ops.fused_softcap_ce_bwd(
            grad_loss.contiguous(), sigmoid_saved, log_prob, target,
            _SOFTCAP_SCALE, _SOFTCAP_CAP,
        )
        return grad_logits, None

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

_polar_express_buf_cache = {}


@torch.compile(backend="npu", dynamic=False)
def _polar_express_core(X, coeffs_a, coeffs_b, coeffs_c):
    """Core polar_express loop — compile-friendly (no dict, no out=)."""
    for i in range(5):
        A = X @ X.mT
        B = A @ A
        B = B * coeffs_c[i] + A * coeffs_b[i]
        X = B @ X + X * coeffs_a[i]
    return X


def polar_express(G: Tensor, split_baddbmm: bool = False):
    X = G if G.dtype == torch.bfloat16 else G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()

    if not hasattr(polar_express, '_coeffs_tensors'):
        polar_express._coeffs_tensors = (
            torch.tensor([a for a, b, c in polar_express_coeffs], dtype=torch.bfloat16, device=X.device),
            torch.tensor([b for a, b, c in polar_express_coeffs], dtype=torch.bfloat16, device=X.device),
            torch.tensor([c for a, b, c in polar_express_coeffs], dtype=torch.bfloat16, device=X.device),
        )
    coeffs_a, coeffs_b, coeffs_c = polar_express._coeffs_tensors

    X = _polar_express_core(X, coeffs_a, coeffs_b, coeffs_c)

    if transposed:
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


@torch.compile(backend="npu", dynamic=False)
def cautious_wd_and_update_batched(params, vs, wd_vec, lr_vec):
    """Vectorized cautious WD + update for a batch of params."""
    ndim = params.ndim
    expand_shape = (-1,) + (1,) * (ndim - 1)
    wd_dev = wd_vec.view(expand_shape)
    lr_dev = lr_vec.view(expand_shape)
    mask = ((vs * params) >= 0).to(params.dtype)
    params.sub_(params * mask * wd_dev * lr_dev + vs * lr_dev)


@torch.compile(backend="npu", dynamic=False)
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
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def register_rs_hooks(self):
        """Register backward hooks to trigger reduce_scatter during backward.
        For each group, hook on the FIRST param (which gets grad LAST in backward).
        When that hook fires, all params in the group have grads ready → trigger RS."""
        self._rs_hooks = []
        self._rs_futures = {}
        self._rs_enabled = False
        
        # Initialize flat param buffers and buf_cache now (needed by hooks)
        if not hasattr(self, '_flat_params_initialized'):
            self._flat_params_initialized = True
            self._buf_cache = {}
            self._flat_param_bufs = {}
            for group_idx, group in enumerate(self.param_groups):
                params = group["params"]
                if not params:
                    continue
                chunk_size = group["chunk_size"]
                padded_num_params = chunk_size * self.world_size
                flat_buf = torch.zeros(
                    (padded_num_params, *params[0].shape),
                    dtype=params[0].dtype, device=params[0].device
                )
                for i, p in enumerate(params):
                    flat_buf[i].copy_(p.data)
                    p.data = flat_buf[i]
                self._flat_param_bufs[group_idx] = flat_buf
                self._buf_cache[group_idx] = dict(
                    stacked_grads=torch.zeros_like(flat_buf),
                    grad_chunk=torch.empty(
                        (chunk_size, *params[0].shape),
                        dtype=params[0].dtype, device=params[0].device
                    ),
                )
        
        for group_idx, group in enumerate(self.param_groups):
            params = group["params"]
            if not params:
                continue
            first_param = params[0]
            
            def make_hook(gidx):
                def hook(param):
                    if not self._rs_enabled:
                        return
                    self._trigger_rs_for_group(gidx)
                return hook
            
            h = first_param.register_post_accumulate_grad_hook(make_hook(group_idx))
            self._rs_hooks.append(h)

    @torch.no_grad()
    def _trigger_rs_for_group(self, group_idx):
        """Called from backward hook: stack grads and issue async RS."""
        group = self.param_groups[group_idx]
        params = group["params"]
        chunk_size = group["chunk_size"]
        
        bufs = self._buf_cache[group_idx]
        stacked_grads = bufs['stacked_grads']
        grad_chunk = bufs['grad_chunk']
        
        for i, p in enumerate(params):
            stacked_grads[i].copy_(p.grad, non_blocking=True)
        
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        self._rs_futures[group_idx] = (grad_chunk, future)

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

        for group_idx, group in enumerate(self.param_groups):
            params: list[Tensor] = group["params"]
            if not params:
                continue
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            bufs = self._buf_cache[group_idx]
            grad_chunk = bufs['grad_chunk']

            if hasattr(self, '_rs_futures') and group_idx in self._rs_futures:
                grad_chunk_from_hook, reduce_future = self._rs_futures[group_idx]
                grad_chunk = grad_chunk_from_hook
            else:
                stacked_grads = bufs['stacked_grads']
                for i, p in enumerate(params):
                    stacked_grads[i].copy_(p.grad, non_blocking=True)
                reduce_future = dist.reduce_scatter_tensor(
                    grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()

            group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future, buf_key=group_idx))

        if hasattr(self, '_rs_futures'):
            self._rs_futures.clear()

        all_gather_infos = []
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()    # 等待自己的这个chunk 完成同步
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
                group["param_lr_npu"] = torch.tensor(lr_mults, dtype=torch.bfloat16, device=params[0].device)
                group["param_wd_npu"] = torch.tensor(wd_mults, dtype=torch.bfloat16, device=params[0].device)

            lr_scalar = group["lr"]
            wd_scalar = group["weight_decay"] * group["lr"]
            eff_lr_dev = group["param_lr_npu"][module_idx:module_idx + num_params] * lr_scalar
            eff_wd_dev = group["param_wd_npu"][module_idx:module_idx + num_params] * wd_scalar

            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads, split_baddbmm=(ref_param.label == 'mlp'))

            red_dim = -1 if (is_gate or param_shape[-2] >= param_shape[-1]) else -2

            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum_buffer, group["beta2"], red_dim
            )
            v_chunk = v_chunk.view(grad_shape)

            bufs = self._buf_cache[info["buf_key"]]
            flat_params = self._flat_param_bufs[info["buf_key"]]
            if num_params > 0:
                param_slice = flat_params[start_idx:start_idx + num_params]
                cautious_wd_and_update_batched(
                    param_slice,
                    v_chunk[:num_params],
                    eff_wd_dev,
                    eff_lr_dev,
                )
            else:
                pass

            updated_chunk = flat_params[start_idx:start_idx + chunk_size]
            gather_future = dist.all_gather_into_tensor(
                flat_params, updated_chunk, async_op=True
            ).get_future()
            all_gather_infos.append({
                "gather_future": gather_future,
                "flat_params": flat_params,
                "orig_params": params,
            })

        # Store pending AG futures for deferred wait (overlap with next forward)
        self._pending_ag_infos = all_gather_infos

    def complete_allgather(self):
        """Deferred AG wait. Since AG writes directly into flat_params buffer
        which the params' .data already points to, no copy needed."""
        if not hasattr(self, '_pending_ag_infos') or not self._pending_ag_infos:
            return
        for info in self._pending_ag_infos:
            info["gather_future"].wait()
        self._pending_ag_infos = []

    def complete_allgather_partial(self, group_indices=None):
        """Wait only specific group AGs, leave others pending for later overlap."""
        if not hasattr(self, '_pending_ag_infos') or not self._pending_ag_infos:
            return
        if group_indices is None:
            self.complete_allgather()
            return
        remaining = []
        for i, info in enumerate(self._pending_ag_infos):
            if i in group_indices:
                info["gather_future"].wait()
            else:
                remaining.append(info)
        self._pending_ag_infos = remaining # rank 1干完这个可以干下面的事情？


        # 以当前 num_layers=11 估算，NorMuon 参数组数量是：

        # smear_gate: 1 个参数
        # attn_gate: 11 个参数，每层 1 个
        # attn:      11 个参数，每层一个 qkvo_w
        # mlp:       22 个参数，每层 c_fc + c_proj

        # 16 张 NPU 下的切分：

        # smear_gate: 1 个参数
        # chunk_size = ceil(1 / 16) = 1
        # rank 0 更新 1 个；rank 1-15 没有实际参数

        # attn_gate: 11 个参数
        # chunk_size = ceil(11 / 16) = 1
        # rank 0-10 各更新 1 个；rank 11-15 没有实际参数

        # attn: 11 个参数
        # chunk_size = ceil(11 / 16) = 1
        # rank 0-10 各更新 1 个；rank 11-15 没有实际参数

        # mlp: 22 个参数
        # chunk_size = ceil(22 / 16) = 2
        # rank 0 更新 params[0:2]
        # rank 1 更新 params[2:4]
        # ...
        # rank 10 更新 params[20:22]
        # rank 11-15 没有实际参数

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
        self._grad_slice_bufs = {}
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p[0].device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
            self._grad_slice_bufs[p] = torch.empty_like(p[:chunk_size])

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
        grad_slice = self._grad_slice_bufs[param]
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
        self._pending_ag_futures = all_gather_futures

    def complete_allgather(self):
        """Deferred AG wait for DistAdam."""
        if not hasattr(self, '_pending_ag_futures') or not self._pending_ag_futures:
            return
        torch.futures.collect_all(self._pending_ag_futures).wait()
        self._pending_ag_futures = []

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

_rms_norm_gamma_cache = {}

def norm(x: Tensor):
    dim = x.size(-1)
    key = (dim, x.dtype, x.device)
    if key not in _rms_norm_gamma_cache:
        _rms_norm_gamma_cache[key] = torch.ones(dim, dtype=x.dtype, device=x.device)
    return torch_npu.npu_rms_norm(x, _rms_norm_gamma_cache[key], epsilon=1e-6)[0]


class FusedResidualScale(torch.autograd.Function):
    """Fuse: output = a * x + b * x0 (3 ops → 1 Python dispatch)"""
    @staticmethod
    def forward(ctx, x, x0, a, b):
        ctx.save_for_backward(x, x0, a, b)
        return a * x + b * x0

    @staticmethod
    def backward(ctx, grad_output):
        x, x0, a, b = ctx.saved_tensors
        grad_x = grad_output * a
        grad_x0 = grad_output * b
        grad_a = (grad_output * x).sum(dim=(0, 1, 2), keepdim=False).reshape_as(a)
        grad_b = (grad_output * x0).sum(dim=(0, 1, 2), keepdim=False).reshape_as(b)
        return grad_x, grad_x0, grad_a, grad_b


def fused_residual_scale(x, x0, a, b):
    return FusedResidualScale.apply(x, x0, a, b)


@torch.compile(backend="npu", dynamic=False)
def _gate_backward_fused(grad_output, y_BTHD, gate):
    """Fuse: sum(grad*y, dim=-1) * gate * (1-gate) into one compiled kernel."""
    grad_gate_expanded = (grad_output * y_BTHD).sum(dim=-1)
    return grad_gate_expanded * gate * (1 - gate)


class FusedGatedOutput(torch.autograd.Function):
    """Fuse: output = y.view(B,T,H,D) * sigmoid(F.linear(x_slice, gate_w)).view(B,T,H,1)
    Combines: linear + sigmoid + view + mul = 4 ops → 1 Python dispatch
    Then reshape + output_proj: reshape + linear = 2 ops
    """
    @staticmethod
    def forward(ctx, y_BTHD, x_slice, gate_weight, num_heads):
        gate_logits = F.linear(x_slice, gate_weight)
        gate = torch.sigmoid(gate_logits)
        B, T = y_BTHD.size(0), y_BTHD.size(1)
        gated_y = y_BTHD * gate.view(B, T, num_heads, 1)
        ctx.save_for_backward(y_BTHD, x_slice, gate_weight, gate)
        ctx.num_heads = num_heads
        return gated_y

    @staticmethod
    def backward(ctx, grad_output):
        y_BTHD, x_slice, gate_weight, gate = ctx.saved_tensors
        B, T, H = y_BTHD.size(0), y_BTHD.size(1), ctx.num_heads

        gate_expanded = gate.view(B, T, H, 1)
        grad_y = grad_output * gate_expanded

        grad_gate_logits = _gate_backward_fused(grad_output, y_BTHD, gate)

        grad_gate_weight = grad_gate_logits.reshape(-1, grad_gate_logits.size(-1)).T @ x_slice.reshape(-1, x_slice.size(-1))
        grad_x_slice = F.linear(grad_gate_logits, gate_weight.T)

        return grad_y, grad_x_slice, grad_gate_weight, None


def fused_gated_output(y_BTHD, x_slice, gate_weight, num_heads):
    return FusedGatedOutput.apply(y_BTHD, x_slice, gate_weight, num_heads)


class FusedScaledLinear(torch.autograd.Function):
    """Fuse: output = F.linear(x, scale * weight)
    Saves scaled_w to avoid recomputing in backward.
    Uses matmul directly to avoid transpose ops.
    """
    @staticmethod
    def forward(ctx, x, weight, scale):
        scaled_w = scale * weight
        output = F.linear(x, scaled_w)
        ctx.save_for_backward(x, weight, scaled_w, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, scaled_w, scale = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, scaled_w)
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
        x_2d = x.reshape(-1, x.size(-1))
        grad_scaled_w = grad_output_2d.T @ x_2d
        grad_weight = scale * grad_scaled_w
        grad_scale = (grad_scaled_w * weight).sum()
        return grad_x, grad_weight, grad_scale.reshape_as(scale)


def fused_scaled_linear(x, weight, scale):
    return FusedScaledLinear.apply(x, weight, scale)

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
        return F.linear(x, self.weight)

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
        self.neg_sin = nn.Buffer((-theta.sin()).to(torch.bfloat16), persistent=False)
        self.cos_full = nn.Buffer(torch.cat([theta.cos(), theta.cos()], dim=-1).to(torch.bfloat16), persistent=False)
        self.neg_sin_full = nn.Buffer(torch.cat([(-theta.sin()), (-theta.sin())], dim=-1).to(torch.bfloat16), persistent=False)
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
        self.neg_sin.copy_(-theta.sin())
        self.cos_full.copy_(torch.cat([theta.cos().bfloat16(), theta.cos().bfloat16()], dim=-1))
        self.neg_sin_full.copy_(torch.cat([(-theta.sin()).bfloat16(), (-theta.sin()).bfloat16()], dim=-1))
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos_full: Tensor, neg_sin_full: Tensor):
    T = x_BTHD.size(-3)
    r1 = cos_full[None, :T, None, :]
    r2 = neg_sin_full[None, :T, None, :]
    return torch_npu.npu_rotary_mul(x_BTHD, r1, r2)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos_full: torch.Tensor
    neg_sin_full: torch.Tensor
    attn_scale: float
    key_shift: bool
    cached_seqlens: list = None


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

    @staticmethod
    def _extract_actual_seqlens(seqlens: Tensor, total_tokens: int) -> list[int]:
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
        cos_full, neg_sin_full = attn_args.cos_full, attn_args.neg_sin_full
        ve, sa_lambdas, key_shift = attn_args.ve, attn_args.sa_lambdas, attn_args.key_shift
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = fused_scaled_linear(x, self.qkvo_w[:self.dim * 3], sa_lambdas[0]).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = rotary(q, cos_full, neg_sin_full), rotary(k, cos_full, neg_sin_full)
        if key_shift:
            k[:, 1:, :, self.head_dim//4:self.head_dim//2] = k[:, :-1, :, self.head_dim//4:self.head_dim//2]
            k[:, 1:, :, self.head_dim//4+self.head_dim//2:] = k[:, :-1, :, self.head_dim//4+self.head_dim//2:]
        if ve is not None:
            v = v + ve.view_as(v)

        actual_seq_qlen = attn_args.cached_seqlens if attn_args.cached_seqlens is not None else self._extract_actual_seqlens(seqlens, T)
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
        y = fused_gated_output(y, x[..., :self.attn_gate.weight.size(-1)], self.attn_gate.weight, self.num_heads)
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        y = fused_scaled_linear(y, self.qkvo_w[self.dim * 3:], sa_lambdas[1])
        return y

@torch.compile(backend="npu", dynamic=False)
def relu_square(x):
    r = F.relu(x)
    return r * r

@torch.compile(backend="npu", dynamic=False)
def softcap_logits(logits, scale, cap):
    return torch.sigmoid(logits * scale) * cap

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
        x = F.linear(x, self.c_fc)
        x = relu_square(x)
        x = torch.matmul(x, self.c_proj)
        return x

def _get_rms_gamma(dim, dtype, device):
    key = (dim, dtype, device)
    if key not in _rms_norm_gamma_cache:
        _rms_norm_gamma_cache[key] = torch.ones(dim, dtype=dtype, device=device)
    return _rms_norm_gamma_cache[key]


class FusedAddRMSNorm(torch.autograd.Function):
    """Custom autograd for fused add + RMSNorm.
    Forward: uses npu_add_rms_norm (single kernel for add + norm)
    Backward: uses npu_rms_norm_backward (proven to work)
    """
    @staticmethod
    def forward(ctx, x, residual, gamma, epsilon):
        result = torch_npu.npu_add_rms_norm(x, residual, gamma, epsilon)
        normed = result[0]
        rstd = result[1]
        x_sum = result[2]
        ctx.save_for_backward(x_sum, rstd, gamma)
        ctx.epsilon = epsilon
        return normed, x_sum

    @staticmethod
    def backward(ctx, grad_normed, grad_x_sum):
        x_sum, rstd, gamma = ctx.saved_tensors
        dx_sum = torch_npu.npu_rms_norm_backward(grad_normed, x_sum, gamma, rstd)[0]
        dx_sum = dx_sum + grad_x_sum
        return dx_sum, dx_sum, None, None


def fused_add_rms_norm(x, residual, gamma, epsilon=1e-6):
    """Returns (normed, x+residual) using fused kernel."""
    return FusedAddRMSNorm.apply(x, residual, gamma, epsilon)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx != 6 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, attn_args: AttnArgs):
        if self.attn is not None:
            attn_out = self.attn(norm(x), attn_args)
            gamma = _get_rms_gamma(x.size(-1), x.dtype, x.device)
            x_normed, x = fused_add_rms_norm(x, attn_out, gamma, 1e-6)
        else:
            x_normed = norm(x)
        if self.mlp is not None:
            x = x + self.mlp(x_normed)
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
        self.scalars.wd_mul = 0.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int, precomputed_seqlens: list = None):
        assert input_seq.ndim == 1
        skip_connections = []
        skip_in = {3}
        skip_out = {6}
        x_backout = None
        backout_layer = 7

        scalars_bf16 = self.scalars.bfloat16()
        resid_lambdas = scalars_bf16[: 1 * self.num_layers]
        x0_lambdas = scalars_bf16[1 * self.num_layers: 2 * self.num_layers]
        sa_lambdas = scalars_bf16[2 * self.num_layers: 4 * self.num_layers].view(-1, 2)
        smear_lambda = scalars_bf16[4 * self.num_layers]
        backout_lambda = scalars_bf16[4 * self.num_layers+1]
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

        if hasattr(self, '_normuon_ref') and hasattr(self._normuon_ref, '_pending_ag_infos') and self._normuon_ref._pending_ag_infos:
            self._normuon_ref.complete_allgather()

        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        _cached_seqlens = precomputed_seqlens if precomputed_seqlens is not None else CausalSelfAttention._extract_actual_seqlens(seqlens, x.size(1))

        for i in range(self.num_layers):
            attn_args = AttnArgs(
                ve=ve[i], sa_lambdas=sa_lambdas[i], seqlens=seqlens,
                bm_size=bm_sizes[i], cos_full=self.yarn.cos_full,
                neg_sin_full=self.yarn.neg_sin_full,
                attn_scale=self.yarn.attn_scale, key_shift=key_shift[i],
                cached_seqlens=_cached_seqlens,
            )
            if i in skip_out:
                gate = torch.sigmoid(skip_lambda)
                x = x + gate * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = fused_residual_scale(x, x0, resid_lambdas[i], x0_lambdas[i])
            x = self.blocks[i](x, attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        if self.training:
            logits = self.lm_head(x)
            loss = FusedSoftcapCrossEntropy.apply(
                logits.view(-1, logits.size(-1)), target_seq
            )
        else:
            chunk_size = 4096
            x_2d = x.view(-1, x.size(-1))
            total_loss = 0.0
            num_tokens = x_2d.size(0)
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                logits_chunk = self.lm_head(x_2d[start:end])
                logits_chunk = softcap_logits(logits_chunk, logits_chunk.new_tensor(0.13333333333333333), logits_chunk.new_tensor(30.0))
                logits_chunk = logits_chunk.float()
                total_loss += F.cross_entropy(logits_chunk, target_seq[start:end], reduction="sum").item()
            loss = torch.tensor(total_loss / num_tokens, dtype=torch.float32, device=x.device)
        return loss


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

    _static_bufs = {}

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

        # Pre-compute actual_seqlens on CPU to avoid .tolist() sync on NPU tensor
        _cpu_cum = _cum_lengths[1:]
        _real_mask = torch.ones(_cpu_cum.numel(), dtype=torch.bool)
        _real_mask[1:] = _cpu_cum[1:] > _cpu_cum[:-1]
        _actual_cum = _cpu_cum[_real_mask]
        _actual_cum = _actual_cum[_actual_cum > 0]
        _actual_seqlens = _actual_cum.tolist()
        if not _actual_seqlens or _actual_seqlens[-1] != num_tokens_local:
            _actual_seqlens.append(num_tokens_local)

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        buf_key = (num_tokens_local, max_num_docs)
        if buf_key not in _static_bufs:
            _static_bufs[buf_key] = (
                torch.empty(num_tokens_local, dtype=torch.int32, device="npu"),
                torch.empty(num_tokens_local, dtype=torch.int64, device="npu"),
                torch.empty(max_num_docs, dtype=torch.int32, device="npu"),
            )
        static_inputs, static_targets, static_cum = _static_bufs[buf_key]
        static_inputs.copy_(_inputs, non_blocking=True)
        static_targets.copy_(_targets, non_blocking=True)
        static_cum.copy_(_cum_lengths, non_blocking=True)

        new_params = yield (static_inputs, static_targets, static_cum, _actual_seqlens)

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
    num_scheduled_iterations: int = 2050
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.55
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
assert torch.npu.is_available()
device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
torch.npu.set_device(device)
dist.init_process_group(backend="hccl")
dist.barrier()
master_process = (rank == 0)

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
    if param.ndim >= 2:
        param.data = param.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

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
    weight_decay=0.005,
)

optimizer2 = NorMuon(hidden_matrix_params + gate_params, lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2)
model._normuon_ref = optimizer2
optimizer2.register_rs_hooks()
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
        optimizers[0].step()
        optimizers[1].step()
        optimizers[0].complete_allgather()
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
        inputs, targets, cum_seqlens, actual_seqlens = train_loader.send(send_args)
        if step % 2 == 1:
            optimizers[0].should_sync = True
        model(inputs, targets, cum_seqlens, ws_long//2, ws_long, actual_seqlens).backward()
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
        inputs, targets, cum_seqlens, actual_seqlens = next(val_loader)
        ws_idx = step % len(ws_schedule)
        if ws_idx == 0:
            model.yarn.reset()
            ws_long = ws_schedule[0]
        else:
            new_ws_long = ws_schedule[ws_idx]
            model.yarn.apply(ws_long, new_ws_long)
            ws_long = new_ws_long
        val_loss += model(inputs, targets, cum_seqlens, ws_long // 2, ws_long, actual_seqlens)

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
                inputs, targets, cum_seqlens, actual_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long, actual_seqlens)
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
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    new_step_batch_size = get_bs(step)
    send_args = (new_step_batch_size, args.train_max_seq_len, grad_accum_steps) if new_step_batch_size != step_batch_size else None
    step_batch_size = new_step_batch_size
    for idx in range(grad_accum_steps):
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync = True
        if idx == grad_accum_steps - 1:
            optimizer2._rs_enabled = True
        inputs, targets, cum_seqlens, actual_seqlens = train_loader.send(send_args)
        optimizer1.complete_allgather()
        (model(inputs, targets, cum_seqlens, ws_short, ws_long, actual_seqlens) / grad_accum_steps).backward()
        optimizer2._rs_enabled = False
    step_optimizers(step, optimizers, model)

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)




print0(f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
