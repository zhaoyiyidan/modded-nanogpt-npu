import os
import sys
_cache_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
_rank_cache = os.path.join(_cache_root, "rank_" + os.environ.get("RANK", "0"))
os.makedirs(_rank_cache, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = os.path.join(_rank_cache, "triton")
os.environ["TMPDIR"] = os.path.join("/dev/shm", "record009_triton_rank_" + os.environ.get("RANK", "0"))
os.environ["TMP"] = os.environ["TMPDIR"]
os.environ["TEMP"] = os.environ["TMPDIR"]
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = os.path.join(_rank_cache, "npu_compile")
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import time
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_npu
from triton_relu_square import TritonFusedReluSquare
from triton_shortcut import TritonShortcutAffine
from triton_cap_cancel_chunked16k_nomaterialize import TritonCancelledChunked16KNoMaterializeCapCast
torch.empty(1, device="npu", requires_grad=True).backward()

from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='npu', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class FusedRotaryWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(q, cos, sin)
        batch, seq, _, dim = q.shape
        full_cos = torch.cat((cos, cos), dim=-1).expand(batch, seq, 1, dim)
        full_sin = torch.cat((sin, sin), dim=-1).expand(batch, seq, 1, dim)
        return torch_npu.npu_apply_rotary_pos_emb(
            q, k, full_cos, -full_sin, layout="BSND", rotary_mode="half"
        )

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        q, cos, sin = ctx.saved_tensors
        batch, seq, _, dim = q.shape
        full_cos = torch.cat((cos, cos), dim=-1).expand(batch, seq, 1, dim)
        full_sin = torch.cat((sin, sin), dim=-1).expand(batch, seq, 1, dim)
        grad_q, _, _ = torch_npu.npu_rotary_mul_backward(
            grad_q, q, full_cos, -full_sin, "half"
        )
        grad_k = apply_rotary_emb(grad_k, cos, -sin)
        return grad_q, grad_k, None, None

class FusedRotaryBothBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(q, k, cos, sin)
        batch, seq, _, dim = q.shape
        full_cos = torch.cat((cos, cos), dim=-1).expand(batch, seq, 1, dim)
        full_sin = torch.cat((sin, sin), dim=-1).expand(batch, seq, 1, dim)
        return torch_npu.npu_apply_rotary_pos_emb(
            q, k, full_cos, -full_sin, layout="BSND", rotary_mode="half"
        )

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        q, k, cos, sin = ctx.saved_tensors
        batch, seq, _, dim = q.shape
        full_cos = torch.cat((cos, cos), dim=-1).expand(batch, seq, 1, dim)
        full_sin = torch.cat((sin, sin), dim=-1).expand(batch, seq, 1, dim)
        grad_q, _, _ = torch_npu.npu_rotary_mul_backward(
            grad_q, q, full_cos, -full_sin, "half"
        )
        grad_k, _, _ = torch_npu.npu_rotary_mul_backward(
            grad_k, k, full_cos, -full_sin, "half"
        )
        return grad_q, grad_k, None, None
class FusedAddRMSWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gamma):
        ctx.branch_dtype = branch.dtype
        branch = branch.to(x.dtype)
        norm, rstd, residual = torch_npu.npu_add_rms_norm(
            x, branch, gamma, torch.finfo(torch.float32).eps
        )
        ctx.save_for_backward(residual, rstd, gamma)
        return residual, norm

    @staticmethod
    def backward(ctx, grad_residual, grad_norm):
        residual, rstd, gamma = ctx.saved_tensors
        grad, _grad_gamma = torch_npu.npu_rms_norm_backward(
            grad_norm, residual, gamma, rstd
        )
        grad = grad + grad_residual
        return grad, grad.to(ctx.branch_dtype), None

class InplaceReluSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.mark_dirty(x)
        x.relu_().square_()
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return grad * (2 * torch.sqrt(y))

_RMS_GAMMA_CACHE = {}

def fast_rms_norm(x):
    key = (x.size(-1), str(x.device), x.dtype)
    if key not in _RMS_GAMMA_CACHE:
        _RMS_GAMMA_CACHE[key] = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
    return torch_npu.npu_rms_norm(x, _RMS_GAMMA_CACHE[key], torch.finfo(torch.float32).eps)[0]

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.official_k = layer_idx >= 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, v1=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        cos, sin = self.rotary(q)
        gamma = self._get_norm_gamma(self.head_dim, q.device, q.dtype)
        q = torch_npu.npu_rms_norm(q, gamma, torch.finfo(torch.float32).eps)[0]
        k = torch_npu.npu_rms_norm(k, gamma, torch.finfo(torch.float32).eps)[0]
        rotary_fn = FusedRotaryBothBackward if self.official_k else FusedRotaryWithBackward
        q, k = rotary_fn.apply(q, k, cos, sin)
        # Use npu_fusion_attention instead of F.scaled_dot_product_attention
        scale = 1.0 / math.sqrt(self.head_dim)
        y = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=self.n_head,
            input_layout="BSND",
            scale=scale,
            pre_tockens=T,
            next_tockens=0,
            sparse_mode=0,
            atten_mask=self._get_causal_mask(T, x.device),
        )[0]
        y = y.reshape_as(x)
        y = self.c_proj(y)
        return y, v1

    _mask_cache = {}
    _gamma_cache = {}

    @classmethod
    def _get_norm_gamma(cls, dim, device, dtype):
        key = (dim, str(device), dtype)
        if key not in cls._gamma_cache:
            cls._gamma_cache[key] = torch.ones(dim, device=device, dtype=dtype)
        return cls._gamma_cache[key]

    @classmethod
    def _get_causal_mask(cls, size, device):
        if size not in cls._mask_cache:
            mask = torch.ones(size, size, dtype=torch.bool, device=device).triu(diagonal=1)
            cls._mask_cache[size] = mask
        return cls._mask_cache[size]

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = TritonFusedReluSquare.apply(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        x = TritonShortcutAffine.apply(x, x0, self.lambdas)
        x1, v1 = self.attn(fast_rms_norm(x), v1)
        gamma = CausalSelfAttention._get_norm_gamma(x.size(-1), x.device, x.dtype)
        x, x_norm = FusedAddRMSWithBackward.apply(x, x1, gamma)
        x = x + self.mlp(x_norm)
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        x = fast_rms_norm(x)
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = fast_rms_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = TritonCancelledChunked16KNoMaterializeCapCast.apply(logits)[0]
            logits = logits.float()
            loss = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean", ignore_index=-1
            )[0].mean()
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = TritonCancelledChunked16KNoMaterializeCapCast.apply(logits)[0]
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        self.is_train = "train_" in filename_pattern
        self.executor = ThreadPoolExecutor(max_workers=1) if self.is_train else None
        self.next_future = None
        self.prefetch_stream = torch.npu.Stream() if self.is_train else None
        self.next_device_tokens = None
        self.next_host_tokens = None
        self.prefetch_batches = 16
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def _load_int32(self, shard_idx):
        tokens = _load_data_shard(self.files[shard_idx]).astype(np.int32)
        if self.is_train:
            return torch.from_numpy(tokens).pin_memory()
        return tokens

    def _start_prefetch(self):
        next_idx = (self.current_shard + 1) % len(self.files)
        self.next_future = self.executor.submit(self._load_int32, next_idx)

    def _launch_device_prefetch(self):
        if self.next_device_tokens is not None:
            return
        self.next_host_tokens = self.next_future.result()
        with torch.npu.stream(self.prefetch_stream):
            self.next_device_tokens = self.next_host_tokens.to(
                device=device, non_blocking=True
            )

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = torch.tensor(self._load_int32(self.current_shard), device=device)
        if self.is_train:
            self._start_prefetch()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        if self.is_train:
            self._launch_device_prefetch()
            torch.npu.current_stream().wait_stream(self.prefetch_stream)
            self.tokens = self.next_device_tokens
            self.next_device_tokens = None
            self.next_host_tokens = None
            self._start_prefetch()
        else:
            host_tokens = self._load_int32(self.current_shard)
            self.tokens = torch.tensor(host_tokens, device=device)

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = buf.long()
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.is_train and self.next_device_tokens is None:
            stride = B * T * self.num_processes
            batches_left = max(
                0, (len(self.tokens) - 1 - self.current_position) // stride
            )
            if batches_left <= self.prefetch_batches:
                self._launch_device_prefetch()
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size : int = 8*64
    device_batch_size : int = 32  # 32 per NPU * 16 NPUs = 512 = batch_size
    sequence_length : int = 1024
    num_iterations : int = 3200
    warmup_iters : int = 0
    warmdown_iters : int = 914
    weight_decay : float = 0
    val_loss_every : int = 125
    val_tokens : int = 10485760
    save_every : int = 0
args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.input_bin = os.path.join(data_path, args.input_bin)
args.input_val_bin = os.path.join(data_path, args.input_val_bin)

assert torch.npu.is_available()
dist.init_process_group(backend='hccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'npu:{ddp_local_rank}'
torch.npu.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.to(device)
# No torch.compile on NPU
model = DDP(model, device_ids=[ddp_local_rank], bucket_cap_mb=75, gradient_as_bucket_view=True, static_graph=True)
raw_model = model.module
ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.3,   betas=(0.9, 0.95))
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.002, betas=(0.9, 0.95))
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]
optimizer3 = Muon(matrix_params,           lr=0.02,  momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.02, betas=(0.9, 0.95))
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__}\n")
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
torch.npu.synchronize()
t0 = time.time()
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            if last_step:
                print(f'final_val_loss_precise:{val_loss:.8f}')
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                if last_step:
                    f.write(f'final_val_loss_precise:{val_loss:.8f}\n')
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        torch.npu.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        torch.npu.synchronize()
        t0 = time.time()

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    if not model.training:
        model.train()
    for i in range(1, train_accumulation_steps+1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    if train_accumulation_steps > 1:
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/500, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")

dist.destroy_process_group()
