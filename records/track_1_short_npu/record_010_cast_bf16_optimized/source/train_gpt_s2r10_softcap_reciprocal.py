import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = f"/tmp/record010_compile_cache_{os.environ.get('LOCAL_RANK', '0')}"
import torch
import torch_npu
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


class NpuApplyAdam(torch.optim.Optimizer):
    """Adam with the same state/math expressed through one fused NPU kernel."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay'] != 0:
                    raise RuntimeError("NpuApplyAdam candidate only supports weight_decay=0")
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                torch_npu.npu_apply_adam(
                    beta1 ** state['step'],
                    beta2 ** state['step'],
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    p.grad,
                    False,
                    False,
                    out=(p, state['exp_avg'], state['exp_avg_sq']),
                )


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        world_size = int(os.environ['WORLD_SIZE'])
        self._collective_layouts = []
        for group in self.param_groups:
            payload_sizes = [0] * world_size
            owners = []
            offsets = []
            for i, p in enumerate(group['params']):
                owner = i % world_size
                owners.append(owner)
                offsets.append(payload_sizes[owner])
                payload_sizes[owner] += p.numel()
            payload_elems = max(payload_sizes)
            device = group['params'][0].device
            local_updates = torch.zeros(payload_elems, device=device, dtype=torch.bfloat16)
            gathered_updates = torch.empty(
                world_size * payload_elems, device=device, dtype=torch.bfloat16
            )
            self._collective_layouts.append(
                (owners, offsets, payload_elems, local_updates, gathered_updates)
            )

    def step(self):
        rank = int(os.environ['RANK'])
        for group_index, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            owners, offsets, payload_elems, local_updates, gathered_updates = self._collective_layouts[group_index]
            local_updates.zero_()
            for i, p in enumerate(group['params']):
                if owners[i] == rank:
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
                    local_updates[offsets[i]:offsets[i]+p.numel()].copy_(g.flatten())

            dist.all_gather_into_tensor(gathered_updates, local_updates)

            for i, p in enumerate(group['params']):
                begin = owners[i] * payload_elems + offsets[i]
                g = gathered_updates[begin:begin+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)

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

class CastedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):
    _causal_mask_cache = {}

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        self.register_buffer(
            "qk_norm_gamma",
            torch.ones(self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        self.lamb = nn.Parameter(torch.tensor(0.5))

    @classmethod
    def _get_causal_mask(cls, size, device):
        if size not in cls._causal_mask_cache:
            row = torch.arange(size, device=device).unsqueeze(1)
            col = torch.arange(size, device=device).unsqueeze(0)
            cls._causal_mask_cache[size] = (col > row).bool()
        return cls._causal_mask_cache[size]

    def forward(self, x, v1=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        cos, sin = self.rotary(q)
        q = torch_npu.npu_rms_norm(
            q, self.qk_norm_gamma, epsilon=torch.finfo(torch.bfloat16).eps
        )[0]
        k = torch_npu.npu_rms_norm(
            k, self.qk_norm_gamma, epsilon=torch.finfo(torch.bfloat16).eps
        )[0]
        cos_full = torch.cat((cos, cos), dim=-1)
        neg_sin_full = torch.cat((-sin, -sin), dim=-1)
        q = torch_npu.npu_rotary_mul(q, cos_full, neg_sin_full)
        k = torch_npu.npu_rotary_mul(k, cos_full, neg_sin_full)
        scale = 1.0 / (self.head_dim ** 0.5)
        causal_mask = self._get_causal_mask(T, x.device)
        y = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=self.n_head,
            input_layout="BSND",
            scale=scale,
            pre_tockens=T,
            next_tockens=0,
            sparse_mode=0,
            atten_mask=causal_mask,
        )[0].view_as(x)
        y = self.c_proj(y)
        return y, v1

def mlp_activation(x):
    y = F.relu(x)
    return y * y

compiled_mlp_activation = torch.compile(mlp_activation, backend="npu", dynamic=False)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = compiled_mlp_activation(x)
        x = self.c_proj(x)
        return x

def residual_mix_norm(x, x0, lambdas):
    mixed = lambdas[0] * x + lambdas[1] * x0
    return F.rms_norm(mixed, (mixed.size(-1),)), mixed

def residual_add_norm(x, residual):
    summed = x + residual
    return F.rms_norm(summed, (summed.size(-1),)), summed

compiled_residual_mix_norm = torch.compile(
    residual_mix_norm, backend="npu", dynamic=False
)
compiled_residual_add_norm = torch.compile(
    residual_add_norm, backend="npu", dynamic=False
)


_rms_gamma_cache = {}


def unit_rms_gamma(x):
    key = (x.device.index, x.dtype, x.size(-1))
    gamma = _rms_gamma_cache.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        _rms_gamma_cache[key] = gamma
    return gamma


class NpuRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        gamma = unit_rms_gamma(x)
        normalized, rstd = torch_npu.npu_rms_norm(
            x, gamma, epsilon=torch.finfo(torch.bfloat16).eps
        )
        ctx.save_for_backward(x, rstd, gamma)
        return normalized

    @staticmethod
    def backward(ctx, grad_norm):
        x, rstd, gamma = ctx.saved_tensors
        grad, _ = torch_npu.npu_rms_norm_backward(
            grad_norm, x, gamma, rstd
        )
        return grad


class NpuResidualAddNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual):
        gamma = unit_rms_gamma(x)
        normalized, rstd, summed = torch_npu.npu_add_rms_norm(
            x, residual, gamma, epsilon=torch.finfo(torch.bfloat16).eps
        )
        ctx.save_for_backward(summed, rstd, gamma)
        return normalized, summed

    @staticmethod
    def backward(ctx, grad_norm, grad_sum):
        summed, rstd, gamma = ctx.saved_tensors
        grad, _ = torch_npu.npu_rms_norm_backward(
            grad_norm, summed, gamma, rstd
        )
        grad = grad + grad_sum
        return grad, grad


def npu_residual_mix_backward(
    grad_norm, grad_sum, mixed, rstd, gamma, x, x0, lambdas
):
    grad, _ = torch_npu.npu_rms_norm_backward(
        grad_norm, mixed, gamma, rstd
    )
    grad = grad + grad_sum
    grad_x = (grad * lambdas[0]).to(x.dtype)
    grad_x0 = (grad * lambdas[1]).to(x0.dtype)
    grad_lambdas = torch.stack(
        ((grad * x).sum(), (grad * x0).sum())
    ).float()
    return grad_x, grad_x0, grad_lambdas


compiled_npu_residual_mix_backward = torch.compile(
    npu_residual_mix_backward, backend="npu", dynamic=False
)


class NpuResidualMixNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0, lambdas):
        gamma = unit_rms_gamma(x)
        left = lambdas[0] * x
        right = lambdas[1] * x0
        normalized, rstd, mixed = torch_npu.npu_add_rms_norm(
            left, right, gamma, epsilon=torch.finfo(torch.bfloat16).eps
        )
        ctx.save_for_backward(mixed, rstd, gamma, x, x0, lambdas)
        return normalized, mixed

    @staticmethod
    def backward(ctx, grad_norm, grad_sum):
        mixed, rstd, gamma, x, x0, lambdas = ctx.saved_tensors
        return compiled_npu_residual_mix_backward(
            grad_norm, grad_sum, mixed, rstd, gamma, x, x0, lambdas
        )

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0):
        attn_input, x = NpuResidualMixNorm.apply(x, x0, self.lambdas)
        x1, v1 = self.attn(attn_input, v1)
        mlp_input, x = NpuResidualAddNorm.apply(x, x1)
        x = x + self.mlp(mlp_input)
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6
    n_embd : int = 768

def softcap_logits(logits):
    return (30 * torch.tanh(logits * (1.0 / 30.0))).float()

compiled_softcap_logits = torch.compile(softcap_logits, backend="npu", dynamic=False)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, target):
        x = self.transformer.wte(idx)
        x = NpuRMSNorm.apply(x)
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = NpuRMSNorm.apply(x)
        logits = self.lm_head(x)
        logits = compiled_softcap_logits(logits)
        loss = torch_npu.npu_cross_entropy_loss(
            logits.view(-1, logits.size(-1)), target.view(-1)
        )[0].squeeze()
        return loss.float()

# -----------------------------------------------------------------------------
# Distributed Data Loader

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
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.npu(), y.npu()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    input_bin : str = os.path.join(os.environ['DATA_PATH'], 'data/fineweb10B/fineweb_train_*.bin')
    input_val_bin : str = os.path.join(os.environ['DATA_PATH'], 'data/fineweb10B/fineweb_val_*.bin')
    batch_size : int = 8*64
    device_batch_size : int = 32
    sequence_length : int = 1024
    num_iterations : int = 3242
    warmup_iters : int = 0
    warmdown_iters : int = 926
    weight_decay : float = 0
    val_loss_every : int = 125
    val_tokens : int = 10485760
    save_every : int = 0
args = Hyperparameters()

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
model = model.npu().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

optimizer1 = NpuApplyAdam([raw_model.transformer.wte.weight], lr=0.3,   betas=(0.9, 0.95))
optimizer2 = NpuApplyAdam([raw_model.lm_head.weight],         lr=0.002, betas=(0.9, 0.95))
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]
optimizer3 = Muon(matrix_params,           lr=0.02,  momentum=0.95)
optimizer4 = NpuApplyAdam(scalar_params, lr=0.02, betas=(0.9, 0.95))
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

if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__}\nnpu-smi:\n")
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
pending_train_logs = []
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
        if master_process and pending_train_logs:
            loss_values = torch.stack([entry[1] for entry in pending_train_logs]).float().cpu().tolist()
            with open(logfile, "a") as f:
                for (logged_step, _, approx_time, logged_timed_steps), loss_value in zip(pending_train_logs, loss_values):
                    line = (
                        f"step:{logged_step}/{args.num_iterations} train_loss:{loss_value:.4f} "
                        f"train_time:{approx_time:.0f}ms step_avg:{approx_time/logged_timed_steps:.2f}ms"
                    )
                    print(line)
                    f.write(line + "\n")
            pending_train_logs.clear()
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
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

    model.train()
    for i in range(1, train_accumulation_steps+1):
        loss = model(x, y)
        train_loss = loss.detach()
        x, y = train_loader.next_batch()
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    if train_accumulation_steps != 1:
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    frac = min(step/500, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        pending_train_logs.append((step + 1, train_loss, approx_time, timed_steps))

if master_process:
    print(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")

dist.destroy_process_group()
