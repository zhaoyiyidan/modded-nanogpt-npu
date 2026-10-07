import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import time
import json
import math
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
os.environ.setdefault("CPU_AFFINITY_CONF", "1")
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = f"/tmp/record005_rms_l4_{os.environ.get('LOCAL_RANK', '0')}"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_CAUSAL_MASK_CACHE = {}

def get_compressed_causal_mask(device):
    key = str(device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(torch.ones((2048, 2048), device=device, dtype=torch.bool), diagonal=1)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def manual_add_rms_norm(a, b, gamma):
    residual = a + b
    rstd = torch.rsqrt((residual * residual).mean(dim=-1, keepdim=True) + torch.finfo(residual.dtype).eps)
    return residual * rstd * gamma, residual


compiled_add_rms_norm = torch.compile(manual_add_rms_norm, backend='npu', dynamic=False)


def fused_add_rms_norm(a, b, gamma):
    return compiled_add_rms_norm(a, b, gamma)

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

def zeropower_via_newtonschulz5_batched(G, steps=10, eps=1e-7):
    """Apply the same Newton-Schulz iteration independently to a matrix batch."""
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True) + eps)
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if transpose:
        X = X.transpose(-2, -1)
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._collective_layout = None

    def _get_collective_layout(self, params):
        if self._collective_layout is not None:
            return self._collective_layout
        if self.world_size != 16 or len(params) != 48:
            raise RuntimeError(
                f"balanced Muon all-gather requires 48 params on 16 ranks, "
                f"got {len(params)} params on {self.world_size} ranks"
            )
        owners = [
            (index // 4) // 3 if index % 4 == 0 else 4 + index // 4
            for index in range(len(params))
        ]
        offsets = []
        loads = [0] * self.world_size
        for param, owner in zip(params, owners):
            offsets.append(loads[owner])
            loads[owner] += param.numel()
        if len(set(loads)) != 1:
            raise RuntimeError(f"Muon all-gather layout is not balanced: {loads}")
        width = loads[0]
        local_updates = torch.zeros(
            width, device=params[0].device, dtype=torch.bfloat16
        )
        gathered_updates = torch.empty(
            self.world_size * width,
            device=params[0].device,
            dtype=torch.bfloat16,
        )
        self._collective_layout = (
            owners, offsets, width, local_updates, gathered_updates
        )
        return self._collective_layout

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            params = [p for p in group['params'] if p.grad is not None]
            owners, offsets, width, local_updates, gathered_updates = self._get_collective_layout(params)
            local_updates.zero_()
            qkv_batches = []
            for i, p in enumerate(params):
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                if owners[i] == self.rank:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    if g.size(0) == 3 * g.size(1) and group['backend'] == 'newtonschulz5':
                        qkv_batches.append((i, p, g.view(3, g.size(1), g.size(1))))
                        continue
                    elif g.size(0) == 3 * g.size(1):
                        g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    else:
                        g = zeropower_backend(g, steps=group['backend_steps'])
                    start = offsets[i]
                    local_updates[start:start + p.numel()].copy_(g.reshape(-1))
            if qkv_batches:
                batched = torch.cat([entry[2] for entry in qkv_batches], dim=0)
                batched = zeropower_via_newtonschulz5_batched(
                    batched, steps=group['backend_steps']
                )
                for update, (i, p, _) in zip(batched.split(3), qkv_batches):
                    start = offsets[i]
                    local_updates[start:start + p.numel()].copy_(update.reshape(-1))
            dist.all_gather_into_tensor(gathered_updates, local_updates)
            for i, p in enumerate(params):
                start = owners[i] * width + offsets[i]
                g = gathered_updates[start:start + p.numel()].view_as(p)
                scale = (g.size(1) if g.size(0) == 3 * g.size(1) else max(g.size(0), g.size(1)))**0.5
                p.data.add_(g, alpha=-lr * scale)

    @torch.no_grad()
    def synchronize_state(self):
        """Materialize owner momentum on every rank before checkpointing."""
        for group in self.param_groups:
            params = list(group['params'])
            owners, offsets, width, local_buffer, gathered_buffer = self._get_collective_layout(params)
            local_buffer.zero_()
            for i, p in enumerate(params):
                if owners[i] == self.rank:
                    start = offsets[i]
                    local_buffer[start:start + p.numel()].copy_(
                        self.state[p]['momentum_buffer'].reshape(-1)
                    )
            dist.all_gather_into_tensor(gathered_buffer, local_buffer)
            for i, p in enumerate(params):
                start = owners[i] * width + offsets[i]
                self.state[p]['momentum_buffer'].copy_(
                    gathered_buffer[start:start + p.numel()].view_as(p)
                )

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
            cos = freqs.cos().bfloat16()
            sin = freqs.sin().bfloat16()
            self.cos_cached = torch.cat((cos, cos), dim=-1)
            self.sin_cached = torch.cat((-sin, -sin), dim=-1)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()
        self.rotary = Rotary(self.head_dim)
        self.register_buffer('qk_norm_weight', torch.ones(self.head_dim, dtype=torch.bfloat16), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        fused_key = prefix + 'c_qkv.weight'
        legacy_keys = [prefix + f'c_{name}.weight' for name in ('q', 'k', 'v')]
        if fused_key not in state_dict and all(key in state_dict for key in legacy_keys):
            state_dict[fused_key] = torch.cat([state_dict.pop(key) for key in legacy_keys], dim=0)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        cos, sin = self.rotary(q)
        q = torch_npu.npu_rms_norm(q, self.qk_norm_weight, torch.finfo(torch.float32).eps)[0]
        k = torch_npu.npu_rms_norm(k, self.qk_norm_weight, torch.finfo(torch.float32).eps)[0]
        q = torch_npu.npu_rotary_mul(q, cos, sin, 'half')
        k = torch_npu.npu_rotary_mul(k, cos, sin, 'half')
        y = torch_npu.npu_fusion_attention(
            q, k, v, self.n_head, "BSND",
            atten_mask=get_compressed_causal_mask(q.device),
            scale=1.0 / math.sqrt(self.head_dim), keep_prob=1.0,
            pre_tockens=65536, next_tockens=0, inner_precise=0, sparse_mode=2,
        )[0].view_as(x)
        y = self.c_proj(y)
        return y

def mlp_activation_where(x):
    return torch.where(x > 0, x * x, torch.zeros((), device=x.device, dtype=x.dtype))


compiled_mlp_activation = torch.compile(mlp_activation_where, backend='npu', dynamic=False)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = compiled_mlp_activation(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.register_buffer('rms_weight', torch.ones(config.n_embd), persistent=False)

    def forward(self, x, normalized_x):
        normalized_x, x = fused_add_rms_norm(x, self.attn(normalized_x), self.rms_weight)
        normalized_x, x = fused_add_rms_norm(x, self.mlp(normalized_x), self.rms_weight)
        return x, normalized_x

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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        normalized_x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x, normalized_x = block(x, normalized_x)
        x = normalized_x
        if targets is not None:
            logits = self.lm_head(x)
            if self.training:
                loss = torch_npu.npu_cross_entropy_loss(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    None,
                    "mean",
                    -1,
                    0.0,
                    0.0,
                    False,
                )[0].squeeze(0)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
        else:
            logits = self.lm_head(x[:, [-1], :])
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
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size : int = 8*64
    device_batch_size : int = 32
    sequence_length : int = 1024
    num_iterations : int = 5100
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450
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
    print(f"train_accumulation_steps: {train_accumulation_steps}")
x, y = train_loader.next_batch()

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.npu()
model = DDP(
    model,
    device_ids=[ddp_local_rank],
    broadcast_buffers=False,
    gradient_as_bucket_view=True,
    static_graph=True,
    bucket_cap_mb=50,
)
raw_model = model.module
ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=False)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95,
                  rank=ddp_rank, world_size=ddp_world_size)
optimizers = [optimizer1, optimizer2]

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

logfile = None
run_dir = os.environ.get("RUN_DIR", None)
if master_process:
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    if run_dir is None:
        run_dir = f'logs/{run_id}'
    os.makedirs(run_dir, exist_ok=True)
    logfile = os.path.join(run_dir, 'train.log')
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
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            if logfile:
                with open(logfile, "a") as f:
                    f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        torch.npu.synchronize()
        t0 = time.time()

    save_step = last_step or (args.save_every > 0 and step % args.save_every == 0)
    if save_step:
        torch.npu.synchronize()
        if master_process:
            training_time_ms += 1000 * (time.time() - t0)
        optimizer2.synchronize_state()
    if master_process and save_step:
        if run_dir:
            log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, os.path.join(run_dir, f'state_step{step:06d}.pt'))
    if save_step:
        torch.npu.synchronize()
        if master_process:
            t0 = time.time()

    if last_step:
        break

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
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:deferred train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        if logfile:
            with open(logfile, "a") as f:
                f.write(f"step:{step+1}/{args.num_iterations} train_loss:deferred train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")

    if run_dir and logfile:
        final_val = float(val_loss.item()) if val_loss is not None else None
        metrics = {"final_val_loss": final_val, "training_time_ms": training_time_ms}
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

dist.destroy_process_group()
