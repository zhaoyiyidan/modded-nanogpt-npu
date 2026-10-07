import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
os.environ.setdefault(
    "TORCH_NPU_COMPILE_CACHE_DIR",
    f"/tmp/npu_compile_cache_batched_ns_{os.environ.get('LOCAL_RANK', '0')}",
)
import uuid
import glob
import time
import json
from dataclasses import dataclass

import numpy as np
import torch
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_rms_weights = {}
_causal_masks = {}

def fast_rms_norm(x):
    key = (x.device.type, x.device.index, x.dtype, x.size(-1))
    weight = _rms_weights.get(key)
    if weight is None:
        weight = torch.ones((x.size(-1),), device=x.device, dtype=x.dtype)
        _rms_weights[key] = weight
    return torch_npu.npu_rms_norm(x, weight, epsilon=torch.finfo(x.dtype).eps)[0]

def fast_causal_attention(q, k, v, head_num):
    key = (q.device.type, q.device.index)
    mask = _causal_masks.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones((2048, 2048), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        _causal_masks[key] = mask
    return torch_npu.npu_fusion_attention(
        q, k, v,
        head_num=head_num,
        input_layout="BSND",
        atten_mask=mask,
        scale=q.size(-1) ** -0.5,
        keep_prob=1.0,
        next_tockens=0,
        sparse_mode=2,
    )[0]

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
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

def _newtonschulz5_batched_polynomial(x, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        A = torch.bmm(x, x.transpose(1, 2))
        B = torch.bmm(A, x)
        x = a * x + b * B + c * torch.bmm(A, B)
    return x

compiled_newtonschulz5_batched_polynomial = torch.compile(
    _newtonschulz5_batched_polynomial,
    backend="npu",
    dynamic=False,
    fullgraph=True,
)

def zeropower_via_newtonschulz5_batched(gradients, steps=5, eps=1e-7):
    xs = []
    transposed = []
    for gradient in gradients:
        x = gradient.bfloat16()
        x /= x.norm() + eps
        needs_transpose = gradient.size(0) > gradient.size(1)
        if needs_transpose:
            x = x.T
        xs.append(x)
        transposed.append(needs_transpose)
    x = compiled_newtonschulz5_batched_polynomial(torch.stack(xs), steps)
    outputs = x.unbind(0)
    return tuple(output.T if flag else output
                 for output, flag in zip(outputs, transposed))

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self._collective_layouts = []
        world_size = dist.get_world_size()
        for group in self.param_groups:
            sizes = [p.numel() for p in group['params']]
            loads = [0] * world_size
            owners = [-1] * len(sizes)
            for index in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i)):
                owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
                owners[index] = owner
                loads[owner] += sizes[index]
            offsets = [0] * len(sizes)
            cursors = [0] * world_size
            for index, size in enumerate(sizes):
                owner = owners[index]
                offsets[index] = cursors[owner]
                cursors[owner] += size
            max_load = max(loads)
            device = group['params'][0].device
            reduce_input = torch.zeros(world_size * max_load, device=device, dtype=torch.float32)
            reduced_grad = torch.empty(max_load, device=device, dtype=torch.float32)
            update_shard = torch.zeros(max_load, device=device, dtype=torch.bfloat16)
            gathered_updates = torch.empty(world_size * max_load, device=device, dtype=torch.bfloat16)
            self._collective_layouts.append(
                (owners, offsets, max_load, reduce_input, reduced_grad,
                 update_shard, gathered_updates)
            )

    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for group_index, group in enumerate(self.param_groups):
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            (owners, offsets, max_load, reduce_input, reduced_grad,
             update_shard, gathered_updates) = self._collective_layouts[group_index]

            for i, p in enumerate(group['params']):
                owner = owners[i]
                start = owner * max_load + offsets[i]
                reduce_input[start:start+p.numel()].copy_(p.grad.flatten())

            dist.reduce_scatter_tensor(reduced_grad, reduce_input, op=dist.ReduceOp.SUM)
            reduced_grad.div_(world_size)

            owned = []
            for i, p in enumerate(group['params']):
                if owners[i] == rank:
                    start = offsets[i]
                    g = reduced_grad[start:start+p.numel()].view_as(p)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    owned.append((p, start, g))

            shape_groups = {}
            for item in owned:
                p, _, _ = item
                key = (min(p.size(0), p.size(1)), max(p.size(0), p.size(1)))
                shape_groups.setdefault(key, []).append(item)
            for items in shape_groups.values():
                gradients = [item[2] for item in items]
                if len(gradients) == 1 or group['backend'] != 'newtonschulz5':
                    updates = tuple(zeropower_backend(
                        gradient, steps=group['backend_steps'])
                        for gradient in gradients)
                else:
                    updates = zeropower_via_newtonschulz5_batched(
                        gradients, steps=group['backend_steps'])
                for (p, start, _), g in zip(items, updates):
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    update_shard[start:start+p.numel()].copy_(g.flatten())

            dist.all_gather_into_tensor(gathered_updates, update_shard)
            gathered_updates = gathered_updates.view(world_size, max_load)
            for i, p in enumerate(group['params']):
                owner = owners[i]
                start = offsets[i]
                update = gathered_updates[owner, start:start+p.numel()].view_as(p.data)
                p.data.add_(update, alpha=-lr)


class ShardedAdam(torch.optim.Adam):
    """Exact data-parallel Adam with optimizer state sharded by vocabulary rows."""

    def __init__(self, full_param, **kwargs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert full_param.ndim == 2
        assert full_param.size(0) % world_size == 0
        rows_per_rank = full_param.size(0) // world_size
        start = rank * rows_per_rank
        local_data = full_param.data.narrow(0, start, rows_per_rank).clone()
        self.full_param = full_param
        self.world_size = world_size
        self.grad_shard = torch.empty_like(local_data)
        self.shard_param = torch.nn.Parameter(local_data, requires_grad=False)
        super().__init__([self.shard_param], **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        assert self.full_param.grad is not None
        dist.reduce_scatter_tensor(
            self.grad_shard, self.full_param.grad, op=dist.ReduceOp.SUM
        )
        self.grad_shard.div_(self.world_size)
        self.shard_param.grad = self.grad_shard
        loss = super().step(closure)
        dist.all_gather_into_tensor(self.full_param.data, self.shard_param.data)
        return loss

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
    cos_full = torch.cat((cos, cos), dim=-1)
    neg_sin_full = torch.cat((-sin, -sin), dim=-1)
    return torch_npu.npu_rotary_mul(x, cos_full, neg_sin_full, 'half')

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = fast_rms_norm(q), fast_rms_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = fast_causal_attention(q, k, v, self.n_head).view_as(x)
        y = self.c_proj(y)
        return y

def exact_relu_mul(x):
    x = F.relu(x)
    return x * x


compiled_relu_mul = torch.compile(
    exact_relu_mul, backend="npu", dynamic=False, fullgraph=True
)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = compiled_relu_mul(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(fast_rms_norm(x))
        x = x + self.mlp(fast_rms_norm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        x = fast_rms_norm(x)
        for block in self.transformer.h:
            x = block(x)
        x = fast_rms_norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction='mean', ignore_index=-1,
            )[0].squeeze()
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

    def next_batch(self, non_blocking=False):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.from_numpy(buf.astype(np.int64, copy=True))
        if non_blocking:
            buf = buf.pin_memory()
            self._inflight_host = buf
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        device_buf = buf.to("npu", non_blocking=non_blocking)
        return device_buf[:-1].view(B, T), device_buf[1:].view(B, T)

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size: int = 8*64
    device_batch_size: int = 32
    sequence_length: int = 1024
    num_iterations: int = 4578
    warmup_iters: int = 0
    warmdown_iters: int = 1308
    weight_decay: float = 0
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0
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
prefetch_stream = torch.npu.Stream()

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.to(device)
for parameter in model.parameters():
    dist.broadcast(parameter.data, src=0)
raw_model = model
ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

optimizer1 = ShardedAdam(raw_model.transformer.wte.weight, lr=0.3, betas=(0.9, 0.95))
optimizer2 = ShardedAdam(raw_model.lm_head.weight, lr=0.002, betas=(0.9, 0.95))
optimizer3 = Muon(raw_model.transformer.h.parameters(), lr=0.02, momentum=0.95)
optimizers = [optimizer1, optimizer2, optimizer3]

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

seed = int(os.environ.get("SEED", "0"))
torch.manual_seed(seed)
torch.npu.manual_seed(seed)

logdir = os.environ.get("LOG_DIR", "logs")
run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
if master_process:
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f'{run_id}.txt')
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__}\n")
        f.write(f"torch_npu version: {torch_npu.__version__}\n")
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
torch.npu.synchronize()
t0 = time.time()
train_loader.reset()
val_losses = []
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
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            if last_step:
                val_losses.append({"step": step, "val_loss": float(val_loss.item())})
        torch.npu.synchronize()
        t0 = time.time()

    if last_step:
        break

    model.train()
    for i in range(1, train_accumulation_steps+1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        with torch.npu.stream(prefetch_stream):
            next_x, next_y = train_loader.next_batch(non_blocking=True)
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
        consumer_stream = torch.npu.current_stream()
        consumer_stream.wait_stream(prefetch_stream)
        next_x.record_stream(consumer_stream)
        next_y.record_stream(consumer_stream)
        x, y = next_x, next_y
    if train_accumulation_steps != 1:
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")
    metrics_path = os.path.join(logdir, "metrics.json")
    final_val = val_losses[-1]["val_loss"] if val_losses else None
    with open(metrics_path, "w") as f:
        json.dump({"val_losses": val_losses, "final_val_loss": final_val, "seed": seed}, f, indent=2)

dist.destroy_process_group()
