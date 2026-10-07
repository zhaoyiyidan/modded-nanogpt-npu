# Derived from: train_gpt.py @ commit d70a1e6332607cf05c5f8838657dd372739191fb
# Full parameters: num_iterations=5100, warmdown_iters=1450, val_loss_every=125.
# Short-run mapping ratio: 128/5100; warmdown round(1450*128/5100)=36;
# validation only at step 0 and the final step.
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import time
import json
from dataclasses import dataclass

import numpy as np

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = f"/tmp/npu_compile_cache_{os.environ.get('RANK', '0')}"
import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DDPNoRebuild(DDP):
    # Preserve reducer prepare/post hooks while skipping bucket rebuild.
    def _pre_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return inputs, kwargs
        if not self._lazy_init_ran and not torch.compiler.is_compiling():
            self._lazy_init()
        if self._delay_all_reduce_all_params:
            return inputs, kwargs
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            assert self.logger is not None
            self.logger.set_runtime_stats_and_log()
            self.reducer.prepare_for_forward()
        return inputs, kwargs

# -----------------------------------------------------------------------------
# Muon optimizer — Distributed variant (Record 6 trick)

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile(backend="npu", dynamic=False)
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
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='npu', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            chunk_mib = float(os.environ.get("LCCL_ALLREDUCE_CHUNK_MIB", "0"))
            if chunk_mib > 0:
                chunk_numel = int(
                    chunk_mib * 1024 * 1024 // updates_flat.element_size()
                )
                for chunk in updates_flat.split(chunk_numel):
                    dist.all_reduce(chunk, op=dist.ReduceOp.SUM)
            else:
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
            cos_half = freqs.cos().bfloat16()
            sin_half = freqs.sin().bfloat16()
            self.cos_cached = torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :]
            self.sin_cached = torch.cat((-sin_half, -sin_half), dim=-1)[None, :, None, :]
        return self.cos_cached, self.sin_cached

def apply_rotary_emb(x, cos, neg_sin):
    return torch_npu.npu_rotary_mul(x, cos, neg_sin, "half")

_CAUSAL_MASK_CACHE = {}

def get_compressed_causal_mask(device):
    key = str(device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones((2048, 2048), device=device, dtype=torch.bool).triu(1)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask

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
        self.register_buffer(
            "rms_weight", torch.ones(self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = torch_npu.npu_rms_norm(q, self.rms_weight, torch.finfo(q.dtype).eps)[0]
        k = torch_npu.npu_rms_norm(k, self.rms_weight, torch.finfo(k.dtype).eps)[0]
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y, *_ = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=self.n_head,
            input_layout="BSND",
            atten_mask=get_compressed_causal_mask(q.device),
            scale=self.head_dim ** -0.5,
            sparse_mode=2,
        )
        y = y.view_as(x)
        y = self.c_proj(y)
        return y

class ReluSquareExact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        relu_x = F.relu(x)
        ctx.save_for_backward(relu_x)
        return relu_x * relu_x

    @staticmethod
    def backward(ctx, grad_output):
        (relu_x,) = ctx.saved_tensors
        return grad_output * relu_x * 2.0

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = ReluSquareExact.apply(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.register_buffer(
            "rms_weight", torch.ones(config.n_embd, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x):
        norm_x = torch_npu.npu_rms_norm(
            x, self.rms_weight, torch.finfo(x.dtype).eps
        )[0]
        x = x + self.attn(norm_x)
        norm_x = torch_npu.npu_rms_norm(
            x, self.rms_weight, torch.finfo(x.dtype).eps
        )[0]
        x = x + self.mlp(norm_x)
        return x

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
        self.register_buffer(
            "final_rms_weight", torch.ones(config.n_embd, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = torch_npu.npu_rms_norm(
            x, self.final_rms_weight, torch.finfo(x.dtype).eps
        )[0]
        if targets is not None:
            logits = self.lm_head(x)
            if not self.training:
                logits = logits.float()
            loss, _, _, _ = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), targets.view(-1)
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

DATA_ROOT = os.environ.get("DATA_ROOT", "/models/modded-nanogpt_record50_cautious_wd/data")
METRICS_FILE = os.environ.get("METRICS_FILE", "metrics.json")

@dataclass
class Hyperparameters:
    input_bin : str = f'{DATA_ROOT}/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = f'{DATA_ROOT}/fineweb10B/fineweb_val_*.bin'
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
if "SMOKE_NUM_ITERATIONS" in os.environ:
    args.num_iterations = int(os.environ["SMOKE_NUM_ITERATIONS"])
    args.warmdown_iters = max(
        1, round(36 * args.num_iterations / 128)
    )
    args.val_loss_every = args.num_iterations
    args.batch_size = int(os.environ.get("SMOKE_BATCH_SIZE", args.batch_size))
    args.device_batch_size = int(
        os.environ.get("SMOKE_DEVICE_BATCH_SIZE", args.device_batch_size)
    )
    args.sequence_length = int(
        os.environ.get("SMOKE_SEQUENCE_LENGTH", args.sequence_length)
    )
    args.val_tokens = int(os.environ.get("SMOKE_VAL_TOKENS", "256"))
    print(
        "FUNCTIONAL_SMOKE_ONLY: "
        f"iterations={args.num_iterations} warmdown={args.warmdown_iters} "
        f"global_batch={args.batch_size} device_batch={args.device_batch_size} "
        f"sequence_length={args.sequence_length} val_tokens={args.val_tokens}"
    )

assert torch.npu.is_available()
comm_backend = os.environ.get("COMM_BACKEND", "hccl")
dist.init_process_group(backend=comm_backend)
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
    print(f"val_steps: {val_steps}")
x, y = train_loader.next_batch()

num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.npu()
# The node's AICPU package cannot open the libraries used by DDP's NPU init
# verification/sync path. Preserve exact rank-0 initialization semantics over a
# CPU/Gloo side group; all timed training collectives remain on the HCCL group.
init_sync_group = dist.new_group(backend="gloo")
with torch.no_grad():
    for tensor in [parameter.data for parameter in model.parameters()] + list(model.buffers()):
        host_tensor = tensor.detach().cpu() if ddp_rank == 0 else torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu"
        )
        dist.broadcast(host_tensor, src=0, group=init_sync_group)
        if ddp_rank != 0:
            tensor.copy_(host_tensor.to(device=tensor.device))
dist.barrier(group=init_sync_group)
dist.destroy_process_group(init_sync_group)
ddp_kwargs = {"init_sync": False, "broadcast_buffers": False}
if "DDP_BUCKET_CAP_MB" in os.environ:
    ddp_kwargs["bucket_cap_mb"] = float(os.environ["DDP_BUCKET_CAP_MB"])
    if master_process:
        print(f"DDP bucket_cap_mb: {ddp_kwargs['bucket_cap_mb']}")
if os.environ.get("DDP_FIND_UNUSED_PARAMETERS", "0") == "1":
    ddp_kwargs["find_unused_parameters"] = True
    if master_process:
        print("DDP find_unused_parameters: True")
if os.environ.get("DDP_STATIC_GRAPH", "0") == "1":
    ddp_kwargs["static_graph"] = True
    if master_process:
        print("DDP static_graph: True")
ddp_cls = DDPNoRebuild if os.environ.get("DDP_SKIP_REBUILD", "0") == "1" else DDP
if master_process and ddp_cls is DDPNoRebuild:
    print("DDP skip_rebuild: True")
model = ddp_cls(model, device_ids=[ddp_local_rank], **ddp_kwargs)
if "DDP_LCCL_CHUNK_MIB" in os.environ:
    ddp_chunk_mib = float(os.environ["DDP_LCCL_CHUNK_MIB"])
    ddp_comm_state = {
        "group": dist.group.WORLD,
        "world_size": ddp_world_size,
        "chunk_numel": int(ddp_chunk_mib * 1024 * 1024 // 4),
    }

    def chunked_lccl_ddp_hook(state, bucket):
        buffer = bucket.buffer()
        for chunk in buffer.split(state["chunk_numel"]):
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=state["group"])
        buffer.div_(state["world_size"])
        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    model.register_comm_hook(ddp_comm_state, chunked_lccl_ddp_hook)
    if master_process:
        print(f"DDP LCCL chunk_mib: {ddp_chunk_mib}")
raw_model = model.module
ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay)
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

metrics_list = []
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
                with torch.no_grad():
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
        if comm_backend == "lccl":
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= ddp_world_size
        else:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        vl = val_loss.item()
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{vl:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{vl:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            metrics_list.append({"step": step, "val_loss": vl, "train_time_ms": training_time_ms})
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
        with ctx:
            _, loss = model(x, y, return_logits=False)
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
    with open(METRICS_FILE, "w") as mf:
        json.dump({"metrics": metrics_list}, mf, indent=2)
    print(f"Metrics saved to {METRICS_FILE}")

dist.destroy_process_group()
