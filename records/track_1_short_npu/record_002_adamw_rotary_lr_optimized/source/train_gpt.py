import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

with open(sys.argv[0]) as f:
    code = f.read()

PADDED_VOCAB_SIZE = 50432

def remap_state_dict_for_padded_vocab(state_dict, padded_vocab_size=PADDED_VOCAB_SIZE):
    """Pad an original 50257-row checkpoint without changing valid weights."""
    remapped = dict(state_dict)
    for key in ("transformer.wte.weight", "lm_head.weight"):
        weight = remapped.get(key)
        if weight is not None and weight.shape[0] < padded_vocab_size:
            remapped[key] = F.pad(weight, (0, 0, 0, padded_vocab_size - weight.shape[0]))
    return remapped

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = -torch.cat((sin, sin), dim=-1)
    return torch_npu.npu_rotary_mul(x, cos_full, sin_full, "half")

_rms_gamma_cache = {}
_lm_head_bias_cache = {}

def rmsnorm(x0, eps=1e-6):
    key = (x0.device, x0.dtype, x0.shape[-1])
    gamma = _rms_gamma_cache.get(key)
    if gamma is None:
        gamma = torch.ones(x0.shape[-1], device=x0.device, dtype=x0.dtype)
        _rms_gamma_cache[key] = gamma
    return torch_npu.npu_rms_norm(x0, gamma, epsilon=eps)[0]

def masked_lm_head_linear(x, weight, effective_vocab):
    """Fuse the padded-vocabulary mask into Linear without a logits-wide COW."""
    key = (x.device, x.dtype, weight.shape[0], effective_vocab)
    bias = _lm_head_bias_cache.get(key)
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
        bias[effective_vocab:].fill_(float("-inf"))
        _lm_head_bias_cache[key] = bias
    return F.linear(x, weight, bias)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        y = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=self.n_head,
            input_layout="BSND",
            atten_mask=self.causal_mask,
            scale=1.0 / math.sqrt(self.head_dim),
            keep_prob=1.0,
            sparse_mode=2,
            pre_tockens=2147483647,
            next_tockens=0,
            inner_precise=0,
            gen_mask_parallel=True,
            sync=False,
        )[0]
        y = y.reshape(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

        # Keep the baseline RNG/initialization for all 50257 effective rows,
        # then pad only the physical GEMM dimension for NPU alignment.
        padded_weight = nn.Parameter(F.pad(
            self.lm_head.weight.detach(),
            (0, 0, 0, PADDED_VOCAB_SIZE - config.vocab_size),
        ))
        self.lm_head.weight = padded_weight
        self.lm_head.out_features = PADDED_VOCAB_SIZE
        self.transformer.wte.weight = padded_weight
        self.transformer.wte.num_embeddings = PADDED_VOCAB_SIZE

    def load_compatible_state_dict(self, state_dict, strict=True):
        return self.load_state_dict(
            remap_state_dict_for_padded_vocab(state_dict), strict=strict
        )

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)
        if targets is not None:
            logits = masked_lm_head_linear(
                x, self.lm_head.weight, self.config.vocab_size
            )
            loss = torch_npu.npu_cross_entropy_loss(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                reduction="mean", ignore_index=-1,
            )[0].mean()
        else:
            logits = self.lm_head(x[:, [-1], :])[:, :, :self.config.vocab_size]
            loss = None
        if not return_logits:
            logits = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, fused=True)
        return optimizer

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
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")
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

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin")
    parser.add_argument("--input_val_bin", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model", type=str, default="d12")
    # Public 2024-06-06 rotary source/run.sh at 3e48a54; NPU differences are
    # explicit in the record-local run.sh. Its separate 5B-token log ran 9536 steps.
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--num_iterations", type=int, default=12288)
    parser.add_argument("--learning_rate", type=float, default=0.0018)
    parser.add_argument("--warmup_iters", type=int, default=256)
    parser.add_argument("--warmdown_iters", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--val_loss_every", type=int, default=128)
    parser.add_argument("--val_max_steps", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5000)
    args = parser.parse_args()

    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.model in {"d12", "d24", "d36", "d48"}

    assert torch.npu.is_available(), "NPU is required"
    init_process_group(backend='hccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'npu:{ddp_local_rank}'
    torch.npu.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = 0
    print(f"using device: {device}")

    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total_batch_size: {args.total_batch_size}, tokens_per_fwdbwd: {tokens_per_fwdbwd}, grad_accum_steps: {grad_accum_steps}")

    ctx = torch.amp.autocast(device_type='npu', dtype=torch.bfloat16)

    model_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.train().npu()
    print0("torch.compile skipped on NPU")

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device)

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        with open(logfile, "w") as f:
            pass

    log_every = int(os.environ.get("LOG_EVERY", "100"))
    assert log_every > 0
    training_time_ms = 0.0
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            torch.npu.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            step_avg_ms = training_time_ms / max(step, 1)
            print0(f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f train_time:%.0fms step_avg:%.2fms\n" % (step, val_loss, training_time_ms, step_avg_ms))
            model.train()
            torch.npu.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum_steps):
            with ctx:
                _, loss = model(x, y, return_logits=False)
            loss = loss / grad_accum_steps
            x, y = train_loader.next_batch()
            if micro_step < grad_accum_steps - 1 and grad_accum_steps > 1:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        grads = [p.grad for p in model.parameters()]
        grad_norms = torch._foreach_norm(grads)
        grad_denoms = torch._foreach_add(grad_norms, 1e-6)
        torch._foreach_div_(grads, grad_denoms)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        should_log = ((step + 1) % log_every == 0 or
                      step + 1 == args.num_iterations)
        if should_log:
            torch.npu.synchronize()
            t1 = time.perf_counter()
            approx_training_time_ms = training_time_ms + 1000 * (t1 - t0)
            lossf = loss.item() * grad_accum_steps
            step_avg_ms = approx_training_time_ms / (step + 1)
            print0(f"step:{step+1}/{args.num_iterations} train_loss:{lossf:.6f} lr:{lr:.2e} train_time:{approx_training_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d trl:%f train_time:%.0fms step_avg:%.2fms\n" % (step, lossf, approx_training_time_ms, step_avg_ms))

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs('logs/%s' % run_id, exist_ok=True)
            torch.save(log, 'logs/%s/model_step%06d.pt' % (run_id, step))

    print0(f"peak memory consumption: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB")
    print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)")

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs('logs/%s' % run_id, exist_ok=True)
        torch.save(log, 'logs/%s/final.pt' % run_id)

    destroy_process_group()
