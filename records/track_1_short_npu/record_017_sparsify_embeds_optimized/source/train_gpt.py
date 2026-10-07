import os
import sys
import json

with open(sys.argv[0]) as f:
    code = f.read()

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
import contextlib
import copy
import glob
import time
import uuid
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu
torch.empty(1, device="npu", requires_grad=True).backward()

import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7) -> Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: "list[Tensor]" = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [
            {
                'params': [p for p in params if p.numel() == size],
                'update_buffer': [
                    torch.empty(size, device='npu', dtype=torch.bfloat16)
                    for _ in range(world_size)
                ],
            }
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers: "list[Tensor]" = group['update_buffer']
            params: "list[Tensor]" = group['params']
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    param_lr = getattr(p_world, "lr", 1.0)
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * param_lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf: Tensor = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffers[rank]
                update_prev()
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + world_size]
            update_prev()
def compiled_softcap(logits):
    return 30 * torch.tanh(logits / 30)

compiled_softcap = torch.compile(compiled_softcap, backend="npu", dynamic=False)

_RMS_GAMMA_CACHE = {}
def norm(x):
    if not torch.is_grad_enabled():
        return F.rms_norm(x, (x.size(-1),))
    key = (x.device.index, x.dtype, x.size(-1))
    gamma = _RMS_GAMMA_CACHE.get(key)
    if gamma is None:
        gamma = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        _RMS_GAMMA_CACHE[key] = gamma
    return torch_npu.npu_rms_norm(x, gamma, epsilon=torch.finfo(x.dtype).eps)[0]
def relu_squared(x):
    x = F.relu(x)
    return x * x

relu_squared = torch.compile(relu_squared, backend="npu", dynamic=False)


def weighted_sum(a, b, weights):
    return weights[0] * a + weights[1] * b

weighted_sum = torch.compile(weighted_sum, backend="npu", dynamic=False)

def scaled_add(a, b, scale):
    return a + scale * b

scaled_add = torch.compile(scaled_add, backend="npu", dynamic=False)

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = (1 / 1024) ** torch.linspace(0.0, 1.0, steps=dim // 4, dtype=torch.float32)
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i, j -> ij", t, inv_freq)
        cos_half = theta.cos()
        sin_half = theta.sin()
        self.register_buffer(
            'cos', torch.cat((cos_half, cos_half), dim=-1).to(torch.bfloat16), persistent=False
        )
        self.register_buffer(
            'neg_sin', (-torch.cat((sin_half, sin_half), dim=-1)).to(torch.bfloat16), persistent=False
        )

    def forward(self, x: Tensor):
        cos = self.cos[None, :x.size(-3), None, :]
        neg_sin = self.neg_sin[None, :x.size(-3), None, :]
        return torch_npu.npu_rotary_mul(x, cos, neg_sin)

class CausalSelfAttention(nn.Module):
    _shared_mask_cache: dict = {}

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()

    @classmethod
    def _get_window_causal_mask(cls, size: int, window: int, device: torch.device) -> Tensor:
        key = ('compressed_causal_2048',)
        if key not in cls._shared_mask_cache:
            cls._shared_mask_cache[key] = torch.triu(
                torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1
            )
        return cls._shared_mask_cache[key]

    @classmethod
    def clear_mask_cache(cls, keep_sizes: "set[int] | None" = None):
        if keep_sizes is None:
            cls._shared_mask_cache.clear()
        else:
            cls._shared_mask_cache = {k: v for k, v in cls._shared_mask_cache.items() if k[0] in keep_sizes}

    @staticmethod
    def _extract_actual_seqlens(seqlens: Tensor, total_tokens: int) -> "list[int]":
        cum = seqlens[1:]
        real_mask = torch.ones(cum.numel(), dtype=torch.bool, device=cum.device)
        real_mask[1:] = cum[1:] > cum[:-1]
        actual_cum = cum[real_mask]
        actual_cum = actual_cum[actual_cum > 0]
        result = actual_cum.tolist()
        if not result or result[-1] != total_tokens:
            result.append(total_tokens)
        return result

    def forward(self, x: Tensor, vi: "Tensor | None", actual_seq_qlen: "list[int]", attn_mask: Tensor, bm_size: int):
        B, T = x.size(0), x.size(1)
        assert B == 1, "varlen sequences requires B == 1"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        if vi is None:
            v = self.lambdas[0] * v
        else:
            v = weighted_sum(v, vi.view_as(v), self.lambdas)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)


        scale = self.head_dim ** -0.5
        y = torch_npu.npu_fusion_attention(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            head_num=self.num_heads,
            input_layout="TND",
            scale=scale,
            pre_tockens=bm_size - 1,
            next_tockens=0,
            atten_mask=attn_mask,
            sparse_mode=4,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0].unsqueeze(0)

        y = y.contiguous().view_as(x)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = relu_squared(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: "GPTConfig", layer_idx: int):
        super().__init__()
        if layer_idx != 7:
            self.attn = CausalSelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.layer_idx = layer_idx

    def forward(self, x, vi, x0, actual_seq_qlen, attn_mask, bm_size):
        x = weighted_sum(x, x0, self.lambdas)
        if self.layer_idx != 7:
            x = x + self.attn(norm(x), vi, actual_seq_qlen, attn_mask, bm_size)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim)
            for _ in range(3)
        ])

    def forward(self, inputs) -> "list[Tensor | None]":
        ve = [emb(inputs) for emb in self.embed]
        ve = [
            ve[0], ve[1], ve[2],
            None, None, None,
            None, None, None,
            ve[0], ve[1], ve[2],
        ]
        return ve

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    num_layers : int = 12
    num_heads : int = 6
    model_dim : int = 768
    def vocab_size_next_multiple_of(self, n: int):
        v = self.vocab_size
        return next(x for x in range(v + n)[::n] if x >= v)

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.num_layers)])
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.model_dim, config.vocab_size_next_multiple_of(128))
        self.lm_head.weight.data.zero_()

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, sliding_window_num_blocks: int):
        BLOCK_SIZE = 128
        assert input_seq.ndim == 1
        T = input_seq.size(0)
        bm_size = sliding_window_num_blocks * BLOCK_SIZE

        x = self.embed(input_seq[None])
        x = norm(x)
        x0 = x
        ve = self.value_embeds(input_seq)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        actual_seq_qlen = seqlens
        prev_bounds = [0] + actual_seq_qlen[:-1]
        max_doc_len = max(a - p for a, p in zip(actual_seq_qlen, prev_bounds))
        attn_mask = CausalSelfAttention._get_window_causal_mask(max_doc_len, bm_size, x.device)

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, actual_seq_qlen, attn_mask, bm_size)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = scaled_add(x, skip_connections.pop(), self.skip_weights[i])
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, actual_seq_qlen, attn_mask, bm_size)

        x = norm(x)
        if self.training:
            logits = self.lm_head(x)
            logits = compiled_softcap(logits)
            logits = logits.float()
            loss = torch_npu.npu_cross_entropy_loss(logits.view(-1, logits.size(-1)), target_seq.view(-1))[0].squeeze()
        else:
            chunk_size = 4096
            x_2d = x.view(-1, x.size(-1))
            total_loss = 0.0
            num_tokens = x_2d.size(0)
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                logits_chunk = self.lm_head(x_2d[start:end])
                logits_chunk = 30 * torch.tanh(logits_chunk / 30)
                logits_chunk = logits_chunk.float()
                total_loss += F.cross_entropy(logits_chunk, target_seq.view(-1)[start:end], reduction="sum").item()
            loss = torch.tensor(total_loss / num_tokens, dtype=torch.float32, device=x.device)
        return loss
BOS_ID = 50256

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

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
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                starts[r].append(cur)
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

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, align_to_bos: bool = True):
    _rank = dist.get_rank() if dist.is_initialized() else 0
    _world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % _world_size == 0

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=_world_size, quickload=True)
        preloader = DataPreloader(file_iter, _world_size)
        preloader.start()
    else:
        pos = 0

    while True:
        num_tokens_local = num_tokens // _world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[_rank]), torch.tensor(seq_ends[_rank])
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
            pos_local = pos + _rank * num_tokens_local
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

        _actual_seq_qlen = []
        _previous_length = 0
        for _length in _cum_lengths[1:].tolist():
            if _length > _previous_length:
                _actual_seq_qlen.append(int(_length))
                _previous_length = int(_length)
        yield (
            _inputs.to(device="npu", non_blocking=True),
            _targets.to(device="npu", non_blocking=True),
            _actual_seq_qlen
        )
@dataclass
class Hyperparameters:
    train_files : str = 'data/fineweb10B/fineweb_train_*.bin'
    val_files : str = 'data/fineweb10B/fineweb_val_*.bin'
    train_batch_tokens : int = 8 * 64 * 1024
    val_batch_tokens : int = 64 * 1024 * 16
    val_tokens : int = 10485760
    train_max_seq_len : int = 65536
    num_iterations : int = 1490
    warmup_iters : int = 0
    cooldown_iters : int = 600
    weight_decay : float = 0
    val_loss_every : int = 125
    save_every : int = 0
args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert torch.npu.is_available()
device = torch.device(f"npu:{local_rank}")
torch.npu.set_device(device)
print(f"using device: {device}")
dist.init_process_group(backend='hccl')
dist.barrier()
master_process = (rank == 0)

logfile = None
run_dir = None
if master_process:
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    run_dir = os.environ.get("LOG_DIR", f"logs/{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    logfile = os.path.join(run_dir, "train.log")
    src_basename = os.path.basename(sys.argv[0])
    with open(os.path.join(run_dir, src_basename), "w") as f:
        f.write(code)
    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(f"cwd: {os.getcwd()}\n")
        f.write(f"argv: {' '.join(sys.argv)}\n")
        f.write(f"world_size: {world_size}\n")
        for k in ("TORCHELASTIC_RUN_ID", "MASTER_ADDR", "MASTER_PORT", "DATA_PATH", "LOG_DIR", "RUN_ID"):
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

print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__}")

def npu_smi():
    import subprocess
    return subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(npu_smi())
print0("="*100)

grad_accum_steps = max(1, 8 // world_size)
assert args.val_tokens % args.val_batch_tokens == 0
val_steps = grad_accum_steps * args.val_tokens // args.val_batch_tokens

print0(f"Grad accumulation steps: {grad_accum_steps}")
print0(f"Val steps: {val_steps}")
print0(f"Train batch tokens: {args.train_batch_tokens}")
print0('='*100)

train_loader = distributed_data_generator(args.train_files, args.train_batch_tokens, args.train_max_seq_len, align_to_bos=True)
inputs_train, targets_train, seqlens_train = next(train_loader)

model = GPT(GPTConfig(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768))
model = model.to(device)
for m in model.modules():
    if isinstance(m, (nn.Embedding, CastedLinear)):
        m.bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()

for param in model.parameters():
    dist.broadcast(param.detach(), 0)

from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
raw_model = model.module
assert isinstance(raw_model, nn.Module)

embed_params = [*raw_model.embed.parameters(), *raw_model.value_embeds.parameters()]
optimizer1 = torch.optim.Adam(embed_params, lr=0.6, betas=(0.8, 0.95))
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.008, betas=(0.8, 0.95))
params = list(raw_model.blocks.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95))
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

BLOCK_SIZE = 128
sliding_window_num_blocks = 1
training_time_ms = 0
torch.npu.synchronize()
t0 = time.perf_counter()

val_losses_record = []

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    frac_done = step / args.num_iterations
    sw_num_blocks = int(((1 - frac_done) * 64 + frac_done * 1792 + 64) // 128)
    sliding_window_num_blocks = sw_num_blocks

    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.npu.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        torch.npu.empty_cache()
        val_loader = distributed_data_generator(
            args.val_files,
            args.val_batch_tokens,
            -1,
            align_to_bos=False
        )
        val_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        with torch.no_grad():
            for _ in range(val_steps):
                inputs_val, targets_val, seqlens_val = next(val_loader)
                val_loss += model(inputs_val, targets_val, seqlens_val, sliding_window_num_blocks).float()
        del val_loader
        CausalSelfAttention.clear_mask_cache()
        torch.npu.empty_cache()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_loss_val = val_loss.item()
        val_losses_record.append({"step": step, "val_loss": val_loss_val})
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss_val:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
        model.train()
        torch.npu.synchronize()
        t0 = time.perf_counter()

    if last_step:
        break

    model.train()
    loss = model(inputs_train, targets_train, seqlens_train, sliding_window_num_blocks)
    loss.backward()
    del loss
    inputs_train, targets_train, seqlens_train = next(train_loader)

    frac = min(step/300, 1)
    for group in optimizer3.param_groups:
        group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

print0(
    f"peak memory allocated: {torch.npu.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.npu.max_memory_reserved() // 1024 // 1024} MiB",
    console=True
)
print0(f"total training time: {training_time_ms/1000:.2f}s ({training_time_ms:.0f}ms)", console=True)

if master_process and run_dir:
    metrics_path = os.path.join(run_dir, "metrics.json")
    final_val_loss = val_losses_record[-1]["val_loss"] if val_losses_record else None
    metrics = {
        "val_losses": val_losses_record,
        "final_val_loss": final_val_loss,
        "training_time_ms": training_time_ms,
        "num_iterations": args.num_iterations,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print0(f"Metrics saved to {metrics_path}", console=True)
    with open(os.path.join(run_dir, "final_val_loss.txt"), "w") as f:
        f.write(f"{final_val_loss}\n")

dist.destroy_process_group()
