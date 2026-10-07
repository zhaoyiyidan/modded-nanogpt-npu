"""Full validation: safe stack + QKVO shared cast + band-sparse attention."""
from pathlib import Path

source_path = Path(__file__).with_name("train_gpt_short_stage2_combo_safe.py")
source = source_path.read_text()
source = source.replace(
    "f\"/tmp/r037_s2_combo_safe_{os.environ.get('LOCAL_RANK', '0')}\"",
    "f\"/tmp/r037_s2_qkvo_band2048_full_{os.environ.get('LOCAL_RANK', '0')}\"",
)
source = source.replace(
    "class SquaredReLU(torch.autograd.Function):",
    "@torch.compile(backend=\"npu\", dynamic=False)\n"
    "def compiled_attention_gate(y: Tensor, gate_logits: Tensor):\n"
    "    return y * torch.sigmoid(gate_logits).unsqueeze(-1)\n\n"
    "class SquaredReLU(torch.autograd.Function):",
)
source = source.replace(
    "class CastedLinear(nn.Linear):",
    "def npu_linear_nd(x: Tensor, weight: Tensor):\n"
    "    shape = x.shape\n"
    "    y = torch_npu.npu_linear(x.reshape(-1, shape[-1]), weight)\n"
    "    return y.reshape(*shape[:-1], weight.size(0))\n\n"
    "class CastedLinear(nn.Linear):",
)
source = source.replace(
    "return F.linear(x, self.weight.type_as(x))",
    "return npu_linear_nd(x, self.weight.type_as(x))",
)
source = source.replace(
    "q, k, v = F.linear(x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)",
    "qkvo_w = self.qkvo_w.type_as(x).view(4, self.hdim, self.dim)\n"
    "        q, k, v = npu_linear_nd(x, qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)",
)
source = source.replace(
    "x = F.linear(x, self.c_fc.T.type_as(x))",
    "x = npu_linear_nd(x, self.c_fc.T.type_as(x))",
)
source = source.replace(
    "y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)",
    "gate_logits = self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])\n"
    "        y = compiled_attention_gate(y, gate_logits)",
)
source = source.replace(
    "y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))",
    "y = F.linear(y, qkvo_w[3])",
)
source = source.replace(
    "        attn_mask = self._get_window_causal_mask(max_doc_len, bm_size, x.device)\n",
    "        attn_mask = self._get_window_causal_mask(args.train_max_seq_len, bm_size, x.device)\n",
)
source = source.replace(
    "            atten_mask=attn_mask,\n"
    "            sparse_mode=0,\n",
    "            atten_mask=attn_mask,\n"
    "            pre_tockens=bm_size - 1,\n"
    "            next_tockens=0,\n"
    "            sparse_mode=4,\n",
)
source = source.replace("num_iterations: int = 164\n", "num_iterations: int = 1640\n")
source = source.replace("iteration_extension: int = 4\n", "iteration_extension: int = 40\n")
source = source.replace("val_loss_every: int = 168\n", "val_loss_every: int = 125\n")

assert "def npu_linear_nd" in source
assert "def compiled_attention_gate" in source
assert "qkvo_w = self.qkvo_w.type_as(x)" in source
assert "y = F.linear(y, qkvo_w[3])" in source
assert "_get_window_causal_mask(args.train_max_seq_len, bm_size" in source
assert "pre_tockens=bm_size - 1" in source
assert "            sparse_mode=0,\n" not in source
assert "num_iterations: int = 1640" in source
assert "iteration_extension: int = 40" in source
assert "val_loss_every: int = 125" in source
exec(compile(source, str(source_path), "exec"), globals(), globals())
