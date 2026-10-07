"""Full validation for QKVO + band sparse + vertically packed embeddings."""
from pathlib import Path

source_path = Path(__file__).with_name("train_gpt_stage2_qkvo_band2048_full.py")
wrapper = source_path.read_text()
wrapper = wrapper.replace(
    "f\"/tmp/r037_s2_qkvo_band2048_full_{os.environ.get('LOCAL_RANK', '0')}\"",
    "f\"/tmp/r037_s2_qkvo_band_packed_embed_full_{os.environ.get('LOCAL_RANK', '0')}\"",
)
marker = 'exec(compile(source, str(source_path), "exec"), globals(), globals())'
packing = '''source = source.replace(
    "        ve = [value_embed(input_seq) for value_embed in self.value_embeds]\\n"
    "        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]\\n",
    "        packed_embed = F.embedding(input_seq.unsqueeze(0) + self.embedding_offsets, self.packed_embeddings)\\n"
    "        x, ve0, ve1, ve2 = packed_embed.unbind(0)\\n"
    "        ve = [None, ve1, ve2] + [None] * (len(self.blocks) - 6) + [ve0, ve1, ve2]\\n",
)
source = source.replace("        x = self.embed(input_seq)\\n", "")
model_anchor = \'''for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)
\'''
model_replacement = \'''for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
with torch.no_grad():
    packed_embeddings = torch.cat(
        [model.embed.weight] + [module.weight for module in model.value_embeds], dim=0
    )
model.packed_embeddings = nn.Parameter(packed_embeddings)
model.packed_embeddings.lr_mul = 75.0
model.embedding_offsets = nn.Buffer(
    torch.arange(4, dtype=torch.int64, device=device).unsqueeze(1) * model.embed.num_embeddings,
    persistent=False,
)
del model.embed
del model.value_embeds
for param in model.parameters():
    dist.broadcast(param.detach(), 0)
\'''
source = source.replace(model_anchor, model_replacement)
assert "self.packed_embeddings" in source
assert "model.packed_embeddings = nn.Parameter" in source
assert "self.embed(input_seq)" not in source
assert "num_iterations: int = 1640\\n" in source
exec(compile(source, str(source_path), "exec"), globals(), globals())'''
wrapper = wrapper.replace(marker, packing)
assert "model.packed_embeddings = nn.Parameter" in wrapper
exec(compile(wrapper, str(source_path), "exec"), globals(), globals())
