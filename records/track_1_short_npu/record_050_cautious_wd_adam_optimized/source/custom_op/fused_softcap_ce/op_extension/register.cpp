#include <torch/library.h>
#include "ops.h"

TORCH_LIBRARY(codex_ops, m) {
    m.def("fused_softcap_ce_fwd(Tensor logits, Tensor target, float scale, float cap) -> (Tensor, Tensor, Tensor)");
    m.def("fused_softcap_ce_bwd(Tensor grad_loss, Tensor sigmoid_saved, Tensor log_prob, Tensor target, float scale, float cap) -> Tensor");
}

TORCH_LIBRARY_IMPL(codex_ops, PrivateUse1, m) {
    m.impl("fused_softcap_ce_fwd", &fused_softcap_ce_forward);
    m.impl("fused_softcap_ce_bwd", &fused_softcap_ce_backward);
}
