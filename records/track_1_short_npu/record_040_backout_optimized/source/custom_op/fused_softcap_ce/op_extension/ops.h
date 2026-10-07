#pragma once

#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_softcap_ce_forward(
    const at::Tensor& logits, const at::Tensor& target, double scale, double cap);

at::Tensor fused_softcap_ce_backward(
    const at::Tensor& grad_loss, const at::Tensor& sigmoid_saved,
    const at::Tensor& log_prob, const at::Tensor& target,
    double scale, double cap);
