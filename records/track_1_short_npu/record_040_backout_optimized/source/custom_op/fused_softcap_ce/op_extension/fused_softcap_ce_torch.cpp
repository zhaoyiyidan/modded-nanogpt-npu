#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "ops.h"
#include "../op_kernel/fused_softcap_ce_tiling.h"

void fused_softcap_ce_fwd_kernel(
    uint32_t blockDim, void* l2Ctrl, void* stream,
    uint8_t* logits, uint8_t* target, uint8_t* loss, uint8_t* logProb,
    uint8_t* sigmoidSaved, uint8_t* workspace, uint8_t* tiling);

void fused_softcap_ce_bwd_kernel(
    uint32_t blockDim, void* l2Ctrl, void* stream,
    uint8_t* gradLoss, uint8_t* sigmoidSaved, uint8_t* logProb, uint8_t* target,
    uint8_t* gradLogits, uint8_t* tiling);

namespace {
constexpr int64_t kVocabSize = 50304;
constexpr int64_t kTileSize = 8192;
constexpr double kExpectedScale = 0.13333333333333333;
// try_96 creates scale with logits.new_tensor(), so the runtime value used by
// its BF16 softcap graph is the BF16-rounded representation of 2/15.
constexpr float kBf16Scale = 0.1337890625f;
constexpr double kExpectedCap = 30.0;

struct PersistentResources {
    at::Tensor tiling;
    at::Tensor workspace;
    uint32_t block_num;
};

using ResourceMap = std::unordered_map<std::string, PersistentResources>;

ResourceMap& resource_map() {
    static auto* resources = new ResourceMap();
    return *resources;
}

std::mutex& resource_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

void validate_inputs(const at::Tensor& logits, const at::Tensor& target, double scale, double cap) {
    TORCH_CHECK(logits.is_privateuseone(), "logits must be on an NPU");
    TORCH_CHECK(target.is_privateuseone(), "target must be on an NPU");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be BF16");
    TORCH_CHECK(target.scalar_type() == at::kLong, "target must be INT64");
    TORCH_CHECK(logits.dim() == 2, "logits must have shape [N, V]");
    TORCH_CHECK(logits.size(1) == kVocabSize, "only V=50304 is supported");
    TORCH_CHECK(target.dim() == 1 && target.size(0) == logits.size(0), "target must have shape [N]");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(std::abs(scale - kExpectedScale) < 1e-12, "softcap scale must be 1/7.5");
    TORCH_CHECK(std::abs(cap - kExpectedCap) < 1e-12, "softcap cap must be 30");
}

PersistentResources& get_resources(const at::Tensor& logits, double scale, double cap) {
    int32_t device = -1;
    TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "aclrtGetDevice failed");
    const int64_t rows = logits.size(0);
    const std::string key = std::to_string(device) + ":" + std::to_string(rows);

    std::lock_guard<std::mutex> guard(resource_mutex());
    auto& resources = resource_map();
    auto found = resources.find(key);
    if (found != resources.end()) {
        return found->second;
    }

    int64_t vector_cores = 0;
    TORCH_CHECK(
        aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_cores) == ACL_SUCCESS,
        "failed to query vector core count");
    TORCH_CHECK(vector_cores > 0, "device reported no vector cores");
    uint32_t block_num = static_cast<uint32_t>(std::min<int64_t>(rows, vector_cores));

    FusedSoftcapCETilingData host{};
    host.vocabSize = kVocabSize;
    const uint64_t rows_per_core = rows / block_num;
    const uint64_t remainder = rows % block_num;
    if (remainder == 0) {
        host.frontCoreNum = block_num;
        host.frontRows = rows_per_core;
        host.tailCoreNum = 0;
        host.tailRows = 0;
    } else {
        host.frontCoreNum = remainder;
        host.frontRows = rows_per_core + 1;
        host.tailCoreNum = block_num - remainder;
        host.tailRows = rows_per_core;
    }
    host.tileSize = kTileSize;
    host.tileLoops = kVocabSize / kTileSize;
    host.tileTail = kVocabSize % kTileSize;
    host.ignoreIndex = -100;
    host.softcapScale = kBf16Scale;
    host.softcapCap = static_cast<float>(cap);

    PersistentResources value;
    value.block_num = block_num;
    value.tiling = at::empty(
        {static_cast<int64_t>(sizeof(FusedSoftcapCETilingData))},
        logits.options().dtype(at::kByte));
    value.workspace = at::empty({64}, logits.options().dtype(at::kFloat));
    TORCH_CHECK(
        aclrtMemcpy(
            value.tiling.mutable_data_ptr(), sizeof(FusedSoftcapCETilingData),
            &host, sizeof(FusedSoftcapCETilingData), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        "failed to copy tiling data to NPU");

    return resources.emplace(key, std::move(value)).first->second;
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_softcap_ce_forward(
    const at::Tensor& logits, const at::Tensor& target, double scale, double cap)
{
    validate_inputs(logits, target, scale, cap);
    auto& resources = get_resources(logits, scale, cap);
    at::Tensor loss = at::empty({}, logits.options());
    at::Tensor log_prob = at::empty_like(logits);
    at::Tensor sigmoid_saved = at::empty_like(logits);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);

    fused_softcap_ce_fwd_kernel(
        resources.block_num, nullptr, reinterpret_cast<void*>(stream),
        reinterpret_cast<uint8_t*>(logits.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(target.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(loss.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(log_prob.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(sigmoid_saved.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(resources.workspace.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(resources.tiling.mutable_data_ptr()));
    return {loss, log_prob, sigmoid_saved};
}

at::Tensor fused_softcap_ce_backward(
    const at::Tensor& grad_loss, const at::Tensor& sigmoid_saved,
    const at::Tensor& log_prob, const at::Tensor& target,
    double scale, double cap)
{
    validate_inputs(sigmoid_saved, target, scale, cap);
    TORCH_CHECK(grad_loss.is_privateuseone() && grad_loss.numel() == 1, "grad_loss must be an NPU scalar");
    TORCH_CHECK(grad_loss.scalar_type() == at::kBFloat16, "grad_loss must be BF16");
    TORCH_CHECK(log_prob.is_privateuseone() && log_prob.scalar_type() == at::kBFloat16,
                "log_prob must be an NPU BF16 tensor");
    TORCH_CHECK(log_prob.sizes() == sigmoid_saved.sizes() && log_prob.is_contiguous(),
                "log_prob must be contiguous and match sigmoid_saved");

    auto& resources = get_resources(sigmoid_saved, scale, cap);
    at::Tensor grad_logits = at::empty_like(sigmoid_saved);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    fused_softcap_ce_bwd_kernel(
        resources.block_num, nullptr, reinterpret_cast<void*>(stream),
        reinterpret_cast<uint8_t*>(grad_loss.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(sigmoid_saved.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(log_prob.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(target.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(grad_logits.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(resources.tiling.mutable_data_ptr()));
    return grad_logits;
}
