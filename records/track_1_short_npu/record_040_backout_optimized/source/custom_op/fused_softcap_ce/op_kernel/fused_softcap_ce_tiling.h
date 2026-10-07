#pragma once

#include <cstdint>

struct FusedSoftcapCETilingData {
    uint64_t vocabSize;
    uint64_t frontCoreNum;
    uint64_t frontRows;
    uint64_t tailCoreNum;
    uint64_t tailRows;
    uint64_t tileSize;
    uint64_t tileLoops;
    uint64_t tileTail;
    int64_t ignoreIndex;
    float softcapScale;
    float softcapCap;
};
