cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(codex_fused_softcap_ce LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH)

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_INCLUDE_DIR "${ASCEND_HOME_PATH}/aarch64-linux/include")
set(ASCEND_LIB_DIR "${ASCEND_HOME_PATH}/aarch64-linux/lib64")

add_library(codex_fused_softcap_ce_ops SHARED
    op_kernel/fused_softcap_ce_fwd_kernel.asc
    op_kernel/fused_softcap_ce_bwd_kernel.asc
    op_extension/fused_softcap_ce_torch.cpp
    op_extension/register.cpp)

target_include_directories(codex_fused_softcap_ce_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ASCEND_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS})

target_link_directories(codex_fused_softcap_ce_ops PRIVATE
    ${ASCEND_LIB_DIR}
    ${TORCH_NPU_PATH}/lib)

target_link_libraries(codex_fused_softcap_ce_ops PRIVATE
    torch_npu
    ascendcl
    ascendc_runtime
    tiling_api
    register
    platform
    unified_dlog
    graph_base
    dl
    m)

target_compile_options(codex_fused_softcap_ce_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)

