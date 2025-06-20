#pragma once

#include "configs.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 3: case_macro(3); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false && "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case CUDA_R_16BF: case_macro(nv_bfloat16); \
        case CUDA_R_32F:  case_macro(float); \
        default: EP_HOST_ASSERT(false && "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2560: case_macro(2560); \
        case 4096: case_macro(4096); \
        case 5120: case_macro(5120); \
        case 7168: case_macro(7168); \
        case 8192: case_macro(8192); \
        default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
    } while (false)

#define DISPATCH_HIDDEN_SIZE(hidden, kHidden, ...) \
  if (hidden == 7168) {                            \
    constexpr size_t kHidden = 7168;               \
    __VA_ARGS__                                    \
  } else if (hidden == 8192) {                     \
    constexpr size_t kHidden = 8192;               \
    __VA_ARGS__                                    \
  } else {                                         \
    EP_HOST_ASSERT(false && "Unsupported hidden"); \
  }

#define DISPATCH_NUM_TOPK(num_topk, kTopk, ...)     \
  if (num_topk == 8) {                              \
    constexpr int kTopk = 8;                        \
    __VA_ARGS__                                     \
  } else {                                          \
    EP_HOST_ASSERT(false && "Unsupported num_topk");\
  }

#define DISPATCH_RDMA_RANKS(num_rdma_ranks, kNumRdmaRanks, ...) \
  if (num_rdma_ranks == 1) {                                    \
    constexpr int kNumRdmaRanks = 1;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 2) {                             \
    constexpr int kNumRdmaRanks = 2;                            \
    __VA_ARGS__                                                 \
  } else if (num_rdma_ranks == 4) {                             \
    constexpr int kNumRdmaRanks = 4;                            \
    __VA_ARGS__                                                 \
  } else {                                                      \
    EP_HOST_ASSERT(false && "Unsupported num_rdma_ranks");      \
  }

#define DISPATCH_NUM_EXPERTS(num_experts, kNumExperts, ...) \
  if (num_experts == 64) {                                  \
    constexpr int kNumExperts = 64;                         \
    __VA_ARGS__                                             \
  } else {                                                  \
    EP_HOST_ASSERT(false && "num_experts num_topk");        \
  }

