#pragma once

#ifdef MULTI_GPU
#include <mpi.h>
#include <rccl/rccl.h>
#endif

#include <hipblaslt/hipblaslt.h>
//#include <hip/hip_fp16.h>
//#include <hip/hip_bfloat16.h>
//#include <hip/amd_detail/amd_hip_bf16.h>

#define hipProfilerStart(x) hipSuccess
#define hipProfilerStop(x) hipSuccess
#define nvtxRangePush(x) {}
#define nvtxRangePop(x) {}
#define nvtxNameCudaStreamA(x,y) {}
#define __stcg(ptr, val) {*(ptr) = val;}
#define __stcs(ptr, val) {*(ptr) = val;}
#define nvtxNameCudaEventA(x,y) {}

static __device__ __forceinline__ hip_bfloat16 __float2bfloat16_rn(float f) {
    return hip_bfloat16::round_to_bfloat16(f);
}

static __device__ __forceinline__ float __bfloat162float(hip_bfloat16 f) {
    return static_cast<float>(f);
}

template <typename T>
static __device__ __forceinline__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_xor(var, laneMask, width);
}

template <typename T>
static __device__ __forceinline__ T __shfl_down_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_down(var, laneMask, width);
}

static __device__ __hip_bfloat16 __ldcs(const __hip_bfloat16* ptr) { return *ptr; }
static __device__ __forceinline__ int4 __ldcs(const int4 *addr) {
    const int *a = (const int *) addr;
    return make_int4(__builtin_nontemporal_load(a),
        __builtin_nontemporal_load(a+1),
        __builtin_nontemporal_load(a+2),
        __builtin_nontemporal_load(a+3));
}
