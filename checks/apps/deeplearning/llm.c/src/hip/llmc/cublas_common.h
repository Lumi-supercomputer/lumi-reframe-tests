/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#if defined(ENABLE_FP32)
#define CUBLAS_LOWP HIP_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP HIP_R_16F
#else // default to bfloat16
#define CUBLAS_LOWP HIP_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
hipblasComputeType_t cublas_compute = HIPBLAS_COMPUTE_32F;
hipblasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void cublasCheck(hipblasStatus_t status, const char *file, int line)
{
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H
