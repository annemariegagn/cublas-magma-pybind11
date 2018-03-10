#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "magma_v2.h"
#include "magma_lapack.h" 


__host__ void run_kernelAmulB
(const float* Ain, const float* Bin, float* Cin, const float alpha, const float beta, int numRowA, 
 int numColA, int numRowB, int numColB) {

    cublasStatus_t *d_status;
    cublasStatus_t status;

    if (cudaMalloc((void **) &d_status, sizeof(cublasStatus_t)) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate d_status)\n");
        exit(EXIT_FAILURE);
    }

    cublasHandle_t cnpHandle;
    cublasStatus_t statuscublas = cublasCreate(&cnpHandle);


    // CALL CUBLAS KERNEL
    // lda, ldb, ldc, typical dimensions for Numpy array ordering
    // A[A_R][A_C] X B[B_R][B_C]=C[A_R][B_C]
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
    // A_R, B_C, A_C, 1.0, A, A_C, B, B_C, 0.0, C, B_C);
    int lda = numColA; int ldb = numColB; int ldc = numColB;
    status = cublasSgemm(cnpHandle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        numRowA, numColB, numColA,
                        &alpha,
                        Ain, lda,
                        Bin, ldb,
                        &beta,
                        Cin, ldc);

    cublasDestroy(cnpHandle);

    cudaError_t error;
    if ((error = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "!!!! kernel execution error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(&status, d_status, sizeof(cublasStatus_t), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device to host memory copy error\n");
        exit(EXIT_FAILURE);
    }

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS API call failed with code %d\n", status);
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_status) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (d_status)\n");
        exit(EXIT_FAILURE);
    }
}

//Ain, Bin, on device
__host__
void run_kernelLinearMagmaSolve(float* Ain, float* Bin, float* Xout, int numRowA, int numColA, int numColB) {
    
    printf("%s \n", "dans runKernewlLinearSolve");
    magma_init();
    magma_int_t info = 0;
    real_Double_t gpu_time;
    magma_trans_t trans = MagmaNoTrans;
    gpu_time = magma_sync_wtime(NULL);

    printf("%s \n", "en haut de magma");

    float gpu_error, gpu_perf, cpu_perf;
    float c_one = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    magma_int_t M = (magma_int_t)numRowA;
    magma_int_t N = (magma_int_t)numColA;
    magma_int_t min_mn = std::min(numRowA, numColA);
    magma_int_t max_mn = std::max(numRowA, numColA);
    magma_int_t nrhs = numColB;
    magma_int_t lda = numRowA;
    magma_int_t ldb = max_mn;
    magma_int_t size = lda * numColA;
    float *h_A2, *tau, *h_work, tmp[1];  
    
    magma_int_t ldda = ((numRowA + 31)/32)*32; //multiple of 32 by default
    magma_int_t lddb = ((max_mn + 31)/32)*32; //multiple of 32 by default
    magma_int_t nb = magma_get_sgeqrf_nb(numRowA, numColA);

    magma_int_t lworkgpu = (numRowA - numColA + nb)*(nrhs + nb) + nrhs*nb;

    //preparing gpu workspace
    
    magma_int_t lhwork = -1;
    lapackf77_sgeqrf(&M, &N, NULL, &M, NULL, tmp, &lhwork, &info);

    
    magma_sgels_gpu(trans, numRowA, numColA, numColB, Ain, ldda, Bin, lddb, Xout, lworkgpu, &info);

    gpu_time = magma_sync_wtime(NULL) - gpu_time;
    printf("magma_sgesv_gpu time: %7.5f sec. \n", gpu_time);
  
    magma_finalize();
}
