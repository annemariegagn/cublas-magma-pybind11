#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "cublas_v2.h"
#include "magma_v2.h"
#include "magma.h"
#include "magma_lapack.h"
#include <cuda_runtime.h>


//size is number of elements
//void run_kernel
//(double *vec1, double *vec2, double scalar, int size);

void run_kernelLinearMagmaSolve
(float* Ain, float* Bin, float* Xout, int numRowA, int numColA, int numColB);

void run_kernelAmulB
(const float* Ain, const float* Bin, float* Cin, const float alpha, const float beta, int numRowA,  int numColA, int numRowB, int numColB);

//Multiply input matrices A, B on GPU. 
//Dimensions must be verified in advance. No broadcast. Will raise an error. 
pybind11::array_t<float> 
multiply_matrices(pybind11::array_t<float> A, pybind11::array_t<float> B, float alpha, float beta, int numRowA, int numColA, int numRowB, int numColB) {
    
    float *gpu_ptrA;
    float *gpu_ptrB;
    float *gpu_ptrC;
    int nbelemA = numRowA * numColA;
    int nbelemB = numRowB * numColB;
    int nbelemC = numRowA * numColB;
    cudaError_t errorA = cudaMalloc(&gpu_ptrA, nbelemA * sizeof(float));
    cudaError_t errorB = cudaMalloc(&gpu_ptrB, nbelemB * sizeof(float));
    cudaError_t errorC = cudaMalloc(&gpu_ptrC, nbelemC * sizeof(float));

    if (errorA != cudaSuccess) {
      std::stringstream strstr;
      strstr << "error with memory allocation on the GPU!" << std::endl;
      throw std::runtime_error(cudaGetErrorString(errorA));
    }
    if (errorB != cudaSuccess) {
      std::stringstream strstr;
      strstr << "error with memory allocation on the GPU!" << std::endl;
      throw std::runtime_error(cudaGetErrorString(errorB));
    }
    if (errorC != cudaSuccess) {
      std::stringstream strstr;
      strstr << "error with memory allocation on the GPU!" << std::endl;
      throw std::runtime_error(cudaGetErrorString(errorC));
    }

    auto haA = A.request(); //request un array numpy
    auto haB = B.request(); //request un array numpy

    float* ptrA = reinterpret_cast<float*>(haA.ptr);
    float* ptrB = reinterpret_cast<float*>(haB.ptr);
    errorA = cudaMemcpy(gpu_ptrA, ptrA, nbelemA * sizeof(float), cudaMemcpyHostToDevice);
    errorB = cudaMemcpy(gpu_ptrB, ptrB, nbelemB * sizeof(float), cudaMemcpyHostToDevice);
    if (errorA != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(errorA));
    }
    if (errorB != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(errorB));
    }
    if (errorC != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(errorC));
    }

    run_kernelAmulB(gpu_ptrA, gpu_ptrB, gpu_ptrC, alpha, beta, numRowA, numColA, numRowB, numColB);

    // ptC hold result
    float* ptrC = (float *)malloc(nbelemC * sizeof(float));
    errorC = cudaMemcpy(ptrC, gpu_ptrC, nbelemC * sizeof(float), cudaMemcpyDeviceToHost);
    if (errorC != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(errorC));
    }

    errorA = cudaFree(gpu_ptrA);
    errorB = cudaFree(gpu_ptrB);
    if (errorA != cudaSuccess) {
        std::stringstream strstr;
        strstr << "erreur!" << std::endl;
        throw std::runtime_error(cudaGetErrorString(errorA));
    }
    if (errorB != cudaSuccess) {
        std::stringstream strstr;
        strstr << "erreur!" << std::endl;
        throw std::runtime_error(cudaGetErrorString(errorB));
    }

    return pybind11::array(nbelemC, ptrC);
}

//solve least-sqaure general problem min_X ||A*X - B|| on GPU, where A is M x N matrix, M >= N, B is an M x nrhs matrix.
//Solution X overwrites B. MagmaNoTrans, only option supported. 

pybind11::array_t<float> 
lin_solver(pybind11::array_t<float> A, pybind11::array_t<float> B, int numRowA, int numColA, int numColB) {

    float *gpu_ptrA;
    float *gpu_ptrB;
    float *gpu_ptrX;
    int nbelemA = numRowA * numColA;
    int nbelemB = numRowA * numColB;
    cudaError_t errorA = cudaMalloc(&gpu_ptrA, nbelemA * sizeof(float));
    cudaError_t errorB = cudaMalloc(&gpu_ptrB, nbelemB * sizeof(float));

    if (errorA != cudaSuccess) {
        std::stringstream strstr;
        strstr << "error with memory allocation on the GPU!" << std::endl;
        throw std::runtime_error(cudaGetErrorString(errorA));
    }
    if (errorB != cudaSuccess) {
        std::stringstream strstr;
        strstr << "error with memory allocation on the GPU!" << std::endl;
        throw std::runtime_error(cudaGetErrorString(errorB));
    }

    auto haA = A.request(); 
    auto haB = B.request(); 

    float* ptrA = reinterpret_cast<float*>(haA.ptr);
    float* ptrB = reinterpret_cast<float*>(haB.ptr);
    
    //Copy to device of input arrays
    errorA = cudaMemcpy(gpu_ptrA, ptrA, nbelemA * sizeof(float), cudaMemcpyHostToDevice);
    errorB = cudaMemcpy(gpu_ptrB, ptrB, nbelemB * sizeof(float), cudaMemcpyHostToDevice);

    printf("%s \n", "cudaMemCpy of arrays on GPU: done");
    if (errorA != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(errorA));
    }
    if (errorB != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(errorB));
    }
  
    magma_init();
    magma_int_t info = 0;
    real_Double_t gpu_time;
    magma_trans_t trans = MagmaNoTrans;
    gpu_time = magma_sync_wtime(NULL);

    magma_int_t M = (magma_int_t)numRowA;
    magma_int_t N = (magma_int_t)numColA;
    magma_int_t min_mn = std::min(numRowA, numColA);
    magma_int_t max_mn = std::max(numRowA, numColA);
    magma_int_t nrhs = numColB;
    magma_int_t lda = numRowA;
    magma_int_t ldb = max_mn;
    magma_int_t size = lda * numColA;
    float *h_work, tmp[1];  
    
    magma_int_t ldda = ((numRowA + 31)/32)*32; //multiple of 32 by default (precision float32)
    magma_int_t lddb = ((max_mn + 31)/32)*32; //multiple of 32 by default
    magma_int_t nb = magma_get_sgeqrf_nb(numRowA, numColA);

    printf("%s %d %d %d %d\n", "ldda, lddb, nb, nrhs", ldda, lddb, nb, nrhs);
    
    magma_int_t lworkgpu = (numRowA - numColA + nb)*(nrhs + nb) + nrhs*nb;

    printf("%s %d\n", "lworkgpu", lworkgpu);

    //preparing gpu workspace
    magma_int_t lhwork = -1;
    lapackf77_sgeqrf(&M, &N, NULL, &M, NULL, tmp, &lhwork, &info);

    magma_int_t lhwork2 = (magma_int_t) MAGMA_S_REAL(tmp[0]);
    lhwork = -1;
    lhwork = std::max(std::max(lhwork, lhwork2), lworkgpu);
    printf("%s %d\n", "lhwork", lhwork);

    //workspace memory allocation for magma
    magma_smalloc_cpu(&h_work, lhwork);
    //magma_smalloc_cpu(&h_work, numColA * numColB);
    
    printf("%s \n", "Call of magma least-square solver");
    magma_sgels_gpu(trans, M, N, nrhs, gpu_ptrA, ldda, gpu_ptrB, lddb, h_work, lworkgpu, &info);
    
    if (info != 0) {
        printf("magma_sgels3_gpu returned error %lld: %s.\n",
        (long long) info, magma_strerror(info));
    }

    gpu_time = magma_sync_wtime(NULL) - gpu_time;
    printf("magma_sgels_gpu time: %7.5f sec. \n", gpu_time);
    
    //transfert solution to array b
    float *b;
    magma_smalloc_cpu(&b , N*nrhs);
    magma_sgetmatrix(N, nrhs, gpu_ptrB, lddb, b, N);
    cudaFree(gpu_ptrA);
    cudaFree(gpu_ptrB);
    return pybind11::array(N*nrhs, b);
}


pybind11::array_t<float> 
solve_cholesky(pybind11::array_t<float> Q, pybind11::array_t<float> z, int numRowQ, int numRowz, int numColz) {

    
}



PYBIND11_MODULE(gpu_library, m)
{
  
  //return array fully owned by Python
  m.def("lin_solver", &lin_solver, pybind11::return_value_policy::move, 
    pybind11::arg("A").noconvert(),
    pybind11::arg("B").noconvert(),
    pybind11::arg("numRowA"),
    pybind11::arg("numColA"),
    pybind11::arg("numColB")
    );
  //return array fully owned by Python
  m.def("multiply_matrices", &multiply_matrices, pybind11::return_value_policy::move,
    pybind11::arg("A").noconvert(),
    pybind11::arg("B").noconvert(),
    pybind11::arg("alpha"),
    pybind11::arg("beta"),
    pybind11::arg("numRowA"),
    pybind11::arg("numColA"),
    pybind11::arg("numRowB"),
    pybind11::arg("numColB")
    );
  m.def("solve_cholesky", &solve_cholesky, pybind11::return_value_policy::move,
    pybind11::arg("Q").noconvert(),
    pybind11::arg("z").noconvert(),
    pybind11::arg("numRowQ"),
    pybind11::arg("numRowz"),
    pybind11::arg("numColz")
    );


}



