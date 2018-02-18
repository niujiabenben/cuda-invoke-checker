#ifndef CUDA_INVOKE_CHECKER_COMMON_H_
#define CUDA_INVOKE_CHECKER_COMMON_H_

#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

//// This must be placed after all cuda-related headers.
#include <helper_cuda.h>

//// Disallow copy and assignment operations for a class.
#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(classname) \
  private:                                  \
  classname(const classname&);              \
  classname& operator=(const classname&)
#endif

//// CUDA: common function checker
#ifndef CUDA_CHECK
#define CUDA_CHECK(condition)                 \
  do {                                        \
    cudaError_t error = condition;            \
    CHECK_EQ(error, cudaSuccess)              \
        << " " << _cudaGetErrorEnum(error);   \
  } while (0)
#endif

//// CUDA: cuBLAS function checker
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(condition)                \
  do {                                         \
    cublasStatus_t status = condition;         \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)    \
        << " " << _cudaGetErrorEnum(status);   \
  } while (0)
#endif

//// CUDA: cuRAND function checker
#ifndef CURAND_CHECK
#define CURAND_CHECK(condition)                \
  do {                                         \
    curandStatus_t status = condition;         \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS)    \
        << " " << _cudaGetErrorEnum(status);   \
  } while (0)
#endif

//// CUDA: grid stride looping
#ifndef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)
#endif

//// CUDA: loop helper 2D
#define CUDA_GET_INDEX_2D_0(index, dimen0, dimen1) \
  ((index) / (dimen1))
#define CUDA_GET_INDEX_2D_1(index, dimen0, dimen1) \
  ((index) % (dimen1))

//// CUDA: loop helper 3D
#define CUDA_GET_INDEX_3D_0(index, dimen0, dimen1, dimen2) \
  ((index) / ((dimen1) * (dimen2)))
#define CUDA_GET_INDEX_3D_1(index, dimen0, dimen1, dimen2) \
  (((index) / (dimen2)) % (dimen1))
#define CUDA_GET_INDEX_3D_2(index, dimen0, dimen1, dimen2) \
  ((index) % (dimen2))

//// CUDA: thread number configuration.
const int CUDA_NUM_THREADS = 1024;

//// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  int cuda_max_blocks = 65535;
  int required_blocks = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return std::min(required_blocks, cuda_max_blocks);
}

#endif  // CUDA_INVOKE_CHECKER_COMMON_H_
