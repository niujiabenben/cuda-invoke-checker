#include "common.h"

void add_host(const int* a, const int* b, int* c, const int N) {
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__
void add_device(const int* a, const int* b, int* c, const int N) {
  if (threadIdx.x < N) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
  }
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  const int NUM = 100;

  // Allocate memory on host
  int* a_host = new int[NUM];
  int* b_host = new int[NUM];
  int* c_host = new int[NUM];
  int* d_host = new int[NUM];

  // Fill vector a & b with random values
  for (int i = 0; i < NUM; ++i) {
    a_host[i] = rand() % 1000;
    b_host[i] = rand() % 1000;
  }
  
  // Allocate memory on device
  const int bytes = sizeof(int) * NUM;
  int* a_device = NULL;
  int* b_device = NULL;
  int* c_device = NULL;
  CUDA_CHECK(cudaMalloc(&a_device, bytes));
  CUDA_CHECK(cudaMalloc(&b_device, bytes));
  CUDA_CHECK(cudaMalloc(&c_device, bytes));
  
  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(a_device, a_host, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_device, b_host, bytes, cudaMemcpyHostToDevice));

  // Call "kernel" routine to execute on GPU
  CUKERNEL_CHECK((add_device<<<1, NUM>>>(a_device, b_device, c_device, NUM)));

  // copy result from device to host
  CUDA_CHECK(cudaMemcpy(d_host, c_device, bytes, cudaMemcpyDeviceToHost));

  // Call host code to execute on CPU
  add_host(a_host, b_host, c_host, NUM);

  // Check the results
  for (int i = 0; i < NUM; ++i) {
    CHECK_EQ(c_host[i], d_host[i])
        << "check failed at " << i << ": " << c_host[i] << " vs " << d_host[i];
  }
  LOG(INFO) << "check passed";

  // Free memory on host
  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] d_host;  

  // free memset on device
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  CUDA_CHECK(cudaFree(c_device));

  return 0;
}
