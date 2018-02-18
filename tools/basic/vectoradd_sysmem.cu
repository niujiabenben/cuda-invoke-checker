#include "common.h"
#include "sync_memory.h"

void add_host(const int* a, const int* b, int* c, const int N) {
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__
void add_device(const int* a, const int* b, int* c, const int N) {
  CUDA_KERNEL_LOOP(i, N) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  const int NUM = 100;
  SyncMemory<int> a(NUM), b(NUM), c(NUM);

  // Get data on host
  int* a_host = a.mutable_cpu_data();
  int* b_host = b.mutable_cpu_data();
  
  // Fill vector a & b with random values
  for (int i = 0; i < NUM; ++i) {
    a_host[i] = rand() % 1000;
    b_host[i] = rand() % 1000;
  }
  
  // Get data on device, automatically synchronized
  const int* a_device = a.gpu_data(); 
  const int* b_device = b.gpu_data();
  int* c_device = c.mutable_gpu_data();

  // Call "kernel" routine to execute on GPU
  CUKERNEL_CHECK((add_device<<<CUDA_GET_BLOCKS(NUM), CUDA_NUM_THREADS>>>(
      a_device, b_device, c_device, NUM)));

  // Call host code to execute on CPU
  int d_host[NUM];
  add_host(a_host, b_host, d_host, NUM);

  // Get vector c on host, automatically synchronized
  const int* c_host = c.cpu_data();

  // Check the results
  for (int i = 0; i < NUM; ++i) {
    CHECK_EQ(c_host[i], d_host[i])
        << "check failed at " << i << ": " << c_host[i] << " vs " << d_host[i];
  }
  LOG(INFO) << "check passed";

  return 0;
}
