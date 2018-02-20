#include "common.h"
#include "interface.h"
#include "sync_memory.h"

float dot_cpu(const float* vec, const int num) {
  float sum = 0.0f;
  for (int i = 0; i < num; ++i) {
    sum += vec[i] * vec[i];
  }
  return sum;
}

////////////////////////////////////////////////////////////////////////

static SyncMemory<float>* kVec = NULL;
static cublasHandle_t kHandle;

int init(int num) {
  kVec = new SyncMemory<float>(num);
  CUBLAS_CHECK(cublasCreate(&kHandle));
  return 0;
}

int process_cpu(char* input, int input_size, char* output, int output_size) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_EQ(kVec->size() * sizeof(float), input_size);
  CHECK_GE(output_size, 4);
  
  float sum = dot_cpu((float*) input, kVec->size());
  ((float*) output)[0] = sum;
  return 4;
}

int process_gpu(char* input, int input_size, char* output, int output_size) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_EQ(kVec->size() * sizeof(float), input_size);
  CHECK_GE(output_size, 4);

  float sum = 0.0f;
  kVec->set_cpu_data((float*) input, kVec->size());
  CUBLAS_CHECK(cublasSdot(
      kHandle, kVec->size(), kVec->gpu_data(), 1, kVec->gpu_data(), 1, &sum));
  ((float*) output)[0] = sum;
  return 4;
}

int release() {
  delete kVec;
  kVec = NULL;
  CUBLAS_CHECK(cublasDestroy(kHandle));
  return 0;
}
