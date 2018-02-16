#ifndef CUDA_INVOKE_CHECKER_SYNC_MEMORY_H_
#define CUDA_INVOKE_CHECKER_SYNC_MEMORY_H_

#include "common.h"


///////////////////////////////////////////////////////////////////
// class SyncMemory synchronizes the memory between CPU and GPU.
///////////////////////////////////////////////////////////////////
template <class Dtype>
class SyncMemory {
 public:
  enum DataLoc {DATA_AT_NONE, DATA_AT_CPU, DATA_AT_GPU, DATA_AT_BOTH};
  explicit SyncMemory(const int size = 0)
      : data_loc_(DATA_AT_NONE),
        cpu_data_(NULL),
        gpu_data_(NULL),
        size_(size) { }
  ~SyncMemory() { clear(); }

  int size() const { return size_; }
  const Dtype* cpu_data() {
    to_cpu();
    return cpu_data_;
  }
  const Dtype* gpu_data() {
    to_gpu();
    return gpu_data_;
  }
  Dtype* mutable_cpu_data() {
    to_cpu();
    data_loc_ = DATA_AT_CPU;
    return cpu_data_;
  }
  Dtype* mutable_gpu_data() {
    to_gpu();
    data_loc_ = DATA_AT_GPU;
    return gpu_data_;
  }
  void set_cpu_data(const Dtype* data, const int size) {
    resize(size);
    const int bytes = sizeof(Dtype) * size;
    CUDA_CHECK(cudaMemcpy(mutable_cpu_data(), data, bytes, cudaMemcpyHostToHost));
  }
  void resize(const int size) {
    if (size != size_) {
      clear();
      size_ = size;
    }
  }

 private:
  void clear();
  void to_cpu();
  void to_gpu();

  DataLoc data_loc_;
  Dtype* cpu_data_;
  Dtype* gpu_data_;
  int size_;
  DISALLOW_COPY_AND_ASSIGN(SyncMemory);
};

template <class Dtype>
void SyncMemory<Dtype>::clear() {
  if ((data_loc_ == DATA_AT_CPU) || (data_loc_ == DATA_AT_BOTH)) {
    CUDA_CHECK(cudaFreeHost(cpu_data_));
    cpu_data_ = NULL;
  }
  if ((data_loc_ == DATA_AT_GPU) || (data_loc_ == DATA_AT_BOTH)) {
    CUDA_CHECK(cudaFree(gpu_data_));
    gpu_data_ = NULL;
  }
  data_loc_ = DATA_AT_NONE;
  size_ = 0;
}

template <class Dtype>
void SyncMemory<Dtype>::to_cpu() {
  const int bytes = size_ * sizeof(Dtype);
  switch (data_loc_) {
    case DATA_AT_NONE:
      CUDA_CHECK(cudaMallocHost(&cpu_data_, bytes));
      memset(cpu_data_, 0, bytes);
      data_loc_ = DATA_AT_CPU;
      break;
    case DATA_AT_GPU:
      if (cpu_data_ == NULL) {
        CUDA_CHECK(cudaMallocHost(&cpu_data_, bytes));
      }
      CUDA_CHECK(cudaMemcpy(cpu_data_, gpu_data_, bytes,
                            cudaMemcpyDeviceToHost));
      data_loc_ = DATA_AT_BOTH;
      break;
    case DATA_AT_BOTH:
    case DATA_AT_CPU:
      break;
  };
}

template <class Dtype>
void SyncMemory<Dtype>::to_gpu() {
  const int bytes = size_ * sizeof(Dtype);
  switch (data_loc_) {
    case DATA_AT_NONE:
      CUDA_CHECK(cudaMalloc(&gpu_data_, bytes));
      CUDA_CHECK(cudaMemset(gpu_data_, 0, bytes));
      data_loc_ = DATA_AT_GPU;
      break;
    case DATA_AT_CPU:
      if (gpu_data_ == NULL) {
        CUDA_CHECK(cudaMalloc(&gpu_data_, bytes));
      }
      CUDA_CHECK(cudaMemcpy(gpu_data_, cpu_data_, bytes,
                            cudaMemcpyHostToDevice));
      data_loc_ = DATA_AT_BOTH;
      break;
    case DATA_AT_BOTH:
    case DATA_AT_GPU:
      break;
  };
}

#endif  // CUDA_INVOKE_CHECKER_SYNC_MEMORY_H_
