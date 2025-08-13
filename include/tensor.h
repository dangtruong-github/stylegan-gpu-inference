#pragma once

#include <vector>
#include <cstdio>
#include <cstddef>
#include <string>
#include <cstring> // For memcpy and memset
#include <cassert> // For assert
#include <stdlib.h>  // For posix_memalign and free

#include "half.hpp" /* for half on CPU ('half_cpu') */
#include "cuda_fp16.h" /* for half on GPU ('half') */

// Define the alignment boundary. 64 bytes is a good value for AVX/AVX-512.
#define TENSOR_ALIGNMENT 64
#define BATCH_SIZE 32
#define NUM_GPUS 4

using std::vector;

/* Namespace for half on CPU ('half_cpu') */
typedef half_float::half half_cpu;
using namespace half_float::literal; 

double get_time_kernel();

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


// #define FP16 /* [Advanced] Uncomment this line only for FP16 */

/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[5] = {1, 1, 1, 1, 1};
  float *buf = nullptr;
  float *d_buf[NUM_GPUS] = {};  // Device buffers for fp32 (one per GPU)
  bool malloc_success = false;
  bool malloc_copy = false;

  #ifdef FP16
    half *d_buf_fp16[NUM_GPUS] = {};  // Device buffers for fp16 (one per GPU)
  #endif

  Tensor(const vector<size_t> &shape_, bool malloc_copy_, cudaStream_t *streams);
  Tensor(const vector<size_t> &shape_, float *buf_, bool malloc_copy_, cudaStream_t *streams);
  Tensor(const vector<size_t> &shape_, float *buf_, bool batch, bool malloc_copy_, cudaStream_t *streams);
  ~Tensor();

  size_t num_elem();
  void reshape(const vector<size_t> &shape_);
  void printShape(const std::string& descr);

  // Helper functions for aligned memory management
  void* aligned_alloc(size_t size);
  void aligned_free(void* ptr);

  void replicate_to_all_devices();
  void malloc_device();
  void to_device(cudaStream_t *streams);
  void from_device(cudaStream_t *streams);
  void free_device();

#ifdef FP16
  void replicate_to_all_devices_fp16();
  void malloc_device_fp16();
  void to_device_fp16(cudaStream_t *streams);
  void from_device_fp16(cudaStream_t *streams);
  void free_device_fp16();
#endif

};

typedef Tensor Parameter;
typedef Tensor Activation;