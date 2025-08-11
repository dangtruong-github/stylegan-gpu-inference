#include "model.h"
#include <cassert> // For assert

double get_time_kernel() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data type.
 * @member - buf   : Data buffer containing elements, aligned to TENSOR_ALIGNMENT
 * @member - shape : Shape of tensor from outermost dimension to innermost dimension
 * e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */

// Helper function for aligned allocation using posix_memalign
void* Tensor::aligned_alloc(size_t size) {
    void* ptr = nullptr;
    // posix_memalign returns 0 on success.
    if (posix_memalign(&ptr, TENSOR_ALIGNMENT, size) != 0) {
        ptr = nullptr;
    }
    return ptr;
}

// Helper function for freeing aligned memory
void Tensor::aligned_free(void* ptr) {
    // Memory from posix_memalign is freed with standard free().
    free(ptr);
}

// Constructor to allocate a zero-initialized tensor
Tensor::Tensor(const std::vector<size_t> &shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    size_t bytes = N_ * sizeof(float);
    
    buf = (float *) aligned_alloc(bytes);
    if (buf) {
        memset(buf, 0, bytes); // Explicitly zero the allocated memory
    }
}

// Constructor to allocate and copy from an existing buffer
Tensor::Tensor(const std::vector<size_t> &shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    size_t bytes = N_ * sizeof(float);
    
    buf = (float *) aligned_alloc(bytes);
    if (buf && buf_) {
        memcpy(buf, buf_, bytes);
    }
}

/* * Allocate batch by copying a single item along the batch dimension
* i.e., shape=(2, 512, 4, 4) and buf_=(1, 512, 4, 4) -> allocate and copy buf_ twice.
*/
Tensor::Tensor(const std::vector<size_t> &shape_, float *buf_, bool batch) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    
    size_t N_ = num_elem();
    size_t bytes = N_ * sizeof(float);
    
    buf = (float *) aligned_alloc(bytes);
    
    if (buf && buf_) {
        size_t single_item_elements = N_ / shape[0];
        size_t single_item_bytes = single_item_elements * sizeof(float);

        for (size_t i = 0; i < shape[0]; ++i) {
            float* destination_pointer = buf + (i * single_item_elements);
            memcpy(destination_pointer, buf_, single_item_bytes);
        }
    }
}

// Destructor using the aligned free function
Tensor::~Tensor() {
    if (buf != nullptr) {
        aligned_free(buf);
        buf = nullptr;
    }
    free_device();  // Free fp32 device buffer

#ifdef FP16
    free_device_fp16();  // Free fp16 device buffer
#endif
}

size_t Tensor::num_elem() {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
    return size;
}

void Tensor::reshape(const std::vector<size_t> &shape_) {
    // Note: This reshape assumes the total number of elements remains the same.
    // It does not reallocate memory.
    size_t n = 1;
    ndim = shape_.size(); // ndim<=5
    for (size_t i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
        n *= shape[i];
    }
    
    // Check that the batch dimension is divisible by 4
    bool has_buffer = d_buf[0] != nullptr;

    #ifdef FP16
        has_buffer = has_buffer || d_buf_fp16[0] != nullptr;
    #endif

    if (has_buffer) {
        assert(shape[0] % NUM_GPUS == 0 && "Batch dimension must be divisible by NUM_GPUS");
    }
}

void Tensor::printShape(const std::string& descr) {
    printf("Shape of %s tensor: ", descr.c_str());
    for (size_t i = 0; i < ndim; i++) {
        printf("%zu ", shape[i]);
    }
    printf("\n");
}

/* GPU ALLOCATION AND FREE */ 

// Allocate device memory (fp32) for 4 GPUs
void Tensor::malloc_device() {
    assert(shape[0] % NUM_GPUS == 0 && "Batch dimension must be divisible by NUM_GPUS");

    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / NUM_GPUS;
    size_t bytes_per_gpu = per_gpu_elements * sizeof(float);

    for (int i = 0; i < NUM_GPUS; i++) {
        if (d_buf[i] != nullptr) continue; // Skip if already allocated

        // Set the current CUDA device
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)&d_buf[i], bytes_per_gpu));
    }

    malloc_success = true;
}

// Copy host to device (fp32) for 4 GPUs
void Tensor::to_device(cudaStream_t *streams) {
    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / NUM_GPUS;
    size_t bytes_per_gpu = per_gpu_elements * sizeof(float);

    // Ensure device memory is allocated
    // malloc_device();

    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        // Copy the segment of host buffer for this GPU
        float* host_segment = buf + i * per_gpu_elements;
        CHECK_CUDA(cudaMemcpy(d_buf[i], host_segment, bytes_per_gpu, cudaMemcpyHostToDevice));
    }
}

// Copy device to host (fp32) for 4 GPUs
void Tensor::from_device(cudaStream_t *streams) {
    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / NUM_GPUS;
    size_t bytes_per_gpu = per_gpu_elements * sizeof(float);

    for (int i = 0; i < NUM_GPUS; i++) {
        if (d_buf[i] == nullptr) continue; // Skip if not allocated

        CHECK_CUDA(cudaSetDevice(i));
        // Copy the segment to the host buffer for this GPU
        float* host_segment = buf + i * per_gpu_elements;
        CHECK_CUDA(cudaMemcpy(host_segment, d_buf[i], bytes_per_gpu, cudaMemcpyDeviceToHost));
    }
}

// Free device memory (fp32) for 4 GPUs
void Tensor::free_device() {
    for (int i = 0; i < NUM_GPUS; i++) {
        if (d_buf[i] != nullptr) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaFree(d_buf[i]));
            d_buf[i] = nullptr;
        }
    }

    malloc_success = false;
}

#ifdef FP16
// Allocate device memory (fp16) for 4 GPUs
void Tensor::malloc_device_fp16() {
    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / NUM_GPUS;
    size_t bytes_per_gpu = per_gpu_elements * sizeof(half);

    for (int i = 0; i < NUM_GPUS; i++) {
        if (d_buf_fp16[i] != nullptr) continue; // Skip if already allocated

        // Set the current CUDA device
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)&d_buf_fp16[i], bytes_per_gpu));
    }
}

// Convert and copy host (fp32) to device (fp16) for 4 GPUs
void Tensor::to_device_fp16(cudaStream_t *streams) {
    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / 4;

    // Ensure device memory is allocated
    // malloc_device_fp16();

    for (int i = 0; i < 4; i++) {
        float* host_segment = buf + i * per_gpu_elements;
        
        // Convert host segment to fp16
        half* temp_fp16 = new half[per_gpu_elements];
        for (size_t j = 0; j < per_gpu_elements; j++) {
            temp_fp16[j] = __float2half(host_segment[j]);
        }

        // Copy to device
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(d_buf_fp16[i], temp_fp16, per_gpu_elements * sizeof(half), cudaMemcpyHostToDevice));
        
        delete[] temp_fp16;
    }
}

// Copy and convert device (fp16) to host (fp32) for 4 GPUs
void Tensor::from_device_fp16(cudaStream_t *streams) {
    assert(shape[0] % NUM_GPUS == 0 && "Batch dimension must be divisible by NUM_GPUS");

    size_t total_elements = num_elem();
    size_t per_gpu_elements = total_elements / 4;

    for (int i = 0; i < 4; i++) {
        if (d_buf_fp16[i] == nullptr) continue; // Skip if not allocated

        // Copy device buffer to temporary host buffer
        half* temp_fp16 = new half[per_gpu_elements];
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMemcpy(temp_fp16, d_buf_fp16[i], per_gpu_elements * sizeof(half), cudaMemcpyDeviceToHost));

        // Convert to fp32 and copy to host buffer
        float* host_segment = buf + i * per_gpu_elements;
        for (size_t j = 0; j < per_gpu_elements; j++) {
            host_segment[j] = __half2float(temp_fp16[j]);
        }

        delete[] temp_fp16;
    }
}

// Free device memory (fp16) for 4 GPUs
void Tensor::free_device_fp16() {
    for (int i = 0; i < NUM_GPUS; i++) {
        if (d_buf_fp16[i] != nullptr) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaFree(d_buf_fp16[i]));
            d_buf_fp16[i] = nullptr;
        }
    }
}
#endif