#include "model.h"

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
        buf = nullptr; // Good practice to avoid dangling pointers
    }
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
}

void Tensor::printShape(const std::string& descr) {
    printf("Shape of %s tensor: ", descr.c_str());
    for (size_t i = 0; i < ndim; i++) {
        printf("%zu ", shape[i]);
    }
    printf("\n");
}