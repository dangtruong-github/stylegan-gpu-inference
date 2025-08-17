#include "layer.h"
#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

// Tiling parameters
#define BLOCK_TILE_SIZE_X 128
#define BLOCK_TILE_SIZE_Y 128
#define BLOCK_TILE_SIZE_K 16
#define WARP_TILE_SIZE_X 32
#define WARP_TILE_SIZE_Y 64
#define THREAD_TILE_SIZE_X 8
#define THREAD_TILE_SIZE_Y 8
#define SKEW 8

#define VEC_LEN 8
#define TILE_W_IM2COL 16
#define NUM_THREADS_MAT_MUL 64

#define NUM_WARPS_X (BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X)
#define NUM_WARPS_Y (BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y)
#define NUM_THREADS_PER_BLOCK (NUM_WARPS_X * NUM_WARPS_Y * 32)

#define TILE_SIZE_LINEAR 32

__global__ void batch_matmul_kernel(
    const float *A, const float *B, float *C,
    int batch, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    const float *A_batch = A + batch_idx * M * K;
    const float *B_batch = B + batch_idx * K * N;
    float *C_batch = C + batch_idx * M * N;

    __shared__ float A_shared[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + SKEW];
    __shared__ float B_shared[2][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + SKEW];

    float accum[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0f};

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_row_idx = warp_id / NUM_WARPS_X;
    int warp_col_idx = warp_id % NUM_WARPS_X;
    int thread_y_in_warp = lane_id / 4;
    int thread_x_in_warp = lane_id % 4;

    int warp_start_y = warp_row_idx * WARP_TILE_SIZE_Y;
    int warp_start_x = warp_col_idx * WARP_TILE_SIZE_X;
    int thread_start_y_in_warp = thread_y_in_warp * THREAD_TILE_SIZE_Y;
    int thread_start_x_in_warp = thread_x_in_warp * THREAD_TILE_SIZE_X;

    int num_k_tiles = (K + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;

    if (tid < 128) {
        int tid_A = tid;
        int thread_y = (tid_A % 32) * 4;
        int thread_k = (tid_A / 32) * 4;
        for (int y_offset = 0; y_offset < 4; y_offset++) {
            int y = thread_y + y_offset;
            int global_k = thread_k;
            int global_y = blockIdx.y * BLOCK_TILE_SIZE_Y + y;
            if (global_y < M && global_k < K) {
                float4 vec = *reinterpret_cast<const float4*>(A_batch + global_y * K + global_k);
                A_shared[0][thread_k][y] = vec.x;
                A_shared[0][thread_k+1][y] = vec.y;
                A_shared[0][thread_k+2][y] = vec.z;
                A_shared[0][thread_k+3][y] = vec.w;
            } else {
                A_shared[0][thread_k][y] = 0.0f;
                A_shared[0][thread_k+1][y] = 0.0f;
                A_shared[0][thread_k+2][y] = 0.0f;
                A_shared[0][thread_k+3][y] = 0.0f;
            }
        }
    } else {
        int tid_B = tid - 128;
        int thread_k = (tid_B % 4) * 4;
        int thread_x = (tid_B / 4) * 4;
        for (int k_offset = 0; k_offset < 4; k_offset++) {
            int k = thread_k + k_offset;
            int global_k = k;
            int global_x = blockIdx.x * BLOCK_TILE_SIZE_X + thread_x;
            if (global_k < K && global_x < N) {
                float4 vec = *reinterpret_cast<const float4*>(B_batch + global_k * N + global_x);
                B_shared[0][k][thread_x] = vec.x;
                B_shared[0][k][thread_x+1] = vec.y;
                B_shared[0][k][thread_x+2] = vec.z;
                B_shared[0][k][thread_x+3] = vec.w;
            } else {
                B_shared[0][k][thread_x] = 0.0f;
                B_shared[0][k][thread_x+1] = 0.0f;
                B_shared[0][k][thread_x+2] = 0.0f;
                B_shared[0][k][thread_x+3] = 0.0f;
            }
        }
    }

    __syncthreads();

    int current_buffer = 0;
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int next_k_tile = k_tile + 1;
        if (next_k_tile < num_k_tiles) {
            if (tid < 128) {
                int tid_A = tid;
                int thread_y = (tid_A % 32) * 4;
                int thread_k = (tid_A / 32) * 4;
                for (int y_offset = 0; y_offset < 4; y_offset++) {
                    int y = thread_y + y_offset;
                    int global_k = next_k_tile * BLOCK_TILE_SIZE_K + thread_k;
                    int global_y = blockIdx.y * BLOCK_TILE_SIZE_Y + y;
                    if (global_y < M && global_k < K) {
                        float4 vec = *reinterpret_cast<const float4*>(A_batch + global_y * K + global_k);
                        A_shared[1-current_buffer][thread_k][y] = vec.x;
                        A_shared[1-current_buffer][thread_k+1][y] = vec.y;
                        A_shared[1-current_buffer][thread_k+2][y] = vec.z;
                        A_shared[1-current_buffer][thread_k+3][y] = vec.w;
                    } else {
                        A_shared[1-current_buffer][thread_k][y] = 0.0f;
                        A_shared[1-current_buffer][thread_k+1][y] = 0.0f;
                        A_shared[1-current_buffer][thread_k+2][y] = 0.0f;
                        A_shared[1-current_buffer][thread_k+3][y] = 0.0f;
                    }
                }
            } else {
                int tid_B = tid - 128;
                int thread_k = (tid_B % 4) * 4;
                int thread_x = (tid_B / 4) * 4;
                for (int k_offset = 0; k_offset < 4; k_offset++) {
                    int k = thread_k + k_offset;
                    int global_k = next_k_tile * BLOCK_TILE_SIZE_K + k;
                    int global_x = blockIdx.x * BLOCK_TILE_SIZE_X + thread_x;
                    if (global_k < K && global_x < N) {
                        float4 vec = *reinterpret_cast<const float4*>(B_batch + global_k * N + global_x);
                        B_shared[1-current_buffer][k][thread_x] = vec.x;
                        B_shared[1-current_buffer][k][thread_x+1] = vec.y;
                        B_shared[1-current_buffer][k][thread_x+2] = vec.z;
                        B_shared[1-current_buffer][k][thread_x+3] = vec.w;
                    } else {
                        B_shared[1-current_buffer][k][thread_x] = 0.0f;
                        B_shared[1-current_buffer][k][thread_x+1] = 0.0f;
                        B_shared[1-current_buffer][k][thread_x+2] = 0.0f;
                        B_shared[1-current_buffer][k][thread_x+3] = 0.0f;
                    }
                }
            }
        }

        for (int k_inner = 0; k_inner < BLOCK_TILE_SIZE_K; k_inner++) {
            float a_reg[THREAD_TILE_SIZE_Y];
            float b_reg[THREAD_TILE_SIZE_X];

            for (int i = 0; i < THREAD_TILE_SIZE_Y; i++) {
                int y = warp_start_y + thread_start_y_in_warp + i;
                a_reg[i] = A_shared[current_buffer][k_inner][y];
            }

            for (int j = 0; j < THREAD_TILE_SIZE_X; j++) {
                int x = warp_start_x + thread_start_x_in_warp + j;
                b_reg[j] = B_shared[current_buffer][k_inner][x];
            }

            for (int i = 0; i < THREAD_TILE_SIZE_Y; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE_X; j++) {
                    accum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        if (next_k_tile < num_k_tiles) {
            __syncthreads();
        }
        current_buffer = 1 - current_buffer;
    }

    for (int i = 0; i < THREAD_TILE_SIZE_Y; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE_X; j++) {
            int global_y = blockIdx.y * BLOCK_TILE_SIZE_Y + warp_start_y + thread_start_y_in_warp + i;
            int global_x = blockIdx.x * BLOCK_TILE_SIZE_X + warp_start_x + thread_start_x_in_warp + j;
            if (global_y < M && global_x < N) {
                C_batch[global_y * N + global_x] = accum[i][j];
            }
        }
    }
}

// --- Batched Matrix Multiplication Wrapper (Multi-GPU, Combined) ---
void bmm_wrapper(Tensor *A, Tensor *B, Tensor *C, bool A_to_device, bool B_to_device, bool C_from_device, cudaStream_t *streams) {
  // 1. Data Movement To GPUs
  if (A_to_device) A->to_device(streams);
  if (B_to_device) B->to_device(streams);

  // 2. Get dimensions from tensor shapes
  const size_t batch_per_gpu = A->shape[0] / NUM_GPUS;
  const size_t M = A->shape[1];
  const size_t K = A->shape[2];
  const size_t N = B->shape[2];

  // 3. Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      
      // Configure the thread block and grid dimensions for the kernel
      dim3 blockDim(NUM_THREADS_PER_BLOCK, 1, 1);
      dim3 gridDim(
          (N + BLOCK_TILE_SIZE_X - 1) / BLOCK_TILE_SIZE_X,
          (M + BLOCK_TILE_SIZE_Y - 1) / BLOCK_TILE_SIZE_Y,
          batch_per_gpu
      );
      
      // Launch the kernel on the current GPU's stream
      batch_matmul_kernel<<<gridDim, blockDim, 0, streams[gpu_id]>>>(
          A->d_buf[gpu_id], 
          B->d_buf[gpu_id], 
          C->d_buf[gpu_id], 
          batch_per_gpu, 
          M, N, K
      );
      CHECK_CUDA(cudaGetLastError());
  }

  // 4. Data Movement From GPUs
  if (C_from_device) {
    C->from_device(streams);
  
    // Block CPU until all data transfers are complete
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief CUDA kernel to transform image patches into a column matrix (im2col).
 * Assumes stride=1, padding=1, and kernel size 3x3, matching the CPU logic.
 * Each thread processes one or more elements of the output column buffer.
 */
__global__ void im2col_kernel(const float* input_data, float* col_data,
                            int batch_size, int C, int H, int W) {
    // Fixed convolution parameters based on the provided CPU im2col function
    const int R = 3, S = 3, pad = 1;
    // For stride=1 and pad=1, output spatial dimensions match input
    const int OH = H, OW = W;

    // Total number of elements in the output column buffer for this GPU's batch
    long long total_elements = (long long)batch_size * C * R * S * OH * OW;
    // Grid-stride loop setup
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = index; i < total_elements; i += stride) {
        // Deconstruct the flat output index `i` into multi-dimensional indices
        long long hw_col = i % (OH * OW);
        long long temp = i / (OH * OW);
        long long crs_col = temp % (C * R * S);
        long long n = temp / (C * R * S);

        int w_col = hw_col % OW;
        int h_col = hw_col / OW;
        int s = crs_col % S;
        int r = (crs_col / S) % R;
        int c = crs_col / (R * S);

        // Calculate corresponding coordinates in the source input tensor
        int input_h = h_col + r - pad;
        int input_w = w_col + s - pad;

        long long dest_idx = i; // The destination index is simply the loop index

        // Bounds check: if the source coordinate is outside the padded input, write zero
        if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
            long long src_idx = (long long)n * (C * H * W) +
                                (long long)c * (H * W) +
                                (long long)input_h * W + input_w;
            col_data[dest_idx] = input_data[src_idx];
        } else {
            col_data[dest_idx] = 0.0f;
        }
    }
}

/**
 * @brief CUDA kernel to transform a column buffer back into an image (col2im).
 * This performs a "scatter-add" operation, requiring atomic adds to prevent race conditions.
 * Assumes a transposed convolution with stride=2 and kernel size 3x3.
 */
__global__ void col2im_kernel(const float* col_data, float* output_data,
                            int batch_size, int K, int OH, int OW, int H, int W) {
    // Fixed convolution parameters from the CPU col2im function
    const int R = 3, S = 3, stride = 2;
    const long long KRS = K * R * S;
    const long long HW = H * W;

    // Total elements in the source column buffer for this GPU's batch
    long long total_elements = (long long)batch_size * KRS * HW;
    // Grid-stride loop setup
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;

    for (long long i = index; i < total_elements; i += grid_stride) {
        float val = col_data[i];
        if (val == 0.0f) continue; // Optimization: skip if value is zero

        // Deconstruct the flat source index `i` from the column buffer
        long long hw = i % HW;
        long long temp = i / HW;
        long long krs = temp % KRS;
        long long n = temp / KRS;

        int w = hw % W;
        int h = hw / W;
        int s = krs % S;
        int r = (krs / S) % R;
        int k = krs / (R * S);

        // Calculate target coordinates in the output image (scatter)
        const int oh = h * stride + r;
        const int ow = w * stride + s;

        // Bounds check before writing to the output tensor
        if (oh < OH && ow < OW) {
            long long output_idx = (long long)n * (K * OH * OW) +
                                   (long long)k * (OH * OW) +
                                   (long long)oh * OW + ow;
            // Use atomicAdd to safely add the value, as multiple threads might
            // target the same output pixel.
            atomicAdd(&output_data[output_idx], val);
        }
    }
}

/**
 * @brief CUDA kernel to transpose a 5D tensor from (N, K, C, R, S) to (N, K, R, S, C).
 * This is a memory-bound operation where each thread re-maps one element.
 */
__global__ void transpose_kernel(const float* weight_data, float* transposed_data,
                               int batch_size, int K, int C, int R, int S) {
    // Total number of elements for this GPU's batch
    long long total_elements = (long long)batch_size * K * C * R * S;

    // Grid-stride loop over the *output* tensor's elements
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = index; i < total_elements; i += stride) {
        // Deconstruct the flat destination index `i` based on the target layout (N, K, R, S, C)
        long long temp = i;
        int c_out = temp % C;
        temp /= C;
        int s_out = temp % S;
        temp /= S;
        int r_out = temp % R;
        temp /= R;
        int k_out = temp % K;
        int n_out = temp / K;

        // Calculate the corresponding source index in the original tensor (N, K, C, R, S)
        long long src_idx = (long long)n_out * (K * C * R * S) +
                            (long long)k_out * (C * R * S) +
                            (long long)c_out * (R * S) +
                            (long long)r_out * S + s_out;
        
        // Copy the element from source to destination
        transposed_data[i] = weight_data[src_idx];
    }
}

/**
 * @brief Multi-GPU wrapper for the im2col operation.
 */
void im2col_wrapper(Tensor *input, Tensor *col_buffer, bool input_to_device, bool col_buffer_from_device, cudaStream_t *streams) {
  // 1. Data Movement To GPUs
  if (input_to_device) input->to_device(streams);

  // 2. Get dimensions from tensor shapes
  const size_t N = input->shape[0];
  const size_t C = input->shape[1];
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];
  const size_t batch_per_gpu = N / NUM_GPUS;

  // 3. Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      
      long long total_elements_gpu = (long long)batch_per_gpu * C * 3 * 3 * H * W;
      int blockSize = 256;
      int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
      
      im2col_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
          input->d_buf[gpu_id], 
          col_buffer->d_buf[gpu_id], 
          batch_per_gpu, C, H, W
      );
      CHECK_CUDA(cudaGetLastError());
  }

  // 4. Data Movement From GPUs
  if (col_buffer_from_device) {
    col_buffer->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}


/**
 * @brief Multi-GPU wrapper for the col2im operation.
 */
void col2im_wrapper(Tensor *col_buffer, Tensor *output, int H, int W, bool col_buffer_to_device, bool output_from_device, cudaStream_t *streams) {
  // 1. Data Movement To GPUs
  if (col_buffer_to_device) col_buffer->to_device(streams);

  // 2. Get dimensions from tensor shapes
  const size_t N = output->shape[0];
  const size_t K = output->shape[1];
  const size_t OH = output->shape[2];
  const size_t OW = output->shape[3];
  const size_t batch_per_gpu = N / NUM_GPUS;

  // 3. Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      
      // IMPORTANT: Initialize output buffer on GPU to zeros before atomic additions
      size_t output_bytes_per_gpu = batch_per_gpu * K * OH * OW * sizeof(float);
      CHECK_CUDA(cudaMemsetAsync(output->d_buf[gpu_id], 0, output_bytes_per_gpu, streams[gpu_id]));
      
      long long total_elements_gpu = (long long)batch_per_gpu * K * 3 * 3 * H * W;
      int blockSize = 256;
      int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
      
      col2im_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
          col_buffer->d_buf[gpu_id],
          output->d_buf[gpu_id],
          batch_per_gpu, K, OH, OW, H, W
      );
      CHECK_CUDA(cudaGetLastError());
  }
  
  // 4. Data Movement From GPUs
  if (output_from_device) {
    output->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}


/**
 * @brief Multi-GPU wrapper for the 5D transpose operation.
 */
void transpose_wrapper(Tensor *weight, Tensor *weight_transpose, bool weight_to_device, bool transpose_from_device, cudaStream_t *streams) {
    // 1. Data Movement To GPUs
    if (weight_to_device) weight->to_device(streams);

    // 2. Get dimensions from tensor shapes
    const size_t N = weight->shape[0];
    const size_t K = weight->shape[1];
    const size_t C = weight->shape[2];
    const size_t R = weight->shape[3];
    const size_t S = weight->shape[4];
    const size_t batch_per_gpu = N / NUM_GPUS;

    // 3. Kernel Execution
    for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
        CHECK_CUDA(cudaSetDevice(gpu_id));
        
        long long total_elements_gpu = (long long)batch_per_gpu * K * C * R * S;
        int blockSize = 256;
        int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
        
        transpose_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
            weight->d_buf[gpu_id],
            weight_transpose->d_buf[gpu_id],
            batch_per_gpu, K, C, R, S
        );
        CHECK_CUDA(cudaGetLastError());
    }
    
    // 4. Data Movement From GPUs
    if (transpose_from_device) {
      weight_transpose->from_device(streams);
      for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
      }
    }
}

/**
 * @brief CUDA kernel for demodulating convolution weights.
 */
__global__ void demodulate_kernel(float *weight, size_t total_filters, size_t weight_inner_dim) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the bounds of the filters to process
    if (idx < total_filters) {
        // Get a pointer to the start of the current filter
        float *filter_ptr = weight + idx * weight_inner_dim;

        // --- Pass 1: Calculate the sum of squares for the filter ---
        float sum_sq = 0.0f;
        for (size_t i = 0; i < weight_inner_dim; ++i) {
            float w = filter_ptr[i];
            sum_sq += w * w;
        }

        // --- Pass 2: Calculate and apply the demodulation factor ---
        // Add a small epsilon for numerical stability
        float demod_factor = rsqrtf(sum_sq + 1e-8f);
        for (size_t i = 0; i < weight_inner_dim; ++i) {
            filter_ptr[i] *= demod_factor;
        }
    }
}

/**
 * @brief Launches demodulation kernels across multiple GPUs using pre-distributed data.
 */
void demodulate_wrapper(Tensor *weight_a, bool weight_to_device, bool weight_from_device, size_t in_C, size_t R, size_t S, cudaStream_t *streams) {
  size_t total_samples = weight_a->shape[0]; // Total batch size across all GPUs
  size_t out_C = weight_a->shape[1];
  size_t weight_inner_dim = in_C * R * S;

  if (weight_to_device) weight_a->to_device(streams);

  // Use OpenMP to parallelize kernel launches, assigning one CPU thread per GPU.
  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    // Set the active GPU for the current CPU thread.
    CHECK_CUDA(cudaSetDevice(i));

    // --- Distribute Workload ---
    size_t start_sample = total_samples * i / NUM_GPUS;
    size_t end_sample = total_samples * (i + 1) / NUM_GPUS;
    size_t num_samples_for_gpu = end_sample - start_sample;

    if (num_samples_for_gpu == 0) {
        continue; // Skip if this GPU has no work.
    }

    float* data_ptr_for_gpu = weight_a->d_buf[i];
    size_t total_filters_on_gpu = num_samples_for_gpu * out_C;
    
    // --- Launch Kernel ---
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_filters_on_gpu + threads_per_block - 1) / threads_per_block;

    demodulate_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>(
        data_ptr_for_gpu,
        total_filters_on_gpu,
        weight_inner_dim
    );
  }

  // --- Synchronize all streams across all devices ---
  if (weight_from_device) {
    weight_a->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

__global__ void modulation_kernel(
    float* weight_a_slice,
    const float* conv_weight,
    const float* style_a_slice,
    float scale,
    size_t total_elements_on_gpu,
    size_t out_C,
    size_t in_C,
    size_t kernel_size)
{
    const size_t element_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_idx < total_elements_on_gpu) {
        // Decompose the element's index to find its (n, oc, ic, k) coordinates.
        const size_t k = element_idx % kernel_size;
        const size_t task_idx = element_idx / kernel_size;
        const size_t n = task_idx / (out_C * in_C);
        const size_t temp = task_idx % (out_C * in_C);
        const size_t oc = temp / in_C;
        const size_t ic = temp % in_C;

        // Calculate the style value.
        const float style_val = style_a_slice[n * in_C + ic] * scale;

        // Find the source weight from the conv_weight tensor.
        const size_t weight_inner_dim = in_C * kernel_size;
        const size_t conv_weight_idx = oc * weight_inner_dim + ic * kernel_size + k;

        // Compute and write the final value to the output tensor.
        weight_a_slice[element_idx] = conv_weight[conv_weight_idx] * style_val;
    }
}

void modulate_wrapper(
    Tensor* conv_weight,
    Tensor* style_a,
    Tensor* weight_a,
    float scale,
    size_t out_C,
    size_t in_C,
    size_t R,
    size_t S,
    bool conv_weight_to_device,
    bool style_a_to_device,
    bool weight_a_from_device,
    cudaStream_t* streams)
{
  if (conv_weight_to_device) conv_weight->to_device(streams);
  if (style_a_to_device) style_a->to_device(streams);

  const size_t total_samples = weight_a->shape[0];
  const size_t kernel_size = R * S;
  const size_t num_samples_per_gpu = total_samples / NUM_GPUS;

  const size_t total_elements_on_gpu = num_samples_per_gpu * out_C * in_C * kernel_size;
  
  const int threads_per_block = 256;
  const int blocks_per_grid = (total_elements_on_gpu + threads_per_block - 1) / threads_per_block;


  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));

      float* weight_a_ptr = weight_a->d_buf[i];
      const float* style_a_ptr = style_a->d_buf[i];
      const float* conv_weight_ptr = conv_weight->d_buf[i];

      modulation_kernel<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>(
          weight_a_ptr, conv_weight_ptr, style_a_ptr, scale,
          total_elements_on_gpu, out_C, in_C, kernel_size
      );
  }

  if (weight_a_from_device) {
    weight_a->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Corrected CUDA kernel for a high-performance tiled matrix multiplication.
 */
__global__ void linear_kernel(float* out, const float* in, const float* w, const float* b,
                              size_t M, size_t N, size_t K, float scale, float lr_mul)
{
    // Shared memory for tiles of 'in' (As) and 'w' (Bs)
    __shared__ float As[TILE_SIZE_LINEAR][TILE_SIZE_LINEAR];
    __shared__ float Bs[TILE_SIZE_LINEAR][TILE_SIZE_LINEAR];

    // Thread identification
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Identify the output element this thread will compute
    int out_row = blockRow * TILE_SIZE_LINEAR + threadRow;
    int out_col = blockCol * TILE_SIZE_LINEAR + threadCol;

    // Register for accumulating the dot product
    float sum = 0.0f;

    // Loop through tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE_LINEAR - 1) / TILE_SIZE_LINEAR; ++t) {
        // Cooperatively load a tile of 'in' (A) into shared memory
        int in_k = t * TILE_SIZE_LINEAR + threadCol;
        if (out_row < M && in_k < K) {
            As[threadRow][threadCol] = in[out_row * K + in_k];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }

        // Cooperatively load a tile of 'w' (B) into shared memory
        int w_k = t * TILE_SIZE_LINEAR + threadRow;
        if (out_col < N && w_k < K) {
            Bs[threadRow][threadCol] = w[out_col * K + w_k];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads(); // Wait for all threads to finish loading tiles

        #pragma unroll
        for (int k = 0; k < TILE_SIZE_LINEAR; ++k) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads(); // Wait for all threads to finish with the current tiles
    }

    // Write the final result to global memory
    if (out_row < M && out_col < N) {
        out[out_row * N + out_col] = sum * scale + b[out_col] * lr_mul;
    }
}

/*
 * @brief Orchestrates a Linear layer computation across multiple GPUs.
 */
void linear_wrapper(Tensor *in, Tensor *w, Tensor *b, Tensor *out,
                    float lr_mul, bool in_to_device, bool out_from_device, cudaStream_t *streams)
{
  if (in_to_device) in->to_device(streams);

  const size_t M = out->shape[0];
  const size_t N = out->shape[1];
  const size_t K = w->shape[1];
  const float scale = (1.0f / sqrtf(K)) * lr_mul;
  const size_t M_for_gpu = M / NUM_GPUS;

  dim3 block_dim(TILE_SIZE_LINEAR, TILE_SIZE_LINEAR, 1);
  dim3 grid_dim((N + TILE_SIZE_LINEAR - 1) / TILE_SIZE_LINEAR, (M_for_gpu + TILE_SIZE_LINEAR - 1) / TILE_SIZE_LINEAR, 1);

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    
    float* out_ptr = out->d_buf[i];
    const float* in_ptr = in->d_buf[i];
    const float* w_ptr = w->d_buf[i];
    const float* b_ptr = b->d_buf[i];

    linear_kernel<<<grid_dim, block_dim, 0, streams[i]>>>(
      out_ptr, in_ptr, w_ptr, b_ptr, M_for_gpu, N, K, scale, lr_mul
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (out_from_device) {
    out->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Corrected fused CUDA kernel to perform addNoise+addBias+LeakyReLU.
 */
__global__ void addNoiseBiasLeakyReLU_kernel(
    float* out_slice,
    const float* noise,
    const float* bias,
    size_t total_elements_on_gpu,
    size_t C, size_t H, size_t W)
{
    const float negative_slope = 0.2f;
    const float scale = sqrtf(2.0f);
    const float neg_slope_scaled = negative_slope * scale;

    const size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < total_elements_on_gpu) {
        const size_t spatial_dim = H * W;
        const size_t c = (global_idx / spatial_dim) % C;
        const size_t spatial_idx = global_idx % spatial_dim;

        float val = out_slice[global_idx] + noise[spatial_idx] + bias[c];

        if (val >= 0.0f) {
            out_slice[global_idx] = val * scale;
        } else {
            out_slice[global_idx] = val * neg_slope_scaled;
        }
    }
}

/**
 * @brief Orchestrates the fused addNoise+addBias+LeakyReLU operation across multiple GPUs.
 */
void addNoiseBiasLeakyReLU_wrapper(Tensor *output, Tensor *noise, Tensor *conv_bias, bool output_to_device, bool noise_to_device, bool conv_bias_to_device, bool output_from_device, cudaStream_t *streams)
{
  if (output_to_device) output->to_device(streams);
  if (noise_to_device) noise->to_device(streams);
  if (conv_bias_to_device) conv_bias->to_device(streams);
  const size_t N = output->shape[0];
  const size_t C = output->shape[1];
  const size_t H = output->shape[2];
  const size_t W = output->shape[3];

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t N_for_gpu = N / NUM_GPUS;
    const size_t total_elements_on_gpu = N_for_gpu * C * H * W;

    if (total_elements_on_gpu == 0) continue;

    float* out_ptr = output->d_buf[i];
    const float* noise_ptr = noise->d_buf[i];
    const float* bias_ptr = conv_bias->d_buf[i];

    const int block_size = 256;
    const int grid_size = (total_elements_on_gpu + block_size - 1) / block_size;

    addNoiseBiasLeakyReLU_kernel<<<grid_size, block_size, 0, streams[i]>>>(
        out_ptr, noise_ptr, bias_ptr, total_elements_on_gpu, C, H, W
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (output_from_device) {
    output->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Performs an in-place bias addition on the GPU.
 */
__global__ void addBias_kernel(
    float* inout_slice,
    const float* bias,
    size_t total_elements_on_gpu,
    size_t C,
    size_t HW)
{
    const size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < total_elements_on_gpu) {
        const size_t c = (global_idx / HW) % C;
        inout_slice[global_idx] += bias[c];
    }
}

/**
 * @brief Orchestrates an in-place bias addition across multiple GPUs.
 */
void addBias_wrapper(Tensor *inout, Tensor *bias, bool inout_to_device, bool bias_to_device, bool inout_from_device, cudaStream_t *streams) {
  if (inout_to_device) inout->to_device(streams);
  if (bias_to_device) bias->to_device(streams);

  const size_t N = inout->shape[0];
  const size_t C = inout->shape[1];
  const size_t H = inout->shape[2];
  const size_t W = inout->shape[3];
  const size_t HW = H * W;

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t N_for_gpu = N / NUM_GPUS;
    const size_t total_elements_on_gpu = N_for_gpu * C * HW;

    if (total_elements_on_gpu == 0) continue;

    float* inout_ptr = inout->d_buf[i];
    const float* bias_ptr = bias->d_buf[i];

    const int block_size = 256;
    const int grid_size = (total_elements_on_gpu + block_size - 1) / block_size;

    addBias_kernel<<<grid_size, block_size, 0, streams[i]>>>(
        inout_ptr, bias_ptr, total_elements_on_gpu, C, HW
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (inout_from_device) {
    inout->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Scatters input elements to an upsampled and padded output tensor.
 */
__global__ void upsample_pad_kernel(
    const float* in_slice,
    float* out_slice,
    size_t total_input_elements_on_gpu,
    size_t C, size_t H, size_t W,
    size_t OH, size_t OW,
    int up, int pad0)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_input_elements_on_gpu) {
        const size_t i_hw = idx % (H * W);
        const size_t i_chw = idx % (C * H * W);
        const size_t n = idx / (C * H * W);
        const size_t c = i_chw / (H * W);
        const size_t h = i_hw / W;
        const size_t w = i_hw % W;

        const size_t out_h = h * up + pad0;
        const size_t out_w = w * up + pad0;

        const size_t dest_idx = n * (C * OH * OW) + c * (OH * OW) + out_h * OW + out_w;

        out_slice[dest_idx] = in_slice[idx];
    }
}

/**
 * @brief Orchestrates an upsample-and-pad operation across multiple GPUs.
 */
void upsample_wrapper(Tensor *input, Tensor *output, int up, int pad0, int pad1,
                      bool input_to_device, bool output_from_device, cudaStream_t *streams)
{
  if (input_to_device) input->to_device(streams);

  const size_t N = input->shape[0];
  const size_t C = input->shape[1];
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];

  const size_t OH = up * H + pad0 + pad1;
  const size_t OW = up * W + pad0 + pad1;

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t N_for_gpu = N / NUM_GPUS;
    const size_t total_input_elements_on_gpu = N_for_gpu * C * H * W;
    const size_t total_output_bytes_on_gpu = N_for_gpu * C * OH * OW * sizeof(float);

    if (total_input_elements_on_gpu == 0) continue;

    const float* in_ptr = input->d_buf[i];
    float* out_ptr = output->d_buf[i];

    CHECK_CUDA(cudaMemsetAsync(out_ptr, 0, total_output_bytes_on_gpu, streams[i]));

    const int block_size = 256;
    const int grid_size = (total_input_elements_on_gpu + block_size - 1) / block_size;

    upsample_pad_kernel<<<grid_size, block_size, 0, streams[i]>>>(
        in_ptr, out_ptr, total_input_elements_on_gpu, C, H, W, OH, OW, up, pad0
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (output_from_device) {
    output->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

#define TILE_DIM_2D_SAME 32
#define MAX_KERNEL_DIM_2D_SAME 7

/**
 * @brief Performs a 2D depthwise convolution using a tiled shared memory approach.
 */
__global__ void conv2d_same_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int H, int W, int R, int S, int OH, int OW,
    int stride, int pad, int dilation)
{
    const int PADDED_TILE_DIM_2D_SAME = TILE_DIM_2D_SAME + MAX_KERNEL_DIM_2D_SAME - 1;
    __shared__ float in_tile[PADDED_TILE_DIM_2D_SAME][PADDED_TILE_DIM_2D_SAME];
    __shared__ float w_tile[MAX_KERNEL_DIM_2D_SAME][MAX_KERNEL_DIM_2D_SAME];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.z;

    const int out_tile_y_start = blockIdx.y * TILE_DIM_2D_SAME;
    const int out_tile_x_start = blockIdx.x * TILE_DIM_2D_SAME;

    const int in_tile_y_start = out_tile_y_start * stride - pad;
    const int in_tile_x_start = out_tile_x_start * stride - pad;

    if (ty < R && tx < S) {
        w_tile[ty][tx] = weight[ty * S + tx];
    }

    for (int y = ty; y < PADDED_TILE_DIM_2D_SAME; y += TILE_DIM_2D_SAME) {
        for (int x = tx; x < PADDED_TILE_DIM_2D_SAME; x += TILE_DIM_2D_SAME) {
            int in_y = in_tile_y_start + y;
            int in_x = in_tile_x_start + x;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                in_tile[y][x] = input[(n * H + in_y) * W + in_x];
            } else {
                in_tile[y][x] = 0.0f;
            }
        }
    }
    __syncthreads();

    const int out_y = out_tile_y_start + ty;
    const int out_x = out_tile_x_start + tx;

    if (out_y < OH && out_x < OW) {
        float acc = 0.0f;
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                acc += in_tile[ty * stride + r * dilation][tx * stride + s * dilation] * w_tile[r][s];
            }
        }
        output[(n * OH + out_y) * OW + out_x] = acc;
    }
}

/**
 * @brief Orchestrates a 2D convolution across multiple GPUs.
 */
void conv2d_same_wrapper(Tensor *input, Tensor *weight, Tensor *output,
                    int stride, int pad, int dilation,
                    bool input_to_device, bool weight_to_device, bool output_from_device, cudaStream_t *streams)
{
  if (input_to_device) input->to_device(streams);
  if (weight_to_device) weight->to_device(streams);

  const size_t N = input->shape[0];
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];
  const size_t R = weight->shape[2];
  const size_t S = weight->shape[3];
  const size_t OH = output->shape[2];
  const size_t OW = output->shape[3];

  const int TILE_DIM = 32;

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t N_for_gpu = N / NUM_GPUS;
    if (N_for_gpu == 0) continue;

    const float* in_ptr = input->d_buf[i];
    const float* weight_ptr = weight->d_buf[i];
    float* out_ptr = output->d_buf[i];

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((OW + TILE_DIM - 1) / TILE_DIM,
                  (OH + TILE_DIM - 1) / TILE_DIM,
                  N_for_gpu);

    conv2d_same_kernel<<<grid_dim, block_dim, 0, streams[i]>>>(
        in_ptr, weight_ptr, out_ptr,
        N_for_gpu, H, W, R, S, OH, OW,
        stride, pad, dilation
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (output_from_device) {
    output->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Performs an in-place element-wise addition on the GPU.
 */
__global__ void elemAdd_kernel(
    float* __restrict__ inout_slice,
    const float* __restrict__ addend_slice,
    size_t total_elements_on_gpu)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  for (; idx < total_elements_on_gpu; idx += stride) {
      inout_slice[idx] += addend_slice[idx];
  }
}

/**
 * @brief Orchestrates an in-place element-wise addition across multiple GPUs.
 */
void elemAdd_wrapper(Tensor *inout, Tensor *addend, bool inout_to_device, bool addend_to_device, bool inout_from_device, cudaStream_t *streams) {
  if (inout_to_device) inout->to_device(streams);
  if (addend_to_device) addend->to_device(streams);

  const size_t total_elements = inout->num_elem();

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t elements_for_gpu = total_elements / NUM_GPUS;
    if (elements_for_gpu == 0) continue;

    float* inout_ptr = inout->d_buf[i];
    const float* addend_ptr = addend->d_buf[i];

    const int block_size = 256;
    const int grid_size = (elements_for_gpu + block_size - 1) / block_size;

    elemAdd_kernel<<<grid_size, block_size, 0, streams[i]>>>(
        inout_ptr, addend_ptr, elements_for_gpu
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (inout_from_device) {
    inout->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Fused kernel to compute Linear + LeakyReLU with corrected bias logic.
 */
__global__ void fusedLinearLeakyReLU_kernel(
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    size_t M, size_t N, size_t K,
    float lr_mul, float linear_scale, float leaky_negative_slope, float leaky_scale)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (m < M && n < N) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
          sum += in[m * K + k] * w[n * K + k];
      }

      float val = sum * linear_scale * lr_mul + b[n] * lr_mul;

      if (val < 0.0f) {
          val *= leaky_negative_slope;
      }
      val *= leaky_scale;

      out[m * N + n] = val;
  }
}

/**
 * @brief Orchestrates a Fused Linear + LeakyReLU operation across multiple GPUs.
 */
void fusedLinearLeakyReLU_wrapper(Tensor *in, Tensor *w, Tensor *b, Tensor *out,
                                  float lr_mul, bool in_to_device, bool out_from_device, cudaStream_t *streams)
{
  if (in_to_device) in->to_device(streams);

  const size_t M = out->shape[0];
  const size_t N = out->shape[1];
  const size_t K = w->shape[1];

  const float linear_scale = (1.0f / sqrtf(K));
  const float leaky_negative_slope = 0.2f;
  const float leaky_scale = sqrtf(2.0f);

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));

      const size_t M_start = M * i / NUM_GPUS;
      const size_t M_end = M * (i + 1) / NUM_GPUS;
      const size_t M_for_gpu = M_end - M_start;
      if (M_for_gpu == 0) continue;

      const float* in_ptr = in->d_buf[i];
      const float* w_ptr = w->d_buf[i];
      const float* b_ptr = b->d_buf[i];
      float* out_ptr = out->d_buf[i];

      dim3 block_dim(16, 16);
      dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                    (M_for_gpu + block_dim.y - 1) / block_dim.y);

      fusedLinearLeakyReLU_kernel<<<grid_dim, block_dim, 0, streams[i]>>>(
          in_ptr, w_ptr, b_ptr, out_ptr,
          M_for_gpu, N, K,
          lr_mul, linear_scale, leaky_negative_slope, leaky_scale
      );
      CHECK_CUDA(cudaGetLastError());
  }

  if (out_from_device) {
    out->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

/**
 * @brief Corrected kernel for in-place PixelNorm.
 */
__global__ void pixelNorm_kernel(float* __restrict__ inout_slice, size_t N_for_gpu, size_t C) {
    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int n = blockIdx.x;

    float* row_ptr = inout_slice + n * C;

    float sum_sq = 0.0f;
    for (unsigned int i = tid; i < C; i += block_size) {
        float val = row_ptr[i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[0] = rsqrtf(sdata[0] / (float)C + 1e-8f);
    }
    __syncthreads();

    const float norm_factor = sdata[0];

    for (unsigned int i = tid; i < C; i += block_size) {
        row_ptr[i] *= norm_factor;
    }
}

/**
 * @brief Orchestrates the PixelNorm operation across multiple GPUs.
 */
void pixelNorm_wrapper(Tensor *inout, bool inout_to_device, bool inout_from_device, cudaStream_t *streams) {
  if (inout_to_device) inout->to_device(streams);

  const size_t N = inout->shape[0];
  const size_t C = inout->shape[1];

  #pragma omp parallel for
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    const size_t N_for_gpu = N / NUM_GPUS;
    if (N_for_gpu == 0) continue;

    float* inout_ptr = inout->d_buf[i];

    const int threads_per_block = 512;
    const int blocks_per_grid = N_for_gpu;

    const size_t shared_mem_size = threads_per_block * sizeof(float);

    pixelNorm_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, streams[i]>>>(
        inout_ptr, N_for_gpu, C
    );
    CHECK_CUDA(cudaGetLastError());
  }

  if (inout_from_device) {
    inout->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
}

// -------------- MAIN FUNCTION ---------------------

static inline float horizontal_sum_avx(__m256 vec) {
  __m128 lo = _mm256_castps256_ps128(vec);
  __m128 hi = _mm256_extractf128_ps(vec, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(sum128);
}

void PixelNorm(Tensor *inout) {
  const size_t N = inout->shape[0];
  const size_t C = inout->shape[1];
  float *__restrict__ buf = inout->buf;

  for (size_t n = 0; n < N; ++n) {
    float *row_ptr = buf + n * C;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (size_t i = 0; i < C; i += 32) {
      __m256 v0 = _mm256_load_ps(row_ptr + i);
      __m256 v1 = _mm256_load_ps(row_ptr + i + 8);
      __m256 v2 = _mm256_load_ps(row_ptr + i + 16);
      __m256 v3 = _mm256_load_ps(row_ptr + i + 24);

      acc0 = _mm256_fmadd_ps(v0, v0, acc0);
      acc1 = _mm256_fmadd_ps(v1, v1, acc1);
      acc2 = _mm256_fmadd_ps(v2, v2, acc2);
      acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    const float sum_squares = horizontal_sum_avx(acc0);
    const float mean_squares = sum_squares / (float)C;
    const float val_to_rsqrt = mean_squares + 1e-8f;

    const __m256 val_vec = _mm256_set1_ps(val_to_rsqrt);
    __m256 norm_vec = _mm256_rsqrt_ps(val_vec);

    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256 muls = _mm256_mul_ps(_mm256_mul_ps(val_vec, norm_vec), norm_vec);
    norm_vec = _mm256_mul_ps(_mm256_mul_ps(half, norm_vec), _mm256_sub_ps(three, muls));

    for (size_t i = 0; i < C; i += 32) {
      __m256 v0 = _mm256_load_ps(row_ptr + i);
      __m256 v1 = _mm256_load_ps(row_ptr + i + 8);
      __m256 v2 = _mm256_load_ps(row_ptr + i + 16);
      __m256 v3 = _mm256_load_ps(row_ptr + i + 24);
      
      v0 = _mm256_mul_ps(v0, norm_vec);
      v1 = _mm256_mul_ps(v1, norm_vec);
      v2 = _mm256_mul_ps(v2, norm_vec);
      v3 = _mm256_mul_ps(v3, norm_vec);

      _mm256_store_ps(row_ptr + i, v0);
      _mm256_store_ps(row_ptr + i + 8, v1);
      _mm256_store_ps(row_ptr + i + 16, v2);
      _mm256_store_ps(row_ptr + i + 24, v3);
    }
  }
}

void UpsamplePad(Tensor *input, Tensor *output, size_t up, size_t pad0, size_t pad1) {
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];
  size_t OH = H * up + pad0 + pad1;
  size_t OW = W * up + pad0 + pad1;

  if (up == 1) {
    #pragma omp parallel for
    for (size_t nc = 0; nc < N * C; ++nc) {
      float *out_channel_ptr = output->buf + nc * OH * OW;
      const float *in_channel_ptr = input->buf + nc * H * W;

      if (pad0 > 0) {
        memset(out_channel_ptr, 0, pad0 * OW * sizeof(float));
      }

      float *out_row_ptr = out_channel_ptr + pad0 * OW;
      const float *in_row_ptr = in_channel_ptr;
      for (size_t h = 0; h < H; ++h) {
        if (pad0 > 0) {
          memset(out_row_ptr, 0, pad0 * sizeof(float));
        }
        memcpy(out_row_ptr + pad0, in_row_ptr, W * sizeof(float));
        if (pad1 > 0) {
          memset(out_row_ptr + pad0 + W, 0, pad1 * sizeof(float));
        }
        out_row_ptr += OW;
        in_row_ptr += W;
      }

      if (pad1 > 0) {
        memset(out_channel_ptr + (pad0 + H) * OW, 0, pad1 * OW * sizeof(float));
      }
    }
  }
  else {
    #pragma omp parallel for
    for (size_t nc = 0; nc < N * C; ++nc) {
      float *out_channel_ptr = output->buf + nc * OH * OW;
      const float *in_channel_ptr = input->buf + nc * H * W;

      if (pad0 > 0) {
        memset(out_channel_ptr, 0, pad0 * OW * sizeof(float));
      }

      for (size_t h = 0; h < H; ++h) {
        float *out_row_ptr = out_channel_ptr + (h * up + pad0) * OW;
        const float *in_row_ptr = in_channel_ptr + h * W;

        if (pad0 > 0) {
            memset(out_row_ptr, 0, pad0 * sizeof(float));
        }

        float *out_pixel_ptr = out_row_ptr + pad0;
        for (size_t w = 0; w < W; ++w) {
            *out_pixel_ptr = in_row_ptr[w];
            
            for (size_t i = 1; i < up; ++i) {
                *(out_pixel_ptr + i) = 0.0f;
            }
            out_pixel_ptr += up;
        }
        
        if (pad1 > 0) {
            memset(out_pixel_ptr, 0, pad1 * sizeof(float));
        }
      }

      for (size_t h = 0; h < H; ++h) {
        for (size_t i = 1; i < up; ++i) {
          float *gap_row_ptr = out_channel_ptr + (h * up + pad0 + i) * OW;
          memset(gap_row_ptr, 0, OW * sizeof(float));
        }
      }

      if (pad1 > 0) {
        memset(out_channel_ptr + (pad0 + H * up) * OW, 0, pad1 * OW * sizeof(float));
      }
    }
  }
}

void Conv2d(Tensor *input, Tensor *weight, Tensor *output, cudaStream_t* streams) {
  size_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  size_t K = weight->shape[1];
  size_t OH = output->shape[2], OW = output->shape[3];

  weight->reshape({N, K, C});
  input->reshape({N, C, H*W});
  output->reshape({N, K, OH*OW});
  bmm_wrapper(weight, input, output, false, false, false, streams);

  weight->reshape({N, K, C, 1, 1});
  input->reshape({N, C, H, W});
  output->reshape({N, K, OH, OW});
}

void Conv2d_same(Tensor *input, Tensor *weight, Tensor *output,
                int stride, int pad, int dilation) {
  size_t N = input->shape[0], H = input->shape[2], W = input->shape[3];
  size_t OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for schedule(static) collapse(2)
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < OH; oh += 2) {
      for (size_t ow = 0; ow < OW; ow += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        const float* in_base = &input->buf[n * H * W];
        const float* w_base = &weight->buf[0];

        for (int r = 0; r < 4; ++r) {
          for (int s = 0; s < 4; ++s) {
            size_t h0 = oh + r;
            size_t h1 = oh + 1 + r;
            size_t w = ow + s;

            float f = w_base[r * 4 + s];
            __m256 filt = _mm256_set1_ps(f);

            __m256 in0 = _mm256_load_ps(&in_base[h0 * W + w]);
            __m256 in1 = _mm256_load_ps(&in_base[h1 * W + w]);

            acc0 = _mm256_fmadd_ps(in0, filt, acc0);
            acc1 = _mm256_fmadd_ps(in1, filt, acc1);
          }
        }

        float* out_base = &output->buf[n * OH * OW];
        
        _mm256_store_ps(&out_base[oh * OW + ow], acc0);
        _mm256_store_ps(&out_base[(oh + 1) * OW + ow], acc1);
      }
    }
  }
}

void Conv2d_im2col(Tensor *input, Tensor *weight, Tensor *output, Tensor *col_buffer, cudaStream_t *streams) {
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t K = weight->shape[1];
  size_t R = weight->shape[3];
  size_t S = weight->shape[4];
  size_t OH = output->shape[2];
  size_t OW = output->shape[3];

  im2col_wrapper(input, col_buffer, false, false, streams);

  weight->reshape({N, K, (size_t)(C * R * S)});
  output->reshape({N, K, (size_t)(OH * OW)});

  bmm_wrapper(weight, col_buffer, output, false, false, false, streams);

  weight->reshape({N, K, C, R, S});
  output->reshape({N, K, OH, OW});
}

void ConvTranspose2d_col2im(Tensor *input, Tensor *weight, Tensor *output,
                            Tensor *col_buffer, cudaStream_t *streams) {
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];
  const size_t C = input->shape[1];
  const size_t N = input->shape[0];
  const size_t K = weight->shape[1];
  const int R = 3, S = 3;

  weight->reshape({N, (size_t)(K * R * S), C});
  input->reshape({N, C, (size_t)(H * W)});
  bmm_wrapper(weight, input, col_buffer, false, false, false, streams);
  input->reshape({N, C, H, W});
  weight->reshape({N, K, (size_t)R, (size_t)S, C});

  col2im_wrapper(col_buffer, output, H, W, false, false, streams);
}

static inline float hsum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    float scale = (1.0f / sqrtf(K)) * lr_mul;

    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            const float *in_row = in->buf + m * K;
            const float *w_row = w->buf + n * K;

            __m256 sum_vec1 = _mm256_setzero_ps();
            __m256 sum_vec2 = _mm256_setzero_ps();
            __m256 sum_vec3 = _mm256_setzero_ps();
            __m256 sum_vec4 = _mm256_setzero_ps();

            for (size_t k = 0; k < K; k += 32) {
                _mm_prefetch((const char*)(in_row + k + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(w_row + k + 64), _MM_HINT_T0);
                
                __m256 in_vec1 = _mm256_loadu_ps(in_row + k);
                __m256 w_vec1  = _mm256_loadu_ps(w_row + k);
                sum_vec1 = _mm256_fmadd_ps(in_vec1, w_vec1, sum_vec1);

                __m256 in_vec2 = _mm256_loadu_ps(in_row + k + 8);
                __m256 w_vec2  = _mm256_loadu_ps(w_row + k + 8);
                sum_vec2 = _mm256_fmadd_ps(in_vec2, w_vec2, sum_vec2);

                __m256 in_vec3 = _mm256_loadu_ps(in_row + k + 16);
                __m256 w_vec3  = _mm256_loadu_ps(w_row + k + 16);
                sum_vec3 = _mm256_fmadd_ps(in_vec3, w_vec3, sum_vec3);

                __m256 in_vec4 = _mm256_loadu_ps(in_row + k + 24);
                __m256 w_vec4  = _mm256_loadu_ps(w_row + k + 24);
                sum_vec4 = _mm256_fmadd_ps(in_vec4, w_vec4, sum_vec4);
            }

            __m256 total_sum_vec = _mm256_add_ps(_mm256_add_ps(sum_vec1, sum_vec2), _mm256_add_ps(sum_vec3, sum_vec4));

            float sum = hsum_avx(total_sum_vec);

            out->buf[m * N + n] = sum * scale + b->buf[n] * lr_mul;
        }
    }
}

void FusedLinearLeakyReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  size_t M = out->shape[0], N = out->shape[1], K = w->shape[1];

  const float linear_scale = (1.0f / sqrtf(K)) * lr_mul;
  const float leaky_negative_slope = 0.2f;
  const float leaky_scale = sqrtf(2.0f);

  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      const float *in_row = in->buf + m * K;
      const float *w_row = w->buf + n * K;
      
      __m256 sum_vec = _mm256_setzero_ps();
      
      size_t k = 0;
      for (; k + 7 < K; k += 8) {
          __m256 in_vec = _mm256_loadu_ps(in_row + k);
          __m256 w_vec = _mm256_loadu_ps(w_row + k);
          sum_vec = _mm256_fmadd_ps(in_vec, w_vec, sum_vec);
      }

      float sum_array[8];
      _mm256_storeu_ps(sum_array, sum_vec);
      float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                  sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

      for (; k < K; k++) {
          sum += in_row[k] * w_row[k];
      }

      float val = sum * linear_scale + b->buf[n] * lr_mul;

      if (val < 0) {
          val *= leaky_negative_slope;
      }
      val *= leaky_scale;

      out->buf[m * N + n] = val;
    }
  }
}

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               size_t up, size_t pad0, size_t pad1, cudaStream_t *streams) {
  upsample_wrapper(input, upsample_a, up, pad0, pad1, false, false, streams);

  size_t N = upsample_a->shape[0];
  size_t C = upsample_a->shape[1];
  size_t H = upsample_a->shape[2];
  size_t W = upsample_a->shape[3];
  size_t OH = H - 3;
  size_t OW = W - 3;

  upsample_a->reshape({N*C, 1, H, W});
  output->reshape({N*C, 1, OH, OW});

  conv2d_same_wrapper(upsample_a, kernel, output, 1, 0, 1, false, false, false, streams);

  upsample_a->reshape({N, C, H, W});
  output->reshape({N, C, OH, OW});
}

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *col_buffer, Tensor *weight_transposed, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, size_t padding, size_t up, cudaStream_t *streams
) {
  size_t N = input->shape[0];
  size_t in_C = input->shape[1];
  size_t out_C = conv_weight->shape[0];
  size_t R = conv_weight->shape[2];
  size_t S = conv_weight->shape[3];
  size_t kernel_size = R * S;

  linear_wrapper(style, modulate_weight, modulate_bias, style_a, 1.0f, false, false, streams);

  float scale = 1.0f / sqrtf((float)(in_C * kernel_size));

  modulate_wrapper(conv_weight, style_a, weight_a, scale, out_C, in_C, R, S, false, false, false, streams);

  if (demodulate) {
    demodulate_wrapper(weight_a, false, false, in_C, R, S, streams);
  }

  if (upsample) {
    transpose_wrapper(weight_a, weight_transposed, false, false, streams);
    ConvTranspose2d_col2im(input, weight_transposed, conv_a, col_buffer, streams);
    upfir2d(conv_a, kernel, output, upsample_a, conv2_a, up, 1, 1, streams);
  }
  else {
    if (weight_a->shape[1] % 64 != 0) {
      Conv2d(input, weight_a, output, streams);
    } else {
      Conv2d_im2col(input, weight_a, output, col_buffer, streams);
    }
  }
}

void elemAdd(Tensor *inout, Tensor *addend) {
  const size_t N = inout->num_elem();
  float *__restrict__ p_inout = inout->buf;
  const float *__restrict__ p_addend = addend->buf;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N / 32; i++) {
    size_t base_idx = i * 32;

    _mm_prefetch((const char *)(p_addend + base_idx + 32), _MM_HINT_T0);
    _mm_prefetch((const char *)(p_inout + base_idx + 32), _MM_HINT_T0);

    __m256 vec_inout1 = _mm256_load_ps(p_inout + base_idx + 0);
    __m256 vec_addend1 = _mm256_load_ps(p_addend + base_idx + 0);
    vec_inout1 = _mm256_add_ps(vec_inout1, vec_addend1);
    _mm256_store_ps(p_inout + base_idx + 0, vec_inout1);

    __m256 vec_inout2 = _mm256_load_ps(p_inout + base_idx + 8);
    __m256 vec_addend2 = _mm256_load_ps(p_addend + base_idx + 8);
    vec_inout2 = _mm256_add_ps(vec_inout2, vec_addend2);
    _mm256_store_ps(p_inout + base_idx + 8, vec_inout2);
    
    __m256 vec_inout3 = _mm256_load_ps(p_inout + base_idx + 16);
    __m256 vec_addend3 = _mm256_load_ps(p_addend + base_idx + 16);
    vec_inout3 = _mm256_add_ps(vec_inout3, vec_addend3);
    _mm256_store_ps(p_inout + base_idx + 16, vec_inout3);

    __m256 vec_inout4 = _mm256_load_ps(p_inout + base_idx + 24);
    __m256 vec_addend4 = _mm256_load_ps(p_addend + base_idx + 24);
    vec_inout4 = _mm256_add_ps(vec_inout4, vec_addend4);
    _mm256_store_ps(p_inout + base_idx + 24, vec_inout4);
  }
}

void StyledConv(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *noise, Tensor *output,
                Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *col_buffer, Tensor *weight_transposed, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a, bool upsample, size_t padding, cudaStream_t *streams) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, demod_a, col_buffer, weight_transposed, conv_a, upsample_a, conv2_a,
                  true, upsample, padding, 1, streams);
  addNoiseBiasLeakyReLU_wrapper(output, noise, conv_bias, false, false, false, false, streams);
}

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *col_buffer, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a, cudaStream_t *streams) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, nullptr, col_buffer, nullptr, nullptr, nullptr, nullptr, false, false, 0, 2, streams);
    
  addBias_wrapper(output, conv_bias, false, false, false, streams);

  if (skip != nullptr) {
    upfir2d(skip, kernel, skip_a, skip_upsample_a, skip_conv_a, 2, 2, 1, streams);
    elemAdd_wrapper(output, skip_a, false, false, false, streams);
  }
}