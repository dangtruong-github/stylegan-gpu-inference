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
// This function orchestrates data movement, kernel launch, and timing 
// for a batched matrix multiplication operation across multiple GPUs.
void bmm_wrapper(Tensor *A, Tensor *B, Tensor *C, bool A_to_device, bool B_to_device, bool C_from_device, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel(); // CPU timer for overall wall-clock time
  double start_time, end_time;
  double to_device_time = 0.0;
  double from_device_time = 0.0;
  
  // CUDA events are the most accurate way to time GPU execution
  cudaEvent_t start_events[NUM_GPUS];
  cudaEvent_t stop_events[NUM_GPUS];
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventCreate(&start_events[i]));
      CHECK_CUDA(cudaEventCreate(&stop_events[i]));
  }
  // --- End of Timing variables ---

  // 1. Time Data Movement To GPUs
  start_time = get_time_kernel();
  if (A_to_device) A->to_device(streams);
  if (B_to_device) B->to_device(streams);

  // Block CPU until all data transfers are complete to get accurate timing
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }
  end_time = get_time_kernel();
  to_device_time = end_time - start_time;

  // 2. Get dimensions from tensor shapes
  const size_t batch_per_gpu = A->shape[0] / NUM_GPUS;
  const size_t M = A->shape[1];
  const size_t K = A->shape[2];
  const size_t N = B->shape[2];

  // 3. Time Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      // Record a start event in the specified stream for this GPU
      CHECK_CUDA(cudaEventRecord(start_events[gpu_id], streams[gpu_id]));
      
      // --- Inlined Kernel Launch Logic ---
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
      // --- End of Inlined Logic ---

      // Record a stop event in the specified stream for this GPU
      CHECK_CUDA(cudaEventRecord(stop_events[gpu_id], streams[gpu_id]));
  }
  
  // Synchronize events and find the maximum time across all GPUs
  float max_kernel_time_ms = 0.0f;
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      float current_gpu_time_ms = 0.0f;
      CHECK_CUDA(cudaSetDevice(gpu_id));
      // Wait for this GPU's stop event to complete
      CHECK_CUDA(cudaEventSynchronize(stop_events[gpu_id]));
      // Calculate the elapsed time between the start and stop events
      CHECK_CUDA(cudaEventElapsedTime(&current_gpu_time_ms, start_events[gpu_id], stop_events[gpu_id]));
      if (current_gpu_time_ms > max_kernel_time_ms) {
          max_kernel_time_ms = current_gpu_time_ms;
      }
  }
  double kernel_exec_time_s = max_kernel_time_ms / 1000.0;

  // 4. Time Data Movement From GPUs
  start_time = get_time_kernel();
  if (C_from_device) {
    C->from_device(streams);
  
    // Block CPU until all data transfers are complete
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
  end_time = get_time_kernel();
  from_device_time = end_time - start_time;

  // --- Cleanup ---
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventDestroy(start_events[i]));
      CHECK_CUDA(cudaEventDestroy(stop_events[i]));
  }
  CHECK_CUDA(cudaSetDevice(0)); // Reset to default device

  // --- Final Performance Report ---
  double func_end_time = get_time_kernel();
  double total_func_time = func_end_time - func_start_time;
  double sum_of_parts = to_device_time + kernel_exec_time_s + from_device_time;
  
  printf("\n--- bmm_wrapper (Multi-GPU) Timing Report ---\n");
  printf("Total Function Time         : %.6f s\n", total_func_time);
  printf("--------------------------------------------\n");
  printf("  - Data Transfer To GPUs   : %.6f s\n", to_device_time);
  printf("  - Kernel Execution (Max)  : %.6f s\n", kernel_exec_time_s);
  printf("  - Data Transfer From GPUs : %.6f s\n", from_device_time);
  printf("--------------------------------------------\n");
  printf("Sum of Timed Parts          : %.6f s\n", sum_of_parts);
  printf("Unaccounted CPU Overhead    : %.6f s\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
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
 * @param input The input tensor of shape (N, C, H, W).
 * @param col_buffer The output column buffer tensor.
 * @param input_to_device Flag to control transferring the input tensor to GPUs.
 * @param col_buffer_from_device Flag to control transferring the result back to the CPU.
 * @param streams Array of CUDA streams, one for each GPU.
 */
void im2col_wrapper(Tensor *input, Tensor *col_buffer, bool input_to_device, bool col_buffer_from_device, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double to_device_time = 0.0;
  double from_device_time = 0.0;
  
  cudaEvent_t start_events[NUM_GPUS];
  cudaEvent_t stop_events[NUM_GPUS];
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventCreate(&start_events[i]));
      CHECK_CUDA(cudaEventCreate(&stop_events[i]));
  }

  // 1. Time Data Movement To GPUs
  start_time = get_time_kernel();
  if (input_to_device) input->to_device(streams);
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }
  end_time = get_time_kernel();
  to_device_time = end_time - start_time;

  // 2. Get dimensions from tensor shapes
  const size_t N = input->shape[0];
  const size_t C = input->shape[1];
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];
  const size_t batch_per_gpu = N / NUM_GPUS;

  // 3. Time Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      CHECK_CUDA(cudaEventRecord(start_events[gpu_id], streams[gpu_id]));
      
      long long total_elements_gpu = (long long)batch_per_gpu * C * 3 * 3 * H * W;
      int blockSize = 256;
      int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
      
      im2col_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
          input->d_buf[gpu_id], 
          col_buffer->d_buf[gpu_id], 
          batch_per_gpu, C, H, W
      );
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(stop_events[gpu_id], streams[gpu_id]));
  }
  
  float max_kernel_time_ms = 0.0f;
  for (int i = 0; i < NUM_GPUS; ++i) {
      float current_gpu_time_ms = 0.0f;
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventSynchronize(stop_events[i]));
      CHECK_CUDA(cudaEventElapsedTime(&current_gpu_time_ms, start_events[i], stop_events[i]));
      if (current_gpu_time_ms > max_kernel_time_ms) {
          max_kernel_time_ms = current_gpu_time_ms;
      }
  }
  double kernel_exec_time_s = max_kernel_time_ms / 1000.0;

  // 4. Time Data Movement From GPUs
  start_time = get_time_kernel();
  if (col_buffer_from_device) {
    col_buffer->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
  end_time = get_time_kernel();
  from_device_time = end_time - start_time;

  // --- Cleanup & Final Performance Report ---
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventDestroy(start_events[i]));
      CHECK_CUDA(cudaEventDestroy(stop_events[i]));
  }
  CHECK_CUDA(cudaSetDevice(0));

  double total_func_time = get_time_kernel() - func_start_time;
  double sum_of_parts = to_device_time + kernel_exec_time_s + from_device_time;
  
  printf("\n--- im2col_wrapper (Multi-GPU) Timing Report ---\n");
  printf("Total Function Time         : %.6f s\n", total_func_time);
  printf("--------------------------------------------\n");
  printf("  - Data Transfer To GPUs   : %.6f s\n", to_device_time);
  printf("  - Kernel Execution (Max)  : %.6f s\n", kernel_exec_time_s);
  printf("  - Data Transfer From GPUs : %.6f s\n", from_device_time);
  printf("--------------------------------------------\n");
  printf("Sum of Timed Parts          : %.6f s\n", sum_of_parts);
  printf("Unaccounted CPU Overhead    : %.6f s\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
}


/**
 * @brief Multi-GPU wrapper for the col2im operation.
 * @param col_buffer The input column buffer tensor.
 * @param output The output tensor of shape (N, K, OH, OW).
 * @param H The original input height (for index calculation).
 * @param W The original input width (for index calculation).
 * @param col_buffer_to_device Flag to control transferring the column buffer to GPUs.
 * @param output_from_device Flag to control transferring the result back to the CPU.
 * @param streams Array of CUDA streams, one for each GPU.
 */
void col2im_wrapper(Tensor *col_buffer, Tensor *output, int H, int W, bool col_buffer_to_device, bool output_from_device, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double to_device_time = 0.0;
  double from_device_time = 0.0;
  
  cudaEvent_t start_events[NUM_GPUS];
  cudaEvent_t stop_events[NUM_GPUS];
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventCreate(&start_events[i]));
      CHECK_CUDA(cudaEventCreate(&stop_events[i]));
  }

  // 1. Time Data Movement To GPUs
  start_time = get_time_kernel();
  if (col_buffer_to_device) col_buffer->to_device(streams);
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }
  end_time = get_time_kernel();
  to_device_time = end_time - start_time;

  // 2. Get dimensions from tensor shapes
  const size_t N = output->shape[0];
  const size_t K = output->shape[1];
  const size_t OH = output->shape[2];
  const size_t OW = output->shape[3];
  const size_t batch_per_gpu = N / NUM_GPUS;

  // 3. Time Kernel Execution
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
      CHECK_CUDA(cudaSetDevice(gpu_id));
      
      // IMPORTANT: Initialize output buffer on GPU to zeros before atomic additions
      size_t output_bytes_per_gpu = batch_per_gpu * K * OH * OW * sizeof(float);
      CHECK_CUDA(cudaMemsetAsync(output->d_buf[gpu_id], 0, output_bytes_per_gpu, streams[gpu_id]));

      CHECK_CUDA(cudaEventRecord(start_events[gpu_id], streams[gpu_id]));
      
      long long total_elements_gpu = (long long)batch_per_gpu * K * 3 * 3 * H * W;
      int blockSize = 256;
      int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
      
      col2im_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
          col_buffer->d_buf[gpu_id],
          output->d_buf[gpu_id],
          batch_per_gpu, K, OH, OW, H, W
      );
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(stop_events[gpu_id], streams[gpu_id]));
  }
  
  float max_kernel_time_ms = 0.0f;
  for (int i = 0; i < NUM_GPUS; ++i) {
      float current_gpu_time_ms = 0.0f;
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventSynchronize(stop_events[i]));
      CHECK_CUDA(cudaEventElapsedTime(&current_gpu_time_ms, start_events[i], stop_events[i]));
      if (current_gpu_time_ms > max_kernel_time_ms) {
          max_kernel_time_ms = current_gpu_time_ms;
      }
  }
  double kernel_exec_time_s = max_kernel_time_ms / 1000.0;

  // 4. Time Data Movement From GPUs
  start_time = get_time_kernel();
  if (output_from_device) {
    output->from_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  }
  end_time = get_time_kernel();
  from_device_time = end_time - start_time;

  // --- Cleanup & Final Performance Report ---
  for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaEventDestroy(start_events[i]));
      CHECK_CUDA(cudaEventDestroy(stop_events[i]));
  }
  CHECK_CUDA(cudaSetDevice(0));

  double total_func_time = get_time_kernel() - func_start_time;
  double sum_of_parts = to_device_time + kernel_exec_time_s + from_device_time;
  
  printf("\n--- col2im_wrapper (Multi-GPU) Timing Report ---\n");
  printf("Total Function Time         : %.6f s\n", total_func_time);
  printf("--------------------------------------------\n");
  printf("  - Data Transfer To GPUs   : %.6f s\n", to_device_time);
  printf("  - Kernel Execution (Max)  : %.6f s\n", kernel_exec_time_s);
  printf("  - Data Transfer From GPUs : %.6f s\n", from_device_time);
  printf("--------------------------------------------\n");
  printf("Sum of Timed Parts          : %.6f s\n", sum_of_parts);
  printf("Unaccounted CPU Overhead    : %.6f s\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
}


/**
 * @brief Multi-GPU wrapper for the 5D transpose operation.
 * @param weight The input tensor with shape (N, K, C, R, S).
 * @param weight_transpose The output tensor with shape (N, K, R, S, C).
 * @param weight_to_device Flag to control transferring the input tensor to GPUs.
 * @param transpose_from_device Flag to control transferring the result back to the CPU.
 * @param streams Array of CUDA streams, one for each GPU.
 */
void transpose_wrapper(Tensor *weight, Tensor *weight_transpose, bool weight_to_device, bool transpose_from_device, cudaStream_t *streams) {
    // --- Timing variables ---
    double func_start_time = get_time_kernel();
    double start_time, end_time;
    double to_device_time = 0.0;
    double from_device_time = 0.0;
    
    cudaEvent_t start_events[NUM_GPUS];
    cudaEvent_t stop_events[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaEventCreate(&start_events[i]));
        CHECK_CUDA(cudaEventCreate(&stop_events[i]));
    }

    // 1. Time Data Movement To GPUs
    start_time = get_time_kernel();
    if (weight_to_device) weight->to_device(streams);
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    end_time = get_time_kernel();
    to_device_time = end_time - start_time;

    // 2. Get dimensions from tensor shapes
    const size_t N = weight->shape[0];
    const size_t K = weight->shape[1];
    const size_t C = weight->shape[2];
    const size_t R = weight->shape[3];
    const size_t S = weight->shape[4];
    const size_t batch_per_gpu = N / NUM_GPUS;

    // 3. Time Kernel Execution
    for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
        CHECK_CUDA(cudaSetDevice(gpu_id));
        CHECK_CUDA(cudaEventRecord(start_events[gpu_id], streams[gpu_id]));
        
        long long total_elements_gpu = (long long)batch_per_gpu * K * C * R * S;
        int blockSize = 256;
        int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
        
        transpose_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
            weight->d_buf[gpu_id],
            weight_transpose->d_buf[gpu_id],
            batch_per_gpu, K, C, R, S
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop_events[gpu_id], streams[gpu_id]));
    }
    
    float max_kernel_time_ms = 0.0f;
    for (int i = 0; i < NUM_GPUS; ++i) {
        float current_gpu_time_ms = 0.0f;
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaEventSynchronize(stop_events[i]));
        CHECK_CUDA(cudaEventElapsedTime(&current_gpu_time_ms, start_events[i], stop_events[i]));
        if (current_gpu_time_ms > max_kernel_time_ms) {
            max_kernel_time_ms = current_gpu_time_ms;
        }
    }
    double kernel_exec_time_s = max_kernel_time_ms / 1000.0;

    // 4. Time Data Movement From GPUs
    start_time = get_time_kernel();
    if (transpose_from_device) {
      weight_transpose->from_device(streams);
      for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
      }
    }
    end_time = get_time_kernel();
    from_device_time = end_time - start_time;

    // --- Cleanup & Final Performance Report ---
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaEventDestroy(start_events[i]));
        CHECK_CUDA(cudaEventDestroy(stop_events[i]));
    }
    CHECK_CUDA(cudaSetDevice(0));

    double total_func_time = get_time_kernel() - func_start_time;
    double sum_of_parts = to_device_time + kernel_exec_time_s + from_device_time;
    
    printf("\n--- transpose_wrapper (Multi-GPU) Timing Report ---\n");
    printf("Total Function Time         : %.6f s\n", total_func_time);
    printf("--------------------------------------------\n");
    printf("  - Data Transfer To GPUs   : %.6f s\n", to_device_time);
    printf("  - Kernel Execution (Max)  : %.6f s\n", kernel_exec_time_s);
    printf("  - Data Transfer From GPUs : %.6f s\n", from_device_time);
    printf("--------------------------------------------\n");
    printf("Sum of Timed Parts          : %.6f s\n", sum_of_parts);
    printf("Unaccounted CPU Overhead    : %.6f s\n", total_func_time - sum_of_parts);
    printf("--- End of Report ---\n\n");
}


// -------------- MAIN FUNCTION ---------------------

/**
 * @brief Horizontally sums the 8 float values in an AVX __m256 vector.
 * @param vec The __m256 vector.
 * @return The scalar float sum.
 */
static inline float horizontal_sum_avx(__m256 vec) {
  // Extract upper and lower 128-bit lanes and add them together
  __m128 lo = _mm256_castps256_ps128(vec);
  __m128 hi = _mm256_extractf128_ps(vec, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  // Use horizontal adds to sum the remaining 4 floats
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(sum128);
}

/**
 * @brief An optimized PixelNorm function using AVX2 intrinsics.
 *
 * This version is optimized for C=512 and small N by focusing
 * on single-thread, vectorized performance. It assumes C is a multiple of 32
 * and that the input buffer is 32-byte aligned for maximum speed.
 *
 * @param [in & out] inout Pointer to a Tensor struct.
 */
void PixelNorm(Tensor *inout) {
  const size_t N = inout->shape[0];
  const size_t C = inout->shape[1];
  // Use the `restrict` keyword to hint to the compiler that this pointer
  // is the only one used to access the memory, enabling better optimization.
  float *__restrict__ buf = inout->buf;

  // Process each row in the batch. For small N, thread overhead is too high.
  for (size_t n = 0; n < N; ++n) {
    float *row_ptr = buf + n * C;

    // --- Pass 1: Calculate sum of squares using AVX2 with 4x unrolling ---
    // Using four accumulators helps hide instruction latency and increases parallelism.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    // Process 32 floats (4 vectors of 8 floats) per loop iteration.
    // NOTE: This assumes C is a multiple of 32 (512 is 16 * 32).
    for (size_t i = 0; i < C; i += 32) {
      // NOTE: Using aligned loads (_mm256_load_ps) requires the input `buf`
      // to be 32-byte aligned. This is much faster than unaligned loads.
      __m256 v0 = _mm256_load_ps(row_ptr + i);
      __m256 v1 = _mm256_load_ps(row_ptr + i + 8);
      __m256 v2 = _mm256_load_ps(row_ptr + i + 16);
      __m256 v3 = _mm256_load_ps(row_ptr + i + 24);

      // Use fused multiply-add (FMA) for v*v + acc. FMA is standard with AVX2.
      acc0 = _mm256_fmadd_ps(v0, v0, acc0);
      acc1 = _mm256_fmadd_ps(v1, v1, acc1);
      acc2 = _mm256_fmadd_ps(v2, v2, acc2);
      acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }

    // Consolidate the four accumulators into one
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    // Horizontally sum the final vector to get the total sum of squares
    const float sum_squares = horizontal_sum_avx(acc0);

    // --- Calculate normalization factor using fast reciprocal square root ---
    const float mean_squares = sum_squares / (float)C;
    const float val_to_rsqrt = mean_squares + 1e-8f;

    // Broadcast the scalar value to a full AVX vector to process in SIMD
    const __m256 val_vec = _mm256_set1_ps(val_to_rsqrt);
    
    // Calculate fast approximate reciprocal square root using AVX2
    __m256 norm_vec = _mm256_rsqrt_ps(val_vec);

    // Refine the result with one Newton-Raphson iteration for better precision:
    // x_new = 0.5 * x_old * (3 - (d * x_old^2))
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256 muls = _mm256_mul_ps(_mm256_mul_ps(val_vec, norm_vec), norm_vec);
    norm_vec = _mm256_mul_ps(_mm256_mul_ps(half, norm_vec), _mm256_sub_ps(three, muls));

    // --- Pass 2: Scale the row using the normalization vector ---
    // Unroll this loop as well to match the read/write pattern and maximize throughput.
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

/*
 * Optimized Upsample and Pad
 *
 * This version avoids the expensive global `memset` by fusing padding (zeroing)
 * and data copying into a single, cache-friendly parallel loop.
 *
 * Optimizations Applied:
 * 1.  No Global `memset`: The entire output tensor is never zeroed at once. Instead,
 * padding is applied selectively around the copied data.
 * 2.  Fused Operations: Zero-padding and data upsampling are done in the same loop,
 * improving temporal locality and reducing memory passes.
 * 3.  Improved Parallelism: The loop is parallelized over the (N, C) dimensions,
 * which is more effective for this memory layout than collapsing deeper loops.
 * This gives each thread an independent channel to process, reducing synchronization.
 * 4.  Improved Memory Locality: Base pointers are calculated for each channel and row,
 * and the innermost loop iterates linearly, which is much friendlier to the CPU cache.
 * 5.  Fast Path for up=1: A separate, highly optimized path using `memcpy` is added
 * for the common case where `up == 1` (padding only, no upsampling).
 */
void UpsamplePad(Tensor *input, Tensor *output, size_t up, size_t pad0, size_t pad1) {
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];
  size_t OH = H * up + pad0 + pad1;
  size_t OW = W * up + pad0 + pad1;

  // --- Fast Path for up=1 (Padding Only) ---
  // If we are not upsampling, the operation is just padding. This can be done
  // much more efficiently by copying entire rows of data between the padded regions.
  if (up == 1) {
    #pragma omp parallel for
    for (size_t nc = 0; nc < N * C; ++nc) {
      // Base pointers for the start of the current channel's data
      float *out_channel_ptr = output->buf + nc * OH * OW;
      const float *in_channel_ptr = input->buf + nc * H * W;

      // 1. Zero top padding
      if (pad0 > 0) {
        memset(out_channel_ptr, 0, pad0 * OW * sizeof(float));
      }

      // 2. Copy H rows from input to output, adding left/right padding for each
      float *out_row_ptr = out_channel_ptr + pad0 * OW;
      const float *in_row_ptr = in_channel_ptr;
      for (size_t h = 0; h < H; ++h) {
        // Zero left padding
        if (pad0 > 0) {
          memset(out_row_ptr, 0, pad0 * sizeof(float));
        }
        // Copy data
        memcpy(out_row_ptr + pad0, in_row_ptr, W * sizeof(float));
        // Zero right padding
        if (pad1 > 0) {
          memset(out_row_ptr + pad0 + W, 0, pad1 * sizeof(float));
        }
        out_row_ptr += OW;
        in_row_ptr += W;
      }

      // 3. Zero bottom padding
      if (pad1 > 0) {
        memset(out_channel_ptr + (pad0 + H) * OW, 0, pad1 * OW * sizeof(float));
      }
    }
  }
  // --- General Path for up > 1 (Upsampling + Padding) ---
  else {
    #pragma omp parallel for
    for (size_t nc = 0; nc < N * C; ++nc) {
      float *out_channel_ptr = output->buf + nc * OH * OW;
      const float *in_channel_ptr = input->buf + nc * H * W;

      // 1. Zero top padding section
      if (pad0 > 0) {
        memset(out_channel_ptr, 0, pad0 * OW * sizeof(float));
      }

      // 2. Process each row: upsample data and fill gaps with zeros
      for (size_t h = 0; h < H; ++h) {
        // Pointer to the start of the current destination row in the padded area
        float *out_row_ptr = out_channel_ptr + (h * up + pad0) * OW;
        const float *in_row_ptr = in_channel_ptr + h * W;

        // Zero the left padding for this row
        if (pad0 > 0) {
            memset(out_row_ptr, 0, pad0 * sizeof(float));
        }

        // Copy input values to upsampled positions and zero the gaps in between
        float *out_pixel_ptr = out_row_ptr + pad0;
        for (size_t w = 0; w < W; ++w) {
            // Copy the actual value
            *out_pixel_ptr = in_row_ptr[w];
            
            // Zero the newly created gaps from upsampling
            for (size_t i = 1; i < up; ++i) {
                *(out_pixel_ptr + i) = 0.0f;
            }
            out_pixel_ptr += up;
        }
        
        // Zero the right padding for this row
        if (pad1 > 0) {
            memset(out_pixel_ptr, 0, pad1 * sizeof(float));
        }
      }

      // 3. Zero the gaps between the upsampled rows
      for (size_t h = 0; h < H; ++h) {
        for (size_t i = 1; i < up; ++i) {
          float *gap_row_ptr = out_channel_ptr + (h * up + pad0 + i) * OW;
          memset(gap_row_ptr, 0, OW * sizeof(float));
        }
      }

      // 4. Zero bottom padding section
      if (pad1 > 0) {
        memset(out_channel_ptr + (pad0 + H * up) * OW, 0, pad1 * OW * sizeof(float));
      }
    }
  }
}

/*
 * Convolution (with per-sample weights)
 * input shape = (N, C, H, W)
 * weight shape = (N, K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 * where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 * OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 * pad = 1, dilation = 1, stride = 1
 */
void Conv2d(Tensor *input, Tensor *weight, Tensor *output, cudaStream_t* streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  // --- End of Timing variables ---

  size_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  size_t K = weight->shape[1];
  size_t OH = output->shape[2], OW = output->shape[3];

  weight->reshape({N, K, C});
  input->reshape({N, C, H*W});
  output->reshape({N, K, OH*OW});
  bmm_wrapper(weight, input, output, true, true, true, streams);

  weight->reshape({N, K, C, 1, 1});
  input->reshape({N, C, H, W});
  output->reshape({N, K, OH, OW});

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_func_time = func_end_time - func_start_time;

  /*
  printf("\n--- Conv2d (1x1 Kernel) Timing Report ---\n");
  input->printShape("input");
  weight->printShape("weight");
  output->printShape("output");
  printf("N: %zu, K: %zu, OH: %zu, OW: %zu, C: %zu\n", N, K, OH, OW, C);
  printf("Total Function Time   : %.6f s\n", total_func_time);
  printf("----------------------------------------\n");
  printf("  (Executed Path: %s)\n", (OW == 4) ? "Specialized OW=4" : "Standard OW>=8");
  printf("--- End of Report ---\n\n");
  */
}

/*
 * Convolution
 * input shape = (N, C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 * where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 * OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */
void Conv2d_same(Tensor *input, Tensor *weight, Tensor *output,
                int stride, int pad, int dilation) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  // --- End of Timing variables ---

  size_t N = input->shape[0], H = input->shape[2], W = input->shape[3];
  size_t OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for schedule(static) collapse(2)
  for (size_t n = 0; n < N; ++n) {
    // Unroll the 'oh' loop by a factor of 2. Step by 2 rows at a time.
    for (size_t oh = 0; oh < OH; oh += 2) {
      for (size_t ow = 0; ow < OW; ow += 8) { // vectorize 8 outputs at once
        // Accumulators for 2 output rows (8 floats each)
        __m256 acc0 = _mm256_setzero_ps(); // Accumulator for row 'oh'
        __m256 acc1 = _mm256_setzero_ps(); // Accumulator for row 'oh + 1'

        // Pre-calculate base pointers to reduce indexing inside the hot loop
        const float* in_base = &input->buf[n * H * W];
        const float* w_base = &weight->buf[0];

        for (int r = 0; r < 4; ++r) {
          for (int s = 0; s < 4; ++s) {
            // Calculate input coordinates for the two rows
            size_t h0 = oh + r;
            size_t h1 = oh + 1 + r;
            size_t w = ow + s;

            // Broadcast kernel weight scalar to 8-wide vector
            // This weight is the same for both rows being processed
            float f = w_base[r * 4 + s];
            __m256 filt = _mm256_set1_ps(f);

            // Load 8 input pixels for the first row
            __m256 in0 = _mm256_load_ps(&in_base[h0 * W + w]);
            // Load 8 input pixels for the second row
            __m256 in1 = _mm256_load_ps(&in_base[h1 * W + w]);

            // Fused Multiply-Add for the first accumulator
            acc0 = _mm256_fmadd_ps(in0, filt, acc0); // acc0 += in0 * filt
            // Fused Multiply-Add for the second accumulator
            acc1 = _mm256_fmadd_ps(in1, filt, acc1); // acc1 += in1 * filt
          }
        }

        // Pre-calculate output base pointer
        float* out_base = &output->buf[n * OH * OW];
        
        // Store result for the first row
        _mm256_store_ps(&out_base[oh * OW + ow], acc0);
        // Store result for the second row
        _mm256_store_ps(&out_base[(oh + 1) * OW + ow], acc1);
      }
    }
  }

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_time = func_end_time - func_start_time;

  /*
  printf("\n--- Conv2d_same Timing Report ---\n");
  input->printShape("input");
  weight->printShape("weight");
  output->printShape("output");
  printf("N: %zu, OH: %zu, OW: %zu, stride: %d, pad: %d, dilation: %d\n", N, OH, OW, stride, pad, dilation);
  printf("Total Function Time: %.6f s\n", total_time);
  printf("--- End of Report ---\n\n");
  */
}

/*
 * Convolution (with per-sample weights)
 * input shape = (N, C, H, W)
 * weight shape = (N, K, C, R, S)
 * col_buffer shape = (N, OH*OW, C*R*S)
 * output shape = (N, K, OH, OW)
 * where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 * OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 * pad = 1, dilation = 1, stride = 1, R = S = 3
 */
void Conv2d_im2col(Tensor *input, Tensor *weight, Tensor *output, Tensor *col_buffer, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double im2col_time = 0.0;
  double bmm_time = 0.0;
  // --- End of Timing variables ---

  // 2. ---- Get dimensions from input tensors ----
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t K = weight->shape[1];
  size_t R = weight->shape[3];
  size_t S = weight->shape[4];
  // We get OH and OW from the output tensor's shape.
  size_t OH = output->shape[2];
  size_t OW = output->shape[3];

  size_t H = input->shape[2];
  size_t W = input->shape[3];

  // 3. ---- Call im2col to populate the provided buffer ----
  // This assumes `col_buffer` has a bmm-compatible shape of (N, OH*OW, C*R*S)
  start_time = get_time_kernel();
  im2col_wrapper(input, col_buffer, true, false, streams);
  end_time = get_time_kernel();
  im2col_time = end_time - start_time;

  // 4. ---- Create "views" for batched matrix multiplication ----
  // A "view" re-interprets the shape of existing data without copying it.

  // View of weights as (N, K, C*R*S)
  weight->reshape({N, K, (size_t)(C * R * S)});

  // View of the output as (N, K, OH*OW)
  output->reshape({N, K, (size_t)(OH * OW)});

  // 5. ---- Perform batched matrix multiplication ----
  // Note that `col_buffer` is already in the correct format and can be used directly.
  // output_view = weight_view @ col_buffer
  start_time = get_time_kernel();
  bmm_wrapper(weight, col_buffer, output, true, false, true, streams);
  end_time = get_time_kernel();
  bmm_time = end_time - start_time;

  // Restore original shapes
  weight->reshape({N, K, C, R, S});
  output->reshape({N, K, OH, OW});

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_func_time = func_end_time - func_start_time;
  double sum_of_parts = im2col_time + bmm_time;

  /*
  printf("\n--- Conv2d_im2col Timing Report ---\n");
  input->printShape("input Conv2d_im2col");
  col_buffer->printShape("col_buffer Conv2d_im2col");
  weight->printShape("weight Conv2d_im2col");
  output->printShape("output Conv2d_im2col");
  printf("N: %zu, C: %zu, K: %zu, H: %zu, W: %zu, OH: %zu, OW: %zu\n", N, C, K, H, W, OH, OW);
  printf("Total Function Time : %.6f s\n", total_func_time);
  printf("----------------------------------\n");
  printf("  - im2col          : %.6f s\n", im2col_time);
  printf("  - bmm_optimized   : %.6f s\n", bmm_time);
  printf("----------------------------------\n");
  printf("Sum of Timed Parts  : %.6f s\n", sum_of_parts);
  printf("Unaccounted Time    : %.6f s (e.g., reshape)\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
  */
}

/**
 *
 * This function implements transposed convolution by transforming the operation
 * into a standard forward convolution on a modified input tensor.
 * input shape = (N, C, H, W)
 * weight shape = (N, K, C, R, S)
 * output shape = (N, K, OH, OW)
 * where OH = (H - 1) * stride - 2 * pad + R
 * OW = (W - 1) * stride - 2 * pad + S
 * stride = 2 and pad = 0
 * weight_transpose (N, K, R, S, C)
 * col_buffer (N, KRS, HW)
 */
void ConvTranspose2d_col2im(Tensor *input, Tensor *weight, Tensor *output,
                            Tensor *col_buffer, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double bmm_time = 0.0;
  double col2im_time = 0.0;
  // --- End of Timing variables ---

  // Extract input dimensions for col2im
  const size_t H = input->shape[2];
  const size_t W = input->shape[3];
  const size_t C = input->shape[1];

  const size_t N = input->shape[0]; // Batch size
  const size_t K = weight->shape[1]; // Output channels

  // Kernel dimensions and parameters
  const int R = 3, S = 3;

  // 2. Perform matmul: (N, K*R*S, C) and (N, C, H*W) --> (N, K*R*S, H*W)
  weight->reshape({N, (size_t)(K * R * S), C});
  input->reshape({N, C, (size_t)(H * W)});
  start_time = get_time_kernel();
  bmm_wrapper(weight, input, col_buffer, false, true, false, streams);
  end_time = get_time_kernel();
  bmm_time = end_time - start_time;
  input->reshape({N, C, H, W});
  weight->reshape({N, K, (size_t)R, (size_t)S, C});

  // 3. col2im: Transform the columns back into the output image
  start_time = get_time_kernel();
  // col2im(col_buffer, output, H, W);
  col2im_wrapper(col_buffer, output, H, W, false, true, streams);
  end_time = get_time_kernel();
  col2im_time = end_time - start_time;

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_func_time = func_end_time - func_start_time;
  double sum_of_parts = bmm_time + col2im_time;

  /*
  printf("\n--- ConvTranspose2d_col2im Timing Report ---\n");
  printf("Total Function Time   : %.6f s\n", total_func_time);
  printf("------------------------------------------\n");
  printf("  - bmm_optimized     : %.6f s\n", bmm_time);
  printf("  - col2im            : %.6f s\n", col2im_time);
  printf("------------------------------------------\n");
  printf("Sum of Timed Parts    : %.6f s\n", sum_of_parts);
  printf("Unaccounted Time      : %.6f s (e.g., reshape)\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
  */
}

/**
 * @brief Horizontally sums the 8 float values in a 256-bit AVX register.
 * @param v The __m256 vector to sum.
 * @return The scalar float sum.
 */
static inline float hsum_avx(__m256 v) {
    // Extract the high 128 bits and add to the low 128 bits
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    // Use horizontal adds to sum the 4 floats in the 128-bit result
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* Optimized Linear Layer
 * @param [in1]   in: [M, K]
 * @param [in2]    w: [N, K]
 * @param [in3]    b: [N]
 * @param [out]  out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1]; // K=512

    // Hoist invariant: Calculate the scaling factor once outside all loops.
    float scale = (1.0f / sqrtf(K)) * lr_mul;

    // 1. Reduce Threading Overhead: Parallelize across both M and N dimensions.
    // The `collapse(2)` clause treats the nested loops as a single large loop,
    // which allows OpenMP to distribute the M*N work items efficiently among
    // threads, minimizing the overhead of creating/destroying parallel regions.
    #pragma omp parallel for collapse(2)
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            const float *in_row = in->buf + m * K;
            const float *w_row = w->buf + n * K;

            // 2. Vectorize & Unroll Inner Loop: Use AVX and loop unrolling.
            // Four __m256 registers are used as accumulators. This unrolls the
            // loop by 4, processing 32 floats per iteration (4 vectors * 8 floats/vector).
            // This hides instruction latency and maximizes computational throughput.
            __m256 sum_vec1 = _mm256_setzero_ps();
            __m256 sum_vec2 = _mm256_setzero_ps();
            __m256 sum_vec3 = _mm256_setzero_ps();
            __m256 sum_vec4 = _mm256_setzero_ps();

            // Since K=512 is a multiple of 32, no remainder loop is needed.
            for (size_t k = 0; k < K; k += 32) {
                // 3. Prefetch Data: Hint to the CPU to load data for a future
                // iteration into the cache, reducing memory latency stalls.
                _mm_prefetch((const char*)(in_row + k + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(w_row + k + 64), _MM_HINT_T0);
                
                // Load data using unaligned loads for robustness.
                // Perform Fused Multiply-Add (FMA): sum_vec += in_vec * w_vec
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

            // Consolidate the four accumulator vectors.
            __m256 total_sum_vec = _mm256_add_ps(_mm256_add_ps(sum_vec1, sum_vec2), _mm256_add_ps(sum_vec3, sum_vec4));

            // Horizontally sum the final vector to get the scalar dot product.
            float sum = hsum_avx(total_sum_vec);

            // 4. Fuse Operations: The scaling and bias addition are performed
            // immediately after the dot product, before writing to memory. This
            // reduces memory bandwidth requirements.
            out->buf[m * N + n] = sum * scale + b->buf[n] * lr_mul;
        }
    }
}

/* FusedLinearLeakyReLU (Optimized)
 * This version is optimized with SIMD (AVX/FMA), enhanced parallelism,
 * and other best practices based on the provided suggestions.
 * @param [in1]   in: [M, K]
 * @param [in2]    w: [N, K]
 * @param [in3]    b: [N]
 * @param [out]  out: [M, N]
 */
void FusedLinearLeakyReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul) {
  size_t M = out->shape[0], N = out->shape[1], K = w->shape[1];

  // --- Precomputed Constants (Suggestion #3) ---
  // Moved outside the loops to avoid redundant calculations.
  const float linear_scale = (1.0f / sqrtf(K)) * lr_mul;
  const float leaky_negative_slope = 0.2f;
  const float leaky_scale = sqrtf(2.0f);

  // --- Parallelize over both M and N (Suggestion #4) ---
  // `collapse(2)` creates M*N work items for better thread distribution,
  // especially when M is small. `schedule(static)` is efficient for
  // uniform workloads, reducing synchronization overhead (Suggestion #7).
  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      // --- Cache Row Pointers (Suggestion #6) ---
      // Pointers are calculated once per (m, n) pair, not inside the k-loop.
      const float *in_row = in->buf + m * K;
      const float *w_row = w->buf + n * K;
      
      // --- SIMD Vectorized Dot Product (Suggestion #2) ---
      // Use 256-bit AVX registers to process 8 floats simultaneously.
      __m256 sum_vec = _mm256_setzero_ps();
      
      size_t k = 0;
      // Main loop: process 8 elements at a time using Fused Multiply-Add.
      for (; k + 7 < K; k += 8) {
          // `loadu` is used for potentially unaligned memory access.
          __m256 in_vec = _mm256_loadu_ps(in_row + k);
          __m256 w_vec = _mm256_loadu_ps(w_row + k);
          // FMA (a*b + c) is faster than separate multiply and add instructions.
          sum_vec = _mm256_fmadd_ps(in_vec, w_vec, sum_vec);
      }

      // Horizontal sum: reduce the 8-float vector to a single float.
      float sum_array[8];
      _mm256_storeu_ps(sum_array, sum_vec);
      float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                  sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

      // Remainder loop: process any leftover elements if K is not a multiple of 8.
      for (; k < K; k++) {
          sum += in_row[k] * w_row[k];
      }

      // --- Fusion Point with Branch-Efficient LeakyReLU (Suggestion #5) ---
      float val = sum * linear_scale + b->buf[n] * lr_mul;

      // This simple `if` is highly efficient for a scalar value. Modern
      // compilers will likely convert it to a branchless `cmov` instruction.
      if (val < 0) {
          val *= leaky_negative_slope;
      }
      val *= leaky_scale;

      // Direct write to the final output location.
      out->buf[m * N + n] = val;
    }
  }
}

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               size_t up, size_t pad0, size_t pad1) {
  UpsamplePad(input, upsample_a, up, pad0, pad1);

  size_t N = upsample_a->shape[0];
  size_t C = upsample_a->shape[1];
  size_t H = upsample_a->shape[2];
  size_t W = upsample_a->shape[3];
  size_t OH = H - 3;
  size_t OW = W - 3;

  // output->printShape("output upfir2d");
  // vector<size_t> orig_output_shape = output->shape;
  upsample_a->reshape({N*C, 1, H, W});
  output->reshape({N*C, 1, OH, OW});

  // Conv2D needs to handle 4D kernel
  Conv2d_same(upsample_a, kernel, output, 1, 0, 1);

  upsample_a->reshape({N, C, H, W});
  output->reshape({N, C, OH, OW});
}

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *col_buffer, Tensor *weight_transposed, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, size_t padding, size_t up, cudaStream_t *streams
) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double linear_time = 0.0;
  double modulation_time = 0.0;
  double demodulation_time = 0.0;
  double transpose_time = 0.0;
  double conv_transpose_time = 0.0;
  double upfir2d_time = 0.0;
  double conv2d_time = 0.0;
  const char* conv_path_taken = "None";
  // --- End of Timing variables ---

  size_t N = input->shape[0];
  size_t in_C = input->shape[1];
  size_t out_C = conv_weight->shape[0];
  size_t R = conv_weight->shape[2];
  size_t S = conv_weight->shape[3];
  size_t kernel_size = R * S;

  start_time = get_time_kernel();
  Linear(style, modulate_weight, modulate_bias, style_a, 1.0f);
  end_time = get_time_kernel();
  linear_time = end_time - start_time;

  // The weight tensor has shape (out_C, in_C, R, S)
  // Let's reshape it logically to (out_C, in_C * kernel_size) for easier processing
  size_t weight_inner_dim = in_C * kernel_size;
  float scale = 1.0f / sqrtf((float)(in_C * kernel_size));

  // --- Modulation Step ---
  start_time = get_time_kernel();
  #pragma omp parallel for collapse(3)
  for (size_t n = 0; n < N; ++n) {
    for (size_t oc = 0; oc < out_C; ++oc) {
      for (size_t ic = 0; ic < in_C; ++ic) {
        float style_val = style_a->buf[n * in_C + ic] * scale;
        size_t conv_weight_idx = oc * weight_inner_dim + ic * kernel_size;
        size_t weight_a_idx = n * out_C * weight_inner_dim + oc * weight_inner_dim + ic * kernel_size;

        for (size_t k = 0; k < kernel_size; ++k) {
          weight_a->buf[weight_a_idx + k] = conv_weight->buf[conv_weight_idx + k] * style_val;
        }
      }
    }
  }
  end_time = get_time_kernel();
  modulation_time = end_time - start_time;

  if (demodulate) {
    start_time = get_time_kernel();
    // --- Demodulation Step ---
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
      size_t n_offset = n * out_C * in_C * R * S;
      for (size_t oc = 0; oc < out_C; oc++) {
        float sum_sq = 0.0f;
        size_t oc_offset = oc * weight_inner_dim;
        for (size_t ic = 0; ic < in_C; ic++) {
          size_t ic_offset = ic * kernel_size;
          for (size_t k = 0; k < kernel_size; k++) {
            float w = weight_a->buf[n_offset + oc_offset + ic_offset + k];
            sum_sq += w * w;
          }
        }
        float demod_factor = rsqrtf(sum_sq + 1e-8f);
        for (size_t i = 0; i < weight_inner_dim; ++i) {
          weight_a->buf[n_offset + oc_offset + i] *= demod_factor;
        }
      }
    }
      end_time = get_time_kernel();
      demodulation_time = end_time - start_time;
  }

  if (upsample) {
    conv_path_taken = "Upsample Path";
    start_time = get_time_kernel();
    // transpose(weight_a, weight_transposed);
    transpose_wrapper(weight_a, weight_transposed, true, false, streams);
    end_time = get_time_kernel();
    transpose_time = end_time - start_time;
    
    start_time = get_time_kernel();
    ConvTranspose2d_col2im(input, weight_transposed, conv_a, col_buffer, streams);
    end_time = get_time_kernel();
    conv_transpose_time = end_time - start_time;

    start_time = get_time_kernel();
    upfir2d(conv_a, kernel, output, upsample_a, conv2_a, up, 1, 1);
    end_time = get_time_kernel();
    upfir2d_time = end_time - start_time;
  }
  else {
    start_time = get_time_kernel();
    // The optimized path requires K (output channels) to be a multiple of 64.
    if (weight_a->shape[1] % 64 != 0) {
      conv_path_taken = "Conv2d (Standard)";
      Conv2d(input, weight_a, output, streams);
    } else {
      conv_path_taken = "Conv2d_Optimized";
      Conv2d_im2col(input, weight_a, output, col_buffer, streams);
    }
    end_time = get_time_kernel();
    conv2d_time = end_time - start_time;
  }

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_func_time = func_end_time - func_start_time;
  double sum_of_parts = linear_time + modulation_time + demodulation_time +
                      transpose_time + conv_transpose_time + upfir2d_time + conv2d_time;

  /*
  printf("\n--- ModulatedConv2d Timing Report ---\n");
  printf("Total Function Time : %.6f s\n", total_func_time);
  printf("-------------------------------------\n");
  printf("  - Linear (Style)    : %.6f s\n", linear_time);
  printf("  - Modulation        : %.6f s\n", modulation_time);
  printf("  - Demodulation      : %.6f s%s\n", demodulation_time, demodulate ? "" : " (skipped)");

  if (upsample) {
      printf("  - Upsample Path:\n");
      printf("    - transpose       : %.6f s\n", transpose_time);
      printf("    - ConvTranspose2d : %.6f s\n", conv_transpose_time);
      printf("    - upfir2d         : %.6f s\n", upfir2d_time);
  } else {
      printf("  - Standard Conv Path (%s): %.6f s\n", conv_path_taken, conv2d_time);
  }
  printf("-------------------------------------\n");
  printf("Sum of Timed Parts  : %.6f s\n", sum_of_parts);
  printf("Unaccounted Time    : %.6f s\n", total_func_time - sum_of_parts);
  printf("--- End of Report ---\n\n");
  */
}

/**
 * Adds bias to the input tensor in-place using AVX2 and optimized parallelism.
 * @param [in & out] inout: [N, C, H, W] tensor
 * @param [in] bias: [C] tensor
 */
void addBias(Tensor *inout, Tensor *bias) {
  // Get tensor dimensions
  size_t N = inout->shape[0];
  size_t C = inout->shape[1];
  size_t H = inout->shape[2];
  size_t W = inout->shape[3];
  size_t HW = H * W;

  // Flatten N and C dimensions for better work distribution among threads
  size_t NC = N * C;
  
  // Define a prefetch distance (in floats) to fetch data into the cache
  // ahead of time. 64 bytes = 16 floats. Let's prefetch 4 cache lines ahead.
  const int PREFETCH_DISTANCE = 64;

  // 3. Optimize thread distribution: Parallelize over the flattened NC dimension.
  // This distributes the work more effectively than `collapse(2)` when N*C is small.
  #pragma omp parallel for
  for (size_t nc = 0; nc < NC; nc++) {
    // 4. Minimize repeated index computations: Calculate the base pointer once per feature map.
    float *p_inout = inout->buf + nc * HW;
    // Determine the current channel and get its bias value
    size_t c = nc % C;
    float bias_cur = bias->buf[c];

    // 2. Use SIMD vectorization (AVX2): Broadcast bias to a 256-bit vector (8 floats)
    __m256 v_bias = _mm256_set1_ps(bias_cur);

    // Process the bulk of the data in chunks of 8 floats
    size_t i = 0;
    size_t vec_end = (HW / 8) * 8; // Loop only over multiples of 8

    for (; i < vec_end; i += 8) {
      // 5. Prefetch for very large HW: Hint to the CPU to fetch future data
      _mm_prefetch((const char *)(p_inout + i + PREFETCH_DISTANCE), _MM_HINT_T0);

      // Load 8 floats from the input tensor (use unaligned load for safety)
      __m256 v_inout = _mm256_loadu_ps(p_inout + i);
      // Add the bias vector
      v_inout = _mm256_add_ps(v_inout, v_bias);
      // Store the 8-float result back to memory
      _mm256_storeu_ps(p_inout + i, v_inout);
    }

    // 1. Reduce loop overhead: The AVX loop handles this by unrolling.
    // Process the remaining elements (HW % 8) with a standard scalar loop.
    for (; i < HW; i++) {
      p_inout[i] += bias_cur;
    }
  }
}

/**
 * @brief Combines addNoise, addBias, and LeakyReLU operations in-place using
 * a highly optimized, SIMD-accelerated implementation.
 */
void addNoiseBiasLeakyReLU(Tensor *output, Tensor *noise, Tensor *conv_bias) {
  // --- Timing variables ---
  // --- 1. Hoist Pointers & Dimensions ---
  float* const out_buf = output->buf;
  const float* const noise_buf = noise->buf;
  const float* const bias_buf = conv_bias->buf;
  const size_t N = output->shape[0], C = output->shape[1], H = output->shape[2], W = output->shape[3];
  const size_t spatial_dim = H * W;

  // --- 2. Hoist & Pre-compute Constants ---
  const float negative_slope = 0.2f;
  const float scale = sqrtf(2.0f);
  const float neg_slope_scaled = negative_slope * scale;

  // --- 3. Prepare SIMD Constants (AVX) ---
  const size_t vec_width = 8;
  const __m256 v_scale = _mm256_set1_ps(scale);
  const __m256 v_neg_slope_scaled = _mm256_set1_ps(neg_slope_scaled);
  const __m256 v_zero = _mm256_setzero_ps();

  // --- 4. Parallelize with OpenMP & Tune Schedule ---
  #pragma omp parallel for collapse(2) schedule(static)
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      const float bias_val = bias_buf[c];
      const __m256 v_bias = _mm256_set1_ps(bias_val);
      float* out_ptr = out_buf + (n * C + c) * spatial_dim;
      const float* noise_ptr = noise_buf;

      // --- 5. Main SIMD Loop (Flattened Spatial Dimensions) ---
      const size_t num_vectors = spatial_dim / vec_width;
      for (size_t i = 0; i < num_vectors; ++i) {
        __m256 v_out = _mm256_loadu_ps(out_ptr);
        const __m256 v_noise = _mm256_loadu_ps(noise_ptr);
        v_out = _mm256_add_ps(v_out, v_noise);
        v_out = _mm256_add_ps(v_out, v_bias);
        const __m256 v_pos = _mm256_mul_ps(v_out, v_scale);
        const __m256 v_neg = _mm256_mul_ps(v_out, v_neg_slope_scaled);
        const __m256 v_mask = _mm256_cmp_ps(v_out, v_zero, _CMP_LT_OS);
        v_out = _mm256_blendv_ps(v_pos, v_neg, v_mask);
        _mm256_storeu_ps(out_ptr, v_out);
        out_ptr += vec_width;
        noise_ptr += vec_width;
      }

      // --- 6. Scalar Remainder Loop ---
      const size_t remainder_count = spatial_dim % vec_width;
      for (size_t i = 0; i < remainder_count; ++i) {
        float val = out_ptr[i];
        val += noise_ptr[i];
        val += bias_val;
        if (val < 0.0f) {
          val *= negative_slope;
        }
        val *= scale;
        out_ptr[i] = val;
      }
    }
  }
}

/*
 * Element-wise addition of two tensors (Optimized with AVX2)
 * @param [in & out] inout: [N, C, H, W] - Assumes buf is 32-byte aligned
 * @param [in] addend: [N, C, H, W] - Assumes buf is 32-byte aligned
 * Adds the elements of addend to inout in-place.
 */
void elemAdd(Tensor *inout, Tensor *addend) {
  const size_t N = inout->num_elem();
  // Use local pointers to avoid repeated struct member lookups inside the loop.
  // The __restrict__ keyword informs the compiler that the pointers do not overlap.
  float *__restrict__ p_inout = inout->buf;
  const float *__restrict__ p_addend = addend->buf;

  // Use OpenMP for multi-threading with a static schedule for better cache performance.
  // Each thread gets a contiguous chunk of iterations.
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N / 32; i++) {
    // Calculate the base index for the start of this block
    size_t base_idx = i * 32;

    // Prefetch data for the *next* iteration to hide memory latency.
    _mm_prefetch((const char *)(p_addend + base_idx + 32), _MM_HINT_T0);
    _mm_prefetch((const char *)(p_inout + base_idx + 32), _MM_HINT_T0);

    // Manually unroll the loop 4x using AVX2 intrinsics.
    // This processes 32 floats (4 vectors * 8 floats/vector) per iteration.
    
    // Vector 1 (elements 0-7)
    __m256 vec_inout1 = _mm256_load_ps(p_inout + base_idx + 0);
    __m256 vec_addend1 = _mm256_load_ps(p_addend + base_idx + 0);
    vec_inout1 = _mm256_add_ps(vec_inout1, vec_addend1);
    _mm256_store_ps(p_inout + base_idx + 0, vec_inout1);

    // Vector 2 (elements 8-15)
    __m256 vec_inout2 = _mm256_load_ps(p_inout + base_idx + 8);
    __m256 vec_addend2 = _mm256_load_ps(p_addend + base_idx + 8);
    vec_inout2 = _mm256_add_ps(vec_inout2, vec_addend2);
    _mm256_store_ps(p_inout + base_idx + 8, vec_inout2);
    
    // Vector 3 (elements 16-23)
    __m256 vec_inout3 = _mm256_load_ps(p_inout + base_idx + 16);
    __m256 vec_addend3 = _mm256_load_ps(p_addend + base_idx + 16);
    vec_inout3 = _mm256_add_ps(vec_inout3, vec_addend3);
    _mm256_store_ps(p_inout + base_idx + 16, vec_inout3);

    // Vector 4 (elements 24-31)
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
  addNoiseBiasLeakyReLU(output, noise, conv_bias);
}

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *col_buffer, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a, cudaStream_t *streams) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, nullptr, col_buffer, nullptr, nullptr, nullptr, nullptr, false, false, 0, 2, streams);
  addBias(output, conv_bias);

  if (skip != nullptr) {
    upfir2d(skip, kernel, skip_a, skip_upsample_a, skip_conv_a, 2, 2, 1);
    elemAdd(output, skip_a);
  }
}