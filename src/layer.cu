#include "layer.h"
#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

#define ITILE 64
#define JTILE 1024
#define KTILE 64
#define VEC_LEN 8
#define TILE_W_IM2COL 16
#define NUM_THREADS_MAT_MUL 64

// --- CUDA Kernel for Matrix Multiplication ---
// This kernel remains unchanged. It operates on the data pointers it's given.
// The logic of which GPU it runs on is handled by the host.
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float C_value = 0.0f;
    for (int k = 0; k < K; ++k) {
      C_value += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = C_value;
  }
}

// --- Host Function to Launch the Kernel ---
// This function also remains unchanged. It simply configures and launches the kernel.
void launch_matmul_kernel(const float *d_A, const float *d_B, float *d_C, int M, int N, int K) {
  const int TILE_SIZE = 16;
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
  CHECK_CUDA(cudaGetLastError());
}

// --- Batched Matrix Multiplication Wrapper (Multi-GPU) ---
// This function now orchestrates the operation across multiple GPUs.
void bmm_wrapper(Tensor *A, Tensor *B, Tensor *C, cudaStream_t *streams) {
  // 1. Move input data to the respective GPUs.
  // to_device() handles splitting the data across all NUM_GPUS.
  A->to_device(streams);
  B->to_device(streams);

  // 2. Get dimensions from tensor shapes
  const size_t batch_per_gpu = A->shape[0] / NUM_GPUS;
  const size_t M = A->shape[1];
  const size_t K = A->shape[2];
  const size_t N = B->shape[2];

  // Calculate the number of elements per single matrix in the batch
  const size_t num_elem_A = M * K;
  const size_t num_elem_B = K * N;
  const size_t num_elem_C = M * N;

  // 3. Loop over each GPU and launch kernels for its portion of the batch
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
    // Set the active GPU for the following CUDA calls
    CHECK_CUDA(cudaSetDevice(gpu_id));

    // Get base pointers to device memory for the current GPU
    float *base_A_d = A->d_buf[gpu_id];
    float *base_B_d = B->d_buf[gpu_id];
    float *base_C_d = C->d_buf[gpu_id];

    // Loop over the sub-batch assigned to this GPU
    for (size_t b = 0; b < batch_per_gpu; ++b) {
      // Calculate pointers for the current matrix in the sub-batch
      float *cur_A_d = base_A_d + b * num_elem_A;
      float *cur_B_d = base_B_d + b * num_elem_B;
      float *cur_C_d = base_C_d + b * num_elem_C;

      // Launch the kernel for the current matrices on the current GPU
      launch_matmul_kernel(cur_A_d, cur_B_d, cur_C_d, M, N, K);
    }
  }
  
  // Synchronize all GPUs to ensure all computations are finished before copying back data
  for (int gpu_id = 0; gpu_id < NUM_GPUS; ++gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // 4. Copy the final result from all GPUs back to the host CPU
  C->from_device(streams);
}

/**
 * @brief Transforms image patches into a column matrix (transposed layout).
 * This is a highly optimized version for large H and W.
 * @param input The input tensor of shape (N, C, H, W).
 * @param col_buffer The output buffer with shape (N, C*R*S, OH*OW).
 * @param R Kernel height.
 * @param S Kernel width.
 * @param stride The convolution stride.
 * @param pad The convolution padding.
 * @param dilation The convolution dilation.
 */
void im2col(Tensor* input, Tensor* col_buffer) {
  // Input tensor dimensions
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];

  // Calculate effective kernel size with dilation
  size_t R = 3, S = 3;
  
  // Get dimensions from the col_buffer's shape
  size_t CRS_col = col_buffer->shape[1]; // C * R * S
  size_t HW_col = col_buffer->shape[2];  // OH * OW

  // 2. --- Optimization: Parallelization (over batch 'N') ---
  // The loop over batch items is embarrassingly parallel.
  for (size_t n = 0; n < N; ++n) {
    // Pointer to the start of the nth item in the input buffer
    const float* input_n = input->buf + n * C * H * W;
    // Pointer to the start of the nth item in the output column buffer
    float* col_buffer_n = col_buffer->buf + n * CRS_col * HW_col;

    // 3. --- Optimization: Loop Reordering & Tiling ---
    // We iterate through channels first, then output locations.
    // Tiling is applied to the output width (w_col) to improve L1/L2 cache re-use.
    #pragma omp parallel for collapse(2) num_threads(NUM_THREADS_MAT_MUL)
    for (size_t c = 0; c < C; ++c) {
      for (size_t h_col = 0; h_col < H; ++h_col) {
        for (size_t w_block_start = 0; w_block_start < W; w_block_start += TILE_W_IM2COL) {
          size_t w_block_end = min(w_block_start + TILE_W_IM2COL, W);

          for (size_t w_col = w_block_start; w_col < w_block_end; ++w_col) {

            // This is now the "hot loop". It processes one entire input patch.
            // By iterating `r` and `s` here, we ensure reads from `input`
            // are localized to a small window, maximizing cache hits.
            for (size_t r = 0; r < R; ++r) {
              int input_h = h_col + r - 1;

              for (size_t s = 0; s < S; ++s) {
                int input_w = w_col + s - 1;
                
                // The bounds check is still needed, but the 'else' is gone.
                if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
                  // Calculate source and destination indices
                  size_t src_idx = c * H * W + input_h * W + input_w;
                  size_t dest_idx = (c * R * S + r * S + s) * HW_col + (h_col * W + w_col);
                  
                  col_buffer_n[dest_idx] = input_n[src_idx];
                }
              }
            }
          }
        }
      }
    }
  }
}

// --- Recommendation #3 & #9: Configure wider H/W tiles for cache blocking ---
#ifndef TILE_SIZE_H
#define TILE_SIZE_H 64
#endif
#ifndef TILE_SIZE_W
#define TILE_SIZE_W 256
#endif

/**
 * @brief Transforms image patches into a column matrix (specialized for 3x3, s=1, p=1).
 *
 * This is a highly optimized AVX2 version specifically for a 3x3 kernel with
 * stride=1, pad=1, and dilation=1. These fixed parameters simplify output dimension
 * calculations (OH=H, OW=W) and remove the need for general-purpose logic,
 * resulting in a faster, more streamlined implementation.
 *
 * @param input The input tensor of shape (N, C, H, W).
 * @param col_buffer The output buffer, pre-allocated with shape (N, C*9, H*W).
 */
void im2col_3x3_s1_p1_avx2(const Tensor* input, Tensor* col_buffer) {
    // Input dimensions
    const size_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    
    // Hardcoded parameters for this specialized function
    constexpr size_t R = 3, S = 3;
    constexpr size_t KERNEL_SIZE = R * S; // Becomes 9

    // With stride=1, pad=1, dilation=1 for a 3x3 kernel, OH=H and OW=W.
    const size_t OH = H;
    const size_t OW = W;
    const size_t COL_WIDTH = OH * OW;
    const size_t COL_HEIGHT = C * KERNEL_SIZE;

    // --- Recommendation #5: Parallelize over larger grains (batch N x channels C) ---
    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const float* input_c = input->buf + (n * C + c) * H * W;
            float* col_buffer_c = col_buffer->buf + (n * COL_HEIGHT + c * KERNEL_SIZE) * COL_WIDTH;

            // --- Recommendation #8: Separate pass for zero padding ---
            // memset(col_buffer_c, 0, KERNEL_SIZE * COL_WIDTH * sizeof(float));
            
            // --- Recommendation #1: Define interior region to avoid bounds checks ---
            // For a 3x3 kernel with pad=1, the interior (non-padded) region of the
            // output is from (1, 1) to (H-2, W-2).
            const int h_col_interior_start = 1;
            const int h_col_interior_end = H - 1;
            const int w_col_interior_start = 1;
            const int w_col_interior_end = W - 1;

            // SLOW PATH: Generic lambda to process border patches where checks are required.
            auto process_border_patch = [&](int h_col, int w_col) {
                const int h_in_start = h_col - 1; // pad=1
                const int w_in_start = w_col - 1; // pad=1
                size_t col_idx = h_col * OW + w_col;
                for (size_t r = 0; r < R; ++r) {
                    for (size_t s = 0; s < S; ++s) {
                        int h_in = h_in_start + r;
                        int w_in = w_in_start + s;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            col_buffer_c[(r * S + s) * COL_WIDTH + col_idx] = input_c[h_in * W + w_in];
                        }
                    }
                }
            };
            
            // Process all border regions using the slow path
            for (int h = 0; h < OH; ++h) {
                if (h < h_col_interior_start || h >= h_col_interior_end) {
                    for (int w = 0; w < OW; ++w) process_border_patch(h, w);
                } else {
                    for (int w = 0; w < w_col_interior_start; ++w) process_border_patch(h, w);
                    for (int w = w_col_interior_end; w < OW; ++w) process_border_patch(h, w);
                }
            }
            
            // --- FAST PATH (HOT LOOP): Process the interior region ---
            for (int h_tile = h_col_interior_start; h_tile < h_col_interior_end; h_tile += TILE_SIZE_H) {
                for (int w_tile = w_col_interior_start; w_tile < w_col_interior_end; w_tile += TILE_SIZE_W) {
                    const int h_tile_end = min(h_tile + TILE_SIZE_H, h_col_interior_end);
                    const int w_tile_end = min(w_tile + TILE_SIZE_W, w_col_interior_end);
                    
                    for (size_t r = 0; r < R; ++r) {
                        for (size_t s = 0; s < S; ++s) {
                            float* col_ptr_base = col_buffer_c + (r * S + s) * COL_WIDTH;

                            for (int h_col = h_tile; h_col < h_tile_end; ++h_col) {
                                // --- Rec #7: Simplified index calculations ---
                                const int h_in = h_col - 1 + r;
                                const float* input_ptr = input_c + h_in * W + (w_tile - 1 + s);
                                float* col_ptr = col_ptr_base + h_col * OW + w_tile;
                                
                                int w_col = w_tile;

                                // --- Rec #4: Vectorize along w_col for 8 elements at a time ---
                                // Since stride=1 and dilation=1, memory access is contiguous.
                                int w_col_end_avx = w_tile + ((w_tile_end - w_tile) / 8) * 8;
                                for (; w_col < w_col_end_avx; w_col += 8) {
                                    _mm256_storeu_ps(col_ptr, _mm256_loadu_ps(input_ptr));
                                    input_ptr += 8;
                                    col_ptr += 8;
                                }
                                
                                // Scalar cleanup for remaining elements
                                for (; w_col < w_tile_end; ++w_col) {
                                    *col_ptr++ = *input_ptr++;
                                }
                            }
                        }
                    }
                }
            } // End of fast path
        } // End of parallel channel loop
    } // End of batch loop
}

/**
 * @brief Transforms a column buffer back into an image tensor (transposed convolution).
 * * @param col_buffer Input column buffer with shape (N, K*R*S, H*W).
 * @param output The output tensor with shape (N, K, OH, OW) to be filled.
 * @param H Input height.
 * @param W Input width.
 * @param R Kernel height.
 * @param S Kernel width.
 * @param stride Convolution stride.
 * @param pad Convolution padding.
 */
void col2im(Tensor *col_buffer, Tensor *output, int H, int W) {
  // Extract dimensions for clarity
  const int N = output->shape[0];
  const int K = output->shape[1];
  const int OH = output->shape[2];
  const int OW = output->shape[3];

  const int KRS = K * 9;
  const int HW = H * W;

  // Initialize the output tensor with zeros. This is crucial.
  memset(output->buf, 0, N * K * OH * OW * sizeof(float));

  // Iterate through each element in the column buffer
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; ++n) {
    for (int krs = 0; krs < KRS; ++krs) {
      for (int hw = 0; hw < HW; ++hw) {
        // Deconstruct the indices
        const int k = krs / 9;
        const int r = (krs % 9) / 3;
        const int s = krs % 3;

        const int h = hw / W;
        const int w = hw % W;

        // Calculate the target coordinates in the output image
        // This is the core formula for the "scattering" operation
        const int oh = h * 2 + r;
        const int ow = w * 2 + s;
        
        // Check if the calculated coordinates are within the output tensor's bounds
        if (oh < OH && ow < OW) {
          // Calculate flat indices
          const int col_idx = n * (KRS * HW) + krs * HW + hw;
          const int output_idx = n * (K * OH * OW) + k * (OH * OW) + oh * OW + ow;
          
          // Add the value from the column buffer to the output image.
          // We use addition because multiple values can map to the same output pixel.
          output->buf[output_idx] += col_buffer->buf[col_idx];
        }
      }
    }
  }
}

/**
 * @brief Transposes a 5D tensor from shape (N, K, C, R, S) to (N, K, R, S, C).
 * * @param weight The input tensor with shape (N, K, C, R, S).
 * @param weight_transpose The output tensor to store the result, with shape (N, K, R, S, C).
 */
void transpose(Tensor *weight, Tensor *weight_transpose) {
  // Extract dimensions for clarity
  const int N = weight->shape[0];
  const int K = weight->shape[1];
  const int C = weight->shape[2];
  const int R = weight->shape[3];
  const int S = weight->shape[4];

  // Iterate through each element of the input tensor
  #pragma omp parallel for collapse(5) num_threads(NUM_THREADS_MAT_MUL)
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
          for (int s = 0; s < S; ++s) {
            // Calculate the source index in the original tensor (row-major)
            int src_idx = n * (K * C * R * S) + k * (C * R * S) + 
                          c * (R * S) + r * S + s;

            // Calculate the destination index in the transposed tensor (row-major)
            int dst_idx = n * (K * R * S * C) + k * (R * S * C) + 
                          r * (S * C) + s * C + c;
            
            // Copy the element
            weight_transpose->buf[dst_idx] = weight->buf[src_idx];
          }
        }
      }
    }
  }
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
void Conv2d(Tensor *input, Tensor *weight, Tensor *output) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  // --- End of Timing variables ---

  size_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  size_t K = weight->shape[1];
  size_t OH = output->shape[2], OW = output->shape[3];

  #pragma omp parallel for collapse(2)
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      // Base pointers for the current (n, k) pair
      const float* p_input_n = input->buf + n * C * H * W;
      // Since R=S=1, the spatial size of the weight tensor is 1.
      const float* p_weight_nk = weight->buf + n * K * C + k * C;
      float* p_output_nk = output->buf + n * K * OH * OW + k * OH * OW;

      // Guaranteed that OW is either 4 or a multiple of 8.
      if (OW == 4) {
        // ====================================================================
        // PATH 1: Specialized AVX for OW=4, R=1, S=1 by unrolling OH by 2.
        // Assumes OH is always a multiple of 2.
        // ====================================================================
        for (size_t oh = 0; oh < OH; oh += 2) {
          // Accumulator for 2x4 = 8 pixels, for the output block at (oh, 0) and (oh+1, 0)
          __m256 o_vec = _mm256_setzero_ps();

          for (size_t c = 0; c < C; ++c) {
            // Get the single weight for this input channel c.
            const float* p_weight_c = p_weight_nk + c;
            __m256 f_vec = _mm256_set1_ps(*p_weight_c);

            const float* p_input_c = p_input_n + c * H * W;

            // --- Load data for the first row (oh) ---
            __m128 i_vec_row0 = _mm_setzero_ps();
            if (oh < H) {
              const float* p_input_row0 = p_input_c + oh * W;
              // Since R=S=1, we load from the start of the 4-element block.
              // A fast path for W>=4 is possible, but this handles all W.
              float temp_i[4] = {0.0f};
              for (size_t i = 0; i < W && i < 4; ++i) {
                  temp_i[i] = p_input_row0[i];
              }
              i_vec_row0 = _mm_loadu_ps(temp_i);
            }

            // --- Load data for the second row (oh + 1) ---
            __m128 i_vec_row1 = _mm_setzero_ps();
            size_t h1 = oh + 1;
            if (h1 < H) {
              const float* p_input_row1 = p_input_c + h1 * W;
              float temp_i[4] = {0.0f};
              for (size_t i = 0; i < W && i < 4; ++i) {
                  temp_i[i] = p_input_row1[i];
              }
              i_vec_row1 = _mm_loadu_ps(temp_i);
            }
            
            // Combine the two 128-bit vectors into one 256-bit vector
            __m256 i_vec = _mm256_castps128_ps256(i_vec_row0);
            i_vec = _mm256_insertf128_ps(i_vec, i_vec_row1, 1);

            // Fused Multiply-Add: o_vec += i_vec * f_vec
            o_vec = _mm256_fmadd_ps(i_vec, f_vec, o_vec);
          }

          // Split the 256-bit result and store it into the two output rows
          __m128 o_vec_row0 = _mm256_castps256_ps128(o_vec);
          __m128 o_vec_row1 = _mm256_extractf128_ps(o_vec, 1);
          _mm_storeu_ps(p_output_nk + oh * OW, o_vec_row0);
          _mm_storeu_ps(p_output_nk + (oh + 1) * OW, o_vec_row1);
        }
      } else {
        // ====================================================================
        // PATH 2: Standard AVX Path for R=1, S=1
        // Assumes OW is a multiple of 8.
        // ====================================================================
        for (size_t oh = 0; oh < OH; ++oh) {
          for (size_t ow = 0; ow < OW; ow += 8) {
            __m256 o_vec = _mm256_setzero_ps();

            for (size_t c = 0; c < C; ++c) {
              const float* p_weight_c = p_weight_nk + c;
              __m256 f_vec = _mm256_set1_ps(*p_weight_c);
              
              __m256 i_vec = _mm256_setzero_ps();
              size_t h = oh;
              if (h < H) {
                  const float* p_input_row = p_input_n + (c * H * W) + (h * W);
                  // Since R=S=1, we load directly from input[h][ow]
                  if (ow + 7 < W) {
                      i_vec = _mm256_load_ps(p_input_row + ow);
                  } else {
                      float temp_i[8] = {0.0f};
                      for (size_t i = 0; i < 8 && (ow + i) < W; ++i) {
                          temp_i[i] = p_input_row[ow + i];
                      }
                      i_vec = _mm256_loadu_ps(temp_i);
                  }
              }
              o_vec = _mm256_fmadd_ps(i_vec, f_vec, o_vec);
            }
            _mm256_store_ps(p_output_nk + oh * OW + ow, o_vec);
          }
        }
      }
    }
  }

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
 * Convolution (with per-sample weights) - Optimized for stride=1, pad=1, dilation=1 and a 3x3 kernel
 *
 * This version fully unrolls the loops over the 3x3 kernel dimensions (R and S) to
 * eliminate loop overhead and maximize instruction-level parallelism.
 *
 * CONSTRAINT: K is a multiple of 64. We will unroll by 4.
 * pad = 1, dilation = 1, stride = 1
 */
void Conv2d_Optimized(Tensor *input, Tensor *weight, Tensor *output) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  // --- End of Timing variables ---

  size_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
  size_t K = weight->shape[1];
  const int R = 3, S = 3;
  const int KERNEL_SIZE = R * S; // Will be 9

  size_t OH = output->shape[2], OW = output->shape[3];

  const int VEC_LEN_128 = 4;
  const int K_UNROLL = 4;

  const size_t input_n_stride = C * H * W;
  const size_t weight_n_stride = K * C * KERNEL_SIZE;
  const size_t output_n_stride = K * OH * OW;
  const size_t weight_k_stride = C * KERNEL_SIZE;
  const size_t output_k_stride = OH * OW;

  memset(output->buf, 0, N * K * OH * OW * sizeof(float));

  #pragma omp parallel for collapse(2)
  for (size_t n = 0; n < N; ++n) {
    for (size_t k_base = 0; k_base < K; k_base += K_UNROLL) {
      const float* p_input_n = input->buf + n * input_n_stride;

      float* p_output_nk[K_UNROLL];
      for (int i = 0; i < K_UNROLL; ++i) {
        p_output_nk[i] = output->buf + n * output_n_stride + (k_base + i) * output_k_stride;
      }

      const float* p_weight_nk[K_UNROLL];
      for (int i = 0; i < K_UNROLL; ++i) {
        p_weight_nk[i] = weight->buf + n * weight_n_stride + (k_base + i) * weight_k_stride;
      }

      // Loop over input channels is now the outer loop
      for (size_t c = 0; c < C; ++c) {
        const float* p_input_c = p_input_n + c * H * W;
        const float* p_weight_c[K_UNROLL];
        for (int i = 0; i < K_UNROLL; ++i) {
          p_weight_c[i] = p_weight_nk[i] + c * KERNEL_SIZE;
        }

        if (OW >= VEC_LEN) {
          // FAST PATH: OW is a multiple of 8. No boundary checks needed for width.
          for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t ow = 0; ow < OW; ow += VEC_LEN) {
              // Load is always safe and aligned
              __m256 o_vec0 = _mm256_load_ps(p_output_nk[0] + oh * OW + ow);
              __m256 o_vec1 = _mm256_load_ps(p_output_nk[1] + oh * OW + ow);
              __m256 o_vec2 = _mm256_load_ps(p_output_nk[2] + oh * OW + ow);
              __m256 o_vec3 = _mm256_load_ps(p_output_nk[3] + oh * OW + ow);

              // V3 Optimization: Manually unroll the 3x3 kernel operations
              #define LOAD_AND_FMA_256(r_offset, s_offset, weight_idx) \
              { \
                const int h_in = (int)oh - 1 + (r_offset); \
                const int w_in_base = (int)ow - 1 + (s_offset); \
                __m256 i_vec; \
                if (h_in >= 0 && h_in < H && w_in_base >= 0 && w_in_base + VEC_LEN <= W) { \
                  i_vec = _mm256_loadu_ps(p_input_c + h_in * W + w_in_base); \
                } else { \
                  float temp_i[VEC_LEN]; \
                  for (int i = 0; i < VEC_LEN; ++i) { \
                    const int w_in = w_in_base + i; \
                    temp_i[i] = (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) \
                                ? p_input_c[h_in * W + w_in] : 0.0f; \
                  } \
                  i_vec = _mm256_loadu_ps(temp_i); \
                } \
                __m256 w_vec0 = _mm256_set1_ps(p_weight_c[0][weight_idx]); \
                o_vec0 = _mm256_fmadd_ps(i_vec, w_vec0, o_vec0); \
                __m256 w_vec1 = _mm256_set1_ps(p_weight_c[1][weight_idx]); \
                o_vec1 = _mm256_fmadd_ps(i_vec, w_vec1, o_vec1); \
                __m256 w_vec2 = _mm256_set1_ps(p_weight_c[2][weight_idx]); \
                o_vec2 = _mm256_fmadd_ps(i_vec, w_vec2, o_vec2); \
                __m256 w_vec3 = _mm256_set1_ps(p_weight_c[3][weight_idx]); \
                o_vec3 = _mm256_fmadd_ps(i_vec, w_vec3, o_vec3); \
              }
              
              // Unroll 3x3 kernel
              LOAD_AND_FMA_256(0, 0, 0); LOAD_AND_FMA_256(0, 1, 1); LOAD_AND_FMA_256(0, 2, 2);
              LOAD_AND_FMA_256(1, 0, 3); LOAD_AND_FMA_256(1, 1, 4); LOAD_AND_FMA_256(1, 2, 5);
              LOAD_AND_FMA_256(2, 0, 6); LOAD_AND_FMA_256(2, 1, 7); LOAD_AND_FMA_256(2, 2, 8);
              
              #undef LOAD_AND_FMA_256

              // Store is always safe and aligned
              _mm256_store_ps(p_output_nk[0] + oh * OW + ow, o_vec0);
              _mm256_store_ps(p_output_nk[1] + oh * OW + ow, o_vec1);
              _mm256_store_ps(p_output_nk[2] + oh * OW + ow, o_vec2);
              _mm256_store_ps(p_output_nk[3] + oh * OW + ow, o_vec3);
            }
          }
        } else {
          // PATH 2: OW is exactly 4. Process with 128-bit vectors only.
          for (size_t oh = 0; oh < OH; ++oh) {
            // Since OW is 4, we only process one 128-bit vector at ow = 0
            const size_t ow = 0;
            
            // Load the 4 output values. Address is 16-byte aligned.
            __m128 o_vec0 = _mm_load_ps(p_output_nk[0] + oh * OW + ow);
            __m128 o_vec1 = _mm_load_ps(p_output_nk[1] + oh * OW + ow);
            __m128 o_vec2 = _mm_load_ps(p_output_nk[2] + oh * OW + ow);
            __m128 o_vec3 = _mm_load_ps(p_output_nk[3] + oh * OW + ow);

            #define LOAD_AND_FMA_128(r_offset, s_offset, weight_idx) \
            { \
              const int h_in = (int)oh - 1 + (r_offset); \
              const int w_in_base = (int)ow - 1 + (s_offset); \
              __m128 i_vec; \
              if (h_in >= 0 && h_in < H && w_in_base >= 0 && w_in_base + VEC_LEN_128 <= W) { \
                i_vec = _mm_loadu_ps(p_input_c + h_in * W + w_in_base); \
              } else { \
                float temp_i[VEC_LEN_128]; \
                for (int i = 0; i < VEC_LEN_128; ++i) { \
                  const int w_in = w_in_base + i; \
                  temp_i[i] = (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) \
                              ? p_input_c[h_in * W + w_in] : 0.0f; \
                } \
                i_vec = _mm_loadu_ps(temp_i); \
              } \
              __m128 w_vec0 = _mm_set1_ps(p_weight_c[0][weight_idx]); \
              o_vec0 = _mm_fmadd_ps(i_vec, w_vec0, o_vec0); \
              __m128 w_vec1 = _mm_set1_ps(p_weight_c[1][weight_idx]); \
              o_vec1 = _mm_fmadd_ps(i_vec, w_vec1, o_vec1); \
              __m128 w_vec2 = _mm_set1_ps(p_weight_c[2][weight_idx]); \
              o_vec2 = _mm_fmadd_ps(i_vec, w_vec2, o_vec2); \
              __m128 w_vec3 = _mm_set1_ps(p_weight_c[3][weight_idx]); \
              o_vec3 = _mm_fmadd_ps(i_vec, w_vec3, o_vec3); \
            }

            // Unroll 3x3 kernel
            LOAD_AND_FMA_128(0, 0, 0); LOAD_AND_FMA_128(0, 1, 1); LOAD_AND_FMA_128(0, 2, 2);
            LOAD_AND_FMA_128(1, 0, 3); LOAD_AND_FMA_128(1, 1, 4); LOAD_AND_FMA_128(1, 2, 5);
            LOAD_AND_FMA_128(2, 0, 6); LOAD_AND_FMA_128(2, 1, 7); LOAD_AND_FMA_128(2, 2, 8);
            
            #undef LOAD_AND_FMA_128

            // Store the 4 updated output values
            _mm_store_ps(p_output_nk[0] + oh * OW + ow, o_vec0);
            _mm_store_ps(p_output_nk[1] + oh * OW + ow, o_vec1);
            _mm_store_ps(p_output_nk[2] + oh * OW + ow, o_vec2);
            _mm_store_ps(p_output_nk[3] + oh * OW + ow, o_vec3);
          }
        }
      }
    }
  }

  // --- Print Timing Results ---
  double func_end_time = get_time_kernel();
  double total_time = func_end_time - func_start_time;

  /*
  printf("\n--- Conv2d_Optimized Timing Report ---\n");
  input->printShape("input");
  weight->printShape("weight");
  output->printShape("output");
  printf("N: %zu, K: %zu, OH: %zu, OW: %zu, H: %zu, W: %zu, C: %zu\n", N, K, OH, OW, H, W, C);
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
  if (OH >= 64) {
    im2col_3x3_s1_p1_avx2(input, col_buffer);
  } else {
    im2col(input, col_buffer);
  }
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
  bmm_wrapper(weight, col_buffer, output, streams);
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
  bmm_wrapper(weight, input, col_buffer, streams);
  end_time = get_time_kernel();
  bmm_time = end_time - start_time;
  input->reshape({N, C, H, W});
  weight->reshape({N, K, (size_t)R, (size_t)S, C});

  // 3. col2im: Transform the columns back into the output image
  start_time = get_time_kernel();
  col2im(col_buffer, output, H, W);
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
    transpose(weight_a, weight_transposed);
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
      Conv2d(input, weight_a, output);
    } else {
      conv_path_taken = "Conv2d_Optimized";
      if (input->shape[2] > 64) {
        Conv2d_im2col(input, weight_a, output, col_buffer, streams);
      } else {
        Conv2d_Optimized(input, weight_a, output);
      }
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
           Tensor *style_a, Tensor *weight_a, Tensor *col_buffer, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a) {
  ModulatedConv2d(input, style, modulate_weight, modulate_bias, conv_weight, kernel, output,
                  style_a, weight_a, nullptr, col_buffer, nullptr, nullptr, nullptr, nullptr, false, false, 0, 2, nullptr);
  addBias(output, conv_bias);

  if (skip != nullptr) {
    upfir2d(skip, kernel, skip_a, skip_upsample_a, skip_conv_a, 2, 2, 1);
    elemAdd(output, skip_a);
  }
}