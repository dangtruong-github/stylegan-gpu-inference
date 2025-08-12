/**
 * @brief CUDA kernel to transform image patches into a column matrix (im2col).
 * Each thread processes one or more elements of the output column buffer.
 * Now accepts dynamic kernel size (R, S) and padding (pad).
 */
__global__ void im2col_kernel(const float* input_data, float* col_data,
                            int batch_size, int C, int H, int W,
                            int R, int S, int pad, int OH, int OW) {
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
        int s_col = crs_col % S;
        int r_col = (crs_col / S) % R;
        int c = crs_col / (R * S);

        // Calculate corresponding coordinates in the source input tensor
        // Note: Assumes stride_h = stride_w = 1
        int input_h = h_col + r_col - pad;
        int input_w = w_col + s_col - pad;

        long long dest_idx = i; // The destination index is the loop index

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
 * @brief Multi-GPU wrapper for the im2col operation.
 * @param input The input tensor of shape (N, C, H, W).
 * @param col_buffer The output column buffer tensor.
 * @param R Kernel height.
 * @param S Kernel width.
 * @param pad Padding size.
 * @param OH Output height.
 * @param OW Output width.
 * @param input_to_device Flag to control transferring the input tensor to GPUs.
 * @param col_buffer_from_device Flag to control transferring the result back to the CPU.
 * @param streams Array of CUDA streams, one for each GPU.
 */
void im2col_wrapper(Tensor *input, Tensor *col_buffer, int R, int S, int pad, int OH, int OW,
                    bool input_to_device, bool col_buffer_from_device, cudaStream_t *streams) {
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
      
      long long total_elements_gpu = (long long)batch_per_gpu * C * R * S * OH * OW;
      int blockSize = 256;
      int gridSize = (total_elements_gpu + blockSize - 1) / blockSize;
      
      im2col_kernel<<<gridSize, blockSize, 0, streams[gpu_id]>>>(
          input->d_buf[gpu_id], 
          col_buffer->d_buf[gpu_id], 
          batch_per_gpu, C, H, W,
          R, S, pad, OH, OW
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
void Conv2d_im2col(Tensor *input, Tensor *weight, Tensor *output, Tensor *col_buffer, int pad, cudaStream_t *streams) {
  // --- Timing variables ---
  double func_start_time = get_time_kernel();
  double start_time, end_time;
  double im2col_time = 0.0;
  double bmm_time = 0.0;
  // --- End of Timing variables ---

  // 2. ---- Get dimensions from input tensors ----
  size_t N = input->shape[0];
  size_t C = input->shape[1];
  size_t H = input->shape[2];
  size_t W = input->shape[3];
  
  size_t K = weight->shape[1];
  size_t R = weight->shape[3];
  size_t S = weight->shape[4];

  // We get OH and OW from the output tensor's shape.
  size_t OH = output->shape[2];
  size_t OW = output->shape[3];

  // 3. ---- Call im2col to populate the provided buffer ----
  // This assumes `col_buffer` has a bmm-compatible shape of (N, OH*OW, C*R*S)
  start_time = get_time_kernel();
  // The wrapper now accepts R, S, pad, OH, and OW.
  // Note: This implementation assumes stride=1 and dilation=1.
  im2col_wrapper(input, col_buffer, R, S, pad, OH, OW, true, false, streams);
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