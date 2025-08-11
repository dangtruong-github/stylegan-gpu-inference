#include <cuda_runtime.h>
#include <unistd.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "model.h"

static size_t num_images = 1;
static bool run_validation = false;
static bool run_warmup = false;

static char input_fname[100] = "/getp/styles.bin";
static char param_fname[100] = "/getp/param.bin";
static char output_fname[100] = "./data/outputs.bin";
static char answer_fname[100] = "/getp/answers.bin";

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_help() {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank != 0) return;
  fprintf(stdout,
          " Usage: ./main [-i 'pth'] [-p 'pth'] [-o 'pth'] [-a 'pth']"
          " [-n 'num_images'] [-v] [-w] [-h]\n");
  fprintf(stdout, " Options:\n");
  fprintf(stdout, 
          "  -i: Input binary path (default: /opt/getp25/inputs.bin)\n");
  fprintf(stdout,
          "  -p: Model parameter path (default: /opt/getp25/params.bin)\n");
  fprintf(stdout, 
          "  -o: Output binary path (default: ./data/outputs.bin)\n");
  fprintf(stdout, 
          "  -a: Answer binary path (default: ./data/answers.bin)\n");
  fprintf(stdout, "  -n: Number of inputs (default: 1)\n");
  fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
  fprintf(stdout, "  -w: Enable warm-up (default: OFF)\n");
  fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
}

void parse_args(int argc, char **argv) {
  int args;
  while ((args = getopt(argc, argv, "i:o:a:p:n:vwh")) != -1) {
    switch (args) {
      case 'i': strcpy(input_fname, optarg); break;
      case 'o': strcpy(output_fname, optarg); break;
      case 'a': strcpy(answer_fname, optarg); break;
      case 'p': strcpy(param_fname, optarg); break;
      case 'n': num_images = atoi(optarg); break;
      case 'v': run_validation = true; break;
      case 'w': run_warmup = true; break;
      case 'h':
        print_help();
        exit(0);
        break;
      default:
        print_help();
        exit(0);
        break;
    }
  }
 
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank != 0) return;
  fprintf(stdout, "\n=============================================\n");
  fprintf(stdout, " Model\n");
  fprintf(stdout, "---------------------------------------------\n");
  fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
  fprintf(stdout, " Warm-up: %s\n", run_warmup ? "ON" : "OFF");
  fprintf(stdout, " Number of images: %ld\n", num_images);
  fprintf(stdout, " Input path: %s\n", input_fname);
  fprintf(stdout, " Model parameter path: %s\n", param_fname);
  fprintf(stdout, " Answer binary path: %s\n", answer_fname);
  fprintf(stdout, " Output binary path: %s\n", output_fname);
  fprintf(stdout, "=============================================\n\n");
}

int validate(float *output, float *answer, int size_) {
#ifdef FP16
  float threshold = 5e-2;
#else
  float threshold = 1e-3;
#endif
  float max_min_err = 0.0f;

  for (int i = 0; i < size_; i++) {
    float abs_err = fabs(output[i] - answer[i]);
    float rel_err = (fabs(answer[i]) > 1e-8) ? abs_err / fabs(answer[i]) : abs_err;
    float min_err = fmin(abs_err, rel_err);
 
    max_min_err = fmax(max_min_err, min_err);

    if (max_min_err > threshold || std::isnan(output[i])) {
      return i;
    }
  }

  return -1; 
}

void *read_binary(const char *fname, size_t *size) {
  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    fprintf(stdout, "[ERROR] Cannot open file \'%s\'\n", fname);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_t size_ = ftell(f);
  rewind(f);

  void *buf;
  CHECK_CUDA(cudaMallocHost(&buf, size_));

  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stdout, "[ERROR] Cannot read file \'%s\'\n", fname);
    exit(-1);
  }
  fclose(f);

  if (size != NULL) *size = (size_t)(size_ / 4);  // 4B per float/int

  return buf;
}

void write_binary(float *output, const char *filename, int size_) {
  FILE *f = (FILE *) fopen(filename, "wb");
  fwrite(output, sizeof(float), size_, f);
  fclose(f);
}

int main(int argc, char **argv) {
  int mpi_rank, mpi_size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  parse_args(argc, argv);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  ////////////////////////////////////////////////////////////////////

  float *inputs = nullptr;
  float *outputs = nullptr;
  float *params = nullptr;
  size_t param_size = 0;

  if (mpi_rank == 0) {
    /* Load inputs (size: [num_images * 512]) from file  */
    fprintf(stdout, "Initializing inputs and parameters...");
    size_t input_size;
    inputs = (float *) read_binary(input_fname, &input_size);

    /* Allocate outputs (size: [num_images * 512 * 512 * 3]) */
    CHECK_CUDA(cudaMallocHost(&outputs, num_images * 512 * 512 * 3 * sizeof(float)));
  }

  /* Initialize parameters and activations */
  params = (float *) read_binary(param_fname, &param_size);
  alloc_and_set_parameters(params, param_size);
  alloc_activations();
  if (mpi_rank == 0) fprintf(stdout, "Done!\n");

  /* Warm-up */
  if (run_warmup){
    if (mpi_rank == 0) fprintf(stdout, "Warm-up...");
    for (size_t i = 0; i < 3; i++) {
      generate(inputs, outputs, 1);
    }
    if (mpi_rank == 0) fprintf(stdout, "Done!\n");
  }
  
  ////////////////////////////////////////////////////////////////////
  // MODEL COMPUTATION                                              //
  ////////////////////////////////////////////////////////////////////

  double st = 0.0, et = 0.0;

  if (mpi_rank == 0) {
    fprintf(stdout, "Generating images..."); fflush(stdout);
  }

   
  for (size_t i = 0; i < 4; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  
  CHECK_CUDA(cudaSetDevice(0));
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0) st = get_time();

  /* Call the main computation (optimization target) of the program. */
	generate(inputs, outputs, num_images);

  
  for (size_t i = 0; i < 4; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  
  CHECK_CUDA(cudaSetDevice(0));
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    et = get_time();
    /* Print the result */
    fprintf(stdout, "Done!\n");
    fprintf(stdout, "Elapsed time: %lf (sec)\n", et - st);
    fprintf(stdout, "Throughput: %lf (images/sec)\n", 
            num_images / (et - st));
  }

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////

  /* Finalize parameters and activations */
  if (mpi_rank == 0) fprintf(stdout, "Finalizing...");
  free_parameters();
  free_activations();
  if (mpi_rank == 0) fprintf(stdout, "Done!\n");

  if (mpi_rank == 0) {
    /* Save outputs */
    fprintf(stdout, "Saving outputs to %s...", output_fname);
    write_binary(outputs, output_fname, num_images * 512 * 512 * 3);
    fprintf(stdout, "Done!\n");

    /* Validation */
    if (run_validation) {
      fprintf(stdout, "Validating...");
      float *answers = (float *) read_binary(answer_fname, nullptr);
      int ret = validate(outputs, answers, num_images * 512 * 512 * 3);
      if (ret == -1) {
        fprintf(stdout, "PASSED!\n");
      } else {
        fprintf(stdout, "FAILED!\nFirst mismatch at image[%d], index[%u] "
                        "(output[%d]=%f <-> answer[%d]=%f)\n", 
                        ret / 2, ret % 2, ret, outputs[ret], 
                        ret, answers[ret]);
      }
    }
  } // end if (mpi_rank == 0)
  MPI_Finalize();

  return 0;
}
