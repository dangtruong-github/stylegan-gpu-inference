#include <cstdio>
#include <mpi.h>
#include "layer.h"
#include "model.h"

cudaStream_t streams[NUM_GPUS];
/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
// Multi-layer perceptron (MLP) parameters
Parameter *mlp0_w, *mlp0_b;
Parameter *mlp1_w, *mlp1_b;
Parameter *mlp2_w, *mlp2_b;
Parameter *mlp3_w, *mlp3_b;
Parameter *mlp4_w, *mlp4_b;
Parameter *mlp5_w, *mlp5_b;
Parameter *mlp6_w, *mlp6_b;
Parameter *mlp7_w, *mlp7_b;
Parameter *constant_input;  // Constant input for the model
Parameter *kernel;  // Blur kernel

// conv1
Parameter *conv1_modulate_w, *conv1_modulate_b;
Parameter *conv1_w, *conv1_b;

// torgb1
Parameter *to_rgb_modulate_w, *to_rgb_modulate_b;
Parameter *to_rgb_w, *to_rgb_b;

// Parameters for 7 blocks
Parameter *block0_conv_up_modulate_w, *block0_conv_up_modulate_b, *block0_conv_up_w, *block0_conv_up_b;
Parameter *block0_conv_modulate_w, *block0_conv_modulate_b, *block0_conv_w, *block0_conv_b;
Parameter *block0_to_rgb_modulate_w, *block0_to_rgb_modulate_b, *block0_to_rgb_w, *block0_to_rgb_b;

Parameter *block1_conv_up_modulate_w, *block1_conv_up_modulate_b, *block1_conv_up_w, *block1_conv_up_b;
Parameter *block1_conv_modulate_w, *block1_conv_modulate_b, *block1_conv_w, *block1_conv_b;
Parameter *block1_to_rgb_modulate_w, *block1_to_rgb_modulate_b, *block1_to_rgb_w, *block1_to_rgb_b;

Parameter *block2_conv_up_modulate_w, *block2_conv_up_modulate_b, *block2_conv_up_w, *block2_conv_up_b;
Parameter *block2_conv_modulate_w, *block2_conv_modulate_b, *block2_conv_w, *block2_conv_b;
Parameter *block2_to_rgb_modulate_w, *block2_to_rgb_modulate_b, *block2_to_rgb_w, *block2_to_rgb_b;

Parameter *block3_conv_up_modulate_w, *block3_conv_up_modulate_b, *block3_conv_up_w, *block3_conv_up_b;
Parameter *block3_conv_modulate_w, *block3_conv_modulate_b, *block3_conv_w, *block3_conv_b;
Parameter *block3_to_rgb_modulate_w, *block3_to_rgb_modulate_b, *block3_to_rgb_w, *block3_to_rgb_b;

Parameter *block4_conv_up_modulate_w, *block4_conv_up_modulate_b, *block4_conv_up_w, *block4_conv_up_b;
Parameter *block4_conv_modulate_w, *block4_conv_modulate_b, *block4_conv_w, *block4_conv_b;
Parameter *block4_to_rgb_modulate_w, *block4_to_rgb_modulate_b, *block4_to_rgb_w, *block4_to_rgb_b;

Parameter *block5_conv_up_modulate_w, *block5_conv_up_modulate_b, *block5_conv_up_w, *block5_conv_up_b;
Parameter *block5_conv_modulate_w, *block5_conv_modulate_b, *block5_conv_w, *block5_conv_b;
Parameter *block5_to_rgb_modulate_w, *block5_to_rgb_modulate_b, *block5_to_rgb_w, *block5_to_rgb_b;

Parameter *block6_conv_up_modulate_w, *block6_conv_up_modulate_b, *block6_conv_up_w, *block6_conv_up_b;
Parameter *block6_conv_modulate_w, *block6_conv_modulate_b, *block6_conv_w, *block6_conv_b;
Parameter *block6_to_rgb_modulate_w, *block6_to_rgb_modulate_b, *block6_to_rgb_w, *block6_to_rgb_b;

// Noise parameters for each layer
Parameter *conv1_noise;
Parameter *block0_noise1, *block0_noise2;
Parameter *block1_noise1, *block1_noise2;
Parameter *block2_noise1, *block2_noise2;
Parameter *block3_noise1, *block3_noise2;
Parameter *block4_noise1, *block4_noise2;
Parameter *block5_noise1, *block5_noise2;
Parameter *block6_noise1, *block6_noise2;

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  mlp0_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp0_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp1_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp1_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp2_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp2_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp3_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp3_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp4_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp4_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp5_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp5_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp6_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp6_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  mlp7_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  mlp7_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  constant_input = new Parameter({BATCH_SIZE, 512, 4, 4}, param + pos, true, false, streams); pos += 512 * 4 * 4;

  kernel = new Parameter({1, 1, 4, 4}, param + pos, true, streams); pos += 4 * 4;

  conv1_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  conv1_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  conv1_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  conv1_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  to_rgb_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  to_rgb_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  to_rgb_w = new Parameter({3, 512, 1, 1}, param + pos, true, streams); pos += 3 * 512 * 1 * 1;
  to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block0_conv_up_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block0_conv_up_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block0_conv_up_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block0_conv_up_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block0_conv_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block0_conv_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block0_conv_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block0_conv_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block0_to_rgb_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block0_to_rgb_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block0_to_rgb_w = new Parameter({3, 512, 1, 1}, param + pos, true, streams); pos += 3 * 512;
  block0_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block1_conv_up_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block1_conv_up_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block1_conv_up_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block1_conv_up_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block1_conv_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block1_conv_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block1_conv_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block1_conv_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block1_to_rgb_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block1_to_rgb_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block1_to_rgb_w = new Parameter({3, 512, 1, 1}, param + pos, true, streams); pos += 3 * 512;
  block1_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block2_conv_up_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block2_conv_up_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block2_conv_up_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block2_conv_up_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block2_conv_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block2_conv_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block2_conv_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block2_conv_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block2_to_rgb_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block2_to_rgb_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block2_to_rgb_w = new Parameter({3, 512, 1, 1}, param + pos, true, streams); pos += 3 * 512;
  block2_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block3_conv_up_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block3_conv_up_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block3_conv_up_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block3_conv_up_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  
  block3_conv_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block3_conv_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block3_conv_w = new Parameter({512, 512, 3, 3}, param + pos, true, streams); pos += 512 * 512 * 3 * 3;
  block3_conv_b = new Parameter({512}, param + pos, true, streams); pos += 512;

  block3_to_rgb_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block3_to_rgb_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block3_to_rgb_w = new Parameter({3, 512, 1, 1}, param + pos, true, streams); pos += 3 * 512;
  block3_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block4_conv_up_modulate_w = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block4_conv_up_modulate_b = new Parameter({512}, param + pos, true, streams); pos += 512;
  block4_conv_up_w = new Parameter({256, 512, 3, 3}, param + pos, true, streams); pos += 256 * 512 * 3 * 3;
  block4_conv_up_b = new Parameter({256}, param + pos, true, streams); pos += 256;

  block4_conv_modulate_w = new Parameter({256, 512}, param + pos, true, streams); pos += 256 * 512;
  block4_conv_modulate_b = new Parameter({256}, param + pos, true, streams); pos += 256;
  block4_conv_w = new Parameter({256, 256, 3, 3}, param + pos, true, streams); pos += 256 * 256 * 3 * 3;
  block4_conv_b = new Parameter({256}, param + pos, true, streams); pos += 256;

  block4_to_rgb_modulate_w = new Parameter({256, 512}, param + pos, true, streams); pos += 256 * 512;
  block4_to_rgb_modulate_b = new Parameter({256}, param + pos, true, streams); pos += 256;
  block4_to_rgb_w = new Parameter({3, 256, 1, 1}, param + pos, true, streams); pos += 3 * 256;
  block4_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block5_conv_up_modulate_w = new Parameter({256, 512}, param + pos, true, streams); pos += 256 * 512;
  block5_conv_up_modulate_b = new Parameter({256}, param + pos, true, streams); pos += 256;
  block5_conv_up_w = new Parameter({128, 256, 3, 3}, param + pos, true, streams); pos += 128 * 256 * 3 * 3;
  block5_conv_up_b = new Parameter({128}, param + pos, true, streams); pos += 128;

  block5_conv_modulate_w = new Parameter({128, 512}, param + pos, true, streams); pos += 128 * 512;
  block5_conv_modulate_b = new Parameter({128}, param + pos, true, streams); pos += 128;
  block5_conv_w = new Parameter({128, 128, 3, 3}, param + pos, true, streams); pos += 128 * 128 * 3 * 3;
  block5_conv_b = new Parameter({128}, param + pos, true, streams); pos += 128;

  block5_to_rgb_modulate_w = new Parameter({128, 512}, param + pos, true, streams); pos += 128 * 512;
  block5_to_rgb_modulate_b = new Parameter({128}, param + pos, true, streams); pos += 128;
  block5_to_rgb_w = new Parameter({3, 128, 1, 1}, param + pos, true, streams); pos += 3 * 128;
  block5_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  block6_conv_up_modulate_w = new Parameter({128, 512}, param + pos, true, streams); pos += 128 * 512;
  block6_conv_up_modulate_b = new Parameter({128}, param + pos, true, streams); pos += 128;
  block6_conv_up_w = new Parameter({64, 128, 3, 3}, param + pos, true, streams); pos += 64 * 128 * 3 * 3;
  block6_conv_up_b = new Parameter({64}, param + pos, true, streams); pos += 64;

  block6_conv_modulate_w = new Parameter({64, 512}, param + pos, true, streams); pos += 64 * 512;
  block6_conv_modulate_b = new Parameter({64}, param + pos, true, streams); pos += 64;
  block6_conv_w = new Parameter({64, 64, 3, 3}, param + pos, true, streams); pos += 64 * 64 * 3 * 3;
  block6_conv_b = new Parameter({64}, param + pos, true, streams); pos += 64;

  block6_to_rgb_modulate_w = new Parameter({64, 512}, param + pos, true, streams); pos += 64 * 512;
  block6_to_rgb_modulate_b = new Parameter({64}, param + pos, true, streams); pos += 64;
  block6_to_rgb_w = new Parameter({3, 64, 1, 1}, param + pos, true, streams); pos += 3 * 64;
  block6_to_rgb_b = new Parameter({3}, param + pos, true, streams); pos += 3;

  conv1_noise = new Parameter({4, 4}, param + pos, true, streams); pos += 4 * 4;
  block0_noise1 = new Parameter({8, 8}, param + pos, true, streams); pos += 8 * 8;
  block0_noise2 = new Parameter({8, 8}, param + pos, true, streams); pos += 8 * 8;
  block1_noise1 = new Parameter({16, 16}, param + pos, true, streams); pos += 16 * 16;
  block1_noise2 = new Parameter({16, 16}, param + pos, true, streams); pos += 16 * 16;
  block2_noise1 = new Parameter({32, 32}, param + pos, true, streams); pos += 32 * 32;
  block2_noise2 = new Parameter({32, 32}, param + pos, true, streams); pos += 32 * 32;
  block3_noise1 = new Parameter({64, 64}, param + pos, true, streams); pos += 64 * 64;
  block3_noise2 = new Parameter({64, 64}, param + pos, true, streams); pos += 64 * 64;
  block4_noise1 = new Parameter({128, 128}, param + pos, true, streams); pos += 128 * 128;
  block4_noise2 = new Parameter({128, 128}, param + pos, true, streams); pos += 128 * 128;
  block5_noise1 = new Parameter({256, 256}, param + pos, true, streams); pos += 256 * 256;
  block5_noise2 = new Parameter({256, 256}, param + pos, true, streams); pos += 256 * 256;
  block6_noise1 = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;
  block6_noise2 = new Parameter({512, 512}, param + pos, true, streams); pos += 512 * 512;

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  delete mlp0_w;
  delete mlp0_b;
  delete mlp1_w;
  delete mlp1_b;
  delete mlp2_w;
  delete mlp2_b;
  delete mlp3_w;
  delete mlp3_b;
  delete mlp4_w;
  delete mlp4_b;
  delete mlp5_w;
  delete mlp5_b;
  delete mlp6_w;
  delete mlp6_b;
  delete mlp7_w;
  delete mlp7_b;

  delete constant_input;
  delete kernel;
  delete conv1_modulate_w;
  delete conv1_modulate_b;
  delete conv1_w;
  delete conv1_b;
  delete to_rgb_modulate_w;
  delete to_rgb_modulate_b;
  delete to_rgb_w;
  delete to_rgb_b;

  delete block0_conv_up_modulate_w; delete block0_conv_up_modulate_b; delete block0_conv_up_w; delete block0_conv_up_b;
  delete block0_conv_modulate_w; delete block0_conv_modulate_b; delete block0_conv_w; delete block0_conv_b;
  delete block0_to_rgb_modulate_w; delete block0_to_rgb_modulate_b; delete block0_to_rgb_w; delete block0_to_rgb_b;

  delete block1_conv_up_modulate_w; delete block1_conv_up_modulate_b; delete block1_conv_up_w; delete block1_conv_up_b;
  delete block1_conv_modulate_w; delete block1_conv_modulate_b; delete block1_conv_w; delete block1_conv_b;
  delete block1_to_rgb_modulate_w; delete block1_to_rgb_modulate_b; delete block1_to_rgb_w; delete block1_to_rgb_b;

  delete block2_conv_up_modulate_w; delete block2_conv_up_modulate_b; delete block2_conv_up_w; delete block2_conv_up_b;
  delete block2_conv_modulate_w; delete block2_conv_modulate_b; delete block2_conv_w; delete block2_conv_b;
  delete block2_to_rgb_modulate_w; delete block2_to_rgb_modulate_b; delete block2_to_rgb_w; delete block2_to_rgb_b;

  delete block3_conv_up_modulate_w; delete block3_conv_up_modulate_b; delete block3_conv_up_w; delete block3_conv_up_b;
  delete block3_conv_modulate_w; delete block3_conv_modulate_b; delete block3_conv_w; delete block3_conv_b;
  delete block3_to_rgb_modulate_w; delete block3_to_rgb_modulate_b; delete block3_to_rgb_w; delete block3_to_rgb_b;

  delete block4_conv_up_modulate_w; delete block4_conv_up_modulate_b; delete block4_conv_up_w; delete block4_conv_up_b;
  delete block4_conv_modulate_w; delete block4_conv_modulate_b; delete block4_conv_w; delete block4_conv_b;
  delete block4_to_rgb_modulate_w; delete block4_to_rgb_modulate_b; delete block4_to_rgb_w; delete block4_to_rgb_b;

  delete block5_conv_up_modulate_w; delete block5_conv_up_modulate_b; delete block5_conv_up_w; delete block5_conv_up_b;
  delete block5_conv_modulate_w; delete block5_conv_modulate_b; delete block5_conv_w; delete block5_conv_b;
  delete block5_to_rgb_modulate_w; delete block5_to_rgb_modulate_b; delete block5_to_rgb_w; delete block5_to_rgb_b;

  delete block6_conv_up_modulate_w; delete block6_conv_up_modulate_b; delete block6_conv_up_w; delete block6_conv_up_b;
  delete block6_conv_modulate_w; delete block6_conv_modulate_b; delete block6_conv_w; delete block6_conv_b;
  delete block6_to_rgb_modulate_w; delete block6_to_rgb_modulate_b; delete block6_to_rgb_w; delete block6_to_rgb_b;

  delete conv1_noise;
  delete block0_noise1; delete block0_noise2;
  delete block1_noise1; delete block1_noise2;
  delete block2_noise1; delete block2_noise2;
  delete block3_noise1; delete block3_noise2;
  delete block4_noise1; delete block4_noise2;
  delete block5_noise1; delete block5_noise2;
  delete block6_noise1; delete block6_noise2;
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *mlp0_a, *mlp1_a, *mlp2_a, *mlp3_a, *mlp4_a, *mlp5_a, *mlp6_a, *mlp7_a;
Activation *constant_input_a;

// conv1 activations
Activation *conv1_style_a, *conv1_weight_a, *conv1_demod_a;
Activation *conv1_output_a;

// ToRGB activations
Activation *to_rgb_style_a, *to_rgb_weight_a;
Activation *to_rgb_output_a;

// Activations for 7 blocks
Activation *block0_conv_up_style_a, *block0_conv_up_weight_a, *block0_conv_up_demod_a;
Activation *block0_conv_up_conv_a, *block0_conv_up_upsample_a, *block0_conv_up_conv2_a, *block0_conv_up_output_a;
Activation *block0_conv_style_a, *block0_conv_weight_a, *block0_conv_demod_a;
Activation *block0_conv_output_a;
Activation *block0_to_rgb_style_a, *block0_to_rgb_weight_a;
Activation *block0_to_rgb_output_a;
Activation *block0_skip_a;
Activation *block0_to_rgb_skip_upsample_a, *block0_to_rgb_skip_conv_a;

Activation *block1_conv_up_style_a, *block1_conv_up_weight_a, *block1_conv_up_demod_a;
Activation *block1_conv_up_conv_a, *block1_conv_up_upsample_a, *block1_conv_up_conv2_a, *block1_conv_up_output_a;
Activation *block1_conv_style_a, *block1_conv_weight_a, *block1_conv_demod_a;
Activation *block1_conv_output_a;
Activation *block1_to_rgb_style_a, *block1_to_rgb_weight_a;
Activation *block1_to_rgb_output_a;
Activation *block1_skip_a;
Activation *block1_to_rgb_skip_upsample_a, *block1_to_rgb_skip_conv_a;

Activation *block2_conv_up_style_a, *block2_conv_up_weight_a, *block2_conv_up_demod_a;
Activation *block2_conv_up_conv_a, *block2_conv_up_upsample_a, *block2_conv_up_conv2_a, *block2_conv_up_output_a;
Activation *block2_conv_style_a, *block2_conv_weight_a, *block2_conv_demod_a;
Activation *block2_conv_output_a;
Activation *block2_to_rgb_style_a, *block2_to_rgb_weight_a;
Activation *block2_to_rgb_output_a;
Activation *block2_skip_a;
Activation *block2_to_rgb_skip_upsample_a, *block2_to_rgb_skip_conv_a;

Activation *block3_conv_up_style_a, *block3_conv_up_weight_a, *block3_conv_up_demod_a;
Activation *block3_conv_up_conv_a, *block3_conv_up_upsample_a, *block3_conv_up_conv2_a, *block3_conv_up_output_a;
Activation *block3_conv_style_a, *block3_conv_weight_a, *block3_conv_demod_a;
Activation *block3_conv_output_a;
Activation *block3_to_rgb_style_a, *block3_to_rgb_weight_a;
Activation *block3_to_rgb_output_a;
Activation *block3_skip_a;
Activation *block3_to_rgb_skip_upsample_a, *block3_to_rgb_skip_conv_a;

Activation *block4_conv_up_style_a, *block4_conv_up_weight_a, *block4_conv_up_demod_a;
Activation *block4_conv_up_conv_a, *block4_conv_up_upsample_a, *block4_conv_up_conv2_a, *block4_conv_up_output_a;
Activation *block4_conv_style_a, *block4_conv_weight_a, *block4_conv_demod_a;
Activation *block4_conv_output_a;
Activation *block4_to_rgb_style_a, *block4_to_rgb_weight_a;
Activation *block4_to_rgb_output_a;
Activation *block4_skip_a;
Activation *block4_to_rgb_skip_upsample_a, *block4_to_rgb_skip_conv_a;

Activation *block5_conv_up_style_a, *block5_conv_up_weight_a, *block5_conv_up_demod_a;
Activation *block5_conv_up_conv_a, *block5_conv_up_upsample_a, *block5_conv_up_conv2_a, *block5_conv_up_output_a;
Activation *block5_conv_style_a, *block5_conv_weight_a, *block5_conv_demod_a;
Activation *block5_conv_output_a;
Activation *block5_to_rgb_style_a, *block5_to_rgb_weight_a;
Activation *block5_to_rgb_output_a;
Activation *block5_skip_a;
Activation *block5_to_rgb_skip_upsample_a, *block5_to_rgb_skip_conv_a;

Activation *block6_conv_up_style_a, *block6_conv_up_weight_a, *block6_conv_up_demod_a;
Activation *block6_conv_up_conv_a, *block6_conv_up_upsample_a, *block6_conv_up_conv2_a, *block6_conv_up_output_a;
Activation *block6_conv_style_a, *block6_conv_weight_a, *block6_conv_demod_a;
Activation *block6_conv_output_a;
Activation *block6_to_rgb_style_a, *block6_to_rgb_weight_a;
Activation *block6_to_rgb_output_a;
Activation *block6_skip_a;
Activation *block6_to_rgb_skip_upsample_a, *block6_to_rgb_skip_conv_a;

Activation *conv1_col_buffer, *block0_conv_col_buffer, *block1_conv_col_buffer, *block2_conv_col_buffer, *block3_conv_col_buffer, *block4_conv_col_buffer, *block5_conv_col_buffer, *block6_conv_col_buffer;

Activation *block0_trans_weight_transposed, *block1_trans_weight_transposed, *block2_trans_weight_transposed, *block3_trans_weight_transposed, *block4_trans_weight_transposed, *block5_trans_weight_transposed, *block6_trans_weight_transposed;

Activation *block0_trans_col_buffer, *block1_trans_col_buffer, *block2_trans_col_buffer, *block3_trans_col_buffer, *block4_trans_col_buffer, *block5_trans_col_buffer, *block6_trans_col_buffer;

void alloc_activations_new() {
  conv1_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 4*4}, false, streams);
  block0_conv_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 8*8}, false, streams);
  block1_conv_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 16*16}, false, streams);
  block2_conv_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 32*32}, false, streams);
  block3_conv_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 64*64}, false, streams);
  block4_conv_col_buffer = new Activation({BATCH_SIZE, 256*3*3, 128*128}, false, streams);
  block5_conv_col_buffer = new Activation({BATCH_SIZE, 128*3*3, 256*256}, false, streams);
  block6_conv_col_buffer = new Activation({BATCH_SIZE, 64*3*3, 512*512}, false, streams);

  block0_trans_weight_transposed = new Activation({BATCH_SIZE, 512, 3, 3, 512}, false, streams);
  block1_trans_weight_transposed = new Activation({BATCH_SIZE, 512, 3, 3, 512}, false, streams);
  block2_trans_weight_transposed = new Activation({BATCH_SIZE, 512, 3, 3, 512}, false, streams);
  block3_trans_weight_transposed = new Activation({BATCH_SIZE, 512, 3, 3, 512}, false, streams);
  block4_trans_weight_transposed = new Activation({BATCH_SIZE, 256, 3, 3, 512}, false, streams);
  block5_trans_weight_transposed = new Activation({BATCH_SIZE, 128, 3, 3, 256}, false, streams);
  block6_trans_weight_transposed = new Activation({BATCH_SIZE, 64, 3, 3, 128}, false, streams);

  block0_trans_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 4*4}, false, streams);
  block1_trans_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 8*8}, false, streams);
  block2_trans_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 16*16}, false, streams);
  block3_trans_col_buffer = new Activation({BATCH_SIZE, 512*3*3, 32*32}, false, streams);
  block4_trans_col_buffer = new Activation({BATCH_SIZE, 256*3*3, 64*64}, false, streams);
  block5_trans_col_buffer = new Activation({BATCH_SIZE, 128*3*3, 128*128}, false, streams);
  block6_trans_col_buffer = new Activation({BATCH_SIZE, 64*3*3, 256*256}, false, streams);

  /*
  conv1_output_a->malloc_device();
  block0_conv_output_a->malloc_device();
  block1_conv_output_a->malloc_device();
  block2_conv_output_a->malloc_device();
  block3_conv_output_a->malloc_device();
  block4_conv_output_a->malloc_device();
  block5_conv_output_a->malloc_device();
  block6_conv_output_a->malloc_device();

  conv1_weight_a->malloc_device();
  block0_conv_weight_a->malloc_device();
  block1_conv_weight_a->malloc_device();
  block2_conv_weight_a->malloc_device();
  block3_conv_weight_a->malloc_device();
  block4_conv_weight_a->malloc_device();
  block5_conv_weight_a->malloc_device();
  block6_conv_weight_a->malloc_device();

  conv1_col_buffer->malloc_device();
  block0_conv_col_buffer->malloc_device();
  block1_conv_col_buffer->malloc_device();
  block2_conv_col_buffer->malloc_device();
  block3_conv_col_buffer->malloc_device();
  block4_conv_col_buffer->malloc_device();
  block5_conv_col_buffer->malloc_device();
  block6_conv_col_buffer->malloc_device();

  block0_conv_up_output_a->malloc_device();
  block1_conv_up_output_a->malloc_device();
  block2_conv_up_output_a->malloc_device();
  block3_conv_up_output_a->malloc_device();
  block4_conv_up_output_a->malloc_device();
  block5_conv_up_output_a->malloc_device();
  block6_conv_up_output_a->malloc_device();

  block0_trans_weight_transposed->malloc_device();
  block1_trans_weight_transposed->malloc_device();
  block2_trans_weight_transposed->malloc_device();
  block3_trans_weight_transposed->malloc_device();
  block4_trans_weight_transposed->malloc_device();
  block5_trans_weight_transposed->malloc_device();
  block6_trans_weight_transposed->malloc_device();

  block0_trans_col_buffer->malloc_device();
  block1_trans_col_buffer->malloc_device();
  block2_trans_col_buffer->malloc_device();
  block3_trans_col_buffer->malloc_device();
  block4_trans_col_buffer->malloc_device();
  block5_trans_col_buffer->malloc_device();
  block6_trans_col_buffer->malloc_device();
  */

  for (int i = 0; i < NUM_GPUS; ++i) {
    // Select the current GPU. All subsequent CUDA calls will target this device.
    CHECK_CUDA(cudaSetDevice(i));
    // Create a stream for the current GPU
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }
}

void free_activations_new() {
  for (int i = 0; i < NUM_GPUS; ++i) {
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }

  delete conv1_col_buffer;
  delete block0_conv_col_buffer; delete block1_conv_col_buffer;
  delete block2_conv_col_buffer; delete block3_conv_col_buffer;
  delete block4_conv_col_buffer; delete block5_conv_col_buffer;
  delete block6_conv_col_buffer;
  
  delete block0_trans_weight_transposed; delete block1_trans_weight_transposed;
  delete block2_trans_weight_transposed; delete block3_trans_weight_transposed;
  delete block4_trans_weight_transposed; delete block5_trans_weight_transposed;
  delete block6_trans_weight_transposed;

  delete block0_trans_col_buffer; delete block1_trans_col_buffer;
  delete block2_trans_col_buffer; delete block3_trans_col_buffer;
  delete block4_trans_col_buffer; delete block5_trans_col_buffer;
  delete block6_trans_col_buffer;
}

void alloc_activations() {
  printf("BATCH SIZE: %d\n", BATCH_SIZE);
  mlp0_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp1_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp2_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp3_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp4_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp5_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp6_a = new Activation({BATCH_SIZE, 512}, false, streams);
  mlp7_a = new Activation({BATCH_SIZE, 512}, false, streams);

  constant_input_a = new Activation({BATCH_SIZE, 512, 4, 4}, false, streams);

  // ModulatedConv2d activations for conv1
  conv1_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  conv1_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  conv1_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  conv1_output_a = new Activation({BATCH_SIZE, 512, 4, 4}, false, streams);

  // ToRGB activations
  to_rgb_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 512, 1, 1}, false, streams);
  to_rgb_output_a = new Activation({BATCH_SIZE, 3, 4, 4}, false, streams);

  // Block 0: 8x8, 512 channels
  block0_conv_up_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block0_conv_up_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block0_conv_up_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block0_conv_up_conv_a = new Activation({BATCH_SIZE, 512, 9, 9}, false, streams);
  block0_conv_up_upsample_a = new Activation({BATCH_SIZE, 512, 11, 11}, false, streams);
  block0_conv_up_conv2_a = new Activation({BATCH_SIZE, 512, 8, 8}, false, streams);
  block0_conv_up_output_a = new Activation({BATCH_SIZE, 512, 8, 8}, false, streams);
  
  block0_conv_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block0_conv_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block0_conv_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block0_conv_output_a = new Activation({BATCH_SIZE, 512, 8, 8}, false, streams);
  
  block0_to_rgb_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block0_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 512, 1, 1}, false, streams);
  block0_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 8, 8}, false, streams);
  block0_skip_a = new Activation({BATCH_SIZE, 3, 8, 8}, false, streams);
  block0_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 11, 11}, false, streams);
  block0_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 8, 8}, false, streams);

  // Block 1: 16x16, 512 channels
  block1_conv_up_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block1_conv_up_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block1_conv_up_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block1_conv_up_conv_a = new Activation({BATCH_SIZE, 512, 17, 17}, false, streams);
  block1_conv_up_upsample_a = new Activation({BATCH_SIZE, 512, 19, 19}, false, streams);
  block1_conv_up_conv2_a = new Activation({BATCH_SIZE, 512, 16, 16}, false, streams);
  block1_conv_up_output_a = new Activation({BATCH_SIZE, 512, 16, 16}, false, streams);
  
  block1_conv_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block1_conv_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block1_conv_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block1_conv_output_a = new Activation({BATCH_SIZE, 512, 16, 16}, false, streams);
  
  block1_to_rgb_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block1_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 512, 1, 1}, false, streams);
  block1_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 16, 16}, false, streams);
  block1_skip_a = new Activation({BATCH_SIZE, 3, 16, 16}, false, streams);
  block1_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 19, 19}, false, streams);
  block1_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 16, 16}, false, streams);

  // Block 2: 32x32, 512 channels
  block2_conv_up_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block2_conv_up_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block2_conv_up_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block2_conv_up_conv_a = new Activation({BATCH_SIZE, 512, 33, 33}, false, streams);
  block2_conv_up_upsample_a = new Activation({BATCH_SIZE, 512, 35, 35}, false, streams);
  block2_conv_up_conv2_a = new Activation({BATCH_SIZE, 512, 32, 32}, false, streams);
  block2_conv_up_output_a = new Activation({BATCH_SIZE, 512, 32, 32}, false, streams);
  
  block2_conv_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block2_conv_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block2_conv_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block2_conv_output_a = new Activation({BATCH_SIZE, 512, 32, 32}, false, streams);
  
  block2_to_rgb_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block2_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 512, 1, 1}, false, streams);
  block2_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 32, 32}, false, streams);
  block2_skip_a = new Activation({BATCH_SIZE, 3, 32, 32}, false, streams);
  block2_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 35, 35}, false, streams);
  block2_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 32, 32}, false, streams);

  // Block 3: 64x64, 512 channels
  block3_conv_up_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block3_conv_up_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block3_conv_up_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block3_conv_up_conv_a = new Activation({BATCH_SIZE, 512, 65, 65}, false, streams);
  block3_conv_up_upsample_a = new Activation({BATCH_SIZE, 512, 67, 67}, false, streams);
  block3_conv_up_conv2_a = new Activation({BATCH_SIZE, 512, 64, 64}, false, streams);
  block3_conv_up_output_a = new Activation({BATCH_SIZE, 512, 64, 64}, false, streams);
  
  block3_conv_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block3_conv_weight_a = new Activation({BATCH_SIZE, 512, 512, 3, 3}, false, streams);
  block3_conv_demod_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block3_conv_output_a = new Activation({BATCH_SIZE, 512, 64, 64}, false, streams);
  
  block3_to_rgb_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block3_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 512, 1, 1}, false, streams);
  block3_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 64, 64}, false, streams);
  block3_skip_a = new Activation({BATCH_SIZE, 3, 64, 64}, false, streams);
  block3_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 67, 67}, false, streams);
  block3_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 64, 64}, false, streams);

  // Block 4: 128x128, 256 channels  
  block4_conv_up_style_a = new Activation({BATCH_SIZE, 512}, false, streams);
  block4_conv_up_weight_a = new Activation({BATCH_SIZE, 256, 512, 3, 3}, false, streams);
  block4_conv_up_demod_a = new Activation({BATCH_SIZE, 256}, false, streams);
  block4_conv_up_conv_a = new Activation({BATCH_SIZE, 256, 129, 129}, false, streams);
  block4_conv_up_upsample_a = new Activation({BATCH_SIZE, 256, 131, 131}, false, streams);
  block4_conv_up_conv2_a = new Activation({BATCH_SIZE, 256, 128, 128}, false, streams);
  block4_conv_up_output_a = new Activation({BATCH_SIZE, 256, 128, 128}, false, streams);
  
  block4_conv_style_a = new Activation({BATCH_SIZE, 256}, false, streams);
  block4_conv_weight_a = new Activation({BATCH_SIZE, 256, 256, 3, 3}, false, streams);
  block4_conv_demod_a = new Activation({BATCH_SIZE, 256}, false, streams);
  block4_conv_output_a = new Activation({BATCH_SIZE, 256, 128, 128}, false, streams);
  
  block4_to_rgb_style_a = new Activation({BATCH_SIZE, 256}, false, streams);
  block4_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 256, 1, 1}, false, streams);
  block4_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 128, 128}, false, streams);
  block4_skip_a = new Activation({BATCH_SIZE, 3, 128, 128}, false, streams);
  block4_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 131, 131}, false, streams);
  block4_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 128, 128}, false, streams);

  // Block 5: 256x256, 128 channels
  block5_conv_up_style_a = new Activation({BATCH_SIZE, 256}, false, streams);
  block5_conv_up_weight_a = new Activation({BATCH_SIZE, 128, 256, 3, 3}, false, streams);
  block5_conv_up_demod_a = new Activation({BATCH_SIZE, 128}, false, streams);
  block5_conv_up_conv_a = new Activation({BATCH_SIZE, 128, 257, 257}, false, streams);
  block5_conv_up_upsample_a = new Activation({BATCH_SIZE, 128, 259, 259}, false, streams);
  block5_conv_up_conv2_a = new Activation({BATCH_SIZE, 128, 256, 256}, false, streams);
  block5_conv_up_output_a = new Activation({BATCH_SIZE, 128, 256, 256}, false, streams);
  
  block5_conv_style_a = new Activation({BATCH_SIZE, 128}, false, streams);
  block5_conv_weight_a = new Activation({BATCH_SIZE, 128, 128, 3, 3}, false, streams);
  block5_conv_demod_a = new Activation({BATCH_SIZE, 128}, false, streams);
  block5_conv_output_a = new Activation({BATCH_SIZE, 128, 256, 256}, false, streams);
  
  block5_to_rgb_style_a = new Activation({BATCH_SIZE, 128}, false, streams);
  block5_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 128, 1, 1}, false, streams);
  block5_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 256, 256}, false, streams);
  block5_skip_a = new Activation({BATCH_SIZE, 3, 256, 256}, false, streams);
  block5_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 259, 259}, false, streams);
  block5_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 256, 256}, false, streams);

  // Block 6: 512x512, 64 channels
  block6_conv_up_style_a = new Activation({BATCH_SIZE, 128}, false, streams);
  block6_conv_up_weight_a = new Activation({BATCH_SIZE, 64, 128, 3, 3}, false, streams);
  block6_conv_up_demod_a = new Activation({BATCH_SIZE, 64}, false, streams);
  block6_conv_up_conv_a = new Activation({BATCH_SIZE, 64, 513, 513}, false, streams);
  block6_conv_up_upsample_a = new Activation({BATCH_SIZE, 64, 515, 515}, false, streams);
  block6_conv_up_conv2_a = new Activation({BATCH_SIZE, 64, 512, 512}, false, streams);
  block6_conv_up_output_a = new Activation({BATCH_SIZE, 64, 512, 512}, false, streams);
  
  block6_conv_style_a = new Activation({BATCH_SIZE, 64}, false, streams);
  block6_conv_weight_a = new Activation({BATCH_SIZE, 64, 64, 3, 3}, false, streams);
  block6_conv_demod_a = new Activation({BATCH_SIZE, 64}, false, streams);
  block6_conv_output_a = new Activation({BATCH_SIZE, 64, 512, 512}, false, streams);
  
  block6_to_rgb_style_a = new Activation({BATCH_SIZE, 64}, false, streams);
  block6_to_rgb_weight_a = new Activation({BATCH_SIZE, 3, 64, 1, 1}, false, streams);
  block6_to_rgb_output_a = new Activation({BATCH_SIZE, 3, 512, 512}, false, streams);
  block6_skip_a = new Activation({BATCH_SIZE, 3, 512, 512}, false, streams);
  block6_to_rgb_skip_upsample_a = new Activation({BATCH_SIZE, 3, 515, 515}, false, streams);
  block6_to_rgb_skip_conv_a = new Activation({BATCH_SIZE, 3, 512, 512}, false, streams);

  alloc_activations_new();
}

void free_activations() {
  free_activations_new();

  delete mlp0_a;
  delete mlp1_a;
  delete mlp2_a;
  delete mlp3_a;
  delete mlp4_a;
  delete mlp5_a;
  delete mlp6_a;
  delete mlp7_a;

  delete constant_input_a;

  delete conv1_style_a;
  delete conv1_weight_a;
  delete conv1_demod_a;
  delete conv1_output_a;

  delete to_rgb_style_a;
  delete to_rgb_weight_a;
  delete to_rgb_output_a;

  // Free block activations - All blocks
  delete block0_conv_up_style_a; delete block0_conv_up_weight_a; delete block0_conv_up_demod_a;
  delete block0_conv_up_conv_a; delete block0_conv_up_upsample_a; delete block0_conv_up_conv2_a; delete block0_conv_up_output_a;
  delete block0_conv_style_a; delete block0_conv_weight_a; delete block0_conv_demod_a; 
  delete block0_conv_output_a;
  delete block0_to_rgb_style_a; delete block0_to_rgb_weight_a;
  delete block0_to_rgb_output_a;
  delete block0_skip_a;
  delete block0_to_rgb_skip_upsample_a; delete block0_to_rgb_skip_conv_a;

  delete block1_conv_up_style_a; delete block1_conv_up_weight_a; delete block1_conv_up_demod_a;
  delete block1_conv_up_conv_a; delete block1_conv_up_upsample_a; delete block1_conv_up_conv2_a; delete block1_conv_up_output_a;
  delete block1_conv_style_a; delete block1_conv_weight_a; delete block1_conv_demod_a; 
  delete block1_conv_output_a;
  delete block1_to_rgb_style_a; delete block1_to_rgb_weight_a;
  delete block1_to_rgb_output_a;
  delete block1_skip_a;
  delete block1_to_rgb_skip_upsample_a; delete block1_to_rgb_skip_conv_a;

  delete block2_conv_up_style_a; delete block2_conv_up_weight_a; delete block2_conv_up_demod_a;
  delete block2_conv_up_conv_a; delete block2_conv_up_upsample_a; delete block2_conv_up_conv2_a; delete block2_conv_up_output_a;
  delete block2_conv_style_a; delete block2_conv_weight_a; delete block2_conv_demod_a; 
  delete block2_conv_output_a;
  delete block2_to_rgb_style_a; delete block2_to_rgb_weight_a;
  delete block2_to_rgb_output_a;
  delete block2_skip_a;
  delete block2_to_rgb_skip_upsample_a; delete block2_to_rgb_skip_conv_a;

  delete block3_conv_up_style_a; delete block3_conv_up_weight_a; delete block3_conv_up_demod_a;
  delete block3_conv_up_conv_a; delete block3_conv_up_upsample_a; delete block3_conv_up_conv2_a; delete block3_conv_up_output_a;
  delete block3_conv_style_a; delete block3_conv_weight_a; delete block3_conv_demod_a; 
  delete block3_conv_output_a;
  delete block3_to_rgb_style_a; delete block3_to_rgb_weight_a;
  delete block3_to_rgb_output_a;
  delete block3_skip_a;
  delete block3_to_rgb_skip_upsample_a; delete block3_to_rgb_skip_conv_a;

  delete block4_conv_up_style_a; delete block4_conv_up_weight_a; delete block4_conv_up_demod_a;
  delete block4_conv_up_conv_a; delete block4_conv_up_upsample_a; delete block4_conv_up_conv2_a; delete block4_conv_up_output_a;
  delete block4_conv_style_a; delete block4_conv_weight_a; delete block4_conv_demod_a; 
  delete block4_conv_output_a;
  delete block4_to_rgb_style_a; delete block4_to_rgb_weight_a;
  delete block4_to_rgb_output_a;
  delete block4_skip_a;
  delete block4_to_rgb_skip_upsample_a; delete block4_to_rgb_skip_conv_a;

  delete block5_conv_up_style_a; delete block5_conv_up_weight_a; delete block5_conv_up_demod_a;
  delete block5_conv_up_conv_a; delete block5_conv_up_upsample_a; delete block5_conv_up_conv2_a; delete block5_conv_up_output_a;
  delete block5_conv_style_a; delete block5_conv_weight_a; delete block5_conv_demod_a; 
  delete block5_conv_output_a;
  delete block5_to_rgb_style_a; delete block5_to_rgb_weight_a;
  delete block5_to_rgb_output_a;
  delete block5_skip_a;
  delete block5_to_rgb_skip_upsample_a; delete block5_to_rgb_skip_conv_a;

  delete block6_conv_up_style_a; delete block6_conv_up_weight_a; delete block6_conv_up_demod_a;
  delete block6_conv_up_conv_a; delete block6_conv_up_upsample_a; delete block6_conv_up_conv2_a; delete block6_conv_up_output_a;
  delete block6_conv_style_a; delete block6_conv_weight_a; delete block6_conv_demod_a; 
  delete block6_conv_output_a;
  delete block6_to_rgb_style_a; delete block6_to_rgb_weight_a;
  delete block6_to_rgb_output_a;
  delete block6_skip_a;
  delete block6_to_rgb_skip_upsample_a; delete block6_to_rgb_skip_conv_a;
}

/* [Model Computation] */
void generate(float *inputs, float *outputs, size_t n_samples) {
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (n_samples == 1) {
    n_samples = BATCH_SIZE * mpi_size;
  }

  // --- Input Data Scattering ---
  size_t batches_per_rank = n_samples / (BATCH_SIZE * mpi_size);
  size_t local_input_size = batches_per_rank * BATCH_SIZE * 512;
  // float* local_inputs;
  // CHECK_CUDA(cudaMallocHost(&local_inputs, local_input_size * sizeof(float)));
  float* local_inputs = new float[local_input_size];

  MPI_Scatter(inputs, local_input_size, MPI_FLOAT,
              local_inputs, local_input_size, MPI_FLOAT, 0,
              MPI_COMM_WORLD);
  size_t local_output_size = batches_per_rank * BATCH_SIZE * 3 * 512 * 512;
  float* local_outputs = new float[local_output_size];

  // --- Parallel Computation Loop ---
  // This loop runs on all processes in parallel.

  for (size_t n = 0; n < batches_per_rank; n++) {
    /* Load a style from the local inputs */
    Tensor *input = new Tensor({BATCH_SIZE, 512}, local_inputs + BATCH_SIZE * n * 512, false, streams);

    /* Get latent from style through an 8-layer MLP */
    PixelNorm(input);
    // pixelNorm_wrapper(input, true, false, streams);

    input->to_device(streams);

    fusedLinearLeakyReLU_wrapper(input, mlp0_w, mlp0_b, mlp0_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp0_a, mlp1_w, mlp1_b, mlp1_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp1_a, mlp2_w, mlp2_b, mlp2_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp2_a, mlp3_w, mlp3_b, mlp3_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp3_a, mlp4_w, mlp4_b, mlp4_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp4_a, mlp5_w, mlp5_b, mlp5_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp5_a, mlp6_w, mlp6_b, mlp6_a, 0.01f, false, false, streams);
    fusedLinearLeakyReLU_wrapper(mlp6_a, mlp7_w, mlp7_b, mlp7_a, 0.01f, false, false, streams);

    StyledConv(constant_input, mlp7_a, conv1_modulate_w, conv1_modulate_b, conv1_w, conv1_b, kernel, conv1_noise, conv1_output_a,
               conv1_style_a, conv1_weight_a, conv1_demod_a, conv1_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(conv1_output_a, nullptr, mlp7_a, to_rgb_modulate_w, to_rgb_modulate_b, to_rgb_w, to_rgb_b, kernel, to_rgb_output_a,
          to_rgb_style_a, to_rgb_weight_a, nullptr, nullptr, nullptr, nullptr, streams); // Creates the first skip connection

    // Block 0: 4x4 -> 8x8
    StyledConv(conv1_output_a, mlp7_a, block0_conv_up_modulate_w, block0_conv_up_modulate_b, block0_conv_up_w, block0_conv_up_b, kernel, block0_noise1, block0_conv_up_output_a,
               block0_conv_up_style_a, block0_conv_up_weight_a, block0_conv_up_demod_a, block0_trans_col_buffer, block0_trans_weight_transposed, block0_conv_up_conv_a, block0_conv_up_upsample_a, block0_conv_up_conv2_a, true, 0, streams);
    StyledConv(block0_conv_up_output_a, mlp7_a, block0_conv_modulate_w, block0_conv_modulate_b, block0_conv_w, block0_conv_b, kernel, block0_noise2, block0_conv_output_a,
               block0_conv_style_a, block0_conv_weight_a, block0_conv_demod_a, block0_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block0_conv_output_a, to_rgb_output_a, mlp7_a, block0_to_rgb_modulate_w, block0_to_rgb_modulate_b, block0_to_rgb_w, block0_to_rgb_b, kernel, block0_to_rgb_output_a,
          block0_to_rgb_style_a, block0_to_rgb_weight_a, nullptr, block0_to_rgb_skip_upsample_a, block0_to_rgb_skip_conv_a, block0_skip_a, streams);

    // Block 1: 8x8 -> 16x16
    StyledConv(block0_conv_output_a, mlp7_a, block1_conv_up_modulate_w, block1_conv_up_modulate_b, block1_conv_up_w, block1_conv_up_b, kernel, block1_noise1, block1_conv_up_output_a,
               block1_conv_up_style_a, block1_conv_up_weight_a, block1_conv_up_demod_a,  block1_trans_col_buffer, block1_trans_weight_transposed, block1_conv_up_conv_a, block1_conv_up_upsample_a, block1_conv_up_conv2_a, true, 0, streams);
    StyledConv(block1_conv_up_output_a, mlp7_a, block1_conv_modulate_w, block1_conv_modulate_b, block1_conv_w, block1_conv_b, kernel, block1_noise2, block1_conv_output_a,
               block1_conv_style_a, block1_conv_weight_a, block1_conv_demod_a, block1_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block1_conv_output_a, block0_to_rgb_output_a, mlp7_a, block1_to_rgb_modulate_w, block1_to_rgb_modulate_b, block1_to_rgb_w, block1_to_rgb_b, kernel, block1_to_rgb_output_a,
          block1_to_rgb_style_a, block1_to_rgb_weight_a, nullptr, block1_to_rgb_skip_upsample_a, block1_to_rgb_skip_conv_a, block1_skip_a, streams);

    // Block 2: 16x16 -> 32x32
    StyledConv(block1_conv_output_a, mlp7_a, block2_conv_up_modulate_w, block2_conv_up_modulate_b, block2_conv_up_w, block2_conv_up_b, kernel, block2_noise1, block2_conv_up_output_a,
               block2_conv_up_style_a, block2_conv_up_weight_a, block2_conv_up_demod_a,  block2_trans_col_buffer, block2_trans_weight_transposed, block2_conv_up_conv_a, block2_conv_up_upsample_a, block2_conv_up_conv2_a, true, 0, streams);
    StyledConv(block2_conv_up_output_a, mlp7_a, block2_conv_modulate_w, block2_conv_modulate_b, block2_conv_w, block2_conv_b, kernel, block2_noise2, block2_conv_output_a,
               block2_conv_style_a, block2_conv_weight_a, block2_conv_demod_a, block2_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block2_conv_output_a, block1_to_rgb_output_a, mlp7_a, block2_to_rgb_modulate_w, block2_to_rgb_modulate_b, block2_to_rgb_w, block2_to_rgb_b, kernel, block2_to_rgb_output_a,
          block2_to_rgb_style_a, block2_to_rgb_weight_a, nullptr, block2_to_rgb_skip_upsample_a, block2_to_rgb_skip_conv_a, block2_skip_a, streams);

    // Block 3: 32x32 -> 64x64
    StyledConv(block2_conv_output_a, mlp7_a, block3_conv_up_modulate_w, block3_conv_up_modulate_b, block3_conv_up_w, block3_conv_up_b, kernel, block3_noise1, block3_conv_up_output_a,
               block3_conv_up_style_a, block3_conv_up_weight_a, block3_conv_up_demod_a,  block3_trans_col_buffer, block3_trans_weight_transposed, block3_conv_up_conv_a, block3_conv_up_upsample_a, block3_conv_up_conv2_a, true, 0, streams);
    StyledConv(block3_conv_up_output_a, mlp7_a, block3_conv_modulate_w, block3_conv_modulate_b, block3_conv_w, block3_conv_b, kernel, block3_noise2, block3_conv_output_a,
               block3_conv_style_a, block3_conv_weight_a, block3_conv_demod_a, block3_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block3_conv_output_a, block2_to_rgb_output_a, mlp7_a, block3_to_rgb_modulate_w, block3_to_rgb_modulate_b, block3_to_rgb_w, block3_to_rgb_b, kernel, block3_to_rgb_output_a,
          block3_to_rgb_style_a, block3_to_rgb_weight_a, nullptr, block3_to_rgb_skip_upsample_a, block3_to_rgb_skip_conv_a, block3_skip_a, streams);

    // Block 4: 64x64 -> 128x128
    StyledConv(block3_conv_output_a, mlp7_a, block4_conv_up_modulate_w, block4_conv_up_modulate_b, block4_conv_up_w, block4_conv_up_b, kernel, block4_noise1, block4_conv_up_output_a,
               block4_conv_up_style_a, block4_conv_up_weight_a, block4_conv_up_demod_a, block4_trans_col_buffer, block4_trans_weight_transposed, block4_conv_up_conv_a, block4_conv_up_upsample_a, block4_conv_up_conv2_a, true, 0, streams);
    StyledConv(block4_conv_up_output_a, mlp7_a, block4_conv_modulate_w, block4_conv_modulate_b, block4_conv_w, block4_conv_b, kernel, block4_noise2, block4_conv_output_a,
               block4_conv_style_a, block4_conv_weight_a, block4_conv_demod_a, block4_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block4_conv_output_a, block3_to_rgb_output_a, mlp7_a, block4_to_rgb_modulate_w, block4_to_rgb_modulate_b, block4_to_rgb_w, block4_to_rgb_b, kernel, block4_to_rgb_output_a,
          block4_to_rgb_style_a, block4_to_rgb_weight_a, nullptr, block4_to_rgb_skip_upsample_a, block4_to_rgb_skip_conv_a, block4_skip_a, streams);

    // Block 5: 128x128 -> 256x256
    StyledConv(block4_conv_output_a, mlp7_a, block5_conv_up_modulate_w, block5_conv_up_modulate_b, block5_conv_up_w, block5_conv_up_b, kernel, block5_noise1, block5_conv_up_output_a,
               block5_conv_up_style_a, block5_conv_up_weight_a, block5_conv_up_demod_a, block5_trans_col_buffer, block5_trans_weight_transposed, block5_conv_up_conv_a, block5_conv_up_upsample_a, block5_conv_up_conv2_a, true, 0, streams);
    StyledConv(block5_conv_up_output_a, mlp7_a, block5_conv_modulate_w, block5_conv_modulate_b, block5_conv_w, block5_conv_b, kernel, block5_noise2, block5_conv_output_a,
               block5_conv_style_a, block5_conv_weight_a, block5_conv_demod_a, block5_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block5_conv_output_a, block4_to_rgb_output_a, mlp7_a, block5_to_rgb_modulate_w, block5_to_rgb_modulate_b, block5_to_rgb_w, block5_to_rgb_b, kernel, block5_to_rgb_output_a,
          block5_to_rgb_style_a, block5_to_rgb_weight_a, nullptr, block5_to_rgb_skip_upsample_a, block5_to_rgb_skip_conv_a, block5_skip_a, streams);

    // Block 6: 256x256 -> 512x512
    StyledConv(block5_conv_output_a, mlp7_a, block6_conv_up_modulate_w, block6_conv_up_modulate_b, block6_conv_up_w, block6_conv_up_b, kernel, block6_noise1, block6_conv_up_output_a,
               block6_conv_up_style_a, block6_conv_up_weight_a, block6_conv_up_demod_a, block6_trans_col_buffer, block6_trans_weight_transposed, block6_conv_up_conv_a, block6_conv_up_upsample_a, block6_conv_up_conv2_a, true, 0, streams);
    StyledConv(block6_conv_up_output_a, mlp7_a, block6_conv_modulate_w, block6_conv_modulate_b, block6_conv_w, block6_conv_b, kernel, block6_noise2, block6_conv_output_a,
               block6_conv_style_a, block6_conv_weight_a, block6_conv_demod_a, block6_conv_col_buffer, nullptr, nullptr, nullptr, nullptr, false, 1, streams);
    ToRGB(block6_conv_output_a, block5_to_rgb_output_a, mlp7_a, block6_to_rgb_modulate_w, block6_to_rgb_modulate_b, block6_to_rgb_w, block6_to_rgb_b, kernel, block6_to_rgb_output_a,
          block6_to_rgb_style_a, block6_to_rgb_weight_a, nullptr, block6_to_rgb_skip_upsample_a, block6_to_rgb_skip_conv_a, block6_skip_a, streams);

    block6_to_rgb_output_a->from_device(streams);
    /*
    for (int i = 0; i < NUM_GPUS; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    */

    /* Copy the final 512x512 RGB image to the local output buffer */
    memcpy(local_outputs + BATCH_SIZE * n * 3 * 512 * 512, block6_to_rgb_output_a->buf, BATCH_SIZE * 3 * 512 * 512 * sizeof(float));
    
    // Clean up the Tensor created for this iteration's input
    delete input;
  }

  // --- Synchronization Barrier ---
  MPI_Barrier(MPI_COMM_WORLD);

  // --- Result Gathering ---
  MPI_Gather(local_outputs, local_output_size, MPI_FLOAT,
             outputs, local_output_size, MPI_FLOAT, 0,
             MPI_COMM_WORLD);

  // --- Cleanup ---
  // CHECK_CUDA(cudaFreeHost(local_inputs));
  delete[] local_inputs;
  delete[] local_outputs;
}
