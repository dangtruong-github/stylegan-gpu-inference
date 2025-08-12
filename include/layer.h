#pragma once

#include <omp.h>
#include <immintrin.h> // Required for AVX, FMA intrinsics
#include "tensor.h"

/* Layers (Operations) */
void PixelNorm(Tensor *inout);

void UpsamplePad(Tensor *input, Tensor *output, int up, int pad0, int pad1);

void Conv2d(Tensor *input, Tensor *weight, Tensor *output);
void Conv2d_same(Tensor *input, Tensor *weight, Tensor *output,
            int stride, int pad, int dilation);

void Conv2d_im2col(Tensor *input, Tensor *weight, Tensor *output, Tensor *col_buffer);

void ConvTranspose2d_col2im(Tensor *input, Tensor *weight, Tensor *output,
                            Tensor *col_buffer);

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul);

void LeakyReLU(Tensor *inout);

void FusedLinearLeakyReLU(Tensor *in, Tensor *w, Tensor *b, Tensor *out, float lr_mul);

void upfir2d(Tensor *input, Tensor *kernel, Tensor *output,
               Tensor *upsample_a, Tensor *conv_a,
               size_t up, size_t pad0, size_t pad1);

void ModulatedConv2d(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *kernel, Tensor *output,
                     Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *col_buffer, Tensor *weight_transposed, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a,
                     bool demodulate, bool upsample, size_t padding, size_t up, cudaStream_t *streams);
void addBias(Tensor *inout, Tensor *bias);

void elemAdd(Tensor *inout, Tensor *addend);

void StyledConv(Tensor *input, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *noise, Tensor *output,
                Tensor *style_a, Tensor *weight_a, Tensor *demod_a, Tensor *col_buffer, Tensor *weight_transposed, Tensor *conv_a, Tensor *upsample_a, Tensor *conv2_a, bool upsample, size_t padding, cudaStream_t *streams);

void ToRGB(Tensor *input, Tensor *skip, Tensor *style, Tensor *modulate_weight, Tensor *modulate_bias, Tensor *conv_weight, Tensor *conv_bias, Tensor *kernel, Tensor *output,
           Tensor *style_a, Tensor *weight_a, Tensor *col_buffer, Tensor *skip_upsample_a, Tensor *skip_conv_a, Tensor *skip_a);