#pragma once

#include "tensor.h"

void alloc_and_set_parameters(float *param, size_t param_size);
void alloc_activations();
void generate(float *inputs, float *outputs, size_t n_samples);
void free_parameters();
void free_activations();