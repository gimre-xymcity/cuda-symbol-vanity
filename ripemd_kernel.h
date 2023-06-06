#pragma once

void __global__ ripemd_kernel(const uint8_t* input_buffer, uint8_t* output_buffer, size_t limit);
