#pragma once

void __global__ sha512_kernel(const uint8_t* rnd, uint8_t* output_buffer, size_t limit);