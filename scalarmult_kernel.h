#pragma once

void __global__ scalarmult_kernel(const uint8_t* private_input, uint8_t* output_buffer, size_t limit);
