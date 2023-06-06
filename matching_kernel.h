#pragma once

void __global__ matching_kernel(uint64_t counter, const uint8_t *input_buffer, uint8_t *output_buffer, const uint8_t* pattern, size_t patternSize);