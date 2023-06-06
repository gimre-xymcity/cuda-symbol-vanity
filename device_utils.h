#pragma once

#include <stdint.h>

void __device__ dump_nibble(char* dest, uint8_t value);

void __device__ hexdump(uint64_t id, const uint8_t* buffer, size_t buffer_size);