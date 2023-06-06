#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#include "device_utils.h"

void __device__ dump_nibble(char* dest, uint8_t value) {
	char map[] = "0123456789abcdef";
	dest[0] = map[value];
}

void __device__ hexdump(uint64_t id, const uint8_t* buffer, size_t buffer_size) {
	if (buffer_size > 128) {
		printf("buffer to long: %lld\n", buffer_size);
		return;
	}

	char dest[128*3 + 2];
	for (size_t i = 0; i < buffer_size; ++i) {
		dump_nibble(dest + i*3 + 0, (buffer[i] >> 4) & 0xF);
		dump_nibble(dest + i*3 + 1, (buffer[i] ) & 0xF);
		dest[i*3 + 2] = ' ';
	}
	dest[buffer_size*3] = '\n';
	dest[buffer_size*3 + 1] = 0;
	printf("%lld: %s", id, dest);
}