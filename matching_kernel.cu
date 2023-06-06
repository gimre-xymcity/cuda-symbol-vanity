#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "vg_constants.h"

#include "device_utils.h"
#include "gpu_errors.h"

void __global__ matching_kernel(uint64_t counter, const uint8_t *input_buffer, uint8_t *output_buffer, const uint8_t* pattern, size_t patternSize)
{
	int input_buffer_id = threadIdx.x + (blockIdx.x * blockDim.x);

	const uint8_t* input = input_buffer + input_buffer_id * RIPEMD_ALIGNED_SIZE;
	uint8_t* output = output_buffer + blockIdx.x * 32;

	__shared__ uint8_t output_data[32];

// #define CMP(x) !!(input[x] == pattern[x])

// 	switch (patternSize) {
// 		case 1:
// 			output_data[threadIdx.x] = CMP(0);
// 			break;
// 		case 2:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1);
// 			break;
// 		case 3:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2);
// 			break;
// 		case 4:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3);
// 			break;
// 		case 5:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4);
// 			break;
// 		case 6:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5);
// 			break;
// 		case 7:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6);
// 			break;
// 		case 8:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7);
// 			break;
// 		case 9:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8);
// 			break;
// 		case 10:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9);
// 			break;
// 		case 11:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10);
// 			break;
// 		case 12:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11);
// 			break;
// 		case 13:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12);
// 			break;
// 		case 14:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13);
// 			break;
// 		case 15:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13) + CMP(14);
// 			break;
// 		case 16:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13) + CMP(14) + CMP(15);
// 			break;
// 		case 17:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13) + CMP(14) + CMP(15) + CMP(16);
// 			break;
// 		case 18:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13) + CMP(14) + CMP(15) + CMP(16) + CMP(17);
// 			break;
// 		case 19:
// 			output_data[threadIdx.x] = CMP(0) + CMP(1) + CMP(2) + CMP(3) + CMP(4) + CMP(5) + CMP(6) + CMP(7) + CMP(8) + CMP(9) + CMP(10) + CMP(11) + CMP(12) + CMP(13) + CMP(14) + CMP(15) + CMP(16) + CMP(17) + CMP(15);
// 			break;
// 	}

// #undef CMP

	int count = 0;
	for (; count < 32; ++count) {
		if (input[count] != pattern[count])
			break;
	}
	output_data[threadIdx.x] = count;

	// todo: that's likely not nice...
	for (int i = 0; i < 32; i++) {
		output[i] = output_data[i];
	}
}
