// gimre@symbol.dev

// code is designed only to handle 32b input buffer

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "vg_constants.h"

#include "device_utils.h"
#include "gpu_errors.h"


namespace {
	__constant__ uint8_t rhopi_shuffle[25][2] = {
		{0, 0}, {6, 44}, {12, 43}, {18, 21}, {24, 14}, {3, 28}, {9, 20}, {10, 3}, {16, 45}, {22, 61}, {1, 1}, {7, 6}, {13, 25}, {19, 8}, {20, 18}, {4, 27}, {5, 36}, {11, 10}, {17, 15}, {23, 56}, {2, 62}, {8, 55}, {14, 39}, {15, 41}, {21, 2}
	};
	__constant__ uint8_t chi[25][2] = {
		{1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 1}, {6, 7}, {7, 8}, {8, 9}, {9, 5}, {5, 6}, {11, 12}, {12, 13}, {13, 14}, {14, 10}, {10, 11}, {16, 17}, {17, 18}, {18, 19}, {19, 15}, {15, 16}, {21, 22}, {22, 23}, {23, 24}, {24, 20}, {20, 21}
	};

	__constant__ uint64_t iota[24] = {
		0x0000000000000001L, 0x0000000000008082L, 0x800000000000808aL, 0x8000000080008000L, 0x000000000000808bL,
		0x0000000080000001L, 0x8000000080008081L, 0x8000000000008009L, 0x000000000000008aL, 0x0000000000000088L,
		0x0000000080008009L, 0x000000008000000aL, 0x000000008000808bL, 0x800000000000008bL, 0x8000000000008089L,
		0x8000000000008003L, 0x8000000000008002L, 0x8000000000000080L, 0x000000000000800aL, 0x800000008000000aL,
		0x8000000080008081L, 0x8000000000008080L, 0x0000000080000001L, 0x8000000080008008L
	};
}

// 200 - 2*32
#define BUF_SIZE 136

__device__ uint64_t rotate(uint64_t val, unsigned n) { return val << n | val >> (64 - n); }

__device__ void stateRounds(uint64_t *A)
{
  const size_t t = threadIdx.x;
  const size_t s = threadIdx.x % 5;

  __shared__ uint64_t C[25];

#pragma unroll
  for (int roundId = 0; roundId < 24; ++roundId) {
    // theta
    C[t] = A[s] ^ A[s + 5] ^ A[s + 10] ^ A[s + 15] ^ A[s + 20];
    A[t] ^= C[s + 5 - 1] ^ rotate(C[s + 1], 1);

    // rho and pi
    C[t] = rotate(A[rhopi_shuffle[t][0]], rhopi_shuffle[t][1]);

    // chi
    A[t] = C[t] ^ (~C[chi[t][0]] & C[chi[t][1]]);

    // iota
    A[t] ^= t == 0 ? iota[roundId] : 0;
  }
}

void __global__ sha3_256_kernel(const uint8_t* input_buffer, uint8_t* output_buffer, size_t limit) {
	int buffer_id = blockIdx.x;
	if (buffer_id > limit)
		return;

	// 32 byte pub key
	const uint8_t* input = input_buffer + buffer_id * PUBKEY_ALIGNED_SIZE;

	// init + padding
	// bytes (32, BUF_SIZE - 2) are  zeroed already
	uint8_t block[BUF_SIZE] = { 0 };
	for (size_t i = 0; i < 32; ++i) {
		block[i] = input[i];
	}
	block[32] = 0x06;
	block[BUF_SIZE - 1] = 0x80;

	const auto* ptr64 = reinterpret_cast<const uint64_t *>(block);
	__shared__ uint64_t state[25];
	__shared__ uint64_t A[25];

	// split into threads
	const size_t t = threadIdx.x;
	state[t] = 0;

	if (t < 25) {
		A[t] = state[t];
		if (t < (BUF_SIZE / 8)) {
			A[t] ^= ptr64[t];
		}

		stateRounds(A);
		state[t] = A[t];
	}

	const uint8_t* state_ptr = reinterpret_cast<const uint8_t *>(state);
	uint8_t* output = output_buffer + buffer_id * SHA3_256_ALIGNED_SIZE;

	// todo: that's likely not nice...
	for (int i = 0; i < 32; i++) {
		output[i] = state_ptr[i];
	}
}
