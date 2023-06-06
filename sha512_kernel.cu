#include <cuda.h>
#include <cuda_runtime.h>

#include "vg_constants.h"

// ugly, but don't care
#include "sha512.cu"

// optimized for a single block of 32b
void __global__ sha512_kernel(const uint8_t* rnd, uint8_t* output_buffer, size_t limit) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	if (id > limit)
		return;

	sha512_context md;

	// unrolled sha512_init + sha512_update

	md.state[0] = UINT64_C(0x6a09e667f3bcc908);
    md.state[1] = UINT64_C(0xbb67ae8584caa73b);
    md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
    md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
    md.state[4] = UINT64_C(0x510e527fade682d1);
    md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
    md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
    md.state[7] = UINT64_C(0x5be0cd19137e2179);

	// random memory is already generated via chacha

	for (size_t i = 0; i < 32; i++) {
		md.buf[i] = rnd[id * CHACHA_STATE_SIZE + i];
	}
	md.curlen = 32;

	// unrolled sha512_finalize

	md.length = md.curlen * UINT64_C(8);
	md.buf[32] = (unsigned char)0x80;
	md.buf[33] = (unsigned char)0;
	md.buf[34] = (unsigned char)0;
	md.buf[35] = (unsigned char)0;
	md.buf[36] = (unsigned char)0;
	md.buf[37] = (unsigned char)0;
	md.buf[38] = (unsigned char)0;
	md.buf[39] = (unsigned char)0;

	md.curlen += 8;

	for (size_t i = 40; i < 120; ++i) {
		md.buf[i] = (unsigned char)0;
	}
	md.curlen += (120 - 40);

	STORE64H(md.length, md.buf+120);

	sha512_compress(&md, md.buf);

	uint8_t* privateKey = output_buffer + id * SHA512_ALIGNED_SIZE;
	for (int i = 0; i < 8; i++) {
		STORE64H(md.state[i], privateKey + (8*i));
	}

	// moved here to have const input in scalarmult_kernel
	// clamp
	privateKey[0]  &= 248;
	privateKey[31] &= 63;
	privateKey[31] |= 64;
}
