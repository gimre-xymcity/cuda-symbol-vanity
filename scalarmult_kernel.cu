#include <cuda.h>
#include <cuda_runtime.h>

#include "vg_constants.h"
#include "ed25519_ge.h"

void __global__ scalarmult_kernel(const uint8_t* private_input, uint8_t* output_buffer, size_t limit) {
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	if (id > limit)
		return;

	// Keypair creation code
	// | ge_p3 A;
    // |
    // | sha512(seed, 32, private_key);
    // | private_key[0] &= 248;
    // | private_key[31] &= 63;
    // | private_key[31] |= 64;
    // |
    // | ge_scalarmult_base(&A, private_key);
    // | ge_p3_tobytes(public_key, &A);
	//
	// privateKey below is already after sha512 pass

	const uint8_t* privateKey = private_input + id * SHA512_ALIGNED_SIZE;

	// basepoint scalar multiplication, to get public key
	ge_p3 A;
	uint8_t* publicKey = output_buffer + id * PUBKEY_ALIGNED_SIZE;
	ge_scalarmult_base(&A, privateKey);
	ge_p3_tobytes(publicKey, &A);

	//hexdump(id, privateKey, 64);
	//hexdump(id, publicKey, 32);
}