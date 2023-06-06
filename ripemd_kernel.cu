#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "vg_constants.h"

#include "device_utils.h"
#include "gpu_errors.h"

struct ripemd160_ctx
{
	uint32_t h[5];

	uint32_t w0[4];
	uint32_t w1[4];
	uint32_t w2[4];
	uint32_t w3[4];

	int len;
};

enum ripemd160_constants
{
	RIPEMD160M_A = 0x67452301U,
	RIPEMD160M_B = 0xefcdab89U,
	RIPEMD160M_C = 0x98badcfeU,
	RIPEMD160M_D = 0x10325476U,
	RIPEMD160M_E = 0xc3d2e1f0U,

	RIPEMD160C00 = 0x00000000U,
	RIPEMD160C10 = 0x5a827999U,
	RIPEMD160C20 = 0x6ed9eba1U,
	RIPEMD160C30 = 0x8f1bbcdcU,
	RIPEMD160C40 = 0xa953fd4eU,
	RIPEMD160C50 = 0x50a28be6U,
	RIPEMD160C60 = 0x5c4dd124U,
	RIPEMD160C70 = 0x6d703ef3U,
	RIPEMD160C80 = 0x7a6d76e9U,
	RIPEMD160C90 = 0x00000000U,

	RIPEMD160S00 = 11u,
	RIPEMD160S01 = 14u,
	RIPEMD160S02 = 15u,
	RIPEMD160S03 = 12u,
	RIPEMD160S04 = 5u,
	RIPEMD160S05 = 8u,
	RIPEMD160S06 = 7u,
	RIPEMD160S07 = 9u,
	RIPEMD160S08 = 11u,
	RIPEMD160S09 = 13u,
	RIPEMD160S0A = 14u,
	RIPEMD160S0B = 15u,
	RIPEMD160S0C = 6u,
	RIPEMD160S0D = 7u,
	RIPEMD160S0E = 9u,
	RIPEMD160S0F = 8u,

	RIPEMD160S10 = 7u,
	RIPEMD160S11 = 6u,
	RIPEMD160S12 = 8u,
	RIPEMD160S13 = 13u,
	RIPEMD160S14 = 11u,
	RIPEMD160S15 = 9u,
	RIPEMD160S16 = 7u,
	RIPEMD160S17 = 15u,
	RIPEMD160S18 = 7u,
	RIPEMD160S19 = 12u,
	RIPEMD160S1A = 15u,
	RIPEMD160S1B = 9u,
	RIPEMD160S1C = 11u,
	RIPEMD160S1D = 7u,
	RIPEMD160S1E = 13u,
	RIPEMD160S1F = 12u,

	RIPEMD160S20 = 11u,
	RIPEMD160S21 = 13u,
	RIPEMD160S22 = 6u,
	RIPEMD160S23 = 7u,
	RIPEMD160S24 = 14u,
	RIPEMD160S25 = 9u,
	RIPEMD160S26 = 13u,
	RIPEMD160S27 = 15u,
	RIPEMD160S28 = 14u,
	RIPEMD160S29 = 8u,
	RIPEMD160S2A = 13u,
	RIPEMD160S2B = 6u,
	RIPEMD160S2C = 5u,
	RIPEMD160S2D = 12u,
	RIPEMD160S2E = 7u,
	RIPEMD160S2F = 5u,

	RIPEMD160S30 = 11u,
	RIPEMD160S31 = 12u,
	RIPEMD160S32 = 14u,
	RIPEMD160S33 = 15u,
	RIPEMD160S34 = 14u,
	RIPEMD160S35 = 15u,
	RIPEMD160S36 = 9u,
	RIPEMD160S37 = 8u,
	RIPEMD160S38 = 9u,
	RIPEMD160S39 = 14u,
	RIPEMD160S3A = 5u,
	RIPEMD160S3B = 6u,
	RIPEMD160S3C = 8u,
	RIPEMD160S3D = 6u,
	RIPEMD160S3E = 5u,
	RIPEMD160S3F = 12u,

	RIPEMD160S40 = 9u,
	RIPEMD160S41 = 15u,
	RIPEMD160S42 = 5u,
	RIPEMD160S43 = 11u,
	RIPEMD160S44 = 6u,
	RIPEMD160S45 = 8u,
	RIPEMD160S46 = 13u,
	RIPEMD160S47 = 12u,
	RIPEMD160S48 = 5u,
	RIPEMD160S49 = 12u,
	RIPEMD160S4A = 13u,
	RIPEMD160S4B = 14u,
	RIPEMD160S4C = 11u,
	RIPEMD160S4D = 8u,
	RIPEMD160S4E = 5u,
	RIPEMD160S4F = 6u,

	RIPEMD160S50 = 8u,
	RIPEMD160S51 = 9u,
	RIPEMD160S52 = 9u,
	RIPEMD160S53 = 11u,
	RIPEMD160S54 = 13u,
	RIPEMD160S55 = 15u,
	RIPEMD160S56 = 15u,
	RIPEMD160S57 = 5u,
	RIPEMD160S58 = 7u,
	RIPEMD160S59 = 7u,
	RIPEMD160S5A = 8u,
	RIPEMD160S5B = 11u,
	RIPEMD160S5C = 14u,
	RIPEMD160S5D = 14u,
	RIPEMD160S5E = 12u,
	RIPEMD160S5F = 6u,

	RIPEMD160S60 = 9u,
	RIPEMD160S61 = 13u,
	RIPEMD160S62 = 15u,
	RIPEMD160S63 = 7u,
	RIPEMD160S64 = 12u,
	RIPEMD160S65 = 8u,
	RIPEMD160S66 = 9u,
	RIPEMD160S67 = 11u,
	RIPEMD160S68 = 7u,
	RIPEMD160S69 = 7u,
	RIPEMD160S6A = 12u,
	RIPEMD160S6B = 7u,
	RIPEMD160S6C = 6u,
	RIPEMD160S6D = 15u,
	RIPEMD160S6E = 13u,
	RIPEMD160S6F = 11u,

	RIPEMD160S70 = 9u,
	RIPEMD160S71 = 7u,
	RIPEMD160S72 = 15u,
	RIPEMD160S73 = 11u,
	RIPEMD160S74 = 8u,
	RIPEMD160S75 = 6u,
	RIPEMD160S76 = 6u,
	RIPEMD160S77 = 14u,
	RIPEMD160S78 = 12u,
	RIPEMD160S79 = 13u,
	RIPEMD160S7A = 5u,
	RIPEMD160S7B = 14u,
	RIPEMD160S7C = 13u,
	RIPEMD160S7D = 13u,
	RIPEMD160S7E = 7u,
	RIPEMD160S7F = 5u,

	RIPEMD160S80 = 15u,
	RIPEMD160S81 = 5u,
	RIPEMD160S82 = 8u,
	RIPEMD160S83 = 11u,
	RIPEMD160S84 = 14u,
	RIPEMD160S85 = 14u,
	RIPEMD160S86 = 6u,
	RIPEMD160S87 = 14u,
	RIPEMD160S88 = 6u,
	RIPEMD160S89 = 9u,
	RIPEMD160S8A = 12u,
	RIPEMD160S8B = 9u,
	RIPEMD160S8C = 12u,
	RIPEMD160S8D = 5u,
	RIPEMD160S8E = 15u,
	RIPEMD160S8F = 8u,

	RIPEMD160S90 = 8u,
	RIPEMD160S91 = 5u,
	RIPEMD160S92 = 12u,
	RIPEMD160S93 = 9u,
	RIPEMD160S94 = 12u,
	RIPEMD160S95 = 5u,
	RIPEMD160S96 = 14u,
	RIPEMD160S97 = 6u,
	RIPEMD160S98 = 8u,
	RIPEMD160S99 = 13u,
	RIPEMD160S9A = 6u,
	RIPEMD160S9B = 5u,
	RIPEMD160S9C = 15u,
	RIPEMD160S9D = 13u,
	RIPEMD160S9E = 11u,
	RIPEMD160S9F = 11u
};

uint32_t __device__ hc_rotl32_S (const uint32_t a, const int n)
{
//  return rotl32_S (a, n);
//   #else
//   #ifdef USE_ROTATE
//   return rotate (a, (u32) (n));
//   #else
   return ((a << n) | (a >> (32 - n)));
//   #endif
//   #endif
}

#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

#define RIPEMD160_F(x,y,z)    ((x) ^ (y) ^ (z))
#define RIPEMD160_G(x,y,z)    ((z) ^ ((x) & ((y) ^ (z)))) /* x ? y : z */
#define RIPEMD160_H(x,y,z)    (((x) | ~(y)) ^ (z))
#define RIPEMD160_I(x,y,z)    ((y) ^ ((z) & ((x) ^ (y)))) /* z ? x : y */
#define RIPEMD160_J(x,y,z)    ((x) ^ ((y) | ~(z)))

#define RIPEMD160_Go(x,y,z)   (bitselect ((z), (y), (x)))
#define RIPEMD160_Io(x,y,z)   (bitselect ((y), (x), (z)))


#define RIPEMD160_STEP_S(f, a, b, c, d, e, x, K, s) \
	{                                               \
		a += K;                                     \
		a += x;                                     \
		a += f(b, c, d);                            \
		a = hc_rotl32_S(a, s);                      \
		a += e;                                     \
		c = hc_rotl32_S(c, 10u);                    \
	}

#define ROTATE_LEFT_WORKAROUND_BUG(a,n) ((a << n) | (a >> (32 - n)))

#define RIPEMD160_STEP_S_WORKAROUND_BUG(f, a, b, c, d, e, x, K, s) \
	{                                                              \
		a += K;                                                    \
		a += x;                                                    \
		a += f(b, c, d);                                           \
		a = ROTATE_LEFT_WORKAROUND_BUG(a, s);                      \
		a += e;                                                    \
		c = hc_rotl32_S(c, 10u);                                   \
	}

void __device__ ripemd160_transform(
	const uint32_t *w0,
	const uint32_t *w1,
	const uint32_t *w2,
	const uint32_t *w3,
	uint32_t *digest)
{
	uint32_t a1 = digest[0];
	uint32_t b1 = digest[1];
	uint32_t c1 = digest[2];
	uint32_t d1 = digest[3];
	uint32_t e1 = digest[4];

	RIPEMD160_STEP_S(RIPEMD160_F, a1, b1, c1, d1, e1, w0[0], RIPEMD160C00, RIPEMD160S00);
	RIPEMD160_STEP_S(RIPEMD160_F, e1, a1, b1, c1, d1, w0[1], RIPEMD160C00, RIPEMD160S01);
	RIPEMD160_STEP_S(RIPEMD160_F, d1, e1, a1, b1, c1, w0[2], RIPEMD160C00, RIPEMD160S02);
	RIPEMD160_STEP_S(RIPEMD160_F, c1, d1, e1, a1, b1, w0[3], RIPEMD160C00, RIPEMD160S03);
	RIPEMD160_STEP_S(RIPEMD160_F, b1, c1, d1, e1, a1, w1[0], RIPEMD160C00, RIPEMD160S04);
	RIPEMD160_STEP_S(RIPEMD160_F, a1, b1, c1, d1, e1, w1[1], RIPEMD160C00, RIPEMD160S05);
	RIPEMD160_STEP_S(RIPEMD160_F, e1, a1, b1, c1, d1, w1[2], RIPEMD160C00, RIPEMD160S06);
	RIPEMD160_STEP_S(RIPEMD160_F, d1, e1, a1, b1, c1, w1[3], RIPEMD160C00, RIPEMD160S07);
	RIPEMD160_STEP_S(RIPEMD160_F, c1, d1, e1, a1, b1, w2[0], RIPEMD160C00, RIPEMD160S08);
	RIPEMD160_STEP_S(RIPEMD160_F, b1, c1, d1, e1, a1, w2[1], RIPEMD160C00, RIPEMD160S09);
	RIPEMD160_STEP_S(RIPEMD160_F, a1, b1, c1, d1, e1, w2[2], RIPEMD160C00, RIPEMD160S0A);
	RIPEMD160_STEP_S(RIPEMD160_F, e1, a1, b1, c1, d1, w2[3], RIPEMD160C00, RIPEMD160S0B);
	RIPEMD160_STEP_S(RIPEMD160_F, d1, e1, a1, b1, c1, w3[0], RIPEMD160C00, RIPEMD160S0C);
	RIPEMD160_STEP_S(RIPEMD160_F, c1, d1, e1, a1, b1, w3[1], RIPEMD160C00, RIPEMD160S0D);
	RIPEMD160_STEP_S(RIPEMD160_F, b1, c1, d1, e1, a1, w3[2], RIPEMD160C00, RIPEMD160S0E);
	RIPEMD160_STEP_S(RIPEMD160_F, a1, b1, c1, d1, e1, w3[3], RIPEMD160C00, RIPEMD160S0F);

	RIPEMD160_STEP_S(RIPEMD160_Go, e1, a1, b1, c1, d1, w1[3], RIPEMD160C10, RIPEMD160S10);
	RIPEMD160_STEP_S(RIPEMD160_Go, d1, e1, a1, b1, c1, w1[0], RIPEMD160C10, RIPEMD160S11);
	RIPEMD160_STEP_S(RIPEMD160_Go, c1, d1, e1, a1, b1, w3[1], RIPEMD160C10, RIPEMD160S12);
	RIPEMD160_STEP_S(RIPEMD160_Go, b1, c1, d1, e1, a1, w0[1], RIPEMD160C10, RIPEMD160S13);
	RIPEMD160_STEP_S(RIPEMD160_Go, a1, b1, c1, d1, e1, w2[2], RIPEMD160C10, RIPEMD160S14);
	RIPEMD160_STEP_S(RIPEMD160_Go, e1, a1, b1, c1, d1, w1[2], RIPEMD160C10, RIPEMD160S15);
	RIPEMD160_STEP_S(RIPEMD160_Go, d1, e1, a1, b1, c1, w3[3], RIPEMD160C10, RIPEMD160S16);
	RIPEMD160_STEP_S(RIPEMD160_Go, c1, d1, e1, a1, b1, w0[3], RIPEMD160C10, RIPEMD160S17);
	RIPEMD160_STEP_S(RIPEMD160_Go, b1, c1, d1, e1, a1, w3[0], RIPEMD160C10, RIPEMD160S18);
	RIPEMD160_STEP_S(RIPEMD160_Go, a1, b1, c1, d1, e1, w0[0], RIPEMD160C10, RIPEMD160S19);
	RIPEMD160_STEP_S(RIPEMD160_Go, e1, a1, b1, c1, d1, w2[1], RIPEMD160C10, RIPEMD160S1A);
	RIPEMD160_STEP_S(RIPEMD160_Go, d1, e1, a1, b1, c1, w1[1], RIPEMD160C10, RIPEMD160S1B);
	RIPEMD160_STEP_S(RIPEMD160_Go, c1, d1, e1, a1, b1, w0[2], RIPEMD160C10, RIPEMD160S1C);
	RIPEMD160_STEP_S(RIPEMD160_Go, b1, c1, d1, e1, a1, w3[2], RIPEMD160C10, RIPEMD160S1D);
	RIPEMD160_STEP_S(RIPEMD160_Go, a1, b1, c1, d1, e1, w2[3], RIPEMD160C10, RIPEMD160S1E);
	RIPEMD160_STEP_S(RIPEMD160_Go, e1, a1, b1, c1, d1, w2[0], RIPEMD160C10, RIPEMD160S1F);

	RIPEMD160_STEP_S(RIPEMD160_H, d1, e1, a1, b1, c1, w0[3], RIPEMD160C20, RIPEMD160S20);
	RIPEMD160_STEP_S(RIPEMD160_H, c1, d1, e1, a1, b1, w2[2], RIPEMD160C20, RIPEMD160S21);
	RIPEMD160_STEP_S(RIPEMD160_H, b1, c1, d1, e1, a1, w3[2], RIPEMD160C20, RIPEMD160S22);
	RIPEMD160_STEP_S(RIPEMD160_H, a1, b1, c1, d1, e1, w1[0], RIPEMD160C20, RIPEMD160S23);
	RIPEMD160_STEP_S(RIPEMD160_H, e1, a1, b1, c1, d1, w2[1], RIPEMD160C20, RIPEMD160S24);
	RIPEMD160_STEP_S(RIPEMD160_H, d1, e1, a1, b1, c1, w3[3], RIPEMD160C20, RIPEMD160S25);
	RIPEMD160_STEP_S(RIPEMD160_H, c1, d1, e1, a1, b1, w2[0], RIPEMD160C20, RIPEMD160S26);
	RIPEMD160_STEP_S(RIPEMD160_H, b1, c1, d1, e1, a1, w0[1], RIPEMD160C20, RIPEMD160S27);
	RIPEMD160_STEP_S(RIPEMD160_H, a1, b1, c1, d1, e1, w0[2], RIPEMD160C20, RIPEMD160S28);
	RIPEMD160_STEP_S(RIPEMD160_H, e1, a1, b1, c1, d1, w1[3], RIPEMD160C20, RIPEMD160S29);
	RIPEMD160_STEP_S(RIPEMD160_H, d1, e1, a1, b1, c1, w0[0], RIPEMD160C20, RIPEMD160S2A);
	RIPEMD160_STEP_S(RIPEMD160_H, c1, d1, e1, a1, b1, w1[2], RIPEMD160C20, RIPEMD160S2B);
	RIPEMD160_STEP_S(RIPEMD160_H, b1, c1, d1, e1, a1, w3[1], RIPEMD160C20, RIPEMD160S2C);
	RIPEMD160_STEP_S(RIPEMD160_H, a1, b1, c1, d1, e1, w2[3], RIPEMD160C20, RIPEMD160S2D);
	RIPEMD160_STEP_S(RIPEMD160_H, e1, a1, b1, c1, d1, w1[1], RIPEMD160C20, RIPEMD160S2E);
	RIPEMD160_STEP_S(RIPEMD160_H, d1, e1, a1, b1, c1, w3[0], RIPEMD160C20, RIPEMD160S2F);

	RIPEMD160_STEP_S(RIPEMD160_Io, c1, d1, e1, a1, b1, w0[1], RIPEMD160C30, RIPEMD160S30);
	RIPEMD160_STEP_S(RIPEMD160_Io, b1, c1, d1, e1, a1, w2[1], RIPEMD160C30, RIPEMD160S31);
	RIPEMD160_STEP_S(RIPEMD160_Io, a1, b1, c1, d1, e1, w2[3], RIPEMD160C30, RIPEMD160S32);
	RIPEMD160_STEP_S(RIPEMD160_Io, e1, a1, b1, c1, d1, w2[2], RIPEMD160C30, RIPEMD160S33);
	RIPEMD160_STEP_S(RIPEMD160_Io, d1, e1, a1, b1, c1, w0[0], RIPEMD160C30, RIPEMD160S34);
	RIPEMD160_STEP_S(RIPEMD160_Io, c1, d1, e1, a1, b1, w2[0], RIPEMD160C30, RIPEMD160S35);
	RIPEMD160_STEP_S(RIPEMD160_Io, b1, c1, d1, e1, a1, w3[0], RIPEMD160C30, RIPEMD160S36);
	RIPEMD160_STEP_S(RIPEMD160_Io, a1, b1, c1, d1, e1, w1[0], RIPEMD160C30, RIPEMD160S37);
	RIPEMD160_STEP_S(RIPEMD160_Io, e1, a1, b1, c1, d1, w3[1], RIPEMD160C30, RIPEMD160S38);
	RIPEMD160_STEP_S(RIPEMD160_Io, d1, e1, a1, b1, c1, w0[3], RIPEMD160C30, RIPEMD160S39);
	RIPEMD160_STEP_S(RIPEMD160_Io, c1, d1, e1, a1, b1, w1[3], RIPEMD160C30, RIPEMD160S3A);
	RIPEMD160_STEP_S(RIPEMD160_Io, b1, c1, d1, e1, a1, w3[3], RIPEMD160C30, RIPEMD160S3B);
	RIPEMD160_STEP_S(RIPEMD160_Io, a1, b1, c1, d1, e1, w3[2], RIPEMD160C30, RIPEMD160S3C);
	RIPEMD160_STEP_S(RIPEMD160_Io, e1, a1, b1, c1, d1, w1[1], RIPEMD160C30, RIPEMD160S3D);
	RIPEMD160_STEP_S(RIPEMD160_Io, d1, e1, a1, b1, c1, w1[2], RIPEMD160C30, RIPEMD160S3E);
	RIPEMD160_STEP_S(RIPEMD160_Io, c1, d1, e1, a1, b1, w0[2], RIPEMD160C30, RIPEMD160S3F);

	RIPEMD160_STEP_S(RIPEMD160_J, b1, c1, d1, e1, a1, w1[0], RIPEMD160C40, RIPEMD160S40);
	RIPEMD160_STEP_S(RIPEMD160_J, a1, b1, c1, d1, e1, w0[0], RIPEMD160C40, RIPEMD160S41);
	RIPEMD160_STEP_S(RIPEMD160_J, e1, a1, b1, c1, d1, w1[1], RIPEMD160C40, RIPEMD160S42);
	RIPEMD160_STEP_S(RIPEMD160_J, d1, e1, a1, b1, c1, w2[1], RIPEMD160C40, RIPEMD160S43);
	RIPEMD160_STEP_S(RIPEMD160_J, c1, d1, e1, a1, b1, w1[3], RIPEMD160C40, RIPEMD160S44);
	RIPEMD160_STEP_S(RIPEMD160_J, b1, c1, d1, e1, a1, w3[0], RIPEMD160C40, RIPEMD160S45);
	RIPEMD160_STEP_S(RIPEMD160_J, a1, b1, c1, d1, e1, w0[2], RIPEMD160C40, RIPEMD160S46);
	RIPEMD160_STEP_S(RIPEMD160_J, e1, a1, b1, c1, d1, w2[2], RIPEMD160C40, RIPEMD160S47);
	RIPEMD160_STEP_S(RIPEMD160_J, d1, e1, a1, b1, c1, w3[2], RIPEMD160C40, RIPEMD160S48);
	RIPEMD160_STEP_S(RIPEMD160_J, c1, d1, e1, a1, b1, w0[1], RIPEMD160C40, RIPEMD160S49);
	RIPEMD160_STEP_S(RIPEMD160_J, b1, c1, d1, e1, a1, w0[3], RIPEMD160C40, RIPEMD160S4A);
	RIPEMD160_STEP_S(RIPEMD160_J, a1, b1, c1, d1, e1, w2[0], RIPEMD160C40, RIPEMD160S4B);
	RIPEMD160_STEP_S(RIPEMD160_J, e1, a1, b1, c1, d1, w2[3], RIPEMD160C40, RIPEMD160S4C);
	RIPEMD160_STEP_S(RIPEMD160_J, d1, e1, a1, b1, c1, w1[2], RIPEMD160C40, RIPEMD160S4D);
	RIPEMD160_STEP_S(RIPEMD160_J, c1, d1, e1, a1, b1, w3[3], RIPEMD160C40, RIPEMD160S4E);
	RIPEMD160_STEP_S(RIPEMD160_J, b1, c1, d1, e1, a1, w3[1], RIPEMD160C40, RIPEMD160S4F);

	uint32_t a2 = digest[0];
	uint32_t b2 = digest[1];
	uint32_t c2 = digest[2];
	uint32_t d2 = digest[3];
	uint32_t e2 = digest[4];

	RIPEMD160_STEP_S_WORKAROUND_BUG(RIPEMD160_J, a2, b2, c2, d2, e2, w1[1], RIPEMD160C50, RIPEMD160S50);
	RIPEMD160_STEP_S(RIPEMD160_J, e2, a2, b2, c2, d2, w3[2], RIPEMD160C50, RIPEMD160S51);
	RIPEMD160_STEP_S(RIPEMD160_J, d2, e2, a2, b2, c2, w1[3], RIPEMD160C50, RIPEMD160S52);
	RIPEMD160_STEP_S(RIPEMD160_J, c2, d2, e2, a2, b2, w0[0], RIPEMD160C50, RIPEMD160S53);
	RIPEMD160_STEP_S(RIPEMD160_J, b2, c2, d2, e2, a2, w2[1], RIPEMD160C50, RIPEMD160S54);
	RIPEMD160_STEP_S(RIPEMD160_J, a2, b2, c2, d2, e2, w0[2], RIPEMD160C50, RIPEMD160S55);
	RIPEMD160_STEP_S(RIPEMD160_J, e2, a2, b2, c2, d2, w2[3], RIPEMD160C50, RIPEMD160S56);
	RIPEMD160_STEP_S(RIPEMD160_J, d2, e2, a2, b2, c2, w1[0], RIPEMD160C50, RIPEMD160S57);
	RIPEMD160_STEP_S(RIPEMD160_J, c2, d2, e2, a2, b2, w3[1], RIPEMD160C50, RIPEMD160S58);
	RIPEMD160_STEP_S(RIPEMD160_J, b2, c2, d2, e2, a2, w1[2], RIPEMD160C50, RIPEMD160S59);
	RIPEMD160_STEP_S(RIPEMD160_J, a2, b2, c2, d2, e2, w3[3], RIPEMD160C50, RIPEMD160S5A);
	RIPEMD160_STEP_S(RIPEMD160_J, e2, a2, b2, c2, d2, w2[0], RIPEMD160C50, RIPEMD160S5B);
	RIPEMD160_STEP_S(RIPEMD160_J, d2, e2, a2, b2, c2, w0[1], RIPEMD160C50, RIPEMD160S5C);
	RIPEMD160_STEP_S(RIPEMD160_J, c2, d2, e2, a2, b2, w2[2], RIPEMD160C50, RIPEMD160S5D);
	RIPEMD160_STEP_S(RIPEMD160_J, b2, c2, d2, e2, a2, w0[3], RIPEMD160C50, RIPEMD160S5E);
	RIPEMD160_STEP_S(RIPEMD160_J, a2, b2, c2, d2, e2, w3[0], RIPEMD160C50, RIPEMD160S5F);

	RIPEMD160_STEP_S(RIPEMD160_Io, e2, a2, b2, c2, d2, w1[2], RIPEMD160C60, RIPEMD160S60);
	RIPEMD160_STEP_S(RIPEMD160_Io, d2, e2, a2, b2, c2, w2[3], RIPEMD160C60, RIPEMD160S61);
	RIPEMD160_STEP_S(RIPEMD160_Io, c2, d2, e2, a2, b2, w0[3], RIPEMD160C60, RIPEMD160S62);
	RIPEMD160_STEP_S(RIPEMD160_Io, b2, c2, d2, e2, a2, w1[3], RIPEMD160C60, RIPEMD160S63);
	RIPEMD160_STEP_S(RIPEMD160_Io, a2, b2, c2, d2, e2, w0[0], RIPEMD160C60, RIPEMD160S64);
	RIPEMD160_STEP_S(RIPEMD160_Io, e2, a2, b2, c2, d2, w3[1], RIPEMD160C60, RIPEMD160S65);
	RIPEMD160_STEP_S(RIPEMD160_Io, d2, e2, a2, b2, c2, w1[1], RIPEMD160C60, RIPEMD160S66);
	RIPEMD160_STEP_S(RIPEMD160_Io, c2, d2, e2, a2, b2, w2[2], RIPEMD160C60, RIPEMD160S67);
	RIPEMD160_STEP_S(RIPEMD160_Io, b2, c2, d2, e2, a2, w3[2], RIPEMD160C60, RIPEMD160S68);
	RIPEMD160_STEP_S(RIPEMD160_Io, a2, b2, c2, d2, e2, w3[3], RIPEMD160C60, RIPEMD160S69);
	RIPEMD160_STEP_S(RIPEMD160_Io, e2, a2, b2, c2, d2, w2[0], RIPEMD160C60, RIPEMD160S6A);
	RIPEMD160_STEP_S(RIPEMD160_Io, d2, e2, a2, b2, c2, w3[0], RIPEMD160C60, RIPEMD160S6B);
	RIPEMD160_STEP_S(RIPEMD160_Io, c2, d2, e2, a2, b2, w1[0], RIPEMD160C60, RIPEMD160S6C);
	RIPEMD160_STEP_S(RIPEMD160_Io, b2, c2, d2, e2, a2, w2[1], RIPEMD160C60, RIPEMD160S6D);
	RIPEMD160_STEP_S(RIPEMD160_Io, a2, b2, c2, d2, e2, w0[1], RIPEMD160C60, RIPEMD160S6E);
	RIPEMD160_STEP_S(RIPEMD160_Io, e2, a2, b2, c2, d2, w0[2], RIPEMD160C60, RIPEMD160S6F);

	RIPEMD160_STEP_S(RIPEMD160_H, d2, e2, a2, b2, c2, w3[3], RIPEMD160C70, RIPEMD160S70);
	RIPEMD160_STEP_S(RIPEMD160_H, c2, d2, e2, a2, b2, w1[1], RIPEMD160C70, RIPEMD160S71);
	RIPEMD160_STEP_S(RIPEMD160_H, b2, c2, d2, e2, a2, w0[1], RIPEMD160C70, RIPEMD160S72);
	RIPEMD160_STEP_S(RIPEMD160_H, a2, b2, c2, d2, e2, w0[3], RIPEMD160C70, RIPEMD160S73);
	RIPEMD160_STEP_S(RIPEMD160_H, e2, a2, b2, c2, d2, w1[3], RIPEMD160C70, RIPEMD160S74);
	RIPEMD160_STEP_S(RIPEMD160_H, d2, e2, a2, b2, c2, w3[2], RIPEMD160C70, RIPEMD160S75);
	RIPEMD160_STEP_S(RIPEMD160_H, c2, d2, e2, a2, b2, w1[2], RIPEMD160C70, RIPEMD160S76);
	RIPEMD160_STEP_S(RIPEMD160_H, b2, c2, d2, e2, a2, w2[1], RIPEMD160C70, RIPEMD160S77);
	RIPEMD160_STEP_S(RIPEMD160_H, a2, b2, c2, d2, e2, w2[3], RIPEMD160C70, RIPEMD160S78);
	RIPEMD160_STEP_S(RIPEMD160_H, e2, a2, b2, c2, d2, w2[0], RIPEMD160C70, RIPEMD160S79);
	RIPEMD160_STEP_S(RIPEMD160_H, d2, e2, a2, b2, c2, w3[0], RIPEMD160C70, RIPEMD160S7A);
	RIPEMD160_STEP_S(RIPEMD160_H, c2, d2, e2, a2, b2, w0[2], RIPEMD160C70, RIPEMD160S7B);
	RIPEMD160_STEP_S(RIPEMD160_H, b2, c2, d2, e2, a2, w2[2], RIPEMD160C70, RIPEMD160S7C);
	RIPEMD160_STEP_S(RIPEMD160_H, a2, b2, c2, d2, e2, w0[0], RIPEMD160C70, RIPEMD160S7D);
	RIPEMD160_STEP_S(RIPEMD160_H, e2, a2, b2, c2, d2, w1[0], RIPEMD160C70, RIPEMD160S7E);
	RIPEMD160_STEP_S(RIPEMD160_H, d2, e2, a2, b2, c2, w3[1], RIPEMD160C70, RIPEMD160S7F);

	RIPEMD160_STEP_S(RIPEMD160_Go, c2, d2, e2, a2, b2, w2[0], RIPEMD160C80, RIPEMD160S80);
	RIPEMD160_STEP_S(RIPEMD160_Go, b2, c2, d2, e2, a2, w1[2], RIPEMD160C80, RIPEMD160S81);
	RIPEMD160_STEP_S(RIPEMD160_Go, a2, b2, c2, d2, e2, w1[0], RIPEMD160C80, RIPEMD160S82);
	RIPEMD160_STEP_S(RIPEMD160_Go, e2, a2, b2, c2, d2, w0[1], RIPEMD160C80, RIPEMD160S83);
	RIPEMD160_STEP_S(RIPEMD160_Go, d2, e2, a2, b2, c2, w0[3], RIPEMD160C80, RIPEMD160S84);
	RIPEMD160_STEP_S(RIPEMD160_Go, c2, d2, e2, a2, b2, w2[3], RIPEMD160C80, RIPEMD160S85);
	RIPEMD160_STEP_S(RIPEMD160_Go, b2, c2, d2, e2, a2, w3[3], RIPEMD160C80, RIPEMD160S86);
	RIPEMD160_STEP_S(RIPEMD160_Go, a2, b2, c2, d2, e2, w0[0], RIPEMD160C80, RIPEMD160S87);
	RIPEMD160_STEP_S(RIPEMD160_Go, e2, a2, b2, c2, d2, w1[1], RIPEMD160C80, RIPEMD160S88);
	RIPEMD160_STEP_S(RIPEMD160_Go, d2, e2, a2, b2, c2, w3[0], RIPEMD160C80, RIPEMD160S89);
	RIPEMD160_STEP_S(RIPEMD160_Go, c2, d2, e2, a2, b2, w0[2], RIPEMD160C80, RIPEMD160S8A);
	RIPEMD160_STEP_S(RIPEMD160_Go, b2, c2, d2, e2, a2, w3[1], RIPEMD160C80, RIPEMD160S8B);
	RIPEMD160_STEP_S(RIPEMD160_Go, a2, b2, c2, d2, e2, w2[1], RIPEMD160C80, RIPEMD160S8C);
	RIPEMD160_STEP_S(RIPEMD160_Go, e2, a2, b2, c2, d2, w1[3], RIPEMD160C80, RIPEMD160S8D);
	RIPEMD160_STEP_S(RIPEMD160_Go, d2, e2, a2, b2, c2, w2[2], RIPEMD160C80, RIPEMD160S8E);
	RIPEMD160_STEP_S(RIPEMD160_Go, c2, d2, e2, a2, b2, w3[2], RIPEMD160C80, RIPEMD160S8F);

	RIPEMD160_STEP_S(RIPEMD160_F, b2, c2, d2, e2, a2, w3[0], RIPEMD160C90, RIPEMD160S90);
	RIPEMD160_STEP_S(RIPEMD160_F, a2, b2, c2, d2, e2, w3[3], RIPEMD160C90, RIPEMD160S91);
	RIPEMD160_STEP_S(RIPEMD160_F, e2, a2, b2, c2, d2, w2[2], RIPEMD160C90, RIPEMD160S92);
	RIPEMD160_STEP_S(RIPEMD160_F, d2, e2, a2, b2, c2, w1[0], RIPEMD160C90, RIPEMD160S93);
	RIPEMD160_STEP_S(RIPEMD160_F, c2, d2, e2, a2, b2, w0[1], RIPEMD160C90, RIPEMD160S94);
	RIPEMD160_STEP_S(RIPEMD160_F, b2, c2, d2, e2, a2, w1[1], RIPEMD160C90, RIPEMD160S95);
	RIPEMD160_STEP_S(RIPEMD160_F, a2, b2, c2, d2, e2, w2[0], RIPEMD160C90, RIPEMD160S96);
	RIPEMD160_STEP_S(RIPEMD160_F, e2, a2, b2, c2, d2, w1[3], RIPEMD160C90, RIPEMD160S97);
	RIPEMD160_STEP_S(RIPEMD160_F, d2, e2, a2, b2, c2, w1[2], RIPEMD160C90, RIPEMD160S98);
	RIPEMD160_STEP_S(RIPEMD160_F, c2, d2, e2, a2, b2, w0[2], RIPEMD160C90, RIPEMD160S99);
	RIPEMD160_STEP_S(RIPEMD160_F, b2, c2, d2, e2, a2, w3[1], RIPEMD160C90, RIPEMD160S9A);
	RIPEMD160_STEP_S(RIPEMD160_F, a2, b2, c2, d2, e2, w3[2], RIPEMD160C90, RIPEMD160S9B);
	RIPEMD160_STEP_S(RIPEMD160_F, e2, a2, b2, c2, d2, w0[0], RIPEMD160C90, RIPEMD160S9C);
	RIPEMD160_STEP_S(RIPEMD160_F, d2, e2, a2, b2, c2, w0[3], RIPEMD160C90, RIPEMD160S9D);
	RIPEMD160_STEP_S(RIPEMD160_F, c2, d2, e2, a2, b2, w2[1], RIPEMD160C90, RIPEMD160S9E);
	RIPEMD160_STEP_S(RIPEMD160_F, b2, c2, d2, e2, a2, w2[3], RIPEMD160C90, RIPEMD160S9F);

	const uint32_t a = digest[1] + c1 + d2;
	const uint32_t b = digest[2] + d1 + e2;
	const uint32_t c = digest[3] + e1 + a2;
	const uint32_t d = digest[4] + a1 + b2;
	const uint32_t e = digest[0] + b1 + c2;

	digest[0] = a;
	digest[1] = b;
	digest[2] = c;
	digest[3] = d;
	digest[4] = e;
}

void __global__ ripemd_kernel(const uint8_t *input_buffer, uint8_t *output_buffer, size_t limit)
{
	int buffer_id = threadIdx.x + (blockIdx.x * blockDim.x);
	if (buffer_id > limit)
		return;

	ripemd160_ctx ctx;

	// init

	ctx.h[0] = RIPEMD160M_A;
	ctx.h[1] = RIPEMD160M_B;
	ctx.h[2] = RIPEMD160M_C;
	ctx.h[3] = RIPEMD160M_D;
	ctx.h[4] = RIPEMD160M_E;

	ctx.w0[0] = 0;
	ctx.w0[1] = 0;
	ctx.w0[2] = 0;
	ctx.w0[3] = 0;
	ctx.w1[0] = 0;
	ctx.w1[1] = 0;
	ctx.w1[2] = 0;
	ctx.w1[3] = 0;
	ctx.w2[0] = 0;
	ctx.w2[1] = 0;
	ctx.w2[2] = 0;
	ctx.w2[3] = 0;
	ctx.w3[0] = 0;
	ctx.w3[1] = 0;
	ctx.w3[2] = 0;
	ctx.w3[3] = 0;

	ctx.len = 0;

	// update
	//int buffer_id = blockIdx.x;

	const uint8_t* input = input_buffer + buffer_id * SHA3_256_ALIGNED_SIZE;

	uint8_t block[64] = {0};
	for (size_t i = 0; i < 32; ++i)
	{
		block[i] = input[i];
	}
	block[32] = 0x80;
	// 56575859 60616263
	// 0 1 0 0  0 0 0 0
	block[64 - 8 + 1] = 1;

	uint32_t w0[4];
	uint32_t w1[4];
	uint32_t w2[4];
	uint32_t w3[4];

	uint32_t* w = reinterpret_cast<uint32_t*>(block);

	w0[0] = w[0];
	w0[1] = w[1];
	w0[2] = w[2];
	w0[3] = w[3];
	w1[0] = w[4];
	w1[1] = w[5];
	w1[2] = w[6];
	w1[3] = w[7];
	w2[0] = w[8];
	w2[1] = w[9];
	w2[2] = w[10];
	w2[3] = w[11];
	w3[0] = w[12];
	w3[1] = w[13];
	w3[2] = w[14];
	w3[3] = w[15];

	// ripemd160_update_64 (ctx, w0, w1, w2, w3, len - pos1);

	ctx.len += 32;

	ctx.w0[0] = w0[0];
	ctx.w0[1] = w0[1];
	ctx.w0[2] = w0[2];
	ctx.w0[3] = w0[3];
	ctx.w1[0] = w1[0];
	ctx.w1[1] = w1[1];
	ctx.w1[2] = w1[2];
	ctx.w1[3] = w1[3];
	ctx.w2[0] = w2[0];
	ctx.w2[1] = w2[1];
	ctx.w2[2] = w2[2];
	ctx.w2[3] = w2[3];
	ctx.w3[0] = w3[0];
	ctx.w3[1] = w3[1];
	ctx.w3[2] = w3[2];
	ctx.w3[3] = w3[3];

	ripemd160_transform(ctx.w0, ctx.w1, ctx.w2, ctx.w3, ctx.h);

	const uint8_t* digest_ptr = reinterpret_cast<const uint8_t *>(ctx.h);
	uint8_t* output = output_buffer + buffer_id * RIPEMD_ALIGNED_SIZE;
	for (int i = 0; i < 5 * sizeof(uint32_t); i++) {
		output[i] = digest_ptr[i];
	}
}
