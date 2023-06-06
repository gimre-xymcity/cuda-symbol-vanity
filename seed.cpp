#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS

#include <bcrypt.h>
#include <ntstatus.h>
#else
#include <stdio.h>
#endif

#include "seed.hpp"

int create_random_seed(uint8_t *seed, uint32_t length) {
#ifdef _WIN32
	BCRYPT_ALG_HANDLE prov;

	if (STATUS_SUCCESS != BCryptOpenAlgorithmProvider(&prov, BCRYPT_RNG_ALGORITHM, NULL, 0)) {
		return 1;
	}

	if (STATUS_SUCCESS != BCryptGenRandom(prov, seed, length, 0)) {
		BCryptCloseAlgorithmProvider(prov, 0);
		return 1;
	}

	BCryptCloseAlgorithmProvider(prov, 0);

#else
    FILE *f = fopen("/dev/urandom", "rb");

    if (f == NULL) {
        return 1;
    }

    size_t res = fread(seed, 1, length, f);
    if (res != length) {
        return 1;
    }

    fclose(f);
#endif

    return 0;
}
