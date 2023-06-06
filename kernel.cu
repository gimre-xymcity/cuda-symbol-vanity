
#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>
#include <vector>

#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "DeviceBuffer.hpp"
#include "chacha_gpu_kernel.hpp"
#include "seed.hpp"

#include "device_utils.h"

#include "sha512_kernel.h"
#include "scalarmult_kernel.h"
#include "sha3_kernel.h"
#include "ripemd_kernel.h"
#include "matching_kernel.h"

#include "vg_constants.h"

#include "gpu_errors.h"

struct GpuState {
	DeviceBuffer<CHACHA_STATE_SIZE * NUM_STATES> random_device;
	DeviceBuffer<SHA512_ALIGNED_SIZE * NUM_STATES> private_device;
	DeviceBuffer<PUBKEY_ALIGNED_SIZE * NUM_STATES> public_device;
	DeviceBuffer<SHA3_256_ALIGNED_SIZE * NUM_STATES> a_sha3_device;
	DeviceBuffer<RIPEMD_ALIGNED_SIZE * NUM_STATES> a_ripemd_device;

	DeviceBuffer<NUM_STATES> matching_device;
	DeviceBuffer<32> pattern;
};

struct VanityContext {
	std::vector<GpuState> states;
};


void xxd(const uint8_t* buffer, size_t buffer_length, size_t grouping = 16, bool showOffset = true) {
	size_t offset = 0;
	while (offset < buffer_length) {
		if (showOffset)
			printf("%08zX: ", offset);

		for (int i = 0; i < grouping; ++i) {
			printf("%02X ", buffer[offset + i]);
		}

		printf("\n");

		offset += grouping;
	}
}

template<size_t N>
void show_all(const DeviceBuffer<N>& deviceBuffer, size_t buf_size) {
	std::vector<uint8_t> buffer;
	deviceBuffer.read(buffer);
	for (size_t counter = 0; counter < NUM_STATES; ++counter) {
		uint8_t* ptr = buffer.data() + counter * buf_size;
		printf("%5d ", counter);
		xxd(ptr, buf_size, buf_size, false);
	}
}

uint8_t unbase32(const char val) {
	if (val >= 'A' && val <= 'Z')
		return static_cast<uint8_t>(val - 'A');

	if (val >= 'a' && val <= 'z')
		return static_cast<uint8_t>(val - 'a');

	if (val >= '2' && val <= '7')
		return static_cast<uint8_t>(val - '2' + 26);

	fprintf(stderr, "character outside of base32 alphabet: %c", val);
	exit(2);
}

size_t unbase32(std::array<uint8_t, 32>& dest, const char* pattern, size_t patternLengthChars) {
	auto first_char_code = unbase32(pattern[0]);
	if (first_char_code > 4) {
		printf("first letter needs to be A, B, C or D\n");
		exit(3);
	}

	size_t dest_index = 0;
	uint16_t working_buffer = first_char_code;
	uint8_t bits_occupied = 2;

	for (size_t i = 1; i < patternLengthChars; ++i) {
		auto code = unbase32(pattern[i]);

		working_buffer <<= 5;
		working_buffer |= code;
		bits_occupied += 5;

		if (bits_occupied > 8) {
			dest[dest_index++] = (working_buffer >> (bits_occupied - 8)) & 0xFF;
			bits_occupied -= 8;
		}
	}

	if (bits_occupied > 0) {
		dest[dest_index++] = (working_buffer << (8 - bits_occupied)) & 0xFF;
	}

	return dest_index;
}

struct Configuration {
	int hash_blockSize = 0;
	int hash_minGridSize = 0;
	int hash_maxActiveBlocks = 0;

	int scalarmult_blockSize = 0;
	int scalarmult_minGridSize = 0;
	int scalarmult_maxActiveBlocks = 0;

	int sha3_maxActiveBlocks = 0;

	int ripe_blockSize = 0;
	int ripe_minGridSize = 0;
	int ripe_maxActiveBlocks = 0;

	int matching_maxActiveBlocks = 0;

	// computed
	int hash_numBlocks;
	int scalarmult_numBlocks;
	int sha3_numBlocks;
	int ripe_numBlocks;
	int matching_numBlocks;
};

void preparePerGpuConfiguration(Configuration& config, int multiProcessorCount) {
	printf("PREPARING CONFIGURATION\n");
	cudaOccupancyMaxPotentialBlockSize(&config.hash_minGridSize, &config.hash_blockSize, sha512_kernel, 0, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.hash_maxActiveBlocks, sha512_kernel, config.hash_blockSize, 0);

	cudaOccupancyMaxPotentialBlockSize(&config.scalarmult_minGridSize, &config.scalarmult_blockSize, scalarmult_kernel, 0, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.scalarmult_maxActiveBlocks, scalarmult_kernel, config.scalarmult_blockSize, 0);

	// specialized per 25 block size
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.sha3_maxActiveBlocks, sha3_256_kernel, 25, 0);

	cudaOccupancyMaxPotentialBlockSize(&config.ripe_minGridSize, &config.ripe_blockSize, ripemd_kernel, 0, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.ripe_maxActiveBlocks, ripemd_kernel, config.ripe_blockSize, 0);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.matching_maxActiveBlocks, matching_kernel, 32, 0);

	// settings check
	//int cpuCount = cudaDevAttrMultiProcessorCount;
	int cpuCount = multiProcessorCount;

	config.hash_numBlocks = config.hash_maxActiveBlocks * cpuCount;
	if (config.hash_numBlocks * config.hash_blockSize > NUM_STATES) {
		fprintf(stderr, "1. bump NUM_STATES %d * %d = %d",
			config.hash_numBlocks,
			config.hash_blockSize,
			config.hash_numBlocks * config.hash_blockSize);
		exit(1);
	}

	config.scalarmult_numBlocks = config.scalarmult_maxActiveBlocks * cpuCount;
	if (config.scalarmult_numBlocks * config.scalarmult_blockSize > NUM_STATES) {
		fprintf(stderr, "2. bump NUM_STATES %d * %d = %d",
			config.scalarmult_numBlocks,
			config.scalarmult_blockSize,
			config.scalarmult_numBlocks * config.scalarmult_blockSize);
		exit(1);
	}

	config.sha3_numBlocks = config.sha3_maxActiveBlocks * cpuCount;
	if (config.sha3_numBlocks * 25 > NUM_STATES) {
		fprintf(stderr, "3. bump NUM_STATES %d * %d = %d", config.sha3_numBlocks, 25, config.sha3_numBlocks * 25);
		exit(1);
	}

	config.ripe_numBlocks = config.ripe_maxActiveBlocks * cpuCount;
	if (config.ripe_numBlocks * config.ripe_blockSize > NUM_STATES) {
		config.ripe_blockSize = NUM_STATES / config.ripe_numBlocks;
	}

	config.matching_numBlocks = config.matching_maxActiveBlocks * cpuCount;

	printf(" *        sha will run with %d %d (%d)\n", config.hash_numBlocks, config.hash_blockSize, config.hash_minGridSize);
	printf(" * scalarmult will run with %d %d (%d)\n", config.scalarmult_numBlocks, config.scalarmult_blockSize, config.scalarmult_minGridSize);
	printf(" *       sha3 will run with %d %d\n", config.sha3_numBlocks, 25);
	printf(" *       ripe will run with %d %d (%d)\n", config.ripe_numBlocks, config.ripe_blockSize, config.ripe_minGridSize);
	printf(" *   matching will run with %d %d\n", config.matching_numBlocks, 32);

}

using MatchCounter = std::array<uint64_t, 20>;

void showMatches(const GpuState& gpuState, int  repetition, MatchCounter& matchCounter, size_t patternSize, std::chrono::steady_clock::time_point& lastShow) {
	std::vector<uint8_t> buffer;
	gpuState.matching_device.read(buffer);

	auto now  = std::chrono::high_resolution_clock::now();
	for (size_t counter = 0; counter < NUM_STATES; ++counter) {
		auto matchCount = buffer.data()[counter];
		if (matchCount != 0) {
			matchCounter[matchCount]++;
			std::chrono::duration<double> elapsed = now - lastShow;
			if (elapsed.count() > REFRESH_RATE) {
				printf("\r[%5d . %5zd] match counters (%6lld, %6lld, %6lld, %6lld, %6lld)",
					repetition,
					counter,
					matchCounter[1],
					matchCounter[2],
					matchCounter[3],
					matchCounter[4],
					matchCounter[5]);

				lastShow = now;
			}

			if (matchCount == patternSize) {
				std::array<uint8_t, CHACHA_STATE_SIZE> priv;
				std::array<uint8_t, PUBKEY_ALIGNED_SIZE> pub;
				std::array<uint8_t, RIPEMD_ALIGNED_SIZE> ripe;

				gpuState.random_device.read(priv, counter);
				gpuState.public_device.read(pub, counter);
				gpuState.a_ripemd_device.read(ripe, counter);

				printf("\npriv: ");
				xxd(priv.data(), 32, 32, false);
				printf(" pub: ");
				xxd(pub.data(), 32, 32, false);
				printf("ripe: ");
				xxd(ripe.data(), 20, 20, false);
			}
		}
	}
}

namespace selfchecks {
	void logAssert(const char* name, bool success) {
		printf(" *  %s self-check %s\n", name, success ? "SUCCESS" : "failure");
		if (!success)
			exit(2);
	}

	void chacha20() {
		// Arrange: https://datatracker.ietf.org/doc/html/rfc7539 2.3.2 Test Vector for the ChaCha20 Block Function
		DeviceBuffer<ChaChaStateSizeInBytes> input_device;
		DeviceBuffer<ChaChaStateSizeInBytes> output_device;

		input_device.write(std::array<uint8_t, ChaChaStateSizeInBytes>({
			0x65, 0x78, 0x70, 0x61,  0x6e, 0x64, 0x20, 0x33,  0x32, 0x2d, 0x62, 0x79,  0x74, 0x65, 0x20, 0x6b,
			0x00, 0x01, 0x02, 0x03,  0x04, 0x05, 0x06, 0x07,  0x08, 0x09, 0x0a, 0x0b,  0x0c, 0x0d, 0x0e, 0x0f,
			0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17,  0x18, 0x19, 0x1a, 0x1b,  0x1c, 0x1d, 0x1e, 0x1f,
			0x01, 0x00, 0x00, 0x00,  0x00, 0x00, 0x00, 0x09,  0x00, 0x00, 0x00, 0x4a,  0x00, 0x00, 0x00, 0x00,
		}));

		// Act:
		std::array<void*, 2> buffers = { input_device.raw(), output_device.raw() };
		gpu_chacha20_block(nullptr, buffers.data(), nullptr, 0);

		// Assert:
		auto result = output_device.read();
		std::array<uint8_t, ChaChaStateSizeInBytes> expected = {
			0x10, 0xf1, 0xe7, 0xe4,  0xd1, 0x3b, 0x59, 0x15,  0x50, 0x0f, 0xdd, 0x1f,  0xa3, 0x20, 0x71, 0xc4,
			0xc7, 0xd1, 0xf4, 0xc7,  0x33, 0xc0, 0x68, 0x03,  0x04, 0x22, 0xaa, 0x9a,  0xc3, 0xd4, 0x6c, 0x4e,
			0xd2, 0x82, 0x64, 0x46,  0x07, 0x9f, 0xaa, 0x09,  0x14, 0xc2, 0xd7, 0x05,  0xd9, 0x8b, 0x02, 0xa2,
			0xb5, 0x12, 0x9c, 0xd1,  0xde, 0x16, 0x4e, 0xb9,  0xcb, 0xd0, 0x83, 0xe8,  0xa2, 0x50, 0x3c, 0x4e
		};

		logAssert("chacha20 block", result == expected);
	}

	void sha512() {
		// Arrange: NIST test vectors
		DeviceBuffer<32> input_device;
		DeviceBuffer<SHA512_ALIGNED_SIZE> output_device;

		// Act:
		input_device.write(std::array<uint8_t, 32>({
			0x8c, 0xcb, 0x8, 0xd2, 0xa1, 0xa2, 0x82, 0xaa, 0x8c, 0xc9, 0x99, 0x2, 0xec, 0xaf, 0xf, 0x67, 0xa9,
			0xf2, 0x1c, 0xff, 0xe2, 0x80, 0x5, 0xcb, 0x27, 0xfc, 0xf1, 0x29, 0xe9, 0x63, 0xf9, 0x9d
		}));

		sha512_kernel<<<1, 1>>>(input_device.raw(), output_device.raw(), 1);

		// Assert:
		auto result = output_device.read();
		std::array<uint8_t, 64> expected = {
			0x45, 0x51, 0xde, 0xf2, 0xf9, 0x12, 0x73, 0x86, 0xee, 0xa8, 0xd4, 0xda, 0xe1, 0xea, 0x8d, 0x8e,
			0x49, 0xb2, 0xad, 0xd0, 0x50, 0x9f, 0x27, 0xcc, 0xbc, 0xe7, 0xd9, 0xe9, 0x50, 0xac, 0x7d, 0xb0,
			0x1d, 0x5b, 0xca, 0x57, 0x9c, 0x27, 0x1b, 0x9f, 0x2d, 0x80, 0x67, 0x30, 0xd8, 0x8f, 0x58, 0x25,
			0x2f, 0xd0, 0xc2, 0x58, 0x78, 0x51, 0xc3, 0xac, 0x8a, 0x0e, 0x72, 0xb4, 0xe1, 0xdc, 0x0d, 0xa6
		};

		// apply clamp
		expected[0] &= 248;
		expected[31] &= 63;
		expected[31] |= 64;

		logAssert("sha2-512", result == expected);
	}

	void scalarmult() {
		DeviceBuffer<32> input_device;
		DeviceBuffer<PUBKEY_ALIGNED_SIZE> output_device;

		// Act: from symbol test vectors
		// vector: 575dbb30... (passed via sha512 and clamped)
		input_device.write(std::array<uint8_t, 32>({
			0x38, 0x28, 0x0f, 0xb3, 0xd9, 0x9f, 0xe7, 0x76, 0x4c, 0x66, 0x32, 0x0f, 0xe0, 0xcb, 0x09, 0x1b,
			0xb0, 0xdb, 0x32, 0x37, 0xc8, 0x28, 0x88, 0xf3, 0x7e, 0xdf, 0xfb, 0xdf, 0x37, 0x54, 0x9d, 0x43
		}));

		scalarmult_kernel<<<1, 1>>>(input_device.raw(), output_device.raw(), 1);

		// Assert:
		auto result = output_device.read();
		std::array<uint8_t, 32> expected = {
			0x2e, 0x83, 0x41, 0x40, 0xfd, 0x66, 0xcf, 0x87, 0xb2, 0x54, 0xa6, 0x93, 0xa2, 0xc7, 0x86, 0x2c,
			0x81, 0x92, 0x17, 0xb6, 0x76, 0xd3, 0x94, 0x32, 0x67, 0x15, 0x66, 0x25, 0xe8, 0x16, 0xec, 0x6f
		};

		logAssert("scalarmult (priv-to-pub)", result == expected);
	}

	void sha3() {
		DeviceBuffer<32> input_device;
		DeviceBuffer<SHA3_256_ALIGNED_SIZE> output_device;

		// Act: http://mumble.net/~campbell/hg/sha3/kat/ShortMsgKAT_SHA3-256.txt
		input_device.write(std::array<uint8_t, 32>({
			0x9f, 0x2f, 0xcc, 0x7c, 0x90, 0xde, 0x09, 0x0d, 0x6b, 0x87, 0xcd, 0x7e, 0x97, 0x18, 0xc1, 0xea,
			0x6c, 0xb2, 0x11, 0x18, 0xfc, 0x2d, 0x5d, 0xe9, 0xf9, 0x7e, 0x5d, 0xb6, 0xac, 0x1e, 0x9c, 0x10
		}));

		sha3_256_kernel<<<1, 25>>>(input_device.raw(), output_device.raw(), 1);

		// Assert:
		auto result = output_device.read();
		std::array<uint8_t, 32> expected = {
			0x2f, 0x1a, 0x5f, 0x71, 0x59, 0xe3, 0x4e, 0xa1, 0x9c, 0xdd, 0xc7, 0x0e, 0xbf, 0x9b, 0x81, 0xf1,
			0xa6, 0x6d, 0xb4, 0x06, 0x15, 0xd7, 0xea, 0xd3, 0xcc, 0x1f, 0x1b, 0x95, 0x4d, 0x82, 0xa3, 0xaf
		};

		logAssert("sha3-256", result == expected);
	}

	void ripemd160() {
		DeviceBuffer<32> input_device;
		DeviceBuffer<RIPEMD_ALIGNED_SIZE> output_device;

		// Act: (same input vec as sha3 above)
		input_device.write(std::array<uint8_t, 32>({
			0x9f, 0x2f, 0xcc, 0x7c, 0x90, 0xde, 0x09, 0x0d, 0x6b, 0x87, 0xcd, 0x7e, 0x97, 0x18, 0xc1, 0xea,
			0x6c, 0xb2, 0x11, 0x18, 0xfc, 0x2d, 0x5d, 0xe9, 0xf9, 0x7e, 0x5d, 0xb6, 0xac, 0x1e, 0x9c, 0x10
		}));

		ripemd_kernel<<<1, 1>>>(input_device.raw(), output_device.raw(), 1);

		// Assert:
		auto result = output_device.read();
		std::array<uint8_t, 20> expected = {
			0xdc, 0xa1, 0xe2, 0x46, 0xd8, 0x9f, 0x25, 0xca, 0x39, 0x9c, 0xa8, 0x50, 0x71, 0x90, 0x77, 0xb3, 0xa2, 0x7e, 0x3e, 0x0f
		};

		logAssert("ripemd-160", 0 == memcmp(result.data(), expected.data(), 20));
	}
}

void runSelfChecks() {
	printf("RUNNING SELF-CHECKS\n");
	selfchecks::chacha20();
	selfchecks::sha512();
	selfchecks::scalarmult();
	selfchecks::sha3();
	selfchecks::ripemd160();
}

int main(int argc, char** argv) {
	int gpuCount = 0;
	cudaGetDeviceCount(&gpuCount);

	if (argc < 3) {
		printf(R"(Syntax:
  cuda-symbol-vanity-gen.exe [pattern] [#loops]

  `pattern` is only searched at the beginning of an address and needs to exclude network byte i.e. ATURE rather than NATURE
  `loops` is number of repetitions
)");
		exit(0);
	}

	// process arguments

	int num_repetitions = atoi(argv[2]);;
	std::array<uint8_t, 32> pattern = {};
	size_t patternLengthChars = strlen(argv[1]);

	if (patternLengthChars > 10) {
		printf("pattern has > 10 chars, this will likely take quite a long time");
	}

	if (patternLengthChars > 19) {
		printf("invalid pattern - too long, pattern has > 19 chars");
		exit(1);
	}

	auto patternSize = unbase32(pattern, argv[1], patternLengthChars);
	printf("pattern ex network byte, as hex:\n > ");
	xxd(pattern.data(), 20, 20);

	VanityContext context;
	context.states.resize(gpuCount);

	for (int gpuId = 0; gpuId < gpuCount; ++gpuId) {
		// initialize seed via CPU and copy it to GPU
		std::vector<uint8_t> seed(ChaChaStateSizeInBytes * NUM_STATES);
		create_random_seed(seed.data(), static_cast<uint32_t>(seed.size()));

		DeviceBuffer<ChaChaStateSizeInBytes * NUM_STATES> seed_device;
		seed_device.write(seed);

		// get info
		cudaSetDevice(gpuId);

		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, gpuId);
		printf("device: %s\n", device.name);
		printf(" : maxThreadsPerBlock: %d\n", device.maxThreadsPerBlock);
		printf(" : multiProcessorCount: %d\n", device.multiProcessorCount);

		runSelfChecks();

		Configuration config;
		preparePerGpuConfiguration(config, device.multiProcessorCount);

		// ---

		auto start  = std::chrono::high_resolution_clock::now();

		// prepare randomness source
		auto& gpuState = context.states[gpuId];
		std::array<void*, 2> buffers = { seed_device.raw(), gpuState.random_device.raw() };
		uint32_t num_inputs = NUM_STATES;

		// copy pattern to gpu
		gpuState.pattern.write(pattern);

		auto match_previousShow = start;
		MatchCounter match_counter = { 0 };
		for (int repetition = 0; repetition < num_repetitions; ++repetition) {
			// CHACHA KERNEL
			gpu_chacha20_block(
				nullptr,
				buffers.data(),
				reinterpret_cast<const char*>(&num_inputs),
				sizeof(num_inputs)
			);

			// SHA512 KERNEL
			for (size_t counter = 0; counter < NUM_STATES; counter += (config.hash_numBlocks * config.hash_blockSize)) {
				const uint8_t* rnd_source = gpuState.random_device.raw() + counter * ChaChaStateSizeInBytes;
				uint8_t* output = gpuState.private_device.raw() + counter * SHA512_ALIGNED_SIZE;

				size_t limit = std::min<size_t>(config.hash_numBlocks * config.hash_blockSize, NUM_STATES - counter);
				sha512_kernel<<<config.hash_numBlocks, config.hash_blockSize>>>(rnd_source, output, limit);
			}

			for (size_t counter = 0; counter < NUM_STATES; counter += (config.scalarmult_numBlocks * config.scalarmult_blockSize)) {
				const uint8_t* input = gpuState.private_device.raw() + counter * SHA512_ALIGNED_SIZE;
				uint8_t* output =  gpuState.public_device.raw() + counter * PUBKEY_ALIGNED_SIZE;

				size_t limit = std::min<size_t>(config.scalarmult_numBlocks * config.scalarmult_blockSize, NUM_STATES - counter);
				scalarmult_kernel<<<config.scalarmult_numBlocks, config.scalarmult_blockSize>>>(input, output, limit);
			}


			// sha3 is processing only num_blocks buffers at a time, as each invocation is specialized for 25 threads
			for (size_t counter = 0; counter < NUM_STATES; counter += config.sha3_numBlocks) {
				const uint8_t* input = gpuState.public_device.raw() + counter * PUBKEY_ALIGNED_SIZE;
				uint8_t* output =  gpuState.a_sha3_device.raw() + counter * SHA3_256_ALIGNED_SIZE;

				size_t limit = std::min<size_t>(config.sha3_numBlocks, NUM_STATES - counter);
				sha3_256_kernel<<<config.sha3_numBlocks, 25>>>(input, output, limit);
			}

			for (size_t counter = 0; counter < NUM_STATES; counter += (config.ripe_numBlocks * config.ripe_blockSize)) {
				const uint8_t* input = gpuState.a_sha3_device.raw() + counter * SHA3_256_ALIGNED_SIZE;
				uint8_t* output =  gpuState.a_ripemd_device.raw() + counter * RIPEMD_ALIGNED_SIZE;

				size_t limit = std::min<size_t>(config.ripe_numBlocks * config.ripe_blockSize, NUM_STATES - counter);
				ripemd_kernel<<<config.ripe_numBlocks, config.ripe_blockSize>>>(input, output, limit);
			}

			for (uint64_t counter = 0; counter < NUM_STATES; counter += (config.matching_numBlocks * 32)) {
				const uint8_t* input = gpuState.a_ripemd_device.raw() + counter * RIPEMD_ALIGNED_SIZE;
				uint8_t* output = gpuState.matching_device.raw() + counter;

				int num_blocks = std::min<int>(config.matching_numBlocks, (NUM_STATES - static_cast<int>(counter)) / 32);
				matching_kernel<<<num_blocks, 32>>>(
					counter,
					input,
					output,
					gpuState.pattern.raw(),
					patternSize);
			}

			showMatches(gpuState, repetition, match_counter, patternSize, match_previousShow);
			cudaMemcpy(seed_device.raw(), context.states[gpuId].random_device.raw(), seed_device.SIZE, cudaMemcpyDeviceToDevice);
		}

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;

		auto hps = (num_repetitions * NUM_STATES) / elapsed.count();
		auto mhps = hps / 1000 / 1000;
		printf("\ntook %f %d (%f mhps, %f)\n", elapsed.count(), (num_repetitions * NUM_STATES), mhps, hps);
	}

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return 0;
}