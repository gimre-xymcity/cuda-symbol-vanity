#pragma once
#include <array>
#include <vector>
#include <cuda_runtime.h>


template <size_t BYTE_SIZE>
class DeviceBuffer {
public:
	static const size_t SIZE = BYTE_SIZE;

	DeviceBuffer() : m_ptr(nullptr) {
		if (cudaMalloc(&m_ptr, BYTE_SIZE) != cudaSuccess)
			throw std::runtime_error("Could not allocate device memory!");
	}

	DeviceBuffer(const std::array<uint8_t, BYTE_SIZE>& data) : DeviceBuffer() {
		write(data);
	}

	~DeviceBuffer() {
		if (m_ptr != nullptr) {
			cudaFree(m_ptr);
			m_ptr = nullptr;
		}
	}

	void write(const std::vector<uint8_t>& data) {
		if (data.size() < BYTE_SIZE)
			throw std::runtime_error("Destination buffer too small");

		if (cudaMemcpy(m_ptr, data.data(), BYTE_SIZE, cudaMemcpyHostToDevice) != cudaSuccess)
			throw std::runtime_error("Failed to copy data to device memory!");
	}

	void write(const std::array<uint8_t, BYTE_SIZE>& data) {
		if (cudaMemcpy(m_ptr, data.data(), BYTE_SIZE, cudaMemcpyHostToDevice) != cudaSuccess)
			throw std::runtime_error("Failed to copy data to device memory!");
	}

	void read(std::vector<uint8_t>& buffer) const {
		buffer.resize(BYTE_SIZE);

		if (cudaMemcpy(buffer.data(), m_ptr, BYTE_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess)
			throw std::runtime_error("Failed to copy data from device memory!");
	}

	template<size_t OBJECT_SIZE>
	void read(std::array<uint8_t, OBJECT_SIZE>& buffer, size_t idx) const {
		if (cudaMemcpy(buffer.data(), m_ptr + idx * OBJECT_SIZE, OBJECT_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess)
			throw std::runtime_error("Failed to copy data from device memory!");
	}

	void read(std::array<uint8_t, BYTE_SIZE>& buffer) const {
		if (cudaMemcpy(buffer.data(), m_ptr, BYTE_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess)
			throw std::runtime_error("Failed to copy data from device memory!");

	}

	std::array<uint8_t, BYTE_SIZE> read() const {
		std::array<uint8_t, BYTE_SIZE> buffer;
		read(buffer);
		return buffer;
	}

	uint8_t* raw() { return m_ptr; }

private:
	uint8_t* m_ptr;

};
