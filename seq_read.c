#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cufile.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#define BLOCK_SIZE (128 * 1024)  // 128KB
#define FILE_SIZE (10UL * 1024 * 1024 * 1024)  // 10GB
#define FILE_PATH "/mnt/test/testfile"

int main() {
	CUfileError_t status;
	CUfileHandle_t fh;
	CUfileDescr_t desc;
	int fd;
	void *devPtr;
	size_t total_blocks = FILE_SIZE / BLOCK_SIZE;
	struct timespec start, end;
	double time_used, throughput;

	cudaMalloc(&devPtr, BLOCK_SIZE);

	status = cuFileDriverOpen();

	fd = open(FILE_PATH, O_RDONLY);

	memset(&desc, 0, sizeof(CUfileDescr_t));
	desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	desc.handle.fd = fd;
	status = cuFileHandleRegister(&fh, &desc);

	status = cuFileBufRegister(devPtr, BLOCK_SIZE, 0);

	printf("Starting sequential read test...\n");
	printf("File: %s\n", FILE_PATH);
	printf("Block size: %d KB\n", BLOCK_SIZE / 1024);
	printf("Total blocks: %lu\n", total_blocks);

	clock_gettime(1, &start);

	for(int i = 0; i < total_blocks; i++) {
		ssize_t ret = cuFileRead(fh, devPtr, BLOCK_SIZE, i * BLOCK_SIZE, 0);
	}

	clock_gettime(1, &end);

	time_used = (end.tv_sec - start.tv_sec) * 1000.0 + 
		(end.tv_nsec - start.tv_nsec) / 1000000.0;
	throughput = (double)FILE_SIZE / (1024 * 1024) / time_used * 1000;

	printf("total time: %.2f ms\n", time_used);
	printf("bw is %.2f MB/s", throughput);

	cuFileBufDeregister(devPtr);
	cuFileHandleDeregister(fh);
	close(fd);
	cudaFree(devPtr);
	cuFileDriverClose();

	return 0;
}
