#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <sys/time.h>

#include <cufile.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile_sample_utils.h"

#define BLOCK_SIZE (128 * 1024)  // 128KB
#define FILE_SIZE (25UL * 1024 * 1024)  // 100MB
#define FILE_PATH "/mnt/test/testfile"

typedef struct _timer {
	struct timeval start_real_time;
	struct timeval end_real_time;


} Timer;

typedef struct _latency {
	double avg, max, count;
} Latency;

typedef struct _thread_data {
	Timer timer;
	Latency lat;

	CUfileHandle_t fh;
	CUfileDescr_t desc;

	int block_size;
	void* dev_ptr;
} ThreadData;

void timer_start(Timer* timer) {
	gettimeofday(&(timer->start_real_time), NULL);
}

void timer_stop(Timer* timer) {
	gettimeofday(&(timer->end_real_time), NULL);
}

static void update_latency_info(Latency* lat,
	struct timeval start, struct timeval end) 
{
	double val; // ms
	val = (end.tv_sec - start.tv_sec) * 1000.0;
	val += (end.tv_usec - start.tv_usec) / 1000.0;

	if(val > lat->max) {
		lat->max = val;
	}
	lat->avg += val;
	lat->count++;
}

// construct a struct timeval from start and end
static void add_timer(struct timeval* t,
	struct timeval* start, struct timeval* end) 
{
	assert(end->tv_sec >= start->tv_sec);
	if(end->tv_sec == start->tv_sec) {
		assert(end->tv_usec >= start->tv_usec);
	}

	t->tv_sec = end->tv_sec - start->tv_sec;
	t->tv_usec = end->tv_usec - start->tv_usec;
}

static double timeval_to_msec(struct timeval t) {
	double ret;
	ret = t.tv_sec * 1000.0;
	ret += t.tv_usec / 1000000.0;
	return ret;
}

int main() {
	CUfileError_t status;
	CUfileHandle_t fh;
	CUfileDescr_t desc;
	int fd;
	void *devPtr;
	size_t total_blocks = FILE_SIZE / BLOCK_SIZE;
	struct timespec start, end;
	double time_used_ms, bps, iops;
	
	Timer* timer = (Timer*)malloc(sizeof(Timer));
	Latency* lat = (Latency*)malloc(sizeof(Latency));
	memset(timer, 0, sizeof(Timer));
	memset(lat, 0, sizeof(Latency));

	check_cudaruntimecall(cudaSetDevice(0));

	check_cudaruntimecall(cudaMalloc(&devPtr, FILE_SIZE));

	if(cuFileDriverOpen().err != CU_FILE_SUCCESS) {
		printf("driver open failed\n");
		exit(-1);
	}

	fd = open(FILE_PATH, O_RDONLY | O_DIRECT, 0600);
	printf("open file, fd is %d\n", fd);

	memset((void*)&desc, 0, sizeof(CUfileDescr_t));
	desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	desc.handle.fd = fd;
	status = cuFileHandleRegister(&fh, &desc);
	if(status.err != CU_FILE_SUCCESS) {
		std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		close(fd);
		return -1;
	}

	// if(cuFileBufRegister(devPtr, BLOCK_SIZE, 0).err != CU_FILE_SUCCESS) {
	// 	printf("file buf register failed\n");
	// 	exit(-1);
	// }
	check_cudaruntimecall(cudaStreamSynchronize(0));

	printf("Starting sequential read test...\n");
	printf("File: %s\n", FILE_PATH);
	printf("Block size: %d KB\n", BLOCK_SIZE / 1024);
	printf("Total blocks: %lu\n", total_blocks);

	timer_start(timer);
	cuFileRead(fh, devPtr, FILE_SIZE, 0, 0);
	// for(int i = 0; i < total_blocks; i++) {
	// 	struct timeval start, end;
	// 	gettimeofday(&start, NULL);
	// 	ssize_t ret = cuFileRead(fh, devPtr, BLOCK_SIZE, i * BLOCK_SIZE, 0);
	// 	if(ret != BLOCK_SIZE) {
	// 		printf("read failed! return %ld\n", ret);
	// 		exit(-1);
	// 	}
	// 	gettimeofday(&end, NULL);
	// 	update_latency_info(lat, start, end);
	// }
	timer_stop(timer);

	struct timeval realtime_read;
	add_timer(&realtime_read, &timer->start_real_time, &timer->end_real_time);
	time_used_ms = timeval_to_msec(realtime_read);

	bps = (double)FILE_SIZE / time_used_ms * 1000;
	double Mbps = bps / 1024.0 / 1024.0;
	// iops = total_blocks / time_used_ms * 1000;

	printf("total time: %.2f ms\n", time_used_ms);
	printf("bw is %.2f MB/s\n", Mbps);
	printf("average lat is %.2f ms\n", lat->avg / lat->count);
	printf("max lat is %.2f ms\n", lat->max);
	// printf("iops: %.2f\n", iops);

	cuFileBufDeregister(devPtr);
	cuFileHandleDeregister(fh);
	close(fd);
	cudaFree(devPtr);
	cuFileDriverClose();

	return 0;
}
