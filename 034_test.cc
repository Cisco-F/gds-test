/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/*
 *  This is a data-integrity test for cuFileReadAsync/WriteAsync APIs with cuFile stream registration.
 *  This shows how the async apis can be used in a batch mode.
 *  The test does the following:
 *  1. Creates a Test file with pattern
 *  2. Test file is loaded to device memory (cuFileReadAsync)
 *  3. From device memory, data is written to a new file (cuFileWriteAsync)
 *  4. Test file and new file are compared for data integrity
 *
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * e9d2f73120b2f2b1d2782e8ef5a42a3259b3c2badc5edb6ee04d4bc7b7633a
 * SHA SUM Match
 * API Version :
 * 440-442(us) : 1
 * 510-512(us) : 1
 *
 */
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
#include <openssl/sha.h>

#include <cuda_runtime.h>
#include <sys/stat.h>

// include this header
#include "cufile.h"
#include "cufile_sample_utils.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
using namespace std::chrono;

using namespace std;

// copy bytes
#define MAX_BUF_SIZE (8 * 1024 * 1024UL)
#define BLOCK_SIZE (8 * 1024 * 1024UL)
#define MAX_BATCH_SIZE 500
#define MAX_STREAM_CNT 1
typedef struct io_args
{
   void *devPtr;
   size_t max_size;
   off_t offset;
   off_t buf_off;
   ssize_t read_bytes_done;
   ssize_t write_bytes_done;
} io_args_t;

// buffer pointer offset is set at submission time 
#define CU_FILE_STREAM_FIXED_BUF_OFFSET         1
// file offset is set at submission time 
#define CU_FILE_STREAM_FIXED_FILE_OFFSET        2
// file size is set at submission time 
#define CU_FILE_STREAM_FIXED_FILE_SIZE          4
// size, offset and buffer offset are 4k aligned 
#define CU_FILE_STREAM_PAGE_ALIGNED_INPUTS      8 
#define CU_FILE_STREAM_FIXED_AND_ALIGNED        15

void print_device_buffer(float* dev_ptr, size_t num_elements) {
    // 1. 在主机端分配缓冲区
    float* host_buffer = new float[num_elements];

    // 2. 从设备拷贝到主机
    cudaMemcpy(host_buffer, dev_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // 3. 打印前几个元素（例如前 10 个）
    for (size_t i = 0; i < num_elements; ++i) {
        std::cout << "Index " << i << ": " << host_buffer[i] << std::endl;
    }

    // 4. 清理
    delete[] host_buffer;
}

int main(int argc, char *argv[]) {
        int fd = -1;
        ssize_t ret = 0;
        io_args_t args[MAX_BATCH_SIZE];
	CUfileError_t status;
        CUfileDescr_t cf_descr;
        CUfileHandle_t cf_handle;
	const char *TEST_READFILE;
	// io stream associated with the I/O
	cudaStream_t io_stream[MAX_STREAM_CNT];

        char env_str[] = "CUFILE_ENV_PATH_JSON=/root/anaconda3/envs/deepspeed/lib/python3.10/site-packages/deepspeed/cufile.json";
        putenv(env_str);
        cuFileDriverOpen();
        size_t direct_io_size = (size_t)BLOCK_SIZE / 1024;
        // std::cout<<"direct_io_size:"<<direct_io_size<<std::endl;
        status = cuFileDriverSetMaxDirectIOSize(direct_io_size);
        if (status.err != CU_FILE_SUCCESS) {
            std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
            exit(EXIT_FAILURE);
        }

        if(argc < 2) {
                std::cerr << argv[0] << " <readfilepath>"<< std::endl;
                exit(1);
        }

        TEST_READFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(0));

        memset(&args, 0, sizeof(args));

        fd = open(TEST_READFILE, O_RDONLY);
        if (fd == -1) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }
        struct stat stat_buf;
        fstat(fd, &stat_buf);
        off_t filesize = stat_buf.st_size;
        size_t num_partitions = (filesize + MAX_BUF_SIZE - 1) / MAX_BUF_SIZE;
        // size_t remainder = filesize - (num_partitions - 1) * MAX_BUF_SIZE;

        //allocate Memory
        for (unsigned i = 0; i < num_partitions; i++) {
                args[i].max_size = MAX_BUF_SIZE;
                // Allocate device Memory and register with cuFile
                check_cudaruntimecall(cudaMalloc(&args[i].devPtr, args[i].max_size));
                // Register buffers. For unregistered buffers, this call is not required.
                status = cuFileBufRegister(args[i].devPtr, args[i].max_size, 0);
                if (status.err != CU_FILE_SUCCESS) {
                        std::cerr << "buf register failed: "
                                << cuFileGetErrorString(status) << std::endl;
                        ret = -1;
                        return ret;
                }
                if(i > 0)
                        args[i].offset += args[i -1].offset + args[i].max_size;
                else
                        args[i].offset = 0;

		/* Create a stream for each of the batch entries. One can create a single stream as well for all I/Os */
                if (io_stream[i % MAX_STREAM_CNT] == nullptr) {
                    check_cudaruntimecall(cudaStreamCreateWithFlags(&io_stream[i % MAX_STREAM_CNT], cudaStreamNonBlocking));
                }
                // special case for holes
                check_cudaruntimecall(cudaMemsetAsync(args[i].devPtr, 0, args[i].max_size, io_stream[i % MAX_STREAM_CNT]));
                // std::cout << "register stream " << io_stream[i] << " with cuFile" << std::endl;
                cuFileStreamRegister(io_stream[i % MAX_STREAM_CNT], CU_FILE_STREAM_FIXED_AND_ALIGNED);
        }


        // Register the filehandles
        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
                std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		ret = -1;
                return ret;
        }

        auto start_time = system_clock::now();

	for (unsigned i = 0; i < num_partitions; i++) {
		status = cuFileReadAsync(cf_handle, (unsigned char *)args[i].devPtr,
                                         &args[i].max_size, &args[i].offset,
                                         &args[i].buf_off, &args[i].read_bytes_done,
                                         io_stream[i % MAX_STREAM_CNT]);
		if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "read failed : "
				<< cuFileGetErrorString(status) << std::endl;
                        ret = -1;
                        return ret;
		}

                check_cudaruntimecall(cudaStreamSynchronize(io_stream[i % MAX_STREAM_CNT]));

                printf("[Debug]: cuFileReadAsync parameters:\n");
                printf("  devPtr        = %p\n", args[i].devPtr);
                printf("  max_size      = %zu\n", args[i].max_size);
                printf("  offset        = %jd\n", (intmax_t)args[i].offset);
                printf("  buf_off       = %jd\n", (intmax_t)args[i].buf_off);
                printf("  read_bytes_done (before) = %zd\n", args[i].read_bytes_done);
                printf("  io_stream     = %p\n", io_stream[i % MAX_STREAM_CNT]);
	}

        //synchronize streams and check for result
        for (unsigned i = 0; i < num_partitions; i++) {
                check_cudaruntimecall(cudaStreamSynchronize(io_stream[i % MAX_STREAM_CNT]));
        }

        auto end_time = system_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double seconds = elapsed.count();
        double filesize_in_gb = static_cast<double>(filesize) / (1024 * 1024 * 1024);
        double bandwidth = (double)filesize_in_gb / (seconds); // GB/s

        std::cout << "File size: " << filesize_in_gb << " GB " << std::endl;
        std::cout << "Elapsed time: " << seconds << " seconds" << std::endl;
        std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl << std::endl ;

        print_device_buffer((float *)args[0].devPtr, 10);

        for (unsigned i = 0; i < num_partitions; i++) {
                if(args[i].devPtr) {
                        cuFileBufDeregister(args[i].devPtr);
                        check_cudaruntimecall(cudaFree(args[i].devPtr));
                }
        }
        for (unsigned i = 0; i < MAX_STREAM_CNT; i++) {
                if(io_stream[i % MAX_STREAM_CNT]) {
                        cuFileStreamDeregister(io_stream[i % MAX_STREAM_CNT]);
                        check_cudaruntimecall(cudaStreamDestroy(io_stream[i % MAX_STREAM_CNT]));
                }
        }
        if(cf_handle)
                cuFileHandleDeregister(cf_handle);
        // if(cf_whandle)
        //         cuFileHandleDeregister(cf_whandle);
	close(fd);
	// close(wfd);
	return ret;
}
