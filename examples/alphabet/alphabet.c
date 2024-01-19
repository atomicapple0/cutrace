#include <cuda.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN_UP(ptr, align) (((uintptr_t)(ptr) + (align)-1) & ~((uintptr_t)(align)-1))
#define NARGS 12

struct arg_data {
	size_t size;
	size_t value;
};

void cuda_error(int status, CUresult result, const char *format, ...);

int main(int argc, char **argv)
{
	int fd;
	struct stat stat;
	void *img;
	size_t i;
	CUcontext ctx;
	CUdevice dev;
	CUmodule mod;
	CUfunction f;
	CUresult result;

	if ((result = cuInit(0)) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuInit");
	if ((result = cuDeviceGet(&dev, 0)) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuDeviceGet");
	if ((result = cuDevicePrimaryCtxRetain(&ctx, dev)) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuDevicePrimaryCtxRetain");
	if ((result = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuCtxSetCurrent");
	if ((result = cuModuleLoad(&mod, "mod.ptx")) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuModuleLoad");
	if ((result = cuModuleGetFunction(&f, mod, "alphabet")) != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuModuleGetFunction");
	
	struct arg_data data[NARGS] = {
		{sizeof(int), 0},
		{sizeof(float*), 1},
		{sizeof(char), 2},
		{sizeof(char), 3},
		{sizeof(char), 4},
		{sizeof(float*), 5},
		{sizeof(float), 6},
		{sizeof(char), 7},
		{sizeof(int), 8},
		{sizeof(char), 9},
		{sizeof(int*), 10},
		{sizeof(char), 11},
	};
	
	char argBuffer[512];
	size_t offset = 0;
	for (int i = 0; i < NARGS; i++) {
		offset = ALIGN_UP(offset, data[i].size);
		memcpy(&argBuffer[offset], &data[i].value, data[i].size);
		offset += data[i].size;
	}

	void *config[5] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
		CU_LAUNCH_PARAM_END
	};
	
	result = cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, NULL,
				NULL, config);
	if (result != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuLaunchKernel");
	result = cuCtxSynchronize();
	if (result != CUDA_SUCCESS)
		cuda_error(EXIT_FAILURE, result, "cuCtxSynchronize");

	// expect a=0, b=1, ..., k=a, l=b
	
	return 0;
}
