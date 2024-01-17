# cutrace - a Cuda API trace tool similar to strace

This is cutrace -- a diagnostic, debugging and instructional utility for CUDA.
CUDA Driver and Runtime API calls are intercepted via a dynamically linked
library installed with LD_PRELOAD. The function, arguments, and return values
for each call is printed to console. Additionally, returned values via output
arguments are printed as well.

This tool is a WIP. Many CUDA calls are not yet supported. Some intercepted lack
proper argument or return parsing. 

## Requirements
This has been tested on Ubuntu 22.04 with CUDA 12.2.

## Build
```
% git clone https://github.com/atomicapple0/cutrace.git
% cd cutrace
% cargo build --release
% export CUTRACE_PATH=$PWD/target/release/libcutrace.so
```

## Usage
```
% LD_PRELOAD=$CUTRACE_PATH ./sample-cuda-program
cuInit(0) = CUDA_SUCCESS
cuDeviceGet(&dev=07ffff100, 0) = CUDA_SUCCESS
  | dev = 0
cuCtxCreate_v2(&ctx=07ffff200, 0x0, dev=0) = CUDA_SUCCESS
  | ctx = 0x30000000
cuModuleLoadData(&mod=07ffff300, 0xbeefdead) = CUDA_SUCCESS
  | mod = 0x30000100
cuModuleGetFunction(&func=07ffff400, mod=0x30000100, "my_kernel") = CUDA_SUCCESS
  | func = 0x30000200
cuMemAlloc_v2(&dptr=07ffff500, bytes=0x1000) = CUDA_SUCCESS
  | dptr = 0x60000000
cuLaunchKernel_v2(func=0x30000200, grid={1, 1, 1}, block={1, 1, 1}, shm_bytes=0x0, stream=CU_NULL_STREAM, ...) = CUDA_SUCCESS
  | my_kernel<<<{1,1,1}, {1,1,1}, 0x0, CU_NULL_STREAM>>>(0x60000000, 0x1000)
...
```