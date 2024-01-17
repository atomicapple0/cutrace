# cutrace - a debugging trace tool analagous to [strace](https://strace.io/) for CUDA

This is cutrace -- a diagnostic, debugging and instructional utility for CUDA.
CUDA Driver and Runtime API calls are intercepted via a dynamically linked
custom CUDA lightweight wrapper library installed with LD_PRELOAD. The function
name, arguments, and return values for each call are printed to the console.
Additionally, returned values via output arguments are printed as well.

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
% cd example
% LD_PRELOAD=$CUTRACE_PATH ./saxpy
cuInit(0) = CUDA_SUCCESS
cuDeviceGet(&dev=07ffff100, 0) = CUDA_SUCCESS
  > dev = 0
cuCtxCreate_v2(&ctx=07ffff200, 0x0, dev=0) = CUDA_SUCCESS
  > ctx = 0x30000000
cuModuleLoadData(&mod=07ffff300, 0xbeefdead) = CUDA_SUCCESS
  > mod = 0x30000100
cuModuleGetFunction(&func=07ffff400, mod=0x30000100, "my_kernel") = CUDA_SUCCESS
  > func = 0x30000200
cuMemAlloc_v2(&dptr=07ffff500, bytes=0x1000) = CUDA_SUCCESS
  > dptr = 0x60000000
cuLaunchKernel_v2(func=0x30000200, grid={1, 1, 1}, block={1, 1, 1}, shm_bytes=0x0, stream=CU_NULL_STREAM, ...) = CUDA_SUCCESS
  > my_kernel<<<{1,1,1}, {1,1,1}, 0x0, CU_NULL_STREAM>>>(0x60000000, 0x1000)
...
```

## Feature Progress
This tool is a WIP. Many CUDA calls are not yet supported. Some intercepted
calls lack proper argument or return parsing. 

Progress
- [x] formatting for basic CUDA Driver API calls
- [x] formatting for basic CUDA Runtime API calls
- [ ] start time and duration of API calls
- [ ] adjustable verbosity levels
- [ ] toggleable printing of arguments and return values
- [ ] toggleable printing by API variant (eg: `--only-kernel-launches`)
- [ ] formatting for all CUDA Driver API calls
- [ ] formatting for all CUDA Runtime API calls

## What about [nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)?
nsys is a great tool for profiling CUDA applications. However the api trace
dump is rather terse, lacking the ability to print the arguments and return
values of each API call. cutrace aims to provide more extensive debugging info.
```
% cd example
% nsys profile --trace=cuda --sample=none --cpuctxsw=none ./saxpy
% nsys stats --format csv --output - --report cuda_api_trace report1.nsys-rep
Generating SQLite file report1.sqlite from report1.nsys-rep
Exporting 1374 events: [===================================================100%]
Processing [report1.sqlite] with [/usr/local/cuda-12.2/nsight-systems-2023.2.3/host-linux-x64/reports/cuda_api_trace.py]... 
Start (ns),Duration (ns),Name,Result,CorrID,Pid,Tid,T-Pri,Thread Name
456173024,2155,cuModuleGetLoadingMode,0,1,811552,811552,20,saxpy
459348415,170736162,cudaMalloc,0,118,811552,811552,20,saxpy
630087558,1270617,cudaMalloc,0,120,811552,811552,20,saxpy
637060753,367537,cudaMemcpy,0,122,811552,811552,20,saxpy
...
```
