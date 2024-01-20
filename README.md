# cutrace - a debugging trace tool analogous to [strace](https://strace.io/) for CUDA

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
% cargo build
% export CUTRACE_PATH=$PWD/target/debug/libcutrace.so
```

## Usage
### Basic Usage
```
% cd examples/alphabet
% make
% LD_PRELOAD=$CUTRACE_PATH ./alphabet 
[0ms] cuInit(.Flags=0) = CUDA_SUCCESS
[58ms] cuDeviceGet(.device=0x7fff6d156c8c, .ordinal=0) = CUDA_SUCCESS
  > *.device = 0
[58ms] cuDevicePrimaryCtxRetain(.pctx=0x7fff6d156c90, .dev=0) = CUDA_SUCCESS
  > *.pctx = CUctx_0x55c5115095a0
[407ms] cuCtxSetCurrent(.ctx=CUctx_0x55c5115095a0) = CUDA_SUCCESS
[407ms] cuModuleLoadData(.module=0x7fff6d156c80, .image=0x7f917de92000) = CUDA_SUCCESS
  > *.module = CUmod_0x55c511d746e0
[411ms] cuModuleGetFunction(.hfunc=0x7fff6d156c78, .hmod=CUmod_0x55c511d746e0, .name="alphabet") = CUDA_SUCCESS
  > *.hfunc = CUfunc_alphabet
[411ms] cuLaunchKernel(.f=CUfunc_alphabet, .gridDimX=1, .gridDimY=1, .gridDimZ=1, .blockDimX=1, .blockDimY=1, .blockDimZ=1, .sharedMemBytes=0, .hStream=CUstream_NULL, .kernelParams=0x0, .extra=0x7fff6d156970) = CUDA_SUCCESS
  > alphabet<<<{1,1,1},{1,1,1},CUstream_NULL>>>(...)
...
```
### Get unique API calls
```
% LD_PRELOAD=$CUTRACE_PATH ./alphabet | cut -d'(' -f1 | sort | uniq | grep "^cu"
cuCtxSetCurrent
cuCtxSynchronize
cuDeviceGet
cuDevicePrimaryCtxRetain
cuInit
cuLaunchKernel
cuMemcpy
cuModuleGetFunction
cuModuleLoad
```
### See device and pinned buffers
```
~/yolov2$ LD_PRELOAD=$CUTRACE_PATH python3 inference.py -o appA.txt yolov2.1.so -n1
[0ms] cudaSetDevice(.device=0) = cudaSuccess
[265ms] cudaMemGetInfo(.free=0x7fffae342228, .total=0x7fffae342230) = cudaSuccess
  > *.free = 5c2b50000
  > *.total = 5dfe30000
[265ms] cudaMalloc(.devPtr=0x7fffae342220, .size=2076672) = cudaSuccess
  > *.devPtr = 0x7f987d000000 : dev_7f987d000000[0x1fb000]
[402ms] cuLaunchKernel(.f=CUfunc_tvmgen_default_fused_nn_conv2d_add_nn_leaky_relu_3_kernel, .gridDimX=4, .gridDimY=52, .gridDimZ=1, .blockDimX=13, .blockDimY=1, .blockDimZ=16, .sharedMemBytes=0, .hStream=CUstream_NULL, .kernelParams=0x7fffae3425d0, .extra=0x0) = CUDA_SUCCESS
  > tvmgen_default_fused_nn_conv2d_add_nn_leaky_relu_3_kernel<<<{4,52,1},{13,1,16},CUstream_NULL>>>(...)
    > arg0: 0x7f9871600000 : dev_7f9871600000[0x548000]
    > arg1: 0x7f9870000000 : dev_7f9870000000[0x1520000]
    > arg2: 0x7f9871c5a000 : dev_7f9871c5a000[0x8000]
    > arg3: 0x7f987d1fc400 : dev_7f987d1fc400[0x100]
```

## Feature Progress
This tool is a WIP and there will be bugs. Many CUDA calls are not yet
supported. Some intercepted calls lack proper argument or return parsing. 

### Checklist
- [x] formatting for basic CUDA Driver API calls
- [x] formatting for basic CUDA Runtime API calls
- [x] start time and duration of API calls
- [ ] adjustable verbosity levels
- [ ] toggleable printing of arguments and return values
- [ ] toggleable printing by API variant (eg: `--only-kernel-launches`)
- [ ] formatting for all CUDA Driver API calls
- [ ] formatting for all CUDA Runtime API calls
- [ ] cuBlas, cuDNN, cuFFT, cuSPARSE, etc

## What about [nsys](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)?
nsys is a great tool for profiling CUDA applications. However the api trace dump
is rather terse, lacking the ability to print the arguments and return values of
each API call. cutrace aims to provide more extensive debugging info.
```
% cd examples/saxpy
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
