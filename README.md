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
```
% cd examples/alphabet
% make
% LD_PRELOAD=$CUTRACE_PATH ./alphabet 
cuInit(.flags=0) = CUDA_SUCCESS
cuDeviceGet(.dev_ref=0x7ffdc1da9bdc, .ord=0) = CUDA_SUCCESS
  > *.dev_ref = 0
cuDevicePrimaryCtxRetain(.ctx_ref=0x7ffdc1da9be0, .dev=0) = CUDA_SUCCESS
  > *.ctx_ref = 0x560a643f55a0
cuCtxSetCurrent(.ctx=0x560a643f55a0) = CUDA_SUCCESS
cuModuleLoad(.cmod_ref=0x7ffdc1da9bd0, .file_name=0x560a63400040) = CUDA_SUCCESS
  > *.cmod_ref = 0x560a64c604b0
cuModuleGetFunction(.func_ref=0x7ffdc1da9bc8, .cmod=0x560a64c604b0, .func_name=0x560a63400055) = CUDA_SUCCESS
  > *.func_ref = 0x560a64c6c440
cuLaunchKernel(.func=0x560a64c6c440, .grid_x=1, .grid_y=1, .grid_z=1, .block_x=1, .block_y=1, .block_z=1, .nbytes_shared=0, .stream=0x0, .kernel_params=0x0, .extra=0x7ffdc1da98c0) = CUDA_SUCCESS
...
```

## Feature Progress
This tool is a WIP and there will be bugs. Many CUDA calls are not yet
supported. Some intercepted calls lack proper argument or return parsing. 

### Checklist
- [x] formatting for basic CUDA Driver API calls
- [ ] formatting for basic CUDA Runtime API calls
- [ ] start time and duration of API calls
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
