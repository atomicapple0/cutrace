use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};

use cudarc::runtime::sys::{
    cudaArrayMemoryRequirements, cudaArraySparseProperties, cudaArray_const_t, cudaArray_t,
    cudaChannelFormatDesc, cudaChannelFormatKind, cudaDeviceAttr, cudaDeviceP2PAttr,
    cudaDeviceProp, cudaDriverEntryPointQueryResult, cudaError_t, cudaEvent_t, cudaExtent,
    cudaExternalMemoryBufferDesc, cudaExternalMemoryHandleDesc,
    cudaExternalMemoryMipmappedArrayDesc, cudaExternalMemory_t, cudaExternalSemaphoreHandleDesc,
    cudaExternalSemaphoreSignalNodeParams, cudaExternalSemaphoreSignalParams,
    cudaExternalSemaphoreWaitNodeParams, cudaExternalSemaphoreWaitParams, cudaExternalSemaphore_t,
    cudaFlushGPUDirectRDMAWritesScope, cudaFlushGPUDirectRDMAWritesTarget, cudaFuncAttribute,
    cudaFuncAttributes, cudaFuncCache, cudaGraphExecUpdateResultInfo, cudaGraphExec_t,
    cudaGraphInstantiateParams, cudaGraphMemAttributeType, cudaGraphNodeParams, cudaGraphNodeType,
    cudaGraphNode_t, cudaGraph_t, cudaGraphicsResource_t, cudaHostFn_t, cudaHostNodeParams,
    cudaIpcEventHandle_t, cudaIpcMemHandle_t, cudaKernelNodeParams, cudaKernel_t,
    cudaLaunchAttributeID, cudaLaunchAttributeValue, cudaLaunchConfig_t, cudaLaunchParams,
    cudaLimit, cudaMemAccessDesc, cudaMemAccessFlags, cudaMemAllocNodeParams,
    cudaMemAllocationHandleType, cudaMemLocation, cudaMemPoolAttr, cudaMemPoolProps,
    cudaMemPoolPtrExportData, cudaMemPool_t, cudaMemRangeAttribute, cudaMemcpy3DParms,
    cudaMemcpy3DPeerParms, cudaMemcpyKind, cudaMemoryAdvise, cudaMemsetParams,
    cudaMipmappedArray_const_t, cudaMipmappedArray_t, cudaPitchedPtr, cudaPointerAttributes,
    cudaResourceDesc, cudaResourceViewDesc, cudaSharedMemConfig, cudaStreamCallback_t,
    cudaStreamCaptureMode, cudaStreamCaptureStatus, cudaSurfaceObject_t, cudaTextureDesc,
    cudaTextureObject_t, cudaUUID_t, cudaUserObject_t, dim3,
};

use crate::{cudart_fn, gen, print_refs};

use crate::handles::WCUdeviceptr as MutVoidStarDev;
use crate::handles::WCUpinnedptr as MutVoidStarPin;
use crate::handles::WCUfunction as cudaFunction_t;
use crate::handles::WCUunknownptr as MutVoidStarUnknown;
use crate::handles::{WCUstream as cudaStream_t, DEVICE_MEM, PINNED_MEM};

cudart_fn!(cudaDeviceReset() -> cudaError_t);
cudart_fn!(cudaDeviceSynchronize() -> cudaError_t);
cudart_fn!(cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t);
cudart_fn!(cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t, { print_refs!(pValue); });
cudart_fn!(cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, fmtDesc: *const cudaChannelFormatDesc, device: c_int) -> cudaError_t, { print_refs!(maxWidthInElements); });
cudart_fn!(cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t);
cudart_fn!(cudaDeviceGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> cudaError_t, { print_refs!(leastPriority, greatestPriority); });
cudart_fn!(cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t);
cudart_fn!(cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t);
cudart_fn!(cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t);
cudart_fn!(cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> cudaError_t, { print_refs!(device); });
cudart_fn!(cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> cudaError_t, { print_refs!(pciBusId); });
cudart_fn!(cudaIpcGetEventHandle(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaIpcOpenEventHandle(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t);
cudart_fn!(cudaIpcGetMemHandle(handle: *mut cudaIpcMemHandle_t, devPtr: *mut c_void) -> cudaError_t);
cudart_fn!(cudaIpcOpenMemHandle(devPtr: *mut *mut c_void, handle: cudaIpcMemHandle_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaIpcCloseMemHandle(devPtr: *mut c_void) -> cudaError_t);
cudart_fn!(cudaDeviceFlushGPUDirectRDMAWrites(target: cudaFlushGPUDirectRDMAWritesTarget, scope: cudaFlushGPUDirectRDMAWritesScope) -> cudaError_t);
cudart_fn!(cudaThreadExit() -> cudaError_t);
cudart_fn!(cudaThreadSynchronize() -> cudaError_t);
cudart_fn!(cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t);
cudart_fn!(cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t, { print_refs!(pValue); });
cudart_fn!(cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t);
cudart_fn!(cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t);
cudart_fn!(cudaGetLastError() -> cudaError_t);
cudart_fn!(cudaPeekAtLastError() -> cudaError_t);
cudart_fn!(cudaGetErrorName(error: cudaError_t) -> *const c_char);
cudart_fn!(cudaGetErrorString(error: cudaError_t) -> *const c_char);
cudart_fn!(cudaGetDeviceCount(count: *mut c_int) -> cudaError_t, { print_refs!(count); });
cudart_fn!(cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t);
cudart_fn!(cudaDeviceGetAttribute(value: *mut c_int, attr: cudaDeviceAttr, device: c_int) -> cudaError_t, { print_refs!(value); });
cudart_fn!(cudaDeviceGetDefaultMemPool(memPool: *mut cudaMemPool_t, device: c_int) -> cudaError_t);
cudart_fn!(cudaDeviceSetMemPool(device: c_int, memPool: cudaMemPool_t) -> cudaError_t);
cudart_fn!(cudaDeviceGetMemPool(memPool: *mut cudaMemPool_t, device: c_int) -> cudaError_t);
cudart_fn!(cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList: *mut c_void, device: c_int, flags: c_int) -> cudaError_t);
cudart_fn!(cudaDeviceGetP2PAttribute(value: *mut c_int, attr: cudaDeviceP2PAttr, srcDevice: c_int, dstDevice: c_int) -> cudaError_t, { print_refs!(value); });
cudart_fn!(cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> cudaError_t, { print_refs!(device); });
cudart_fn!(cudaInitDevice(device: c_int, deviceFlags: c_uint, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaSetDevice(device: c_int) -> cudaError_t);
cudart_fn!(cudaGetDevice(device: *mut c_int) -> cudaError_t, { print_refs!(device); });
cudart_fn!(cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> cudaError_t, { print_refs!(device_arr); });
cudart_fn!(cudaSetDeviceFlags(flags: c_uint) -> cudaError_t);
cudart_fn!(cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t, { print_refs!(flags); });
cudart_fn!(cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t);
cudart_fn!(cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: c_uint, priority: c_int) -> cudaError_t);
cudart_fn!(cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> cudaError_t, { print_refs!(priority); });
cudart_fn!(cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> cudaError_t, { print_refs!(flags); });
cudart_fn!(cudaStreamGetId(hStream: cudaStream_t, streamId: *mut c_ulonglong) -> cudaError_t, { print_refs!(streamId); });
cudart_fn!(cudaCtxResetPersistingL2Cache() -> cudaError_t);
cudart_fn!(cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t);
cudart_fn!(cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t);
cudart_fn!(cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userData: *mut c_void, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaStreamQuery(stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: *mut c_void, length: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode)-> cudaError_t);
cudart_fn!(cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t);
cudart_fn!(cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaStreamIsCapturing(stream: cudaStream_t, pCaptureStatus: *mut cudaStreamCaptureStatus) -> cudaError_t);
cudart_fn!(cudaStreamGetCaptureInfo_v2(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, numDependencies_out: *mut usize) -> cudaError_t, { print_refs!(id_out, numDependencies_out); });
cudart_fn!(cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, numDependencies: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaEventQuery(event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaEventDestroy(event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t, { print_refs!(ms); });
cudart_fn!(cudaImportExternalMemory(extMem_out: *mut cudaExternalMemory_t, memHandleDesc: *const cudaExternalMemoryHandleDesc) -> cudaError_t);
cudart_fn!(cudaExternalMemoryGetMappedBuffer(devPtr: *mut *mut c_void, extMem: cudaExternalMemory_t, bufferDesc: *const cudaExternalMemoryBufferDesc) -> cudaError_t);
cudart_fn!(cudaExternalMemoryGetMappedMipmappedArray(mipmap: *mut cudaMipmappedArray_t, extMem: cudaExternalMemory_t, mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc) -> cudaError_t);
cudart_fn!(cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t);
cudart_fn!(cudaImportExternalSemaphore(extSem_out: *mut cudaExternalSemaphore_t, semHandleDesc: *const cudaExternalSemaphoreHandleDesc) -> cudaError_t);
cudart_fn!(cudaSignalExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreSignalParams, numExtSems: c_uint, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaWaitExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreWaitParams, numExtSems: c_uint, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t);
cudart_fn!(cudaLaunchKernel(func: *const c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaLaunchKernelExC(config: *const cudaLaunchConfig_t, func: *const c_void, args: *mut *mut c_void) -> cudaError_t);
cudart_fn!(cudaLaunchCooperativeKernel(func: *const c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaLaunchCooperativeKernelMultiDevice(launchParamsList: *mut cudaLaunchParams, numDevices: c_uint, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaFuncSetCacheConfig(func: *const c_void, cacheConfig: cudaFuncCache) -> cudaError_t);
cudart_fn!(cudaFuncSetSharedMemConfig(func: *const c_void, config: cudaSharedMemConfig) -> cudaError_t);
cudart_fn!(cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t);
cudart_fn!(cudaFuncSetAttribute(func: *const c_void, attr: cudaFuncAttribute, value: c_int) -> cudaError_t);
cudart_fn!(cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t);
cudart_fn!(cudaSetDoubleForHost(d: *mut f64) -> cudaError_t);
cudart_fn!(cudaLaunchHostFunc(stream: cudaStream_t, fn_: cudaHostFn_t, userData: *mut c_void) -> cudaError_t);
cudart_fn!(cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut c_int, func: *const c_void, blockSize: c_int, dynamicSMemSize: usize) -> cudaError_t, { print_refs!(numBlocks); });
cudart_fn!(cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: *const c_void, numBlocks: c_int, blockSize: c_int) -> cudaError_t, { print_refs!(dynamicSmemSize); });
cudart_fn!(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut c_int, func: *const c_void, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) -> cudaError_t, { print_refs!(numBlocks); });
cudart_fn!(cudaOccupancyMaxPotentialClusterSize(clusterSize: *mut c_int, func: *const c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t, { print_refs!(clusterSize); });
cudart_fn!(cudaOccupancyMaxActiveClusters(numClusters: *mut c_int, func: *const c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t, { print_refs!(numClusters); });
cudart_fn!(cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaMalloc(devPtr: *mut MutVoidStarDev, size: usize) -> cudaError_t, {
    let base = unsafe { *devPtr }.0 as _;
    let mut device_mem = DEVICE_MEM.lock().unwrap();
    device_mem.insert(
        base..(base + size),
        (base, size),
    );
    drop(device_mem);
    print_refs!(devPtr);
});
cudart_fn!(cudaMallocHost(ptr: *mut MutVoidStarPin, size: usize) -> cudaError_t, {
    let base = unsafe { *ptr }.0 as _;
    let mut pinned_mem = PINNED_MEM.lock().unwrap();
    pinned_mem.insert(
        base..(base + size),
        (base, size),
    );
    drop(pinned_mem);
    print_refs!(ptr);
});
cudart_fn!(cudaMallocPitch(devPtr: *mut *mut c_void, pitch: *mut usize, width: usize, height: usize) -> cudaError_t, { print_refs!(pitch); });
cudart_fn!(cudaMallocArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, width: usize, height: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaFree(devPtr: MutVoidStarDev) -> cudaError_t, {
    let base = devPtr.0 as _;
    let mut device_mem = DEVICE_MEM.lock().unwrap();
    let (_, size) = *device_mem.get(&base).unwrap();
    device_mem.remove(base..(base + size));
});
cudart_fn!(cudaFreeHost(ptr: MutVoidStarPin) -> cudaError_t, {
    let base = ptr.0 as _;
    let mut pinned_mem = PINNED_MEM.lock().unwrap();
    let (_, size) = *pinned_mem.get(&base).unwrap();
    pinned_mem.remove(base..(base + size));
});
cudart_fn!(cudaFreeArray(array: cudaArray_t) -> cudaError_t);
cudart_fn!(cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t);
cudart_fn!(cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaHostUnregister(ptr: *mut c_void) -> cudaError_t);
cudart_fn!(cudaHostGetDevicePointer(pDevice: *mut *mut c_void, pHost: *mut c_void, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaHostGetFlags(pFlags: *mut c_uint, pHost: *mut c_void) -> cudaError_t, { print_refs!(pFlags); });
cudart_fn!(cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t);
cudart_fn!(cudaMalloc3DArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaMallocMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, numLevels: c_uint, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaGetMipmappedArrayLevel(levelArray: *mut cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: c_uint) -> cudaError_t);
cudart_fn!(cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t);
cudart_fn!(cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t);
cudart_fn!(cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpy3DPeerAsync(p: *const cudaMemcpy3DPeerParms, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t, { print_refs!(free, total); });
cudart_fn!(cudaArrayGetInfo(desc: *mut cudaChannelFormatDesc, extent: *mut cudaExtent, flags: *mut c_uint, array: cudaArray_t) -> cudaError_t, { print_refs!(flags); });
cudart_fn!(cudaArrayGetPlane(pPlaneArray: *mut cudaArray_t, hArray: cudaArray_t, planeIdx: c_uint) -> cudaError_t);
cudart_fn!(cudaArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, array: cudaArray_t, device: c_int) -> cudaError_t);
cudart_fn!(cudaMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, mipmap: cudaMipmappedArray_t, device: c_int) -> cudaError_t);
cudart_fn!(cudaArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, array: cudaArray_t) -> cudaError_t);
cudart_fn!(cudaMipmappedArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, mipmap: cudaMipmappedArray_t) -> cudaError_t);
cudart_fn!(cudaMemcpy(dst: MutVoidStarUnknown, src: MutVoidStarUnknown, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyPeer(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize) -> cudaError_t);
cudart_fn!(cudaMemcpy2D(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpy2DToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpy2DFromArray(dst: *mut c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpy2DArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyToSymbol(symbol: *const c_void, src: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyFromSymbol(dst: *mut c_void, symbol: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpyPeerAsync(dst: *mut c_void, dstDevice: c_int, src: *const c_void, srcDevice: c_int, count: usize, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpy2DAsync(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpy2DToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpy2DFromArrayAsync(dst: *mut c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpyToSymbolAsync(symbol: *const c_void, src: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpyFromSymbolAsync(dst: *mut c_void, symbol: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> cudaError_t);
cudart_fn!(cudaMemset2D(devPtr: *mut c_void, pitch: usize, value: c_int, width: usize, height: usize) -> cudaError_t);
cudart_fn!(cudaMemset3D(pitchedDevPtr: cudaPitchedPtr, value: c_int, extent: cudaExtent) -> cudaError_t);
cudart_fn!(cudaMemsetAsync(devPtr: *mut c_void, value: c_int, count: usize, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemset2DAsync(devPtr: *mut c_void, pitch: usize, value: c_int, width: usize, height: usize, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemset3DAsync(pitchedDevPtr: cudaPitchedPtr, value: c_int, extent: cudaExtent, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaGetSymbolAddress(devPtr: *mut *mut c_void, symbol: *const c_void) -> cudaError_t);
cudart_fn!(cudaGetSymbolSize(size: *mut usize, symbol: *const c_void) -> cudaError_t, { print_refs!(size); });
cudart_fn!(cudaMemPrefetchAsync(devPtr: *const c_void, count: usize, dstDevice: c_int, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemPrefetchAsync_v2(devPtr: *const c_void, count: usize, location: cudaMemLocation, flags: c_uint, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemAdvise(devPtr: *const c_void, count: usize, advice: cudaMemoryAdvise, device: c_int) -> cudaError_t);
cudart_fn!(cudaMemAdvise_v2(devPtr: *const c_void, count: usize, advice: cudaMemoryAdvise, location: cudaMemLocation) -> cudaError_t);
cudart_fn!(cudaMemRangeGetAttribute(data: *mut c_void, dataSize: usize, attribute: cudaMemRangeAttribute, devPtr: *const c_void, count: usize) -> cudaError_t);
cudart_fn!(cudaMemRangeGetAttributes(data: *mut *mut c_void, dataSizes: *mut usize, attributes: *mut cudaMemRangeAttribute, numAttributes: usize, devPtr: *const c_void, count: usize) -> cudaError_t, { print_refs!(dataSizes); });
cudart_fn!(cudaMemcpyToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyFromArray(dst: *mut c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaMemcpyToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemcpyFromArrayAsync(dst: *mut c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMallocAsync(devPtr: *mut *mut c_void, size: usize, hStream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaFreeAsync(devPtr: *mut c_void, hStream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t);
cudart_fn!(cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut c_void) -> cudaError_t);
cudart_fn!(cudaMemPoolGetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut c_void) -> cudaError_t);
cudart_fn!(cudaMemPoolSetAccess(memPool: cudaMemPool_t, descList: *const cudaMemAccessDesc, count: usize) -> cudaError_t);
cudart_fn!(cudaMemPoolGetAccess(flags: *mut cudaMemAccessFlags, memPool: cudaMemPool_t, location: *mut cudaMemLocation) -> cudaError_t);
cudart_fn!(cudaMemPoolCreate(memPool: *mut cudaMemPool_t, poolProps: *const cudaMemPoolProps) -> cudaError_t);
cudart_fn!(cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t);
cudart_fn!(cudaMallocFromPoolAsync(ptr: *mut *mut c_void, size: usize, memPool: cudaMemPool_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaMemPoolExportToShareableHandle(shareableHandle: *mut c_void, memPool: cudaMemPool_t, handleType: cudaMemAllocationHandleType, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaMemPoolImportFromShareableHandle(memPool: *mut cudaMemPool_t, shareableHandle: *mut c_void, handleType: cudaMemAllocationHandleType, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaMemPoolExportPointer(exportData: *mut cudaMemPoolPtrExportData, ptr: *mut c_void) -> cudaError_t);
cudart_fn!(cudaMemPoolImportPointer(ptr: *mut *mut c_void, memPool: cudaMemPool_t, exportData: *mut cudaMemPoolPtrExportData) -> cudaError_t);
cudart_fn!(cudaPointerGetAttributes(attributes: *mut cudaPointerAttributes, ptr: *const c_void) -> cudaError_t);
cudart_fn!(cudaDeviceCanAccessPeer(canAccessPeer: *mut c_int, device: c_int, peerDevice: c_int) -> cudaError_t, { print_refs!(canAccessPeer); });
cudart_fn!(cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaDeviceDisablePeerAccess(peerDevice: c_int) -> cudaError_t);
cudart_fn!(cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t);
cudart_fn!(cudaGraphicsResourceSetMapFlags(resource: cudaGraphicsResource_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphicsMapResources(count: c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaGraphicsUnmapResources(count: c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaGraphicsResourceGetMappedPointer(devPtr: *mut *mut c_void, size: *mut usize, resource: cudaGraphicsResource_t) -> cudaError_t, { print_refs!(size); });
cudart_fn!(cudaGraphicsSubResourceGetMappedArray(array: *mut cudaArray_t, resource: cudaGraphicsResource_t, arrayIndex: c_uint, mipLevel: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, resource: cudaGraphicsResource_t) -> cudaError_t);
cudart_fn!(cudaGetChannelDesc(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t);
cudart_fn!(cudaCreateChannelDesc(x: c_int, y: c_int, z: c_int, w: c_int, f: cudaChannelFormatKind) -> cudaChannelFormatDesc);
cudart_fn!(cudaCreateTextureObject(pTexObject: *mut cudaTextureObject_t, pResDesc: *const cudaResourceDesc, pTexDesc: *const cudaTextureDesc, pResViewDesc: *const cudaResourceViewDesc) -> cudaError_t);
cudart_fn!(cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t);
cudart_fn!(cudaGetTextureObjectResourceDesc(pResDesc: *mut cudaResourceDesc, texObject: cudaTextureObject_t) -> cudaError_t);
cudart_fn!(cudaGetTextureObjectTextureDesc(pTexDesc: *mut cudaTextureDesc, texObject: cudaTextureObject_t) -> cudaError_t);
cudart_fn!(cudaGetTextureObjectResourceViewDesc(pResViewDesc: *mut cudaResourceViewDesc, texObject: cudaTextureObject_t) -> cudaError_t);
cudart_fn!(cudaCreateSurfaceObject(pSurfObject: *mut cudaSurfaceObject_t, pResDesc: *const cudaResourceDesc) -> cudaError_t);
cudart_fn!(cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t);
cudart_fn!(cudaGetSurfaceObjectResourceDesc(pResDesc: *mut cudaResourceDesc, surfObject: cudaSurfaceObject_t) -> cudaError_t);
cudart_fn!(cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t, { print_refs!(driverVersion); });
cudart_fn!(cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t, { print_refs!(runtimeVersion); });
cudart_fn!(cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphAddKernelNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphKernelNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaKernelNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphKernelNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphKernelNodeCopyAttributes(hSrc: cudaGraphNode_t, hDst: cudaGraphNode_t) -> cudaError_t);
cudart_fn!(cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t);
cudart_fn!(cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t);
cudart_fn!(cudaGraphAddMemcpyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pCopyParams: *const cudaMemcpy3DParms) -> cudaError_t);
cudart_fn!(cudaGraphAddMemcpyNodeToSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, symbol: *const c_void, src: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphAddMemcpyNodeFromSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut c_void, symbol: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphAddMemcpyNode1D(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphMemcpyNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemcpy3DParms) -> cudaError_t);
cudart_fn!(cudaGraphMemcpyNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t);
cudart_fn!(cudaGraphMemcpyNodeSetParamsToSymbol(node: cudaGraphNode_t, symbol: *const c_void, src: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphMemcpyNodeSetParamsFromSymbol(node: cudaGraphNode_t, dst: *mut c_void, symbol: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphMemcpyNodeSetParams1D(node: cudaGraphNode_t, dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphAddMemsetNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pMemsetParams: *const cudaMemsetParams) -> cudaError_t);
cudart_fn!(cudaGraphMemsetNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemsetParams) -> cudaError_t);
cudart_fn!(cudaGraphMemsetNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t);
cudart_fn!(cudaGraphAddHostNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaHostNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphHostNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaHostNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphHostNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphAddChildGraphNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, childGraph: cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaGraphAddEmptyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t);
cudart_fn!(cudaGraphAddEventRecordNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphEventRecordNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphAddEventWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphEventWaitNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t)-> cudaError_t);
cudart_fn!(cudaGraphAddExternalSemaphoresSignalNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExternalSemaphoresSignalNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreSignalNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExternalSemaphoresSignalNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphAddExternalSemaphoresWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExternalSemaphoresWaitNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreWaitNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExternalSemaphoresWaitNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphAddMemAllocNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaMemAllocNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphMemAllocNodeGetParams(node: cudaGraphNode_t, params_out: *mut cudaMemAllocNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphAddMemFreeNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dptr: *mut c_void) -> cudaError_t);
cudart_fn!(cudaGraphMemFreeNodeGetParams(node: cudaGraphNode_t, dptr_out: *mut c_void) -> cudaError_t);
cudart_fn!(cudaDeviceGraphMemTrim(device: c_int) -> cudaError_t);
cudart_fn!(cudaDeviceGetGraphMemAttribute(device: c_int, attr: cudaGraphMemAttributeType, value: *mut c_void) -> cudaError_t);
cudart_fn!(cudaDeviceSetGraphMemAttribute(device: c_int, attr: cudaGraphMemAttributeType, value: *mut c_void) -> cudaError_t);
cudart_fn!(cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t)-> cudaError_t);
cudart_fn!(cudaGraphNodeFindInClone(pNode: *mut cudaGraphNode_t, originalNode: cudaGraphNode_t, clonedGraph: cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaGraphNodeGetType(node: cudaGraphNode_t, pType: *mut cudaGraphNodeType) -> cudaError_t);
cudart_fn!(cudaGraphGetNodes(graph: cudaGraph_t, nodes: *mut cudaGraphNode_t, numNodes: *mut usize) -> cudaError_t, { print_refs!(numNodes); });
cudart_fn!(cudaGraphGetRootNodes(graph: cudaGraph_t, pRootNodes: *mut cudaGraphNode_t, pNumRootNodes: *mut usize) -> cudaError_t, { print_refs!(pNumRootNodes); });
cudart_fn!(cudaGraphGetEdges(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, numEdges: *mut usize) -> cudaError_t, { print_refs!(numEdges); });
cudart_fn!(cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, pNumDependencies: *mut usize) -> cudaError_t, { print_refs!(pNumDependencies); });
cudart_fn!(cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, pNumDependentNodes: *mut usize) -> cudaError_t, { print_refs!(pNumDependentNodes); });
cudart_fn!(cudaGraphAddDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t);
cudart_fn!(cudaGraphRemoveDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t);
cudart_fn!(cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t);
cudart_fn!(cudaGraphInstantiate(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) -> cudaError_t);
cudart_fn!(cudaGraphInstantiateWithFlags(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: c_ulonglong) -> cudaError_t);
cudart_fn!(cudaGraphInstantiateWithParams(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, instantiateParams: *mut cudaGraphInstantiateParams) -> cudaError_t);
cudart_fn!(cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: *mut c_ulonglong) -> cudaError_t, { print_refs!(flags); });
cudart_fn!(cudaGraphExecKernelNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExecMemcpyNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t);
cudart_fn!(cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, symbol: *const c_void, src: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut c_void, symbol: *const c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphExecMemcpyNodeSetParams1D(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t);
cudart_fn!(cudaGraphExecMemsetNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t);
cudart_fn!(cudaGraphExecHostNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExecChildGraphNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, childGraph: cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaGraphExecEventRecordNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphExecEventWaitNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t);
cudart_fn!(cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphNodeSetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphNodeGetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: *mut c_uint) -> cudaError_t, { print_refs!(isEnabled); });
cudart_fn!(cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, resultInfo: *mut cudaGraphExecUpdateResultInfo) -> cudaError_t);
cudart_fn!(cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t);
cudart_fn!(cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t);
cudart_fn!(cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t);
cudart_fn!(cudaGraphDebugDotPrint(graph: cudaGraph_t, path: *const c_char, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaUserObjectCreate(object_out: *mut cudaUserObject_t, ptr: *mut c_void, destroy: cudaHostFn_t, initialRefcount: c_uint, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaUserObjectRetain(object: cudaUserObject_t, count: c_uint) -> cudaError_t);
cudart_fn!(cudaUserObjectRelease(object: cudaUserObject_t, count: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphRetainUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint, flags: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphReleaseUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: c_uint) -> cudaError_t);
cudart_fn!(cudaGraphAddNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphNodeSetParams(node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t);
cudart_fn!(cudaGraphExecNodeSetParams(graphExec: cudaGraphExec_t, node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t);
cudart_fn!(cudaGetDriverEntryPoint(symbol: *const c_char, funcPtr: *mut *mut c_void, flags: c_ulonglong, driverStatus: *mut cudaDriverEntryPointQueryResult) -> cudaError_t);
cudart_fn!(cudaGetExportTable(ppExportTable: *mut *const c_void, pExportTableId: *const cudaUUID_t) -> cudaError_t);
cudart_fn!(cudaGetFuncBySymbol(functionPtr: *mut cudaFunction_t, symbolPtr: *const c_void) -> cudaError_t);
cudart_fn!(cudaGetKernel(kernelPtr: *mut cudaKernel_t, entryFuncAddr: *const c_void) -> cudaError_t);
