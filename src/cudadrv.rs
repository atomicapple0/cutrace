use std::ffi::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_ushort, c_void, CStr};

use cudarc::driver::sys::{
    cuuint32_t, cuuint64_t, CUaddress_mode, CUarray, CUarrayMapInfo, CUarray_format,
    CUcoredumpSettings, CUdevice, CUdevice_P2PAttribute, CUdevice_attribute, CUdevprop,
    CUdriverProcAddressQueryResult, CUevent, CUexecAffinityParam, CUexecAffinityType,
    CUexternalMemory, CUexternalSemaphore, CUfilter_mode, CUflushGPUDirectRDMAWritesScope,
    CUflushGPUDirectRDMAWritesTarget, CUfunc_cache, CUfunction_attribute, CUgraph, CUgraphExec,
    CUgraphExecUpdateResultInfo, CUgraphMem_attribute, CUgraphNode, CUgraphNodeParams,
    CUgraphNodeType, CUgraphicsResource, CUhostFn, CUipcEventHandle, CUipcMemHandle,
    CUjitInputType, CUjit_option, CUkernel, CUkernelNodeAttrID, CUkernelNodeAttrValue,
    CUlaunchConfig, CUlibrary, CUlibraryOption, CUlimit, CUlinkState, CUmemAccessDesc,
    CUmemAccess_flags, CUmemAllocationGranularity_flags, CUmemAllocationHandleType,
    CUmemAllocationProp, CUmemGenericAllocationHandle, CUmemLocation, CUmemPoolProps,
    CUmemPoolPtrExportData, CUmemPool_attribute, CUmemRangeHandleType, CUmem_advise,
    CUmem_range_attribute, CUmemoryPool, CUmipmappedArray, CUmoduleLoadingMode,
    CUmulticastGranularity_flags, CUmulticastObjectProp, CUoccupancyB2DSize, CUoutput_mode,
    CUpointer_attribute, CUresult, CUsharedconfig, CUstreamAttrID, CUstreamAttrValue,
    CUstreamBatchMemOpParams, CUstreamCallback, CUstreamCaptureMode, CUstreamCaptureStatus,
    CUsurfObject, CUsurfref, CUtensorMap, CUtensorMapDataType, CUtensorMapFloatOOBfill,
    CUtensorMapInterleave, CUtensorMapL2promotion, CUtensorMapSwizzle, CUtexObject, CUtexref,
    CUuserObject, CUuuid, CUDA_ARRAY3D_DESCRIPTOR, CUDA_ARRAY_DESCRIPTOR,
    CUDA_ARRAY_MEMORY_REQUIREMENTS, CUDA_ARRAY_SPARSE_PROPERTIES, CUDA_BATCH_MEM_OP_NODE_PARAMS,
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS, CUDA_EXT_SEM_WAIT_NODE_PARAMS, CUDA_GRAPH_INSTANTIATE_PARAMS,
    CUDA_HOST_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS, CUDA_LAUNCH_PARAMS, CUDA_MEMCPY2D,
    CUDA_MEMCPY3D, CUDA_MEMCPY3D_PEER, CUDA_MEMSET_NODE_PARAMS, CUDA_MEM_ALLOC_NODE_PARAMS,
    CUDA_RESOURCE_DESC, CUDA_RESOURCE_VIEW_DESC, CUDA_TEXTURE_DESC,
};

use crate::{
    cudadrv_fn,
    fatbin::{is_fatbin, parse_func_sigs_from_fatbin, CUfuncParamSize},
    gen,
    handles::{FatBinPtr, WCUunknownptr, CUFUNC_INFO, CUFUNC_SIG, CUMOD_FATBIN, DEVICE_MEM},
    print_refs,
};

use crate::handles::WCUcontext as CUcontext;
use crate::handles::WCUdeviceptr as CUdeviceptr;
use crate::handles::WCUfunction as CUfunction;
use crate::handles::WCUmodule as CUmodule;
use crate::handles::WCUstream as CUstream;
use crate::handles::WConstCharStar as ConstCharStar;

cudadrv_fn!(cuGetErrorString(error: CUresult, pStr: *mut ConstCharStar) -> CUresult);
cudadrv_fn!(cuGetErrorName(error: CUresult, pStr: *mut ConstCharStar) -> CUresult);
cudadrv_fn!(cuInit(Flags: c_uint) -> CUresult);
cudadrv_fn!(cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult, { print_refs!(driverVersion); });
cudadrv_fn!(cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult, { print_refs!(device); });
cudadrv_fn!(cuDeviceGetCount(count: *mut c_int) -> CUresult, { print_refs!(count); });
cudadrv_fn!(cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult, { print_refs!(name); });
cudadrv_fn!(cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult, { print_refs!(uuid); });
cudadrv_fn!(cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult, { print_refs!(uuid); });
cudadrv_fn!(cuDeviceGetLuid(luid: *mut c_char, deviceNodeMask: *mut c_uint, dev: CUdevice) -> CUresult, { print_refs!(luid, deviceNodeMask); });
cudadrv_fn!(cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult, { print_refs!(bytes); });
cudadrv_fn!(cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, format: CUarray_format, numChannels: c_uint, dev: CUdevice) -> CUresult, { print_refs!(maxWidthInElements); });
cudadrv_fn!(cuDeviceGetAttribute(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult, { print_refs!(pi); });
cudadrv_fn!(cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList: *mut c_void, dev: CUdevice, flags: c_int) -> CUresult);
cudadrv_fn!(cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult);
cudadrv_fn!(cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuDeviceGetExecAffinitySupport(pi: *mut c_int, type_: CUexecAffinityType, dev: CUdevice) -> CUresult, { print_refs!(pi); });
cudadrv_fn!(cuFlushGPUDirectRDMAWrites(target: CUflushGPUDirectRDMAWritesTarget, scope: CUflushGPUDirectRDMAWritesScope) -> CUresult);
cudadrv_fn!(cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult, { print_refs!(major, minor); });
cudadrv_fn!(cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult);
cudadrv_fn!(cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) -> CUresult);
cudadrv_fn!(cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult, { print_refs!(flags, active); });
cudadrv_fn!(cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult);
cudadrv_fn!(cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuCtxCreate_v3(pctx: *mut CUcontext, paramsArray: *mut CUexecAffinityParam, numParams: c_int, flags: c_uint, dev: CUdevice) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuCtxDestroy_v2(ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuCtxSetCurrent(ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuCtxGetDevice(device: *mut CUdevice) -> CUresult, { print_refs!(device); });
cudadrv_fn!(cuCtxGetFlags(flags: *mut c_uint) -> CUresult, { print_refs!(flags); });
cudadrv_fn!(cuCtxSetFlags(flags: c_uint) -> CUresult);
cudadrv_fn!(cuCtxGetId(ctx: CUcontext, ctxId: *mut c_ulonglong) -> CUresult, { print_refs!(ctxId); });
cudadrv_fn!(cuCtxSynchronize() -> CUresult);
cudadrv_fn!(cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult);
cudadrv_fn!(cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult, { print_refs!(pvalue); });
cudadrv_fn!(cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult, { print_refs!(pconfig); });
cudadrv_fn!(cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult);
cudadrv_fn!(cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult, { print_refs!(pConfig); });
cudadrv_fn!(cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult);
cudadrv_fn!(cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult, { print_refs!(version); });
cudadrv_fn!(cuCtxGetStreamPriorityRange(leastPriority: *mut c_int, greatestPriority: *mut c_int) -> CUresult, { print_refs!(leastPriority, greatestPriority); });
cudadrv_fn!(cuCtxResetPersistingL2Cache() -> CUresult);
cudadrv_fn!(cuCtxGetExecAffinity(pExecAffinity: *mut CUexecAffinityParam, type_: CUexecAffinityType) -> CUresult);
cudadrv_fn!(cuCtxAttach(pctx: *mut CUcontext, flags: c_uint) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuCtxDetach(ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuModuleLoad(module: *mut CUmodule, fname: ConstCharStar) -> CUresult, { print_refs!(module); });
cudadrv_fn!(cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult, {
    print_refs!(module);
    if is_fatbin(image) {
        CUMOD_FATBIN
            .lock()
            .unwrap()
            .insert(unsafe { *module }, FatBinPtr(image));
    };
});
cudadrv_fn!(cuModuleLoadDataEx(module: *mut CUmodule, image: *const c_void, numOptions: c_uint, options: *mut CUjit_option, optionValues: *mut *mut c_void) -> CUresult, { print_refs!(module); });
cudadrv_fn!(cuModuleLoadFatBinary(module: *mut CUmodule, fatCubin: *const c_void) -> CUresult, { print_refs!(module); });
cudadrv_fn!(cuModuleUnload(hmod: CUmodule) -> CUresult);
cudadrv_fn!(cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult, { print_refs!(mode); });
cudadrv_fn!(cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: ConstCharStar) -> CUresult, {
    let name = unsafe { CStr::from_ptr(name.0) }.to_str().unwrap();
    CUFUNC_INFO
        .lock()
        .unwrap()
        .insert(unsafe { *hfunc }, (hmod, name.to_owned()));
    print_refs!(hfunc);
});
cudadrv_fn!(cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: ConstCharStar) -> CUresult, { print_refs!(dptr, bytes); });
cudadrv_fn!(cuLinkCreate_v2(numOptions: c_uint, options: *mut CUjit_option, optionValues: *mut *mut c_void, stateOut: *mut CUlinkState) -> CUresult);
cudadrv_fn!(cuLinkAddData_v2(state: CUlinkState, type_: CUjitInputType, data: *mut c_void, size: usize, name: ConstCharStar, numOptions: c_uint, options: *mut CUjit_option, optionValues: *mut *mut c_void) -> CUresult);
cudadrv_fn!(cuLinkAddFile_v2(state: CUlinkState, type_: CUjitInputType, path: ConstCharStar, numOptions: c_uint, options: *mut CUjit_option, optionValues: *mut *mut c_void) -> CUresult);
cudadrv_fn!(cuLinkComplete(state: CUlinkState, cubinOut: *mut *mut c_void, sizeOut: *mut usize) -> CUresult, { print_refs!(sizeOut); });
cudadrv_fn!(cuLinkDestroy(state: CUlinkState) -> CUresult);
cudadrv_fn!(cuModuleGetTexRef(pTexRef: *mut CUtexref, hmod: CUmodule, name: ConstCharStar) -> CUresult);
cudadrv_fn!(cuModuleGetSurfRef(pSurfRef: *mut CUsurfref, hmod: CUmodule, name: ConstCharStar) -> CUresult);
cudadrv_fn!(cuLibraryLoadData(library: *mut CUlibrary, code: *const c_void, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut c_void, numJitOptions: c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut c_void, numLibraryOptions: c_uint) -> CUresult, { print_refs!(library); });
cudadrv_fn!(cuLibraryLoadFromFile(library: *mut CUlibrary, fileName: ConstCharStar, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut c_void, numJitOptions: c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut c_void, numLibraryOptions: c_uint) -> CUresult, { print_refs!(library); });
cudadrv_fn!(cuLibraryUnload(library: CUlibrary) -> CUresult);
cudadrv_fn!(cuLibraryGetKernel(pKernel: *mut CUkernel, library: CUlibrary, name: ConstCharStar) -> CUresult, { print_refs!(pKernel); });
cudadrv_fn!(cuLibraryGetModule(pMod: *mut CUmodule, library: CUlibrary) -> CUresult, { print_refs!(pMod); });
cudadrv_fn!(cuKernelGetFunction(pFunc: *mut CUfunction, kernel: CUkernel) -> CUresult, { print_refs!(pFunc); });
cudadrv_fn!(cuLibraryGetGlobal(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: ConstCharStar) -> CUresult, { print_refs!(dptr, bytes); });
cudadrv_fn!(cuLibraryGetManaged(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: ConstCharStar) -> CUresult, { print_refs!(dptr, bytes); });
cudadrv_fn!(cuLibraryGetUnifiedFunction(fptr: *mut *mut c_void, library: CUlibrary, symbol: ConstCharStar) -> CUresult);
cudadrv_fn!(cuKernelGetAttribute(pi: *mut c_int, attrib: CUfunction_attribute, kernel: CUkernel, dev: CUdevice) -> CUresult, { print_refs!(pi); });
cudadrv_fn!(cuKernelSetAttribute(attrib: CUfunction_attribute, val: c_int, kernel: CUkernel, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuKernelSetCacheConfig(kernel: CUkernel, config: CUfunc_cache, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult, { print_refs!(free, total); });
cudadrv_fn!(cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult, { print_refs!(dptr); });
cudadrv_fn!(cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, pPitch: *mut usize, WidthInBytes: usize, Height: usize, ElementSizeBytes: c_uint) -> CUresult, { print_refs!(dptr, pPitch); });
cudadrv_fn!(cuMemFree_v2(dptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult, { print_refs!(pbase, psize); });
cudadrv_fn!(cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult);
cudadrv_fn!(cuMemFreeHost(p: *mut c_void) -> CUresult);
cudadrv_fn!(cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut c_void, Flags: c_uint) -> CUresult, { print_refs!(pdptr); });
cudadrv_fn!(cuMemHostGetFlags(pFlags: *mut c_uint, p: *mut c_void) -> CUresult, { print_refs!(pFlags); });
cudadrv_fn!(cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) -> CUresult, { print_refs!(dptr); });
cudadrv_fn!(cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: ConstCharStar) -> CUresult, { print_refs!(dev); });
cudadrv_fn!(cuDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, dev: CUdevice) -> CUresult, { print_refs!(pciBusId); });
cudadrv_fn!(cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult, { print_refs!(pHandle); });
cudadrv_fn!(cuIpcOpenEventHandle(phEvent: *mut CUevent, handle: CUipcEventHandle) -> CUresult, { print_refs!(phEvent); });
cudadrv_fn!(cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuIpcOpenMemHandle_v2(pdptr: *mut CUdeviceptr, handle: CUipcMemHandle, Flags: c_uint) -> CUresult, { print_refs!(pdptr); });
cudadrv_fn!(cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuMemHostUnregister(p: *mut c_void) -> CUresult);
cudadrv_fn!(cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyPeer(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyDtoH_v2(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyDtoA_v2(dstArray: CUarray, dstOffset: usize, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyAtoD_v2(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyHtoA_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const c_void, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyAtoH_v2(dstHost: *mut c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpyAtoA_v2(dstArray: CUarray, dstOffset: usize, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult);
cudadrv_fn!(cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult);
cudadrv_fn!(cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult);
cudadrv_fn!(cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult);
cudadrv_fn!(cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult);
cudadrv_fn!(cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyPeerAsync(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyHtoDAsync_v2(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyDtoHAsync_v2(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyDtoDAsync_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyHtoAAsync_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const c_void, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpyAtoHAsync_v2(dstHost: *mut c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpy3DAsync_v2(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemcpy3DPeerAsync(pCopy: *const CUDA_MEMCPY3D_PEER, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: c_uchar, N: usize) -> CUresult);
cudadrv_fn!(cuMemsetD16_v2(dstDevice: CUdeviceptr, us: c_ushort, N: usize) -> CUresult);
cudadrv_fn!(cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: c_uint, N: usize) -> CUresult);
cudadrv_fn!(cuMemsetD2D8_v2(dstDevice: CUdeviceptr, dstPitch: usize, uc: c_uchar, Width: usize, Height: usize) -> CUresult);
cudadrv_fn!(cuMemsetD2D16_v2(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize) -> CUresult);
cudadrv_fn!(cuMemsetD2D32_v2(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize) -> CUresult);
cudadrv_fn!(cuMemsetD8Async(dstDevice: CUdeviceptr, uc: c_uchar, N: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD16Async(dstDevice: CUdeviceptr, us: c_ushort, N: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD32Async(dstDevice: CUdeviceptr, ui: c_uint, N: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD2D8Async(dstDevice: CUdeviceptr, dstPitch: usize, uc: c_uchar, Width: usize, Height: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD2D16Async(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemsetD2D32Async(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuArrayCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR) -> CUresult);
cudadrv_fn!(cuArrayGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) -> CUresult);
cudadrv_fn!(cuArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, array: CUarray) -> CUresult);
cudadrv_fn!(cuMipmappedArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, mipmap: CUmipmappedArray) -> CUresult);
cudadrv_fn!(cuArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, array: CUarray, device: CUdevice) -> CUresult);
cudadrv_fn!(cuMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, mipmap: CUmipmappedArray, device: CUdevice) -> CUresult);
cudadrv_fn!(cuArrayGetPlane(pPlaneArray: *mut CUarray, hArray: CUarray, planeIdx: c_uint) -> CUresult);
cudadrv_fn!(cuArrayDestroy(hArray: CUarray) -> CUresult);
cudadrv_fn!(cuArray3DCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR) -> CUresult);
cudadrv_fn!(cuArray3DGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) -> CUresult);
cudadrv_fn!(cuMipmappedArrayCreate(pHandle: *mut CUmipmappedArray, pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: c_uint) -> CUresult);
cudadrv_fn!(cuMipmappedArrayGetLevel(pLevelArray: *mut CUarray, hMipmappedArray: CUmipmappedArray, level: c_uint) -> CUresult);
cudadrv_fn!(cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult);
cudadrv_fn!(cuMemGetHandleForAddressRange(handle: *mut c_void, dptr: CUdeviceptr, size: usize, handleType: CUmemRangeHandleType, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemAddressReserve(ptr: *mut CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: c_ulonglong) -> CUresult, { print_refs!(ptr); });
cudadrv_fn!(cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult);
cudadrv_fn!(cuMemCreate(handle: *mut CUmemGenericAllocationHandle, size: usize, prop: *const CUmemAllocationProp, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult);
cudadrv_fn!(cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemMapArrayAsync(mapInfoList: *mut CUarrayMapInfo, count: c_uint, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult);
cudadrv_fn!(cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) -> CUresult);
cudadrv_fn!(cuMemGetAccess(flags: *mut c_ulonglong, location: *const CUmemLocation, ptr: CUdeviceptr) -> CUresult, { print_refs!(flags); });
cudadrv_fn!(cuMemExportToShareableHandle(shareableHandle: *mut c_void, handle: CUmemGenericAllocationHandle, handleType: CUmemAllocationHandleType, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemImportFromShareableHandle(handle: *mut CUmemGenericAllocationHandle, osHandle: *mut c_void, shHandleType: CUmemAllocationHandleType) -> CUresult);
cudadrv_fn!(cuMemGetAllocationGranularity(granularity: *mut usize, prop: *const CUmemAllocationProp, option: CUmemAllocationGranularity_flags) -> CUresult, { print_refs!(granularity); });
cudadrv_fn!(cuMemGetAllocationPropertiesFromHandle(prop: *mut CUmemAllocationProp, handle: CUmemGenericAllocationHandle) -> CUresult);
cudadrv_fn!(cuMemRetainAllocationHandle(handle: *mut CUmemGenericAllocationHandle, addr: *mut c_void) -> CUresult);
cudadrv_fn!(cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult, { print_refs!(dptr); });
cudadrv_fn!(cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) -> CUresult);
cudadrv_fn!(cuMemPoolSetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void) -> CUresult);
cudadrv_fn!(cuMemPoolGetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void) -> CUresult);
cudadrv_fn!(cuMemPoolSetAccess(pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize) -> CUresult);
cudadrv_fn!(cuMemPoolGetAccess(flags: *mut CUmemAccess_flags, memPool: CUmemoryPool, location: *mut CUmemLocation) -> CUresult);
cudadrv_fn!(cuMemPoolCreate(pool: *mut CUmemoryPool, poolProps: *const CUmemPoolProps) -> CUresult);
cudadrv_fn!(cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult);
cudadrv_fn!(cuMemAllocFromPoolAsync(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hStream: CUstream) -> CUresult, { print_refs!(dptr); });
cudadrv_fn!(cuMemPoolExportToShareableHandle(handle_out: *mut c_void, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemPoolImportFromShareableHandle(pool_out: *mut CUmemoryPool, handle: *mut c_void, handleType: CUmemAllocationHandleType, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMemPoolExportPointer(shareData_out: *mut CUmemPoolPtrExportData, ptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuMemPoolImportPointer(ptr_out: *mut CUdeviceptr, pool: CUmemoryPool, shareData: *mut CUmemPoolPtrExportData) -> CUresult, { print_refs!(ptr_out); });
cudadrv_fn!(cuMulticastCreate(mcHandle: *mut CUmemGenericAllocationHandle, prop: *const CUmulticastObjectProp) -> CUresult);
cudadrv_fn!(cuMulticastAddDevice(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice) -> CUresult);
cudadrv_fn!(cuMulticastBindMem(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMulticastBindAddr(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuMulticastUnbind(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, size: usize) -> CUresult);
cudadrv_fn!(cuMulticastGetGranularity(granularity: *mut usize, prop: *const CUmulticastObjectProp, option: CUmulticastGranularity_flags) -> CUresult, { print_refs!(granularity); });
cudadrv_fn!(cuPointerGetAttribute(data: *mut c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemPrefetchAsync_v2(devPtr: CUdeviceptr, count: usize, location: CUmemLocation, flags: c_uint, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) -> CUresult);
cudadrv_fn!(cuMemAdvise_v2(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, location: CUmemLocation) -> CUresult);
cudadrv_fn!(cuMemRangeGetAttribute(data: *mut c_void, dataSize: usize, attribute: CUmem_range_attribute, devPtr: CUdeviceptr, count: usize) -> CUresult);
cudadrv_fn!(cuMemRangeGetAttributes(data: *mut *mut c_void, dataSizes: *mut usize, attributes: *mut CUmem_range_attribute, numAttributes: usize, devPtr: CUdeviceptr, count: usize) -> CUresult, { print_refs!(dataSizes); });
cudadrv_fn!(cuPointerSetAttribute(value: *const c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuPointerGetAttributes(numAttributes: c_uint, attributes: *mut CUpointer_attribute, data: *mut *mut c_void, ptr: CUdeviceptr) -> CUresult);
cudadrv_fn!(cuStreamCreate(phStream: *mut CUstream, Flags: c_uint) -> CUresult, { print_refs!(phStream); });
cudadrv_fn!(cuStreamCreateWithPriority(phStream: *mut CUstream, flags: c_uint, priority: c_int) -> CUresult, { print_refs!(phStream); });
cudadrv_fn!(cuStreamGetPriority(hStream: CUstream, priority: *mut c_int) -> CUresult, { print_refs!(priority); });
cudadrv_fn!(cuStreamGetFlags(hStream: CUstream, flags: *mut c_uint) -> CUresult, { print_refs!(flags); });
cudadrv_fn!(cuStreamGetId(hStream: CUstream, streamId: *mut c_ulonglong) -> CUresult, { print_refs!(streamId); });
cudadrv_fn!(cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult, { print_refs!(pctx); });
cudadrv_fn!(cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamAddCallback(hStream: CUstream, callback: CUstreamCallback, userData: *mut c_void, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamBeginCapture_v2(hStream: CUstream, mode: CUstreamCaptureMode) -> CUresult);
cudadrv_fn!(cuThreadExchangeStreamCaptureMode(mode: *mut CUstreamCaptureMode) -> CUresult, { print_refs!(mode); });
cudadrv_fn!(cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult, { print_refs!(phGraph); });
cudadrv_fn!(cuStreamIsCapturing(hStream: CUstream, captureStatus: *mut CUstreamCaptureStatus) -> CUresult);
cudadrv_fn!(cuStreamGetCaptureInfo_v2(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t, graph_out: *mut CUgraph, dependencies_out: *mut *const CUgraphNode, numDependencies_out: *mut usize) -> CUresult, { print_refs!(id_out, graph_out, numDependencies_out); });
cudadrv_fn!(cuStreamUpdateCaptureDependencies(hStream: CUstream, dependencies: *mut CUgraphNode, numDependencies: usize, flags: c_uint) -> CUresult, { print_refs!(dependencies); });
cudadrv_fn!(cuStreamAttachMemAsync(hStream: CUstream, dptr: CUdeviceptr, length: usize, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamQuery(hStream: CUstream) -> CUresult);
cudadrv_fn!(cuStreamSynchronize(hStream: CUstream) -> CUresult);
cudadrv_fn!(cuStreamDestroy_v2(hStream: CUstream) -> CUresult);
cudadrv_fn!(cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult);
cudadrv_fn!(cuStreamGetAttribute(hStream: CUstream, attr: CUstreamAttrID, value_out: *mut CUstreamAttrValue) -> CUresult);
cudadrv_fn!(cuStreamSetAttribute(hStream: CUstream, attr: CUstreamAttrID, value: *const CUstreamAttrValue) -> CUresult);
cudadrv_fn!(cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult, { print_refs!(phEvent); });
cudadrv_fn!(cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: c_uint) -> CUresult);
cudadrv_fn!(cuEventQuery(hEvent: CUevent) -> CUresult);
cudadrv_fn!(cuEventSynchronize(hEvent: CUevent) -> CUresult);
cudadrv_fn!(cuEventDestroy_v2(hEvent: CUevent) -> CUresult);
cudadrv_fn!(cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult, { print_refs!(pMilliseconds); });
cudadrv_fn!(cuImportExternalMemory(extMem_out: *mut CUexternalMemory, memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult);
cudadrv_fn!(cuExternalMemoryGetMappedBuffer(devPtr: *mut CUdeviceptr, extMem: CUexternalMemory, bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC) -> CUresult, { print_refs!(devPtr); });
cudadrv_fn!(cuExternalMemoryGetMappedMipmappedArray(mipmap: *mut CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) -> CUresult);
cudadrv_fn!(cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult);
cudadrv_fn!(cuImportExternalSemaphore(extSem_out: *mut CUexternalSemaphore, semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> CUresult);
cudadrv_fn!(cuSignalExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, numExtSems: c_uint, stream: CUstream) -> CUresult);
cudadrv_fn!(cuWaitExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, numExtSems: c_uint, stream: CUstream) -> CUresult);
cudadrv_fn!(cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult);
cudadrv_fn!(cuStreamWaitValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamWaitValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamWriteValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamWriteValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: c_uint) -> CUresult);
cudadrv_fn!(cuStreamBatchMemOp_v2(stream: CUstream, count: c_uint, paramArray: *mut CUstreamBatchMemOpParams, flags: c_uint) -> CUresult);
cudadrv_fn!(cuFuncGetAttribute(pi: *mut c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult, { print_refs!(pi); });
cudadrv_fn!(cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: c_int) -> CUresult);
cudadrv_fn!(cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult);
cudadrv_fn!(cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult);
cudadrv_fn!(cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult, { print_refs!(hmod); });
cudadrv_fn!(cuLaunchKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: CUstream, kernelParams: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult, {
    let cufunc_info = CUFUNC_INFO.lock().unwrap();
    let mut cufunc_sig = CUFUNC_SIG.lock().unwrap();
    let (cmod, func_name) = cufunc_info.get(&f).unwrap();

    if cufunc_sig.get(cmod).is_none() {
        let cumod_fatbin = CUMOD_FATBIN.lock().unwrap();
        let fatbinptr = cumod_fatbin.get(cmod).unwrap();
        let fn_sigs = parse_func_sigs_from_fatbin(*fatbinptr);
        cufunc_sig.insert(*cmod, fn_sigs);
    }

    let param_sizes = cufunc_sig.get(cmod).unwrap().get(func_name).unwrap();

    println!("  > {}<<<{{{},{},{}}},{{{},{},{}}},{:x?}>>>(...)", func_name, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, hStream);

    if !kernelParams.is_null() {
        for (idx, &CUfuncParamSize { size, offset: _ }) in param_sizes.iter().enumerate() {
            let argbufptr = unsafe { *kernelParams.add(idx) };
            println!("woo");
            match size {
                1 => {
                    let arg = unsafe { *(argbufptr as *mut u8) };
                    println!("    > arg{}: {:#x?}", idx, arg);
                }
                4 => {
                    let arg = unsafe { *(argbufptr as *mut u32) };
                    println!("    > arg{}: {:#x?}", idx, arg);
                }
                8 => {
                    let arg = unsafe { *(argbufptr as *mut u64) };
                    println!("    > arg{}: {:#x?}", idx, WCUunknownptr(arg));
                }
                16 => {
                    let arg = unsafe { *(argbufptr as *mut u128) };
                    println!("    > arg{}: {:#x?}", idx, arg);
                }
                _ => {
                    println!("    > &arg{}: {:#x?}", idx, argbufptr);
                }
            }
        }
    }
});
cudadrv_fn!(cuLaunchKernelEx(config: *const CUlaunchConfig, f: CUfunction, kernelParams: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult);
cudadrv_fn!(cuLaunchCooperativeKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, hStream: CUstream, kernelParams: *mut *mut c_void) -> CUresult);
cudadrv_fn!(cuLaunchCooperativeKernelMultiDevice(launchParamsList: *mut CUDA_LAUNCH_PARAMS, numDevices: c_uint, flags: c_uint) -> CUresult);
cudadrv_fn!(cuLaunchHostFunc(hStream: CUstream, fn_: CUhostFn, userData: *mut c_void) -> CUresult);
cudadrv_fn!(cuFuncSetBlockShape(hfunc: CUfunction, x: c_int, y: c_int, z: c_int) -> CUresult);
cudadrv_fn!(cuFuncSetSharedSize(hfunc: CUfunction, bytes: c_uint) -> CUresult);
cudadrv_fn!(cuParamSetSize(hfunc: CUfunction, numbytes: c_uint) -> CUresult);
cudadrv_fn!(cuParamSeti(hfunc: CUfunction, offset: c_int, value: c_uint) -> CUresult);
cudadrv_fn!(cuParamSetf(hfunc: CUfunction, offset: c_int, value: f32) -> CUresult);
cudadrv_fn!(cuParamSetv(hfunc: CUfunction, offset: c_int, ptr: *mut c_void, numbytes: c_uint) -> CUresult);
cudadrv_fn!(cuLaunch(f: CUfunction) -> CUresult);
cudadrv_fn!(cuLaunchGrid(f: CUfunction, grid_width: c_int, grid_height: c_int) -> CUresult);
cudadrv_fn!(cuLaunchGridAsync(f: CUfunction, grid_width: c_int, grid_height: c_int, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuParamSetTexRef(hfunc: CUfunction, texunit: c_int, hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuGraphCreate(phGraph: *mut CUgraph, flags: c_uint) -> CUresult, { print_refs!(phGraph); });
cudadrv_fn!(cuGraphAddKernelNode_v2(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphKernelNodeGetParams_v2(hNode: CUgraphNode, nodeParams: *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphKernelNodeSetParams_v2(hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddMemcpyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphMemcpyNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMCPY3D) -> CUresult);
cudadrv_fn!(cuGraphMemcpyNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMCPY3D) -> CUresult);
cudadrv_fn!(cuGraphAddMemsetNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphMemsetNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMSET_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphMemsetNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMSET_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddHostNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphHostNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_HOST_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphHostNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddChildGraphNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, childGraph: CUgraph) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphChildGraphNodeGetGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult, { print_refs!(phGraph); });
cudadrv_fn!(cuGraphAddEmptyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphAddEventRecordNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphEventRecordNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult, { print_refs!(event_out); });
cudadrv_fn!(cuGraphEventRecordNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult);
cudadrv_fn!(cuGraphAddEventWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphEventWaitNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult, { print_refs!(event_out); });
cudadrv_fn!(cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult);
cudadrv_fn!(cuGraphAddExternalSemaphoresSignalNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphExternalSemaphoresSignalNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExternalSemaphoresSignalNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddExternalSemaphoresWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphExternalSemaphoresWaitNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExternalSemaphoresWaitNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddBatchMemOpNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphBatchMemOpNodeGetParams(hNode: CUgraphNode, nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphBatchMemOpNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExecBatchMemOpNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddMemAllocNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphMemAllocNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphAddMemFreeNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, dptr: CUdeviceptr) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphMemFreeNodeGetParams(hNode: CUgraphNode, dptr_out: *mut CUdeviceptr) -> CUresult, { print_refs!(dptr_out); });
cudadrv_fn!(cuDeviceGraphMemTrim(device: CUdevice) -> CUresult);
cudadrv_fn!(cuDeviceGetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut c_void) -> CUresult);
cudadrv_fn!(cuDeviceSetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut c_void) -> CUresult);
cudadrv_fn!(cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult, { print_refs!(phGraphClone); });
cudadrv_fn!(cuGraphNodeFindInClone(phNode: *mut CUgraphNode, hOriginalNode: CUgraphNode, hClonedGraph: CUgraph) -> CUresult, { print_refs!(phNode); });
cudadrv_fn!(cuGraphNodeGetType(hNode: CUgraphNode, type_: *mut CUgraphNodeType) -> CUresult);
cudadrv_fn!(cuGraphGetNodes(hGraph: CUgraph, nodes: *mut CUgraphNode, numNodes: *mut usize) -> CUresult, { print_refs!(nodes, numNodes); });
cudadrv_fn!(cuGraphGetRootNodes(hGraph: CUgraph, rootNodes: *mut CUgraphNode, numRootNodes: *mut usize) -> CUresult, { print_refs!(rootNodes, numRootNodes); });
cudadrv_fn!(cuGraphGetEdges(hGraph: CUgraph, from: *mut CUgraphNode, to: *mut CUgraphNode, numEdges: *mut usize) -> CUresult, { print_refs!(from, to, numEdges); });
cudadrv_fn!(cuGraphNodeGetDependencies(hNode: CUgraphNode, dependencies: *mut CUgraphNode, numDependencies: *mut usize) -> CUresult, { print_refs!(dependencies, numDependencies); });
cudadrv_fn!(cuGraphNodeGetDependentNodes(hNode: CUgraphNode, dependentNodes: *mut CUgraphNode, numDependentNodes: *mut usize) -> CUresult, { print_refs!(dependentNodes, numDependentNodes); });
cudadrv_fn!(cuGraphAddDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult);
cudadrv_fn!(cuGraphRemoveDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult);
cudadrv_fn!(cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult);
cudadrv_fn!(cuGraphInstantiateWithFlags(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, flags: c_ulonglong) -> CUresult);
cudadrv_fn!(cuGraphInstantiateWithParams(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExecGetFlags(hGraphExec: CUgraphExec, flags: *mut cuuint64_t) -> CUresult, { print_refs!(flags); });
cudadrv_fn!(cuGraphExecKernelNodeSetParams_v2(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExecMemcpyNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuGraphExecMemsetNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult);
cudadrv_fn!(cuGraphExecHostNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExecChildGraphNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, childGraph: CUgraph) -> CUresult);
cudadrv_fn!(cuGraphExecEventRecordNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult);
cudadrv_fn!(cuGraphExecEventWaitNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult);
cudadrv_fn!(cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult);
cudadrv_fn!(cuGraphNodeSetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: c_uint) -> CUresult);
cudadrv_fn!(cuGraphNodeGetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: *mut c_uint) -> CUresult, { print_refs!(isEnabled); });
cudadrv_fn!(cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult);
cudadrv_fn!(cuGraphDestroy(hGraph: CUgraph) -> CUresult);
cudadrv_fn!(cuGraphExecUpdate_v2(hGraphExec: CUgraphExec, hGraph: CUgraph, resultInfo: *mut CUgraphExecUpdateResultInfo) -> CUresult);
cudadrv_fn!(cuGraphKernelNodeCopyAttributes(dst: CUgraphNode, src: CUgraphNode) -> CUresult);
cudadrv_fn!(cuGraphKernelNodeGetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value_out: *mut CUkernelNodeAttrValue) -> CUresult);
cudadrv_fn!(cuGraphKernelNodeSetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value: *const CUkernelNodeAttrValue) -> CUresult);
cudadrv_fn!(cuGraphDebugDotPrint(hGraph: CUgraph, path: ConstCharStar, flags: c_uint) -> CUresult);
cudadrv_fn!(cuUserObjectCreate(object_out: *mut CUuserObject, ptr: *mut c_void, destroy: CUhostFn, initialRefcount: c_uint, flags: c_uint) -> CUresult, { print_refs!(object_out); });
cudadrv_fn!(cuUserObjectRetain(object: CUuserObject, count: c_uint) -> CUresult);
cudadrv_fn!(cuUserObjectRelease(object: CUuserObject, count: c_uint) -> CUresult);
cudadrv_fn!(cuGraphRetainUserObject(graph: CUgraph, object: CUuserObject, count: c_uint, flags: c_uint) -> CUresult);
cudadrv_fn!(cuGraphReleaseUserObject(graph: CUgraph, object: CUuserObject, count: c_uint) -> CUresult);
cudadrv_fn!(cuGraphAddNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUgraphNodeParams) -> CUresult, { print_refs!(phGraphNode); });
cudadrv_fn!(cuGraphNodeSetParams(hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams)-> CUresult);
cudadrv_fn!(cuGraphExecNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult);
cudadrv_fn!(cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut c_int, func: CUfunction, blockSize: c_int, dynamicSMemSize: usize) -> CUresult, { print_refs!(numBlocks); });
cudadrv_fn!(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut c_int, func: CUfunction, blockSize: c_int, dynamicSMemSize: usize, flags: c_uint) -> CUresult, { print_refs!(numBlocks); });
cudadrv_fn!(cuOccupancyMaxPotentialBlockSize(minGridSize: *mut c_int, blockSize: *mut c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: c_int) -> CUresult, { print_refs!(minGridSize, blockSize); });
cudadrv_fn!(cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize: *mut c_int, blockSize: *mut c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: c_int, flags: c_uint) -> CUresult, { print_refs!(minGridSize, blockSize); });
cudadrv_fn!(cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: CUfunction, numBlocks: c_int, blockSize: c_int) -> CUresult, { print_refs!(dynamicSmemSize); });
cudadrv_fn!(cuOccupancyMaxPotentialClusterSize(clusterSize: *mut c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult, { print_refs!(clusterSize); });
cudadrv_fn!(cuOccupancyMaxActiveClusters(numClusters: *mut c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult, { print_refs!(numClusters); });
cudadrv_fn!(cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuTexRefSetMipmappedArray(hTexRef: CUtexref, hMipmappedArray: CUmipmappedArray, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuTexRefSetAddress_v2(ByteOffset: *mut usize, hTexRef: CUtexref, dptr: CUdeviceptr, bytes: usize) -> CUresult, { print_refs!(ByteOffset); });
cudadrv_fn!(cuTexRefSetAddress2D_v3(hTexRef: CUtexref, desc: *const CUDA_ARRAY_DESCRIPTOR, dptr: CUdeviceptr, Pitch: usize) -> CUresult);
cudadrv_fn!(cuTexRefSetFormat(hTexRef: CUtexref, fmt: CUarray_format, NumPackedComponents: c_int) -> CUresult);
cudadrv_fn!(cuTexRefSetAddressMode(hTexRef: CUtexref, dim: c_int, am: CUaddress_mode) -> CUresult);
cudadrv_fn!(cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult);
cudadrv_fn!(cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult);
cudadrv_fn!(cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult);
cudadrv_fn!(cuTexRefSetMipmapLevelClamp(hTexRef: CUtexref, minMipmapLevelClamp: f32, maxMipmapLevelClamp: f32) -> CUresult);
cudadrv_fn!(cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: c_uint) -> CUresult);
cudadrv_fn!(cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: *mut f32) -> CUresult, { print_refs!(pBorderColor); });
cudadrv_fn!(cuTexRefSetFlags(hTexRef: CUtexref, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult, { print_refs!(pdptr); });
cudadrv_fn!(cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuTexRefGetMipmappedArray(phMipmappedArray: *mut CUmipmappedArray, hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuTexRefGetAddressMode(pam: *mut CUaddress_mode, hTexRef: CUtexref, dim: c_int) -> CUresult);
cudadrv_fn!(cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuTexRefGetFormat(pFormat: *mut CUarray_format, pNumChannels: *mut c_int, hTexRef: CUtexref) -> CUresult, { print_refs!(pNumChannels); });
cudadrv_fn!(cuTexRefGetMipmapFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult, { print_refs!(pbias); });
cudadrv_fn!(cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp: *mut f32, pmaxMipmapLevelClamp: *mut f32, hTexRef: CUtexref) -> CUresult, { print_refs!(pminMipmapLevelClamp, pmaxMipmapLevelClamp); });
cudadrv_fn!(cuTexRefGetMaxAnisotropy(pmaxAniso: *mut c_int, hTexRef: CUtexref) -> CUresult, { print_refs!(pmaxAniso); });
cudadrv_fn!(cuTexRefGetBorderColor(pBorderColor: *mut f32, hTexRef: CUtexref) -> CUresult, { print_refs!(pBorderColor); });
cudadrv_fn!(cuTexRefGetFlags(pFlags: *mut c_uint, hTexRef: CUtexref) -> CUresult, { print_refs!(pFlags); });
cudadrv_fn!(cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult);
cudadrv_fn!(cuTexRefDestroy(hTexRef: CUtexref) -> CUresult);
cudadrv_fn!(cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult);
cudadrv_fn!(cuTexObjectCreate(pTexObject: *mut CUtexObject, pResDesc: *const CUDA_RESOURCE_DESC, pTexDesc: *const CUDA_TEXTURE_DESC, pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC) -> CUresult, { print_refs!(pTexObject); });
cudadrv_fn!(cuTexObjectDestroy(texObject: CUtexObject) -> CUresult);
cudadrv_fn!(cuTexObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult);
cudadrv_fn!(cuTexObjectGetTextureDesc(pTexDesc: *mut CUDA_TEXTURE_DESC, texObject: CUtexObject) -> CUresult);
cudadrv_fn!(cuTexObjectGetResourceViewDesc(pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC, texObject: CUtexObject) -> CUresult);
cudadrv_fn!(cuSurfObjectCreate(pSurfObject: *mut CUsurfObject, pResDesc: *const CUDA_RESOURCE_DESC) -> CUresult);
cudadrv_fn!(cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult);
cudadrv_fn!(cuSurfObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, surfObject: CUsurfObject) -> CUresult);
cudadrv_fn!(cuTensorMapEncodeTiled(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, boxDim: *const cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult);
cudadrv_fn!(cuTensorMapEncodeIm2col(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, pixelBoxLowerCorner: *const c_int, pixelBoxUpperCorner: *const c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult);
cudadrv_fn!(cuTensorMapReplaceAddress(tensorMap: *mut CUtensorMap, globalAddress: *mut c_void) -> CUresult);
cudadrv_fn!(cuDeviceCanAccessPeer(canAccessPeer: *mut c_int, dev: CUdevice, peerDev: CUdevice) -> CUresult, { print_refs!(canAccessPeer); });
cudadrv_fn!(cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: c_uint) -> CUresult);
cudadrv_fn!(cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult);
cudadrv_fn!(cuDeviceGetP2PAttribute(value: *mut c_int, attrib: CUdevice_P2PAttribute, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult, { print_refs!(value); });
cudadrv_fn!(cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult);
cudadrv_fn!(cuGraphicsSubResourceGetMappedArray(pArray: *mut CUarray, resource: CUgraphicsResource, arrayIndex: c_uint, mipLevel: c_uint) -> CUresult);
cudadrv_fn!(cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray: *mut CUmipmappedArray, resource: CUgraphicsResource) -> CUresult);
cudadrv_fn!(cuGraphicsResourceGetMappedPointer_v2(pDevPtr: *mut CUdeviceptr, pSize: *mut usize, resource: CUgraphicsResource) -> CUresult, { print_refs!(pDevPtr, pSize); });
cudadrv_fn!(cuGraphicsResourceSetMapFlags_v2(resource: CUgraphicsResource, flags: c_uint) -> CUresult);
cudadrv_fn!(cuGraphicsMapResources(count: c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuGraphicsUnmapResources(count: c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult);
cudadrv_fn!(cuGetProcAddress_v2(symbol: ConstCharStar, pfn: *mut *mut c_void, cudaVersion: c_int, flags: cuuint64_t, symbolStatus: *mut CUdriverProcAddressQueryResult) -> CUresult);
cudadrv_fn!(cuCoredumpGetAttribute(attrib: CUcoredumpSettings, value: *mut c_void, size: *mut usize) -> CUresult, { print_refs!(size); });
cudadrv_fn!(cuCoredumpGetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut c_void, size: *mut usize) -> CUresult, { print_refs!(size); });
cudadrv_fn!(cuCoredumpSetAttribute(attrib: CUcoredumpSettings, value: *mut c_void, size: *mut usize) -> CUresult, { print_refs!(size); });
cudadrv_fn!(cuCoredumpSetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut c_void, size: *mut usize) -> CUresult, { print_refs!(size); });
cudadrv_fn!(cuGetExportTable(ppExportTable: *mut *const c_void, pExportTableId: *const CUuuid) -> CUresult);
cudadrv_fn!(cuProfilerInitialize(configFile: ConstCharStar, outputFile: ConstCharStar, outputMode: CUoutput_mode) -> CUresult);
cudadrv_fn!(cuProfilerStart() -> CUresult);
cudadrv_fn!(cuProfilerStop() -> CUresult);
