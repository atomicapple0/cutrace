# cudaExternalSemaphoreWaitParams,cudaExternalSemaphore_t,cudaGraphMemAttributeType,cudaMemPoolPtrExportData,f32,cudaMemcpy3DParms,cudaIpcMemHandle_t,cudaArraySparseProperties,cudaArray_t,cudaMemcpyKind,cudaFuncAttribute,cudaMemPoolAttr,c_char,cudaEvent_t,c_uint,cudaGraphicsResource_t,cudaStreamCaptureStatus,cudaFuncAttributes,cudaExternalSemaphoreSignalParams,cudaSurfaceObject_t,cudaKernelNodeParams,usize,cudaMemAllocationHandleType,cudaResourceViewDesc,cudaStreamCallback_t,cudaTextureDesc,cudaDriverEntryPointQueryResult,cudaError_t,cudaSharedMemConfig,cudaExternalSemaphoreSignalNodeParams,cudaExternalMemoryMipmappedArrayDesc,cudaMipmappedArray_t,f64,cudaChannelFormatKind,cudaExtent,c_void,cudaFlushGPUDirectRDMAWritesTarget,cudaExternalMemoryBufferDesc,cudaMipmappedArray_const_t,cudaTextureObject_t,cudaMemcpy3DPeerParms,dim3,cudaStreamCaptureMode,cudaGraphExecUpdateResultInfo,cudaDeviceAttr,cudaPointerAttributes,cudaFuncCache,c_int,cudaMemPool_t,cudaLaunchAttributeID,cudaExternalMemoryHandleDesc,cudaMemLocation,cudaMemsetParams,cudaFunction_t,cudaFlushGPUDirectRDMAWritesScope,cudaPitchedPtr,cudaGraphNode_t,cudaStream_t,cudaDeviceProp,cudaLaunchAttributeValue,cudaLimit,cudaExternalMemory_t,cudaMemoryAdvise,cudaGraphNodeType,cudaGraphExec_t,cudaUserObject_t,cudaKernel_t,cudaDeviceP2PAttr,cudaHostFn_t,cudaMemAccessFlags,cudaHostNodeParams,cudaExternalSemaphoreWaitNodeParams,cudaChannelFormatDesc,cudaUUID_t,cudaGraphInstantiateParams,cudaResourceDesc,cudaMemAccessDesc,cudaArray_const_t,cudaGraphNodeParams,cudaExternalSemaphoreHandleDesc,cudaLaunchParams,c_ulonglong,cudaArrayMemoryRequirements,cudaMemRangeAttribute,cudaMemAllocNodeParams,cudaIpcEventHandle_t,cudaMemPoolProps,cudaGraph_t,cudaLaunchConfig_t

with open("cool.txt") as f:
    src = f.read()

import re

ok_types = [
    "c_ulonglong",
    "CUstream",
    "CUkernel",
    "CUevent",
    "usize",
    "c_uint",
    "f32",
    "CUmodule",
    "CUgraph",
    "c_int",
    "CUdeviceptr",
    "c_char",
    "CUcontext",
    "CUipcEventHandle",
    "cuuint64_t",
    "CUdevice",
    "CUlibrary",
    "CUfunc_cache",
    "CUstreamCaptureMode",
    "CUuuid",
    "CUgraphNode",
    "CUtexObject",
    "CUfunction",
    "CUmoduleLoadingMode",
    "CUsharedconfig",
    "CUuserObject",
]

all_types = set()

found = 0

idx = 0
while idx < len(src):
    pub_fn = "pub fn "
    start = src[idx : min(len(src), idx + len(pub_fn))]
    if start == pub_fn:
        idx += len(pub_fn)
        munch = []
        while src[idx] != ";":
            if src[idx] != "\n":
                munch.append(src[idx])
            idx += 1
        munch = "".join(munch)
        for _ in range(5):
            munch = munch.replace("  ", "")
            munch = munch.replace(", ", ",")
        munch = munch.replace(",)", ")")
        munch = munch.replace(",", ", ")
        munch = munch.replace("::core::ffi::", "")

        # match ident : *mut
        idents = re.findall(r"([a-zA-Z0-9_]+): \*mut ([a-zA-Z0-9_]+)", munch)
        # idents = re.findall(r"([a-zA-Z0-9_]+): (\*?[a-zA-Z0-9_\s]+)", munch)
        # print(idents)

        maybe = ""
        idents = [a for a,b in idents if b in ok_types]
        if idents:
            maybe = ", {" + ", ".join(idents) + "}"
            # for ident, ty in idents:
            #     ty = ty.replace("*mut ", "")
            #     ty = ty.replace("*mut ", "")
            #     ty = ty.replace("*mut ", "")
            #     ty = ty.replace("*const ", "")
            #     ty = ty.replace("*const ", "")
            #     ty = ty.replace("*const ", "")
            #     all_types.add(ty)

        # # print("-----")
        print(f"cuda_fn!({munch}{maybe});")
        # print("-----")
        found += 1

        # if found == 10:
        #     break
    idx += 1

print(",".join(all_types))
