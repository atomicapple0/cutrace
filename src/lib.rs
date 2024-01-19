#![feature(c_str_literals)]

#[allow(non_snake_case)]
mod cudadrv;
#[allow(non_snake_case)]
mod cudart;
mod pretty_constcharstar;
mod pretty_cucontext;
mod pretty_cudevice;
mod pretty_cufunction;
mod pretty_cumodule;
mod symbol_loader;

use std::ffi::{c_char, c_float, c_int, c_uint, CStr};

use cudarc::driver::sys::{
    CUcontext, CUdevice, CUdevice_attribute, CUdeviceptr, CUevent, CUfunction,
    CUfunction_attribute, CUmodule, CUmoduleLoadingMode, CUresult, CUstream, CUstreamCaptureStatus,
    CUuuid,
};
use libc::c_void;
use once_cell::sync::Lazy;
use symbol_loader::SymbolLoader;

use crate::pretty_constcharstar::ConstCharStar;

const CUDADRV_PATH: &CStr = c"libcuda.so.535.86.10";

static CUDADRV_SYMBOLS: Lazy<SymbolLoader> = Lazy::new(|| SymbolLoader::new(CUDADRV_PATH));
#[macro_export]
macro_rules! print_ref {
    ($ref_arg: ident) => {
        println!("  > *.{} = {:x?}", stringify!($ref_arg), unsafe {
            *$ref_arg
        });
    };
}
#[macro_export]
macro_rules! gen {
    ($fn_name: ident($($arg: ident: $arg_ty: ty),*) -> $ret_ty: ty, $block:block) => {
        #[no_mangle]
        pub extern "C" fn $fn_name($($arg: $arg_ty),*) -> $ret_ty {
            use std::io::Write;
            use std::io::stdout;
            println!("{}", stringify!($fn_name));
            stdout().flush().unwrap();
            // let args : &[String] = &[$(
            //     format!(".{}={:x?}", stringify!($arg), $arg),
            // )*];
            // print!("({}) = ", args.join(", "));
            // stdout().flush().unwrap();

            let sym = $crate::CUDADRV_SYMBOLS.get_symbol(stringify!($fn_name));
            let fn_ptr: extern "C" fn($($arg_ty),*) -> $ret_ty = unsafe { ::core::mem::transmute(sym) };
            let res = fn_ptr($($arg),*);
            println!("{:?}", res);

            $block

            res
        }
    };
}

#[macro_export]
macro_rules! cuda_fn {
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty) => {
        gen!($fn_name($($arg: $arg_ty),*) -> $ret_ty, {});
    };
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty, { $($ref_arg: ident),* $(,)? }) => {
        gen!($fn_name($($arg: $arg_ty),*) -> $ret_ty, {$(
            print_ref!($ref_arg);
        )*});
    };
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty, $block:block) => {
        gen!($fn_name($($arg: $arg_ty),*) -> $ret_ty, $block);
    }
}

// cuda_fn!(cuInit(flags: c_uint) -> CUresult);
// // cuda_fn!(cuGetErrorString(error: CUresult, str_ref: *mut ConstCharStar) -> CUresult, {str_ref});
// // cuda_fn!(cuGetErrorName(error: CUresult, str_ref: *mut ConstCharStar) -> CUresult, {str_ref});
// // cuda_fn!(cuCtxCreate(ctx_ref: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult, {ctx_ref});
// // cuda_fn!(cuCtxCreate_v2(ctx_ref: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult, {ctx_ref});
// // cuda_fn!(cuCtxSynchronize() -> CUresult);
// // cuda_fn!(cuCtxSetCurrent(ctx: CUcontext) -> CUresult);
// // cuda_fn!(cuCtxGetCurrent(ctx_ref: *mut CUcontext) -> CUresult, {ctx_ref});
// // cuda_fn!(cuCtxGetStreamPriorityRange(low_ref: *mut c_int, high_ref: *mut c_int) -> CUresult, {low_ref, high_ref});
// // cuda_fn!(cuDeviceGetAttribute(pi_ref: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult, {pi_ref});
// cuda_fn!(cuDeviceGet(dev_ref: *mut CUdevice, ord: c_int) -> CUresult, {dev_ref});
// // cuda_fn!(cuDeviceGetCount(count_ref: *mut c_int) -> CUresult, {count_ref});
// // cuda_fn!(cuDeviceGetUuid(uuid_ref: *mut CUuuid, dev: CUdevice) -> CUresult, {uuid_ref});
// // cuda_fn!(cuDeviceGetName(name_ref: *mut c_char, len: c_int, dev: CUdevice) -> CUresult, {name_ref});
// // cuda_fn!(cuDeviceTotalMem(bytes_ref: *mut usize, dev: CUdevice) -> CUresult, {bytes_ref});
// // cuda_fn!(cuDeviceTotalMem_v2(bytes_ref: *mut usize, dev: CUdevice) -> CUresult, {bytes_ref});
// // cuda_fn!(cuDriverGetVersion(version_ref: *mut c_int) -> CUresult, {version_ref});
// // cuda_fn!(cuDevicePrimaryCtxRetain(ctx_ref: *mut CUcontext, dev: CUdevice) -> CUresult, {ctx_ref});
// #[no_mangle]
// pub extern "C" fn cuDevicePrimaryCtxRetain(ctx_ref: *mut CUcontext, dev: CUdevice) -> CUresult {
//     todo!("cuDevicePrimaryCtxRetain")
// }
// #[no_mangle]
// pub extern "C" fn cuMemGetInfo(
//     free_bytes_ref: *mut usize,
//     total_bytes_ref: *mut usize,
// ) -> CUresult {
//     todo!("cuMemGetInfo")
// }
// // cuda_fn!(cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult);
// // cuda_fn!(cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult);
// // cuda_fn!(cuDevicePrimaryCtxGetState(dev: CUdevice, flags_ref: *mut c_uint, active_ref: *mut c_int) -> CUresult, {flags_ref, active_ref});
// // cuda_fn!(cuEventCreate(event_ref: *mut CUevent, flags: c_uint) -> CUresult, {event_ref});
// // cuda_fn!(cuEventDestroy(event: CUevent) -> CUresult);
// // cuda_fn!(cuEventDestroy_v2(event: CUevent) -> CUresult);
// // cuda_fn!(cuEventElapsedTime(time_ref: *mut c_float, start: CUevent, end: CUevent) -> CUresult, {time_ref});
// // cuda_fn!(cuEventRecord(event: CUevent, stream: CUstream) -> CUresult);
// // cuda_fn!(cuEventSynchronize(event: CUevent) -> CUresult);
// // cuda_fn!(cuModuleLoadData(cmod_ref: *mut CUmodule, image: *const c_void) -> CUresult, {cmod_ref});
// // cuda_fn!(cuModuleLoad(cmod_ref: *mut CUmodule, file_name: ConstCharStar) -> CUresult, {cmod_ref});
// cuda_fn!(cuModuleGetFunction(func_ref: *mut CUfunction, cmod: CUmodule, func_name: ConstCharStar) -> CUresult, {
//     // CUfunction::register(unsafe{*func_ref}, unsafe{CStr::from_ptr(func_name.0 as _)});
//     print_ref!(func_ref);
// });
// // cuda_fn!(cuModuleGetGlobal(dptr_ref: *mut CUdeviceptr, bytes_ref: *mut usize, module: CUmodule, name: ConstCharStar) -> CUresult, {dptr_ref, bytes_ref});
// // cuda_fn!(cuModuleGetGlobal_v2(dptr_ref: *mut CUdeviceptr, bytes_ref: *mut usize, module: CUmodule, name: ConstCharStar) -> CUresult, {dptr_ref, bytes_ref});
// // cuda_fn!(cuModuleGetLoadingMode(mode_ref: *mut CUmoduleLoadingMode) -> CUresult, {mode_ref});
// // cuda_fn!(cuModuleUnload(module: CUmodule) -> CUresult);
// // cuda_fn!(cuStreamCreate(stream_ref: *mut CUstream, flags: c_uint) -> CUresult, {stream_ref});
// // cuda_fn!(cuStreamDestroy(stream: CUstream) -> CUresult);
// // cuda_fn!(cuStreamDestroy_v2(stream: CUstream) -> CUresult);
// // cuda_fn!(cuStreamSynchronize(stream: CUstream) -> CUresult);
// // cuda_fn!(cuStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult);
// // cuda_fn!(cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyHtoD(dst_dev: CUdeviceptr, src_host: *const c_void, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyHtoD_v2(dst_device: CUdeviceptr, src_host: *const c_void, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyDtoH(dst_host: *mut c_void, src_device: CUdeviceptr, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyDtoH_v2(dst_host: *mut c_void, src_device: CUdeviceptr, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyHtoDAsync(dst_device: CUdeviceptr, src_host: *const c_void, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyHtoDAsync_v2(dst_device: CUdeviceptr, src_host: *const c_void, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyDtoHAsync(dst_host: *mut c_void, src_device: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyDtoHAsync_v2(dst_host: *mut c_void, src_device: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyDtoD(dst_device: CUdeviceptr, src_device: CUdeviceptr, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyDtoD_v2(dst_device: CUdeviceptr, src_device: CUdeviceptr, bytes: usize) -> CUresult);
// // cuda_fn!(cuMemcpyDtoDAsync(dst_device: CUdeviceptr, src_device: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemcpyDtoDAsync_v2(dst_device: CUdeviceptr, src_device: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemAllocAsync(dptr_ref: *mut CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult, {dptr_ref});
// // cuda_fn!(cuMemAlloc(dptr_ref: *mut CUdeviceptr, bytes: usize) -> CUresult, {dptr_ref});
// // cuda_fn!(cuMemAlloc_v2(dptr_ref: *mut CUdeviceptr, bytes: usize) -> CUresult, {dptr_ref});
// // cuda_fn!(cuMemFreeAsync(dptr: CUdeviceptr, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemFree(dptr: CUdeviceptr) -> CUresult);
// // cuda_fn!(cuMemFree_v2(dptr: CUdeviceptr) -> CUresult);
// // cuda_fn!(cuMemHostAlloc(host_ptr_ref: *mut *mut CUdeviceptr, bytes: usize, flags: c_uint) -> CUresult, {host_ptr_ref});
// // cuda_fn!(cuMemFreeHost(host_ptr: *mut CUdeviceptr) -> CUresult, {host_ptr});
// // cuda_fn!(cuMemGetInfo(free_bytes_ref: *mut usize, total_bytes_ref: *mut usize) -> CUresult, {free_bytes_ref, total_bytes_ref});
// // cuda_fn!(cuMemGetInfo_v2(free_bytes_ref: *mut usize, total_bytes_ref: *mut usize) -> CUresult, {free_bytes_ref, total_bytes_ref});
// // cuda_fn!(cuMemsetD32(dst: CUdeviceptr, value: u32, n: usize) -> CUresult);
// // cuda_fn!(cuMemsetD32_v2(dst: CUdeviceptr, value: u32, n: usize) -> CUresult);
// // cuda_fn!(cuMemsetD8Async(dst: CUdeviceptr, value: u8, n: usize, stream: CUstream) -> CUresult);
// // cuda_fn!(cuMemsetD8(dst: CUdeviceptr, value: u8, n: usize) -> CUresult);
// // cuda_fn!(cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult);
// // cuda_fn!(cuFuncSetAttribute(func: CUfunction, attrib: CUfunction_attribute, value: c_int) -> CUresult);
// // // cuda_fn!(cuLaunchKernel(func: CUfunction, grid_x: c_uint, grid_y: c_uint, grid_z: c_uint, block_x: c_uint, block_y: c_uint, block_z: c_uint, nbytes_shared: c_uint, stream: CUstream, kernel_params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult);
// // cuda_fn!(cuStreamIsCapturing(stream: CUstream, capture_status_ref: *mut CUstreamCaptureStatus) -> CUresult, {capture_status_ref});
// // cuda_fn!(cuFuncGetAttribute(pi_ref: *mut c_int, attrib: CUfunction_attribute, func: CUfunction) -> CUresult, {pi_ref});
// cuda_fn!(cudaSetDevice(device: i32) -> CUresult);
