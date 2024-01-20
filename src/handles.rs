use std::ffi::c_char;

use colored::Colorize;
use cudarc::driver::sys::{CUcontext, CUfunction};
use libc::c_void;

use cudarc::driver::sys::{CUdevice, CUevent, CUkernel, CUlibrary, CUmodule, CUstream};

use std::{collections::HashMap, sync::Mutex};

use cudarc::driver::sys::CUdeviceptr;
use once_cell::sync::Lazy;
use rangemap::RangeMap;

use crate::fatbin::CUfuncParamSize;

/*********************************************************************/
/*                                                                   */
/* Globals                                                           */
/*                                                                   */
/*********************************************************************/

pub static PINNED_MEM: Lazy<Mutex<RangeMap<usize, (usize, usize)>>> =
    Lazy::new(|| Mutex::new(RangeMap::new()));
pub static DEVICE_MEM: Lazy<Mutex<RangeMap<usize, (usize, usize)>>> =
    Lazy::new(|| Mutex::new(RangeMap::new()));
pub static CUFUNC_SIG: Lazy<Mutex<HashMap<WCUmodule, HashMap<String, Vec<CUfuncParamSize>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
pub static CUFUNC_INFO: Lazy<Mutex<HashMap<WCUfunction, (WCUmodule, String)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
pub static CUMOD_FATBIN: Lazy<Mutex<HashMap<WCUmodule, FatBinPtr>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/*********************************************************************/
/*                                                                   */
/* New Types                                                         */
/*                                                                   */
/*********************************************************************/

macro_rules! impl_new_type {
    (NO_DEBUG, $new_type: ident, $type: ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
        pub struct $new_type(pub $type);
        unsafe impl Send for $new_type {}
        unsafe impl Sync for $new_type {}
    };
    ($new_type: ident, $type: ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
        pub struct $new_type(pub $type);
        unsafe impl Send for $new_type {}
        unsafe impl Sync for $new_type {}
    };
}

impl_new_type!(NO_DEBUG, WConstCharStar, *const c_char);
impl_new_type!(Void, *mut c_void);
impl_new_type!(FatBinPtr, *const c_void);
impl_new_type!(NO_DEBUG, WCUcontext, CUcontext);
impl_new_type!(NO_DEBUG, WCUfunction, CUfunction);
impl_new_type!(NO_DEBUG, WCUmodule, CUmodule);
impl_new_type!(NO_DEBUG, WCUstream, CUstream);
impl_new_type!(NO_DEBUG, WCUdevice, CUdevice);
impl_new_type!(WClibrary, CUlibrary);
impl_new_type!(WCUkernel, CUkernel);
impl_new_type!(WCUevent, CUevent);
impl_new_type!(NO_DEBUG, WCUdeviceptr, CUdeviceptr);
impl_new_type!(NO_DEBUG, WCUpinnedptr, CUdeviceptr);
impl_new_type!(NO_DEBUG, WCUunknownptr, CUdeviceptr);

/*********************************************************************/
/*                                                                   */
/* Debug Format                                                      */
/*                                                                   */
/*********************************************************************/

impl std::fmt::Debug for WConstCharStar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cstr = unsafe { std::ffi::CStr::from_ptr(self.0) };
        let s = format!("{:?}", cstr);
        write!(f, "{}", s.red())
    }
}

impl std::fmt::Debug for WCUcontext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUctx_{:x?}", self.0);
        write!(f, "{}", s.cyan())
    }
}

impl std::fmt::Debug for WCUdevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUdev_{:x?}", self.0);
        write!(f, "{}", s.blue())
    }
}

impl std::fmt::Debug for WCUfunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = CUFUNC_INFO.lock().unwrap();
        let s = match map.get(self) {
            Some((_, name)) => {
                format!("CUfunc_{}", name)
            }
            None => "CUfunc_UNKNOWN".to_owned(),
        };
        write!(f, "{}", s.green())
    }
}

impl std::fmt::Debug for WCUmodule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUmod_{:x?}", self.0);
        write!(f, "{}", s.magenta())
    }
}

impl std::fmt::Debug for WCUdeviceptr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device_mem = DEVICE_MEM.lock().unwrap();
        let (base, size) = *device_mem
            .get(&(self.0 as _))
            .expect(format!("{:#x?} not found", self.0).as_str());
        let s = format!(
            "dev_{:x?}({})[{}]",
            base,
            size,
            self.0 as usize - base
        );
        write!(f, "{}", s.red().on_green())
    }
}

impl std::fmt::Debug for WCUpinnedptr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pinned_mem = PINNED_MEM.lock().unwrap();
        let (base, size) = *pinned_mem
            .get(&(self.0 as _))
            .expect(format!("{:#x?} not found", self.0).as_str());
        let s = format!(
            "dev_{:x?}({})[{}]",
            base,
            size,
            self.0 as usize - base
        );
        write!(f, "{}", s.blue().on_green())
    }
}

impl std::fmt::Debug for WCUunknownptr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device_mem = DEVICE_MEM.lock().unwrap();
        let pinned_mem = PINNED_MEM.lock().unwrap();
        let device_lookup = device_mem.get(&(self.0 as _)).cloned();
        let pinned_lookup = pinned_mem.get(&(self.0 as _)).cloned();
        drop(device_mem);
        drop(pinned_mem);
        match (device_lookup, pinned_lookup) {
            (Some(_), Some(_)) => panic!("ptr {:#x?} is both pinned and device?", self.0),
            (Some(_), None) => WCUdeviceptr(self.0).fmt(f),
            (None, Some(_)) => WCUpinnedptr(self.0).fmt(f),
            _ => {
                let s = format!("host_{:x?}", self.0);
                write!(f, "{}", s.yellow().on_green())
            }
        }
    }
}

impl std::fmt::Debug for WCUstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = if self.0.is_null() {
            "CUstream_NULL".to_owned()
        } else {
            format!("CUstream_{:x?}", self.0)
        };
        write!(f, "{}", s.bright_blue())
    }
}
