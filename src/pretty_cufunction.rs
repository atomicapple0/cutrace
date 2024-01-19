use std::{collections::HashMap, ffi::CStr, sync::Mutex};

use colored::Colorize;
use cudarc::driver::sys::CUfunction;
use once_cell::sync::Lazy;

static CUDA_FUNCTIONS: Lazy<Mutex<HashMap<PrettyCUfunction, String>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[repr(C)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct PrettyCUfunction(pub CUfunction);

impl std::fmt::Debug for PrettyCUfunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = CUDA_FUNCTIONS.lock().unwrap();
        let s = match map.get(self) {
            Some(name) => {
                format!("CUfunc_{}", name)
            }
            None => "CUfunc_UNKNOWN".to_owned(),
        };
        write!(f, "{}", s.truecolor(237, 145, 33))
    }
}

unsafe impl Send for PrettyCUfunction {}
unsafe impl Sync for PrettyCUfunction {}

impl PrettyCUfunction {
    pub fn register(self, name: &CStr) {
        CUDA_FUNCTIONS
            .lock()
            .unwrap()
            .insert(self, name.to_str().unwrap().to_string());
    }
    pub fn deregister(self) {
        CUDA_FUNCTIONS.lock().unwrap().remove(&self);
    }
}
