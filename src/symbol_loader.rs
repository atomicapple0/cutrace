use std::{
    collections::HashMap,
    ffi::{c_char, c_uint, CStr, CString},
    sync::RwLock,
};

use cudarc::driver::sys::{CUdeviceptr, CUresult};
use libc::c_void;
use once_cell::sync::Lazy;

struct SharedObj(*mut c_void);
unsafe impl Send for SharedObj {}
unsafe impl Sync for SharedObj {}
pub struct SymbolLoader {
    shared_obj: SharedObj,
    symbols: RwLock<HashMap<&'static str, fn()>>,
}

impl SymbolLoader {
    pub fn new(so_path: &CStr) -> Self {
        let shared_obj = unsafe {
            SharedObj(libc::dlopen(
                so_path.as_ptr() as *const c_char,
                libc::RTLD_NOW | libc::RTLD_NODELETE,
            ))
        };
        let symbols = RwLock::new(HashMap::new());
        Self {
            shared_obj,
            symbols,
        }
    }

    pub fn get_symbol(&self, name: &'static str) -> fn() {
        let map = self.symbols.read().unwrap();
        match map.get(name) {
            Some(f) => *f,
            None => {
                drop(map);
                let mut map = self.symbols.write().unwrap();
                let f = *map.entry(name).or_insert_with(|| {
                    let name = CString::new(name).unwrap();
                    let sym = unsafe { libc::dlsym(self.shared_obj.0, name.as_ptr() as _) };
                    assert!(!sym.is_null());
                    unsafe { std::mem::transmute::<*const c_void, fn()>(sym) }
                });
                f
            }
        }
    }
}
