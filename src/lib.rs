#![feature(c_str_literals)]

#[allow(non_snake_case)]
mod cudadrv;
#[allow(non_snake_case)]
mod cudart;
mod fatbin;
mod handles;
mod log;
mod symbol_loader;

use std::ffi::CStr;

use once_cell::sync::Lazy;
use symbol_loader::SymbolLoader;

static START_TIME: Lazy<std::time::Instant> = Lazy::new(|| std::time::Instant::now());

const CUDADRV_PATH: &CStr = c"libcuda.so.535.86.10";
const CUDADRT_PATH: &CStr = c"libcudart.so.12";

static CUDADRV_SYMBOLS: Lazy<SymbolLoader> = Lazy::new(|| SymbolLoader::new(CUDADRV_PATH));
static CUDART_SYMBOLS: Lazy<SymbolLoader> = Lazy::new(|| SymbolLoader::new(CUDADRT_PATH));

#[macro_export]
macro_rules! print_refs {
    ($($ref_arg: ident),*) => {
        $(
            $crate::log!("  > *.{} = {:x?}\n", stringify!($ref_arg), unsafe {
                *$ref_arg
            });
        )*
    };
}

#[macro_export]
macro_rules! gen {
    ($symbols: path, $fn_name: ident($($arg: ident: $arg_ty: ty),*) -> $ret_ty: ty, $block:block) => {
        #[no_mangle]
        pub extern "C" fn $fn_name($($arg: $arg_ty),*) -> $ret_ty {
            use std::io::Write;
            use std::io::stdout;
            $crate::log!("{}", stringify!($fn_name));
            stdout().flush().unwrap();
            let args : &[String] = &[$(
                format!(".{}={:?}", stringify!($arg), $arg),
            )*];
            $crate::log!("({}) = ", args.join(", "));
            stdout().flush().unwrap();

            let before = std::time::Instant::now();
            let sym = $symbols.get_symbol(stringify!($fn_name));
            let fn_ptr: extern "C" fn($($arg_ty),*) -> $ret_ty = unsafe { ::core::mem::transmute(sym) };
            let res = fn_ptr($($arg),*);
            
            let after = std::time::Instant::now();
            let start = before.duration_since(*$crate::START_TIME);
            let elapsed = after.duration_since(before);
            $crate::log!("{:?}\t[{:?}ms,{:?}ms]\n", res, start.as_millis(), elapsed.as_millis());

            $block

            res
        }
    };
}

#[macro_export]
macro_rules! cudadrv_fn {
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty) => {
        gen!($crate::CUDADRV_SYMBOLS, $fn_name($($arg: $arg_ty),*) -> $ret_ty, {});
    };
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty, $block:block) => {
        gen!($crate::CUDADRV_SYMBOLS, $fn_name($($arg: $arg_ty),*) -> $ret_ty, $block);
    }
}
#[macro_export]
macro_rules! cudart_fn {
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty) => {
        gen!($crate::CUDART_SYMBOLS, $fn_name($($arg: $arg_ty),*) -> $ret_ty, {});
    };
    ($fn_name: ident($($arg: ident: $arg_ty: ty),* $(,)?) -> $ret_ty: ty, $block:block) => {
        gen!($crate::CUDART_SYMBOLS, $fn_name($($arg: $arg_ty),*) -> $ret_ty, $block);
    }
}
