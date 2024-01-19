use std::ffi::c_char;

use colored::Colorize;

pub type ConstCharStar = *const c_char;

#[repr(C)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct PrettyConstCharStar(pub ConstCharStar);

impl std::fmt::Debug for PrettyConstCharStar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cstr = unsafe { std::ffi::CStr::from_ptr(self.0) };
        let s = format!("{:?}", cstr);
        write!(f, "{}", s.truecolor(128, 128, 10))
    }
}
