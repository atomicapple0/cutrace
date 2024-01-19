use colored::Colorize;
use cudarc::driver::sys::CUmodule;

#[repr(C)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct PrettyCUmodule(pub CUmodule);

impl std::fmt::Debug for PrettyCUmodule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUmod_{:x?}", self.0);
        write!(f, "{}", s.truecolor(145, 247, 33))
    }
}
