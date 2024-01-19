use colored::Colorize;
use cudarc::driver::sys::CUdevice;

#[repr(C)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct PrettyCUdevice(pub CUdevice);

impl std::fmt::Debug for PrettyCUdevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUctx_{:x?}", self.0);
        write!(f, "{}", s.truecolor(33, 33, 147))
    }
}
