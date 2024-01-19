use colored::Colorize;
use cudarc::driver::sys::CUcontext;

#[repr(C)]
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct PrettyCUcontext(pub CUcontext);

impl std::fmt::Debug for PrettyCUcontext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = format!("CUctx_{:x?}", self.0);
        write!(f, "{}", s.truecolor(33, 247, 147))
    }
}
