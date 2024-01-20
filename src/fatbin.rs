use std::{
    collections::HashMap,
    ffi::c_char,
    io::BufRead,
    io::BufReader,
    path::PathBuf,
    process::{Command, Stdio},
};

use libc::{
    c_void, strlen, Elf64_Ehdr, EI_CLASS, EI_MAG0, EI_MAG1, EI_MAG2, EI_MAG3, ELFCLASS64, ELFMAG0,
    ELFMAG1, ELFMAG2, ELFMAG3,
};
use parse_int::parse;

use crate::handles::FatBinPtr;

#[derive(Debug, Copy, Clone)]
pub struct CUfuncParamSize {
    pub size: usize,
    pub offset: usize,
}

const FATBIN_MAGIC: usize = 0x00100001ba55ed50;

#[allow(dead_code)]
pub fn is_elf(img: *const c_void) -> bool {
    let mag: *const u8 = img as *const u8;
    unsafe {
        *mag.add(EI_MAG0) == ELFMAG0
            && *mag.add(EI_MAG1) == ELFMAG1
            && *mag.add(EI_MAG2) == ELFMAG2
            && *mag.add(EI_MAG3) == ELFMAG3
    }
}

pub fn is_fatbin(img: *const c_void) -> bool {
    unsafe { *(img as *const usize) == FATBIN_MAGIC }
}

#[allow(dead_code)]
fn img_elf_parse_len(elf: *const c_void) -> usize {
    let hdr_ptr: *const Elf64_Ehdr = elf as *const Elf64_Ehdr;
    let hdr = unsafe { &*hdr_ptr };
    assert!(hdr.e_ident[EI_CLASS] == ELFCLASS64);
    /* Assume that the program header terminates the object. */
    hdr.e_phoff as usize + hdr.e_phnum as usize + hdr.e_phentsize as usize
}

fn img_fatbin_parse_len(fatbin: *const c_void) -> usize {
    /*
     * The size remaining after the 16-byte header is in the second 64-bit
     * integer.
     */
    let fatbin_ptr: *const u64 = unsafe { std::mem::transmute(fatbin) };
    let size = unsafe { *fatbin_ptr.add(1) };
    16 + size as usize
}

/*
 * See cuModuleLoadData for a description of image. It may be an ELF object, a
 * PTX string, or a fatbin object.
 */
#[allow(dead_code)]
pub fn img_parse_len(img: *const c_void) -> usize {
    unsafe {
        if is_elf(img) {
            img_elf_parse_len(img)
        } else if is_fatbin(img) {
            img_fatbin_parse_len(img)
        } else {
            strlen(img as *const c_char) + 1
        }
    }
}

pub fn parse_func_sigs_from_fatbin(fatbin: FatBinPtr) -> HashMap<String, Vec<CUfuncParamSize>> {
    assert!(is_fatbin(fatbin.0));

    let len = img_fatbin_parse_len(fatbin.0);
    let tmp_file = PathBuf::from(format!("/tmp/{:#x}.image", fatbin.0 as usize));
    let tmp_file = tmp_file.as_path();
    let tmp_file = tmp_file.to_str().unwrap();
    let tmp_file = std::ffi::CString::new(tmp_file).unwrap();
    let dump_fd = unsafe { libc::open(tmp_file.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644) };
    if dump_fd == -1 {
        panic!("failed to open");
    }
    let ret = unsafe { libc::write(dump_fd, fatbin.0, len) };
    if ret == -1 {
        panic!("failed to write");
    }
    if unsafe { libc::close(dump_fd) } == -1 {
        panic!("failed to close");
    }

    let stdout = Command::new("cuobjdump")
        .arg("-elf")
        .arg(tmp_file.to_str().unwrap())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap()
        .stdout
        .unwrap();

    let reader = BufReader::new(stdout);
    let mut fn_sigs: HashMap<String, Vec<CUfuncParamSize>> = HashMap::new();
    let mut lines = reader.lines().peekable();

    while lines.peek().is_some() {
        let line = lines.next().unwrap().unwrap();

        // .nv.info._Z5saxpyifPfS_
        let fn_name = match line.strip_prefix(".nv.info.") {
            Some(fn_name) if !fn_name.trim().is_empty() => fn_name.trim().to_owned(),
            _ => continue,
        };

        fn_sigs.insert(fn_name.clone(), Vec::new());

        while lines.peek().is_some() {
            let line = lines.next().unwrap().unwrap();

            if line.trim().is_empty() {
                break;
            }

            if !line.contains("EIATTR_KPARAM_INFO") {
                continue;
            }

            _ = lines.next().unwrap().unwrap();

            let line = lines.next().unwrap().unwrap();

            // Value:  Index : 0x0     Ordinal : 0x3   Offset  : 0x10  Size    : 0x8
            // ["Value:", "Index", ":", "0x0", "Ordinal", ":", "0x3", "Offset", ":", "0x10", "Size", ":", "0x8"]

            let words: Vec<&str> = line.split_whitespace().collect();
            let offset = parse::<usize>(words[9]).unwrap();
            let size = parse::<usize>(words[12]).unwrap();

            // logo!("parsed {:#x} {:#x} {:#x}", index, offset, size);
            fn_sigs
                .get_mut(&fn_name)
                .unwrap()
                .push(CUfuncParamSize { size, offset });
        }

        fn_sigs.get_mut(&fn_name).unwrap().reverse();
    }

    fn_sigs
}
