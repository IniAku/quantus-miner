//! Minimal OpenCL 1.2 API, loaded at runtime (libOpenCL.so.1 / OpenCL.dll), so
//! the binary needs no OpenCL SDK to build and still starts without a driver.
#![allow(non_camel_case_types, non_snake_case, clippy::too_many_arguments)]

use libloading::Library;
use std::ffi::c_void;
use std::sync::OnceLock;

pub type cl_int = i32;
pub type cl_uint = u32;
pub type cl_bitfield = u64;
pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_mem = *mut c_void;
pub type cl_event = *mut c_void;

pub const CL_SUCCESS: cl_int = 0;
pub const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 2;
pub const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: cl_uint = 0x1002;
pub const CL_DEVICE_NAME: cl_uint = 0x102B;
pub const CL_DEVICE_VENDOR: cl_uint = 0x102C;
pub const CL_PLATFORM_NAME: cl_uint = 0x0902;
pub const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183;
pub const CL_MEM_READ_WRITE: cl_bitfield = 1 << 0;
pub const CL_MEM_READ_ONLY: cl_bitfield = 1 << 2;
pub const CL_TRUE: cl_uint = 1;

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        pub struct Api {
            _lib: Library,
            $(pub $name: unsafe extern "system" fn($($arg),*) -> $ret,)*
        }
        impl Api {
            unsafe fn load(lib: Library) -> Result<Self, String> {
                $(let $name = *lib
                    .get::<unsafe extern "system" fn($($arg),*) -> $ret>(concat!(stringify!($name), "\0").as_bytes())
                    .map_err(|e| format!("OpenCL symbol {} missing: {e}", stringify!($name)))?;)*
                Ok(Self { _lib: lib, $($name,)* })
            }
        }
    };
}

api! {
    clGetPlatformIDs: fn(cl_uint, *mut cl_platform_id, *mut cl_uint) -> cl_int;
    clGetPlatformInfo: fn(cl_platform_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clGetDeviceIDs: fn(cl_platform_id, cl_bitfield, cl_uint, *mut cl_device_id, *mut cl_uint) -> cl_int;
    clGetDeviceInfo: fn(cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clCreateContext: fn(*const isize, cl_uint, *const cl_device_id, *const c_void, *mut c_void, *mut cl_int) -> cl_context;
    clCreateCommandQueue: fn(cl_context, cl_device_id, cl_bitfield, *mut cl_int) -> cl_command_queue;
    clCreateProgramWithSource: fn(cl_context, cl_uint, *const *const u8, *const usize, *mut cl_int) -> cl_program;
    clBuildProgram: fn(cl_program, cl_uint, *const cl_device_id, *const u8, *const c_void, *mut c_void) -> cl_int;
    clGetProgramBuildInfo: fn(cl_program, cl_device_id, cl_uint, usize, *mut c_void, *mut usize) -> cl_int;
    clCreateKernel: fn(cl_program, *const u8, *mut cl_int) -> cl_kernel;
    clCreateBuffer: fn(cl_context, cl_bitfield, usize, *mut c_void, *mut cl_int) -> cl_mem;
    clSetKernelArg: fn(cl_kernel, cl_uint, usize, *const c_void) -> cl_int;
    clEnqueueWriteBuffer: fn(cl_command_queue, cl_mem, cl_uint, usize, usize, *const c_void, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clEnqueueReadBuffer: fn(cl_command_queue, cl_mem, cl_uint, usize, usize, *mut c_void, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clEnqueueNDRangeKernel: fn(cl_command_queue, cl_kernel, cl_uint, *const usize, *const usize, *const usize, cl_uint, *const cl_event, *mut cl_event) -> cl_int;
    clReleaseMemObject: fn(cl_mem) -> cl_int;
    clReleaseKernel: fn(cl_kernel) -> cl_int;
    clReleaseCommandQueue: fn(cl_command_queue) -> cl_int;
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

pub fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| {
        let names: &[&str] = if cfg!(windows) {
            &["OpenCL.dll"]
        } else if cfg!(target_os = "macos") {
            &["/System/Library/Frameworks/OpenCL.framework/OpenCL"]
        } else {
            &["libOpenCL.so.1", "libOpenCL.so"]
        };
        let mut last = String::new();
        for name in names {
            match unsafe { Library::new(name) } {
                Ok(lib) => return unsafe { Api::load(lib) },
                Err(e) => last = format!("{name}: {e}"),
            }
        }
        Err(format!("OpenCL runtime not found ({last})"))
    })
    .as_ref()
    .map_err(|e| e.clone())
}

pub fn check(code: cl_int, what: &str) -> Result<(), String> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed (OpenCL error {code})"))
    }
}

pub fn info_string(
    f: unsafe extern "system" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> cl_int,
    obj: *mut c_void,
    param: cl_uint,
) -> String {
    let mut size = 0usize;
    unsafe {
        if f(obj, param, 0, std::ptr::null_mut(), &mut size) != CL_SUCCESS || size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        if f(obj, param, size, buf.as_mut_ptr() as *mut c_void, std::ptr::null_mut()) != CL_SUCCESS {
            return String::new();
        }
        String::from_utf8_lossy(&buf).trim_end_matches('\0').trim().to_string()
    }
}
