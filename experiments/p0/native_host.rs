// Standalone Rust host against the pinned TVM FFI C ABI; no Python or Torch.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::mem::{align_of, size_of};
use std::ptr;

const FUNCTION: i32 = 68;
const MODULE: i32 = 73;
#[repr(C)] #[derive(Default)] struct Version { major: u32, minor: u32, patch: u32 }
#[repr(C)] struct ByteArray { data: *const c_char, size: usize }
#[repr(C)] #[derive(Clone, Copy)] struct Any { type_index: i32, padding: u32, payload: u64 }
impl Any {
    fn empty() -> Self { Self { type_index: 0, padding: 0, payload: 0 } }
    fn raw_str(s: &CString) -> Self { Self { type_index: 8, padding: 0, payload: s.as_ptr() as u64 } }
    fn object(kind: i32, p: *mut c_void) -> Self { Self { type_index: kind, padding: 0, payload: p as u64 } }
    fn tensor(t: &mut DLTensor) -> Self { Self { type_index: 7, padding: 0, payload: t as *mut DLTensor as u64 } }
}
struct Owned(Any);
impl Owned {
    fn view(&self) -> Any { self.0 }
    fn expect(self, kind: i32) -> Result<Self, String> {
        if self.0.type_index != kind { return Err(format!("FFI result type {} != {}", self.0.type_index, kind)); }
        Ok(self)
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        if self.0.type_index >= 64 && self.0.payload != 0 {
            unsafe { TVMFFIObjectDecRef(self.0.payload as *mut c_void); }
        }
    }
}
#[repr(C)] #[derive(Clone, Copy)] struct DLDevice { device_type: i32, device_id: i32 }
#[repr(C)] #[derive(Clone, Copy)] struct DLDataType { code: u8, bits: u8, lanes: u16 }
#[repr(C)] struct DLTensor {
    data: *mut c_void, device: DLDevice, ndim: i32, dtype: DLDataType,
    shape: *mut i64, strides: *mut i64, byte_offset: u64,
}
impl DLTensor {
    fn cpu_f32(data: &mut [f32], size: &mut i64) -> Self {
        Self { data: data.as_mut_ptr().cast(), device: DLDevice { device_type: 1, device_id: 0 },
            ndim: 1, dtype: DLDataType { code: 2, bits: 32, lanes: 1 },
            shape: size as *mut i64, strides: ptr::null_mut(), byte_offset: 0 }
    }
}
#[link(name = "tvm_ffi")]
unsafe extern "C" {
    fn TVMFFIGetVersion(out: *mut Version);
    fn TVMFFIFunctionGetGlobal(name: *const ByteArray, out: *mut *mut c_void) -> c_int;
    fn TVMFFIFunctionCall(func: *mut c_void, args: *mut Any, num_args: i32, out: *mut Any) -> c_int;
    fn TVMFFIObjectDecRef(handle: *mut c_void) -> c_int;
    fn TVMFFIErrorMoveFromRaised(out: *mut *mut c_void);
}
#[link(name = "dl")]
unsafe extern "C" { fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void; fn dlerror() -> *const c_char; }
fn ffi_error(code: i32) -> String {
    let mut handle = ptr::null_mut();
    unsafe { TVMFFIErrorMoveFromRaised(&mut handle); }
    if handle.is_null() { return format!("TVM FFI failure {code}, no raised error"); }
    // TVMFFIError begins with an object header, kind and message ByteArrays.
    let message = unsafe { &*((handle as *const u8).add(24 + size_of::<ByteArray>()) as *const ByteArray) };
    let text = if message.data.is_null() { String::new() } else {
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(message.data as *const u8, message.size) }).into_owned()
    };
    unsafe { TVMFFIObjectDecRef(handle); }
    format!("TVM FFI failure {code}: {text}")
}
fn global(name: &str) -> Result<Owned, String> {
    let name = ByteArray { data: name.as_ptr().cast(), size: name.len() };
    let mut handle = ptr::null_mut();
    let code = unsafe { TVMFFIFunctionGetGlobal(&name, &mut handle) };
    if code != 0 { return Err(ffi_error(code)); }
    if handle.is_null() { return Err("global function absent".into()); }
    Ok(Owned(Any::object(FUNCTION, handle)))
}
fn call(func: &Owned, args: &mut [Any]) -> Result<Owned, String> {
    let mut result = Any::empty();
    let code = unsafe { TVMFFIFunctionCall(func.0.payload as *mut c_void, args.as_mut_ptr(), args.len() as i32, &mut result) };
    if code != 0 { return Err(ffi_error(code)); }
    Ok(Owned(result))
}
fn load_library(path: &str) -> Result<(), String> {
    let path = CString::new(path).map_err(|e| e.to_string())?;
    let handle = unsafe { dlopen(path.as_ptr(), 2 | 256) }; // RTLD_NOW | RTLD_GLOBAL
    if handle.is_null() { return Err(unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy().into_owned()); }
    // Compiler registration tables remain loaded for the process lifetime.
    Ok(())
}
fn run(module_path: &str) -> Result<(), String> {
    let mut version = Version::default();
    unsafe { TVMFFIGetVersion(&mut version); }
    if (version.major, version.minor, version.patch) != (0, 1, 12) { return Err("unexpected TVM FFI version".into()); }
    if size_of::<Any>() != 16 || align_of::<Any>() != 8 || size_of::<DLTensor>() != 48 || size_of::<ByteArray>() != 16 {
        return Err("foreign ABI layout mismatch".into());
    }
    let loader = global("ffi.ModuleLoadFromFile")?;
    let path = CString::new(module_path).map_err(|e| e.to_string())?;
    let module = call(&loader, &mut [Any::raw_str(&path)])?.expect(MODULE)?;
    let get_function = global("ffi.ModuleGetFunction")?;
    let name = CString::new("run").unwrap();
    let function = call(&get_function, &mut [module.view(), Any::raw_str(&name), Any { type_index: 2, padding: 0, payload: 0 }])?.expect(FUNCTION)?;
    let mut size = 129_i64;
    let mut a: Vec<f32> = (0..size).map(|i| i as f32 / 8.0 - 12.0).collect();
    let mut b = vec![0.25_f32; size as usize];
    let mut c = vec![f32::NAN; size as usize];
    let (mut aa, mut bb, mut cc) = (DLTensor::cpu_f32(&mut a, &mut size), DLTensor::cpu_f32(&mut b, &mut size), DLTensor::cpu_f32(&mut c, &mut size));
    call(&function, &mut [Any::tensor(&mut aa), Any::tensor(&mut bb), Any::tensor(&mut cc)])?;
    for i in 0..size as usize {
        if !c[i].is_finite() || c[i] != (2.0 * a[i] + b[i]).max(0.0) { return Err(format!("numeric mismatch at {i}")); }
    }
    aa.dtype.bits = 16;
    let wrong_dtype = call(&function, &mut [Any::tensor(&mut aa), Any::tensor(&mut bb), Any::tensor(&mut cc)]).err()
        .ok_or("wrong dtype was accepted")?;
    if !wrong_dtype.contains("float32") { return Err(format!("unexpected error: {wrong_dtype}")); }
    println!("{{\"status\":\"passed\",\"size\":129,\"max_abs_error\":0,\"wrong_dtype_rejected\":true,\"ffi_version\":\"0.1.12\",\"any_bytes\":16,\"dltensor_bytes\":48}}");
    Ok(())
}
fn ir(compiler: &str, tilelang: &str, path: &str) -> Result<(), String> {
    load_library(compiler)?; load_library(tilelang)?;
    let json = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let load = global("node.LoadJSON")?;
    let save = global("node.SaveJSON")?;
    let raw = CString::new(json.as_bytes()).map_err(|e| e.to_string())?;
    let module = call(&load, &mut [Any::raw_str(&raw)])?;
    let encoded = call(&save, &mut [module.view()])?.expect(65)?;
    let bytes = unsafe { &*((encoded.0.payload as *const u8).add(24) as *const ByteArray) };
    let roundtrip = unsafe { std::slice::from_raw_parts(bytes.data as *const u8, bytes.size) };
    if roundtrip != json.as_bytes() { return Err("native IR roundtrip changed bytes".into()); }
    println!("{{\"status\":\"passed\",\"byte_identical\":true,\"ir_bytes\":{}}}", json.len());
    Ok(())
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.as_slice() {
        [_, mode, module] if mode == "run" => run(module),
        [_, mode, compiler, tilelang, path] if mode == "ir" => ir(compiler, tilelang, path),
        _ => Err("usage: native_host run module.so | ir compiler.so tilelang.so ir.json".into()),
    };
    if let Err(message) = result { eprintln!("{message}"); std::process::exit(1); }
}
