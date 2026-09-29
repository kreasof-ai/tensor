// Standalone Rust CPU host for Tensor ABI 1. No Python or compiler runtime.
use std::ffi::{c_char,c_void,CString};
use std::mem::{size_of,transmute};
#[repr(C)] struct Buffer { address:u64,byte_size:u64,shape:*const i64,strides:*const i64,
  rank:u32,dtype:u32,device_type:u32,device_ordinal:i32 }
#[repr(C)] struct Argument { kind:u32,dtype:u32,buffer:Buffer,scalar:u64 }
#[repr(C)] struct Stream { device_type:u32,device_ordinal:i32,handle:u64 }
#[repr(C)] struct Call { abi_version:u32,struct_size:u32,arguments:*const Argument,argument_count:u32,flags:u32,
  grid:[u32;3],block:[u32;3],shared_memory_bytes:u64,stream:Stream }
#[repr(C)] struct Error { code:i32,message:[c_char;508] }
#[repr(C,align(64))] struct Storage([f32;129]);
#[link(name="dl")] unsafe extern "C" {
  fn dlopen(name:*const c_char,flags:i32)->*mut c_void;
  fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
  fn dlclose(handle:*mut c_void)->i32;
}
type Kernel=unsafe extern "C" fn(*const Call,*mut Error)->i32;
fn main() {
  assert_eq!([size_of::<Buffer>(),size_of::<Argument>(),size_of::<Stream>(),size_of::<Call>(),size_of::<Error>()],
             [48,64,16,72,512]);
  let path=CString::new(std::env::args().nth(1).expect("kernel.so path")).unwrap();
  let name=CString::new("tensor_kernel_v1").unwrap();
  let library=unsafe { dlopen(path.as_ptr(),2) }; assert!(!library.is_null());
  let function=unsafe { dlsym(library,name.as_ptr()) }; assert!(!function.is_null());
  let run:Kernel=unsafe { transmute(function) };
  let mut a=Storage([0.;129]);let b=Storage([1.;129]);let mut out=Storage([f32::NAN;129]);
  for(i,value)in a.0.iter_mut().enumerate(){*value=i as f32-64.;}
  let shape=129i64;let stride=4i64;
  let pointers=[a.0.as_ptr(),b.0.as_ptr(),out.0.as_mut_ptr()];
  let mut args:Vec<Argument>=pointers.iter().map(|pointer| Argument {kind:1,dtype:11,scalar:0,
    buffer:Buffer{address:*pointer as u64,byte_size:516,shape:&shape,strides:&stride,rank:1,dtype:11,
                  device_type:1,device_ordinal:0}}).collect();
  let mut call=Call{abi_version:1,struct_size:72,arguments:args.as_ptr(),argument_count:3,flags:0,
    grid:[1;3],block:[1;3],shared_memory_bytes:0,stream:Stream{device_type:1,device_ordinal:0,handle:0}};
  let mut error=Error{code:0,message:[0;508]};
  assert_eq!(unsafe{run(&call,&mut error)},0);
  for i in 0..129 { assert_eq!(out.0[i],(2.*a.0[i]+b.0[i]).max(0.)); }
  args[0].dtype=12;assert_eq!(unsafe{run(&call,&mut error)},2);args[0].dtype=11;
  call.abi_version=2;assert_eq!(unsafe{run(&call,&mut error)},1);
  assert_eq!(unsafe{dlclose(library)},0);
  println!("{{\"status\":\"passed\",\"provider\":\"cpu\",\"abi\":1,\"elements\":129,\"negative_checks\":2}}");
}
