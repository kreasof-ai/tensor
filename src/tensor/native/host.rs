// Standalone Rust CPU host for Tensor ABI 1. No Python or compiler runtime.
use std::ffi::{c_char,c_void,CString};
use std::mem::{size_of,transmute};
use std::sync::atomic::{AtomicU64,Ordering};
static IDENTITIES:AtomicU64=AtomicU64::new(1);
#[repr(C)] struct Buffer { address:u64,byte_size:u64,shape:*const i64,strides:*const i64,
  rank:u32,dtype:u32,device_type:u32,device_ordinal:i32 }
#[repr(C)] struct Argument { kind:u32,dtype:u32,buffer:Buffer,scalar:u64 }
#[repr(C)] struct Stream { device_type:u32,device_ordinal:i32,handle:u64 }
#[repr(C)] struct Call { abi_version:u32,struct_size:u32,arguments:*const Argument,argument_count:u32,flags:u32,
  grid:[u32;3],block:[u32;3],shared_memory_bytes:u64,stream:Stream }
#[repr(C)] struct Error { code:i32,message:[c_char;508] }
#[derive(Clone,Copy,PartialEq)]
#[repr(C)] struct Workspace { byte_size:u64,alignment:u32,device_type:u32,flags:u32,reserved:u32 }
#[derive(Clone,Copy)]
#[repr(C)] struct Executable { abi_version:u32,struct_size:u32,device_type:u32,device_ordinal:i32,
  session:u64,handle:u64,argument_count:u32,flags:u32,workspace:Workspace }
#[repr(C)] struct Event { abi_version:u32,struct_size:u32,device_type:u32,device_ordinal:i32,
  session:u64,handle:u64,flags:u32,reserved:u32 }
#[repr(C,align(64))] struct Storage([f32;129]);
#[link(name="dl")] unsafe extern "C" {
  fn dlopen(name:*const c_char,flags:i32)->*mut c_void;
  fn dlsym(handle:*mut c_void,name:*const c_char)->*mut c_void;
  fn dlclose(handle:*mut c_void)->i32;
}
type Kernel=unsafe extern "C" fn(*const Call,*mut Error)->i32;
struct Resource { descriptor:Executable,function:Option<Kernel> }
impl Resource {
  fn lookup(&self,snapshot:&Executable)->Option<Kernel> {
    let d=&self.descriptor;
    if snapshot.abi_version!=1 || snapshot.struct_size<64 || snapshot.device_type!=d.device_type ||
       snapshot.device_ordinal!=d.device_ordinal || snapshot.session!=d.session || snapshot.handle!=d.handle ||
       snapshot.argument_count!=d.argument_count || snapshot.flags!=d.flags || snapshot.workspace!=d.workspace {
      None
    } else {self.function}
  }
}
fn main() {
  assert_eq!([size_of::<Buffer>(),size_of::<Argument>(),size_of::<Stream>(),size_of::<Call>(),size_of::<Error>()],
             [48,64,16,72,512]);
  assert_eq!([size_of::<Workspace>(),size_of::<Executable>()],[24,64]);
  assert_eq!(size_of::<Event>(),40);
  let path=CString::new(std::env::args().nth(1).expect("kernel.so path")).unwrap();
  let name=CString::new("tensor_kernel_v1").unwrap();
  let library=unsafe { dlopen(path.as_ptr(),2) }; assert!(!library.is_null());
  let function=unsafe { dlsym(library,name.as_ptr()) }; assert!(!function.is_null());
  let mut resource=Resource{descriptor:Executable{abi_version:1,struct_size:64,device_type:1,device_ordinal:0,
    session:IDENTITIES.fetch_add(1,Ordering::Relaxed),handle:IDENTITIES.fetch_add(1,Ordering::Relaxed),
    argument_count:3,flags:0,workspace:Workspace{byte_size:0,alignment:1,device_type:1,flags:0,reserved:0}},
    function:Some(unsafe{transmute(function)})};
  let mut invalid=resource.descriptor;invalid.workspace.byte_size=16;
  assert!(resource.lookup(&invalid).is_none());
  invalid=resource.descriptor;invalid.session=2;assert!(resource.lookup(&invalid).is_none());
  let run=resource.lookup(&resource.descriptor).unwrap();
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
  resource.function=None;assert!(resource.lookup(&resource.descriptor).is_none());
  assert_eq!(unsafe{dlclose(library)},0);
  println!("{{\"status\":\"passed\",\"provider\":\"cpu\",\"abi\":1,\"minor\":1,\"workspace_bytes\":0,\"elements\":129,\"negative_checks\":5}}");
}
