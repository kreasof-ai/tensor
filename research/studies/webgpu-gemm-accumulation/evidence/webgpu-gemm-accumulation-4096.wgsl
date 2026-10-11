// Function: linear_kernel
//----------------------------------------
// Function: linear_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> cc : array<f32, 1024>;
var<workgroup> aa : array<f32, 512>;
var<workgroup> bb : array<f32, 512>;
@compute @workgroup_size(128, 1, 1)
fn linear_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var wgpu_gemm_loop_acc_1 : array<f32, 8>;
  for (var i : i32 = 0i; i < 2i; i++) {
    var cc_local_cast : array<f32, 4>;
    for (var vec : i32 = 0i; vec < 4i; vec++) {
      cc_local_cast[vec] = 0.000000e+00f;
    }
    for (var vec_copy : i32 = 0i; vec_copy < 4i; vec_copy++) {
      cc[(((i * 512i) + (i32(threadIdx.x) * 4i)) + vec_copy)] = cc_local_cast[vec_copy];
    }
  }
  workgroupBarrier();
  wgpu_gemm_loop_acc_1[0i] = cc[(((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i))];
  wgpu_gemm_loop_acc_1[1i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 1i)];
  wgpu_gemm_loop_acc_1[2i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 32i)];
  wgpu_gemm_loop_acc_1[3i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 33i)];
  wgpu_gemm_loop_acc_1[4i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 512i)];
  wgpu_gemm_loop_acc_1[5i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 513i)];
  wgpu_gemm_loop_acc_1[6i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 544i)];
  wgpu_gemm_loop_acc_1[7i] = cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 545i)];
  for (var tile : i32 = 0i; tile < 256i; tile++) {
    for (var i_1 : i32 = 0i; i_1 < 4i; i_1++) {
      aa[((i32(threadIdx.x) * 4i) + i_1)] = a[(((((i32(blockIdx.y) * 131072i) + ((i32(threadIdx.x)>>2u) * 4096i)) + (tile * 16i)) + ((i32(threadIdx.x) & 3i) * 4i)) + i_1)];
    }
    for (var i_2 : i32 = 0i; i_2 < 4i; i_2++) {
      bb[((i32(threadIdx.x) * 4i) + i_2)] = b[(((((v__1 * 131072i) + ((i32(threadIdx.x)>>2u) * 4096i)) + (tile * 16i)) + ((i32(threadIdx.x) & 3i) * 4i)) + i_2)];
    }
    workgroupBarrier();
    for (var wgpu_loop_k_1 : i32 = 0i; wgpu_loop_k_1 < 16i; wgpu_loop_k_1++) {
      wgpu_gemm_loop_acc_1[0i] = fma(aa[(((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1)], bb[(((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1)], wgpu_gemm_loop_acc_1[0i]);
      wgpu_gemm_loop_acc_1[1i] = fma(aa[(((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1)], bb[((((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1) + 16i)], wgpu_gemm_loop_acc_1[1i]);
      wgpu_gemm_loop_acc_1[2i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 16i)], bb[(((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1)], wgpu_gemm_loop_acc_1[2i]);
      wgpu_gemm_loop_acc_1[3i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 16i)], bb[((((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1) + 16i)], wgpu_gemm_loop_acc_1[3i]);
      wgpu_gemm_loop_acc_1[4i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 256i)], bb[(((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1)], wgpu_gemm_loop_acc_1[4i]);
      wgpu_gemm_loop_acc_1[5i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 256i)], bb[((((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1) + 16i)], wgpu_gemm_loop_acc_1[5i]);
      wgpu_gemm_loop_acc_1[6i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 272i)], bb[(((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1)], wgpu_gemm_loop_acc_1[6i]);
      wgpu_gemm_loop_acc_1[7i] = fma(aa[((((i32(threadIdx.x)>>4u) * 32i) + wgpu_loop_k_1) + 272i)], bb[((((i32(threadIdx.x) & 15i) * 32i) + wgpu_loop_k_1) + 16i)], wgpu_gemm_loop_acc_1[7i]);
    }
    workgroupBarrier();
  }
  cc[(((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i))] = wgpu_gemm_loop_acc_1[0i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 1i)] = wgpu_gemm_loop_acc_1[1i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 32i)] = wgpu_gemm_loop_acc_1[2i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 33i)] = wgpu_gemm_loop_acc_1[3i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 512i)] = wgpu_gemm_loop_acc_1[4i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 513i)] = wgpu_gemm_loop_acc_1[5i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 544i)] = wgpu_gemm_loop_acc_1[6i];
  cc[((((i32(threadIdx.x)>>4u) * 64i) + ((i32(threadIdx.x) & 15i) * 2i)) + 545i)] = wgpu_gemm_loop_acc_1[7i];
  workgroupBarrier();
  for (var i_3 : i32 = 0i; i_3 < 2i; i_3++) {
    for (var vec_1 : i32 = 0i; vec_1 < 4i; vec_1++) {
      out[((((((i32(blockIdx.y) * 131072i) + (i_3 * 65536i)) + ((i32(threadIdx.x)>>3u) * 4096i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 7i) * 4i)) + vec_1)] = cc[(((i_3 * 512i) + (i32(threadIdx.x) * 4i)) + vec_1)];
    }
  }
}


