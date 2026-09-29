"""Compare NVRTC/nvcc builds and compiler-free host launch timing.

Uses fresh compiler processes and independent cold/warm caches. CUDA timing
includes host binding and stream synchronization; it is not GPU kernel duration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=("elementwise","gemm_relu","dynamic_affine","dynamic_gemm","scalar_offset")


def invoke(command):
    start=time.perf_counter()
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=300)
    return json.loads(result.stdout),time.perf_counter()-start


def measure(runtime_python,out,target,nvcc):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in EXAMPLES:
        for compiler in ("nvrtc","nvcc"):
            common=[sys.executable,"-m","tensor","build",str(ROOT / f"examples/{name}.py"),
                    "--target",target,"--compiler",compiler,"--cache-dir",str(out / f"cache-{compiler}")]
            if compiler=="nvcc":common += ["--nvcc",nvcc]
            artifact=out / f"{name}-{compiler}.tbin"
            cold,cold_wall=invoke([*common,"--out",str(artifact)])
            warm,warm_wall=invoke([*common,"--out",str(out / f"{name}-{compiler}-warm.tbin")])
            if cold["cache_hit"] or not warm["cache_hit"]:
                raise RuntimeError("cold/warm cache states not observed")
            code='''
import json,sys
import numpy as np
import tensor as tx
name=sys.argv[2]
with tx.Device() as d:
 k=d.load(sys.argv[1])
 if name=='elementwise': args=(d.arange(129),d.ones((129,))); kw={}
 elif name=='dynamic_affine': args=(d.arange(129),d.ones((129,))); kw={'scale':2.5}
 elif name=='scalar_offset': args=(d.arange(129,dtype='int64'),); kw={'delta':(1<<40)+3}
 elif name=='gemm_relu': args=(d.ones((64,64),'float16'),d.ones((64,64),'float16'),d.ones((64,),'float16')); kw={}
 else: args=(d.ones((33,32),'float16'),d.ones((32,32),'float16')); kw={}
 buffers,dims,outputs=k.prepare(*args,**kw)
 k.launch(*buffers,**dims)
 snapshots=[value.to_numpy().tolist() for value in outputs.values()]
 result=tx.bench(k,buffers,warmup=10,iters=100,**dims)
 result['outputs']=snapshots
assert not {'torch','tilelang','tvm','tvm_ffi'} & sys.modules.keys()
print(json.dumps(result))
'''
            timing,_=invoke([str(runtime_python),"-c",code,str(artifact),name])
            rows.append({"example":name,"compiler":compiler,"cold_build_seconds":cold["seconds"],
                "warm_build_seconds":warm["seconds"],"cold_process_seconds":cold_wall,
                "warm_process_seconds":warm_wall,"compile_seconds":cold["compile_seconds"],
                "artifact_bytes":cold["bytes"],"timing":timing})
    for i in range(0,len(rows),2):
        if rows[i]['timing'].pop('outputs') != rows[i+1]['timing'].pop('outputs'):
            raise RuntimeError("compiler outputs disagree")
    return {"status":"passed","target":target,"same_host":True,
            "outputs_equal":True,"metric":"host enqueue and launch-plus-stream-sync","records":rows}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--target",default="sm_86")
    parser.add_argument("--nvcc",required=True)
    args=parser.parse_args()
    result=measure(args.runtime_python.absolute(),args.out,args.target,args.nvcc)
    (args.out / "metrics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"cases":len(result["records"]),"out":str(args.out)}))
