"""Fresh matched FP32 Tensor schedules versus the complete CLBlast routine.

Use the producer environment with PyOpenCL installed, or append the existing
CLBlast consumer's site-packages after importing Tensor and NumPy. GPU work is
sequential; inputs, outputs and compiled programs stay resident during timing.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.providers.webgpu import Device
from scripts.validation.webgpu_validation import specialize
from benchmarks.inference.clblast_comparison import (
    OpenCL, TimestampAdapter, check, host_samples, inputs, webgpu_measure,
)
from benchmarks.inference.webgpu_outer_product_search import source
from benchmarks.lfm2.tensor_projection_search import Timer, bind

ROOT = Path(__file__).resolve().parents[2]


def compiled(directory, name, text):
    path = directory / (name + '.py')
    path.write_text(text)
    artifact = path.with_suffix('.tbin')
    artifact.unlink(missing_ok=True)
    tensor.build(path, artifact, provider='webgpu', cache_dir=directory / 'cache')
    manifest, files = read_artifact(artifact)
    return artifact, dict(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        wgsl_sha256=hashlib.sha256(files['kernel.wgsl']).hexdigest(),
        launch=manifest['launch'],
        workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes'],
    )


def outer_measure(device, artifact, row, values, reference, flat=True):
    arrays = values if row['mode'] == 'linear' else values[:2]
    uploaded = [device.from_numpy(v.ravel() if flat else v) for v in arrays]
    output = device.full(row['m'] * row['n'] if flat else (row['m'],row['n']), np.nan)
    kernel = device.load(artifact)
    plan = bind(device, kernel, (*uploaded, output))
    timer = Timer(device, plan)
    try:
        quality = check(plan.launch(readback=output).reshape(reference.shape), reference)
        if not quality['passed']:
            raise AssertionError(quality)
        # Completed submissions, without output download or allocating output.
        completed = host_samples(plan.launch, device.synchronize, lambda _: None)
        for _ in range(3):
            timer.sample()
        samples = [timer.sample() * 1000 for _ in range(21)]
        return dict(correctness=quality, prepared_completed=completed,
                    gpu_interval=dict(median_ms=statistics.median(samples), samples_ms=samples))
    finally:
        timer.close()
        plan.close()
        kernel._dispose()
        output.release()
        for value in uploaded:
            value.release()


def heldout(device, directory, config):
    """Validate the selected schedule on tails and scales outside its search."""
    m,n,k=35,131,1027
    validations=[]
    for mode in ('gemm','linear'):
        artifact,metadata=compiled(directory,'heldout-'+mode,source(m,n,k,config,mode))
        kernel=device.load(artifact)
        try:
            for seed,scale in ((804,1),(819,.01),(835,1e-5)):
                rng=np.random.default_rng(seed)
                a=(rng.normal(size=(m,k))*scale).astype(np.float32)
                b=rng.normal(size=(n,k)).astype(np.float32)
                bias=(rng.normal(size=n)*scale).astype(np.float32)
                aa,bb=a.astype(np.float64),b.astype(np.float64)
                reference=aa@bb.T
                bound=(np.abs(aa)@np.abs(bb).T)*3e-6+1e-10
                if mode=='linear':
                    reference=np.maximum(reference+bias.astype(np.float64),0)
                    bound+=np.abs(bias.astype(np.float64))*3e-6
                arrays=(a,b,bias) if mode=='linear' else (a,b)
                uploaded=[device.from_numpy(v.ravel()) for v in arrays]
                output=device.full(m*n,np.nan)
                plan=bind(device,kernel,(*uploaded,output))
                try:
                    actual=plan.launch(readback=output).reshape(m,n)
                    delta=np.abs(actual-reference)
                    passed=bool(np.isfinite(actual).all() and np.all(delta<=bound))
                    validations.append(dict(mode=mode,seed=seed,scale=scale,passed=passed,
                        maximum_absolute_error=float(delta.max()),
                        maximum_error_to_bound_ratio=float(np.max(delta/bound)),artifact=metadata))
                    if not passed:raise AssertionError(validations[-1])
                finally:
                    plan.close();output.release()
                    for v in uploaded:v.release()
        finally:kernel._dispose()
    return validations


def run(directory, search_report, consumer_site=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if consumer_site:
        _sys.path.append(str(Path(consumer_site).resolve()))
    selected = json.loads(Path(search_report).read_text())['best']['config']
    cl = OpenCL(ROOT / 'build/clblast-build/Release/clblast.dll',
                ROOT / 'build/clblast-probe-build/Release/tensor_clblast_probe.dll')
    device = Device()
    device._adapter = TimestampAdapter(device._adapter)
    result = dict(
        schema='tensor.outer-product-clblast.v1', status='running',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        search_report_sha256=hashlib.sha256(Path(search_report).read_bytes()).hexdigest(),
        selected_config=selected, opencl=cl.info,
        clblast_parameters={f: cl.parameters(f) for f in ('Xgemm', 'XgemmDirect')},
        timestamp_period_ns=10, cases=[],
        protocol='Native F32 A[M,K] @ B[N,K].T, F32 accumulation/output; alpha=1 beta=0. '
                 'Linear adds F32 bias and ReLU. Identical seeded inputs per case. '
                 '20 warm and 45 completed calls; 21 outer-product and 20 generic/CLBlast GPU samples. Tensor prepared call '
                 'uses preallocated output; CLBlast preallocated includes internal transforms '
                 'and linear epilogue. GPU intervals differ: Tensor compute-pass timestamps, '
                 'CLBlast completed-start/stop queue barriers. Neither timed call downloads output.',
    )
    def save():
        (directory / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    shapes = [(s,s,s) for s in (512,1024,2048,4096)] + [(32,2560,1024),(32,1024,2560)]
    with device:
        result['webgpu'] = device.info
        result['heldout_validation']=heldout(device,directory,selected)
        for index, (m,n,k) in enumerate(shapes):
            for mode in ('gemm', 'linear'):
                name = f'float32-{m}-{n}-{k}-{mode}'
                row = dict(name=name, m=m, n=n, k=k, dtype='float32', mode=mode)
                values, reference = inputs(row, index*2)
                row['input_sha256'] = [hashlib.sha256(v.tobytes()).hexdigest() for v in values]
                constants = dict(M=m, N=n, K=k, DTYPE='float32', OUTPUT_DTYPE='float32',
                                 TRANSPOSE_B=True, USE_BIAS=mode=='linear', RELU=mode=='linear')
                generic, generic_metadata = compiled(directory, name+'-generic',
                    specialize((ROOT/'examples/webgpu_gemm.py').read_text(), constants))
                outer, outer_metadata = compiled(directory, name+'-outer', source(m,n,k,selected,mode))
                row['generic_artifact'] = generic_metadata
                row['outer_artifact'] = outer_metadata
                row['tensor_generic'] = webgpu_measure(device,generic,row,values,reference,10)
                row['tensor_outer'] = outer_measure(device,outer,row,values,reference)
                row['clblast'] = cl.measure(row,values,reference)
                if not row['clblast']['correctness']['passed']:
                    raise AssertionError('CLBlast failed reference check')
                result['cases'].append(row)
                save()
                print(name, 'generic',round(row['tensor_generic']['preallocated']['median_ms'],3),
                      'outer',round(row['tensor_outer']['prepared_completed']['median_ms'],3),
                      'CLBlast',round(row['clblast']['preallocated']['median_ms'],3),flush=True)
    result['status'] = 'finished'
    save()


def refine(directory, comparison, skinny_reports, consumer_site=None):
    """Repeat the large winner, ablate unrolling, and validate skinny winners."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if consumer_site:_sys.path.append(str(Path(consumer_site).resolve()))
    previous=json.loads(Path(comparison).read_text())
    cl=OpenCL(ROOT/'build/clblast-build/Release/clblast.dll',ROOT/'build/clblast-probe-build/Release/tensor_clblast_probe.dll')
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    result=dict(status='running',sources={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text())
        for p in (Path(__file__),ROOT/'src/tensor/compiler/webgpu.py',ROOT/'src/tensor/compiler/webgpu_lowering.py',ROOT/'src/tensor/compiler/webgpu_search.py')},
        comparison_sha256=hashlib.sha256(Path(comparison).read_bytes()).hexdigest(),
        clblast_dll_sha256=hashlib.sha256((ROOT/'build/clblast-build/Release/clblast.dll').read_bytes()).hexdigest(),
        clblast_commit=subprocess.check_output(['git','-C','build/clblast-source','rev-parse','HEAD'],text=True).strip(),
        large_repeats=[],skinny=[],heldout=[],opencl=cl.info,protocol=previous['protocol'])
    def save():(directory/'refinement.json').write_text(json.dumps(result,indent=2)+'\n')
    with device:
        result['webgpu']=device.info
        row=previous['cases'][6]
        values,reference=inputs(row,6)
        assert [hashlib.sha256(v.tobytes()).hexdigest() for v in values]==row['input_sha256']
        selected=previous['selected_config']
        paths={}
        for explicit in (False,True):
            paths[explicit],metadata=compiled(directory,'large-'+str(explicit),
                source(row['m'],row['n'],row['k'],{**selected,'explicit_unroll':explicit}))
        for iteration in range(2):
            observation=dict(iteration=iteration,config=selected,input_sha256=row['input_sha256'])
            # Alternate the Tensor variant order; CLBlast repeats its complete routine.
            for explicit in ((False,True) if iteration==0 else (True,False)):
                observation['explicit_'+str(explicit)]=outer_measure(device,paths[explicit],row,values,reference)
            observation['clblast']=cl.measure(row,values,reference)
            if iteration==0:
                generic=Path(comparison).parent/(row['name']+'-generic.tbin')
                observation['generic_prepared']=outer_measure(device,generic,row,values,reference,flat=False)
            if not observation['clblast']['correctness']['passed']:raise AssertionError('CLBlast reference')
            result['large_repeats'].append(observation);save()
            print('large repeat',iteration,{s:round(observation['explicit_'+str(s)]['prepared_completed']['median_ms'],3) for s in (False,True)},flush=True)
        for search_path,index in zip(skinny_reports,(4,5),strict=True):
            search=json.loads(Path(search_path).read_text());config=search['best']['config']
            result['heldout']+=heldout(device,directory,config)
            for mode_index in (0,1):
                row=previous['cases'][index*2+mode_index]
                values,reference=inputs(row,index*2)
                assert [hashlib.sha256(v.tobytes()).hexdigest() for v in values]==row['input_sha256']
                artifact,metadata=compiled(directory,row['name']+'-skinny',source(row['m'],row['n'],row['k'],config,row['mode']))
                observation=dict(name=row['name'],shape=[row['m'],row['n'],row['k']],mode=row['mode'],
                    config=config,input_sha256=row['input_sha256'],artifact=metadata,
                    search_report_sha256=hashlib.sha256(Path(search_path).read_bytes()).hexdigest())
                observation['tensor_outer']=outer_measure(device,artifact,row,values,reference)
                generic=Path(comparison).parent/(row['name']+'-generic.tbin')
                observation['tensor_generic']=webgpu_measure(device,generic,row,values,reference,10)
                observation['generic_prepared']=outer_measure(device,generic,row,values,reference,flat=False)
                observation['clblast']=cl.measure(row,values,reference)
                if not observation['clblast']['correctness']['passed']:raise AssertionError('CLBlast reference')
                result['skinny'].append(observation);save()
                print(row['name'],'skinny',round(observation['tensor_outer']['prepared_completed']['median_ms'],3),
                      'CLBlast',round(observation['clblast']['preallocated']['median_ms'],3),flush=True)
    result['status']='finished';save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--search-report',type=Path)
    p.add_argument('--consumer-site',type=Path)
    p.add_argument('--refine-comparison',type=Path)
    p.add_argument('--skinny-reports',type=Path,nargs=2)
    a = p.parse_args()
    if a.refine_comparison:
        if not a.skinny_reports:p.error('refinement needs two skinny reports')
        refine(a.out,a.refine_comparison,a.skinny_reports,a.consumer_site)
    else:
        if not a.search_report:p.error('comparison needs a search report')
        run(a.out,a.search_report,a.consumer_site)
