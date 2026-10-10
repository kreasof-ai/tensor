"""Probe H200 hardware counters separately from serving measurements."""
import modal
from benchmarks.qwen35.modal_h200 import base_image,with_workspace,volume

app=modal.App('tensor-qwen35-h200-counters')
profiler_image=with_workspace(base_image.run_commands(
    'chmod 1777 /tmp',
    "sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources"
).apt_install('cuda-nsight-compute-12-9'))


@app.function(image=profiler_image,gpu='H200',cpu=8,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def profile(prepared):
    import hashlib,json,os,shutil,subprocess,sys
    from pathlib import Path
    os.chdir('/workspace');volume.reload()
    ncu=shutil.which('ncu') or next((str(p) for p in Path('/opt/nvidia/nsight-compute').glob('*/ncu')),None)
    if ncu is None:raise RuntimeError('Nsight Compute CLI missing')
    identity=hashlib.sha256(json.dumps(prepared,sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/counters')/identity;root.mkdir(parents=True,exist_ok=True)
    metrics='dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active'
    result=dict(scope='isolated hardware counters; not serving throughput',cases=[],
        nsight_compute=subprocess.check_output([ncu,'--version'],text=True),prepared=prepared)
    cases=[('copy',prepared['copy'],True)]
    cases += [(f'attention-q{p["schedule"]["query_rows"]}-t{p["schedule"]["threads"]}-j{int(p["schedule"]["joint_kv"])}',p['path'],False)
              for p in prepared['candidates'] if p['compiled']]
    for name,artifact,copy in cases:
        command=[ncu,'--profile-from-start','off','--launch-count','1','--clock-control','none',
                 '--cache-control','none','--metrics',metrics,'--csv','--page','raw',
                 '--export',str(root/name),sys.executable,'-m','benchmarks.qwen35.counter_inputs','--artifact',artifact]
        if copy:command.append('--copy')
        completed=subprocess.run(command,text=True,capture_output=True,timeout=180)
        log=completed.stdout+completed.stderr;(root/(name+'.log')).write_text(log)
        row=dict(name=name,command=command,exit_code=completed.returncode,output=log)
        result['cases'].append(row);print('Counter probe',name,completed.returncode,log[-3500:],flush=True)
        if 'ERR_NVGPUCTRPERM' in log:
            result['hardware_counters_available']=False;break
    else:result['hardware_counters_available']=all(row['exit_code']==0 for row in result['cases'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main(prepared_file:str,out:str='build/qwen35-h200-counters.json'):
    import json
    from pathlib import Path
    result=profile.remote(json.loads(Path(prepared_file).read_text()))
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
