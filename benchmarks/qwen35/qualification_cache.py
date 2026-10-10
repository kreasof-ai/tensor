"""Reuse successful physical kernel checks only for identical source and tools."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


def read_proof(root,fingerprint):
    try:
        record=json.loads((root/'proof.json').read_text())
        log=(root/'kernel-tests.log').read_text()
    except (OSError,ValueError):return None
    if (not isinstance(record,dict) or record.get('fingerprint')!=fingerprint or record.get('exit_code')!=0
            or record.get('log_sha256')!=hashlib.sha256(log.encode()).hexdigest()):return None
    return record,log


def qualify(command):
    from importlib.metadata import version
    files=sorted(Path('src/tensor').rglob('*.py'))
    files+=sorted(Path('packages/tensor-llm').rglob('*.py'))
    files+=sorted(Path('benchmarks/qwen35').glob('*.py'))
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap,driver_version',
                             '--format=csv,noheader'],check=True,text=True,capture_output=True).stdout.strip()
    if not hardware.startswith('NVIDIA H200, 9.0,'):raise RuntimeError('physical H200 qualification required')
    fingerprint=dict(command=command,hardware=hardware,nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME'),
        packages={name:version(name) for name in ('tilelang','torch','numpy','pytest','apache-tvm-ffi')},
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    root=Path('/cache/kernel-qualification')/key
    retained=read_proof(root,fingerprint)
    if retained:
        record,log=retained
        print('Reusing source-bound physical H200 checks',key,flush=True)
        return SimpleNamespace(returncode=0,stdout=log,stderr=''),dict(
            key=key,source='retained',path=str(root),log_sha256=record['log_sha256'])
    result=subprocess.run(command,env=dict(os.environ,TENSOR_QWEN_CUDA='1'),text=True,capture_output=True)
    log=result.stdout+result.stderr
    if result.returncode==0:
        root.mkdir(parents=True,exist_ok=True)
        (root/'kernel-tests.log').write_text(log)
        record=dict(fingerprint=fingerprint,exit_code=0,log_sha256=hashlib.sha256(log.encode()).hexdigest())
        temp=root/'proof.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(root/'proof.json')
    return result,dict(key=key,source='executed',path=str(root),log_sha256=hashlib.sha256(log.encode()).hexdigest())
