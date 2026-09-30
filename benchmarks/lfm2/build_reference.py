"""Build the pinned llama.cpp CUDA baseline and matched Linux API helper."""
import argparse
import os
from pathlib import Path
import subprocess

COMMIT='f7b384c1e5c5b2c5b321a4a7cefea04b15b54cb7'


def build(root,cuda_root,arch,jobs):
    if os.name!='posix':raise RuntimeError('this measured baseline recipe requires Linux')
    source=root/'llama-cpp';binary=source/'out'
    root.mkdir(parents=True,exist_ok=True)
    if not source.exists():subprocess.run(['git','clone','https://github.com/ggml-org/llama.cpp',str(source)],check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',COMMIT],check=True)
    command=['cmake','-S',str(source),'-B',str(binary),'-G','Ninja','-DGGML_CUDA=ON',
        '-DCMAKE_CUDA_ARCHITECTURES='+arch,'-DCMAKE_BUILD_TYPE=Release','-DLLAMA_BUILD_TESTS=OFF',
        '-DLLAMA_BUILD_EXAMPLES=OFF','-DCUDAToolkit_ROOT='+str(cuda_root),
        '-DCMAKE_CUDA_COMPILER='+str(cuda_root/'bin/nvcc')]
    for name in ('cublas','cublasLt'):
        library=cuda_root/f'lib64/lib{name}.so.12'
        if library.exists():command.append(f'-DCUDA_{name}_LIBRARY={library}')
    subprocess.run(command,check=True)
    subprocess.run(['cmake','--build',str(binary),'--target','llama-bench','llama-cli','-j',str(jobs)],check=True)
    subprocess.run(['g++','-O2','-std=c++20','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),
        '-I'+str(source/'vendor'),str(Path(__file__).with_name('reference.cpp')),'-L'+str(binary/'bin'),
        '-lllama','-lggml-base','-lggml','-Wl,-rpath,$ORIGIN/llama-cpp/out/bin',
        '-Wl,-rpath-link,'+str(cuda_root/'lib64'),'-o',str(root/'lfm2-reference')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('build'))
    p.add_argument('--cuda-root',type=Path,required=True);p.add_argument('--arch',default='86');p.add_argument('--jobs',type=int,default=8)
    args=p.parse_args();build(args.out.resolve(),args.cuda_root.resolve(),args.arch,args.jobs)
