import json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path('/tmp/tensor-qwen08');config=json.loads((root/'mtp-comparison-launches.json').read_text())
for profile in sys.argv[1:]:
 info=config['profiles'][profile];env=os.environ|{'PYTHONPATH':'.:src:packages/tensor-llm/src'}
 for k,v in info['environment'].items():
  if k=='PATH_prepend':env['PATH']=os.pathsep.join(v+[env.get('PATH','')])
  else:env[k]=v
 engine=profile.split('-')[0];port=8101 if engine=='vllm' else 8102
 print('launch',profile,flush=True)
 with (root/f'{profile}-server.log').open('wb') as log:
  process=subprocess.Popen(info['command'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:
   for _ in range(300):
    if process.poll() is not None:raise RuntimeError(f'{profile}: server exited {process.returncode}')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:
      if response.status==200:break
    except Exception:pass
    time.sleep(2)
   else:raise RuntimeError(f'{profile}: startup timeout')
   subprocess.run([f'build/{engine}-env/bin/python',str(root/'prewarm.py'),engine,profile,str(port)],check=True,env=env)
   subprocess.run([f'build/{engine}-env/bin/python',str(root/'run_baseline.py'),engine,profile,str(port)],check=True,env=env)
  finally:
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGKILL);process.wait()
 print('stopped',profile,flush=True)
