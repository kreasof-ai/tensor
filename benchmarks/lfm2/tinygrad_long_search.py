"""Sequential, checkpointed projection search with a combined wall-clock cap."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def run(model,bundle,out,fallback_cache,minutes):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();deadline=started+minutes*60;stages=[]
    # Broader default action space, then a wider beam if convergence leaves time.
    configurations=[(4,64,512,300),(8,128,1024,600),(16,256,1024,600)]
    report={'status':'searching','combined_budget_seconds':minutes*60,'stages':stages}
    def save():
        report['wall_seconds']=time.perf_counter()-started
        (out/'search-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for width,upcast,local,nominal in configurations:
        for weight in ('ffn_gate','ffn_down'):
            remaining=deadline-time.perf_counter()
            if remaining<30:break
            # Preserve some time for the other projection in the final tier.
            budget=min(nominal,remaining/2-10 if weight=='ffn_gate' else remaining-20)
            if budget<5:break
            name=f'{weight}-beam{width}-up{upcast}-local{local}';directory=out/name;directory.mkdir(exist_ok=True)
            env={**os.environ,'DEV':'CL','BEAM':'0','JITBEAM':str(width),'BEAM_ESTIMATE':'0',
                 'BEAM_UPCAST_MAX':str(upcast),'BEAM_LOCAL_MAX':str(local),'PARALLEL':'0',
                 'CACHEDB':str(directory/'cache.db')}
            command=[sys.executable,'benchmarks/lfm2/tinygrad_projection_compare.py','--model',str(model),'--bundle',str(bundle),
                     '--out',str(directory),'--rows','32','--weights',weight,'--search-seconds',str(budget),
                     '--fallback-cache',str(fallback_cache)]
            stage={'name':name,'beam':width,'upcast_limit':upcast,'local_limit':local,'search_budget_seconds':budget,
                   'started_seconds':time.perf_counter()-started};stages.append(stage);save()
            print('starting',stage,flush=True)
            with (directory/'run.log').open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                try:code=process.wait(timeout=min(deadline-time.perf_counter(),budget+30))
                except subprocess.TimeoutExpired:
                    # Windows venv launchers may own a second Python process.
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False)
                    else:process.kill()
                    process.wait();code=None;stage['hard_timeout']=True
            stage['exit_code']=code;stage['finished_seconds']=time.perf_counter()-started
            path=directory/'report.json'
            if path.exists():stage['report']=json.loads(path.read_text())
            checkpoint=directory/f'{weight}-r32-search/search-progress.json'
            if checkpoint.exists():stage['checkpoint']=json.loads(checkpoint.read_text())
            save();print('finished',name,'exit',code,'elapsed',round(stage['finished_seconds']-stage['started_seconds'],2),flush=True)
            if code not in (0,None):
                report['status']='failed';save();raise RuntimeError('search child failed: '+name)
    report['status']='finished';save();print('all searches finished',round(report['wall_seconds'],2),'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out','fallback-cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minutes',type=float,default=30)
    a=p.parse_args()
    if not 0<a.minutes<=30:p.error('minutes must be in (0,30]')
    run(a.model,a.bundle,a.out,a.fallback_cache,a.minutes)
