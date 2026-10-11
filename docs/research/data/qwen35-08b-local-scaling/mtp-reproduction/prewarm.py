import asyncio,json,time,sys
from pathlib import Path
import aiohttp
from benchmarks.llm_serving.adapters import Adapter
from benchmarks.llm_serving.runner import request_once
from benchmarks.qwen35.dense_scaling import server,CONCURRENCIES
engine,profile,port=sys.argv[1:]
async def main():
 config=server(engine,int(port),speculative='mtp' in profile);adapter=Adapter(engine,'qwen08');evidence=[]
 async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=300)) as session:
  for c in CONCURRENCIES:
   workload=json.loads(Path(f'/tmp/tensor-qwen08/workloads/train/c{c}.json').read_text());origin=time.perf_counter()
   rows=await asyncio.gather(*(request_once(session,adapter,config,dict(request,output_tokens=64),origin=origin,offered=origin,seed=20261013+i,phase='concurrency-prewarm',timeout=300) for i,request in enumerate(workload['requests'][:c])))
   if any(row['status']!='completed' for row in rows):raise RuntimeError(f'prewarm C{c} failed')
   evidence.append(dict(concurrency=c,requests=rows));print('prewarm',profile,c,flush=True)
 Path(f'/tmp/tensor-qwen08/{profile}-concurrency-prewarm.json').write_text(json.dumps(evidence)+'\n')
asyncio.run(main())
