"""Local native Tensor serving adapter for the common finite-replay harness.

The checkpoint and all request states belong to one owner thread. Chunked
prefill and decode use the same native resources. This fixed slot scheduler is
experimental while full-model numerical qualification remains open.
"""
import argparse,asyncio,time
import ctypes as ct
from dataclasses import dataclass,field
import hashlib,json
from pathlib import Path
import numpy as np


@dataclass
class Request:
    prompt:list[int]
    output:int
    events:asyncio.Queue=field(default_factory=lambda:asyncio.Queue(64))
    position:int=0
    generated:int=0
    next_token:int=0
    cancelled:bool=False


class NativeServer:
    def __init__(self,model,prefill,checkpoint,*,coalesce_seconds=.003,tokenizer=None,speculative=None):
        model._check();self.model=model;self.prefill=prefill
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer=Tokenizer.from_file(str(Path(checkpoint)/'tokenizer.json'))
        self.tokenizer=tokenizer
        self.pending=asyncio.Queue();self.slots=[None]*model.slots
        self.closed=False;self.coalesce_seconds=coalesce_seconds
        self.decode_steps=0;self.output_tokens=0;self.prefill_tokens=0
        self.speculative=speculative

    def info(self):
        target_bytes=sum(b.nbytes for b in self.model.weights.values())
        draft_bytes=sum(b.nbytes for b in self.speculative['draft'].owned_weights.values()) if self.speculative else 0
        return dict(engine='tensor',implementation='native-qwen35-experimental',
            slots=self.model.slots,context_capacity=self.model.context,
            resident_weight_bytes=target_bytes+draft_bytes,target_weight_bytes=target_bytes,draft_weight_bytes=draft_bytes,
            checkpoint_weight_bytes=self.model.checkpoint.weight_bytes,
            state_dtype='float32',kv_dtype=getattr(self.model,'kv_dtype','bfloat16'),weight_format='native-F8_E4M3-block128',
            speculative=bool(self.speculative),prefix_cache=False,cpu_offload='none',
            output_lookup=bool(self.speculative and self.speculative.get('output_lookup')),
            compact_expert_prefill=bool(getattr(self.prefill,'_compact_experts_installed',False)),
            hopper_prefill=bool(getattr(self.prefill,'_hopper_installed',False)),
            model_throughput_qualified=False,full_stress_target_reached=False)

    async def worker(self):
        model,prefill=self.model,self.prefill
        try:
            if self.speculative:
                await self.speculative_worker();return
            while not self.closed:
                if all(r is None for r in self.slots):
                    request=await self.pending.get();self.slots[0]=request
                    model.reset([0]);await asyncio.sleep(self.coalesce_seconds)
                for slot,current in enumerate(self.slots):
                    if current is not None and current.cancelled:self.slots[slot]=None
                    if self.slots[slot] is None:
                        try:request=self.pending.get_nowait()
                        except asyncio.QueueEmpty:continue
                        model.reset([slot]);self.slots[slot]=request
                prefill_slots=[(slot,r) for slot,r in enumerate(self.slots)
                    if r is not None and r.position<len(r.prompt)]
                if prefill_slots:
                    block=np.full((model.slots,prefill.chunk),-1,'int32');lengths=np.zeros(model.slots,'int32')
                    for slot,r in prefill_slots:
                        count=min(prefill.chunk,len(r.prompt)-r.position)
                        block[slot,:count]=r.prompt[r.position:r.position+count];lengths[slot]=count
                    predicted=prefill.forward(block,lengths)
                    self.prefill_tokens+=int(lengths.sum())
                    for slot,r in prefill_slots:
                        r.position+=int(lengths[slot]);r.next_token=int(predicted[slot])
                        if r.position==len(r.prompt):await self.publish(slot,r)
                ready=[(slot,r) for slot,r in enumerate(self.slots)
                    if r is not None and r.position==len(r.prompt)]
                if ready:
                    tokens=np.full(model.slots,-1,'int32')
                    for slot,r in ready:tokens[slot]=r.next_token
                    predicted=model.forward(tokens);self.decode_steps+=1
                    for slot,r in ready:
                        r.next_token=int(predicted[slot]);await self.publish(slot,r)
                # Permit incoming requests, SSE writers and telemetry to run on
                # the owner event loop between GPU calls. Never move model
                # execution into a thread pool or duplicate its checkpoint.
                await asyncio.sleep(0)
        except asyncio.CancelledError:raise
        except BaseException as error:
            self.closed=True
            failed=[r for r in self.slots if r is not None]
            while not self.pending.empty():failed.append(self.pending.get_nowait())
            for r in failed:await r.events.put(dict(error=f'{type(error).__name__}: {error}'))
            raise

    async def speculative_worker(self):
        from tensor_llm import Qwen35Prefill,Qwen35Verifier,Qwen35Speculative
        from tensor_llm.qwen35.mtp.prefill import Qwen35MTPPrefill
        from .spec_run import install_selected
        model=self.model;config=self.speculative;draft=config['draft']
        verifier=repair=draft_prefill=None
        try:
            while not self.closed:
                request=await self.pending.get();self.slots[0]=request
                model.reset();draft.reset();await asyncio.sleep(self.coalesce_seconds)
                if self.prefill.closed:
                    self.prefill=Qwen35Prefill(model,config['prefill_bundle'])
                    if config.get('compact_experts'):
                        from tensor_llm.qwen35.compact_prefill import install
                        install(self.prefill,config['prefill_bundle'])
                    if config.get('hopper_bundle'):
                        from tensor_llm.qwen35.hopper import install as install_hopper
                        install_hopper(self.prefill,config['hopper_bundle'])
                draft_prefill=Qwen35MTPPrefill(draft,config['draft_prefill_bundle'])
                cached_proposal=np.full(model.slots,-1,'int32')
                cached_hidden=np.zeros((model.slots,2048),'uint16')
                # Admit arrivals during prefill. Decode retains this cohort so
                # large prefix workspaces and verification snapshots never
                # overlap in the bounded L40S allocation.
                while True:
                    for slot,current in enumerate(self.slots):
                        if current is not None and current.cancelled:self.slots[slot]=None
                        if self.slots[slot] is None:
                            try:request=self.pending.get_nowait()
                            except asyncio.QueueEmpty:continue
                            model.reset([slot]);draft.reset([slot]);self.slots[slot]=request
                    active=[(s,r) for s,r in enumerate(self.slots) if r is not None and r.position<len(r.prompt)]
                    if not active:break
                    chunk=self.prefill.chunk
                    block=np.zeros((model.slots,chunk),'int32');shifted=np.zeros_like(block)
                    lengths=np.zeros(model.slots,'int32')
                    for slot,r in active:
                        count=min(chunk,len(r.prompt)-r.position)
                        block[slot,:count]=r.prompt[r.position:r.position+count];lengths[slot]=count
                    predicted=self.prefill.forward(block,lengths)
                    for slot,r in active:
                        count=int(lengths[slot])
                        if count>1:shifted[slot,:count-1]=block[slot,1:count]
                        shifted[slot,count-1]=r.prompt[r.position+count] if r.position+count<len(r.prompt) else predicted[slot]
                    proposal=draft_prefill.forward(shifted,self.prefill.buffers['normal'],lengths)
                    self.prefill_tokens+=int(lengths.sum())
                    for slot,r in active:
                        r.position+=int(lengths[slot]);r.next_token=int(predicted[slot])
                        if r.position==len(r.prompt):
                            cached_proposal[slot]=proposal[slot]
                            # Preserve the tiny cached draft control activation
                            # across inactive prefix chunks for unequal prompts.
                            row=cached_hidden[slot]
                            draft.device.driver.call('cuMemcpyDtoH_v2',ct.c_void_p(row.ctypes.data),
                                draft.buffers['normal'].pointer+slot*row.nbytes,row.nbytes)
                            await self.publish(slot,r)
                    await asyncio.sleep(0)
                self.prefill.close();draft_prefill.close();draft_prefill=None
                if not any(r is not None for r in self.slots):continue
                draft._write('tokens',cached_proposal)
                draft.device.driver.call('cuMemcpyHtoD_v2',draft.buffers['normal'].pointer,
                    ct.c_void_p(cached_hidden.ctypes.data),cached_hidden.nbytes)
                draft.device.driver.call('cuStreamSynchronize',None)
                verifier=Qwen35Verifier(model,config['verify_bundle'])
                repair=Qwen35MTPPrefill(draft,config['repair_bundle'])
                install_selected(verifier,config['verify_bundle']);install_selected(repair,config['repair_bundle'])
                engine=Qwen35Speculative(model,draft,verifier,repair,
                    output_lookup=config.get('output_lookup',False),fallback_proposals=config.get('fallback_proposals',3))
                phase_seconds={name:0. for name in ('draft_seconds','verify_seconds','commit_seconds','repair_seconds')}
                round_count=proposal_count=accepted_count=lookup_requests=0
                decode_start=time.perf_counter()
                seed=np.array([r.next_token if r is not None else -1 for r in self.slots],'int32');engine.seed(seed)
                while any(r is not None for r in self.slots):
                    ready=[(s,r) for s,r in enumerate(self.slots) if r is not None and not r.cancelled]
                    for slot,r in enumerate(self.slots):
                        if r is not None and r.cancelled:self.slots[slot]=None
                    if not ready:break
                    pending=np.full(model.slots,-1,'int32');lengths=np.zeros(model.slots,'int32')
                    for slot,r in ready:
                        pending[slot]=r.next_token;lengths[slot]=min(verifier.chunk,r.output-r.generated)
                    result=engine.step(pending,lengths);self.decode_steps+=1
                    round_count+=1
                    proposal_count+=int(result['lengths'].sum())
                    accepted_count+=int(result['counts'].sum())
                    lookup_requests+=result['lookup_requests']
                    for name in phase_seconds:phase_seconds[name]+=result[name]
                    for slot,r in ready:
                        values=result['outputs'][slot];r.next_token=int(result['pending'][slot])
                        await self.publish(slot,r,values)
                    await asyncio.sleep(0)
                print('Native speculative cohort '+json.dumps(dict(
                    phase_seconds=phase_seconds,rounds=round_count,proposals=proposal_count,
                    accepted=accepted_count,lookup_requests=lookup_requests,
                    elapsed_seconds=time.perf_counter()-decode_start)),flush=True)
                repair.close();repair=None;verifier.close();verifier=None
        finally:
            if repair:repair.close()
            if verifier:verifier.close()
            if draft_prefill:draft_prefill.close()
            self.prefill.close()

    async def publish(self,slot,r,tokens=None):
        tokens=[r.next_token] if tokens is None else tokens
        if not tokens or len(tokens)>r.output-r.generated:raise ValueError('invalid generated token count')
        r.generated+=len(tokens);self.output_tokens+=len(tokens)
        finish=r.generated==r.output
        await r.events.put(dict(token_ids=tokens,meta_info=dict(prompt_tokens=len(r.prompt),
            completion_tokens=r.generated,finish_reason=dict(type='length') if finish else None)))
        if finish:self.slots[slot]=None

    async def generate(self,http_request):
        from aiohttp import web
        data=await http_request.json();prompt=data.get('input_ids');sampling=data.get('sampling_params',{})
        count=sampling.get('max_new_tokens')
        if (not isinstance(prompt,list) or not prompt or any(type(t) is not int or not 0<=t<self.model.config.vocab for t in prompt)
                or type(count) is not int or count<=0 or len(prompt)+count>self.model.context
                or sampling.get('temperature',0)!=0 or sampling.get('ignore_eos') is not True):
            raise web.HTTPBadRequest(text='native benchmark requires valid token IDs, greedy ignore_eos, and a bounded context')
        if self.closed:raise web.HTTPServiceUnavailable(text='native engine stopped')
        request=Request(prompt,count);await self.pending.put(request)
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream','Cache-Control':'no-cache'})
        await response.prepare(http_request)
        try:
            while True:
                event=await request.events.get()
                await response.write(('data: '+json.dumps(event,separators=(',',':'))+'\n\n').encode())
                if event.get('error') or event['meta_info']['finish_reason'] is not None:break
            await response.write(b'data: [DONE]\n\n');await response.write_eof()
        finally:request.cancelled=True
        return response

    async def serve(self,host='127.0.0.1',port=8013):
        from aiohttp import web
        app=web.Application(client_max_size=32*1024**2)
        app.router.add_post('/generate',self.generate)
        async def tokenize(request):
            data=await request.json();return web.json_response(dict(tokens=self.tokenizer.encode(data['prompt'],add_special_tokens=False).ids))
        async def info(request):return web.json_response(self.info())
        async def metrics(request):
            running=sum(r is not None for r in self.slots)
            text='\n'.join((f'tensor_num_requests_running {running}',f'tensor_num_requests_waiting {self.pending.qsize()}',
                f'tensor_generation_tokens_total {self.output_tokens}',f'tensor_prompt_tokens_total {self.prefill_tokens}',
                f'tensor_decode_steps_total {self.decode_steps}',f'tensor_last_decode_batch {int(self.model.active.sum())}'))+'\n'
            return web.Response(text=text,content_type='text/plain')
        app.router.add_post('/tokenize',tokenize);app.router.add_get('/server_info',info)
        app.router.add_get('/health',info);app.router.add_get('/metrics',metrics)
        runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,host,port);await site.start()
        task=asyncio.create_task(self.worker());print(f'Native Tensor server ready on {host}:{port}',flush=True)
        try:await task
        finally:
            self.closed=True;task.cancel();await asyncio.gather(task,return_exceptions=True);await runner.cleanup()


def main():
    import tensor
    from tensor_llm import Qwen35Batch,Qwen35Prefill
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--prefill-bundle',type=Path,required=True);p.add_argument('--port',type=int,default=8013)
    p.add_argument('--draft-bundle',type=Path);p.add_argument('--draft-prefill-bundle',type=Path)
    p.add_argument('--verify-bundle',type=Path);p.add_argument('--repair-bundle',type=Path)
    p.add_argument('--output-lookup',action='store_true');p.add_argument('--fallback-proposals',type=int,default=3)
    p.add_argument('--compact-experts',action='store_true')
    p.add_argument('--hopper-bundle',type=Path)
    a=p.parse_args()
    selected=(a.draft_bundle,a.draft_prefill_bundle,a.verify_bundle,a.repair_bundle)
    if any(selected) and not all(selected):p.error('speculative mode needs all four draft/verification/repair bundles')
    with tensor.Device() as d,Qwen35Batch(a.checkpoint,a.bundle,d,progress=lambda s:print(s,flush=True)) as model:
        prefill=Qwen35Prefill(model,a.prefill_bundle)
        draft=None
        try:
            if a.compact_experts:
                from tensor_llm.qwen35.compact_prefill import install
                install(prefill,a.prefill_bundle)
            if a.hopper_bundle:
                from tensor_llm.qwen35.hopper import install as install_hopper
                install_hopper(prefill,a.hopper_bundle)
            config=None
            if all(selected):
                from tensor_llm import Qwen35MTP
                draft=Qwen35MTP(model,a.draft_bundle)
                config=dict(draft=draft,prefill_bundle=a.prefill_bundle,draft_prefill_bundle=a.draft_prefill_bundle,
                    verify_bundle=a.verify_bundle,repair_bundle=a.repair_bundle,output_lookup=a.output_lookup,
                    fallback_proposals=a.fallback_proposals,compact_experts=a.compact_experts,
                    hopper_bundle=a.hopper_bundle)
            asyncio.run(NativeServer(model,prefill,a.checkpoint,speculative=config).serve(port=a.port))
        finally:
            prefill.close()
            if draft:draft.close()


if __name__=='__main__':main()
