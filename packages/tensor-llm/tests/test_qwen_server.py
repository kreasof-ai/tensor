"""Serving slot admission, independent autoregression and exact stream counts."""
import asyncio
from types import SimpleNamespace
import numpy as np


def test_native_scheduler_preserves_eight_independent_streams():
    from benchmarks.qwen35.server import NativeServer,Request
    class Model:
        slots=8;context=100;config=SimpleNamespace(vocab=1000)
        def __init__(self):self.positions=np.zeros(8,'int32');self.resets=[];self.active=np.zeros(8,'int32')
        def _check(self):pass
        def reset(self,slots):
            self.positions[slots]=0;self.resets.extend(slots)
        def forward(self,tokens):
            self.active=(tokens>=0).astype('int32');self.positions+=self.active
            return np.where(tokens>=0,tokens+1,-1)
    class Prefill:
        chunk=4
        def __init__(self,model):self.model=model
        def forward(self,block,lengths):
            self.model.positions+=lengths
            return np.array([block[s,lengths[s]-1]+1 if lengths[s]>0 else -1 for s in range(8)],'int32')
    async def check():
        model=Model();server=NativeServer(model,Prefill(model),None,tokenizer=object(),coalesce_seconds=0)
        requests=[Request([10+slot]*count,slot+1) for slot,count in enumerate((11,7,2,8,1,4,13,6))]
        for request in requests:await server.pending.put(request)
        task=asyncio.create_task(server.worker())
        async def collect(request):
            events=[]
            while True:
                event=await asyncio.wait_for(request.events.get(),5);events.append(event)
                if event['meta_info']['finish_reason'] is not None:return events
        try:
            streams=await asyncio.gather(*(collect(r) for r in requests))
            for slot,(request,events) in enumerate(zip(requests,streams)):
                assert len(events)==request.output
                assert [event['token_ids'][0] for event in events]==list(range(11+slot,11+slot+request.output))
                assert [event['meta_info']['completion_tokens'] for event in events]==list(range(1,request.output+1))
                assert all(event['meta_info']['prompt_tokens']==len(request.prompt) for event in events)
                assert model.positions[slot]==len(request.prompt)+request.output-1
            assert server.output_tokens==sum(r.output for r in requests)
            assert server.prefill_tokens==sum(len(r.prompt) for r in requests)
            assert set(model.resets)==set(range(8))
            assert all(r is None for r in server.slots)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    asyncio.run(check())


def test_native_adapter_counts_server_usage_without_text_estimates():
    from benchmarks.llm_serving.adapters import Adapter
    adapter=Adapter('tensor','qwen35')
    request=dict(prompt_token_ids=[12,24],output_tokens=3)
    payload=adapter.payload(request,seed=7,prefix_cache=False)
    assert payload['input_ids']==request['prompt_token_ids']
    assert payload['sampling_params']['ignore_eos'] is True
    event=adapter.parse(dict(token_ids=[91],meta_info=dict(completion_tokens=3,prompt_tokens=2,finish_reason=dict(type='length'))),2)
    assert event.count==3 and event.prompt_tokens==2 and event.finished


def test_verified_chunks_preserve_exact_stream_counts_and_completion():
    from benchmarks.qwen35.server import NativeServer,Request
    from benchmarks.llm_serving.adapters import Adapter
    model=SimpleNamespace(slots=1,_check=lambda:None)
    async def check():
        server=NativeServer(model,None,None,tokenizer=object());request=Request([12,24],5)
        server.slots[0]=request;adapter=Adapter('tensor','qwen35')
        await server.publish(0,request,[30,31]);first=await request.events.get()
        assert first['token_ids']==[30,31]
        assert adapter.parse(first,0).count==2
        assert first['meta_info']['finish_reason'] is None
        assert server.slots[0] is request
        await server.publish(0,request,[32,33,34]);last=await request.events.get()
        parsed=adapter.parse(last,2)
        assert parsed.count==5 and parsed.finished
        assert server.output_tokens==5 and request.generated==5 and server.slots[0] is None
    asyncio.run(check())


def test_resident_speculative_graphs_remain_owned_by_caller_on_worker_exit():
    from benchmarks.qwen35.server import NativeServer,Request
    class Graph:
        def __init__(self):self.closes=0
        def close(self):self.closes+=1
    class Prefill:
        def __init__(self):self.closed=False
        def close(self):self.closed=True
    async def check(fail):
        def reset():raise RuntimeError('injected prefix initialization failure')
        model=SimpleNamespace(slots=1,_check=lambda:None,reset=reset)
        graphs=Graph(),Graph();prefill=Prefill()
        server=NativeServer(model,prefill,None,tokenizer=object(),
                            speculative=dict(draft=object(),resident_pair=graphs))
        if fail:await server.pending.put(Request([12],2))
        task=asyncio.create_task(server.speculative_worker())
        if fail:
            import pytest
            with pytest.raises(RuntimeError,match='injected prefix'):await task
        else:
            await asyncio.sleep(0);task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        assert prefill.closed and all(graph.closes==0 for graph in graphs)
        # The same caller that prepared the graphs closes them after the worker.
        for graph in graphs:graph.close()
        assert all(graph.closes==1 for graph in graphs)
    asyncio.run(check(False));asyncio.run(check(True))


def test_resident_speculative_graphs_reuse_owners_across_fresh_cohorts(monkeypatch):
    import tensor_llm
    import tensor_llm.qwen35.mtp.prefill as mtp_module
    import benchmarks.qwen35.spec_run as spec_run
    from benchmarks.qwen35.server import NativeServer,Request
    class Buffer:
        pointer=0
        def __init__(self):self.value=np.zeros(1,'int32')
        def to_numpy(self):return self.value.copy()
    class Model:
        slots=1;context=100;config=SimpleNamespace(vocab=1000)
        def __init__(self):
            self.position=np.zeros(1,'int32');self.buffers={k:Buffer() for k in ('tokens','normal')}
            self.device=SimpleNamespace(driver=SimpleNamespace(call=lambda *args:None))
        def _check(self):pass
        def reset(self,slots=None):self.position[:]=0
        def _write(self,name,value):
            if name in self.buffers:self.buffers[name].value=np.array(value)
        def draft(self,tokens,hidden):
            self.position+=(tokens>=0).astype('int32');return tokens+1
    class Prefill:
        chunk=2
        def __init__(self,model,bundle=None):
            self.model=model;self.closed=False;self.buffers={'normal':Buffer()}
        def forward(self,block,lengths):
            self.model.position+=lengths
            return np.array([block[0,lengths[0]-1]+1],'int32')
        def close(self):self.closed=True
    class DraftPrefill(Prefill):
        def forward(self,tokens,hidden,lengths):return super().forward(tokens,lengths)
    class Verifier:
        chunk=8
        def __init__(self,model):self.model=model;self.buffers={'normal':Buffer()};self.closes=0
        def forward(self,tokens,lengths):return tokens+1
        def commit(self,counts):self.model.position+=counts
        def close(self):self.closes+=1
    class Repair:
        chunk=8
        def __init__(self,model,verifier):self.model=model;self.verifier=verifier;self.closes=0
        def forward(self,tokens,hidden,counts):
            assert hidden is self.verifier.buffers['normal']
            self.model.position+=counts
            self.model._write('tokens',np.array([tokens[0,counts[0]-1]+1],'int32'))
        def close(self):self.closes+=1
    monkeypatch.setattr(tensor_llm,'Qwen35Prefill',Prefill)
    monkeypatch.setattr(mtp_module,'Qwen35MTPPrefill',DraftPrefill)
    def unexpected_factory(*args):raise AssertionError('resident graph pair was rebuilt')
    monkeypatch.setattr(spec_run,'make_speculative_pair',unexpected_factory)
    async def check():
        model=Model();draft=Model();draft.target=model
        verifier=Verifier(model);repair=Repair(draft,verifier)
        server=NativeServer(model,Prefill(model),None,tokenizer=object(),coalesce_seconds=0,
            speculative=dict(draft=draft,resident_pair=(verifier,repair),
                             prefill_bundle='prefill',draft_prefill_bundle='draft'))
        task=asyncio.create_task(server.worker())
        try:
            for prompt in (10,30):
                request=Request([prompt,prompt],5);await server.pending.put(request);emitted=[]
                while True:
                    event=await asyncio.wait_for(request.events.get(),5)
                    assert 'error' not in event
                    emitted.extend(event['token_ids'])
                    if event['meta_info']['finish_reason'] is not None:break
                assert emitted==list(range(prompt+1,prompt+6))
                assert model.position[0]==draft.position[0]==6
                assert verifier.closes==repair.closes==0
            assert server.output_tokens==10 and server.prefill_tokens==4
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)
        assert verifier.closes==repair.closes==0
        repair.close();verifier.close()
        assert verifier.closes==repair.closes==1
    asyncio.run(check())
