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
