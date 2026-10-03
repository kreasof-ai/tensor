"""Independent causal-window proof and GPU tail/control extraction checks."""
import os
import numpy as np
import pytest


def test_two_width_three_convolutions_preserve_final_output_and_histories():
    rng=np.random.default_rng(1793);c=7
    inputs=[rng.normal(size=(c,3*c))*.2 for _ in range(2)]
    outputs=[rng.normal(size=(c,c))*.2 for _ in range(2)]
    filters=[rng.normal(size=(c,3))*.3 for _ in range(2)]
    feed=rng.normal(size=(c,c))*.1
    def pointwise(x):
        gate=x@feed
        return x+gate/(1+np.exp(-gate))*(x@feed.T)
    def forward(x,states):
        x=pointwise(x);next_states=[]
        for inp,out,w,state in zip(inputs,outputs,filters,states):
            projected=(x@inp).reshape(len(x),3,c);history=state.copy();mixed=[]
            for a,b,d in projected:
                current=a*d
                mixed.append(b*(history[0]*w[:,0]+history[1]*w[:,1]+current*w[:,2]))
                history=np.stack((history[1],current))
            next_states.append(history);x=pointwise(x+np.array(mixed)@out)
        return x[-1],next_states
    states=[rng.normal(size=(2,c)) for _ in range(2)]
    for count in (1,2,4,8,9,17,128,3,129):
        x=rng.normal(size=(count,c))*.3
        complete,next_states=forward(x,states)
        cropped,tail_states=forward(x[-8:],states)
        np.testing.assert_allclose(cropped,complete,rtol=1e-12,atol=1e-12)
        for actual,expected in zip(tail_states,next_states):
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        if count==128:
            insufficient,_=forward(x[-4:],states)
            assert np.max(np.abs(insufficient-complete))>1e-12
        states=next_states


def test_tail_specialization_requires_the_measured_shape_and_suffix():
    from types import SimpleNamespace
    from tensor_llm.model import prefill_tail_rows
    cfg=SimpleNamespace(width=2048,ff=10752,layers=('conv','attention','conv','conv'))
    tensors={}
    for i in (1,2,3):
        names=['ffn_gate','ffn_up','ffn_down']
        if i>1:names += ['shortconv.in_proj','shortconv.out_proj']
        else:names += ['attn_q','attn_output']
        for name in names:tensors[f'blk.{i}.{name}.weight']=SimpleNamespace(type=2)
    gguf=SimpleNamespace(tensors=tensors)
    assert prefill_tail_rows(cfg,gguf,'prefill_mixed')==8
    assert prefill_tail_rows(cfg,gguf,'quant_searched')==0
    assert prefill_tail_rows(SimpleNamespace(width=1024,ff=10752,layers=cfg.layers),gguf,'prefill_mixed')==0
    assert prefill_tail_rows(SimpleNamespace(width=2048,ff=10752,layers=('conv','conv','attention')),gguf,'prefill_mixed')==0
    tensors['blk.2.ffn_gate.weight'].type=14
    assert prefill_tail_rows(cfg,gguf,'prefill_mixed')==0


@pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires native WebGPU')
def test_tail_extraction_partial_chunks_and_position_controls(tmp_path):
    import tensor
    from tensor_llm.webgpu_kernels import source
    r,c,t=128,17,8;rng=np.random.default_rng(1805)
    path=tmp_path/'tail.py';path.write_text(source('prefill_tail',dict(r=r,c=c,t=t)))
    artifact=path.with_suffix('.tbin');tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);x=rng.normal(size=(r,c)).astype(np.float32)
        inp=device.from_numpy(x.ravel());out=device.full(t*c,np.nan)
        control=device.zeros(2,dtype='int32');tail_control=device.zeros(2,dtype='int32')
        for position in (0,127,384):
            for count in (1,2,4,8,17,128):
                device.write(control,np.array([position,count],np.int32))
                kernel.launch(inp,out,control,tail_control)
                expected=np.zeros((t,c),np.float32);active=min(count,t);start=max(count-t,0)
                expected[:active]=x[start:count]
                np.testing.assert_array_equal(out.to_numpy().reshape(t,c),expected)
                np.testing.assert_array_equal(tail_control.to_numpy(),[position+start,active])


@pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires native WebGPU')
def test_attention_tail_uses_absolute_positions_and_all_stored_keys(tmp_path):
    import tensor
    from tensor_llm.webgpu_kernels import source
    r,t,h,kh,d,cap=128,8,4,2,64,192;position=23;rng=np.random.default_rng(1817)
    q=(rng.normal(size=(r,h,d))*.1).astype(np.float32)
    keys=(rng.normal(size=(cap,kh,d))*.1).astype(np.float16)
    values=rng.normal(size=(cap,kh,d)).astype(np.float16)
    artifacts=[]
    for rows in (r,t):
        path=tmp_path/f'attention-{rows}.py';path.write_text(source('attention',dict(r=rows,h=h,kh=kh,d=d,cap=cap,sg=True)))
        artifact=path.with_suffix('.tbin');tensor.build(path,artifact,provider='webgpu');artifacts.append(artifact)
    with tensor.Device(provider='webgpu') as device:
        k,v=[device.from_numpy(x.ravel()) for x in (keys,values)];actual=[]
        for rows,artifact in zip((r,t),artifacts):
            kernel=device.load(artifact);inp=device.from_numpy(q[-rows:].ravel())
            control=device.from_numpy(np.array([position+r-rows,rows],np.int32));out=device.full(rows*h*d,np.nan)
            kernel.launch(inp,k,v,out,control);actual.append(out.to_numpy().reshape(rows,h,d))
        np.testing.assert_allclose(actual[1],actual[0][-t:],rtol=3e-6,atol=1e-8)
        expected=np.zeros((t,h,d),np.float64)
        for row in range(t):
            count=position+r-t+row+1
            for head in range(h):
                group=head//(h//kh)
                scores=q[r-t+row,head].astype(np.float64)@keys[:count,group].astype(np.float64).T/d**.5
                probabilities=np.exp(scores-np.max(scores));probabilities/=np.sum(probabilities)
                expected[row,head]=probabilities@values[:count,group].astype(np.float64)
        np.testing.assert_allclose(actual[1],expected,rtol=3e-5,atol=3e-7)
