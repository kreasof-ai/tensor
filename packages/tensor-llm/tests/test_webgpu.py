"""Opt-in real Radeon execution and exact packed-weight gathers."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import tensor
from tensor_llm import LFM2
from tensor_llm.kernels import identity

pytestmark=pytest.mark.skipif(os.environ.get('TENSOR_LFM2_WEBGPU')!='1',reason='set TENSOR_LFM2_WEBGPU=1 and build the 230M WebGPU bundles')
ROOT=Path(__file__).resolve().parents[3]


def test_explicit_half_rounding_ties_subnormals_and_random_values(tmp_path):
    from tensor_llm.kernels import emit
    from tensor_llm.webgpu_kernels import round_half
    rng=np.random.default_rng(342)
    x=np.concatenate((np.array([0.,-0.,2**-25,-2**-25,np.nextafter(np.float32(2**-25),np.float32(1)),2**-24,-2**-24,2**-14,1+2**-11,1+3*2**-11,-1-2**-11,65504.,-65504.],np.float32),
                      rng.normal(size=256).astype(np.float32),rng.uniform(-1e-4,1e-4,size=256).astype(np.float32)))
    n=len(x);source=tmp_path/'round.py';artifact=tmp_path/'round.tbin'
    source.write_text(emit([('x',n,'float32'),('out',n,'float32')],f'''with T.Kernel(T.ceildiv({n}, 128), threads=128) as block:
    for lane in T.Parallel(128):
        i=block*128+lane
        if i < {n}:
            out[i]={round_half('x[i]')}'''))
    tensor.build(source,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);input=device.from_numpy(x);out=device.zeros(n)
        kernel.launch(input,out)
        np.testing.assert_array_equal(out.to_numpy(),x.astype(np.float16).astype(np.float32))


@pytest.mark.parametrize('kind',['f16','q4_0'])
def test_real_model_reset_reference_and_capacity(kind):
    path=ROOT/f'build/lfm2-230m-models/LFM2.5-230M-{kind.upper()}.gguf'
    bundle=ROOT/f'build/lfm2-230m-{kind}-webgpu'
    evidence=ROOT/f'build/lfm2-230m-{kind}-webgpu-validation'
    report=json.loads((evidence/'report.json').read_text())
    with tensor.Device(provider='webgpu') as device,LFM2(path,bundle,device,context=512) as model:
        first=model.forward(report['validation'][0]['tokens'])
        np.testing.assert_array_equal(first,np.load(evidence/'0-tensor-logits.npy'))
        model.reset();np.testing.assert_array_equal(model.forward(report['validation'][0]['tokens']),first)
        prompt='What is 2 + 2?'
        assert model.generate(prompt,max_tokens=96)==model.generate(prompt,max_tokens=96,gpu_greedy=False)
        # A multi-chunk prompt must sample from its last chunk, then feed the
        # GPU-selected token directly into cached decode.
        for limit in (1,3):
            assert model.generate(prompt*10,max_tokens=limit)==model.generate(prompt*10,max_tokens=limit,gpu_greedy=False)
        model.reset();model.forward(report['validation'][0]['tokens'],read=False)
        queued=model.forward([3097]);model.reset();model.forward(report['validation'][0]['tokens'])
        np.testing.assert_array_equal(queued,model.forward([3097]))
        for tokens,message in (([], 'nonempty'),([65536],'vocabulary'),([1]*513,'capacity')):
            with pytest.raises(ValueError,match=message):model.forward(tokens)
        model.reset();model.forward([1]*512)
        with pytest.raises(ValueError,match='capacity'):model.forward([1])
    with pytest.raises(RuntimeError,match='closed'):model.forward([1])


def test_q6_embedding_gather_matches_cpu_blocks_exactly():
    path=ROOT/'build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf'
    with tensor.Device(provider='webgpu') as device,LFM2(path,ROOT/'build/lfm2-230m-q4_0-webgpu',device,context=512) as model:
        info=model.gguf.tensors['token_embd.weight']
        assert info.type==14
        ws=model.workspaces[1]
        kernel=model.kernels[identity('embedding',dict(r=1,c=1024,v=65536,type=14))]
        for token in (0,1,7,1023,32767,65535):
            device.write(ws['tokens'],np.array([token],np.int32))
            kernel.launch(ws['tokens'],model.weights['token_embd.weight'],ws['hidden'])
            np.testing.assert_array_equal(ws['hidden'].to_numpy(),model.gguf.row('token_embd.weight',token))


def test_q4_embedding_word_boundaries_and_half_subnormal_scales(tmp_path):
    from tensor_llm.gguf import dequantize
    from tensor_llm.webgpu_kernels import source
    # Eighteen-byte blocks alternate between aligned and unaligned u32 loads.
    rng=np.random.default_rng(411);blocks=rng.integers(0,256,(8,18),dtype=np.uint8)
    scales=np.array([0.,1.,-1.,2**-24,-2**-24,2**-14,.001,3.],np.float16)
    blocks[:,:2]=scales.view(np.uint8).reshape(8,2)
    raw=blocks.reshape(-1);expected=dequantize(raw,2).reshape(4,64)
    path=tmp_path/'gather.py';artifact=tmp_path/'gather.tbin'
    path.write_text(source('embedding',dict(r=1,c=64,v=4,type=2)))
    tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);weights=device.from_numpy(raw.view(np.uint32));out=device.zeros(64)
        token=device.from_numpy(np.zeros(1,np.int32))
        for i in range(4):
            device.write(token,np.array([i],np.int32));kernel.launch(token,weights,out)
            np.testing.assert_array_equal(out.to_numpy(),expected[i])


def test_gpu_argmax_ties_tail_and_control(tmp_path):
    from tensor_llm.webgpu_kernels import source
    n=1025;path=tmp_path/'argmax.py';artifact=tmp_path/'argmax.tbin'
    path.write_text(source('argmax',dict(n=n)));tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(852)
    cases=[rng.normal(size=n).astype(np.float32),np.full(n,-np.inf,np.float32),np.zeros(n,np.float32)]
    tied=np.full(n,-1.,np.float32);tied[[512,1024,17]]=np.inf;cases.append(tied)
    tail=np.zeros(n,np.float32);tail[-1]=100;cases.append(tail)
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);logits=device.zeros(n);token=device.zeros(1,'int32');pos=device.from_numpy(np.array([11,32],np.int32))
        for case in cases:
            device.write(logits,case);kernel.launch(logits,token,pos)
            assert token.to_numpy()[0]==np.argmax(case)
            np.testing.assert_array_equal(pos.to_numpy(),np.array([11,1],np.int32))


@pytest.mark.parametrize('kind,unroll,dot',[(2,1,False),(2,2,False),(14,1,False),(14,2,False),(14,1,True),(14,2,True)])
@pytest.mark.parametrize('subgroup',[False,True,'fallback'])
def test_packed_decode_projection_block_fields(tmp_path,kind,subgroup,unroll,dot):
    from tensor_llm.gguf import dequantize,TYPES
    from tensor_llm.webgpu_kernels import source
    k,o=512,8;_,block,size=TYPES[kind]
    rng=np.random.default_rng(524+kind);blocks=rng.integers(0,256,(o*k//block,size),dtype=np.uint8)
    offset=0 if kind==2 else 208
    scales=np.resize(np.array([0.,1.,-1.,2**-24,-2**-24,2**-14,.001,3.],np.float16),len(blocks))
    blocks[:,offset:offset+2]=scales.view(np.uint8).reshape(-1,2)
    raw=blocks.reshape(-1);x=rng.normal(size=k).astype(np.float32)
    decoded=dequantize(raw,kind).reshape(o,k).astype(np.float64)
    expected=decoded@x.astype(np.float64)
    tolerance=np.sum(np.abs(decoded*x),axis=1)*3e-6+1e-10
    path=tmp_path/'projection.py';artifact=tmp_path/'projection.tbin'
    text=source('linear',dict(r=1,k=k,o=o,type=kind,sg=bool(subgroup),gemv_unroll=unroll,gemv_chains=unroll,gemv_q6_dot=dot))
    if subgroup=='fallback':text=text.replace('T.call_extern("uint32", "tensor_subgroup_size")','T.uint32(8)')
    path.write_text(text);tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);out=device.zeros(o)
        kernel.launch(device.from_numpy(x),device.from_numpy(raw.view(np.uint32)),out)
        assert np.all(np.abs(out.to_numpy()-expected)<=tolerance)


@pytest.mark.parametrize('schedule',[
    {'gemv_lanes':8,'gemv_threads':64},
    {'gemv_lanes':16,'gemv_threads':256,'gemv_accumulators':4},
    {'gemv_lanes':32,'gemv_threads':128,'gemv_accumulators':4},
    {'gemv_lanes':32,'gemv_threads':128,'gemv_dot':True},
    {'gemv_lanes':32,'gemv_threads':128,'gemv_dot':True,'gemv_unroll':2,'gemv_chains':2},
    {'gemv_lanes':64,'gemv_threads':128,'gemv_dot':True},
])
@pytest.mark.parametrize('subgroup',[False,True,'fallback'])
@pytest.mark.parametrize('kind',['linear','ffn'])
def test_q4_decode_schedules_tail_and_paired_accumulators(tmp_path,schedule,subgroup,kind):
    from tensor_llm.gguf import dequantize
    from tensor_llm.webgpu_kernels import source
    k,o=512,11;rng=np.random.default_rng(547)
    x=(rng.normal(size=k)*.05).astype(np.float32)
    weights=[];references=[];bounds=[]
    for _ in range(2 if kind=='ffn' else 1):
        blocks=rng.integers(0,256,(o*k//32,18),dtype=np.uint8)
        scales=np.resize(np.array([0.,1.,-1.,2**-24,-2**-24,2**-14,.001,3.],np.float16),len(blocks))
        blocks[:,:2]=scales.view(np.uint8).reshape(-1,2);raw=blocks.ravel();weights.append(raw)
        decoded=dequantize(raw,2).reshape(o,k).astype(np.float64)
        references.append(decoded@x.astype(np.float64))
        bounds.append(np.sum(np.abs(decoded*x),axis=1)*3e-6+1e-10)
    expected=references[0];tolerance=bounds[0]
    if kind=='ffn':
        activated=expected/(1+np.exp(-expected))
        tolerance=1.1*bounds[0]*np.abs(references[1])+np.abs(activated)*bounds[1]+1e-7
        expected=activated*references[1]
    text=source(kind,dict(r=1,k=k,o=o,type=2,sg=bool(subgroup),**schedule))
    if subgroup=='fallback':text=text.replace('T.call_extern("uint32", "tensor_subgroup_size")','T.uint32(4)')
    path=tmp_path/'decode.py';artifact=path.with_suffix('.tbin');path.write_text(text)
    tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        output=device.zeros(o)
        device.load(artifact).launch(device.from_numpy(x),*(device.from_numpy(w.view(np.uint32)) for w in weights),output)
        actual=output.to_numpy()
        assert np.all(np.isfinite(actual))
        assert np.all(np.abs(actual-expected)<=tolerance)


@pytest.mark.parametrize('encoding',[0,1,2,14])
@pytest.mark.parametrize('subgroup',[False,True])
def test_decode_residual_projection_matches_separate_add(tmp_path,encoding,subgroup):
    from tensor_llm.gguf import TYPES
    from tensor_llm.webgpu_kernels import source
    k,o=512,11;rng=np.random.default_rng(829)
    x=rng.normal(size=k).astype(np.float32);residual=rng.normal(size=o).astype(np.float32)
    if encoding in (0,1):
        weights=(rng.normal(size=k*o)*.1).astype(np.float16 if encoding==1 else np.float32)
    else:
        _,block,size=TYPES[encoding];packed=rng.integers(0,256,(k*o//block,size),dtype=np.uint8)
        offset=0 if encoding==2 else 208
        packed[:,offset:offset+2]=np.full(len(packed),.01,np.float16).view(np.uint8).reshape(-1,2)
        weights=packed.ravel().view(np.uint32)
    artifacts=[]
    for kind in ('linear','linear_add'):
        path=tmp_path/(kind+'.py');artifact=path.with_suffix('.tbin')
        path.write_text(source(kind,dict(r=1,k=k,o=o,type=encoding,sg=subgroup)))
        tensor.build(path,artifact,provider='webgpu');artifacts.append(artifact)
    with tensor.Device(provider='webgpu') as device:
        input=device.from_numpy(x);weight=device.from_numpy(weights);base=device.zeros(o);fused=device.zeros(o)
        device.load(artifacts[0]).launch(input,weight,base)
        device.load(artifacts[1]).launch(input,weight,device.from_numpy(residual),fused)
        np.testing.assert_array_equal(fused.to_numpy(),residual+base.to_numpy())


def test_native_half_unpack_all_bit_patterns(tmp_path):
    from tensor_llm.kernels import emit
    from tensor_llm.webgpu_kernels import half_bits
    n=65536;path=tmp_path/'half.py';artifact=path.with_suffix('.tbin')
    path.write_text(emit([('x',n,'uint32'),('out',n,'float32')],f'''with T.Kernel({n//128},threads=128) as block:
    tx=T.get_thread_binding()
    i=block * 128 + tx
    out[i]={half_bits('x[i]')}'''))
    tensor.build(path,artifact,provider='webgpu')
    bits=np.arange(n,dtype=np.uint16);expected=bits.view(np.float16).astype(np.float32)
    with tensor.Device(provider='webgpu') as device:
        out=device.zeros(n);device.load(artifact).launch(device.from_numpy(bits.astype(np.uint32)),out)
        actual=out.to_numpy();finite=~np.isnan(expected)
        np.testing.assert_array_equal(actual[finite].view(np.uint32),expected[finite].view(np.uint32))
        assert np.isnan(actual[~finite]).all()


def test_q8_activation_packing_and_integer_dot_reference(tmp_path):
    from tensor_llm.gguf import dequantize
    from tensor_llm.webgpu_kernels import source
    k,o=256,8;rng=np.random.default_rng(802)
    raw=rng.integers(0,256,(o*k//32,18),dtype=np.uint8)
    raw[:,:2]=np.full(len(raw),.25,np.float16).view(np.uint8).reshape(-1,2)
    raw=raw.flatten();x=rng.normal(size=k).astype(np.float32);x[:32]=0
    artifacts=[]
    for kind,p in (('quantize_q8',{'k':k}),('linear_q8',{'k':k,'o':o})):
        path=tmp_path/(kind+'.py');artifact=path.with_suffix('.tbin');path.write_text(source(kind,p));tensor.build(path,artifact,provider='webgpu');artifacts.append(artifact)
    scale=np.maximum(np.max(np.abs(x.reshape(-1,32)),axis=1)/127,np.float32(1e-20)).astype(np.float32)
    quant=np.rint(x.reshape(-1,32)/scale[:,None]).astype(np.int8)
    expected=dequantize(raw,2).reshape(o,k).astype(np.float64)@(quant.astype(np.float32)*scale[:,None]).flatten().astype(np.float64)
    with tensor.Device(provider='webgpu') as device:
        packed=device.zeros(k//4,dtype='uint32');scales=device.zeros(k//32);sums=device.zeros(k//32,dtype='int32');out=device.zeros(o)
        device.load(artifacts[0]).launch(device.from_numpy(x),packed,scales,sums)
        np.testing.assert_array_equal(packed.to_numpy().view(np.int8).reshape(-1,32),quant)
        np.testing.assert_allclose(scales.to_numpy(),scale,rtol=1e-6,atol=0)
        np.testing.assert_array_equal(sums.to_numpy(),quant.astype(np.int32).sum(axis=1))
        device.load(artifacts[1]).launch(packed,scales,sums,device.from_numpy(raw.view(np.uint32)),out)
        np.testing.assert_allclose(out.to_numpy(),expected,rtol=3e-6,atol=1e-5)
