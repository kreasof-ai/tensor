"""Compact maps must preserve tails, hot experts and the maximum route budget."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1', reason='native CUDA compact route qualification')
def test_compact_tiles_cover_every_route_once(tmp_path):
    import tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    rows, m = 4096, 64
    source = tmp_path/'tiles.py'
    artifact = source.with_suffix('.tbin')
    source.write_text(export_source('tensor_llm.qwen35.kernels.compact_experts', 'tile_map_kernel',
                                    dict(rows=rows, block_m=m), dependencies=('tensor.compiler.entry',)))
    build_artifact(source, artifact, target=os.environ.get('TENSOR_QWEN_TARGET','sm_89'),
                   compiler='nvrtc', nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    hot = np.zeros(256,'int32'); hot[:8] = rows
    balanced = np.full(256,rows*8//256,'int32')
    # All 256 experts have tails while exhausting the route budget.
    tails = np.ones(256,'int32'); tails[:8] += (rows*8-256)//8
    sparse = np.zeros(256,'int32'); sparse[[0,31,255]] = [1,65,rows]
    capacity = (rows*8+m-1)//m+255
    with tensor.Device() as device:
        kernel = device.load(artifact)
        expert = device.empty((capacity,),'int32'); offset = device.empty((capacity,),'int32')
        for counts in (balanced, hot, tails, sparse, np.zeros(256,'int32')):
            value = device.from_numpy(counts)
            kernel.launch(value,expert,offset)
            actual = list(zip(expert.to_numpy().tolist(),offset.to_numpy().tolist()))
            expected = [(e,t) for e,count in enumerate(counts) for t in range((int(count)+m-1)//m)]
            assert actual[:len(expected)] == expected
            assert all(e==-1 for e,_ in actual[len(expected):])
            assert len(expected)<=capacity
            value.release()


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1', reason='native CUDA compact projection qualification')
@pytest.mark.parametrize('routed', [False, True])
def test_compact_experts_match_original_with_shuffled_routes(tmp_path, routed):
    import torch
    import tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    torch.set_num_threads(1)
    rng = np.random.default_rng(623)
    rows, k, o, top = 257, 256, 128, 8
    ids = np.stack([rng.choice(256,top,replace=False) for _ in range(rows)])
    ids[:128] = np.arange(top)
    counts = np.zeros(256,'int32'); routes = np.full((256,rows),-1,'int32')
    for expert in range(256):
        values = [row*top+rank for row in range(rows) for rank in range(top) if ids[row,rank]==expert]
        rng.shuffle(values)
        counts[expert] = len(values); routes[expert,:len(values)] = values
    shape = (rows,top,k) if routed else (rows,k)
    x = torch.from_numpy(rng.normal(size=shape).astype('float32')).to(torch.float8_e4m3fn)
    w = torch.from_numpy(rng.normal(size=(256,o,k)).astype('float32')).to(torch.float8_e4m3fn)
    a = rng.uniform(.01,.1,(*shape[:-1],k//128)).astype('float32')
    scales = rng.uniform(.01,.1,(256,1,k//128)).astype('float32')
    artifacts = {}
    for name,module,factory,p in [
        ('map','compact_experts','tile_map_kernel',dict(rows=rows,block_m=64)),
        ('control','prefill','expert_kernel',dict(rows=rows,k=k,o=o,routed_input=routed,block_m=64,threads=256)),
        ('compact','prefill','expert_kernel',dict(rows=rows,k=k,o=o,routed_input=routed,block_m=64,threads=256,compact=True))]:
        source = tmp_path/(name+'.py'); artifact = source.with_suffix('.tbin')
        source.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,p,
                                        dependencies=('tensor.compiler.entry',)))
        build_artifact(source,artifact,target=os.environ.get('TENSOR_QWEN_TARGET','sm_89'),
                       compiler='nvrtc',nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        artifacts[name] = artifact
    with tensor.Device() as device:
        inputs = [device.from_numpy(v,dtype=dtype) for v,dtype in
            ((x.view(torch.uint8).numpy(),'uint8'),(a,'float32'),(w.view(torch.uint8).numpy(),'uint8'),
             (scales,'bfloat16'),(counts,'int32'),(routes,'int32'))]
        capacity = (rows*top+63)//64+255
        experts = device.empty((capacity,),'int32'); offsets = device.empty((capacity,),'int32')
        device.load(artifacts['map']).launch(inputs[-2],experts,offsets)
        out = device.from_numpy(np.full((rows,top,o),np.nan,'float32'))
        device.load(artifacts['control']).launch(*inputs,out)
        expected = out.to_numpy()
        device.load(artifacts['compact']).launch(*inputs,out,experts,offsets)
        actual = out.to_numpy()
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, expected)
