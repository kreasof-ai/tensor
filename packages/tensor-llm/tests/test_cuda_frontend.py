"""Production CUDA algorithms remain inspectable by the compiler frontend."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('kind,parameters', [
    ('ffn',dict(r=1,k=256,o=7,type=12)),
    ('ffn',dict(r=32,k=256,o=7,type=12,block_m=32,block_n=64,block_k=64,packed_pairs=True)),
    ('attention_grouped',dict(r=1,h=32,kh=8,d=64,cap=512,splits=16,stage=16,warps=4)),
])
def test_algorithms_have_ir_loops_and_only_typed_hardware_externs(tmp_path,kind,parameters):
    pytest.importorskip('tilelang')
    import tvm
    from tensor.artifacts.portable import export_spec
    from tensor_llm.lfm2.kernels.cuda import source
    path=tmp_path/'kernel.py';path.write_text(source(kind,parameters))
    spec=export_spec(path)
    externs=[];loops=[]
    def visit(node):
        if isinstance(node,tvm.tirx.For):loops.append(node)
        if isinstance(node,tvm.tirx.Call) and getattr(node.op,'name',None)=='tirx.call_extern':
            externs.append(node.args[0].value)
    tvm.tirx.stmt_functor.post_order_visit(spec['kernel'].body,visit)
    assert loops,'the complete algorithm must remain visible as frontend loops'
    assert set(externs)<= {'tensor_load_u16','tensor_pack_f16x2'}
    assert 'prelude' not in path.read_text()
    assert '__device__' not in path.read_text()
