"""Container bounds and exact packed GGML block semantics."""
import struct
import numpy as np
import pytest
from tensor_llm.common.gguf import GGUF,GGUFError,dequantize


def string(value):
    value=value.encode();return struct.pack('<Q',len(value))+value


def container(path,*,offset=0,kind=1,dimensions=(4,2),duplicate=False):
    metadata=string('general.architecture')+struct.pack('<I',8)+string('lfm2')
    entry=string('matrix')+struct.pack('<I',len(dimensions))+struct.pack('<'+'Q'*len(dimensions),*dimensions)+struct.pack('<IQ',kind,offset)
    data=b'GGUF'+struct.pack('<IQQ',3,2 if duplicate else 1,1)+metadata+entry+(entry if duplicate else b'')
    data+=b'\0'*((-len(data))%32)+np.arange(8,dtype=np.float16).tobytes()
    path.write_bytes(data);return path


def test_mapped_tensor_shape_and_row(tmp_path):
    g=GGUF(container(tmp_path/'matrix.gguf'))
    assert g.tensors['matrix'].shape==(2,4)
    np.testing.assert_array_equal(g.array('matrix'),np.arange(8).reshape(2,4))
    np.testing.assert_array_equal(g.row('matrix',1),np.arange(4,8))
    assert not g.packed('matrix').flags.writeable
    with pytest.raises(GGUFError):g.row('matrix',2)


@pytest.mark.parametrize('kwargs',[{'offset':1},{'offset':32},{'kind':999},{'dimensions':(0,2)},{'dimensions':(4,2),'kind':2},{'duplicate':True}])
def test_invalid_tensor_layout_rejected(tmp_path,kwargs):
    with pytest.raises(GGUFError):GGUF(container(tmp_path/'bad.gguf',**kwargs))


def test_truncated_metadata_and_wrong_version_rejected(tmp_path):
    p=container(tmp_path/'bad.gguf');data=p.read_bytes();p.write_bytes(data[:25])
    with pytest.raises(GGUFError):GGUF(p)
    p.write_bytes(data[:4]+struct.pack('<I',2)+data[8:])
    with pytest.raises(GGUFError):GGUF(p)


def test_q4_0_nibble_order_and_signed_scale():
    raw=np.frombuffer(struct.pack('<e',-0.5)+bytes(range(16)),np.uint8)
    expected=np.r_[np.arange(16)-8,np.full(16,-8)]*-0.5
    np.testing.assert_array_equal(dequantize(raw,2),expected)


def test_q8_0_signed_values():
    q=np.arange(-16,16,dtype=np.int8);raw=np.r_[np.frombuffer(struct.pack('<e',0.25),np.uint8),q.view(np.uint8)]
    np.testing.assert_array_equal(dequantize(raw,8),q.astype(np.float32)*0.25)


def test_q4_k_six_bit_scale_and_minimum_layout():
    # Scale/min high bits live in different bytes for the second four groups.
    scales=np.array([1,2,3,4,17,34,51,63],np.uint8);mins=np.array([5,6,7,8,19,35,52,62],np.uint8)
    packed=np.empty(12,np.uint8)
    for j in range(4):
        packed[j]=scales[j]|((scales[j+4]>>4)<<6)
        packed[j+4]=mins[j]|((mins[j+4]>>4)<<6)
        packed[j+8]=(scales[j+4]&15)|((mins[j+4]&15)<<4)
    qs=np.arange(128,dtype=np.uint8)
    raw=np.r_[np.frombuffer(struct.pack('<ee',0.125,0.25),np.uint8),packed,qs]
    expected=[]
    for group in range(8):
        values=qs[(group//2)*32:(group//2+1)*32]
        values=(values&15) if group%2==0 else values>>4
        expected.extend(0.125*int(scales[group])*values.astype(np.float32)-0.25*int(mins[group]))
    np.testing.assert_array_equal(dequantize(raw,12),expected)


def test_q6_k_high_bits_and_signed_subblock_scales():
    raw=np.zeros(210,np.uint8);raw[:128]=0xA5;raw[128:192]=0b11100100
    raw[192:208]=np.arange(-8,8,dtype=np.int8).view(np.uint8);raw[208:210]=np.frombuffer(struct.pack('<e',0.5),np.uint8)
    expected=[]
    for half in range(2):
        for group,quant in enumerate((-27,-11,10,26)):
            for scale in range(half*8+group*2-8,half*8+group*2-6):expected.extend([0.5*scale*quant]*16)
    np.testing.assert_array_equal(dequantize(raw,14),expected)


def test_incomplete_quantized_block_rejected():
    with pytest.raises(GGUFError):dequantize(np.zeros(17,np.uint8),2)
