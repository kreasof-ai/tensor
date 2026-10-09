"""Checkpoint layout validation without loading a second model into RAM."""
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint, Qwen35Config


def config():
    return {'text_config': dict(model_type='qwen3_5_moe_text', hidden_size=2048,
        vocab_size=248320, num_hidden_layers=40, num_attention_heads=16,
        num_key_value_heads=2, head_dim=256, linear_num_key_heads=16,
        linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4,
        num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
        shared_expert_intermediate_size=512, hidden_act='silu',
        attn_output_gate=True, attention_bias=False,
        layer_types=['linear_attention'] * 3 + ['full_attention'],
        max_position_embeddings=262144, rms_norm_eps=1e-6,
        rope_parameters=dict(rope_type='default', partial_rotary_factor=.25,
            mrope_section=[11,11,10], mrope_interleaved=True, rope_theta=10000000)),
        'quantization_config':dict(quant_method='fp8',activation_scheme='dynamic',weight_block_size=[128,128])}


@pytest.fixture
def cfg():
    value=config();value['text_config']['layer_types'] *= 10
    return value


def test_resident_memory_includes_all_eight_full_contexts(cfg):
    c=Qwen35Config.from_json(cfg)
    memory=c.state_bytes(slots=8,context=48000)
    assert memory['kv']==7_864_320_000
    assert memory['recurrent']==503_316_480
    assert memory['convolution']==23_592_960
    assert c.state_bytes(slots=8,context=48001)['capacity']==48128
    for values in ({'slots':0,'context':48000},{'slots':8,'context':262145},
                   {'slots':True,'context':48000}):
        with pytest.raises(ValueError):c.state_bytes(**values)


@pytest.mark.parametrize('change', ['layers','fp4','heads','rope','activation','bias'])
def test_unsupported_variants_are_rejected(cfg,change):
    if change=='layers':cfg['text_config']['layer_types']=cfg['text_config']['layer_types'][:4]
    if change=='fp4':cfg['quantization_config']['quant_method']='fp4'
    if change=='heads':cfg['text_config']['num_key_value_heads']=8
    if change=='rope':cfg['text_config']['rope_parameters']['partial_rotary_factor']=1
    if change=='activation':cfg['quantization_config']['activation_scheme']='static'
    if change=='bias':cfg['text_config']['attention_bias']=True
    with pytest.raises(ValueError):Qwen35Config.from_json(cfg)


@pytest.fixture
def checkpoint(tmp_path,monkeypatch,cfg):
    # Small storage exercises the file protocol. The real full checkpoint is
    # audited separately, with its complete 62,303-tensor shape contract.
    shapes={'model.language_model.norm.weight':(4,),
            'model.language_model.test.weight':(128,128),
            'model.language_model.test.weight_scale_inv':(1,1)}
    monkeypatch.setattr(Qwen35Config,'expected_tensors',lambda self:shapes)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    header={};raw=bytearray()
    for name,shape in shapes.items():
        dtype='F8_E4M3' if name.endswith('test.weight') else 'BF16'
        size=int(np.prod(shape))*(1 if dtype=='F8_E4M3' else 2)
        header[name]=dict(dtype=dtype,shape=shape,data_offsets=[len(raw),len(raw)+size])
        raw.extend(bytes([len(header)])*size)
    def write():
        encoded=json.dumps(header).encode()
        (tmp_path/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+raw)
    write()
    index={'weight_map':{n:'model.safetensors' for n in shapes}}
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(index))
    return tmp_path,header,write,index


def test_checkpoint_native_bytes_are_preserved(checkpoint):
    path,_,_,_=checkpoint
    c=Qwen35Checkpoint(path)
    assert c.weight_bytes==16394
    np.testing.assert_array_equal(c.read('model.language_model.test.weight'),np.full((128,128),2,'uint8'))
    assert c.read('model.language_model.norm.weight').dtype==np.dtype('uint16')


@pytest.mark.parametrize('defect', ['shape','dtype','overlap','past_end','index','missing','escape'])
def test_malformed_checkpoint_rejected_before_gpu_allocation(checkpoint,defect):
    path,header,write,index=checkpoint
    name='model.language_model.test.weight'
    if defect=='shape':header[name]['shape']=[64,256]
    if defect=='dtype':header[name]['dtype']='I8'
    if defect=='overlap':header[name]['data_offsets']=[0,16384]
    if defect=='past_end':header[name]['data_offsets']=[20,16404]
    if defect=='index':index['weight_map'][name]='missing.safetensors'
    if defect=='missing':del header[name]
    if defect=='escape':
        for n in index['weight_map']:index['weight_map'][n]='../outside.safetensors'
        (path.parent/'outside.safetensors').write_bytes((path/'model.safetensors').read_bytes())
    write();(path/'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):Qwen35Checkpoint(path)


def test_partial_gpu_upload_frees_prior_allocations(checkpoint):
    path,_,_,_=checkpoint;c=Qwen35Checkpoint(path);closed=[]
    def empty(shape,dtype):
        return SimpleNamespace(pointer=1,release=lambda:closed.append(True))
    def fail(*args):raise RuntimeError('transfer failure')
    d=SimpleNamespace(empty=empty,_check=lambda:None,driver=SimpleNamespace(call=fail))
    with pytest.raises(RuntimeError,match='transfer failure'):c.upload(d)
    assert closed==[True]*3


def test_mtp_branch_requires_supported_configuration(checkpoint):
    path, _, _, _ = checkpoint
    with pytest.raises(ValueError, match='one decoder layer'):
        Qwen35Checkpoint(path, branch='mtp')
    with pytest.raises(ValueError, match='branch must be'):
        Qwen35Checkpoint(path, branch='vision')


def test_mtp_contract_shares_embedding_and_head(cfg):
    c = Qwen35Config.from_json(cfg)
    shapes = c.expected_mtp_tensors()
    assert shapes['mtp.fc.weight'] == (2048, 4096)
    assert shapes['mtp.layers.0.self_attn.q_proj.weight'] == (8192, 2048)
    assert shapes['mtp.layers.0.mlp.experts.255.down_proj.weight'] == (2048, 512)
    assert not any('embed_tokens' in n or n == 'lm_head.weight' or 'linear_attn' in n for n in shapes)


def test_mtp_branch_reads_native_bytes_and_rejects_missing_tensors(checkpoint, monkeypatch):
    path, header, write, index = checkpoint
    config_value = json.loads((path/'config.json').read_text())
    config_value['text_config'].update(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    (path/'config.json').write_text(json.dumps(config_value))
    shapes = {}
    for name in list(header):
        mtp_name = name.replace('model.language_model.', 'mtp.')
        header[mtp_name] = header.pop(name)
        index['weight_map'][mtp_name] = index['weight_map'].pop(name)
        shapes[mtp_name] = tuple(header[mtp_name]['shape'])
    monkeypatch.setattr(Qwen35Config, 'expected_mtp_tensors', lambda self: shapes)
    write(); (path/'model.safetensors.index.json').write_text(json.dumps(index))
    c = Qwen35Checkpoint(path, branch='mtp')
    assert c.weight_bytes == 16394
    np.testing.assert_array_equal(c.read('mtp.test.weight'), np.full((128, 128), 2, 'uint8'))
    del index['weight_map']['mtp.norm.weight']
    (path/'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='mtp tensor set mismatch'):
        Qwen35Checkpoint(path, branch='mtp')
