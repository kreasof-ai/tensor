"""Create a small mixed-encoding LFM2 GGUF for GPU-free compilation CI.

This is a two-layer synthetic contract fixture, not a substitute for the pinned
30-layer LiquidAI model validation or performance measurements.
"""
import argparse
from pathlib import Path
import struct
import numpy as np


def string(text):
    data=text.encode('utf-8');return struct.pack('<Q',len(data))+data


def fixture(path):
    c,f,d,v=256,512,64,288
    values=list(range(33,127))+list(range(161,173))+list(range(174,256));chars=values.copy()
    for b in range(256):
        if b not in values:values.append(b);chars.append(256+len(chars)-188)
    tokens=list(map(chr,chars))+['<|startoftext|>','<|im_start|>','<|im_end|>','<think>']
    tokens += [f'[unused{i}]' for i in range(v-len(tokens))]
    metadata={'general.architecture':'lfm2','lfm2.embedding_length':c,'lfm2.feed_forward_length':f,
        'lfm2.attention.head_count':4,'lfm2.attention.head_count_kv':[0,1],'lfm2.block_count':2,
        'lfm2.vocab_size':v,'lfm2.shortconv.l_cache':3,'lfm2.attention.layer_norm_rms_epsilon':1e-6,
        'lfm2.rope.freq_base':10000000.,'lfm2.context_length':384,
        'tokenizer.ggml.model':'gpt2','tokenizer.ggml.pre':'lfm2','tokenizer.ggml.tokens':tokens,
        'tokenizer.ggml.token_type':[1]*256+[3]*4+[5]*(v-260), 'tokenizer.ggml.merges':[],
        'tokenizer.ggml.bos_token_id':256,'tokenizer.ggml.eos_token_id':258}
    def encode(value):
        if isinstance(value,str):return struct.pack('<I',8)+string(value)
        if isinstance(value,int):return struct.pack('<II',4,value)
        if isinstance(value,float):return struct.pack('<If',6,value)
        subtype=8 if value and isinstance(value[0],str) else 4
        data=b''.join(string(item) if subtype==8 else struct.pack('<I',item) for item in value)
        return struct.pack('<IIQ',9,subtype,len(value))+data
    shapes={'token_embd.weight':((v,c),14),'token_embd_norm.weight':((c,),0)}
    for i in range(2):
        p=f'blk.{i}.'
        shapes.update({p+'attn_norm.weight':((c,),0),p+'ffn_norm.weight':((c,),0),
            p+'ffn_gate.weight':((f,c),12),p+'ffn_up.weight':((f,c),1),p+'ffn_down.weight':((c,f),14)})
        if i==0:shapes.update({p+'shortconv.in_proj.weight':((3*c,c),2),p+'shortconv.out_proj.weight':((c,c),1),p+'shortconv.conv.weight':((c,3),0)})
        else:shapes.update({p+'attn_q.weight':((c,c),1),p+'attn_k.weight':((d,c),1),p+'attn_v.weight':((d,c),1),
            p+'attn_output.weight':((c,c),1),p+'attn_q_norm.weight':((d,),0),p+'attn_k_norm.weight':((d,),0)})
    rng=np.random.default_rng(291);entries=bytearray();payload=bytearray()
    for name,(shape,kind) in shapes.items():
        payload.extend(b'\0'*((-len(payload))%32));offset=len(payload);n=int(np.prod(shape))
        entries.extend(string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape[::-1])+struct.pack('<IQ',kind,offset))
        if kind in (0,1):
            data=np.ones(shape,np.float32) if len(shape)==1 else rng.normal(0,.015,shape)
            payload.extend(data.astype(np.float32 if kind==0 else np.float16).tobytes())
        else:
            block,bytes_per_block={2:(32,18),12:(256,144),14:(256,210)}[kind]
            data=rng.integers(0,256,(n//block,bytes_per_block),dtype=np.uint8)
            if kind==2:data[:,:2]=np.frombuffer(struct.pack('<e',.004),np.uint8)
            elif kind==12:data[:,:4]=np.frombuffer(struct.pack('<ee',.00015,.0008),np.uint8)
            else:
                data[:,192:208]=rng.integers(-4,5,(n//block,16),dtype=np.int8).view(np.uint8)
                data[:,208:210]=np.frombuffer(struct.pack('<e',.0003),np.uint8)
            payload.extend(data.tobytes())
    header=b'GGUF'+struct.pack('<IQQ',3,len(shapes),len(metadata))
    header+=b''.join(string(key)+encode(value) for key,value in metadata.items())+entries
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(header+b'\0'*((-len(header))%32)+payload)
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();fixture(args.out)
