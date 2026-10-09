"""Architecture contract inferred from GGUF metadata and checked tensor shapes."""
from dataclasses import dataclass
import math
from .gguf import GGUFError


@dataclass(frozen=True)
class Config:
    width: int
    ff: int
    heads: int
    kv_heads: int
    layers: tuple[str,...]
    vocab: int
    conv: int
    epsilon: float
    theta: float
    max_context: int

    @property
    def head_dim(self):return self.width//self.heads

    @classmethod
    def from_gguf(cls,gguf):
        m=gguf.metadata
        if m.get('general.architecture')!='lfm2':raise GGUFError('requires the LFM2 architecture')
        get=lambda name:m['lfm2.'+name]
        kv=get('attention.head_count_kv')
        if not isinstance(kv,list) or len(kv)!=get('block_count'):raise GGUFError('requires per-layer attention metadata')
        unique=set(kv)-{0}
        if len(unique)!=1:raise GGUFError('requires uniform nonzero GQA head count')
        config=cls(get('embedding_length'),get('feed_forward_length'),get('attention.head_count'),unique.pop(),
                   tuple('attention' if n else 'conv' for n in kv),get('vocab_size'),get('shortconv.l_cache'),
                   get('attention.layer_norm_rms_epsilon'),get('rope.freq_base'),get('context_length'))
        if any(type(value) is not int or value <= 0 for value in
               (config.width, config.ff, config.heads, config.kv_heads, config.vocab, config.max_context)):
            raise GGUFError('LFM2 dimensions must be positive integers')
        if not config.layers or any(type(n) is not int or n < 0 for n in kv):
            raise GGUFError('invalid LFM2 layer/head metadata')
        if not all(math.isfinite(value) and value > 0 for value in (config.epsilon, config.theta)):
            raise GGUFError('LFM2 normalization/RoPE constants must be finite and positive')
        if config.width%config.heads or config.heads%config.kv_heads or config.head_dim!=64 or config.ff%32 or config.vocab%4 or config.conv!=3:
            raise GGUFError('unsupported LFM2 head/convolution configuration')
        expected={'token_embd.weight':(config.vocab,config.width),'token_embd_norm.weight':(config.width,)}
        for layer,kind in enumerate(config.layers):
            p=f'blk.{layer}.';c=config.width;f=config.ff;d=config.head_dim;kh=config.kv_heads
            expected.update({p+'attn_norm.weight':(c,),p+'ffn_norm.weight':(c,),p+'ffn_gate.weight':(f,c),
                             p+'ffn_up.weight':(f,c),p+'ffn_down.weight':(c,f)})
            if kind=='conv':
                expected.update({p+'shortconv.in_proj.weight':(3*c,c),p+'shortconv.out_proj.weight':(c,c),p+'shortconv.conv.weight':(c,3)})
            else:
                expected.update({p+'attn_q.weight':(c,c),p+'attn_k.weight':(kh*d,c),p+'attn_v.weight':(kh*d,c),
                                 p+'attn_output.weight':(c,c),p+'attn_q_norm.weight':(d,),p+'attn_k_norm.weight':(d,)})
        if 'output.weight' in gguf.tensors:expected['output.weight']=(config.vocab,config.width)
        if set(expected)!=set(gguf.tensors):raise GGUFError('unsupported/missing LFM2 tensor inventory')
        for name,shape in expected.items():
            if gguf.tensors[name].shape!=shape:raise GGUFError(f'{name}: incompatible LFM2 tensor shape')
            if len(shape)==1 or name.endswith('conv.weight'):
                if gguf.tensors[name].type!=0:raise GGUFError(f'{name}: requires float32 auxiliary weights')
        return config
