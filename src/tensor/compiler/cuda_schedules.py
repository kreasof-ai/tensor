"""CUDA schedule spaces supplied to device-independent discovery."""

PREFILL_SPACE = dict(block_m=(16,32,64),block_n=(64,128),block_k=(32,64,128),
                     stages=(2,3),threads=(128,256),packed_pairs=(False,True))
DECODE_SPACE = dict(threads=(64,128,256),unroll=(1,2,4,8))
ATTENTION_SPACES = {'partial': {'splits': (16,)},
                    'grouped': {'stage': (16,32,64), 'warps': (2,4)}}


def projection_space(phase, *, f16_vectors=False, family='tiles'):
    if phase=='decode':
        return {'gemv': {**DECODE_SPACE, **({'f16_values': (4,8,16)} if f16_vectors else {})}}
    if phase!='prefill':raise ValueError('requires prefill or decode schedule space')
    space=dict(PREFILL_SPACE)
    if family=='paired-loads':space['packed_pairs']=(True,)
    if family=='small-rows':space.update(block_m=(16,32),block_n=(64,),stages=(2,),threads=(128,))
    return {'mma': space}


def projection_legal(config, *, depth, paired=False, packed_pairs=True, shared_limit=99*1024):
    if config['family']=='gemv':
        width=config.get('f16_values',4)
        return depth%256==0 and depth%(32*width)==0
    bk,bm,bn,stages=(config[name] for name in ('block_k','block_m','block_n','stages'))
    # SM86 has a 99 KiB opt-in shared-memory limit per CTA. Padding/layout
    # details remain the compiler's responsibility and may reject a candidate.
    shared=(bm+(2 if paired else 1)*bn)*bk*2*stages
    return depth%bk==0 and shared<=shared_limit and (not config['packed_pairs'] or packed_pairs)
