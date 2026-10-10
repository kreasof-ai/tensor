"""Compare wider batches using each request's last valid control output."""
import numpy as np
import hashlib


def forward(executor,batch,lengths):
    chunk=executor.chunk
    if batch.shape[1]%chunk:raise ValueError('control chunk must divide the comparison batch')
    for offset in range(0,batch.shape[1],chunk):
        counts=np.clip(lengths-offset,0,chunk).astype('int32')
        result,logits=executor.forward(batch[:,offset:offset+chunk],counts,read_logits=True)
        if offset==0:
            final_result=result.copy();final_logits=logits.copy()
        else:
            # Later inactive chunks may zero temporary head storage. Preserve
            # the prediction and logits from the last actual input per slot.
            active=counts>0;final_result[active]=result[active];final_logits[active]=logits[active]
    return final_result,final_logits


def cache_prefix_hashes(model):
    """Hash only initialized cache rows; unused capacity has no defined value."""
    result={}
    for layer,state in model.states.items():
        if model.config.layers[layer]=='linear_attention':continue
        for index,buffer in enumerate(state):
            value=buffer.to_numpy();digest=hashlib.sha256()
            for slot,count in enumerate(model.position):
                digest.update(np.ascontiguousarray(value[slot,:,:int(count),:]).tobytes())
            result[f'{layer}:{index}']=digest.hexdigest()
    return result
