"""Compare AOT schedules on identical complete decoder states and long contexts."""
import ctypes as ct
import importlib,json,time
from pathlib import Path
import numpy as np
from tensor_llm.qwen35.pipeline import replace


class Snapshot:
    def __init__(self,model):
        self.position=model.position.copy();self.active=model.active.copy()
        self.tokens=model.buffers['tokens'].to_numpy()
        # Decode appends KV entries; old prefix entries are immutable. Restoring
        # the position masks appended entries without copying multi-GiB KV.
        self.states=[(buffer,buffer.to_numpy()) for layer,state in model.states.items()
            if model.config.layers[layer]=='linear_attention' for buffer in state]

    def restore(self,model):
        model._check();model.device.synchronize()
        for buffer,array in self.states:
            model.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(array.ctypes.data),array.nbytes)
        model.device.driver.call('cuStreamSynchronize',None)
        model.position=self.position.copy();model.active=self.active.copy()
        for name,value in (('tokens',self.tokens),('positions',self.position),('active',self.active)):model._write(name,value)


def compare(model,bundles,out,*,steps=64,repetitions=3):
    model._check();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    snapshot=Snapshot(model);records=[]
    for repetition in range(repetitions):
        # Alternate order to reduce bias from clock/thermal drift.
        order=list(bundles)
        if repetition%2:order.reverse()
        for bundle in order:
            try:replace(model,bundle)
            except Exception as error:
                if model.closed or model.graph is None:raise
                records.append(dict(bundle=str(bundle),repetition=repetition,status='rejected',
                    error=f'{type(error).__name__}: {error}',full_stress_target_reached=False))
                (out/'timings.json').write_text(json.dumps(records,indent=2)+'\n')
                print('Whole decoder schedule rejected',records[-1],flush=True)
                continue
            snapshot.restore(model)
            started=time.perf_counter();generated=[]
            for _ in range(steps):generated.append(model.step())
            seconds=time.perf_counter()-started
            record=dict(bundle=str(bundle),repetition=repetition,status='measured',context_start=int(snapshot.position.max()),
                steps=steps,output_tokens=steps*int(snapshot.active.sum()),seconds=seconds,
                output_tokens_per_second=steps*int(snapshot.active.sum())/seconds,
                kind='same-state-full-context-decode-diagnostic',full_stress_target_reached=False)
            records.append(record)
            np.save(out/(Path(bundle).name+f'-r{repetition}-tokens.npy'),np.stack(generated))
            (out/'timings.json').write_text(json.dumps(records,indent=2)+'\n')
            print('Whole decoder schedule',record,flush=True)
    snapshot.restore(model)
    return records
