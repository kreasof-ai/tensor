"""Run the complete native Qwen decoder and independently audit its logits."""
import argparse
import code
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensor
from tensor_llm.qwen35.decode import Qwen35Batch


def run(checkpoint,bundle,out,*,steps=3,oracle=True,decode_steps=128,interactive=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='tensor.qwen35-native-qualification.v1',status='running',
        protocol='native block-FP8; BF16 activations/KV; FP32 recurrent state; pure AR',
        model_throughput_qualified=False,full_stress_target_reached=False,validation=[],timing={})
    def save():
        temp=out/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out/'report.json')
    def progress(message):
        print(message,flush=True)
        (out/'progress.txt').write_text(message+'\n')
    save()
    with tensor.Device() as d,Qwen35Batch(checkpoint,bundle,d,progress=progress) as model:
        report['device']=d.info;report['buffer_bytes']=model.allocated_bytes
        report['weight_bytes']=model.checkpoint.weight_bytes
        report['artifact_manifest_sha256']=hashlib.sha256((Path(bundle)/'inference.json').read_bytes()).hexdigest()
        reference=None
        if oracle:
            from .reference import Reference
            reference=Reference(checkpoint,model.slots)
        failure=None
        try:
            for step in range(steps):
                tokens=np.arange(model.slots,dtype='int32')*17+256+step
                actual_stages={}
                def capture(label,value):
                    actual_stages[label]=value
                    np.save(out/f'stage-{step}-{label[0]}-{label[1]}.npy',value)
                _,actual=model.forward(tokens,read_logits=True,debug=capture)
                progress(f'Native C{model.slots} forward {step+1}: finite={bool(np.isfinite(actual).all())}')
                if not np.isfinite(actual).all():raise AssertionError('native logits contain non-finite values')
                np.save(out/f'logits-{step}.npy',actual)
                if reference:
                    row=dict(step=step,stages=[])
                    report['validation'].append(row)
                    def compare(label,want):
                        got=actual_stages[label]
                        error=float(np.linalg.norm(got-want)/max(float(np.linalg.norm(want)),1e-12))
                        row['stages'].append(dict(layer=label[0],stage=label[1],relative_rms=error))
                        np.save(out/f'expected-{step}-{label[0]}-{label[1]}.npy',want)
                        save()
                        if error>.03:raise AssertionError(f'Qwen oracle mismatch at {label}: {error}')
                    expected=reference.forward(tokens,debug=compare,progress=progress)
                    rms=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                    row.update(relative_rms=rms,greedy_matches=int((actual.argmax(-1)==expected.argmax(-1)).sum()));save()
                    if rms>.03:raise AssertionError(f'Qwen final logit oracle mismatch: {rms}')
        except Exception as error:
            failure=error;report.update(status='failed',error=f'{type(error).__name__}: {error}');save()
            progress(report['error'])
            if not interactive:raise
        if interactive:
            progress('Native checkpoint retained for numerical diagnosis and kernel tuning')
            code.interact(local=locals())
            return
        # This small-context timing is diagnostic only, explicitly separated
        # from the 32K/16K finite replay required for the 600 tok/s target.
        model.reset()
        model.forward(np.arange(model.slots,dtype='int32')+256)
        for _ in range(8):model.step()
        started=time.perf_counter()
        for _ in range(decode_steps):model.step()
        elapsed=time.perf_counter()-started
        report['timing']=dict(kind='short-context-decode-diagnostic',steps=decode_steps,
            output_tokens=model.slots*decode_steps,wall_seconds=elapsed,
            output_tokens_per_second=model.slots*decode_steps/elapsed,
            maximum_context=int(model.position.max()))
    report['status']='passed' if oracle else 'native-execution-only';save()
    print(json.dumps(report['timing']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--decode-steps',type=int,default=128);p.add_argument('--no-oracle',action='store_true')
    p.add_argument('--interactive',action='store_true',help='retain native weights for producer-side diagnosis and tuning')
    a=p.parse_args();run(a.checkpoint,a.bundle,a.out,steps=a.steps,oracle=not a.no_oracle,
                        decode_steps=a.decode_steps,interactive=a.interactive)
