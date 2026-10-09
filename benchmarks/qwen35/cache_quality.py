"""Matched teacher-forced cache quality, retaining one resident checkpoint."""
from pathlib import Path
import hashlib,json,time
import numpy as np


def run(model,bf16_bundle,fp8_bundle,checkpoint,out,*,steps=512,tokens=None,full_loss=False):
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(checkpoint)/'tokenizer.json'))
    texts=[
        'Explain how a compiler schedules matrix multiplication on a GPU. ',
        'A farmer has twenty apples and gives seven away. How many remain? ',
        'Write Python code to implement a queue and explain its complexity. ',
        'Summarize the history of astronomy and the discovery of planets. ',
        'Describe a journey through mountains, rivers and a quiet village. ',
        'What makes a scientific experiment reproducible and reliable? ',
        'Translate this sentence into French: The garden is beautiful today. ',
        'Compare memory bandwidth and arithmetic throughput in an inference engine. ']
    if tokens is None:
        streams=[tokenizer.encode(t,add_special_tokens=False).ids for t in texts[:model.slots]]
        tokens=np.stack([np.resize(np.array(s,'int32'),steps) for s in streams],axis=1)
        protocol='short natural-language texts cyclically repeated; calibration stress, not a held-out quality corpus'
    else:
        tokens=np.asarray(tokens)
        if (tokens.shape!=(steps,model.slots) or tokens.dtype.kind not in 'iu'
                or np.any(tokens<0) or np.any(tokens>=model.config.vocab)):
            raise ValueError('quality token matrix must be [steps, slots] with valid token IDs')
        tokens=tokens.astype('int32');protocol='frozen externally supplied teacher-forced calibration tokens'
    indices={i-1 for i in (1,2,8,16,32,64,128,256,512) if i<=steps}
    logits={};timings={};losses={};manifests={}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for dtype,bundle in (('bfloat16',bf16_bundle),('fp8',fp8_bundle)):
        model.reset();model.reset_cache_profile(bundle)
        manifests[dtype]=hashlib.sha256((Path(bundle)/'inference.json').read_bytes()).hexdigest()
        started=time.perf_counter();samples={};nll=[]
        for i,row in enumerate(tokens):
            if i in indices or full_loss:
                _,actual=model.forward(row,read_logits=True)
                if i in indices:samples[i]=actual
                if full_loss and i+1<steps:
                    shifted=actual-actual.max(-1,keepdims=True)
                    log_sum=np.log(np.exp(shifted).sum(-1))
                    nll.extend((log_sum-shifted[np.arange(model.slots),tokens[i+1]]).tolist())
            else:model.forward(row)
        timings[dtype]=time.perf_counter()-started;logits[dtype]=samples
        if nll:losses[dtype]=dict(tokens=len(nll),mean_negative_log_likelihood=float(np.mean(nll)))
        print('teacher forced',dtype,timings[dtype],flush=True)
    rows=[]
    for i in sorted(indices):
        ref=logits['bfloat16'][i];got=logits['fp8'][i]
        ref_log=ref-ref.max(-1,keepdims=True);got_log=got-got.max(-1,keepdims=True)
        ref_log-=np.log(np.exp(ref_log).sum(-1,keepdims=True));got_log-=np.log(np.exp(got_log).sum(-1,keepdims=True))
        kl=(np.exp(ref_log)*(ref_log-got_log)).sum(-1)
        rows.append(dict(context=i+1,relative_logit_rms=float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),
            greedy_matches=int((got.argmax(-1)==ref.argmax(-1)).sum()),slots=model.slots,
            mean_kl=float(kl.mean()),maximum_kl=float(kl.max()),finite=bool(np.isfinite(got).all())))
    # These cache checks supplement the existing complete-model CPU oracle;
    # they cannot override a failed model qualification or certify throughput.
    passed=all(r['finite'] and r['relative_logit_rms']<=.03 for r in rows)
    report=dict(schema='tensor.qwen35-kv-quality.v1',status='passed' if passed else 'failed',
        protocol=protocol,artifact_manifests=manifests,losses=losses,
        token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),steps=steps,
        gates=dict(relative_logit_rms_maximum=.03),samples=rows,diagnostic_wall_seconds=timings,
        model_throughput_qualified=False,full_stress_target_reached=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    model.reset();return report
