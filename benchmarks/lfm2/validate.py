"""Compare Tensor LFM2 tokenization, quantized blocks and logits with llama.cpp."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,ctypes as ct,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import GGUF
from tensor_llm.lfm2.model import LFM2
from tensor_llm.common.tokenizer import Tokenizer
from tensor_llm.common.gguf import dequantize


def block_check(gguf,library):
    lib=ct.CDLL(str(library.resolve()));observations=[]
    for kind in sorted({t.type for t in gguf.tensors.values()}-{0,1}):
        name=next(n for n,t in gguf.tensors.items() if t.type==kind);info=gguf.tensors[name]
        # Several rows and block boundaries from the actual checkpoint.
        for row in (0,1,info.shape[0]-1):
            width=info.nbytes//info.shape[0];packed=np.ascontiguousarray(gguf.packed(name)[row*width:(row+1)*width])
            expected=np.empty(info.shape[1],np.float32);function=getattr(lib,'dequantize_row_'+info.encoding.lower().replace('_k','_K'))
            function.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int64];function.restype=None
            function(packed.ctypes.data,expected.ctypes.data,len(expected))
            actual=dequantize(packed,kind);np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
            observations.append({'type':info.encoding,'tensor':name,'row':row,'max_error':float(np.max(np.abs(actual-expected)))})
    return observations


def validate(model,bundle,reference_executable,ggml_library,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'validation.json').unlink(missing_ok=True);gguf=GGUF(model);tokenizer=Tokenizer(gguf.metadata)
    text='Explain why the sky appears blue in two sentences.'
    prompt=tokenizer.chat(text)
    forced=tokenizer.encode(' The atmosphere scatters blue light more strongly.',add_bos=False)[:12]
    items=[{'tokens':prompt,'reset':True,'trace':True}]+[{'tokens':[int(token)]} for token in forced]
    cases=[{'text':value,'bos':True,'special':False} for value in [text,'Hello, 世界! café\n\n123456789',"It's raining—don't forget your umbrella.",'<|im_start|>']]
    cases.append({'text':'<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>','bos':True,'special':True})
    pattern=tokenizer.encode('The same fixed text checks convolution history and cached attention. ',add_bos=False)
    for length in (127,128,129,512,2048,8192):
        prefix=[tokenizer.bos]+[pattern[j%len(pattern)] for j in range(length-1)]
        items.extend([{'tokens':prefix,'reset':True}, {'tokens':forced[:1]}])
    # Reset after the longest prefix must reproduce the original prompt exactly.
    items.append({'tokens':prompt,'reset':True})
    spec={'model':str(Path(model).resolve()),'out':str((out/'llama').resolve()),'context':8448,'tokenization':cases,'validation':items}
    specfile=out/'reference-spec.json';specfile.write_text(json.dumps(spec,indent=2)+'\n')
    with (out/'llama.log').open('w') as log:subprocess.run([str(reference_executable.resolve()),str(specfile)],check=True,stderr=log,stdout=log)
    reference=json.loads((out/'llama/reference.json').read_text())
    for item in reference['tokenization']:
        assert tokenizer.encode(item['text'],add_bos=item['bos'],parse_special=item['special'])==item['tokens'],item['text']
    blocks=block_check(gguf,ggml_library)
    report={'schema':'tensor.lfm2-validation.v1','status':'running','model_sha256':hashlib.file_digest(Path(model).open('rb'),'sha256').hexdigest(),
            'block_checks':blocks,'tokenization_cases':len(cases),'steps':[]}
    saved=[]
    with tensor.Device() as device,LFM2(model,bundle,device) as network:
        for i,item in enumerate(items):
            if item.get('reset'):network.reset()
            actual=network.forward(item['tokens']);expected=np.fromfile(out/'llama'/f'{i}-logits.bin',np.float32)
            difference=actual-expected
            metrics={'step':i,'position':network.position,'max_error':float(np.max(np.abs(difference))),
                'rmse':float(np.sqrt(np.mean(difference*difference))), 'reference_top1':int(np.argmax(expected)),
                'tensor_top1':int(np.argmax(actual)),'cosine':float(np.dot(actual.astype(np.float64),expected.astype(np.float64))/(np.linalg.norm(actual.astype(np.float64))*np.linalg.norm(expected.astype(np.float64))))}
            np.save(out/f'tensor-{i}-logits.npy',actual)
            assert np.all(np.isfinite(actual)), 'non-finite model logits'
            saved.append(actual);report['steps'].append(metrics);print(metrics,flush=True)
        report.update(adapter=device.info,owned_device_bytes=network.allocated_bytes,launches_per_decode=len(network.plans[1]))
    np.testing.assert_array_equal(saved[0],saved[-1])
    # Same GGUF decoding, independent eager mixed-precision operators.
    from benchmarks.lfm2.torch_reference import Reference
    eager=Reference(model);report['independent_reference']=[]
    for i,item in enumerate(items):
        if item.get('reset'):eager.reset()
        for start in range(0,len(item['tokens']),128):
            expected=eager.forward(item['tokens'][start:start+128])
        actual=saved[i];delta=actual-expected
        normalized_rmse=float(np.linalg.norm(delta)/np.linalg.norm(expected))
        cosine=float(np.dot(actual.astype(np.float64),expected.astype(np.float64))/(np.linalg.norm(actual.astype(np.float64))*np.linalg.norm(expected.astype(np.float64))))
        metrics={'step':i,'position':eager.position,'normalized_rmse':normalized_rmse,'cosine':cosine,
                 'max_error':float(np.max(np.abs(delta))),'top1_equal':bool(np.argmax(actual)==np.argmax(expected))}
        report['independent_reference'].append(metrics);print('independent',metrics,flush=True)
        assert np.all(np.isfinite(expected))
        assert normalized_rmse < 0.01 and cosine > 0.9999, metrics
    report['gates']={'finite_logits':True,'reset_bitwise_equal':True,
        'independent_normalized_rmse_max':0.01,'independent_cosine_min':0.9999,
        'ggml_block_rtol':1e-6,'ggml_block_atol':1e-7,
        'llama_logits':'reported, not an equality gate: different activation quantization'}
    report['status']='passed'
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','ggml-library','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();validate(a.model,a.bundle,a.reference,a.ggml_library,a.out)

if __name__=='__main__':main()
