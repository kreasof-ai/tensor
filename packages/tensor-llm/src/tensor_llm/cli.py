"""Inspect a GGUF or generate text using a precompiled Tensor LFM2 bundle."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from .common.gguf import GGUF
from .lfm2.config import Config


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect',help='inspect model architecture and packed tensor types')
    inspect.add_argument('--model',required=True,type=Path)
    generate=commands.add_parser('generate',help='generate with a precompiled CUDA or WebGPU bundle')
    generate.add_argument('--model',required=True,type=Path);generate.add_argument('--bundle',required=True,type=Path)
    generate.add_argument('--prompt',required=True);generate.add_argument('--max-tokens',default=128,type=int)
    generate.add_argument('--context',type=int,help='default: 512 for WebGPU, 8448 for CUDA');generate.add_argument('--raw',action='store_true',help='use a plain completion prompt')
    generate.add_argument('--out',type=Path,help='save generated token IDs and text as JSON')
    generate.add_argument('--provider',choices=('cuda','webgpu'),default='cuda')
    generate.add_argument('--device',type=int,default=0)
    args=parser.parse_args()
    if args.command=='inspect':
        gguf=GGUF(args.model);print(json.dumps({'architecture':asdict(Config.from_gguf(gguf)),
            'tensors':len(gguf.tensors),'encodings':dict(Counter(t.encoding for t in gguf.tensors.values()))},indent=2));return
    import tensor
    from .lfm2.model import LFM2
    context=args.context if args.context is not None else (512 if args.provider=='webgpu' else 8448)
    with tensor.Device(args.device,provider=args.provider) as device,LFM2(args.model,args.bundle,device,context=context) as model:
        result=model.generate(args.prompt,max_tokens=args.max_tokens,chat=not args.raw)
    print(result['text'])
    if args.out:args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
