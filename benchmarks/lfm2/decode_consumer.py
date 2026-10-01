"""Compiler-free consumer of experimental split-KV plans."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
from benchmarks.lfm2.consumer import Guard,consume
_sys.meta_path.insert(0,Guard())
from benchmarks.lfm2.decode_optimization import OptimizedLFM2
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();consume(a.model,a.bundle,a.reference,a.out,engine_cls=OptimizedLFM2)
