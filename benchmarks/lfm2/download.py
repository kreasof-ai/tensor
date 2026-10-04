"""Download and verify immutable LiquidAI GGUF checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = 'LiquidAI/LFM2.5-2.6B-GGUF'
REVISION = 'e7caca5d835a3901a8e0d63e94009429bafafdfc'
FILES = {
    'F16': ('e041c231351185eb390f9c417d3bfd1815869a50a8589f3f86e5b9add3c529f1', 5403158528),
    'Q4_0': ('e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe', 1593894912),
    'Q4_K_M': ('02a8b7e17487d326e46d68ce0ba24211e1b80a14c4cd0597fa73c1cd697f52ed', 1674455040),
    'QAD-Q4_0': ('a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03', 1593894944),
}
PROFILES = {
    '2.6B': (REPOSITORY, REVISION, FILES),
    '230M': ('LiquidAI/LFM2.5-230M-GGUF', '03502067c64ce32ac4fe87b0cec0310a1a13d3e9', {
        'F16': ('4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f', 461884256),
        'Q4_0': ('430fbec5b1b355e9bb12cd0638c9f2a8f21fedd6eafb4103e42c7e88887daa73', 149080928),
        'Q4_K_M': ('7bbd90384d3deffe4c646ec9643b212802d32d4ce417c90a1ec9282100650062', 153406304),
    }),
}


def download(out, kinds, *, model_size='2.6B'):
    repository, revision, files = PROFILES[model_size]
    if any(kind not in files for kind in kinds):
        raise ValueError(f'unsupported checkpoint format for {model_size}')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in kinds:
        expected, size = files[kind]
        name = f'LFM2.5-{model_size}-{kind}.gguf'; path = out / name
        if not path.exists():
            partial = path.with_suffix('.partial')
            url = f'https://huggingface.co/{repository}/resolve/{revision}/{name}?download=true'
            print(f'downloading {name} ({size:,} bytes)', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as target:
                while block := response.read(8 * 1024 * 1024):
                    target.write(block)
            with partial.open('rb') as data:
                actual = hashlib.file_digest(data, 'sha256').hexdigest()
            if partial.stat().st_size != size or actual != expected:
                raise ValueError(f'{name}: download checksum/size mismatch')
            partial.replace(path)
        with path.open('rb') as data:
            actual = hashlib.file_digest(data, 'sha256').hexdigest()
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f'{name}: installed checkpoint checksum/size mismatch')
        records.append({'file': name, 'bytes': size, 'sha256': actual})
        print(f'verified {name}', flush=True)
    (out / 'provenance.json').write_text(json.dumps(
        {'repository': repository, 'revision': revision, 'files': records}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--model-size', choices=PROFILES, default='2.6B',
                        help='use 230M for faster local GPU iterations')
    parser.add_argument('--formats', nargs='+', choices=FILES, default=['F16','Q4_0','Q4_K_M'])
    args = parser.parse_args()
    out = args.out or Path('build/lfm2-230m-models' if args.model_size == '230M' else 'build/lfm2-models')
    download(out, args.formats, model_size=args.model_size)
