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
}


def download(out, kinds):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in kinds:
        expected, size = FILES[kind]
        name = f'LFM2.5-2.6B-{kind}.gguf'; path = out / name
        if not path.exists():
            partial = path.with_suffix('.partial')
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true'
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
        {'repository': REPOSITORY, 'revision': REVISION, 'files': records}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('build/lfm2-models'))
    parser.add_argument('--formats', nargs='+', choices=FILES, default=list(FILES))
    args = parser.parse_args(); download(args.out, args.formats)
