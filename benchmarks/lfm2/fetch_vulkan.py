"""Fetch and verify the Windows Vulkan release used by vulkan_reference.py."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

RELEASE='b11310'
COMMIT='f872b591121761ac7b2af18283bd99bdc092a63a'
SHA256='b93a7e765c09ce5151457b71028fae2a61d4887af88d5280faa324f87162b4c1'
URL=f'https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/llama-{RELEASE}-bin-win-vulkan-x64.zip'


def fetch(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    archive=out/'release.zip'
    if not archive.exists():
        request=urllib.request.Request(URL,headers={'User-Agent':'tensor-lfm2-benchmark'})
        with urllib.request.urlopen(request) as response,archive.open('wb') as target:
            while chunk:=response.read(1024*1024):target.write(chunk)
    if hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()!=SHA256:
        raise ValueError('llama.cpp release checksum mismatch')
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if not (out/entry.filename).resolve().is_relative_to(out):
                raise ValueError('archive path escapes output')
        bundle.extractall(out)
    result={'release':RELEASE,'commit':COMMIT,'archive_sha256':SHA256,
            'dlls':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.glob('*.dll')}}
    (out/'release.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('build/llama-vulkan-b11310'))
    print(json.dumps(fetch(parser.parse_args().out),indent=2))
