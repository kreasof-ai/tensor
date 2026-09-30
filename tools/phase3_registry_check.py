"""Exercise registry restoration in a compiler-free consumer, optionally on GPU."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading

from phase1_transfer_check import NoCompilerImports, exercise


def check(wheel, *, execute=False):
    sys.meta_path.insert(0, NoCompilerImports())
    from tensor.modules import Project, add, install
    from tensor.registry import read_wheel

    wheel = Path(wheel)
    graph, packages = read_wheel(wheel)
    version = packages[graph['root']].manifest['version']
    distribution = wheel.name.split('-')[0].replace('_', '-')
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='tensor-registry-consumer-') as directory:
        root = Path(directory)
        files = root / 'index/files'
        files.mkdir(parents=True)
        shutil.copyfile(wheel, files / wheel.name)
        page = root / 'index/simple' / distribution
        page.mkdir(parents=True)
        (page / 'index.html').write_text(f'<a href="../../files/{wheel.name}#sha256={wheel_hash}">{wheel.name}</a>', encoding='utf-8')

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(root / 'index')))
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        thread.start()
        project = root / 'app'
        project.mkdir()
        (project / 'tensor.json').write_text(json.dumps({'formatVersion': 1, 'name': 'consumer',
            'version': '0.1.0', 'tensorAbi': 1, 'exports': {}}), encoding='utf-8')
        try:
            added = add(f'pypi:{distribution}=={version}', project, cache_dir=root / 'first-cache',
                        index_url=f'http://127.0.0.1:{server.server_port}/simple/')
            assert added['pypi']['sha256'] == wheel_hash
            lock = (project / 'tensor.lock').read_bytes()
            cache = root / 'fresh-cache'
            install(project, cache_dir=cache, frozen=True)
            assert (project / 'tensor.lock').read_bytes() == lock
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        shutil.rmtree(root / 'index')
        install(project, cache_dir=cache, frozen=True, offline=True)
        assert (project / 'tensor.lock').read_bytes() == lock
        installed = Project(project, cache_dir=cache)
        artifacts, selections = {}, {}
        for name, data in packages.items():
            mod = installed.module(name)
            for export, spec in data.manifest['exports'].items():
                if spec.get('artifacts'):
                    resolved = mod.resolve(export, target='sm_86')
                    assert resolved['selection'] == 'packaged'
                    artifacts[export] = Path(resolved['path'])
                    selections[f'{name}::{export}'] = resolved['selection']
        runtime = exercise(artifacts) if execute else None
        assert not list((cache / 'artifacts').glob('*.tbin'))
        return {'status': 'passed', 'wheel_sha256': wheel_hash, 'distribution': distribution,
                'modules': graph['packages'], 'fresh_cache_restore': True,
                'offline_frozen_install': True, 'compiler_imports': False, 'selections': selections,
                'runtime': runtime}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--execute', action='store_true', help='execute all five validation profiles on sm_86')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.wheel, execute=args.execute)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'exports': len(report['selections']),
                      'fresh_cache_restore': True, 'offline_frozen_install': True,
                      'cases': len(report['runtime']['records']) if report['runtime'] else 0}))
