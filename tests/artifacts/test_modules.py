"""Offline package integrity, graph resolution and portable/export boundaries."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from tensor.modules import ModuleError, Project, add, install, pack

ROOT = Path(__file__).resolve().parents[2]


def module(path, name, *, dependencies=None, exports=None, files=None):
    path.mkdir(parents=True, exist_ok=True)
    value = {"formatVersion":1,"name":name,"version":"0.1.0","tensorAbi":1,
             "exports": {} if exports is None else exports,"dependencies":dependencies or {},"files":files or []}
    (path / "tensor.json").write_text(json.dumps(value))
    return path


def dependency(path):
    return {"path":str(path),"version":"0.1.0"}


def source(path, contents="raise RuntimeError('package operations executed source')\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_deterministic_package_relocates_its_transitive_closure(tmp_path):
    cache = tmp_path / "cache"
    leaf=module(tmp_path / "leaf","leaf",exports={"kernel":"src/kernel.py"})
    source(leaf / "src/kernel.py")
    middle=module(tmp_path / "middle","middle",dependencies={"leaf":dependency("../leaf")})
    top=module(tmp_path / "top","top",dependencies={"middle":dependency("../middle")})
    one,two=tmp_path / "one.tpack",tmp_path / "two.tpack"
    first=pack(top,one,cache_dir=cache)
    second=pack(top,two,cache_dir=cache)
    assert first['sha256']==second['sha256'] and one.read_bytes()==two.read_bytes()
    shutil.rmtree(leaf);shutil.rmtree(middle);shutil.rmtree(top)
    app=module(tmp_path / "consumer","app")
    add(one,app,cache_dir=cache)
    one.unlink()
    install(app,cache_dir=cache,frozen=True)
    project=Project(app,cache_dir=cache)
    assert project.module("leaf").manifest['exports']['kernel']['source']=='src/kernel.py'
    assert set(json.loads((app / 'tensor.lock').read_text())['packages'])=={'app','top','middle','leaf'}


def test_frozen_lock_rejects_changed_source_without_rewriting(tmp_path):
    cache=tmp_path / "cache"
    lib=module(tmp_path / "lib","lib",exports={"kernel":"kernel.py"})
    source(lib / "kernel.py")
    app=module(tmp_path / "app","app",dependencies={"lib":dependency("../lib")})
    install(app,cache_dir=cache)
    previous=(app / 'tensor.lock').read_bytes()
    source(lib / "kernel.py","raise RuntimeError('changed')\n")
    with pytest.raises(ModuleError,match='stale'):
        install(app,cache_dir=cache,frozen=True)
    assert (app / 'tensor.lock').read_bytes()==previous
    install(app,cache_dir=cache)
    assert (app / 'tensor.lock').read_bytes()!=previous
    source(app / 'extra.py')
    manifest=json.loads((app / 'tensor.json').read_text());manifest['files']=['extra.py']
    (app / 'tensor.json').write_text(json.dumps(manifest))
    with pytest.raises(ModuleError,match='changed'):
        Project(app,cache_dir=cache)


def test_cycles_version_conflicts_and_failed_add_preserve_project(tmp_path):
    cache=tmp_path / 'cache'
    a=module(tmp_path / 'a','a',dependencies={'b':dependency('../b')})
    module(tmp_path / 'b','b',dependencies={'a':dependency('../a')})
    with pytest.raises(ModuleError,match='cycle'):
        install(a,cache_dir=cache)
    left=module(tmp_path / 'left','common',exports={'kernel':'kernel.py'})
    right=module(tmp_path / 'right','common',exports={'kernel':'kernel.py'})
    source(left / 'kernel.py','x=1');source(right / 'kernel.py','x=2')
    parent=module(tmp_path / 'parent','parent',dependencies={'common':dependency('../right')})
    app=module(tmp_path / 'app','app',dependencies={'common':dependency('../left')})
    install(app,cache_dir=cache)
    before=[(app/p).read_bytes() for p in ('tensor.json','tensor.lock')]
    with pytest.raises(ModuleError,match='conflicting'):
        add(parent,app,cache_dir=cache)
    assert [(app/p).read_bytes() for p in ('tensor.json','tensor.lock')]==before
    bad=module(tmp_path / 'bad','bad',dependencies={'common':{'path':'../left','version':'0.2.0'}})
    with pytest.raises(ModuleError,match='version mismatch'):
        install(bad,cache_dir=cache)


def test_corrupt_snapshot_fails_and_install_repairs_it(tmp_path):
    lib=module(tmp_path / 'lib','lib',exports={'kernel':'kernel.py'})
    source(lib/'kernel.py')
    cache=tmp_path / 'cache';install(lib,cache_dir=cache)
    digest=Project(lib,cache_dir=cache).module().digest
    cached=cache/'packages'/digest/'kernel.py';cached.write_text('corrupt')
    with pytest.raises(ModuleError,match='hash mismatch'):
        Project(lib,cache_dir=cache)
    install(lib,cache_dir=cache,frozen=True)
    assert cached.read_bytes()==(lib/'kernel.py').read_bytes()


def test_package_archive_rejects_traversal_duplicates_and_corrupt_payload(tmp_path):
    lib=module(tmp_path/'lib','lib',exports={'kernel':'kernel.py'});source(lib/'kernel.py')
    archive=tmp_path/'valid.tpack';pack(lib,archive)
    app=module(tmp_path/'app','app');before=(app/'tensor.json').read_bytes()
    for name,contents in (('../outside',b'x'),('modules/lib/kernel.py',b'changed')):
        path=tmp_path/(('traversal' if name.startswith('..') else 'changed')+'.tpack')
        with zipfile.ZipFile(archive) as original,zipfile.ZipFile(path,'w') as changed:
            for member in original.namelist():
                changed.writestr(member,contents if member==name else original.read(member))
            if name.startswith('..'):
                changed.writestr(name,contents)
        with pytest.raises(ModuleError,match='path|hash'):
            add(path,app,cache_dir=tmp_path/'cache')
    duplicate=tmp_path/'duplicate.tpack'
    with zipfile.ZipFile(archive) as original,zipfile.ZipFile(duplicate,'w') as changed:
        for member in original.namelist(): changed.writestr(member,original.read(member))
        with pytest.warns(UserWarning,match='Duplicate'):
            changed.writestr('package.json',original.read('package.json'))
    with pytest.raises(ModuleError,match='duplicate'):
        add(duplicate,app,cache_dir=tmp_path/'cache')
    assert (app/'tensor.json').read_bytes()==before


def test_manifest_rejects_external_paths_and_capabilities_before_compilation(tmp_path):
    lib=module(tmp_path/'lib','lib',exports={'kernel':'../outside.py'})
    with pytest.raises(ModuleError,match='inside'):
        install(lib,cache_dir=tmp_path/'cache')
    module(lib,'lib',exports={'kernel':'kernel.py'});source(lib/'kernel.py')
    value=json.loads((lib/'tensor.json').read_text());value['capabilities']=['unknown-feature']
    (lib/'tensor.json').write_text(json.dumps(value));install(lib,cache_dir=tmp_path/'cache')
    with pytest.raises(ModuleError,match='capabilities'):
        Project(lib,cache_dir=tmp_path/'cache').module().resolve('kernel',provider='cpu',compile=True)


@pytest.fixture(scope='module')
def cpu_image(tmp_path_factory):
    import platform
    if platform.system()!='Linux' or platform.machine()!='x86_64' or not shutil.which('c++'):
        pytest.skip('Linux x86-64 CPU producer')
    import tensor as tx
    root=tmp_path_factory.mktemp('module-cpu')
    artifact=root/'dynamic.tbin'
    tx.build(ROOT/'examples/dynamic_affine.py',artifact,provider='cpu',cache_dir=root/'compiler')
    return artifact


def test_exact_binary_then_portable_compile_cache_and_corruption_recovery(cpu_image,tmp_path):
    import numpy as np
    import tensor as tx
    cache=tmp_path/'cache'
    lib=module(tmp_path/'lib','lib',exports={'affine':{'artifacts':['kernel.tbin'],'portable':'kernel.tbin'}})
    shutil.copyfile(cpu_image,lib/'kernel.tbin')
    install(lib,cache_dir=cache)
    mod=Project(lib,cache_dir=cache).module()
    assert mod.resolve('affine',provider='cpu')['selection']=='packaged'
    with tx.Device(provider='cpu') as device:
        kernel=mod.load('affine',device)
        tx.assert_close(kernel(device.arange(129),device.ones((129,)),scale=2.5),2.5*np.arange(129,dtype='float32')+1)
    # A portable carrier is not implicitly treated as a runtime binary.
    value=json.loads((lib/'tensor.json').read_text());value['exports']['affine'].pop('artifacts')
    (lib/'tensor.json').write_text(json.dumps(value));install(lib,cache_dir=cache)
    mod=Project(lib,cache_dir=cache).module()
    with pytest.raises(ModuleError,match='--compile'):
        mod.resolve('affine',provider='cpu')
    resolved=mod.resolve('affine',provider='cpu',compile=True,cache_dir=tmp_path/'compiler')
    assert resolved['selection']=='compiled'
    assert mod.resolve('affine',provider='cpu')['selection']=='cached'
    Path(resolved['path']).write_bytes(b'corrupt')
    with pytest.raises(ModuleError,match='--compile'):
        mod.resolve('affine',provider='cpu')
    assert mod.resolve('affine',provider='cpu',compile=True,cache_dir=tmp_path/'compiler')['selection']=='compiled'
    with tx.Device(provider='cpu') as device:
        tx.assert_close(mod.load('affine',device)(device.arange(1),device.ones((1,)),scale=2.),np.ones(1,dtype='float32'))


def test_portable_preflight_rejects_frontend_and_operator_mismatches(cpu_image,tmp_path):
    from tensor.artifact import read_artifact
    from tensor.build import BuildError
    from tensor.portable import preflight
    manifest,files=read_artifact(cpu_image)
    for field,value in (('tilelang_version','future-frontend'),('op_set',['unknown.operator'])):
        changed={**manifest,field:value};path=tmp_path/(field+'.tbin')
        with zipfile.ZipFile(path,'w') as bundle:
            bundle.writestr('manifest.json',json.dumps(changed))
            for name,contents in files.items():bundle.writestr(name,contents)
        with pytest.raises(BuildError,match='version mismatch|operator set'):
            preflight(path)


def test_source_helpers_are_isolated_and_do_not_mutate_snapshots(tmp_path):
    from tensor.portable import export_spec
    for value in (1,2):
        lib=module(tmp_path/str(value),f'lib{value}',exports={'kernel':'src/kernel.py'},files=['src/helper.py'])
        source(lib/'src/kernel.py','import helper\ndef tensor_export(): return {"value":helper.VALUE}\n')
        source(lib/'src/helper.py',f'VALUE={value}\n')
        cache=tmp_path/'cache';install(lib,cache_dir=cache)
        mod=Project(lib,cache_dir=cache).module()
        path=cache/'packages'/mod.digest/'src/kernel.py'
        assert export_spec(path)=={'value':value}
        assert 'helper' not in sys.modules
        assert mod.manifest['name']==f'lib{value}'


def test_cli_package_install_inspect_and_cache_without_source_execution(tmp_path,capsys):
    from tensor.cli import main
    lib=module(tmp_path/'lib','lib',exports={'kernel':'kernel.py'});source(lib/'kernel.py')
    app=module(tmp_path/'app','app');cache=tmp_path/'cache';archive=tmp_path/'lib.tpack'
    assert main(['pack',str(lib),'--out',str(archive),'--module-cache',str(cache)])==0
    assert json.loads(capsys.readouterr().out)['status']=='packed'
    common=['--project',str(app),'--module-cache',str(cache)]
    assert main(['add',str(archive),*common])==0
    assert json.loads(capsys.readouterr().out)['name']=='lib'
    assert main(['install',*common,'--frozen'])==0
    assert json.loads(capsys.readouterr().out)['frozen'] is True
    assert main(['inspect',str(archive)])==0
    assert json.loads(capsys.readouterr().out)['root']=='lib'
    assert main(['cache','--module-cache',str(cache),'--cache-dir',str(tmp_path/'compiler')])==0
    assert json.loads(capsys.readouterr().out)['modules']['packages']==2
    with pytest.raises(SystemExit) as failure:
        main(['resolve','lib::kernel',*common,'--provider','cpu'])
    assert failure.value.code==1 and '--compile' in capsys.readouterr().err


def test_packaged_binary_build_reference_is_compiler_free(cpu_image,tmp_path,capsys):
    from tensor.cli import main
    cache=tmp_path/'cache';lib=module(tmp_path/'lib','lib',exports={'affine':{'artifacts':['kernel.tbin']}})
    shutil.copyfile(cpu_image,lib/'kernel.tbin');install(lib,cache_dir=cache)
    output=tmp_path/'export.tbin'
    common=['--project',str(lib),'--module-cache',str(cache),'--provider','cpu']
    assert main(['build','lib::affine',*common,'--out',str(output)])==0
    assert json.loads(capsys.readouterr().out)['selection']=='packaged'
    assert output.read_bytes()==cpu_image.read_bytes()
    assert main(['inspect','lib::affine',*common])==0
    assert json.loads(capsys.readouterr().out)['provider']=='cpu'
    assert main(['resolve','lib::affine',*common])==0
    assert json.loads(capsys.readouterr().out)['selection']=='packaged'
    with pytest.raises(SystemExit):main(['build','lib::affine',*common,'--out',str(output)])
    assert output.read_bytes()==cpu_image.read_bytes()


def test_legacy_and_current_artifacts_share_a_logical_module_signature(tmp_path):
    lib=module(tmp_path/'lib','lib',exports={'kernel':{'artifacts':['legacy.tbin','current.tbin']}})
    files={'kernel.cubin':b'\x7fELFmetadata-only','kernel.tirx.json':b'{"nodes":[]}'}
    old={'format':'tensor.cuda','format_version':1,'kind':'cubin','target':'sm_86','entrypoint':'kernel',
        'launch':{'grid':[1,1,1],'block':[32,1,1],'shared_memory_bytes':0},
        'arguments':[{'name':'out','dtype':'float32','shape':[32]}],'outputs':['out'],
        'source_sha256':'0'*64,'tilelang_version':'0.1.14','tvm_ffi_version':'0.1.12','op_set':[],
        'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    new={**old,'format':'tensor.module','format_version':3,'provider':'cuda','target':'sm_80',
        'arguments':[{'kind':'buffer','name':'out','dtype':'float32','shape':[32],'alignment':64}],
        'abi':[{'kind':'buffer','name':'out','dtype':'float32'}],'symbols':{},
        'compiler':{'name':'nvrtc','version':'12.9.86'},'workspace':{'bytes':0,'alignment':1},
        'runtime_abi':{'major':1,'minor':1,'required_capabilities':['contiguous','executable_descriptors','no_external_workspace']}}
    for name,manifest in [('legacy',old),('current',new)]:
        with zipfile.ZipFile(lib/f'{name}.tbin','w') as bundle:
            bundle.writestr('manifest.json',json.dumps(manifest))
            for member,contents in files.items():bundle.writestr(member,contents)
    cache=tmp_path/'cache';install(lib,cache_dir=cache)
    mod=Project(lib,cache_dir=cache).module()
    assert mod.resolve('kernel',target='sm_86')['path'].endswith('legacy.tbin')
    assert mod.resolve('kernel',target='sm_80')['path'].endswith('current.tbin')
