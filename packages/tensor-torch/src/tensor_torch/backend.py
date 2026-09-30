"""Inference-first Dynamo backend with visible per-region specialization/fallback."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule

from tensor.artifact import read_artifact
from .bridge import Kernel, LaunchPlan
from .lowering import emit, regions

_CACHE_SCHEMA = 1
_reports = []


class Region:
    def __init__(self, original, root, nodes, external, owner):
        self.nodes, self.external, self.owner = nodes, external, owner
        graph = Graph()
        mapping = {n: graph.placeholder(n.name) for n in external}
        for node in nodes:
            mapping[node] = graph.node_copy(node, lambda n: mapping[n])
        graph.output(mapping[root])
        self.reference = GraphModule(original, graph)
        self.nodes = [n for n in graph.nodes if n.op not in {"placeholder", "output"}]
        self.external = [n for n in graph.nodes if n.op == "placeholder"]
        self.specializations = {}
        self.lock = threading.RLock()
        self.record = {'nodes': [n.name for n in nodes], 'specializations': [], 'fallbacks': []}
        owner.report['regions'].append(self.record)
        self.__name__ = 'tensor_region_' + root.name
        metadata = [n.meta.get('example_value', n.meta.get('val')) for n in external]
        # Dynamo's tensor guards already protect static shapes, strides, dtypes
        # and devices. Dynamic FX graphs retain our concrete specialization guard.
        self.static = all(isinstance(v, torch.Tensor) and
                          all(not isinstance(d, torch.SymInt) for d in v.shape) for v in metadata)
        self.static_plan = False

    def __call__(self, *args):
        if self.static_plan is not False:
            if self.static_plan is None:
                return self.reference(*args)
            if any(v.data_ptr() % 64 for v in args):
                reason = 'FX inputs need 64-byte pointer alignment'
                if reason not in self.record['fallbacks']:
                    self.record['fallbacks'].append(reason)
                return self.reference(*args)
            kernel, order = self.static_plan
            return kernel(*(args[i] for i in order))
        key = tuple((tuple(v.shape), v.dtype, v.device, tuple(v.stride()), v.data_ptr() % 64)
                    if isinstance(v, torch.Tensor) else (type(v).__name__, str(v)) for v in args)
        specialized = self.specializations.get(key, False)
        if specialized is not False:
            if specialized is None:
                return self.reference(*args)
            kernel, order = specialized
            return kernel(*(args[i] for i in order))
        with self.lock:
            if key not in self.specializations:
                if len(self.specializations) >= self.owner.max_specializations:
                    reason = 'region specialization limit exceeded'
                    if reason not in self.record['fallbacks']:
                        self.record['fallbacks'].append(reason)
                    return self.reference(*args)
                started = time.perf_counter()
                try:
                    source, order, category = emit(self.nodes, self.external, args)
                    if any(v.data_ptr() % 64 for v in args):
                        raise ValueError('FX inputs need 64-byte pointer alignment')
                except ValueError as error:
                    self.record['fallbacks'].append(str(error))
                    self.specializations[key] = None
                else:
                    capability = torch.cuda.get_device_capability(args[0].device)
                    target = 'sm_' + ''.join(map(str, capability))
                    identity = {'schema': _CACHE_SCHEMA, 'source': source, 'target': target,
                                'frontend': 'tilelang==0.1.14', 'runtime_abi': '1.1', 'compiler': 'nvrtc'}
                    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    path = self.owner.cache_dir / (digest + '.tbin')
                    hit = False
                    if path.is_file():
                        try:
                            manifest, _ = read_artifact(path)
                            hit = (manifest['target'] == target
                                   and manifest['source_sha256'] == hashlib.sha256(source.encode()).hexdigest())
                        except ValueError:
                            pass
                    if not hit:
                        from tensor import build
                        self.owner.cache_dir.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=self.owner.cache_dir, prefix='.region-') as temp:
                            source_path = Path(temp) / 'kernel.py'
                            source_path.write_bytes(source.encode())
                            output = Path(temp) / 'kernel.tbin'
                            build(source_path, output, target=target, compiler='nvrtc',
                                  nvrtc_home=self.owner.nvrtc_home,
                                  cache_dir=self.owner.cache_dir / 'compiler')
                            os.replace(output, path)
                    kernel = Kernel(path)
                    # Load/validate before publishing the specialization. First call
                    # below binds concrete tensor metadata and records stream usage.
                    self.specializations[key] = (LaunchPlan(kernel, tuple(args[i] for i in order)), order)
                    self.record['specializations'].append({'kind': category, 'target': target,
                        'artifact': str(path), 'cache_hit': hit,
                        'prepare_seconds': time.perf_counter() - started,
                        'inputs': [list(v.shape) for v in args]})
            specialized = self.specializations[key]
            if self.static:
                self.static_plan = specialized
        if specialized is None:
            return self.reference(*args)
        kernel, order = specialized
        return kernel(*(args[i] for i in order))


class Backend:
    """Pass an instance to torch.compile to inspect its JSON-compatible report.

    Unsupported nodes execute through their original FX targets. NVRTC compiler
    failures propagate; they are never reported as successfully compiled regions.
    training=True evaluates AOTAutograd, with unsupported forward/backward nodes
    still in PyTorch. It does not claim full compiled-training coverage.
    """
    def __init__(self, *, cache_dir=None, nvrtc_home=None, training=False, max_specializations=64):
        self.cache_dir = Path(cache_dir or os.environ.get('TENSOR_TORCH_CACHE_DIR')
                              or Path.home() / '.cache' / 'tensor' / 'torch')
        self.nvrtc_home = nvrtc_home
        self.training = training
        if type(max_specializations) is not int or max_specializations < 1:
            raise ValueError('max_specializations must be a positive integer')
        self.max_specializations = max_specializations
        self.report = {'regions': [], 'fallback_nodes': [], 'autograd': [], 'graphs': 0}

    def _compile(self, gm, example_inputs):
        self.report['graphs'] += 1
        original = copy.deepcopy(gm)
        selected = regions(original.graph)
        claimed = {n.name for _, nodes, _ in selected for n in nodes}
        self.report['fallback_nodes'].extend(str(n.target) for n in original.graph.nodes
                                            if n.op.startswith('call_') and n.name not in claimed)
        for root, nodes, external in reversed(selected):
            region = Region(original, root, nodes, external, self)
            with original.graph.inserting_before(root):
                replacement = original.graph.call_function(region, tuple(external))
            replacement.meta = root.meta.copy()
            root.replace_all_uses_with(replacement)
            for node in reversed(nodes):
                original.graph.erase_node(node)
        original.graph.lint()
        original.recompile()
        return original.forward

    def __call__(self, gm, example_inputs, **kwargs):
        if kwargs:
            raise ValueError(f'unsupported backend settings: {sorted(kwargs)}; configure a Backend instance')
        if self.training:
            from torch._dynamo.backends.common import aot_autograd
            from functorch.compile import make_boxed_func
            def compiler(stage):
                def compile_graph(graph, inputs):
                    self.report['autograd'].append({'stage': stage,
                        'operators': [str(n.target) for n in graph.graph.nodes if n.op.startswith('call_')]})
                    return make_boxed_func(self._compile(graph, inputs))
                return compile_graph
            return aot_autograd(fw_compiler=compiler('forward'), bw_compiler=compiler('backward'))(gm, example_inputs)
        if torch.is_grad_enabled() and any(isinstance(v, torch.Tensor) and v.requires_grad for v in example_inputs):
            self.report['autograd'].append({'stage': 'fallback', 'reason': 'inference backend called with gradients enabled'})
            self.report['fallback_nodes'].extend(str(n.target) for n in gm.graph.nodes if n.op.startswith('call_'))
            return gm.forward
        return self._compile(gm, example_inputs)


def backend(gm, example_inputs, *, options=None, **kwargs):
    """torch_dynamo_backends entry point; options configure a fresh compiler."""
    compiler = Backend(**(options or {}))
    _reports.append(compiler.report)
    return compiler(gm, example_inputs, **kwargs)


def reports():
    return copy.deepcopy(_reports)
