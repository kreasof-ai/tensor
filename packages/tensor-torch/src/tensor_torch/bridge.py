"""PyTorch storage/stream ownership around the compiler-independent Tensor ABI."""
from __future__ import annotations

import atexit
from contextlib import nullcontext
from collections import OrderedDict
import ctypes as c
import importlib
import os
import threading
from pathlib import Path

import torch

try:
    from ._launch import submit as _submit
except ImportError:
    _submit = None

_executor = None
if os.environ.get('TENSOR_TORCH_NATIVE', '1') != '0':
    # The C++ executor uses Torch's non-stable ABI: only load a module built for
    # this major/minor. Portable adapter wheels have no such module.
    version = '_'.join(torch.__version__.split('.')[:2])
    try:
        _executor = importlib.import_module('._executor_' + version, __package__)
    except ImportError:
        pass


def _native_plan(prepared, args, outputs, *, fixed):
    if _executor is None:
        return None
    manifest = prepared.executable.manifest
    names = manifest['outputs']
    inputs = [d for d in manifest['arguments'] if d['name'] not in names]
    tensors = tuple(v for d, v in zip(inputs, args) if buffer_argument(d))
    positions = {d['name']: i for i, d in enumerate(d for d in inputs if buffer_argument(d))}
    positions.update((name, len(tensors) + i) for i, name in enumerate(names))
    bindings = []
    for argument, descriptor in zip(prepared.arguments, manifest.get('abi', manifest['arguments'])):
        if buffer_argument(descriptor):
            bindings.append((positions[descriptor['name']], descriptor.get('alignment', 256)))
        else:
            bindings.append(c.string_at(c.addressof(argument) + type(argument).scalar.offset,
                                        c.sizeof(SCALAR_TYPES[descriptor['dtype']])))
    return _executor.Plan(prepared.executable.function.value,
        (*prepared.launch['grid'], *prepared.launch['block'], prepared.launch['shared_memory_bytes']),
        tensors, tuple(outputs), bindings, fixed)


def _enqueue(executable, launch, stream, parameters):
    if _submit is None:
        executable.device.driver.call('cuLaunchKernel', executable.function,
            *launch['grid'], *launch['block'], launch['shared_memory_bytes'],
            c.c_void_p(stream), parameters, None)
    else:
        code = _submit(executable.function.value, *launch['grid'], *launch['block'],
                       launch['shared_memory_bytes'], stream, c.addressof(parameters))
        if code:
            from tensor.providers.cuda import CudaError
            raise CudaError(f'cuLaunchKernel failed with CUDA error {code}')

from tensor.artifacts.format import read_artifact
from tensor.providers.cuda import Device
from tensor.runtime.abi import BoundCall
from tensor.runtime.signature import SCALAR_TYPES, bind_shapes, buffer_argument, resolve_shape, scalar_value

_sessions = {}
_lock = threading.RLock()
_streams = {}


def _current_stream(ordinal):
    # PyTorch exposes this low-overhead handle API to its own compiler backends.
    # Keep a public-API fallback for supported versions that omit the helper.
    getter = getattr(torch._C, '_cuda_getCurrentRawStream', None)
    if getter is None:
        return torch.cuda.current_stream(ordinal)
    handle = getter(ordinal)
    key = (ordinal, handle)
    stream = _streams.get(key)
    if stream is None:
        stream = torch.cuda.current_stream(ordinal)
        _streams[key] = stream
    return stream


def session(ordinal):
    # Retain the primary context once; launch streams are selected for every call.
    with _lock:
        if ordinal not in _sessions:
            _sessions[ordinal] = Device(ordinal).__enter__()
        return _sessions[ordinal]


def close():
    """Synchronize and release adapter-owned modules, never PyTorch allocations."""
    with _lock:
        for device in _sessions.values():
            # Launches may have used any PyTorch stream on this device.
            with torch.cuda.device(device.ordinal):
                torch.cuda.synchronize()
            device.__exit__(None, None, None)
        _sessions.clear()
        _streams.clear()


atexit.register(close)


class _Metadata:
    def __init__(self, value):
        self.shape = tuple(value.shape)
        self.dtype = str(value.dtype).removeprefix('torch.')


class Kernel:
    """A functional artifact with PyTorch-owned outputs and explicit output names.

    Input buffers must be read-only in the exported kernel. Declared output buffers
    must be completely written and must not alias inputs. These are the functional
    custom-operator contract; artifacts do not encode arbitrary alias analysis.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.manifest, _ = read_artifact(self.path)
        if self.manifest.get('provider', 'cuda') != 'cuda':
            raise ValueError('tensor-torch currently supports CUDA artifacts')
        names = self.manifest.get('outputs', [])
        if not names:
            raise ValueError('functional kernels need declared outputs')
        self.inputs = [d for d in self.manifest['arguments'] if d['name'] not in names]
        self.outputs = [next(d for d in self.manifest['arguments'] if d['name'] == n) for n in names]
        self._executables = {}
        self._prepared = OrderedDict()
        self._lock = threading.RLock()
        self._op = None
        self._into_op = None

    def metadata(self, args, *, symbolic=False):
        if len(args) != len(self.inputs):
            raise ValueError(f'expected {len(self.inputs)} inputs, received {len(args)}')
        values = {}
        device = None
        for descriptor, value in zip(self.inputs, args):
            if buffer_argument(descriptor):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{descriptor['name']} must be a PyTorch tensor")
                if value.device.type != 'cuda' or not value.is_contiguous():
                    raise ValueError('Tensor kernels require contiguous CUDA tensors')
                if device is not None and value.device != device:
                    raise ValueError('all inputs must be on the same CUDA device')
                device = value.device
                values[descriptor['name']] = _Metadata(value)
            else:
                values[descriptor['name']] = value
        if device is None:
            raise ValueError('a PyTorch kernel needs at least one tensor input')
        if not symbolic:
            symbols, _ = bind_shapes(self.manifest, values, {})
            shapes = [resolve_shape(d['shape'], symbols) for d in self.outputs]
        else:
            symbols = {}
            for descriptor in self.inputs:
                value = values[descriptor['name']]
                if buffer_argument(descriptor):
                    if len(value.shape) != len(descriptor['shape']) or value.dtype != descriptor['dtype']:
                        raise ValueError('input rank or dtype mismatch')
                    for dim, actual in zip(descriptor['shape'], value.shape):
                        if isinstance(dim, dict) and 'var' in dim:
                            name = dim['var']
                            if name in symbols:
                                torch._check(symbols[name] == actual)
                            symbols[name] = actual
                elif descriptor['name'] in self.manifest.get('symbols', {}):
                    symbols[descriptor['name']] = value
            def evaluate(expr):
                if type(expr) is int:
                    return expr
                if 'var' in expr:
                    return symbols[expr['var']]
                if 'cast' in expr:
                    return evaluate(expr['value'])
                lhs, rhs = map(evaluate, expr['args'])
                import operator
                return {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul,
                        'floordiv': operator.floordiv, 'floormod': operator.mod,
                        'min': torch.sym_min, 'max': torch.sym_max}[expr['op']](lhs, rhs)
            for descriptor in self.inputs:
                if buffer_argument(descriptor):
                    for dim, actual in zip(descriptor['shape'], values[descriptor['name']].shape):
                        torch._check(evaluate(dim) == actual)
            shapes = [tuple(map(evaluate, d['shape'])) for d in self.outputs]
            for shape in shapes:
                for extent in shape:
                    torch._check(extent > 0)
        return [(shape, getattr(torch, d['dtype'])) for shape, d in zip(shapes, self.outputs)], device

    def fake(self, args):
        outputs, device = self.metadata(args, symbolic=True)
        return [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in outputs]

    def _launch(self, args, outputs, *, contract=None, trusted_outputs=False, enqueue=True):
        expected, placement = contract or self.metadata(args)
        if len(outputs) != len(expected):
            raise ValueError('wrong output count')
        tensors = [v for d, v in zip(self.inputs, args) if buffer_argument(d)]
        for output_index, (out, (shape, dtype)) in enumerate(zip(outputs, expected)):
            if (not isinstance(out, torch.Tensor) or tuple(out.shape) != shape or out.dtype != dtype
                    or out.device != placement or not out.is_contiguous()):
                raise ValueError('output shape, dtype, layout or device mismatch')
            if not trusted_outputs and any(out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr() for v in tensors + list(outputs[:output_index])):
                raise ValueError('outputs must not alias inputs or each other')
        ordinal = placement.index
        with self._lock, torch.cuda.device(ordinal):
            device = session(ordinal)
            executable = self._executables.get(ordinal)
            if executable is None or not executable.device._open:
                executable = self._executables[ordinal] = device.load(self.path)
                self._prepared.clear()
            stream = torch.cuda.current_stream(ordinal)
            # record_stream protects allocator reuse after Python references disappear.
            # PyTorch owns producer ordering; current-stream DLPack import performs
            # the first binding handshake and validates pointers with the provider.
            all_values = dict(zip((d['name'] for d in self.inputs), args))
            all_values.update(zip((d['name'] for d in self.outputs), outputs))
            key = (ordinal, tuple((d['name'], tuple(v.shape), v.dtype) if buffer_argument(d)
                                else (d['name'], type(v), v if d['name'] in self.manifest.get('symbols', {}) else None) for d in self.manifest['arguments']
                                for v in [all_values[d['name']]]))
            prepared = self._prepared.get(key)
            if prepared is None:
                borrowed = []
                try:
                    bound = {}
                    for d in self.manifest['arguments']:
                        v = all_values[d['name']]
                        if buffer_argument(d):
                            b = device.from_dlpack(v.detach())
                            borrowed.append(b)
                            bound[d['name']] = b
                        else:
                            bound[d['name']] = v
                    bound, symbols, launch = executable._bind((), bound, include_outputs=True)
                    call = BoundCall(device, self.manifest, bound, symbols, launch, validated=True)
                    parameters = call.cuda_parameters()
                    # Keep descriptor storage, not DLPack owners, in the prepared cache.
                    # Disposed Buffer objects retain ABI shape/stride backing only.
                    prepared = self._prepared[key] = (call, parameters, launch)
                    if len(self._prepared) > 64:
                        self._prepared.popitem(last=False)
                    shared = launch['shared_memory_bytes']
                    if shared > executable._shared_limit:
                        device.driver.call('cuFuncSetAttribute', executable.function, 8, shared)
                        executable._shared_limit = shared
                finally:
                    # Imports queued producer->Tensor-stream waits. Hand back that
                    # ordering before releasing their DLPack owners.
                    device.handoff(stream.cuda_stream)
                    for b in borrowed:
                        b._dispose()
            self._prepared.move_to_end(key)
            call, parameters, launch = prepared
            for argument, d in zip(call.arguments, self.manifest.get('abi', self.manifest['arguments'])):
                if buffer_argument(d):
                    value = all_values[d['name']]
                    pointer = value.data_ptr()
                    if pointer % d.get('alignment', 256):
                        raise ValueError(f"{d['name']} has insufficient pointer alignment")
                    value.record_stream(stream)
                    argument.buffer.address = pointer
                elif d['name'] in all_values:
                    scalar = scalar_value(all_values[d['name']], d['dtype'], d['name'])
                    scalar = SCALAR_TYPES[d['dtype']](scalar)
                    c.memmove(c.addressof(argument) + type(argument).scalar.offset, c.byref(scalar), c.sizeof(scalar))
            # The context is selected by torch.cuda.device. No default-stream launch
            # or stream synchronization occurs on the prepared path.
            if enqueue:
                device.driver.call('cuLaunchKernel', executable.function, *launch['grid'], *launch['block'],
                                   launch['shared_memory_bytes'], c.c_void_p(stream.cuda_stream), parameters, None)
            return executable, call, launch, tuple(tensors) + tuple(outputs), stream

    def raw(self, *args):
        contract = self.metadata(args)
        outputs = [torch.empty(shape, dtype=dtype, device=contract[1]) for shape, dtype in contract[0]]
        self._launch(args, outputs, contract=contract, trusted_outputs=True)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def prepare(self, *args, outputs=None):
        """Bind fixed tensors/scalars once, retaining storage for repeated submission.

        The returned call launches on the current PyTorch stream and never allocates
        outputs. Preparation imports DLPack; call outputs are available as .outputs.
        Changing input data in place is supported; changing tensor metadata is not.
        """
        if outputs is None:
            outputs = self.fake(args)
        state = self._launch(args, list(outputs), enqueue=False)
        return Prepared(state, args, tuple(outputs))

    def register(self):
        if self._op is not None:
            return self._op
        import hashlib
        name = 'kernel_' + hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        # Multiple loads are independent custom ops with the same artifact semantics.
        name += '_' + str(next(_operator_ids))
        def unpack(tensors, scalars):
            ti, si = iter(tensors), iter(scalars)
            return tuple(next(ti) if buffer_argument(d) else next(si) for d in self.inputs)
        @torch.library.custom_op('tensor_torch::' + name, mutates_args=(),
                                 schema='(Tensor[] tensors, Scalar[] scalars) -> Tensor[]')
        def op(tensors, scalars):
            args = unpack(tensors, scalars)
            contract = self.metadata(args)
            outputs = [torch.empty(shape, dtype=dtype, device=contract[1]) for shape, dtype in contract[0]]
            self._launch(args, outputs, contract=contract, trusted_outputs=True)
            return outputs
        @op.register_fake
        def fake(tensors, scalars):
            return self.fake(unpack(tensors, scalars))
        @torch.library.custom_op('tensor_torch::' + name + '_into', mutates_args={'outputs'},
                                 schema='(Tensor[] tensors, Scalar[] scalars, Tensor(a!)[] outputs) -> ()')
        def into(tensors, scalars, outputs):
            self._launch(unpack(tensors, scalars), outputs)
        @into.register_fake
        def into_fake(tensors, scalars, outputs):
            expected = self.fake(unpack(tensors, scalars))
            if len(expected) != len(outputs):
                raise ValueError('wrong output count')
            for a, b in zip(expected, outputs):
                if a.dtype != b.dtype or a.device != b.device or a.ndim != b.ndim or not b.is_contiguous():
                    raise ValueError('output metadata mismatch')
                for lhs, rhs in zip(a.shape, b.shape):
                    torch._check(lhs == rhs)
        self._op, self._into_op = op, into
        return op

    def __call__(self, *args):
        op = self._op or self.register()
        result = op([v for d, v in zip(self.inputs, args) if buffer_argument(d)],
                    [v for d, v in zip(self.inputs, args) if not buffer_argument(d)])
        return result[0] if len(result) == 1 else tuple(result)

    def into(self, *args, outputs):
        """Explicitly mutate disjoint output buffers, with a registered mutation schema."""
        if self._into_op is None:
            self.register()
        self._into_op([v for d, v in zip(self.inputs, args) if buffer_argument(d)],
                      [v for d, v in zip(self.inputs, args) if not buffer_argument(d)], list(outputs))


from itertools import count
_operator_ids = count()


class Prepared:
    def __init__(self, state, args, outputs):
        self.executable, call, self.launch, self.tensors, stream = state
        self._descriptor_storage = call.storage
        self.args, self.outputs = args, outputs
        self.device = self.executable.device
        self.arguments = type(call.arguments).from_buffer_copy(call.arguments)
        from tensor.runtime.abi import Argument, BufferDescriptor
        self.parameters = (c.c_void_p * len(self.arguments))(*(
            c.addressof(arg) + (Argument.buffer.offset + BufferDescriptor.address.offset if arg.kind == 1
                              else Argument.scalar.offset) for arg in self.arguments))
        self.streams = {stream.cuda_stream: stream}
        self.lock = threading.RLock()
        self.metadata = [(tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.data_ptr()) for t in self.tensors]
        self._native = _native_plan(self, args, outputs, fixed=True)

    def __call__(self):
        if not self.device._open or self.executable._released:
            raise RuntimeError('prepared call belongs to a closed session')
        if self._native is not None:
            self._native()
            return
        for tensor, (shape, stride, dtype, device, pointer) in zip(self.tensors, self.metadata):
            if (tensor.data_ptr() != pointer or tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride
                    or tensor.dtype != dtype or tensor.device != device):
                raise ValueError('prepared tensor metadata or storage changed')
        with self.lock, (torch.cuda.device(self.device.ordinal) if torch.cuda.current_device() != self.device.ordinal else nullcontext()):
            stream = _current_stream(self.device.ordinal)
            if stream.cuda_stream not in self.streams:
                for tensor in self.tensors:
                    tensor.record_stream(stream)
                self.streams[stream.cuda_stream] = stream
            _enqueue(self.executable, self.launch, stream.cuda_stream, self.parameters)


class LaunchPlan:
    """Internal fast path, reached only after Region's concrete metadata guard."""
    def __init__(self, kernel, args, fallback):
        self.fallback = fallback
        self.prepared = kernel.prepare(*args)
        self.inputs = kernel.inputs
        self.outputs = kernel.outputs
        self.descriptors = kernel.manifest.get('abi', kernel.manifest['arguments'])
        self.output_specs = [(tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in self.prepared.outputs]
        self.allocate = getattr(torch._C._dynamo.guards, '_empty_strided_cuda', None)
        self.positions = {d['name']:i for i,d in enumerate(self.inputs)}
        self.output_positions = {d['name']:i for i,d in enumerate(self.outputs)}
        self.kernel = kernel
        self._native = _native_plan(self.prepared, args, self.prepared.outputs, fixed=False)
        # Plans rebind all pointers before launch; retain descriptors, not the
        # first invocation's PyTorch input/output allocations.
        self.prepared.args = self.prepared.tensors = self.prepared.outputs = ()
        self.prepared.metadata = []
        # FX plans must not retain the initial tensors through the fixed executor.
        self.prepared._native = None

    def __call__(self, *args):
        prepared = self.prepared
        if not prepared.device._open or prepared.executable._released:
            raise RuntimeError('launch plan belongs to a closed session')
        if self._native is not None:
            output = self._native(*args)
            return self.fallback(*args) if output is None else output
        with prepared.lock, (torch.cuda.device(prepared.device.ordinal) if torch.cuda.current_device() != prepared.device.ordinal else nullcontext()):
            outputs = [self.allocate(shape,stride,dtype) if self.allocate is not None
                       else torch.empty(shape,dtype=dtype,device=device)
                       for shape,stride,dtype,device in self.output_specs]
            stream = _current_stream(prepared.device.ordinal)
            for argument, descriptor in zip(prepared.arguments, self.descriptors):
                name = descriptor['name']
                value = outputs[self.output_positions[name]] if name in self.output_positions else args[self.positions[name]]
                if buffer_argument(descriptor):
                    pointer = value.data_ptr()
                    if pointer % descriptor.get('alignment',256):
                        return self.fallback(*args)
                    value.record_stream(stream)
                    argument.buffer.address = pointer
                else:
                    # FX emitters currently produce only buffer arguments.
                    raise ValueError('scalar launch plans are unsupported')
            _enqueue(prepared.executable, prepared.launch, stream.cuda_stream, prepared.parameters)
        return outputs[0] if len(outputs)==1 else tuple(outputs)
