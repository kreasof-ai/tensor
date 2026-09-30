"""Conservative FX region selection and standalone TileLang source emission."""
from __future__ import annotations

import math
import operator
from pathlib import Path

import torch
from torch.fx import Node

POINTWISE = {
    operator.add: 'add', operator.sub: 'sub', operator.mul: 'mul', operator.truediv: 'div',
    operator.neg: 'neg', torch.add: 'add', torch.sub: 'sub', torch.mul: 'mul',
    torch.div: 'div', torch.relu: 'relu', torch.sigmoid: 'sigmoid', torch.tanh: 'tanh',
    torch.nn.functional.relu: 'relu',
    torch.ops.aten.add.Tensor: 'add', torch.ops.aten.sub.Tensor: 'sub',
    torch.ops.aten.mul.Tensor: 'mul', torch.ops.aten.mul.Scalar: 'mul',
    torch.ops.aten.div.Tensor: 'div', torch.ops.aten.neg.default: 'neg',
    torch.ops.aten.relu.default: 'relu', torch.ops.aten.sigmoid.default: 'sigmoid',
    torch.ops.aten.tanh.default: 'tanh',
}
GEMM = {operator.matmul: False, torch.matmul: False, torch.mm: False,
        torch.nn.functional.linear: True, torch.ops.aten.mm.default: False}
SDPA = {torch.nn.functional.scaled_dot_product_attention}


def kind(node):
    if node.op == 'call_method' and node.target in {'relu', 'sigmoid', 'tanh'}:
        return node.target
    if node.op == 'call_function':
        return POINTWISE.get(node.target) or ('gemm' if node.target in GEMM else
                                             'attention' if node.target in SDPA else None)
    return None


def eligible(node):
    value = node.meta.get('example_value', node.meta.get('val'))
    return (kind(node) is not None and isinstance(value, torch.Tensor)
            and value.device.type == 'cuda' and value.dtype in {torch.float16, torch.float32}
            and value.is_contiguous() and not node.kwargs.get('inplace', False)
            and not node.kwargs.get('out') and not node.kwargs.get('rounding_mode'))


def regions(graph):
    claimed = set()
    result = []
    for root in reversed(list(graph.nodes)):
        if root in claimed or not eligible(root):
            continue
        selected = set()
        bases = []
        def collect(node):
            if node in selected or node in claimed or not eligible(node):
                return
            category = kind(node)
            if category in {'gemm', 'attention'}:
                if bases or (category == 'attention' and node is not root):
                    return
                bases.append(node)
                selected.add(node)
                return
            selected.add(node)
            for predecessor in node.all_input_nodes:
                if len(predecessor.users) == 1:
                    collect(predecessor)
        collect(root)
        ordered = [n for n in graph.nodes if n in selected]
        external = list(dict.fromkeys(n for item in ordered for n in item.all_input_nodes if n not in selected))
        result.append((root, ordered, external))
        claimed.update(selected)
    return list(reversed(result))


def _dtype(tensor):
    return str(tensor.dtype).removeprefix('torch.')


def emit(nodes, external, inputs):
    values = dict(zip(external, inputs))
    root = nodes[-1]
    base = next((n for n in nodes if kind(n) in {'gemm', 'attention'}), None)
    if any(not isinstance(v, torch.Tensor) for v in inputs):
        raise ValueError('symbolic scalar FX inputs currently remain in PyTorch')
    if any(v.device.type != 'cuda' or not v.is_contiguous() or (v.numel() == 0 or v.ndim == 0) for v in inputs):
        raise ValueError('region needs nonempty contiguous CUDA tensors')
    if len({v.device for v in inputs}) != 1 or len({v.dtype for v in inputs}) != 1:
        raise ValueError('region needs one device and one dtype')
    dtype = _dtype(inputs[0])
    if dtype not in {'float16', 'float32'}:
        raise ValueError('region supports float16 and float32')
    names = {n: 'arg' + str(i) for i, n in enumerate(external)}
    arguments = [f'{names[n]}: T.Tensor({tuple(v.shape)!r}, "{dtype}")' for n, v in values.items()]
    if base is not None and kind(base) == 'attention':
        if root is not base or len(nodes) != 1:
            raise ValueError('attention epilogues remain separate regions')
        q, k, v = [values[n] for n in base.args[:3]]
        options = dict(zip(('attn_mask', 'dropout_p', 'is_causal', 'scale', 'enable_gqa'), base.args[3:]))
        options.update(base.kwargs)
        if (dtype != 'float16' or q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
                or q.shape[-1] not in (64, 128) or options.get('attn_mask') is not None
                or options.get('dropout_p', 0) != 0 or options.get('enable_gqa', False)
                or options.get('scale') not in (None, q.shape[-1] ** -0.5)):
            raise ValueError('SDPA profile requires FP16 BHSD self-attention, D64/128, no mask/dropout/GQA')
        import re
        source = (Path(__file__).parent / 'templates' / 'attention.py').read_text()
        for name, value in zip(('BATCH', 'HEADS', 'SEQ_LEN', 'HEAD_DIM', 'IS_CAUSAL'),
                               (*q.shape, bool(options.get('is_causal', False)))):
            source = re.sub(rf'^{name} = .*$', f'{name} = {value!r}', source, flags=re.M)
        # Reorder external operands to q/k/v when FX names appeared differently.
        return source, tuple(external.index(n) for n in base.args[:3]), 'attention'
    if base is None:
        shape = tuple(torch.broadcast_shapes(*(v.shape for v in inputs)))
        if any(tuple(v.shape) not in (shape, (shape[-1],)) and v.numel() != 1 for v in inputs):
            raise ValueError('pointwise broadcast supports full tensors, trailing bias, or singleton')
        size = math.prod(shape)
        index = tuple(f'(index // {math.prod(shape[i+1:])}) % {d}' for i, d in enumerate(shape))
        prefix = ['with T.Kernel(T.ceildiv(SIZE, 256), threads=256) as block:',
                  '    for lane in T.Parallel(256):', '        index = block * 256 + lane',
                  '        if index < SIZE:']
        indent = '            '
        expressions = {}
        for n, v in values.items():
            coordinates = index if tuple(v.shape) == shape else ('index % ' + str(shape[-1]),) if v.numel() > 1 else tuple('0' for _ in v.shape)
            expressions[n] = f'{names[n]}[{", ".join(coordinates)}]'
        out_index = ', '.join(index)
        declarations = f'SIZE = {size}\n'
    else:
        if kind(base) != 'gemm' or dtype != 'float16':
            raise ValueError('matrix multiplication currently needs float16')
        a, b = [values[n] for n in base.args[:2]]
        transpose = GEMM[base.target]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError('matrix multiplication currently needs rank 2')
        m, k = a.shape
        n, kb = (b.shape[0], b.shape[1]) if transpose else (b.shape[1], b.shape[0])
        if k != kb:
            raise ValueError('GEMM reduction dimension mismatch')
        shape = (m, n)
        if any(tuple(v.shape) not in ((m,k),tuple(b.shape),shape,(n,)) for v in inputs):
            raise ValueError('unsupported GEMM epilogue broadcast')
        if any(base.kwargs):
            raise ValueError('GEMM keyword arguments unsupported')
        declarations = f'M = {m}\nN = {n}\nK = {k}\n'
        prefix = ['with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 32), threads=128) as (bx, by):',
                  '    lhs = T.alloc_shared((32, 32), "float16")',
                  f'    rhs = T.alloc_shared({(64,32) if transpose else (32,64)!r}, "float16")',
                  '    accum = T.alloc_fragment((32, 64), "float32")', '    T.clear(accum)',
                  '    for tile in T.Pipelined(T.ceildiv(K, 32), num_stages=3):',
                  f'        T.copy({names[base.args[0]]}[by * 32, tile * 32], lhs)',
                  f'        T.copy({names[base.args[1]]}[{"bx * 64, tile * 32" if transpose else "tile * 32, bx * 64"}], rhs)',
                  f'        T.gemm(lhs, rhs, accum, transpose_B={transpose!r})',
                  '    for row, col in T.Parallel(32, 64):',
                  '        if (by * 32 + row < M) & (bx * 64 + col < N):']
        indent = '            '
        expressions = {base: 'T.cast(accum[row, col], "float16")'}
        for operand, v in values.items():
            if operand in base.args[:2]:
                continue
            coordinates = 'bx * 64 + col' if v.ndim == 1 else 'by * 32 + row, bx * 64 + col'
            expressions[operand] = f'{names[operand]}[{coordinates}]'
        if transpose and len(base.args) > 2 and base.args[2] is not None:
            expressions[base] = f'T.cast(accum[row, col] + {expressions[base.args[2]]}, "float16")'
        out_index = 'by * 32 + row, bx * 64 + col'
    def expression(value):
        if isinstance(value, Node):
            if value not in expressions:
                raise ValueError('unsupported region dependency')
            return expressions[value]
        if type(value) in (int, float) and math.isfinite(value):
            return repr(value)
        raise ValueError('unsupported pointwise operand')
    body = list(prefix)
    for node in nodes:
        if node is base:
            continue
        category = kind(node)
        args = [expression(a) for a in node.args]
        kwargs = node.kwargs
        if set(kwargs) - {'alpha', 'inplace'}:
            raise ValueError('unsupported pointwise keyword arguments')
        if category in {'add', 'sub', 'mul', 'div'}:
            if len(args) != 2:
                raise ValueError('binary operation needs two arguments')
            op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[category]
            alpha = kwargs.get('alpha', 1)
            if type(alpha) not in (float, int) or not math.isfinite(alpha):
                raise ValueError('alpha must be a finite scalar')
            rhs = f'({args[1]} * {alpha!r})' if alpha != 1 else args[1]
            result = f'({args[0]} {op} {rhs})'
        elif category == 'neg':
            result = f'(-{args[0]})'
        elif category == 'relu':
            # Propagate NaN: CUDA fmax alone would incorrectly turn NaN into zero.
            result = f'T.if_then_else(T.isnan({args[0]}), {args[0]}, T.max({args[0]}, 0))'
        elif category == 'sigmoid':
            result = f'(1 / (1 + T.exp(-T.cast({args[0]}, "float32"))))'
        elif category == 'tanh':
            result = f'T.tanh(T.cast({args[0]}, "float32"))'
        else:
            raise ValueError('unsupported epilogue')
        body.append(indent + f'{node.name} = T.cast({result}, "{dtype}")')
        expressions[node] = node.name
    body.append(indent + f'out[{out_index}] = {expressions[root]}')
    arguments.append(f'out: T.Tensor({shape!r}, "{dtype}")')
    source = 'import tilelang.language as T\n' + declarations + '\n@T.prim_func\ndef kernel(' + ',\n           '.join(arguments) + '):\n'
    source += '\n'.join('    ' + line for line in body) + '\n\ndef tensor_export():\n    return {"kernel": kernel, "outputs": ["out"]}\n'
    return source, tuple(range(len(inputs))), 'gemm' if base else 'pointwise'
