"""Native Qwen3.5-35B-A3B FP8 checkpoint and resident state contracts.

This module reads safetensors directly. Loading one mapped shard at a time
avoids a second model-sized allocation in host RAM. It does not implement a
forward pass; execution must qualify these contracts before serving requests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import ctypes as ct
import json
import math
import mmap
from pathlib import Path
import struct
from types import MappingProxyType

import numpy as np


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _json(value):
    return json.loads(value, object_pairs_hook=_object)


@dataclass(frozen=True)
class Qwen35Config:
    layers: tuple[str, ...]
    context_limit: int
    epsilon: float
    rope_theta: float
    width: int = 2048
    vocab: int = 248320
    heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    key_heads: int = 16
    value_heads: int = 32
    key_dim: int = 128
    value_dim: int = 128
    conv_width: int = 4
    experts: int = 256
    top_k: int = 8
    expert_width: int = 512

    @classmethod
    def from_json(cls, value):
        text = value['text_config']
        required = dict(model_type='qwen3_5_moe_text', hidden_size=2048,
            vocab_size=248320, num_hidden_layers=40, num_attention_heads=16,
            num_key_value_heads=2, head_dim=256, linear_num_key_heads=16,
            linear_num_value_heads=32, linear_key_head_dim=128,
            linear_value_head_dim=128, linear_conv_kernel_dim=4,
            num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
            shared_expert_intermediate_size=512, hidden_act='silu',
            attn_output_gate=True, attention_bias=False)
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"unsupported Qwen35 configuration: {name}")
        layers = tuple(text['layer_types'])
        if layers != (('linear_attention',) * 3 + ('full_attention',)) * 10:
            raise ValueError('unsupported Qwen35 layer order')
        quant = value['quantization_config']
        if (quant.get('quant_method') != 'fp8' or quant.get('activation_scheme') != 'dynamic'
                or quant.get('weight_block_size') != [128, 128]):
            raise ValueError('Qwen35 requires dynamic block-128 native FP8 weights')
        rope = text['rope_parameters']
        if (rope.get('rope_type') != 'default' or rope.get('partial_rotary_factor') != .25
                or rope.get('mrope_section') != [11, 11, 10]
                or not rope.get('mrope_interleaved')):
            raise ValueError('unsupported Qwen35 rotary contract')
        epsilon, theta = float(text['rms_norm_eps']), float(rope['rope_theta'])
        limit = text['max_position_embeddings']
        if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(theta)
                or theta <= 0 or type(limit) is not int or limit < 48000):
            raise ValueError('invalid Qwen35 numeric configuration')
        if text.get('mlp_only_layers') or text.get('layer_scale', False):
            raise ValueError('unsupported Qwen35 decoder variant')
        return cls(layers, limit, epsilon, theta)

    def state_bytes(self, *, slots, context, block=128):
        """Logical BF16 KV and FP32 recurrent/conv storage; no workspaces.

        Context rounds up to the allocator's page size. Each request retains
        its own states, including during prefill and until completion.
        """
        if any(type(v) is not int or v <= 0 for v in (slots, context, block)):
            raise ValueError('state sizes must be positive integers')
        if context > self.context_limit:
            raise ValueError('context exceeds the checkpoint limit')
        capacity = (context + block - 1) // block * block
        full, linear = self.layers.count('full_attention'), self.layers.count('linear_attention')
        return dict(slots=slots, context=context, capacity=capacity,
            kv=full * slots * capacity * self.kv_heads * self.head_dim * 2 * 2,
            recurrent=linear * slots * self.value_heads * self.key_dim * self.value_dim * 4,
            convolution=linear * slots * (2 * self.key_heads * self.key_dim
                + self.value_heads * self.value_dim) * (self.conv_width - 1) * 4)

    def expected_tensors(self):
        shapes = {'model.language_model.embed_tokens.weight': (self.vocab, self.width),
                  'model.language_model.norm.weight': (self.width,),
                  'lm_head.weight': (self.vocab, self.width)}
        def vector(name, shape):
            shapes[name] = shape
        def fp8(name, rows, columns):
            shapes[name + '.weight'] = (rows, columns)
            shapes[name + '.weight_scale_inv'] = ((rows + 127) // 128, (columns + 127) // 128)
        for layer, kind in enumerate(self.layers):
            root = f'model.language_model.layers.{layer}.'
            for name in ('input_layernorm', 'post_attention_layernorm'):
                vector(root + name + '.weight', (self.width,))
            if kind == 'linear_attention':
                prefix = root + 'linear_attn.'
                vector(prefix + 'A_log', (self.value_heads,))
                vector(prefix + 'dt_bias', (self.value_heads,))
                vector(prefix + 'conv1d.weight', (8192, 1, self.conv_width))
                vector(prefix + 'norm.weight', (self.value_dim,))
                for name in ('a', 'b'):
                    vector(prefix + 'in_proj_' + name + '.weight', (self.value_heads, self.width))
                for name, rows, columns in (('in_proj_qkv', 8192, self.width),
                        ('in_proj_z', 4096, self.width), ('out_proj', self.width, 4096)):
                    fp8(prefix + name, rows, columns)
            else:
                prefix = root + 'self_attn.'
                for name, rows, columns in (('q_proj', 8192, self.width),
                        ('k_proj', 512, self.width), ('v_proj', 512, self.width),
                        ('o_proj', self.width, 4096)):
                    fp8(prefix + name, rows, columns)
                for name in ('q_norm', 'k_norm'):
                    vector(prefix + name + '.weight', (self.head_dim,))
            prefix = root + 'mlp.'
            vector(prefix + 'gate.weight', (self.experts, self.width))
            vector(prefix + 'shared_expert_gate.weight', (1, self.width))
            for expert in ('shared_expert', *(f'experts.{i}' for i in range(self.experts))):
                for name, rows, columns in (('gate_proj', self.expert_width, self.width),
                        ('up_proj', self.expert_width, self.width),
                        ('down_proj', self.width, self.expert_width)):
                    fp8(prefix + expert + '.' + name, rows, columns)
        return shapes

    def expected_mtp_tensors(self):
        """The official one-layer draft branch; embedding/head are shared."""
        prefix = 'model.language_model.layers.3.'
        shapes = {'mtp.layers.0.' + name[len(prefix):]: shape
                  for name, shape in self.expected_tensors().items()
                  if name.startswith(prefix)}
        shapes.update({'mtp.' + name + '.weight': (self.width,)
                       for name in ('norm', 'pre_fc_norm_embedding', 'pre_fc_norm_hidden')})
        shapes['mtp.fc.weight'] = (self.width, 2 * self.width)
        return shapes


@dataclass(frozen=True)
class TensorInfo:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class Qwen35Checkpoint:
    """Validate the complete text checkpoint before allocating GPU weights."""
    storage = {'F8_E4M3': ('uint8', 1), 'BF16': ('uint16', 2), 'F32': ('float32', 4)}

    def __init__(self, directory, *, branch='text'):
        self.directory = Path(directory).resolve()
        value = _json((self.directory / 'config.json').read_bytes())
        self.config = Qwen35Config.from_json(value)
        if branch not in ('text', 'mtp'):
            raise ValueError('checkpoint branch must be text or mtp')
        self.branch = branch
        if branch == 'mtp' and (value['text_config'].get('mtp_num_hidden_layers') != 1
                or value['text_config'].get('mtp_use_dedicated_embeddings') is not False):
            raise ValueError('MTP requires one decoder layer and shared embeddings')
        index = _json((self.directory / 'model.safetensors.index.json').read_bytes())['weight_map']
        selected = (self.config.expected_tensors() if branch == 'text'
                    else self.config.expected_mtp_tensors())
        actual = ({n for n in index if n.startswith('model.language_model.') or n == 'lm_head.weight'}
                  if branch == 'text' else {n for n in index if n.startswith('mtp.')})
        if actual != selected.keys():
            raise ValueError(f'Qwen35 {branch} tensor set mismatch: missing={len(selected.keys() - actual)}, '
                             f'unexpected={len(actual - selected.keys())}')
        tensors = {}
        for filename in sorted({index[n] for n in selected}):
            shard = (self.directory / filename).resolve()
            if shard.parent != self.directory or not shard.is_file():
                raise ValueError('checkpoint shard must be a local file within its directory')
            size = shard.stat().st_size
            with shard.open('rb') as stream:
                encoded = stream.read(8)
                if len(encoded) != 8:
                    raise ValueError('truncated safetensors header')
                header_size = struct.unpack('<Q', encoded)[0]
                if not 2 <= header_size <= min(size - 8, 100_000_000):
                    raise ValueError('invalid safetensors header length')
                header = _json(stream.read(header_size))
            intervals = []
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                shape, offsets = info.get('shape'), info.get('data_offsets')
                if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(x) is not int for x in offsets)
                        or not 0 <= offsets[0] <= offsets[1] <= size - header_size - 8):
                    raise ValueError(f'invalid safetensors extent: {name}')
                intervals.append(tuple(offsets))
                if name not in selected:
                    continue
                dtype = info.get('dtype')
                if (name in tensors or index[name] != filename or dtype not in self.storage
                        or tuple(shape) != selected[name]
                        or offsets[1] - offsets[0] != math.prod(shape) * self.storage[dtype][1]):
                    raise ValueError(f'invalid Qwen35 tensor layout: {name}')
                expected_dtype = ('F8_E4M3' if name.endswith('.weight')
                    and name + '_scale_inv' in selected else 'BF16')
                if name.endswith('.linear_attn.A_log') or name.endswith('.linear_attn.norm.weight'):
                    expected_dtype = 'F32'
                if dtype != expected_dtype:
                    raise ValueError(f'unsupported Qwen35 tensor dtype: {name}: {dtype}')
                tensors[name] = TensorInfo(shard, dtype, tuple(shape),
                    8 + header_size + offsets[0], offsets[1] - offsets[0])
            ordered = sorted(intervals)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError('overlapping safetensors storage')
        if tensors.keys() != selected.keys():
            raise ValueError('index references missing safetensors tensors')
        self.tensors = MappingProxyType(tensors)

    @property
    def weight_bytes(self):
        return sum(info.nbytes for info in self.tensors.values())

    def inventory(self):
        counts, sizes = Counter(), Counter()
        for info in self.tensors.values():
            counts[info.dtype] += 1
            sizes[info.dtype] += info.nbytes
        return dict(weight_bytes=self.weight_bytes, tensors=dict(counts), bytes_by_dtype=dict(sizes))

    def read(self, name):
        """Copy a single raw tensor for producer/reference use, preserving bits."""
        info = self.tensors[name]
        with info.shard.open('rb') as stream:
            stream.seek(info.offset)
            raw = stream.read(info.nbytes)
        if len(raw) != info.nbytes:
            raise ValueError('checkpoint changed after validation')
        return np.frombuffer(raw, dtype=self.storage[info.dtype][0]).reshape(info.shape)

    def upload(self, device, *, names=None, pack_experts=False):
        """Upload native bytes; close all allocations if any transfer fails.

        FP8 is stored as uint8 with an explicit E4M3FN kernel decoder; BF16
        scales remain BF16. A synchronous default-stream barrier completes each
        transfer before unmapping its host shard or launching on another stream.
        The returned buffers are owned by the caller.
        """
        wanted = list(self.tensors) if names is None else list(names)
        if len(set(wanted)) != len(wanted) or any(n not in self.tensors for n in wanted):
            raise ValueError('unknown or duplicate requested checkpoint tensor')
        layouts = {}
        for name in wanted:
            info = self.tensors[name]
            if pack_experts and '.mlp.experts.' in name:
                prefix, suffix = name.split('.mlp.experts.', 1)
                expert, tail = suffix.split('.', 1)
                group = prefix + '.mlp.experts.' + tail
                layouts[name] = (group, int(expert) * info.nbytes,
                                 (self.config.experts, *info.shape))
            else:
                layouts[name] = (name, 0, info.shape)
        if pack_experts:
            counts = Counter(v[0] for v in layouts.values())
            for group, count in counts.items():
                if '.mlp.experts.' in group and count != self.config.experts:
                    raise ValueError('packed loading requires every expert in each selected projection')
        buffers = {}
        try:
            for name in wanted:
                group, _, shape = layouts[name]
                if group not in buffers:
                    info = self.tensors[name]
                    dtype = 'bfloat16' if info.dtype == 'BF16' else self.storage[info.dtype][0]
                    buffers[group] = device.empty(shape, dtype)
            for shard in sorted({self.tensors[n].shard for n in wanted}):
                with shard.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    for name in wanted:
                        info = self.tensors[name]
                        if info.shard != shard:
                            continue
                        group, offset, _ = layouts[name]
                        buffer = buffers[group]
                        view = np.frombuffer(mapped, dtype='uint8', count=info.nbytes, offset=info.offset)
                        try:
                            device._check()
                            device.driver.call('cuMemcpyHtoD_v2', buffer.pointer + offset,
                                ct.c_void_p(view.ctypes.data), info.nbytes)
                            device.driver.call('cuStreamSynchronize', None)
                        finally:
                            del view
            return MappingProxyType(buffers)
        except BaseException:
            for buffer in buffers.values():
                buffer.release()
            raise
