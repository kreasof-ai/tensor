"""Read official dense BF16 Qwen checkpoints without a framework runtime."""

from dataclasses import dataclass
import ctypes as ct
import json
import math
from pathlib import Path
import struct

import numpy as np

from ..checkpoint import TensorInfo, _json


@dataclass(frozen=True)
class DenseConfig:
    width: int
    intermediate: int
    vocab: int
    layers: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    epsilon: float
    theta: float
    rotary_dim: int
    context_limit: int

    @classmethod
    def from_json(cls, value):
        text = value["text_config"]
        rope = text["rope_parameters"]
        if (
            text["model_type"] != "qwen3_5_text"
            or text["hidden_act"] != "silu"
            or not text["attn_output_gate"]
            or text["attention_bias"]
            or text["linear_conv_kernel_dim"] != 4
            or not text["tie_word_embeddings"]
            or rope["rope_type"] != "default"
            or not rope["mrope_interleaved"]
        ):
            raise ValueError("unsupported dense Qwen variant")
        result = cls(
            text["hidden_size"],
            text["intermediate_size"],
            text["vocab_size"],
            tuple(text["layer_types"]),
            text["num_attention_heads"],
            text["num_key_value_heads"],
            text["head_dim"],
            text["linear_num_key_heads"],
            text["linear_num_value_heads"],
            text["linear_key_head_dim"],
            text["linear_value_head_dim"],
            text["rms_norm_eps"],
            rope["rope_theta"],
            int(text["head_dim"] * rope["partial_rotary_factor"]),
            text["max_position_embeddings"],
        )
        if (
            len(result.layers) != text["num_hidden_layers"]
            or result.key_dim != 128
            or result.value_dim != 128
            or result.head_dim != 256
            or result.rotary_dim != 64
            or result.heads % result.kv_heads
            or result.value_heads % result.key_heads
        ):
            raise ValueError("unsupported dense Qwen geometry")
        return result

    @property
    def qkv_width(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @property
    def value_width(self):
        return self.value_heads * self.value_dim

    def expected_tensors(self):
        shapes = {
            "model.language_model.embed_tokens.weight": (self.vocab, self.width),
            "model.language_model.norm.weight": (self.width,),
        }
        for layer, kind in enumerate(self.layers):
            root = f"model.language_model.layers.{layer}."
            shapes.update(
                {
                    root + n + ".weight": (self.width,)
                    for n in ("input_layernorm", "post_attention_layernorm")
                }
            )
            for name, shape in (
                ("gate", (self.intermediate, self.width)),
                ("up", (self.intermediate, self.width)),
                ("down", (self.width, self.intermediate)),
            ):
                shapes[root + "mlp." + name + "_proj.weight"] = shape
            if kind == "linear_attention":
                prefix = root + "linear_attn."
                shapes.update(
                    {
                        prefix + "A_log": (self.value_heads,),
                        prefix + "dt_bias": (self.value_heads,),
                        prefix + "conv1d.weight": (self.qkv_width, 1, 4),
                        prefix + "norm.weight": (self.value_dim,),
                    }
                )
                for name, shape in (
                    ("qkv", (self.qkv_width, self.width)),
                    ("z", (self.value_width, self.width)),
                    ("a", (self.value_heads, self.width)),
                    ("b", (self.value_heads, self.width)),
                ):
                    shapes[prefix + "in_proj_" + name + ".weight"] = shape
                shapes[prefix + "out_proj.weight"] = (self.width, self.value_width)
            elif kind == "full_attention":
                prefix = root + "self_attn."
                for name, shape in (
                    ("q", (2 * self.heads * self.head_dim, self.width)),
                    ("k", (self.kv_heads * self.head_dim, self.width)),
                    ("v", (self.kv_heads * self.head_dim, self.width)),
                    ("o", (self.width, self.heads * self.head_dim)),
                ):
                    shapes[prefix + name + "_proj.weight"] = shape
                shapes.update(
                    {prefix + n + "_norm.weight": (self.head_dim,) for n in ("q", "k")}
                )
            else:
                raise ValueError("unsupported dense layer type")
        full = next(i for i, kind in enumerate(self.layers) if kind == "full_attention")
        prefix = f"model.language_model.layers.{full}."
        shapes.update(
            {
                "mtp.layers.0." + name[len(prefix) :]: shape
                for name, shape in tuple(shapes.items())
                if name.startswith(prefix)
            }
        )
        shapes.update(
            {
                "mtp." + n + ".weight": (self.width,)
                for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")
            }
        )
        shapes["mtp.fc.weight"] = (self.width, 2 * self.width)
        return shapes


class DenseCheckpoint:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = DenseConfig.from_json(
            _json((self.directory / "config.json").read_bytes())
        )
        index = _json((self.directory / "model.safetensors.index.json").read_bytes())[
            "weight_map"
        ]
        self.tensors = {}
        selected = {
            name for name in index if name.startswith(("model.language_model.", "mtp."))
        }
        expected = self.config.expected_tensors()
        if selected != expected.keys():
            raise ValueError("dense checkpoint tensor set mismatch")
        for filename in sorted({index[name] for name in selected}):
            path = (self.directory / filename).resolve()
            if path.parent != self.directory:
                raise ValueError("checkpoint shard escapes its directory")
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= length <= min(100_000_000, path.stat().st_size - 8):
                    raise ValueError("invalid safetensors header")
                header = _json(stream.read(length))
            for name in selected:
                if index[name] != filename:
                    continue
                info = header[name]
                size = {"BF16": 2, "F32": 4}.get(info["dtype"])
                begin, end = info["data_offsets"]
                expected_dtype = (
                    "F32"
                    if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
                    else "BF16"
                )
                if (
                    size is None
                    or tuple(info["shape"]) != expected[name]
                    or info["dtype"] != expected_dtype
                    or end - begin != math.prod(info["shape"]) * size
                    or not 0 <= begin <= end <= path.stat().st_size - length - 8
                ):
                    raise ValueError(f"invalid dense checkpoint tensor: {name}")
                self.tensors[name] = TensorInfo(
                    path,
                    info["dtype"],
                    tuple(info["shape"]),
                    8 + length + begin,
                    end - begin,
                )

    def read(self, name):
        info = self.tensors[name]
        with info.shard.open("rb") as stream:
            stream.seek(info.offset)
            raw = stream.read(info.nbytes)
        if len(raw) != info.nbytes:
            raise ValueError("checkpoint changed after validation")
        return np.frombuffer(
            raw, dtype="uint16" if info.dtype == "BF16" else "float32"
        ).reshape(info.shape)

    def upload(self, device, *, branch="text"):
        if branch not in ("text", "mtp"):
            raise ValueError("invalid checkpoint branch")
        weights = {}

        def upload(raw, dtype):
            # BF16 uint16 arrays contain storage bits, not numeric uint16
            # values. from_numpy(dtype='bfloat16') would convert the latter.
            raw = np.ascontiguousarray(raw)
            buffer = device.empty(raw.shape, dtype)
            device.driver.call(
                "cuMemcpyHtoD_v2",
                buffer.pointer,
                ct.c_void_p(raw.ctypes.data),
                raw.nbytes,
            )
            device.driver.call("cuStreamSynchronize", None)
            return buffer

        for name, info in self.tensors.items():
            if name.startswith("mtp.") != (branch == "mtp"):
                continue
            weights[name] = upload(
                self.read(name), "bfloat16" if info.dtype == "BF16" else "float32"
            )
        # Pack adjacent independent projections once. This changes no weights.
        c = self.config
        layers = c.layers if branch == "text" else ("full_attention",)
        for layer, kind in enumerate(layers):
            root = (
                f"model.language_model.layers.{layer}."
                if branch == "text"
                else f"mtp.layers.{layer}."
            )
            if kind == "linear_attention":
                names = [
                    root + "linear_attn.in_proj_" + n + ".weight"
                    for n in ("qkv", "z", "a", "b")
                ]
                packed = root + "linear_attn.input.weight"
            else:
                names = [
                    root + "self_attn." + n + "_proj.weight" for n in ("q", "k", "v")
                ]
                packed = root + "self_attn.input.weight"
            weights[packed] = upload(
                np.concatenate([self.read(n) for n in names]), "bfloat16"
            )
            names += [root + "mlp." + n + "_proj.weight" for n in ("gate", "up")]
            weights[root + "mlp.input.weight"] = upload(
                np.concatenate([self.read(n) for n in names[-2:]]), "bfloat16"
            )
            for name in names:
                weights.pop(name).release()
        return weights
