"""Static geometry and initialization for the nanoGPT example."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GPTConfig:
    layers: int = 12
    heads: int = 12
    width: int = 768
    vocab: int = 50304
    batch: int = 2
    sequence: int = 512
    loss_scale: float = 128.0
    learning_rate: float = 0.0006
    clip: float = 1.0

    def __post_init__(self):
        for name in ("layers", "heads", "width", "vocab", "batch", "sequence"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or self.width // self.heads % 8:
            raise ValueError("head dimensions must be a multiple of eight")
        if self.sequence % 16 or self.width % 16 or self.vocab % 16:
            raise ValueError("training dimensions must be multiples of sixteen")
        for name in ("loss_scale", "learning_rate", "clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def rows(self):
        return self.batch * self.sequence

    @classmethod
    def diagnostic(cls):
        return cls(layers=1, heads=2, width=32, vocab=256, batch=1, sequence=16)


def parameter_shapes(config):
    c = config.width
    result = {"token": (config.vocab, c), "position": (config.sequence, c)}
    for layer in range(config.layers):
        result.update(
            {
                f"{layer}.ln1": (c,),
                f"{layer}.qkv": (3 * c, c),
                f"{layer}.proj": (c, c),
                f"{layer}.ln2": (c,),
                f"{layer}.fc": (4 * c, c),
                f"{layer}.fcproj": (c, 4 * c),
            }
        )
    result["ln_f"] = (c,)
    return result


def initial_weights(config, seed=20260930):
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in parameter_shapes(config).items():
        values = (
            np.ones(shape, np.float32)
            if len(shape) == 1
            else rng.standard_normal(shape).astype(np.float32) * np.float32(0.02)
        )
        if name.endswith((".proj", ".fcproj")):
            values /= np.float32(math.sqrt(2 * config.layers))
        result[name] = values
    return result
