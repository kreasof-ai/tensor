"""Readable native Tensor module templates with explicit backward callbacks."""

from tensor_nn.templates.linear import Linear
from tensor_nn.templates.normalization import LayerNorm
from tensor_nn.templates.activation import GELU
from tensor_nn.templates.attention import CausalSelfAttention
from tensor_nn.templates.embedding import TokenPositionEmbedding
from tensor_nn.templates.loss import CrossEntropy

__all__ = [
    "Linear",
    "LayerNorm",
    "GELU",
    "CausalSelfAttention",
    "TokenPositionEmbedding",
    "CrossEntropy",
]
