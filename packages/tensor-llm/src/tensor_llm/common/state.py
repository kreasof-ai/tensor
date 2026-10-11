"""Persistent hybrid state shared by dense, MoE, target, and draft adapters."""

import numpy as np


def allocate_hybrid_states(
    device, config, capacity, context, *, kv_dtype="bfloat16", zero_recurrent=False
):
    """Allocate GDN history and attention KV; model execution owns slot resets."""
    if kv_dtype not in ("bfloat16", "fp8"):
        raise ValueError("unsupported hybrid KV dtype")
    states = {}
    try:
        for layer, kind in enumerate(config.layers):
            values = states[layer] = []
            if kind == "linear_attention":
                width = (
                    2 * config.key_heads * config.key_dim
                    + config.value_heads * config.value_dim
                )
                shapes = (
                    (capacity, config.value_heads, config.value_dim, config.key_dim),
                    (capacity, width, getattr(config, "conv_width", 4) - 1),
                )
                for shape in shapes:
                    values.append(
                        device.from_numpy(np.zeros(shape, np.float32))
                        if zero_recurrent
                        else device.empty(shape, "float32")
                    )
            elif kind == "full_attention":
                shape = (capacity, config.kv_heads, context, config.head_dim)
                for _ in range(2):
                    values.append(
                        device.empty(
                            shape, "uint8" if kv_dtype == "fp8" else "bfloat16"
                        )
                    )
                if kv_dtype == "fp8":
                    scales = shape[:-1] + ((config.head_dim + 127) // 128,)
                    for _ in range(2):
                        values.append(device.empty(scales, "float32"))
            else:
                raise ValueError(f"unsupported hybrid layer: {kind}")
        return states
    except BaseException:
        for values in states.values():
            for buffer in values:
                buffer.release()
        raise
