"""Extract TileLang's static TMA descriptors into compiler-owned launch metadata."""


def descriptors(kernel, lowered):
    """Map by-value CUDA tensor-map parameters back to frontend buffers."""
    import tvm

    buffers = {str(b.data.name): b for b in kernel.buffer_map.values()}
    result = {}
    seen = set()
    for function in lowered.host_mod.functions.values():
        for parameter, fields in function.attrs.get("tma_descriptor_args", {}).items():
            if str(fields[0].value) != "__tvm_tensormap_create_tiled":
                raise ValueError("unsupported CUDA tensor-map encoding")

            def integer(value):
                if not isinstance(value, tvm.tirx.IntImm):
                    raise ValueError("tensor maps currently require static geometry")
                return int(value.value)

            rank = integer(fields[3])
            if not 1 <= rank <= 5 or len(fields) != 9 + 4 * rank:
                raise ValueError("invalid lowered tensor-map descriptor")
            source = fields[4]
            if not isinstance(source, tvm.tirx.Var) or str(source.name) not in buffers:
                raise ValueError("tensor-map source must be an exported buffer")
            buffer = buffers[str(source.name)]
            name = str(buffer.name)
            if name in seen:
                raise ValueError(
                    "multiple tensor maps for one buffer are not yet supported"
                )
            seen.add(name)
            dims = [integer(v) for v in fields[5 : 5 + rank]]
            strides = [integer(v) for v in fields[5 + rank : 5 + 2 * rank]]
            from tensor.runtime.tensor_map import ELEMENT_BYTES

            if strides[0] != ELEMENT_BYTES.get(integer(fields[2])):
                raise ValueError(
                    "tensor-map innermost stride must match its element width"
                )
            result[str(parameter.name)] = dict(
                buffer=name,
                dtype=str(buffer.dtype),
                descriptor=dict(
                    data_type=integer(fields[2]),
                    global_dims=dims,
                    global_strides=strides[1:],
                    box_dims=[integer(v) for v in fields[5 + 2 * rank : 5 + 3 * rank]],
                    element_strides=[
                        integer(v) for v in fields[5 + 3 * rank : 5 + 4 * rank]
                    ],
                    interleave=integer(fields[-4]),
                    swizzle=integer(fields[-3]),
                    l2_promotion=integer(fields[-2]),
                    oob_fill=integer(fields[-1]),
                ),
            )
    return result
