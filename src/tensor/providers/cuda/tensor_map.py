"""Encode compiler-produced TMA launch descriptors once when binding a call."""

import ctypes as c


def encode(driver, buffer, spec):
    from tensor.runtime.tensor_map import validate_descriptor

    validate_descriptor(spec, buffer.nbytes)
    if buffer.pointer % 16:
        raise ValueError("TMA source address must be aligned to 16 bytes")
    rank = len(spec["global_dims"])
    dims = (c.c_uint64 * rank)(*spec["global_dims"])
    strides = (c.c_uint64 * max(1, rank - 1))(*spec["global_strides"])
    box = (c.c_uint32 * rank)(*spec["box_dims"])
    element = (c.c_uint32 * rank)(*spec["element_strides"])
    storage = c.create_string_buffer(128 + 63)
    address = (c.addressof(storage) + 63) & ~63
    function = driver.lib.cuTensorMapEncodeTiled
    function.argtypes = [
        c.c_void_p,
        c.c_int,
        c.c_uint32,
        c.c_void_p,
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_uint64),
        c.POINTER(c.c_uint32),
        c.POINTER(c.c_uint32),
        c.c_int,
        c.c_int,
        c.c_int,
        c.c_int,
    ]
    function.restype = c.c_int
    driver.call(
        "cuTensorMapEncodeTiled",
        address,
        spec["data_type"],
        rank,
        buffer.pointer,
        dims,
        strides,
        box,
        element,
        spec["interleave"],
        spec["swizzle"],
        spec["l2_promotion"],
        spec["oob_fill"],
    )
    return storage, address
