"""Validation of static CUDA tensor-map metadata, without compiler imports."""

# CUtensorMapDataType values with plain element storage.
ELEMENT_BYTES = {
    0: 1,
    1: 2,
    2: 4,
    3: 4,
    4: 8,
    5: 8,
    6: 2,
    7: 4,
    8: 8,
    9: 2,
    10: 4,
    11: 4,
}
FIELDS = {
    "data_type",
    "global_dims",
    "global_strides",
    "box_dims",
    "element_strides",
    "interleave",
    "swizzle",
    "l2_promotion",
    "oob_fill",
}


def validate_descriptor(spec, byte_size=None):
    if not isinstance(spec, dict) or set(spec) != FIELDS:
        raise ValueError("invalid CUDA tensor-map metadata")
    dtype = spec["data_type"]
    if type(dtype) is not int or dtype not in ELEMENT_BYTES:
        raise ValueError("unsupported tensor-map storage type")
    dims, strides, box, elements = (
        spec[n]
        for n in ("global_dims", "global_strides", "box_dims", "element_strides")
    )
    if not all(isinstance(v, list) for v in (dims, strides, box, elements)):
        raise ValueError("tensor-map dimensions must be lists")
    rank = len(dims)
    if not 1 <= rank <= 5 or (len(strides), len(box), len(elements)) != (
        rank - 1,
        rank,
        rank,
    ):
        raise ValueError("invalid tensor-map rank")
    if any(type(v) is not int or not 1 <= v <= 2**32 for v in dims):
        raise ValueError("invalid tensor-map global dimension")
    if any(type(v) is not int or not 1 <= v <= 256 for v in box):
        raise ValueError("invalid tensor-map tile dimension")
    if any(type(v) is not int or not 1 <= v <= 8 for v in elements):
        raise ValueError("invalid tensor-map element stride")
    size = ELEMENT_BYTES[dtype]
    for i, extent in enumerate(dims):
        if i and (type(strides[i - 1]) is not int or strides[i - 1] != size):
            raise ValueError("tensor-map source must use contiguous storage")
        size *= extent
    if byte_size is not None and size != byte_size:
        raise ValueError("tensor-map geometry does not cover its bound buffer")
    for key, choices in (
        ("interleave", (0, 1, 2)),
        ("swizzle", (0, 1, 2, 3)),
        ("l2_promotion", (0, 1, 2, 3)),
        ("oob_fill", (0, 1)),
    ):
        if type(spec[key]) is not int or spec[key] not in choices:
            raise ValueError(f"invalid tensor-map {key}")


def validate_manifest(manifest):
    maps = manifest.get("cuda_tensor_maps", {})
    if not isinstance(maps, dict):
        raise ValueError("invalid CUDA tensor-map table")
    if not maps:
        return
    if (
        manifest.get("provider", "cuda") != "cuda"
        or "cuda_tensor_maps" not in manifest["runtime_abi"]["required_capabilities"]
    ):
        raise ValueError("tensor maps require a CUDA runtime capability")
    frontend = {a["name"]: a for a in manifest["arguments"] if a["kind"] == "buffer"}
    abi = {a["name"]: a for a in manifest["abi"] if a["kind"] == "buffer"}
    for name, spec in maps.items():
        if name not in frontend or name not in abi:
            raise ValueError("unmapped CUDA tensor-map source")
        validate_descriptor(spec)
