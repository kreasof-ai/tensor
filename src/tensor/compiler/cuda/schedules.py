"""CUDA schedule spaces supplied to device-independent discovery."""

PREFILL_SPACE = {
    "block_m": (16, 32, 64),
    "block_n": (64, 128),
    "block_k": (32, 64, 128),
    "stages": (2, 3),
    "threads": (128, 256),
    "packed_pairs": (False, True),
}
DECODE_SPACE = {"threads": (64, 128, 256), "unroll": (1, 2, 4, 8)}
ATTENTION_SPACES = {
    "partial": {"splits": (16,)},
    "grouped": {"stage": (16, 32, 64), "warps": (2, 4)},
}


def projection_space(phase, *, f16_vectors=False, family="tiles"):
    if phase == "decode":
        return {
            "gemv": {
                **DECODE_SPACE,
                **({"f16_values": (4, 8, 16)} if f16_vectors else {}),
            }
        }
    if phase != "prefill":
        raise ValueError("requires prefill or decode schedule space")
    space = dict(PREFILL_SPACE)
    if family == "paired-loads":
        space["packed_pairs"] = (True,)
    if family == "small-rows":
        space.update(block_m=(16, 32), block_n=(64,), stages=(2,), threads=(128,))
    return {"mma": space}


def projection_legal(
    config, *, depth, paired=False, packed_pairs=True, shared_limit=99 * 1024
):
    if config["family"] == "gemv":
        width = config.get("f16_values", 4)
        return depth % 256 == 0 and depth % (32 * width) == 0
    bk, bm, bn, stages = (
        config[name] for name in ("block_k", "block_m", "block_n", "stages")
    )
    # SM86 has a 99 KiB opt-in shared-memory limit per CTA. Padding/layout
    # details remain the compiler's responsibility and may reject a candidate.
    shared = (bm + (2 if paired else 1) * bn) * bk * 2 * stages
    return (
        depth % bk == 0
        and shared <= shared_limit
        and (not config["packed_pairs"] or packed_pairs)
    )


def matmul_space(rows, *, target="sm_90a", partitioned=False, dtype="float16"):
    """Full-operation geometry, including which operand owns the MMA M axis."""
    result = {
        "matmul_partials" if partitioned else "matmul": {
            "block_m": (128, 256)
            if rows >= 256
            else (64, 128)
            if rows >= 128
            else (16, 32),
            "block_n": (16, 32, 64, 128),
            "block_k": (32, 64, 128),
            "stages": (2, 3, 4),
            "threads": (128, 256, 512),
            "orientation": ("weight", "activation"),
        }
    }
    if dtype == "float32":
        result = {"matmul_partials" if partitioned else "matmul": {
            "block_m": (16, 8, 32, 64), "block_n": (32, 8, 16, 64, 128),
            "block_k": (32, 16, 64), "stages": (2, 1, 3),
            "threads": (128, 256), "orientation": ("activation", "weight"),
            "reduction_lanes": (1, 4, 8, 16, 32),
        }}
    if partitioned:
        result["matmul_partials"]["splits"] = (2, 4, 8)
    return result


def matmul_shared(config, *, dtype="float16"):
    """Estimate the selected implementation's staged shared storage."""
    bm, bn, bk, stages = (config[k] for k in ("block_m", "block_n", "block_k", "stages"))
    if dtype == "float32":
        right = bm if config["orientation"] == "weight" else bn
        return ((bm + bn) * bk + right) * 4 * stages
    return (bm + bn) * bk * 2 * stages


def matmul_legal(config, *, target, depth=None, dtype="float16"):
    limit = 227328 if target in ("sm_90", "sm_90a") else 99 * 1024
    left, right = (
        (config["block_n"], config["block_m"])
        if config["orientation"] == "weight"
        else (config["block_m"], config["block_n"])
    )
    warps = config["threads"] // 32
    partition = (config["threads"] in (128, 256, 512)
                 and all(v > 0 and v & (v - 1) == 0 for v in (left, right, config["block_k"]))
                 and config.get("reduction_lanes", 1) in (1, 2, 4, 8, 16, 32)
                 and config["block_k"] % config.get("reduction_lanes", 1) == 0
                 and left * right * config.get("reduction_lanes", 1) >= config["threads"]
                 and left * right * config.get("reduction_lanes", 1) % config["threads"] == 0) if dtype == "float32" else any(
        warps % rows == 0
        and left % (16 * rows) == 0
        and right % (8 * (warps // rows)) == 0
        for rows in range(1, warps + 1)
    )
    if not partition:
        return False
    if "splits" in config:
        if depth is None:
            raise ValueError("partitioned matmul legality requires the reduction depth")
        if config["splits"] < 2 or depth % (config["splits"] * config["block_k"]):
            return False
    return matmul_shared(config, dtype=dtype) <= limit
