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


def matmul_space(rows, *, target="sm_90a", partitioned=False):
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
    if partitioned:
        result["matmul_partials"]["splits"] = (2, 4, 8)
    return result


def matmul_legal(config, *, target, depth=None):
    limit = 227328 if target in ("sm_90", "sm_90a") else 99 * 1024
    left, right = (
        (config["block_n"], config["block_m"])
        if config["orientation"] == "weight"
        else (config["block_m"], config["block_n"])
    )
    warps = config["threads"] // 32
    partition = any(
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
    return (config["block_m"] + config["block_n"]) * config["block_k"] * 2 * config[
        "stages"
    ] <= limit


def hopper_space(rows, *, paired=False):
    """Native-adapter schedules: geometry, TMA storage and async queue together."""
    return {
        "hopper_pipeline": {
            "block_m": (8, 16, 32, 64) if rows < 128 else (16, 32, 64, 128),
            "block_n": (32, 64) if paired else (64, 128),
            "block_k": (128, 256, 512),
            "stages": (2, 3, 4),
            "queue_depth": (0, 1, 2),
            "splits": (1, 2, 4, 8),
            "transposed_weights": (False, True),
        }
    }


def hopper_legal(config, *, target, depth, paired=False, outputs=None):
    if target != "sm_90a":
        return False
    bm, bn, bk, stages, queue, splits = (
        config[name]
        for name in ("block_m", "block_n", "block_k", "stages", "queue_depth", "splits")
    )
    if any(
        type(value) is not int or value < 1 for value in (bm, bn, bk, stages, splits)
    ):
        return False
    rows = (2 if paired else 1) * bn
    shared = stages * (rows + bm) * bk * 2 + rows * bm * 4
    clustered = config.get("cluster_reduce", False)
    if config.get("register_epilogue", False) and (clustered or rows != 64):
        return False
    if config.get("coalesced_epilogue", False) and (
        not paired or not config.get("register_epilogue", False)
    ):
        return False
    if config.get("manual_sync", False) and (
        clustered or not config.get("register_epilogue", False)
    ):
        return False
    if clustered:
        if splits not in (2, 4, 8):
            return False
        shared += (splits + 1) * bm * bn * 4
    return (
        type(queue) is int
        and 0 <= queue <= 7
        and stages > queue
        and rows % 64 == 0
        and bm % 8 == 0
        and bk % 16 == 0
        and depth % (bk * splits) == 0
        and shared <= 227328
        and (
            not config["transposed_weights"]
            or outputs is None
            or ((2 if paired else 1) * outputs) % 8 == 0
        )
    )


def hopper_cluster_space(rows, *, paired=False):
    axes = dict(hopper_space(rows, paired=paired)["hopper_pipeline"])
    axes.update(cluster_reduce=(True,), splits=(2, 4, 8))
    return {"hopper_cluster": axes}


def hopper_register_space(rows, *, paired=False):
    axes = dict(hopper_space(rows, paired=paired)["hopper_pipeline"])
    axes.update(
        register_epilogue=(False, True),
        manual_sync=(False, True),
        coalesced_epilogue=(False, True) if paired else (False,),
        block_n=(32 if paired else 64,),
    )
    return {"hopper_register": axes}
