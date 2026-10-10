"""GPU control flow for a complete greedy speculative CUDA graph."""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, window = p["slots"], p["window"]
    if kind == "spec_setup":

        @T.prim_func
        def kernel(
            control: T.Tensor((slots, 5), "int32"),
            mapping: T.Tensor((slots,), "int32"),
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            tokens: T.Tensor((slots * window,), "int32"),
            head_position: T.Tensor((slots * window,), "int32"),
            head_active: T.Tensor((slots * window,), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        mapping[row] = control[row, 0]
                        position[row] = control[row, 1]
                        lengths[row] = control[row, 2]
                        for t in T.serial(window):
                            tokens[row * window + t] = T.if_then_else(
                                t == 0, control[row, 3], control[row, 4]
                            )
                            head_position[row * window + t] = 0
                            head_active[row * window + t] = T.cast(
                                t < control[row, 2], "int32"
                            )

        return kernel

    if kind == "spec_draft":
        depth = p["depth"]

        @T.prim_func
        def kernel(
            control: T.Tensor((slots, 5), "int32"),
            proposed: T.Tensor((slots * window,), "int32"),
            mapping: T.Tensor((slots,), "int32"),
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            tokens: T.Tensor((slots,), "int32"),
            head_position: T.Tensor((slots,), "int32"),
            head_active: T.Tensor((slots,), "int32"),
            mode: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        mapping[row] = control[row, 0]
                        position[row] = control[row, 1] + depth - 1
                        lengths[row] = T.cast(control[row, 2] > depth + 1, "int32")
                        tokens[row] = proposed[row * window + depth]
                        head_position[row] = 0
                        head_active[row] = lengths[row]
                        if row == 0:
                            mode[0] = 1

        return kernel

    if kind == "spec_proposal":
        depth = p["depth"]

        @T.prim_func
        def kernel(
            predicted: T.Tensor((slots,), "int32"),
            proposed: T.Tensor((slots * window,), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        proposed[row * window + depth + 1] = predicted[row]

        return kernel

    if kind == "spec_accept":

        @T.prim_func
        def kernel(
            control: T.Tensor((slots, 5), "int32"),
            proposed: T.Tensor((slots * window,), "int32"),
            predicted: T.Tensor((slots * window,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            result: T.Tensor((slots, window + 2), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        accepted = T.alloc_var("int32")
                        accepted = T.cast(control[row, 2] > 0, "int32")
                        for t in T.serial(window - 1):
                            if (
                                (accepted == t + 1)
                                & (t + 1 < control[row, 2])
                                & (
                                    predicted[row * window + t]
                                    == proposed[row * window + t + 1]
                                )
                            ):
                                accepted += 1
                        lengths[row] = accepted
                        result[row, 0] = accepted
                        for t in T.serial(window):
                            result[row, t + 1] = predicted[row * window + t]

        return kernel

    if kind == "spec_repair":
        width = p["width"]

        @T.prim_func
        def kernel(
            control: T.Tensor((slots, 5), "int32"),
            result: T.Tensor((slots, window + 2), "int32"),
            target_hidden: T.Tensor((slots * window, width), "bfloat16"),
            hidden: T.Tensor((slots * window, width), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            tokens: T.Tensor((slots * window,), "int32"),
            head_position: T.Tensor((slots,), "int32"),
            head_active: T.Tensor((slots,), "int32"),
            mode: T.Tensor((1,), "int32"),
        ):
            with T.Kernel(slots, threads=128) as row:
                for t, col in T.Parallel(window, width):
                    hidden[row * window + t, col] = target_hidden[row * window + t, col]
                for t in T.Parallel(window):
                    tokens[row * window + t] = result[row, t + 1]
                for j in T.Parallel(1):
                    mapping[row] = control[row, 0]
                    position[row] = control[row, 1]
                    lengths[row] = result[row, 0]
                    head_position[row] = 0
                    head_active[row] = T.cast(result[row, 0] > 0, "int32")
                    if row == 0:
                        mode[0] = 0

        return kernel

    if kind == "spec_result":

        @T.prim_func
        def kernel(
            predicted: T.Tensor((slots,), "int32"),
            result: T.Tensor((slots, window + 2), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        result[row, window + 1] = predicted[row]

        return kernel
    raise ValueError("unknown speculative control kernel")
