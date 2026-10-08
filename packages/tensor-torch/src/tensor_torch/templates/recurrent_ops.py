"""Native elementwise operations for recurrent/gated architecture profiles."""


def unary(p):
    import tilelang.language as T

    n, dt, gd, kind = p
    if kind.endswith("_dx"):

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), dt), dy: T.Tensor((n,), gd), out: T.Tensor((n,), dt)
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        z = T.cast(x[i], "float32")
                        sig = 1 / (1 + T.exp(-z))
                        derivative = (
                            sig * (1 - sig)
                            if kind == "sigmoid_dx"
                            else sig * (1 + z * (1 - sig))
                        )
                        out[i] = T.cast(dy[i], "float32") * derivative

    else:

        @T.prim_func
        def kernel(x: T.Tensor((n,), dt), out: T.Tensor((n,), dt)):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        z = T.cast(x[i], "float32")
                        sig = 1 / (1 + T.exp(-z))
                        out[i] = sig if kind == "sigmoid" else z * sig

    return kernel


def multiply(p):
    import tilelang.language as T

    n, xd, yd, od = p

    @T.prim_func
    def kernel(x: T.Tensor((n,), xd), y: T.Tensor((n,), yd), out: T.Tensor((n,), od)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = T.cast(x[i], "float32") * T.cast(y[i], "float32")

    return kernel


def blend(p):
    import tilelang.language as T

    n, gd, xd, zd, od = p

    @T.prim_func
    def kernel(
        g: T.Tensor((n,), gd),
        x: T.Tensor((n,), xd),
        z: T.Tensor((n,), zd),
        out: T.Tensor((n,), od),
    ):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    gate = T.cast(g[i], "float32")
                    out[i] = gate * T.cast(x[i], "float32") + (1 - gate) * T.cast(
                        z[i], "float32"
                    )

    return kernel


def blend_backward(p):
    import tilelang.language as T

    n, gd, xd, zd, od = p

    @T.prim_func
    def kernel(
        g: T.Tensor((n,), gd),
        x: T.Tensor((n,), xd),
        z: T.Tensor((n,), zd),
        dy: T.Tensor((n,), od),
        dg: T.Tensor((n,), gd),
        dx: T.Tensor((n,), xd),
        dz: T.Tensor((n,), zd),
    ):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    gate = T.cast(g[i], "float32")
                    go = T.cast(dy[i], "float32")
                    dg[i] = (T.cast(x[i], "float32") - T.cast(z[i], "float32")) * go
                    dx[i] = gate * go
                    dz[i] = (1 - gate) * go

    return kernel
