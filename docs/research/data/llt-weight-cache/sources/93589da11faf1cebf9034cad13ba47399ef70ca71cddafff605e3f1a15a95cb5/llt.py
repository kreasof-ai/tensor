"""Explicit Tensor CUDA operators for loop-latent training and inference.

PyTorch owns autograd scheduling and storage. Every call listed in the coverage
report executes a Tensor artifact; unsupported inputs raise, never silently fall
back. First use compiles; warmed artifacts execute without compiler packages.
"""

from pathlib import Path
import hashlib
import os
import threading
import math
import weakref
import torch
from torch.autograd.function import once_differentiable
from .bridge import Kernel, LaunchPlan


class Operators:
    def __init__(self, cache_dir=None, *, compute_dtype=torch.bfloat16, gemm_profile="optimized", cache_inference_weights=False):
        if compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("compute_dtype must be FP16 or BF16")
        self.compute_dtype = compute_dtype
        if gemm_profile not in ("optimized", "legacy"):
            raise ValueError("gemm_profile must be optimized or legacy")
        self.gemm_profile = gemm_profile
        self.cache_inference_weights = bool(cache_inference_weights)
        self.inference_weight_casts = {}
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("TENSOR_LLT_CACHE_DIR")
            or Path.home() / ".cache/tensor/llt"
        )
        self.kernels = {}
        self.plans = {}
        self.factories = {}
        self.lock = threading.RLock()
        self.report = {
            "artifacts": [],
            "calls": {},
            "fallbacks": [],
            "layout_copies": 0,
            "compute_dtype": str(compute_dtype),
            "gemm_profile": gemm_profile,
            "cache_inference_weights": self.cache_inference_weights,
            "inference_weight_cast_hits": 0,
        }

    def kernel(
        self, factory, parameters, outputs, *, module="tensor_torch.templates.llt"
    ):
        from tensor.compiler.entry import export_source

        target = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
        specialization = (module, factory, repr(parameters), tuple(outputs), target)
        if specialization in self.factories:
            return self.factories[specialization]
        source = export_source(
            module,
            factory,
            parameters,
            dependencies=(module, "tensor.compiler.entry", "tensor.runtime.abi"),
            outputs=outputs,
        )
        key = hashlib.sha256((source + target).encode()).hexdigest()
        with self.lock:
            if key not in self.kernels:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path = self.cache_dir / (key + ".tbin")
                hit = path.exists()
                if not hit:
                    import tensor

                    entry = self.cache_dir / (key + ".py")
                    entry.write_text(source)
                    tensor.build(
                        entry,
                        path,
                        target=target,
                        cache_dir=self.cache_dir / "compiler",
                    )
                mutable = (
                    ("accumulator",)
                    if module == "tensor_torch.templates.llt_accumulate"
                    else ()
                )
                if module == "tensor_torch.templates.llt_ops":
                    mutable = {
                        "zero": ("out",),
                        "embedding_scatter": ("dw",),
                        "adam_step": ("step",),
                        "adamw": ("weight", "moment", "variance"),
                    }.get(parameters["kind"], ())
                if factory == "cache_append":
                    mutable = ("keys",) if parameters[-1] else ("keys", "values")
                if factory == "cache_advance":
                    mutable = ("lengths", "overflow")
                self.kernels[key] = Kernel(path, mutable_inputs=mutable)
                self.report["artifacts"].append(
                    {
                        "factory": factory,
                        "module": module,
                        "parameters": parameters,
                        "target": target,
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "cache_hit": hit,
                    }
                )
        self.factories[specialization] = self.kernels[key]
        return self.kernels[key]

    def call(
        self, factory, parameters, outputs, *args, module="tensor_torch.templates.llt"
    ):
        kernel = self.kernel(factory, parameters, outputs, module=module)
        normalized = []
        copies = {}
        for descriptor, x in zip(kernel.inputs, args):
            if not x.is_contiguous() or x.data_ptr() % descriptor.get("alignment", 256):
                if descriptor["name"] in kernel.mutable_inputs:
                    raise ValueError(
                        "mutable Tensor buffers require contiguous aligned storage"
                    )
                identity = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype)
                if identity not in copies:
                    copies[identity] = x.clone(memory_format=torch.contiguous_format)
                    self.report["layout_copies"] += 1
                x = copies[identity]
            normalized.append(x)
        args = tuple(normalized)
        key = (
            kernel,
            tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device) for x in args),
        )
        with self.lock:
            if key not in self.plans:
                self.plans[key] = LaunchPlan(kernel, args, lambda *a: kernel.raw(*a))
        label = factory + (
            ":" + parameters["kind"]
            if isinstance(parameters, dict) and "kind" in parameters
            else ""
        )
        self.report["calls"][label] = self.report["calls"].get(label, 0) + 1
        result = self.plans[key](*args)
        for descriptor, arg in zip(kernel.inputs, args):
            if descriptor["name"] in kernel.mutable_inputs:
                torch.autograd.graph.increment_version(arg)
        return result

    def attention(self, q, k, v, *, causal=False, scale=None, query_offset=0):
        if any(
            x.ndim != 4 or x.device.type != "cuda" or not x.is_contiguous()
            for x in (q, k, v)
        ):
            raise ValueError("attention requires contiguous CUDA BHSD tensors")
        b, h, m, d = q.shape
        bk, kh, n, dk = k.shape
        if (
            bk != b
            or v.shape[:3] != (b, kh, n)
            or dk != d
            or kh < 1
            or h % kh
            or any(size < 1 for size in (*q.shape, *k.shape, *v.shape))
            or d % 16
            or v.shape[-1] % 16
            or d > 256
            or v.shape[-1] > 128
        ):
            raise ValueError(
                "attention score/value dimensions must be multiples of 16, at most 256/128"
            )
        if (
            len({x.dtype for x in (q, k, v)}) != 1
            or q.dtype not in (torch.float16, torch.bfloat16)
            or len({x.device for x in (q, k, v)}) != 1
        ):
            raise ValueError("attention requires matching FP16/BF16 dtype and device")
        if type(query_offset) != int or query_offset < 0:
            raise ValueError(
                "query_offset must be a nonnegative integer; causal masking is j <= query_offset+i"
            )
        import math

        scale = d**-0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("attention scale must be finite")
        p = (
            b,
            h,
            kh,
            m,
            n,
            d,
            v.shape[-1],
            str(q.dtype).removeprefix("torch."),
            bool(causal),
            query_offset,
            scale,
        )
        return _Attention.apply(q, k, v, self, p)

    def decode(self, q, cache, *, scale=None, partitions=None):
        if torch.is_grad_enabled() and q.requires_grad:
            raise ValueError("persistent-cache decode is inference-only")
        if (
            q.shape[:1] != (cache.keys.shape[0],)
            or q.ndim != 4
            or q.shape[2] != 1
            or q.shape[3] != cache.keys.shape[-1]
            or q.shape[1] % cache.keys.shape[1]
            or q.dtype != cache.keys.dtype
            or q.device != cache.keys.device
            or not q.is_contiguous()
        ):
            raise ValueError("decode geometry/dtype does not match cache")
        if partitions is None:
            partitions = 64 if cache.capacity >= 32768 and q.shape[-1] <= 64 else 32
        if partitions not in (8, 16, 32, 64, 128, 256):
            raise ValueError("decode partitions must be 8/16/32/64/128/256")
        if cache.length < 1:
            raise ValueError("decode needs a nonempty prefix")
        b, h, _, d = q.shape
        kh, capacity = cache.keys.shape[1:3]
        dv = cache.values.shape[-1]
        scale = d**-0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("decode scale must be finite")
        p = (b, h, kh, capacity, d, dv, _dtype(q), scale, partitions)
        partial, stats = self.call(
            "decode_partial",
            p,
            ["partial", "stats"],
            q,
            cache.keys,
            cache.values,
            cache.lengths,
            module="tensor_torch.templates.llt_decode",
        )
        return self.call(
            "decode_merge",
            (b, h, dv, _dtype(q), partitions),
            ["out"],
            partial,
            stats,
            module="tensor_torch.templates.llt_decode",
        )

    def training(self, p, outputs, *args):
        return self.call(
            "make_kernel", p, outputs, *args, module="tensor_torch.templates.llt_ops"
        )

    def cast(self, x, dtype):
        if x.dtype == dtype:
            return x.contiguous()
        cacheable = (self.cache_inference_weights and torch.is_inference_mode_enabled()
                     and isinstance(x, torch.nn.Parameter))
        if cacheable:
            key = (id(x), dtype)
            version = (x._version, x.data_ptr())
            entry = self.inference_weight_casts.get(key)
            if entry is not None and entry[0]() is x and entry[1] == version:
                self.report["inference_weight_cast_hits"] += 1
                return entry[2]
        result = self.training(
            {
                "kind": "cast",
                "n": x.numel(),
                "dtype": _dtype(x),
                "out_dtype": str(dtype).removeprefix("torch."),
            },
            ["out"],
            x.contiguous().reshape(-1),
        ).reshape(x.shape)
        if cacheable:
            cache = self.inference_weight_casts
            def discard(reference, key=key, cache=cache):
                entry = cache.get(key)
                if entry is not None and entry[0] is reference:
                    cache.pop(key)
            cache[key] = (weakref.ref(x, discard), version, result)
        return result

    def clear_inference_weight_cache(self):
        """Release prepared casts after disposing graphs that reference them."""
        self.inference_weight_casts.clear()

    def gemm(self, x, y, ta=False, tb=False, out_dtype=None):
        out_dtype = self.compute_dtype if out_dtype is None else out_dtype
        x = self.cast(x, self.compute_dtype)
        y = self.cast(y, self.compute_dtype)
        m, k = (x.shape[1], x.shape[0]) if ta else x.shape
        ky, n = (y.shape[1], y.shape[0]) if tb else y.shape
        if ky != k:
            raise ValueError("matmul contraction mismatch")
        p = {
                "kind": "gemm",
                "m": m,
                "k": k,
                "c": n,
                "ta": ta,
                "tb": tb,
                "dtype": _dtype(x),
                "out_dtype": str(out_dtype).removeprefix("torch."),
            }
        if self.gemm_profile == "legacy":
            return self.training(p, ["out"], x, y)
        # L40S beam-search winners. Performance on other targets is unqualified;
        # these choices are explicit, not implicit autotuning.
        if m <= 8 and tb and not ta:
            schedule = dict(family="gemv", rows=8, chunk=256, threads=256)
        elif ta:
            schedule = dict(family="mma", bm=128, bn=64, bk=32, threads=128, stages=2)
        elif n >= 16384:
            schedule = dict(family="mma", bm=128, bn=128, bk=32, threads=128, stages=3)
        else:
            schedule = dict(family="mma", bm=64, bn=128, bk=64, threads=256, stages=3)
        p["schedule"] = schedule
        return self.call("make_kernel", p, ["out"], x, y,
                         module="tensor_torch.templates.llt_gemm")

    def matmul(self, x, y, *, transpose_a=False, transpose_b=False):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("matmul requires rank-two inputs")
        return _Matmul.apply(x, y, self, transpose_a, transpose_b)

    def batched_gemm(self, x, y, ta=False, tb=False):
        x = self.cast(x, self.compute_dtype)
        y = self.cast(y, self.compute_dtype)
        m, k = (x.shape[2], x.shape[1]) if ta else x.shape[1:]
        ky, n = (y.shape[2], y.shape[1]) if tb else y.shape[1:]
        if x.shape[0] != y.shape[0] or k != ky:
            raise ValueError("batched matmul geometry mismatch")
        p = {
            "batch": x.shape[0],
            "m": m,
            "k": k,
            "n": n,
            "ta": ta,
            "tb": tb,
            "dtype": _dtype(x),
        }
        return self.call(
            "make_kernel", p, ["out"], x, y, module="tensor_torch.templates.llt_bmm"
        )

    def bmm(self, x, y, *, transpose_a=False, transpose_b=False):
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("bmm requires rank-three inputs")
        return _BMM.apply(x, y, self, transpose_a, transpose_b)

    def linear(self, x, weight, bias=None):
        out = self.matmul(
            x.reshape(-1, x.shape[-1]), weight, transpose_b=True
        ).reshape(*x.shape[:-1], weight.shape[0])
        return _Bias.apply(out,bias,self) if bias is not None else out

    def layer_norm(self,x,weight,bias=None,eps=1e-5):
        if weight.dtype!=torch.float32 or weight.shape!=(x.shape[-1],):
            raise ValueError('LayerNorm requires FP32 channel weights')
        if bias is None:
            bias=torch.zeros_like(weight)
        if bias.dtype!=torch.float32 or bias.shape!=weight.shape or eps<=0:
            raise ValueError('invalid LayerNorm bias or epsilon')
        return _LayerNorm.apply(x,weight,bias,self,float(eps))

    def column_sum(self,x):
        while x.shape[0]>1:
            x=self.call('make_kernel',dict(kind='column_sum',r=x.shape[0],c=x.shape[1],dtype=_dtype(x)),['out'],x,
                        module='tensor_torch.templates.llt_norm')
        return x.reshape(-1)

    def rms_norm(self, x, weight, eps=1e-6):
        if weight.dtype != torch.float32 or weight.numel() != x.shape[-1]:
            raise ValueError(
                "RMS weight must be an FP32 vector matching the last dimension"
            )
        return _RMS.apply(x, weight, self, eps)

    def gelu(self, x):
        return _GELU.apply(x, self)

    def add(self, x, y):
        if x.shape != y.shape:
            raise ValueError("residual add requires matching shapes")
        return _Add.apply(x, y, self)

    def embedding(self, index, weight):
        if index.dtype != torch.int64 or weight.dtype != torch.float32:
            raise ValueError("embedding requires int64 indices and FP32 master weights")
        return _Embedding.apply(index, weight, self)

    def rotary(self, x, *, offset=0, base=10000.0):
        if x.ndim != 4 or x.shape[-1] % 2:
            raise ValueError("rotary requires BHSD with an even last dimension")
        if base <= 0 or not math.isfinite(base):
            raise ValueError("invalid rotary base")
        if type(offset) is int:
            if offset < 0:
                raise ValueError("rotary offset must be nonnegative")
            offset = torch.full(
                (x.shape[0],), offset, dtype=torch.int64, device=x.device
            )
        if (
            not isinstance(offset, torch.Tensor)
            or offset.dtype != torch.int64
            or offset.device != x.device
            or offset.shape != (x.shape[0],)
        ):
            raise ValueError(
                "rotary offset must be an integer or CUDA int64 batch vector"
            )
        return _Rotary.apply(x, self, offset.contiguous(), float(base))

    def cross_entropy(self, logits, target, *, ignore_index=-100):
        if (
            logits.ndim != 2
            or target.shape != (logits.shape[0],)
            or target.dtype != torch.int64
        ):
            raise ValueError("cross_entropy requires rank-two logits and int64 labels")
        return _CE.apply(logits, target, self, ignore_index)

    def linear_cross_entropy(
        self, x, weight, target, *, chunk_size=32, ignore_index=-100
    ):
        """Stream classifier rows and explicitly accumulate weight gradients.

        Tensor supplies GEMM/loss/backward and in-place FP32 accumulation.
        PyTorch supplies integer counts, scalar aggregation, and dX slice copies.
        """
        if (
            chunk_size < 1
            or x.ndim != 2
            or weight.ndim != 2
            or target.shape != (x.shape[0],)
            or target.dtype != torch.int64
            or x.shape[1] != weight.shape[1]
        ):
            raise ValueError("invalid chunked classifier inputs")
        return _LinearCE.apply(x, weight, target, self, chunk_size, ignore_index)

    def zero(self, shape, dtype=torch.float32, device="cuda"):
        import math

        out = torch.empty(shape, dtype=dtype, device=device)
        self.training(
            {
                "kind": "zero",
                "n": math.prod(shape),
                "dtype": str(dtype).removeprefix("torch."),
            },
            [],
            out.reshape(-1),
        )
        return out


def _dtype(x):
    return str(x.dtype).removeprefix("torch.")


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, ops, ta, tb):
        ctx.ops, ctx.ta, ctx.tb = ops, ta, tb
        ctx.save_for_backward(x, y)
        return ops.gemm(x, y, ta, tb)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, y = ctx.saved_tensors
        ops = ctx.ops
        # Match autocast: GEMMs accumulate in FP32, round their result to BF16,
        # then propagate through the storage cast to FP32 master inputs.
        dx = (
            ops.gemm(y, dy, ctx.tb, True)
            if ctx.ta
            else ops.gemm(dy, y, False, not ctx.tb)
        )
        dw = (
            ops.gemm(dy, x, True, ctx.ta)
            if ctx.tb
            else ops.gemm(x, dy, not ctx.ta, False)
        )
        return ops.cast(dx, x.dtype), ops.cast(dw, y.dtype), None, None, None


class _RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, ops, eps):
        p = {
            "kind": "rms",
            "r": x.numel() // x.shape[-1],
            "c": x.shape[-1],
            "dtype": _dtype(x),
            "eps": eps,
        }
        xx = x.contiguous().reshape(-1, x.shape[-1])
        out, inv = ops.training(p, ["out", "inv"], xx, w)
        ctx.ops, ctx.p, ctx.shape = ops, p, x.shape
        ctx.save_for_backward(xx, w, inv)
        return out.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, w, inv = ctx.saved_tensors
        dy = dy.contiguous().reshape(x.shape)
        dx = ctx.ops.training({**ctx.p, "kind": "rms_dx"}, ["out"], x, w, dy, inv)
        dw = (ctx.ops.training({**ctx.p, "kind": "rms_dw"}, ["out"], x, dy, inv)
              if ctx.needs_input_grad[1] else None)
        return dx.reshape(ctx.shape), dw, None, None


class _Bias(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bias,ops):
        ctx.ops,ctx.dtype=ops,bias.dtype
        r,c=x.numel()//x.shape[-1],x.shape[-1]
        if bias.shape!=(c,):raise ValueError('bias must match the last dimension')
        return ops.call('make_kernel',dict(kind='bias',r=r,c=c,dtype=_dtype(x)),['out'],
            x.contiguous().reshape(r,c),ops.cast(bias,x.dtype),module='tensor_torch.templates.llt_norm').reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        db=ctx.ops.column_sum(dy.contiguous().reshape(-1,dy.shape[-1]))
        # Match autocast's rounded bias-gradient storage before FP32 masters.
        db=ctx.ops.cast(ctx.ops.cast(db,dy.dtype),ctx.dtype)
        return dy,db,None


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias,ops,eps):
        p=dict(kind='ln',r=x.numel()//x.shape[-1],c=x.shape[-1],dtype=_dtype(x),eps=eps)
        xx=x.contiguous().reshape(p['r'],p['c'])
        out,mean,inv=ops.call('make_kernel',p,['out','mean','inv'],xx,w,bias,module='tensor_torch.templates.llt_norm')
        ctx.save_for_backward(xx,w,mean,inv)
        ctx.ops,ctx.p,ctx.shape=ops,p,x.shape
        return out.reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        x,w,mean,inv=ctx.saved_tensors
        ops,p=ctx.ops,ctx.p
        dy=dy.contiguous().reshape(x.shape)
        dx=ops.call('make_kernel',{**p,'kind':'ln_dx'},['out'],x,w,dy,mean,inv,module='tensor_torch.templates.llt_norm')
        dw,db=ops.call('make_kernel',{**p,'kind':'params'},['dw','db'],x,dy,mean,inv,module='tensor_torch.templates.llt_norm')
        return dx.reshape(ctx.shape),ops.column_sum(dw),ops.column_sum(db),None,None


class _BMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, ops, ta, tb):
        ctx.ops, ctx.ta, ctx.tb = ops, ta, tb
        ctx.save_for_backward(x, y)
        return ops.batched_gemm(x, y, ta, tb)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, y = ctx.saved_tensors
        ops = ctx.ops
        dx = (
            ops.batched_gemm(y, dy, ctx.tb, True)
            if ctx.ta
            else ops.batched_gemm(dy, y, False, not ctx.tb)
        )
        dw = (
            ops.batched_gemm(dy, x, True, ctx.ta)
            if ctx.tb
            else ops.batched_gemm(x, dy, not ctx.ta, False)
        )
        return ops.cast(dx, x.dtype), ops.cast(dw, y.dtype), None, None, None


class _GELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ops):
        ctx.ops = ops
        ctx.save_for_backward(x)
        return ops.training(
            {"kind": "gelu", "n": x.numel(), "dtype": _dtype(x)},
            ["out"],
            x.contiguous().reshape(-1),
        ).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return ctx.ops.training(
            {"kind": "gelu_dx", "n": x.numel(), "dtype": _dtype(x)},
            ["out"],
            x.contiguous().reshape(-1),
            dy.contiguous().reshape(-1),
        ).reshape(x.shape), None


class _Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, ops):
        ctx.xdtype, ctx.ydtype = x.dtype, y.dtype
        return ops.training(
            {
                "kind": "add",
                "n": x.numel(),
                "dtype": _dtype(x),
                "other_dtype": _dtype(y),
                "out_dtype": _dtype(x),
            },
            ["out"],
            x.contiguous().reshape(-1),
            y.contiguous().reshape(-1),
        ).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        return dy.to(ctx.xdtype), dy.to(ctx.ydtype), None


class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index, w, ops):
        p = {
            "kind": "embedding",
            "r": index.numel(),
            "c": w.shape[1],
            "v": w.shape[0],
            "dtype": _dtype(w),
        }
        ctx.ops, ctx.p, ctx.shape = ops, p, index.shape
        ctx.save_for_backward(index)
        return ops.training(p, ["out"], index.contiguous().reshape(-1), w).reshape(
            *index.shape, w.shape[1]
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (index,) = ctx.saved_tensors
        p = ctx.p
        dw = ctx.ops.zero((p["v"], p["c"]))
        ctx.ops.training(
            {**p, "kind": "embedding_scatter"},
            [],
            index.contiguous().reshape(-1),
            dy.contiguous().reshape(p["r"], p["c"]),
            dw,
        )
        return None, dw, None


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ops, offset, base):
        p = {
            "kind": "rotary",
            "shape": tuple(x.shape),
            "base": base,
            "sign": 1,
            "dtype": _dtype(x),
        }
        ctx.ops, ctx.p = ops, p
        ctx.save_for_backward(offset)
        return ops.training(p, ["out"], x.contiguous(), offset)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (offset,) = ctx.saved_tensors
        return (
            ctx.ops.training({**ctx.p, "sign": -1}, ["out"], dy.contiguous(), offset),
            None,
            None,
            None,
        )


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target, ops, chunk_size, ignore):
        total = (target != ignore).sum().float()
        losses = []
        for start in range(0, x.shape[0], chunk_size):
            labels = target[start : start + chunk_size]
            logits = ops.gemm(x[start : start + chunk_size], weight, tb=True)
            value = ops.cross_entropy(logits, labels, ignore_index=ignore)
            count = (labels != ignore).sum()
            losses.append(
                torch.where(count > 0, value * count, torch.zeros_like(value))
            )
        ctx.ops, ctx.chunk, ctx.ignore = ops, chunk_size, ignore
        ctx.save_for_backward(x, weight, target, total)
        return torch.stack(losses).sum() / total

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, target, total = ctx.saved_tensors
        ops = ctx.ops
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dw = (
            ops.zero(weight.shape, device=weight.device)
            if ctx.needs_input_grad[1]
            else None
        )
        norm = torch.where(total > 0, dy / total, torch.zeros_like(dy))
        for start in range(0, x.shape[0], ctx.chunk):
            xx = x[start : start + ctx.chunk]
            labels = target[start : start + ctx.chunk]
            logits = ops.gemm(xx, weight, tb=True)
            p = {
                "kind": "ce_parts",
                "r": xx.shape[0],
                "v": weight.shape[0],
                "dtype": _dtype(logits),
                "ignore_index": ctx.ignore,
            }
            parts = ops.training(p, ["parts"], logits)
            lse, loss = ops.training(
                {**p, "kind": "ce_finish"}, ["lse", "loss"], logits, labels, parts
            )
            dlogits = ops.training(
                {**p, "kind": "ce_dx"},
                ["out"],
                logits,
                labels,
                lse,
                norm.expand(xx.shape[0]).contiguous(),
            )
            del logits, parts, lse, loss
            if dx is not None:
                gx = ops.gemm(dlogits, weight)
                dx[start : start + ctx.chunk].copy_(ops.cast(gx, x.dtype))
                del gx
            if dw is not None:
                partial = ops.gemm(dlogits, xx, ta=True)
                ops.call(
                    "accumulate",
                    (partial.numel(), _dtype(partial)),
                    [],
                    partial.reshape(-1),
                    dw.reshape(-1),
                    module="tensor_torch.templates.llt_accumulate",
                )
                del partial
            del dlogits
        return (
            dx,
            ops.cast(dw, weight.dtype) if dw is not None else None,
            None,
            None,
            None,
            None,
        )


class _CE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, ops, ignore):
        p = {
            "kind": "ce_parts",
            "r": x.shape[0],
            "v": x.shape[1],
            "dtype": _dtype(x),
            "ignore_index": ignore,
        }
        parts = ops.training(p, ["parts"], x.contiguous())
        lse, loss = ops.training(
            {**p, "kind": "ce_finish"},
            ["lse", "loss"],
            x.contiguous(),
            target.contiguous(),
            parts,
        )
        out, count = ops.training(
            {**p, "kind": "ce_reduce"}, ["out", "count"], loss, target.contiguous()
        )
        ctx.ops, ctx.p = ops, p
        ctx.save_for_backward(x, target, lse, count)
        return out.reshape(())

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, target, lse, count = ctx.saved_tensors
        seed = ctx.ops.training(
            {**ctx.p, "kind": "ce_seed"}, ["out"], dy.contiguous().reshape(1), count
        )
        dx = ctx.ops.training(
            {**ctx.p, "kind": "ce_dx"},
            ["out"],
            x.contiguous(),
            target.contiguous(),
            lse,
            seed,
        )
        return dx, None, None, None


class AdamW(torch.optim.Optimizer):
    """FP32 master/state Tensor updates with global clipping and finite skip.

    Hyperparameter upload and optimizer orchestration use PyTorch. CUDA kernels
    perform norm reduction and updates. Weight tensors must be contiguous FP32.
    """

    def __init__(
        self,
        params,
        ops,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        max_norm=1.0,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0 or max_norm <= 0:
            raise ValueError(
                "invalid AdamW learning rate, epsilon, decay, or clipping norm"
            )
        if not all(
            math.isfinite(x) for x in (lr, eps, weight_decay, *betas)
        ) or math.isnan(max_norm):
            raise ValueError(
                "AdamW hyperparameters must be finite (max_norm may be infinity)"
            )
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )
        self.ops, self.max_norm = ops, max_norm
        self.hyper = {}

    def state_dict(self):
        state = super().state_dict()
        state["tensor_max_norm"] = self.max_norm
        return state

    def load_state_dict(self, state):
        state = dict(state)
        self.max_norm = state.pop("tensor_max_norm", self.max_norm)
        super().load_state_dict(state)
        self.hyper.clear()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if not params:
            return loss
        parts = []
        for p in params:
            if p.dtype != torch.float32 or not p.is_contiguous() or p.grad.is_sparse:
                raise ValueError(
                    "Tensor AdamW requires contiguous FP32 dense weights/gradients"
                )
            parts.append(
                self.ops.training(
                    {"kind": "sumsq", "n": p.numel(), "dtype": "float32"},
                    ["out"],
                    p.grad.contiguous().reshape(-1),
                )
            )
        parts = torch.cat(parts)
        while parts.numel() > 1024:
            parts = self.ops.training(
                {"kind": "sum", "n": parts.numel(), "dtype": "float32"}, ["out"], parts
            )
        clip = self.ops.training(
            {
                "kind": "clip",
                "n": parts.numel(),
                "limit": min(float(self.max_norm), torch.finfo(torch.float32).max),
                "dtype": "float32",
            },
            ["out"],
            parts,
        )
        self.last_norm = clip[1]
        for group_index, group in enumerate(self.param_groups):
            b1, b2 = group["betas"]
            values = (group["lr"], b1, b2, group["eps"], group["weight_decay"])
            key = (values, params[0].device)
            if group_index not in self.hyper:
                self.hyper[group_index] = (
                    key,
                    torch.tensor(values, device=params[0].device, dtype=torch.float32),
                )
            elif self.hyper[group_index][0] != key:
                hyper = self.hyper[group_index][1]
                hyper.copy_(torch.tensor(values, dtype=torch.float32))
                self.hyper[group_index] = (key, hyper)
            hyper = self.hyper[group_index][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = self.ops.zero((1,))
                    state["exp_avg"] = self.ops.zero(p.shape)
                    state["exp_avg_sq"] = self.ops.zero(p.shape)
                self.ops.training({"kind": "adam_step"}, [], state["step"], clip)
                self.ops.training(
                    {"kind": "adamw", "n": p.numel()},
                    [],
                    p.reshape(-1),
                    p.grad.contiguous().reshape(-1),
                    state["exp_avg"].reshape(-1),
                    state["exp_avg_sq"].reshape(-1),
                    state["step"],
                    clip,
                    hyper,
                )
        return loss


class KVCache:
    """Preallocated batched cache; CUDA lengths support captured append/replay.

    check() synchronizes and detects overflow after captured execution. Regular
    append rejects overflow before launch. Storage never grows or copies history.
    """

    def __init__(
        self,
        ops,
        batch,
        heads,
        capacity,
        score_dim,
        value_dim=None,
        *,
        dtype=torch.bfloat16,
        shared=False,
        device="cuda",
    ):
        value_dim = score_dim if value_dim is None else value_dim
        if (
            min(batch, heads, capacity, score_dim, value_dim) < 1
            or score_dim % 16
            or value_dim % 16
            or score_dim > 256
            or value_dim > 128
            or dtype not in (torch.float16, torch.bfloat16)
            or torch.device(device).type != "cuda"
            or (shared and score_dim != value_dim)
        ):
            raise ValueError("invalid cache geometry")
        self.ops, self.capacity, self.shared = ops, capacity, shared
        self.keys = torch.empty(
            (batch, heads, capacity, score_dim), dtype=dtype, device=device
        )
        self.values = (
            self.keys
            if shared
            else torch.empty(
                (batch, heads, capacity, value_dim), dtype=dtype, device=device
            )
        )
        self.lengths = torch.zeros(batch, dtype=torch.int32, device=device)
        self.overflow = torch.zeros_like(self.lengths)
        self.length = 0
        self.captured_mutation = False

    @property
    def nbytes(self):
        return (
            self.keys.numel() * self.keys.element_size()
            + (0 if self.shared else self.values.numel() * self.values.element_size())
            + self.lengths.numel() * 8
        )

    def reset(self):
        self.length = 0
        self.lengths.zero_()
        self.overflow.zero_()
        self.captured_mutation = False

    def check(self):
        if self.overflow.any().item():
            raise ValueError(
                "captured cache append exceeded capacity; reset before reuse"
            )
        self.length = int(self.lengths.max().item())
        return self.length

    @torch.no_grad()
    def append(self, k, v=None):
        v = k if v is None else v
        if (
            k.ndim != 4
            or k.shape[0:2] != self.keys.shape[0:2]
            or v.shape[:3] != k.shape[:3]
            or k.shape[-1] != self.keys.shape[-1]
            or v.shape[-1] != self.values.shape[-1]
            or k.dtype != self.keys.dtype
            or v.dtype != self.values.dtype
            or k.device != self.keys.device
            or v.device != self.values.device
        ):
            raise ValueError("cache append metadata mismatch")
        if self.shared and k.data_ptr() != v.data_ptr():
            raise ValueError("shared cache append requires aliased K=V")
        size = k.shape[2]
        if size < 1:
            raise ValueError("cannot append an empty chunk")
        capturing = torch.cuda.is_current_stream_capturing()
        if self.captured_mutation and not capturing:
            self.check()
        if self.length + size > self.capacity:
            raise ValueError("cache capacity exceeded")
        b, heads, capacity, d = self.keys.shape
        dv = self.values.shape[-1]
        self.ops.call(
            "cache_advance",
            (b, capacity, size),
            [],
            self.lengths,
            self.overflow,
            module="tensor_torch.templates.llt_decode",
        )
        args = (
            (k.contiguous(), self.lengths, self.overflow, self.keys)
            if self.shared
            else (
                k.contiguous(),
                v.contiguous(),
                self.lengths,
                self.overflow,
                self.keys,
                self.values,
            )
        )
        self.ops.call(
            "cache_append",
            (b, heads, capacity, d, dv, size, _dtype(k), self.shared),
            [],
            *args,
            module="tensor_torch.templates.llt_decode",
        )
        self.length += size
        self.captured_mutation |= capturing


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ops, p):
        out, lse, exact = ops.call(
            "attention_forward", p, ["out", "lse", "exact"], q, k, v
        )
        ctx.ops, ctx.p = ops, p
        ctx.save_for_backward(q, k, v, lse, exact)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, lse, exact = ctx.saved_tensors
        ops, p = ctx.ops, ctx.p
        do = do.contiguous()
        b, h, _, m, _, _, dv, dtype, _, _, _ = p
        delta = ops.call("attention_delta", (b, h, m, dv, dtype), ["delta"], exact, do)
        dq = ops.call("attention_dq", p, ["dq"], q, k, v, do, lse, delta)
        kh, n, d = p[2], p[4], p[5]
        split = m >= 256 and h // kh >= 4
        dk, dvout = ops.call(
            "attention_dkv", (*p, split), ["dk", "dvout"], q, k, v, do, lse, delta
        )
        if split:
            dk, dvout = ops.call(
                "attention_sum_heads",
                (b, h, kh, n, d, dv, dtype),
                ["dk", "dvout"],
                dk,
                dvout,
            )
        return dq, dk, dvout, None, None
