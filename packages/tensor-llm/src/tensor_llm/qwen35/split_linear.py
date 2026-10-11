"""Install selected Split-K projections into an existing native chunk graph."""

from tensor_llm.common.artifacts import backbone_hashes
import hashlib
import json
from pathlib import Path
from tensor_llm.common.plan import bind as bind_call
from tensor_llm.common.plan import submit as submit_calls
from tensor_llm.common.artifacts import artifact_path
from tensor.providers.cuda_graph import CudaGraph
from tensor_llm.qwen35.artifacts import identity


def implementation_hashes():
    from tensor_llm.kernels import speculative_linear
    from tensor_llm.kernels import matmul

    return (
        {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (speculative_linear, matmul)
        }
        | {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        | backbone_hashes()
    )


def install(executor, bundle):
    executor.model._check()
    if executor.closed or getattr(executor, "pending_verification", False):
        raise RuntimeError("executor must be idle")
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "prefill.json").read_text())
    if manifest.get("split_linear_implementation") != implementation_hashes():
        raise ValueError("Split-K implementation mismatch")
    if (manifest["slots"], manifest["chunk"], manifest["context"]) != (
        executor.model.slots,
        executor.chunk,
        executor.model.context,
    ):
        raise ValueError("Split-K capacity mismatch")
    d = executor.device
    loaded = {}
    buffers = {}
    replacements = {}
    try:
        if "fixed_routes" in manifest:
            key = manifest["fixed_routes"]
            row = manifest["kernels"][key]
            path = artifact_path(bundle, row)
            loaded[key] = d.load(path)
            buffers["groups"] = d.empty((executor.rows * 8,), "int32")
            buffers["group_routes"] = d.empty(
                (executor.rows * 8, executor.rows), "int32"
            )
        for logical, record in manifest["split_linear"].items():
            old = executor.kernels[logical]
            for key in (record["matmul"], record.get("merge")):
                if key is None:
                    continue
                if key in loaded:
                    continue
                row = manifest["kernels"][key]
                path = artifact_path(bundle, row)
                loaded[key] = d.load(path)
            if "partial_shape" in record:
                buffers[logical] = d.empty(tuple(record["partial_shape"]))
            replacements[id(old)] = (logical, record)
        plan = []

        def bind(kernel, values):
            args = tuple(values[a["name"]] for a in kernel.manifest["arguments"])
            return kernel, bind_call(d, kernel, args)

        for kernel, bound in executor.plan:
            if "fixed_routes" in manifest and kernel is executor.kernels.get(
                identity(
                    "expert_routes",
                    dict(slots=executor.model.slots, chunk=executor.chunk),
                )
            ):
                values = dict(
                    zip(
                        (
                            a["name"]
                            for a in kernel.manifest.get(
                                "abi", kernel.manifest["arguments"]
                            )
                        ),
                        bound.storage,
                    )
                )
                plan.append(
                    bind(
                        loaded[manifest["fixed_routes"]],
                        dict(
                            ids=values["ids"],
                            experts=buffers["groups"],
                            routes=buffers["group_routes"],
                        ),
                    )
                )
                continue
            record = replacements.get(id(kernel))
            if record is None:
                plan.append((kernel, bound))
                continue
            logical, record = record
            values = dict(
                zip(
                    (
                        a["name"]
                        for a in kernel.manifest.get(
                            "abi", kernel.manifest["arguments"]
                        )
                    ),
                    bound.storage,
                )
            )
            if "merge" not in record:
                plan.append(bind(loaded[record["matmul"]], values))
                continue
            if record.get("fixed_rows"):
                values.pop("counts")
                values.update(experts=buffers["groups"], routes=buffers["group_routes"])
            out = values["out"]
            values["out"] = buffers[logical]
            plan.append(bind(loaded[record["matmul"]], values))
            plan.append(
                bind(loaded[record["merge"]], dict(partial=buffers[logical], out=out))
            )
        graph = CudaGraph(
            d,
            lambda: submit_calls(d, plan),
            resources=(*executor.graph.resources, *buffers.values(), *loaded.values()),
        )
    except BaseException:
        for value in buffers.values():
            value.release()
        for kernel in loaded.values():
            kernel.release()
        raise
    executor.graph.close()
    executor.graph = graph
    executor.plan = plan
    executor.kernels.update(loaded)
    executor.buffers.update({"_split_linear_" + k: v for k, v in buffers.items()})
