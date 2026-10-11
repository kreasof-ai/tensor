"""Install explicit Hopper cubins in an existing native execution graph."""

from tensor_llm.common.artifacts import backbone_hashes
import hashlib
import json
from pathlib import Path
from tensor.providers.cuda_graph import CudaGraph
from tensor_llm.common.plan import bind as bind_call
from tensor_llm.common.plan import submit as submit_calls
from tensor_llm.common.artifacts import artifact_path
from tensor.runtime.cuda_target import matches_device


def implementation_hashes():
    return {
        __name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    } | backbone_hashes()


def install(executor, bundle):
    executor.model._check()
    if executor.closed or getattr(executor, "pending_verification", False):
        raise RuntimeError("executor must be idle")
    if getattr(executor, "_hopper_installed", False):
        raise ValueError("Hopper kernels already installed")
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "hopper.json").read_text())
    if manifest["implementation"] != implementation_hashes():
        raise ValueError("Hopper implementation mismatch; rebuild bundle")
    if not matches_device(manifest["target"], executor.device.info["arch"]):
        raise ValueError("Hopper device mismatch")
    if (manifest["slots"], manifest["chunk"], manifest["context"]) != (
        executor.model.slots,
        executor.chunk,
        executor.model.context,
    ):
        raise ValueError("Hopper execution capacity mismatch")
    loaded = {}
    replacements = {}
    device = executor.device
    try:
        for key, record in manifest["kernels"].items():
            old = executor.kernels.get(key)
            if old is None:
                raise ValueError("Hopper source kernel absent: " + key)
            if old.manifest["files"]["kernel.cubin"] != record["original_cubin_sha256"]:
                raise ValueError("Hopper source kernel mismatch: " + key)
            path = artifact_path(bundle, record)
            new = device.load(path)
            loaded[key] = new
            if new.manifest["arguments"] != old.manifest["arguments"]:
                raise ValueError("Hopper kernel signature changed")
            replacements[id(old)] = new
        plan = []
        for kernel, bound in executor.plan:
            selected = replacements.get(id(kernel))
            if selected is None:
                plan.append((kernel, bound))
                continue
            values = dict(
                zip((a["name"] for a in kernel.manifest["abi"]), bound.storage)
            )
            args = tuple(values[a["name"]] for a in selected.manifest["arguments"])
            plan.append(
                (
                    selected,
                    bind_call(device, selected, args),
                )
            )
        graph = CudaGraph(
            device,
            lambda: submit_calls(device, plan),
            resources=(*executor.graph.resources, *loaded.values()),
        )
    except BaseException:
        for kernel in loaded.values():
            kernel.release()
        raise
    executor.graph.close()
    executor.graph = graph
    executor.plan = plan
    executor.kernels.update({"_hopper_" + k: v for k, v in loaded.items()})
    executor._hopper_installed = True
