"""Check the returned WebGPU acceptance evidence against the transferred suite."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(suite, result, *, suite_sha256, require_second_gpu=True):
    if suite.get("schema") != "tensor.webgpu-validation.v1" or result.get("schema") != "tensor.webgpu-result.v1" or result.get("status") != "passed":
        raise ValueError("missing successful WebGPU suite/result")
    for field, expected in (("suite_sha256", suite_sha256), ("module_sha256",suite["module_sha256"]),
                            ("producer",suite["producer"]), ("consumer_source_sha256",suite["consumer_source_sha256"])):
        if result.get(field) != expected:
            raise ValueError(f"WebGPU evidence {field} mismatch")
    expected = {case["name"]:case for case in suite["cases"]}
    observed = {case["name"]:case for case in result["cases"]}
    if (len(expected) != len(suite["cases"]) or len(observed) != len(result["cases"])
            or set(observed) != set(expected) | {"mlp_chain"}):
        raise ValueError("WebGPU evidence is missing or duplicates coverage")
    for name, case in observed.items():
        if case.get("status") != "passed":
            raise ValueError(f"WebGPU case failed: {name}")
        if name != "mlp_chain":
            if case.get("artifact_sha256") != expected[name]["sha256"]:
                raise ValueError(f"WebGPU artifact mismatch: {name}")
            timing = case.get("timing",{})
            if type(timing.get("iters")) is not int or timing["iters"] < 3:
                raise ValueError("WebGPU evidence needs at least three timing samples per case")
            for field in ("median_launch_and_sync_seconds", "median_host_enqueue_seconds"):
                value = timing.get(field)
                if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid WebGPU timing evidence")
            if not math.isfinite(case["maximum_absolute_error"]):
                raise ValueError("invalid WebGPU numerical error")
    if result.get("compiler_imports") != [] or result.get("compiler_import_guard") is not True:
        raise ValueError("WebGPU consumer import isolation was not established")
    if {name.lower().replace('_','-') for name in result.get("packages",[])} & {"tilelang","apache-tvm-ffi","torch","triton"}:
        raise ValueError("WebGPU consumer contains compiler/framework packages")
    adapter = result["adapter"]["adapter"]
    hardware = adapter["adapter_type"] in ("DiscreteGPU", "IntegratedGPU")
    vendor = str(adapter.get("vendor","")).lower()
    second = hardware and (adapter.get("vendor_id") in (0x1002,0x106b) or "amd" in vendor or "apple" in vendor)
    if result.get("physical_second_gpu") is not second or result.get("software_adapter") is not (adapter["adapter_type"] == "CPU"):
        raise ValueError("WebGPU hardware classification contradicts adapter metadata")
    two_hosts = bool(result.get("hostname")) and result["hostname"] != suite["producer"].get("hostname")
    if require_second_gpu and (not second or not two_hosts):
        raise ValueError("Phase 5 needs a transferred suite on a physical AMD or Apple GPU")
    return {"status":"passed","cases":len(observed),"physical_second_gpu":second,"two_hosts":two_hosts,
            "phase5_hardware_gate":"passed" if second and two_hosts else "open"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--allow-software",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit(json.loads(args.suite.read_text()),json.loads(args.result.read_text()),
                           suite_sha256=hashlib.sha256(args.suite.read_bytes()).hexdigest(),
                           require_second_gpu=not args.allow_software),indent=2))
