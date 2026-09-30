"""Physical-GPU acceptance cannot be inferred from software or partial evidence."""
import copy

import pytest

from tools.webgpu_audit import audit


def evidence():
    suite = {"schema":"tensor.webgpu-validation.v1","module_sha256":"module",
             "consumer_source_sha256":{"runtime":"hash"},"producer":{"hostname":"build"},
             "cases":[{"name":"profile","sha256":"artifact"}]}
    result = {"schema":"tensor.webgpu-result.v1","status":"passed","suite_sha256":"suite",
              "module_sha256":"module","consumer_source_sha256":{"runtime":"hash"},
              "producer":suite["producer"],"hostname":"consumer","compiler_imports":[],
              "compiler_import_guard":True,"packages":["numpy","tensor-workspace","wgpu"],
              "adapter":{"adapter":{"adapter_type":"DiscreteGPU","vendor":"AMD","vendor_id":0x1002}},
              "physical_second_gpu":True,"software_adapter":False,
              "cases":[{"name":"profile","status":"passed","artifact_sha256":"artifact",
                        "maximum_absolute_error":.001,"timing":{"iters":3,"median_launch_and_sync_seconds":.001,"median_host_enqueue_seconds":.0001}},
                       {"name":"mlp_chain","status":"passed"}]}
    return suite,result


def test_physical_transfer_and_explicit_software_boundary():
    suite,result = evidence()
    assert audit(suite,result,suite_sha256="suite")["phase5_hardware_gate"] == "passed"
    result["adapter"]["adapter"] = {"adapter_type":"CPU","vendor":"llvmpipe"}
    result.update(physical_second_gpu=False,software_adapter=True)
    with pytest.raises(ValueError,match="physical AMD or Apple"):
        audit(suite,result,suite_sha256="suite")
    assert audit(suite,result,suite_sha256="suite",require_second_gpu=False)["phase5_hardware_gate"] == "open"


@pytest.mark.parametrize("failure", ["imports","packages","source","missing","duplicate","hostname","timing","classification"])
def test_incomplete_or_misleading_evidence_fails(failure):
    suite,result = evidence()
    if failure == "imports":
        result["compiler_import_guard"] = False
    elif failure == "packages":
        result["packages"].append("tilelang")
    elif failure == "source":
        result["consumer_source_sha256"] = {"runtime":"old"}
    elif failure == "missing":
        result["cases"].pop()
    elif failure == "duplicate":
        result["cases"].append(copy.deepcopy(result["cases"][0]))
    elif failure == "hostname":
        result["hostname"] = "build"
    elif failure == "timing":
        result["cases"][0]["timing"]["median_launch_and_sync_seconds"] = float("nan")
    else:
        result["software_adapter"] = True
    with pytest.raises(ValueError):
        audit(suite,result,suite_sha256="suite")
