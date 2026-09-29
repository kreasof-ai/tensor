"""P0 ownership and ordering probe on borrowed PyTorch CUDA streams and buffers."""

import ctypes as C
import hashlib


def probe(root):
    import torch
    from experiments.p0.artifact_build import prepare, compile_bundle
    from experiments.p0.artifact_format import read_bundle
    from experiments.p0.cuda_driver import Driver

    torch.cuda.init()
    torch.empty(1, device="cuda")  # attach PyTorch's primary context to this thread
    driver = Driver()
    _, info = driver.device_info()
    if info["arch"] != "sm_86":
        raise RuntimeError("foreign-stream probe requires the measured sm_86 hardware")
    context_before = C.c_void_p()
    driver.call("cuCtxGetCurrent", C.byref(context_before))
    assert context_before.value
    source, executable = root / "source.zip", root / "kernel.tbin"
    prepare(source, size=129, arch=info["arch"])
    compile_bundle(source, executable)
    manifest, files = read_bundle(executable, kind="cubin")

    # These are borrowed primary-context resources. The probe owns only its
    # module and events. No new CUDA context or stream is created.
    events = ("cuEventCreate", [C.POINTER(C.c_void_p), C.c_uint]),
    for name, args in [*events, ("cuEventRecord", [C.c_void_p, C.c_void_p]),
                       ("cuEventSynchronize", [C.c_void_p]),
                       ("cuEventDestroy_v2", [C.c_void_p]),
                       ("cuStreamWaitEvent", [C.c_void_p, C.c_void_p, C.c_uint])]:
        method = getattr(driver.lib, name)
        method.argtypes, method.restype = args, C.c_int
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    default = torch.cuda.current_stream()
    ready, finished, module, function = (C.c_void_p() for _ in range(4))
    rows = []
    try:
        driver.call("cuEventCreate", C.byref(ready), 2)  # disable timing
        driver.call("cuEventCreate", C.byref(finished), 2)
        image = C.create_string_buffer(files["kernel.cubin"])
        driver.call("cuModuleLoadData", C.byref(module), image)
        driver.call("cuModuleGetFunction", C.byref(function), module, manifest["entrypoint"].encode())
        for seed in range(3):
            with torch.cuda.stream(producer):
                torch.manual_seed(seed)
                a = torch.randn(129, dtype=torch.float32, device="cuda")
                b = torch.randn(129, dtype=torch.float32, device="cuda")
                out = torch.full((129,), float("nan"), dtype=torch.float32, device="cuda")
            addresses = [a.data_ptr(), b.data_ptr(), out.data_ptr()]
            driver.call("cuEventRecord", ready, C.c_void_p(producer.cuda_stream))
            driver.call("cuStreamWaitEvent", C.c_void_p(consumer.cuda_stream), ready, 0)
            holders = [C.c_uint64(value) for value in addresses]
            params = (C.c_void_p * 3)(*(C.addressof(x) for x in holders))
            launch = manifest["launch"]
            driver.call("cuLaunchKernel", function, *launch["grid"], *launch["block"],
                        launch["shared_memory_bytes"], C.c_void_p(consumer.cuda_stream), params, None)
            driver.call("cuEventRecord", finished, C.c_void_p(consumer.cuda_stream))
            driver.call("cuStreamWaitEvent", C.c_void_p(default.cuda_stream), finished, 0)
            driver.call("cuEventSynchronize", finished)
            expected = torch.relu(2*a+b)
            torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
            assert addresses == [a.data_ptr(), b.data_ptr(), out.data_ptr()]
            rows.append({"seed": seed, "size": 129, "max_abs_error": float((out-expected).abs().max()),
                         "borrowed_buffer_addresses_stable": True})
    finally:
        # Synchronize owned work before unloading. Destroy only owned handles.
        if finished.value:
            driver.call("cuEventSynchronize", finished)
        if module.value:
            driver.call("cuModuleUnload", module)
        for event in (finished, ready):
            if event.value:
                driver.call("cuEventDestroy_v2", event)
    context_after = C.c_void_p()
    driver.call("cuCtxGetCurrent", C.byref(context_after))
    assert context_after.value == context_before.value
    with torch.cuda.stream(producer):
        still_usable = torch.ones(1, device="cuda") + 1
    producer.synchronize()
    assert still_usable.item() == 2
    return {"rows": rows, "device": info, "artifact_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "context_preserved": True, "borrowed_stream_survived_cleanup": True,
            "ordering": "producer event -> consumer wait/launch -> completion event -> default stream wait",
            "ownership": "Torch streams and tensors borrowed; CUDA module and events owned by probe",
            "limit": "CUDA-specific same-device interop, not a provider-neutral event ABI or distributed signal contract"}
