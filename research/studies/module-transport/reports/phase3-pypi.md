# Phase 3 PyPI transport acceptance

Date: 2026-09-30. Validated implementation:
`c7cc1acad51c6ff80a4f6e33645c40b3dff7584b`.

Tensor now publishes a verified closed `.tpack` as a data-only wheel using
optional Twine and installs exact versions through a Python Simple Index.
Registry dependencies pin both transport and module hashes; frozen installation
can restore an empty cache and offline installation avoids registry requests.
The [module guide](../guides/modules.md) documents usage and
[ADR 0014](../adr/0014-pypi-module-transport.md) describes the transport profile.

## Results

| Check | Result |
|---|---|
| Full existing GPU suite plus registry tests | 116 passed, zero skipped, 71.05 s |
| Registry acceptance tests | 12 passed, including actual local Twine upload |
| Linux CI suite | 97 passed, 19 skipped, 17.78 s |
| Windows CI suite | 87 passed, 29 skipped, 24.92 s |
| Clean Linux/Windows CI consumers | Five packaged exports each; fresh-cache restoration and frozen offline installation passed |
| Linux-produced kernels through local registry on A10G | 21 numerical/interop cases and eight diagnostics passed |
| Windows-produced kernels through local registry on A10G | 21 numerical/interop cases and eight diagnostics passed |

CI [36651240548](https://github.com/kreasof-ai/tensor/actions/runs/36651240548)
passed for the exact implementation above. GPU-free runners skip the existing
hardware-dependent checks; Windows additionally skips the Linux CPU/native host
profile. Each runner performs an actual Twine subprocess upload to a local
multipart server, retrieves the received wheel through the Tensor resolver,
then tests a clean consumer with only Tensor and NumPy.

The A10G acceptance downloaded each verified CI archive and installed its exact
runtime wheel. Installed runtime files match the corresponding producer wheel;
repository comparisons normalize only the Windows checkout's CRLF line endings.
Each producer's clean revision, package graph and all five source payloads match
the validated commit. Each downloaded `.tpack` was wrapped on the consumer into
a transport wheel, served by a local HTML Simple Index, restored into a second
empty cache, and installed frozen again after the server stopped. All five
selections were packaged images, with compiler imports prohibited and no
generated kernel images. Windows production was exercised on a Linux GPU
consumer; this does not establish Windows GPU execution.

Consumer-prepared transport wheels:

| Producer | Bytes | SHA-256 |
|---|---:|---|
| Linux | 142398 | `001d54ec3742c3391239fa90fa9baee071d7ef4a6c0cac4b5443d8992c0527ec` |
| Windows | 141856 | `7b557152dd588f5feea2f67fd7d60d14c0da96b7274abcadc1eeebbfe0815288` |

These are module payload wheels, not compiler installations. Raw evidence:
[CI metadata](data/phase3-pypi-ci.json),
[Linux acceptance](data/phase3-pypi-linux.json),
[Windows acceptance](data/phase3-pypi-windows.json).
The full local test log is retained at `build/phase3-registry-pytest.log`, SHA-256
`badaa71188b53ac96bf41497b260ff11696fc0274e18ed3b45c59e5cc72228ad`.
The final test change strengthened three tampering cases with valid outer RECORD
hashes; its registry suite was rerun locally and the committed suite passed CI.

## Boundaries

No package was uploaded to public PyPI or TestPyPI. Upload acceptance used a real
Twine client against a local endpoint; public publication still needs an account,
available project name and configured Twine authentication. The consumer supports
public JSON/HTML Simple Index endpoints, exact versions and this Tensor data-wheel
profile. Authenticated private downloads, version ranges, fusion/autotuning and
broader CUDA compatibility remain separate work. NVRTC remains the compiler
default and direct PTX remains experimental work after Tensor v1.
