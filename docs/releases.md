# Prepare a release

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md) · [Changelog](../CHANGELOG.md)

Tensor is pre-1.0. Distribution checks prepare files for review; they do not
upload to a registry or create a GitHub release. The package names remain
`tensor-workspace`, `tensor-nn`, `tensor-llm`, and `tensor-torch`.

## Build reviewable files

With Python 3.12 and uv, from a clean checkout:

```sh
uv sync --locked
uv run --locked python -m pytest
uv run --locked python scripts/validation/distribution_check.py --out build/release-candidate
```

Use a fresh output directory for each candidate. The script builds all four
wheels and source archives, verifies metadata, READMEs, licenses, native source
inclusions, and package boundaries, then runs Twine's metadata checks. It rebuilds
each source archive in isolation and installs the core wheel into a temporary
consumer containing only Tensor and NumPy. Import and CLI checks run outside the
checkout with compiler/framework imports blocked. The example's `--help` is also
checked there; actual GPU execution is a separate validation step.

To recheck existing files:

```sh
uv run --locked python scripts/validation/distribution_check.py --out build/release-candidate --check-only
```

The [Distribution checks workflow](../.github/workflows/distribution.yml) runs
this on Linux and Windows and retains the candidate files as CI artifacts. It
requires no GPU, TileLang, or NVRTC. GPU/provider acceptance workflows remain
responsible for execution validation. Default candidates omit the opt-in WebGPU
and PyTorch native executors; explicitly built native wheels need their own
platform, Python, and framework validation.

## Before publishing

1. Select a version and update the root `pyproject.toml`, `tensor.__version__`,
   all optional-package versions, their exact core dependency pins, and the
   workspace development pins. Run `uv lock`; update versioned wheel examples.
2. Move user-facing `Unreleased` entries into a dated changelog entry. Describe
   any API, ABI, target, or implementation-fingerprint changes and rebuild steps.
3. Pass the distribution checks on Linux and Windows and run the relevant
   provider/workload acceptance on actual hardware. Record skips and unvalidated
   platforms in the release notes; don't infer physical support from software CI.
4. Test the quickstart from a fresh checkout and the produced wheel in a fresh
   consumer. Keep the wheels used for any fingerprint-bound bundles together.
5. Review the license notices, bundled third-party code, and any model-weight
   terms for the files actually shipped. Tensor's MIT license does not replace
   dependency or model licenses.
6. Confirm registry ownership and choose the public distribution channel. Upload
   the reviewed files and publish release notes only as an explicit release action.

Publish the core and matching optional packages together. Avoid reusing a version
for different wheel contents. Keep source archives available alongside wheels,
and describe the supported Python/hardware profiles in the release notes.

## Core packages versus kernel modules

`uv build` prepares Tensor's Python distributions. `tensor publish` packages a
kernel module into a transport wheel; it is a different release operation. See
the [module guide](guides/modules.md) and use `tensor publish --dry-run` to inspect
a module candidate before upload.
