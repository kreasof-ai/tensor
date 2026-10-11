#!/usr/bin/env bash
# Install Tensor's CLI in an isolated, user-owned Python environment.
set -euo pipefail

tensor_install_root="${TENSOR_INSTALL_DIR:-$HOME/.tensor}"
tensor_source="${TENSOR_SOURCE:-https://github.com/kreasof-ai/tensor/archive/${TENSOR_REF:-main}.tar.gz}"
tensor_extras="${TENSOR_EXTRAS:-}"
case "$tensor_extras" in
    ''|compiler|webgpu|compiler,webgpu|webgpu,compiler) ;;
    *) echo 'TENSOR_EXTRAS must be compiler, webgpu, or compiler,webgpu.' >&2; exit 1 ;;
esac
case "$(uname -s)" in
    Linux|Darwin) ;;
    *) echo 'Use install.ps1 on Windows.' >&2; exit 1 ;;
esac

mkdir -p "$tensor_install_root/bin"
tensor_install_root="$(cd "$tensor_install_root" && pwd)"
if [[ -x "$tensor_install_root/bin/uv" ]]; then
    tensor_uv="$tensor_install_root/bin/uv"
elif command -v uv >/dev/null 2>&1; then
    tensor_uv="$(command -v uv)"
else
    tensor_installer="$(mktemp)"
    trap 'rm -f "$tensor_installer"' EXIT
    curl -fsSL --retry 3 https://astral.sh/uv/0.12.9/install.sh -o "$tensor_installer"
    UV_UNMANAGED_INSTALL="$tensor_install_root/bin" sh "$tensor_installer"
    tensor_uv="$tensor_install_root/bin/uv"
fi

export UV_TOOL_DIR="$tensor_install_root/tools"
export UV_TOOL_BIN_DIR="$tensor_install_root/bin"
export UV_PYTHON_INSTALL_DIR="$tensor_install_root/python"
tensor_package='tensor-workspace'
if [[ -n "$tensor_extras" ]]; then
    tensor_package="tensor-workspace[$tensor_extras]"
fi
"$tensor_uv" tool install --managed-python --python 3.12 --reinstall --force "$tensor_package @ $tensor_source"
"$tensor_install_root/bin/tensor" --version
if [[ "${TENSOR_NO_MODIFY_PATH:-0}" != 1 ]]; then
    "$tensor_uv" tool update-shell
fi
printf '\nTensor is installed in %s\n' "$tensor_install_root"
printf 'Open a new terminal, or run:\n  export PATH="%s/bin:$PATH"\n  tensor --help\n' "$tensor_install_root"
