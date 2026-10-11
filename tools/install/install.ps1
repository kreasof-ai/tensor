# Install Tensor's CLI in an isolated, user-owned Python environment.
& {
    $ErrorActionPreference = 'Stop'
    $installRoot = if ($env:TENSOR_INSTALL_DIR) { $env:TENSOR_INSTALL_DIR } else { Join-Path $env:USERPROFILE '.tensor' }
    $revision = if ($env:TENSOR_REF) { $env:TENSOR_REF } else { 'main' }
    $source = if ($env:TENSOR_SOURCE) { $env:TENSOR_SOURCE } else { "https://github.com/kreasof-ai/tensor/archive/$revision.tar.gz" }
    $extras = $env:TENSOR_EXTRAS
    if ($extras -and $extras -notin @('compiler', 'webgpu', 'compiler,webgpu', 'webgpu,compiler')) {
        throw 'TENSOR_EXTRAS must be compiler, webgpu, or compiler,webgpu.'
    }
    $installRoot = [System.IO.Path]::GetFullPath($installRoot)
    $bin = Join-Path $installRoot 'bin'
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    $uv = Join-Path $bin 'uv.exe'
    $savedEnvironment = @{}
    foreach ($name in @('UV_UNMANAGED_INSTALL', 'UV_TOOL_DIR', 'UV_TOOL_BIN_DIR', 'UV_PYTHON_INSTALL_DIR')) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    try {
        if (-not (Test-Path -LiteralPath $uv)) {
            $existingUv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
            if ($existingUv) {
                $uv = $existingUv.Source
            } else {
                $env:UV_UNMANAGED_INSTALL = $bin
                Invoke-Expression (Invoke-RestMethod 'https://astral.sh/uv/0.12.9/install.ps1')
                if (-not (Test-Path -LiteralPath $uv)) { throw 'uv bootstrap failed.' }
            }
        }
        $env:UV_TOOL_DIR = Join-Path $installRoot 'tools'
        $env:UV_TOOL_BIN_DIR = $bin
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $installRoot 'python'
        $package = if ($extras) { "tensor-workspace[$extras]" } else { 'tensor-workspace' }
        & $uv tool install --managed-python --python 3.12 --reinstall --force "$package @ $source"
        if ($LASTEXITCODE -ne 0) { throw 'Tensor installation failed.' }
        & (Join-Path $bin 'tensor.exe') --version
        if ($LASTEXITCODE -ne 0) { throw 'Tensor CLI validation failed.' }
        if ($env:TENSOR_NO_MODIFY_PATH -ne '1') {
            & $uv tool update-shell
            if ($LASTEXITCODE -ne 0) { throw 'Tensor installed, but PATH setup failed.' }
        }
        Write-Host "`nTensor is installed in $installRoot"
        Write-Host "Open a new terminal and run: tensor --help"
    } finally {
        foreach ($name in $savedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
        }
    }
}
