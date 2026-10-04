<#
.SYNOPSIS
    Set up the Tensor development environment on Windows.

.DESCRIPTION
    Synchronizes the locked compiler and development environment with Python 3.12.

    Python 3.12 specifically, not 3.13/3.14: tilelang declares
    `torch-c-dlpack-ext; python_version < "3.14"`, so on 3.14 the DLPack tensor
    bridge is silently dropped. See docs/research/phase0-ground-truth.md section 4.

.PARAMETER Force
    Recreate the virtual environment even if it already exists.

.EXAMPLE
    .\tools\bootstrap.ps1
#>
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repoRoot '.venv'
$pythonVersion = '3.12'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it with: winget install astral-sh.uv"
}

if ($Force -and (Test-Path $venv)) {
    $resolvedVenv = (Resolve-Path -LiteralPath $venv).Path
    $expectedVenv = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '.venv'))
    if ($resolvedVenv -ne $expectedVenv) { throw "Unexpected environment path: $resolvedVenv" }
    Write-Host "Removing existing environment at $venv" -ForegroundColor Yellow
    # Move aside rather than hard-delete, so a failed setup is recoverable.
    $bak = "$venv.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Move-Item -LiteralPath $resolvedVenv -Destination $bak
    Write-Host "  -> moved to $bak"
}

Write-Host "Synchronizing the locked development environment..." -ForegroundColor Cyan
Push-Location $repoRoot
try {
    uv sync --locked --python $pythonVersion
    if ($LASTEXITCODE -ne 0) { throw "locked dependency sync failed" }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Environment ready. Verify with:" -ForegroundColor Green
Write-Host "  $venv\Scripts\python.exe -m tensor --version"
Write-Host "  $venv\Scripts\python.exe -m pytest"
Write-Host "Build your first kernel: docs/guides/quickstart.md"
