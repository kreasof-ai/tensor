# Build a pinned OpenCL comparison toolchain in the workspace; no driver installation.
param([string]$BuildRoot = 'build')
$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $repositoryRoot
$buildDirectory = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $BuildRoot))
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) { $cmakeBinary = $cmakeCommand.Source }
else {
    $vswhere = 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
    $visualStudio = & $vswhere -latest -products '*' -property installationPath
    $cmakeBinary = Join-Path $visualStudio 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
}
function Run-CMake([string[]]$Arguments) {
    & $cmakeBinary @Arguments
    if ($LASTEXITCODE -ne 0) { throw "CMake failed: $LASTEXITCODE" }
}
function Get-PinnedSource([string]$Name, [string]$Url, [string]$Commit) {
    $sourceDirectory = Join-Path $buildDirectory $Name
    if (!(Test-Path -LiteralPath $sourceDirectory)) {
        & git init $sourceDirectory | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
        & git -C $sourceDirectory remote add origin $Url | Out-Host
        & git -C $sourceDirectory fetch --depth 1 origin $Commit | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
        & git -C $sourceDirectory checkout --detach FETCH_HEAD | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'git checkout failed' }
    }
    $actualCommit = & git -C $sourceDirectory rev-parse HEAD
    $changes = & git -C $sourceDirectory status --porcelain
    if ($actualCommit -ne $Commit -or $changes) { throw "Expected clean pinned source: $sourceDirectory" }
    return $sourceDirectory.Replace('\', '/')
}
$clblastSource = Get-PinnedSource 'clblast-source' 'https://github.com/CNugteren/CLBlast.git' '2a081972b20911ddf76a6b40df717c7d0c181268'
$openclHeaders = Get-PinnedSource 'opencl-headers' 'https://github.com/KhronosGroup/OpenCL-Headers.git' '30bc20a8e90468e231d7c639805ae61ad1fefa4f'
$openclLoader = Get-PinnedSource 'opencl-icd-source' 'https://github.com/KhronosGroup/OpenCL-ICD-Loader.git' '5192c84f8059e5f703e5452929b613f9487f6e4c'
$loaderBuild = (Join-Path $buildDirectory 'opencl-icd-build').Replace('\', '/')
$clblastBuild = (Join-Path $buildDirectory 'clblast-build').Replace('\', '/')
$probeBuild = (Join-Path $buildDirectory 'clblast-probe-build').Replace('\', '/')
$generatorArguments = @('-G', 'Visual Studio 18 2026', '-A', 'x64')
Run-CMake (@('-S', $openclLoader, '-B', $loaderBuild) + $generatorArguments + @("-DOPENCL_ICD_LOADER_HEADERS_DIR=$openclHeaders", '-DBUILD_TESTING=OFF'))
Run-CMake (@('--build', $loaderBuild, '--config', 'Release', '--target', 'OpenCL', '--parallel', '6'))
Run-CMake (@('-S', $clblastSource, '-B', $clblastBuild) + $generatorArguments + @("-DOPENCL_INCLUDE_DIRS=$openclHeaders", "-DOPENCL_LIBRARIES=$loaderBuild/Release/OpenCL.lib", '-DTUNERS=ON', '-DCLIENTS=OFF', '-DTESTS=OFF', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'))
Run-CMake (@('--build', $clblastBuild, '--config', 'Release', '--target', 'clblast', 'clblast_tuner_xgemm', 'clblast_tuner_xgemm_direct', '--parallel', '6'))
Run-CMake (@('-S', 'benchmarks/inference/clblast_probe', '-B', $probeBuild) + $generatorArguments + @("-DCLBLAST_SOURCE=$clblastSource", "-DOPENCL_HEADERS=$openclHeaders", "-DCLBLAST_LIBRARY=$clblastBuild/Release/clblast.lib"))
Run-CMake (@('--build', $probeBuild, '--config', 'Release', '--parallel', '6'))
