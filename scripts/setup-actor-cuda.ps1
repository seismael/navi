<#
.SYNOPSIS
  Install CUDA-enabled PyTorch stack into actor virtual environment.

.DESCRIPTION
  Uses the official PyTorch CUDA wheel index to install GPU-enabled
  `torch`, `torchvision`, and `torchaudio` into `projects/actor/.venv`.

  This script targets native Windows first and is also usable on Linux/WSL2
  when PowerShell is available.

.EXAMPLE
  ./scripts/setup-actor-cuda.ps1

.EXAMPLE
  ./scripts/setup-actor-cuda.ps1 -CudaTag cu121 -TorchVersion 2.5.1
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$CudaTag = "cu121",
  [string]$TorchVersion = "2.5.1",
  [switch]$SkipActorSync
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

$repoRoot = Get-RepoRoot
$actorProject = Join-Path $repoRoot "projects/actor"
$pyWin = Join-Path $actorProject ".venv/Scripts/python.exe"
$pyLinux = Join-Path $actorProject ".venv/bin/python"
$py = ""

if (Test-Path $pyWin) {
    $py = $pyWin
}
elseif (Test-Path $pyLinux) {
    $py = $pyLinux
}
else {
    throw "Actor virtualenv Python not found. Run: uv sync --project projects/actor"
}

$indexUrl = "https://download.pytorch.org/whl/$CudaTag"

if (-not $env:CUDA_HOME -and $env:CUDA_PATH) {
  $env:CUDA_HOME = $env:CUDA_PATH
}
if (-not $env:CUDA_PATH -and $env:CUDA_HOME) {
  $env:CUDA_PATH = $env:CUDA_HOME
}
if (-not $env:CUDA_HOME) {
  $candidate = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
  if (Test-Path $candidate) {
    $env:CUDA_HOME = $candidate
    $env:CUDA_PATH = $candidate
  }
}

if ($env:CUDA_HOME) {
  $cudaBin = Join-Path $env:CUDA_HOME "bin"
  $cudaNvvp = Join-Path $env:CUDA_HOME "libnvvp"
  $env:PATH = "$cudaBin;$cudaNvvp;$env:PATH"
  Write-Host "  CUDA_HOME: $($env:CUDA_HOME)"
}
else {
  Write-Host "WARNING: CUDA_HOME is unset. Native CUDA torch install may fail." -ForegroundColor Yellow
}

Write-Host "Installing CUDA PyTorch packages into actor environment..."
Write-Host "  Python : $py"
Write-Host "  Index  : $indexUrl"
Write-Host "  Torch  : $TorchVersion"

uv pip install --python $py --index-url $indexUrl --upgrade --force-reinstall `
    "torch==$TorchVersion" `
    "torchvision" `
    "torchaudio"

if ($LASTEXITCODE -ne 0) {
    throw "CUDA package install failed with exit code $LASTEXITCODE"
}

Write-Host "CUDA package install completed."
Write-Host "Run CUDA preflight:"
Write-Host "  $py .\\scripts\\check_gpu.py"

Write-Host "Running CUDA preflight now..."
& $py (Join-Path $repoRoot "scripts/check_gpu.py")
if ($LASTEXITCODE -ne 0) {
  throw "CUDA preflight failed with exit code $LASTEXITCODE. Check GPU compute capability compatibility."
}

if (-not $SkipActorSync) {
    Write-Host "Syncing actor project dependencies (canonical mambapy runtime)..."
    uv sync --project $actorProject --python 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "Actor dependency sync failed with exit code $LASTEXITCODE"
    }
}
