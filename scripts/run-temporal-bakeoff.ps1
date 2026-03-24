<#!
.SYNOPSIS
    Run the canonical temporal microbenchmark bakeoff and persist a JSON artifact.

.DESCRIPTION
    Executes `projects/actor/scripts/bench_temporal_backends.py` for the
    selected temporal candidates and writes a timestamped JSON result under:
    `artifacts/benchmarks/temporal/`.

    This is diagnostic profiling only and does not alter production runtime.

.EXAMPLE
    ./scripts/run-temporal-bakeoff.ps1

.EXAMPLE
    ./scripts/run-temporal-bakeoff.ps1 -Device cpu -AllowCpuDiagnostic
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Batch = 16,
    [int]$SeqLen = 128,
    [int]$DModel = 128,
    [int]$Repeats = 40,
    [int]$Warmup = 10,
    [ValidateSet("gru", "mambapy", "mamba2")]
    [string[]]$Candidates = @("mamba2"),
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$AllowCpuDiagnostic,
    [string]$ActorPython = "",
    [string]$OutputDir = "artifacts/benchmarks/temporal"
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Resolve-ActorPython {
    param([string]$ProjectRoot)

    $winPython = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
    if (Test-Path $winPython) {
        return $winPython
    }

    $linuxPython = Join-Path $ProjectRoot ".venv/bin/python"
    if (Test-Path $linuxPython) {
        return $linuxPython
    }

    return ""
}

$repoRoot = Get-RepoRoot
$actorProject = Join-Path $repoRoot "projects/actor"
$harness = Join-Path $repoRoot "projects/actor/scripts/bench_temporal_backends.py"

if ($Device -eq "cpu" -and -not $AllowCpuDiagnostic.IsPresent) {
    throw "CPU benchmark runs are diagnostics only. Use -Device cuda for canonical selection, or pass -AllowCpuDiagnostic to run CPU diagnostics explicitly."
}

if (-not (Test-Path $harness)) {
    throw "Benchmark harness not found: $harness"
}

$selectedCandidates = @($Candidates)

$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
}
else {
    Join-Path $repoRoot $OutputDir
}

if (-not (Test-Path $resolvedOutputDir)) {
    New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outJson = Join-Path $resolvedOutputDir "temporal-bakeoff-$timestamp.json"

$scriptArgs = @(
    $harness,
    "--candidates"
)
$scriptArgs += $selectedCandidates
$scriptArgs += @(
    "--batch", "$Batch",
    "--seq-len", "$SeqLen",
    "--d-model", "$DModel",
    "--repeats", "$Repeats",
    "--warmup", "$Warmup",
    "--device", $Device,
    "--json"
)

$actorPythonResolved = $ActorPython
if ([string]::IsNullOrWhiteSpace($actorPythonResolved)) {
    $actorPythonResolved = Resolve-ActorPython -ProjectRoot $actorProject
}

$oldPythonPath = $env:PYTHONPATH
$actorSrc = Join-Path $actorProject "src"
$pathSep = [System.IO.Path]::PathSeparator
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
    $env:PYTHONPATH = $actorSrc
}
else {
    $env:PYTHONPATH = "$actorSrc$pathSep$oldPythonPath"
}

try {
    if ([string]::IsNullOrWhiteSpace($actorPythonResolved)) {
        Write-Host "Running bake-off via uv project environment..."
        $jsonText = uv run --project $actorProject python @scriptArgs
    }
    else {
        if (-not (Test-Path $actorPythonResolved)) {
            throw "Actor Python executable not found: $actorPythonResolved"
        }
        Write-Host "Running bake-off via actor Python: $actorPythonResolved"
        $jsonText = & $actorPythonResolved @scriptArgs
    }
}
finally {
    if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $oldPythonPath
    }
}

if ($LASTEXITCODE -ne 0) {
    if ($Device -eq "cuda") {
        Write-Host "CUDA benchmark failed. If torch is CPU-only, install CUDA wheels:" -ForegroundColor Yellow
        Write-Host "  powershell -ExecutionPolicy Bypass -File ./scripts/setup-actor-cuda.ps1" -ForegroundColor Yellow
    }
    throw "Canonical temporal profile execution failed with exit code $LASTEXITCODE"
}

if ([string]::IsNullOrWhiteSpace($jsonText)) {
    throw "Canonical temporal profile returned empty output"
}

$jsonText | Set-Content -Path $outJson -Encoding UTF8

$results = $jsonText | ConvertFrom-Json
$ranked = @($results | Where-Object { $_.available -eq $true -and $null -ne $_.tokens_per_second } | Sort-Object tokens_per_second -Descending)

Write-Host ""
Write-Host "Canonical temporal profile artifact: $outJson"
Write-Host "Benchmark device: $Device"
if ($ranked.Count -eq 0) {
    Write-Host "No selected temporal runtime produced a tokens_per_second result."
    exit 1
}

Write-Host "Measured runtime (tokens_per_second):"
$ranked | ForEach-Object {
    Write-Host ("  {0,-8} tps={1,10:N1} fwd_ms={2,8:N3} step_ms={3,8:N3}" -f $_.candidate, $_.tokens_per_second, $_.forward_ms, $_.step_ms)
}

$winner = $ranked[0]
Write-Host "Fastest measured candidate: $($winner.candidate)"
