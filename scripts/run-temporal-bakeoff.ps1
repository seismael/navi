<#
.SYNOPSIS
  Run temporal-core bake-off benchmarks and persist JSON artifacts.

.DESCRIPTION
  Executes `projects/actor/scripts/bench_temporal_backends.py` with configurable
  candidates and writes a timestamped JSON result under:
  `artifacts/benchmarks/temporal/`.

  This is migration tooling only and does not alter production runtime.

.EXAMPLE
  ./scripts/run-temporal-bakeoff.ps1

.EXAMPLE
    ./scripts/run-temporal-bakeoff.ps1 -SkipMamba -Candidates "gru,lstm" -Batch 16 -SeqLen 128 -AllowCpuDiagnostic
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Candidates = "mamba2,gru,lstm",
    [switch]$SkipMamba,
    [int]$Batch = 16,
    [int]$SeqLen = 128,
    [int]$DModel = 128,
    [int]$Repeats = 40,
    [int]$Warmup = 10,
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

function Resolve-Candidates {
    param(
        [string]$RawCandidates,
        [bool]$SkipMambaFlag
    )

    $parts = $RawCandidates.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
    if ($SkipMambaFlag) {
        $parts = $parts | Where-Object { $_ -ne "mamba2" }
    }

    $valid = @("mamba2", "gru", "lstm")
    foreach ($name in $parts) {
        if ($valid -notcontains $name) {
            throw "Invalid candidate '$name'. Allowed: mamba2,gru,lstm"
        }
    }

    if ($parts.Count -eq 0) {
        throw "No candidates selected after filtering"
    }

    return ($parts -join ",")
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

$selectedCandidates = Resolve-Candidates -RawCandidates $Candidates -SkipMambaFlag $SkipMamba.IsPresent

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
    "--candidates", $selectedCandidates,
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
    throw "Temporal bake-off execution failed with exit code $LASTEXITCODE"
}

if ([string]::IsNullOrWhiteSpace($jsonText)) {
    throw "Temporal bake-off returned empty output"
}

$jsonText | Set-Content -Path $outJson -Encoding UTF8

$results = $jsonText | ConvertFrom-Json
$ranked = @($results | Where-Object { $_.available -eq $true -and $null -ne $_.tokens_per_second } | Sort-Object tokens_per_second -Descending)

Write-Host ""
Write-Host "Temporal bake-off artifact: $outJson"
Write-Host "Benchmark device: $Device"
if ($ranked.Count -eq 0) {
    Write-Host "No available candidates produced tokens_per_second."
    exit 1
}

Write-Host "Ranked candidates (tokens_per_second):"
$ranked | ForEach-Object {
    Write-Host ("  {0,-8} tps={1,10:N1} fwd_ms={2,8:N3} step_ms={3,8:N3}" -f $_.candidate, $_.tokens_per_second, $_.forward_ms, $_.step_ms)
}

$winner = $ranked[0]
Write-Host "Winner by throughput: $($winner.candidate)"
