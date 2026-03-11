param(
    [string]$Scene = "",
    [string]$Manifest = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [switch]$AutoCompileGmDag,
    [int]$GmDagResolution = 512,
    [int]$TotalSteps = 0,
    [string]$CheckpointDir = "checkpoints/all_night",
    [string]$ResumeCheckpoint = "",
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [int]$MinibatchSize = 64,
    [int]$PpoEpochs = 1,
    [double]$ExistentialTax = -0.02,
    [double]$EntropyCoeff = 0.02,
    [double]$LearningRate = 5e-4,
    [int]$BpttLen = 8,
    [int]$RolloutLength = 512,
    [int]$CheckpointEvery = 25000,
    [string]$LogDir = "scripts/logs/all_night",
    [string]$PythonVersion = "3.12"
)

# Standard Ghost-Matrix Fleet Size
$NumActors = 4

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Stop-ProcessTreeById {
    param([int]$ProcessId)

    if ($ProcessId -le 0) {
        return
    }

    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return
    }

    try {
        & taskkill /PID $ProcessId /T /F *> $null
    }
    catch {
    }
}

function Initialize-CudaEnvironment {
    if (-not $env:CUDA_HOME -and $env:CUDA_PATH) {
        $env:CUDA_HOME = $env:CUDA_PATH
    }
    if (-not $env:CUDA_PATH -and $env:CUDA_HOME) {
        $env:CUDA_PATH = $env:CUDA_HOME
    }

    if (-not $env:CUDA_HOME) {
        foreach ($candidate in @(
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
        )) {
            if (Test-Path $candidate) {
                $env:CUDA_HOME = $candidate
                $env:CUDA_PATH = $candidate
                break
            }
        }
    }

    if (-not $env:CUDA_HOME) {
        throw "CUDA_HOME could not be resolved. Install the CUDA toolkit or set CUDA_HOME/CUDA_PATH before launching canonical sdfdag training."
    }

    $cudaBin = Join-Path $env:CUDA_HOME "bin"
    $cudaNvvp = Join-Path $env:CUDA_HOME "libnvvp"
    $env:PATH = "$cudaBin;$cudaNvvp;$env:PATH"
}

function Stop-NaviProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*navi-environment*" -or
            $_.CommandLine -like "*navi-actor*"
        )
    } | ForEach-Object {
        Stop-ProcessTreeById -ProcessId $_.ProcessId
    }

    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        $held = netstat -ano 2>$null | Select-String "5559|5560|5557"
        if (-not $held) { break }
        foreach ($line in $held) {
            if ($line -match '\s(\d+)\s*$') {
                $procId = [int]$Matches[1]
                if ($procId -gt 0) {
                    Stop-ProcessTreeById -ProcessId $procId
                }
            }
        }
        Start-Sleep -Seconds 1
    }
}

$repoRoot = Get-RepoRoot
Initialize-CudaEnvironment

$resolvedGmDagFile = if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) {
    (Resolve-Path $GmDagFile).Path
} else {
    ""
}

$resolvedScene = if (-not [string]::IsNullOrWhiteSpace($Scene)) {
    (Resolve-Path $Scene).Path
} else {
    ""
}

$resolvedManifest = if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
    (Resolve-Path $Manifest).Path
} else {
    ""
}

$resolvedCorpusRoot = if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
    (Resolve-Path $CorpusRoot).Path
} else {
    ""
}

$resolvedGmDagRoot = if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) {
    (Resolve-Path $GmDagRoot).Path
} else {
    ""
}

foreach ($dir in @($CheckpointDir, $LogDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "========================================================"
Write-Host "  Navi All-Night Continuous Training"
Write-Host "  Corpus     : $(if ($resolvedGmDagFile) { $resolvedGmDagFile } elseif ($resolvedScene) { $resolvedScene } elseif ($resolvedManifest) { $resolvedManifest } elseif ($resolvedCorpusRoot) { $resolvedCorpusRoot } else { 'auto-discovered canonical corpus' })"
Write-Host "  Steps      : $(if ($TotalSteps -le 0) { 'continuous until stopped' } else { "$TotalSteps (Total)" })"
Write-Host "  Actors     : $NumActors (Standard Fleet)"
Write-Host "  Resolution : ${AzimuthBins}x${ElevationBins}"
Write-Host "  Runtime    : sdfdag (canonical)"
Write-Host "  Checkpoint : $CheckpointDir (every $CheckpointEvery)"
Write-Host "  Optimizer  : LR=$LearningRate, Batch=$MinibatchSize, Epochs=$PpoEpochs"
Write-Host "  Rollout    : $RolloutLength"
Write-Host "  Reward     : Tax=$ExistentialTax, Entropy=$EntropyCoeff"
Write-Host "========================================================"
Write-Host ""

Stop-NaviProcesses

$actorArgs = @(
    "run",
    "--python", $PythonVersion,
    "--project", (Join-Path $repoRoot "projects\actor"),
    "navi-actor", "train",
    "--actors", "$NumActors",
    "--azimuth-bins", "$AzimuthBins",
    "--elevation-bins", "$ElevationBins",
    "--total-steps", "$TotalSteps",
    "--checkpoint-every", "$CheckpointEvery",
    "--checkpoint-dir", $CheckpointDir,
    "--minibatch-size", "$MinibatchSize",
    "--ppo-epochs", "$PpoEpochs",
    "--existential-tax", "$ExistentialTax",
    "--entropy-coeff", "$EntropyCoeff",
    "--learning-rate", "$LearningRate",
    "--bptt-len", "$BpttLen",
    "--rollout-length", "$RolloutLength",
    "--compile-resolution", "$GmDagResolution",
    "--shuffle"
)

if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagFile)) {
    $actorArgs += @("--gmdag-file", $resolvedGmDagFile)
}
elseif (-not [string]::IsNullOrWhiteSpace($resolvedScene)) {
    $actorArgs += @("--scene", $resolvedScene)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedManifest)) {
    $actorArgs += @("--manifest", $resolvedManifest)
}
if (-not [string]::IsNullOrWhiteSpace($resolvedCorpusRoot)) {
    $actorArgs += @("--corpus-root", $resolvedCorpusRoot)
}
if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagRoot)) {
    $actorArgs += @("--gmdag-root", $resolvedGmDagRoot)
}
if ($AutoCompileGmDag) {
    $actorArgs += "--force-corpus-refresh"
}

if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $actorArgs += @("--checkpoint", $ResumeCheckpoint)
}

$logOut = Join-Path $LogDir "train_continuous.out.log"
$logErr = Join-Path $LogDir "train_continuous.err.log"

Write-Host "  Starting continuous training engine..."
Write-Host "  Logs: $logErr"

Start-Process -FilePath "uv" -ArgumentList $actorArgs `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -NoNewWindow -Wait
