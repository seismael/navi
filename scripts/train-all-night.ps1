param(
    [string]$Scene = "",
    [string]$Manifest = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [switch]$AutoCompileGmDag,
    [int]$GmDagResolution = 512,
    [int]$TotalSteps = 0,
    [string]$CheckpointDir = "",
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
    [string]$LogDir = "",
    [string]$PythonVersion = "3.12"
)

# Standard Ghost-Matrix Fleet Size
$NumActors = 4
$ActorTelemetryPort = 5557

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

function Wait-ForTrainingReady {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$Port,
        [string[]]$LogFiles,
        [int]$TimeoutSeconds = 180
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $Process) {
            try {
                $Process.Refresh()
                if ($Process.HasExited) {
                    return @{
                        Ready = $false
                        Reason = "process-exited"
                        ExitCode = $Process.ExitCode
                    }
                }
            }
            catch {
            }
        }

        $hit = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+\d+\s*$"
        if ($hit) {
            return @{
                Ready = $true
                Reason = "port-bound"
            }
        }

        foreach ($logFile in $LogFiles) {
            if (-not $logFile -or -not (Test-Path $logFile)) {
                continue
            }

            $tail = Get-Content $logFile -Tail 40 -ErrorAction SilentlyContinue
            if (-not $tail) {
                continue
            }

            $tailText = ($tail -join "`n")
            if (
                $tailText -match "Async telemetry worker started on .*:$Port" -or
                $tailText -match "Canonical PPO trainer started: .* pub=tcp://localhost:$Port"
            ) {
                return @{
                    Ready = $true
                    Reason = "log-ready"
                }
            }
        }

        Start-Sleep -Milliseconds 500
    }

    return @{
        Ready = $false
        Reason = "timeout"
    }
}

$repoRoot = Get-RepoRoot
$observabilityModule = Join-Path $repoRoot "tools\Navi.Observability.psm1"
Import-Module $observabilityModule -Force
$cleanupStartedAt = Get-Date
$cleanupSummary = Invoke-NaviGeneratedCleanup -RepoRoot $repoRoot
$runContext = New-NaviRunContext -RepoRoot $repoRoot -Profile "all-night-train" -BaseRelativeRoot "artifacts\runs"
Write-NaviRunManifest -RunContext $runContext -Metadata ([ordered]@{
    actors = $NumActors
    cleanup_removed = @($cleanupSummary.removed)
}) -FileName "train-all-night-wrapper.json"
Write-NaviPhaseMetric -RunContext $runContext -Operation "wrapper_cleanup" -StartedAt $cleanupStartedAt -Metadata ([ordered]@{
    removed_count = @($cleanupSummary.removed).Count
}) | Out-Null
Initialize-CudaEnvironment

if ([string]::IsNullOrWhiteSpace($CheckpointDir)) {
    $CheckpointDir = $runContext.CheckpointRoot
}
if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = $runContext.LogRoot
}

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
Write-Host "  Run ID     : $($runContext.RunId)"
Write-Host "  Run Root   : $($runContext.RunRoot)"
Write-Host "  Checkpoint : $CheckpointDir (every $CheckpointEvery)"
Write-Host "  Optimizer  : LR=$LearningRate, Batch=$MinibatchSize, Epochs=$PpoEpochs"
Write-Host "  Rollout    : $RolloutLength"
Write-Host "  Reward     : Tax=$ExistentialTax, Entropy=$EntropyCoeff"
Write-Host "  Metrics    : $($runContext.MetricsRoot)"
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
    if (-not [System.IO.Path]::IsPathRooted($ResumeCheckpoint)) {
        $ResumeCheckpoint = Join-Path $repoRoot "projects\actor\$ResumeCheckpoint"
    }
    if (-not (Test-Path $ResumeCheckpoint)) {
        throw "Checkpoint file not found: $ResumeCheckpoint"
    }
    $actorArgs += @("--checkpoint", $ResumeCheckpoint)
}

$logOut = Join-Path $LogDir "train_continuous.out.log"
$logErr = Join-Path $LogDir "train_continuous.err.log"

Write-Host "  Starting continuous training engine..."
Write-Host "  Logs: $logErr"

$trainProc = $null
trap [System.Management.Automation.BreakException] {
    Write-Host "`nInterrupted. Cleaning up..."
    if ($null -ne $trainProc -and -not $trainProc.HasExited) {
        Stop-ProcessTreeById -ProcessId $trainProc.Id
    }
    exit 1
}

try {
    $trainLaunchStartedAt = Get-Date
    $trainProc = Start-Process -FilePath "uv" -ArgumentList $actorArgs `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr `
        -NoNewWindow -PassThru

    Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_launch" -StartedAt $trainLaunchStartedAt -ProcessId $trainProc.Id -Metadata ([ordered]@{
        actors = $NumActors
        telemetry_port = $ActorTelemetryPort
    }) | Out-Null

    Write-Host "  PID: $($trainProc.Id)"
    Write-Host "  Metrics : $($runContext.MetricsRoot)"
    Write-Host "  Verifying training telemetry readiness ($ActorTelemetryPort)..."

    $trainingReadyStartedAt = Get-Date
    $trainingReady = Wait-ForTrainingReady -Process $trainProc -Port $ActorTelemetryPort -LogFiles @($logErr) -TimeoutSeconds 180
    Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_ready" -StartedAt $trainingReadyStartedAt -ProcessId $(if ($null -ne $trainProc) { $trainProc.Id } else { 0 }) -Metadata ([ordered]@{
        ready = [bool]$trainingReady.Ready
        reason = [string]$trainingReady.Reason
        exit_code = if ($trainingReady.ContainsKey('ExitCode')) { $trainingReady.ExitCode } else { $null }
        telemetry_port = $ActorTelemetryPort
    }) | Out-Null

    if (-not $trainingReady.Ready) {
        if ($trainingReady.Reason -eq "process-exited") {
            throw "Continuous training exited before telemetry became ready (exit code $($trainingReady.ExitCode))."
        }
        throw "Continuous training telemetry failed readiness check ($ActorTelemetryPort, reason=$($trainingReady.Reason))."
    }

    $trainExitStartedAt = Get-Date
    Wait-Process -Id $trainProc.Id
    $trainProc.Refresh()
    $exitCode = if ($null -eq $trainProc.ExitCode) { 0 } else { [int]$trainProc.ExitCode }
    Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_exit" -StartedAt $trainExitStartedAt -Metadata ([ordered]@{
        exit_code = $exitCode
    }) | Out-Null

    if ($exitCode -ne 0) {
        throw "Continuous training exited with code $exitCode. See $logErr"
    }
}
finally {
    if ($null -ne $trainProc -and -not $trainProc.HasExited) {
        Write-Host "Stopping continuous training engine (PID $($trainProc.Id))..."
        Stop-ProcessTreeById -ProcessId $trainProc.Id
    }
}
