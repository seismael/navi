<# .SYNOPSIS
  Launch the full Navi Ghost-Matrix stack.

    Three modes:
        1. Legacy Inference (default) — canonical sdfdag Environment + Actor + Dashboard as 3 processes.
        2. Training  (-Train)         — canonical sdfdag train (in-process env+actor), with dashboard attach optional.
        3. Inference (-Infer)         — canonical sdfdag infer (in-process env+actor), with dashboard by default.

.EXAMPLE
    # Inference on the canonical compiled runtime
    .\run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\apartment_1.gmdag

    # In-process inference with a trained checkpoint (dashboard by default)
    .\run-ghost-stack.ps1 -Infer -Checkpoint ".\artifacts\runs\<run_id>\checkpoints\policy_step_0010000.pt"

    # In-process inference on a specific dataset with deterministic actions
    .\run-ghost-stack.ps1 -Infer -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt" -Datasets "ai-habitat_ReplicaCAD_baked_lighting" -Deterministic

    # In-process inference for 10000 steps without dashboard
    .\run-ghost-stack.ps1 -Infer -Checkpoint ".\my_model.pt" -TotalSteps 10000 -NoDashboard

    # Canonical PPO training on the full discovered corpus without the dashboard
    # (auto-continues from artifacts\models\latest.pt if available)
    .\run-ghost-stack.ps1 -Train

    # Canonical PPO training on the full discovered corpus with 8 actors and dashboard
    .\run-ghost-stack.ps1 -Train -Actors 8 -WithDashboard

    # Canonical PPO training with an explicit passive dashboard attach
    .\run-ghost-stack.ps1 -Train -WithDashboard

    # Resume from an explicit prior run checkpoint
    .\run-ghost-stack.ps1 -Train -TotalSteps 500000 -Checkpoint ".\artifacts\runs\<run_id>\checkpoints\policy_step_0010000.pt"

    # Inference using the latest promoted model (auto-selected)
    .\run-ghost-stack.ps1 -Infer

    # Train on only the best dataset (ReplicaCAD baked lighting)
    .\run-ghost-stack.ps1 -Train -Datasets "ai-habitat_ReplicaCAD_baked_lighting" -WithDashboard
#>
param(
    # ── Mode ──
    [switch]$Train,
    [switch]$Infer,

    # ── Common ──
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [string]$Scene = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [switch]$AutoCompileGmDag,
    [int]$GmDagResolution = 512,
    [string]$PythonVersion = "3.12",
    [ValidateRange(1, 65535)]
    [int]$Actors = 4,
    [switch]$NoPreKill,
    [switch]$NoDashboard,
    [switch]$WithDashboard,

    # ── Training params ──
    [string]$Manifest = "",
    [string]$Datasets = "",
    [string]$ExcludeDatasets = "",
    [int]$TotalSteps = 0,
    [int]$TotalEpisodes = 0,
    [switch]$Deterministic,
    [int]$CheckpointEvery = 25000,
    [string]$CheckpointDir = "checkpoints",
    [string]$Checkpoint = "",
    [int]$LogEvery = 100,
    [int]$RolloutLength = 256,
    [int]$ActorTelemetryPort = 5557,
    [ValidateSet("gru", "mambapy", "mamba2")]
    [string]$TemporalCore = "mamba2",

    # ── Inference-mode ZMQ addresses ──
    [string]$EnvironmentPub = "tcp://*:5559",
    [string]$EnvironmentRep = "tcp://*:5560",
    [string]$ActorSub = "tcp://localhost:5559",
    [string]$ActorPub = "tcp://*:5557",
    [string]$ActorStepEndpoint = "tcp://localhost:5560",
    [string]$ActorPolicyCheckpoint = "",

    # ── Orchestration ──
    [switch]$Foreground
)

# Requested Ghost-Matrix Fleet Size
# Inference defaults to 1 actor unless user explicitly sets -Actors
if ($Infer -and -not $PSBoundParameters.ContainsKey('Actors')) {
    $NumActors = 1
} else {
    $NumActors = $Actors
}

$ErrorActionPreference = "Stop"

trap [System.Management.Automation.BreakException] {
    Write-Host "`nInterrupted. Cleaning up Navi processes..."
    Stop-NaviProcesses
    exit 1
}

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

function Stop-ListenersOnPorts {
    param([int[]]$Ports)

    $pids = @()
    foreach ($port in $Ports) {
        $lines = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+(\d+)\s*$"
        foreach ($line in $lines) {
            $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
            if ($parts.Length -ge 5) {
                $procId = 0
                if ([int]::TryParse($parts[-1], [ref]$procId)) {
                    $pids += $procId
                }
            }
        }
    }

    foreach ($procId in ($pids | Sort-Object -Unique)) {
        Stop-ProcessTreeById -ProcessId $procId
    }
}

function Stop-NaviProcesses {
    $patterns = @(
        "*navi-environment*",
        "*navi-actor*",
        "*navi-auditor*"
    )
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $cmd -and ($patterns | Where-Object { $cmd -like $_ })
    }

    foreach ($proc in $targets) {
        Stop-ProcessTreeById -ProcessId $proc.ProcessId
    }

    # Also kill any detached child process still holding Navi ports.
    $ports = @(5559, 5560, 5557)
    if ($ActorTelemetryPort -ne 5557) {
        $ports += $ActorTelemetryPort
    }
    Stop-ListenersOnPorts -Ports $ports
}

function Start-BackgroundUv {
    param(
        [string]$RepoRoot,
        [string[]]$UvArgs,
        [string]$StdOutFile,
        [string]$StdErrFile
    )

    $logDir = Split-Path $StdOutFile -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    # Force unbuffered Python output so logs appear in real time
    $env:PYTHONUNBUFFERED = "1"
    return Start-Process -FilePath "uv" -ArgumentList $UvArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $StdOutFile -RedirectStandardError $StdErrFile -PassThru
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

    if ($env:CUDA_HOME) {
        $cudaBin = Join-Path $env:CUDA_HOME "bin"
        $cudaNvvp = Join-Path $env:CUDA_HOME "libnvvp"
        $env:PATH = "$cudaBin;$cudaNvvp;$env:PATH"
        Write-Host "CUDA_HOME: $($env:CUDA_HOME)"
    }
    else {
        throw "CUDA_HOME could not be resolved. Install the CUDA toolkit or set CUDA_HOME/CUDA_PATH before launching sdfdag runtime."
    }
}

function Resolve-SdfDagAsset {
    param(
        [string]$RepoRoot,
        [string]$GmDagFile,
        [string]$Scene,
        [switch]$AutoCompile,
        [int]$Resolution,
        [string]$PythonVersion
    )

    Initialize-CudaEnvironment

    if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) {
        $resolved = Resolve-Path $GmDagFile
        return $resolved.Path
    }

    $compiledCorpusRoot = Join-Path $RepoRoot "artifacts\gmdag\corpus"
    if ((-not $AutoCompile) -and [string]::IsNullOrWhiteSpace($Scene) -and (Test-Path $compiledCorpusRoot)) {
        $compiledCandidates = Get-ChildItem -Path $compiledCorpusRoot -Recurse -File -Filter "*.gmdag" | Sort-Object FullName
        if ($compiledCandidates.Count -gt 0) {
            return $compiledCandidates[0].FullName
        }
    }

    if (-not $AutoCompile) {
        throw "GmDagFile is required unless a compiled corpus asset already exists or -AutoCompileGmDag is set with -Scene."
    }
    if ([string]::IsNullOrWhiteSpace($Scene)) {
        throw "Scene is required for -AutoCompileGmDag on the canonical runtime."
    }

    $sourcePath = (Resolve-Path $Scene).Path
    $cacheDir = Join-Path $RepoRoot "artifacts\gmdag\corpus\manual"
    if (-not (Test-Path $cacheDir)) {
        New-Item -ItemType Directory -Path $cacheDir | Out-Null
    }
    $outputPath = Join-Path $cacheDir (([System.IO.Path]::GetFileNameWithoutExtension($sourcePath)) + ".gmdag")

    Write-Host "Compiling canonical gmdag cache..."
    & uv run --python $PythonVersion --project (Join-Path $RepoRoot "projects\environment") `
        navi-environment compile-gmdag `
        --source $sourcePath `
        --output $outputPath `
        --resolution $Resolution

    if ($LASTEXITCODE -ne 0) {
        throw "gmdag compilation failed with exit code $LASTEXITCODE"
    }

    return $outputPath
}

function Wait-ForPorts {
    param(
        [int[]]$Ports,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $allReady = $true
        foreach ($port in $Ports) {
            $hit = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+\d+\s*$"
            if (-not $hit) {
                $allReady = $false
                break
            }
        }
        if ($allReady) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
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

        if (Wait-ForPorts -Ports @($Port) -TimeoutSeconds 1) {
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
$runProfile = if ($Train) { "ghost-stack-train" } elseif ($Infer) { "ghost-stack-infer" } else { "ghost-stack-inference" }
$runContext = New-NaviRunContext -RepoRoot $repoRoot -Profile $runProfile -BaseRelativeRoot "artifacts\runs"
Write-NaviRunManifest -RunContext $runContext -Metadata ([ordered]@{
    train = [bool]$Train
    infer = [bool]$Infer
    actors = $Actors
    temporal_core = $TemporalCore
    actor_telemetry_port = $ActorTelemetryPort
    cleanup_removed = @($cleanupSummary.removed)
}) -FileName "run-ghost-stack.json"
Write-NaviPhaseMetric -RunContext $runContext -Operation "wrapper_cleanup" -StartedAt $cleanupStartedAt -Metadata ([ordered]@{
    removed_count = @($cleanupSummary.removed).Count
}) | Out-Null
$logsDir = $runContext.LogRoot

if ($CheckpointDir -eq "checkpoints") {
    $CheckpointDir = $runContext.CheckpointRoot
}

if ($Train -and $Infer) {
    throw "-Train and -Infer cannot be used together. Choose one mode."
}

$resolvedGmDagFile = ""
if ($Train -or $Infer) {
    Initialize-CudaEnvironment
    if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) {
        $resolvedGmDagFile = (Resolve-Path $GmDagFile).Path
    }
}
else {
    $resolvedGmDagFile = Resolve-SdfDagAsset -RepoRoot $repoRoot -GmDagFile $GmDagFile -Scene $Scene -AutoCompile:$AutoCompileGmDag -Resolution $GmDagResolution -PythonVersion $PythonVersion
}

if (-not $NoPreKill) {
    Write-Host "Stopping stale Navi processes..."
    Stop-NaviProcesses
    Start-Sleep -Milliseconds 500
}

# ═══════════════════════════════════════════════════════════════════
# Canonical training launch: train (in-process env+actor), dashboard optional
# ═══════════════════════════════════════════════════════════════════
if ($Train) {
    if ($NoDashboard -and $WithDashboard) {
        throw "-NoDashboard and -WithDashboard cannot be used together."
    }

    $dashboardEnabled = $WithDashboard -and (-not $NoDashboard)

    $sceneOverride = if (-not [string]::IsNullOrWhiteSpace($Scene)) {
        $Scene
    }
    else { "" }

    $resolvedScene = if (-not [string]::IsNullOrWhiteSpace($sceneOverride)) {
        (Resolve-Path $sceneOverride).Path
    }
    else {
        ""
    }

    $resolvedManifest = if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        (Resolve-Path $Manifest).Path
    }
    else {
        ""
    }

    $resolvedCorpusRoot = if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
        (Resolve-Path $CorpusRoot).Path
    }
    else {
        ""
    }

    $resolvedGmDagRoot = if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) {
        (Resolve-Path $GmDagRoot).Path
    }
    else {
        ""
    }

    # Resolve checkpoint dir to absolute
    if (-not [System.IO.Path]::IsPathRooted($CheckpointDir)) {
        $CheckpointDir = Join-Path $repoRoot "projects\actor\$CheckpointDir"
    }

    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        if (-not [System.IO.Path]::IsPathRooted($Checkpoint)) {
            $Checkpoint = Join-Path $repoRoot $Checkpoint
        }
        if (-not (Test-Path $Checkpoint)) {
            throw "Checkpoint file not found: $Checkpoint"
        }
    }
    else {
        # Auto-continue: use latest promoted model if available
        $latestModel = Join-Path $repoRoot "artifacts\models\latest.pt"
        if (Test-Path $latestModel) {
            $Checkpoint = $latestModel
            Write-Host "[ghost-stack] Auto-continuing from latest promoted model: $Checkpoint" -ForegroundColor Cyan
        }
    }

    $trainArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\actor"),
        "python", "-m", "navi_actor.cli", "train",
        "--actors", "$NumActors",
        "--temporal-core", "$TemporalCore",
        "--total-steps", $TotalSteps,
        "--actor-pub", "tcp://localhost:$ActorTelemetryPort",
        "--shuffle",
        "--checkpoint-every", $CheckpointEvery,
        "--checkpoint-dir", $CheckpointDir,
        "--log-every", $LogEvery,
        "--rollout-length", $RolloutLength,
        "--compile-resolution", $GmDagResolution,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )

    if (-not $dashboardEnabled) {
        $trainArgs += "--no-emit-observation-stream"
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagFile)) {
        $trainArgs += @("--gmdag-file", $resolvedGmDagFile)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($resolvedScene)) {
        $trainArgs += @("--scene", $resolvedScene)
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedManifest)) {
        $trainArgs += @("--manifest", $resolvedManifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($resolvedCorpusRoot)) {
        $trainArgs += @("--corpus-root", $resolvedCorpusRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagRoot)) {
        $trainArgs += @("--gmdag-root", $resolvedGmDagRoot)
    }
    if ($AutoCompileGmDag) {
        $trainArgs += "--force-corpus-refresh"
    }

    if (-not [string]::IsNullOrWhiteSpace($Datasets)) {
        $trainArgs += @("--datasets", $Datasets)
    }
    if (-not [string]::IsNullOrWhiteSpace($ExcludeDatasets)) {
        $trainArgs += @("--exclude-datasets", $ExcludeDatasets)
    }

    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $trainArgs += @("--checkpoint", $Checkpoint)
    }

    $trainLogOut = Join-Path $logsDir "train.out.log"
    $trainLogErr = Join-Path $logsDir "train.err.log"
    $trainUnifiedLog = Join-Path $repoRoot "logs\navi_actor_train.log"

    $trainProc = $null
    try {
        Write-Host "========================================================"
        Write-Host "  Navi Ghost-Matrix Training"
        Write-Host "  Backend    : sdfdag"
        Write-Host "  Run ID     : $($runContext.RunId)"
        Write-Host "  Run Root   : $($runContext.RunRoot)"
        Write-Host "  Corpus     : $(if ($resolvedGmDagFile) { $resolvedGmDagFile } elseif ($resolvedScene) { $resolvedScene } elseif ($resolvedManifest) { $resolvedManifest } elseif ($resolvedCorpusRoot) { $resolvedCorpusRoot } else { 'auto-discovered canonical corpus' })"
        Write-Host "  Actors     : $NumActors (Standard Fleet)"
        Write-Host "  Total Steps: $(if ($TotalSteps -le 0) { 'continuous until stopped' } else { $TotalSteps })"
        Write-Host "  Temporal   : $TemporalCore"
        if (-not [string]::IsNullOrWhiteSpace($Datasets)) {
            Write-Host "  Datasets   : $Datasets"
        }
        if (-not [string]::IsNullOrWhiteSpace($ExcludeDatasets)) {
            Write-Host "  Exclude DS : $ExcludeDatasets"
        }
        Write-Host "  Telemetry  : tcp://localhost:$ActorTelemetryPort"
        Write-Host "  Checkpoints: every $CheckpointEvery -> $CheckpointDir"
        if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
            Write-Host "  Resume     : $Checkpoint"
        }
        Write-Host "  Dashboard  : $(if ($dashboardEnabled) { 'enabled (passive observer)' } else { 'disabled by default for canonical training; observation stream off' })"
        Write-Host "  Metrics    : $($runContext.MetricsRoot)"
        Write-Host "========================================================"

        Write-Host "`nStarting canonical train (background)..."
        $trainLaunchStartedAt = Get-Date
        $trainProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $trainArgs -StdOutFile $trainLogOut -StdErrFile $trainLogErr
        Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_launch" -StartedAt $trainLaunchStartedAt -ProcessId $trainProc.Id -Metadata ([ordered]@{
            dashboard_enabled = [bool]$dashboardEnabled
            actors = $NumActors
            temporal_core = $TemporalCore
        }) | Out-Null
        Write-Host "  PID: $($trainProc.Id)"
        Write-Host "  Logs: $trainLogOut"
        Write-Host "        $trainLogErr"
        Write-Host "        $trainUnifiedLog"
        Write-Host "  Metrics: $($runContext.MetricsRoot)"

        Write-Host "Verifying training telemetry readiness ($ActorTelemetryPort)..."
        $trainingReadyStartedAt = Get-Date
        $trainingReady = Wait-ForTrainingReady -Process $trainProc -Port $ActorTelemetryPort -LogFiles @($trainLogErr, $trainUnifiedLog) -TimeoutSeconds 180
        Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_ready" -StartedAt $trainingReadyStartedAt -ProcessId $(if ($null -ne $trainProc) { $trainProc.Id } else { 0 }) -Metadata ([ordered]@{
            ready = [bool]$trainingReady.Ready
            reason = [string]$trainingReady.Reason
            exit_code = if ($trainingReady.ContainsKey('ExitCode')) { $trainingReady.ExitCode } else { $null }
            telemetry_port = $ActorTelemetryPort
        }) | Out-Null
        if (-not $trainingReady.Ready) {
            $reason = [string]$trainingReady.Reason
            if ($reason -eq "process-exited") {
                Write-Host "ERROR: canonical train exited before telemetry became ready (exit code $($trainingReady.ExitCode))."
            }
            else {
                Write-Host "ERROR: training telemetry failed readiness check ($ActorTelemetryPort, reason=$reason)."
            }
            Write-Host "Check logs: $trainLogErr"
            Write-Host "            $trainUnifiedLog"
            if ($null -ne $trainProc -and -not $trainProc.HasExited) {
                Stop-ProcessTreeById -ProcessId $trainProc.Id
            }
            Stop-ListenersOnPorts -Ports @($ActorTelemetryPort, 5559, 5560)
            exit 1
        }
        else {
            Write-Host "  OK: actor telemetry is ready ($($trainingReady.Reason))."
        }

        if ($dashboardEnabled) {
            # The canonical trainer only guarantees actor telemetry. The auditor
            # remains resilient when matrix/step streams are absent.
            Write-Host "`nWaiting for actor telemetry before launching dashboard..."
            $dashboardReady = Wait-ForTrainingReady -Process $trainProc -Port $ActorTelemetryPort -LogFiles @($trainLogErr, $trainUnifiedLog) -TimeoutSeconds 180
            if (-not $dashboardReady.Ready) {
                Write-Host "ERROR: Dashboard launch aborted, actor telemetry not ready ($ActorTelemetryPort, reason=$($dashboardReady.Reason))."
                if ($null -ne $trainProc -and -not $trainProc.HasExited) {
                    Stop-ProcessTreeById -ProcessId $trainProc.Id
                }
                Stop-ListenersOnPorts -Ports @($ActorTelemetryPort, 5559, 5560)
                exit 1
            }

            Write-Host "Launching Dashboard (foreground)..."
            Write-Host "  Tab = toggle manual/AI | WASD = move | ESC = quit"
            & uv run --project (Join-Path $repoRoot "projects\auditor") `
                --python $PythonVersion `
                navi-auditor dashboard `
                --actor-sub "tcp://localhost:$ActorTelemetryPort" `
                --passive
        }
        else {
            Write-Host "`nDashboard not launched. Canonical training runs without observer attachment by default."
            Write-Host "  Tail logs: Get-Content '$trainUnifiedLog' -Wait"
            Write-Host "  Telemetry: actor.training.* remains available on tcp://localhost:$ActorTelemetryPort"
            Write-Host "  Live view: relaunch with -WithDashboard when passive observation attach is needed"
            Write-Host "  Stop:      Stop-Process -Id $($trainProc.Id) -Force"

            # In no-dashboard mode, wait for canonical train to finish naturally.
            $trainWaitStartedAt = Get-Date
            Wait-Process -Id $trainProc.Id
            $trainProc.Refresh()
            $exitCode = if ($null -eq $trainProc.ExitCode) { 0 } else { [int]$trainProc.ExitCode }
            Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_exit" -StartedAt $trainWaitStartedAt -ProcessId $trainProc.Id -Metadata ([ordered]@{
                exit_code = $exitCode
                dashboard_enabled = $false
            }) | Out-Null
            if ($exitCode -ne 0) {
                Write-Host "ERROR: canonical train exited with code $exitCode."
                exit $exitCode
            }
            Write-Host "Training completed successfully."
        }
    }
    finally {
        if ($null -ne $trainProc -and -not $trainProc.HasExited) {
            Write-Host "`nStopping canonical train (PID $($trainProc.Id))..."
            $trainStopStartedAt = Get-Date
            Stop-ProcessTreeById -ProcessId $trainProc.Id
            Write-NaviPhaseMetric -RunContext $runContext -Operation "train_process_stop" -StartedAt $trainStopStartedAt -Metadata ([ordered]@{
                process_id = $trainProc.Id
            }) | Out-Null
        }
        Stop-ListenersOnPorts -Ports @($ActorTelemetryPort, 5559, 5560)
    }
    exit 0
}

# ═══════════════════════════════════════════════════════════════════
# In-process inference: infer (in-process env+actor), dashboard by default
# ═══════════════════════════════════════════════════════════════════
if ($Infer) {
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        # Auto-select latest promoted model if available
        $latestModel = Join-Path $repoRoot "artifacts\models\latest.pt"
        if (Test-Path $latestModel) {
            $Checkpoint = $latestModel
            Write-Host "[ghost-stack] Using latest promoted model for inference: $Checkpoint" -ForegroundColor Cyan
        }
        else {
            throw "-Infer requires -Checkpoint <path> to specify the trained model (no latest promoted model found)."
        }
    }

    if ($NoDashboard -and $WithDashboard) {
        throw "-NoDashboard and -WithDashboard cannot be used together."
    }

    # Dashboard enabled by default for inference (opposite of training)
    $dashboardEnabled = (-not $NoDashboard) -or $WithDashboard

    $resolvedScene = ""
    if (-not [string]::IsNullOrWhiteSpace($Scene)) {
        $resolvedScene = (Resolve-Path $Scene).Path
    }

    $resolvedManifest = ""
    if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        $resolvedManifest = (Resolve-Path $Manifest).Path
    }

    $resolvedCorpusRoot = ""
    if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
        $resolvedCorpusRoot = (Resolve-Path $CorpusRoot).Path
    }

    $resolvedGmDagRoot = ""
    if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) {
        $resolvedGmDagRoot = (Resolve-Path $GmDagRoot).Path
    }

    if (-not [System.IO.Path]::IsPathRooted($Checkpoint)) {
        $Checkpoint = Join-Path $repoRoot $Checkpoint
    }
    if (-not (Test-Path $Checkpoint)) {
        throw "Checkpoint file not found: $Checkpoint"
    }

    $inferArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\actor"),
        "python", "-m", "navi_actor.cli", "infer",
        "--checkpoint", $Checkpoint,
        "--actors", "$NumActors",
        "--temporal-core", "$TemporalCore",
        "--actor-pub", "tcp://localhost:$ActorTelemetryPort",
        "--log-every", $LogEvery,
        "--compile-resolution", $GmDagResolution,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )

    if ($TotalSteps -gt 0) {
        $inferArgs += @("--total-steps", $TotalSteps)
    }
    if ($TotalEpisodes -gt 0) {
        $inferArgs += @("--total-episodes", $TotalEpisodes)
    }
    if ($Deterministic) {
        $inferArgs += "--deterministic"
    }

    if (-not $dashboardEnabled) {
        $inferArgs += "--no-emit-observation-stream"
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagFile)) {
        $inferArgs += @("--gmdag-file", $resolvedGmDagFile)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($resolvedScene)) {
        $inferArgs += @("--scene", $resolvedScene)
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedManifest)) {
        $inferArgs += @("--manifest", $resolvedManifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($resolvedCorpusRoot)) {
        $inferArgs += @("--corpus-root", $resolvedCorpusRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($resolvedGmDagRoot)) {
        $inferArgs += @("--gmdag-root", $resolvedGmDagRoot)
    }

    if (-not [string]::IsNullOrWhiteSpace($Datasets)) {
        $inferArgs += @("--datasets", $Datasets)
    }
    if (-not [string]::IsNullOrWhiteSpace($ExcludeDatasets)) {
        $inferArgs += @("--exclude-datasets", $ExcludeDatasets)
    }

    $inferLogOut = Join-Path $logsDir "infer.out.log"
    $inferLogErr = Join-Path $logsDir "infer.err.log"
    $inferUnifiedLog = Join-Path $repoRoot "logs\navi_actor_train.log"

    $inferProc = $null
    try {
        Write-Host "========================================================"
        Write-Host "  Navi Ghost-Matrix Inference (In-Process)"
        Write-Host "  Backend    : sdfdag"
        Write-Host "  Run ID     : $($runContext.RunId)"
        Write-Host "  Run Root   : $($runContext.RunRoot)"
        Write-Host "  Checkpoint : $Checkpoint"
        Write-Host "  Corpus     : $(if ($resolvedGmDagFile) { $resolvedGmDagFile } elseif ($resolvedScene) { $resolvedScene } elseif ($resolvedManifest) { $resolvedManifest } elseif ($resolvedCorpusRoot) { $resolvedCorpusRoot } else { 'auto-discovered canonical corpus' })"
        Write-Host "  Actors     : $NumActors"
        Write-Host "  Deterministic: $Deterministic"
        Write-Host "  Total Steps: $(if ($TotalSteps -le 0) { 'unlimited' } else { $TotalSteps })"
        Write-Host "  Total Ep.  : $(if ($TotalEpisodes -le 0) { 'unlimited' } else { $TotalEpisodes })"
        Write-Host "  Temporal   : $TemporalCore"
        if (-not [string]::IsNullOrWhiteSpace($Datasets)) {
            Write-Host "  Datasets   : $Datasets"
        }
        if (-not [string]::IsNullOrWhiteSpace($ExcludeDatasets)) {
            Write-Host "  Exclude DS : $ExcludeDatasets"
        }
        Write-Host "  Telemetry  : tcp://localhost:$ActorTelemetryPort"
        Write-Host "  Dashboard  : $(if ($dashboardEnabled) { 'enabled (passive observer)' } else { 'disabled' })"
        Write-Host "  Metrics    : $($runContext.MetricsRoot)"
        Write-Host "========================================================"

        Write-Host "`nStarting canonical inference (background)..."
        $inferLaunchStartedAt = Get-Date
        $inferProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $inferArgs -StdOutFile $inferLogOut -StdErrFile $inferLogErr
        Write-NaviPhaseMetric -RunContext $runContext -Operation "infer_process_launch" -StartedAt $inferLaunchStartedAt -ProcessId $inferProc.Id -Metadata ([ordered]@{
            dashboard_enabled = [bool]$dashboardEnabled
            actors = $NumActors
            temporal_core = $TemporalCore
            deterministic = [bool]$Deterministic
        }) | Out-Null
        Write-Host "  PID: $($inferProc.Id)"
        Write-Host "  Logs: $inferLogOut"
        Write-Host "        $inferLogErr"
        Write-Host "        $inferUnifiedLog"

        Write-Host "Verifying inference telemetry readiness ($ActorTelemetryPort)..."
        $inferReadyStartedAt = Get-Date
        $inferReady = Wait-ForTrainingReady -Process $inferProc -Port $ActorTelemetryPort -LogFiles @($inferLogErr, $inferUnifiedLog) -TimeoutSeconds 180
        Write-NaviPhaseMetric -RunContext $runContext -Operation "infer_process_ready" -StartedAt $inferReadyStartedAt -ProcessId $(if ($null -ne $inferProc) { $inferProc.Id } else { 0 }) -Metadata ([ordered]@{
            ready = [bool]$inferReady.Ready
            reason = [string]$inferReady.Reason
            exit_code = if ($inferReady.ContainsKey('ExitCode')) { $inferReady.ExitCode } else { $null }
            telemetry_port = $ActorTelemetryPort
        }) | Out-Null
        if (-not $inferReady.Ready) {
            $reason = [string]$inferReady.Reason
            if ($reason -eq "process-exited") {
                Write-Host "ERROR: inference exited before telemetry became ready (exit code $($inferReady.ExitCode))."
            }
            else {
                Write-Host "ERROR: inference telemetry failed readiness check ($ActorTelemetryPort, reason=$reason)."
            }
            Write-Host "Check logs: $inferLogErr"
            Write-Host "            $inferUnifiedLog"
            if ($null -ne $inferProc -and -not $inferProc.HasExited) {
                Stop-ProcessTreeById -ProcessId $inferProc.Id
            }
            Stop-ListenersOnPorts -Ports @($ActorTelemetryPort, 5559, 5560)
            exit 1
        }
        else {
            Write-Host "  OK: inference telemetry is ready ($($inferReady.Reason))."
        }

        if ($dashboardEnabled) {
            Write-Host "`nLaunching Dashboard (foreground, passive)..."
            Write-Host "  ESC = quit"
            & uv run --project (Join-Path $repoRoot "projects\auditor") `
                --python $PythonVersion `
                navi-auditor dashboard `
                --actor-sub "tcp://localhost:$ActorTelemetryPort" `
                --passive
        }
        else {
            Write-Host "`nDashboard not launched."
            Write-Host "  Tail logs: Get-Content '$inferUnifiedLog' -Wait"
            Write-Host "  Telemetry: actor.inference.* on tcp://localhost:$ActorTelemetryPort"
            Write-Host "  Stop:      Stop-Process -Id $($inferProc.Id) -Force"

            $inferWaitStartedAt = Get-Date
            Wait-Process -Id $inferProc.Id
            $inferProc.Refresh()
            $exitCode = if ($null -eq $inferProc.ExitCode) { 0 } else { [int]$inferProc.ExitCode }
            Write-NaviPhaseMetric -RunContext $runContext -Operation "infer_process_exit" -StartedAt $inferWaitStartedAt -ProcessId $inferProc.Id -Metadata ([ordered]@{
                exit_code = $exitCode
                dashboard_enabled = $false
            }) | Out-Null
            if ($exitCode -ne 0) {
                Write-Host "ERROR: inference exited with code $exitCode."
                exit $exitCode
            }
            Write-Host "Inference completed successfully."
        }
    }
    finally {
        if ($null -ne $inferProc -and -not $inferProc.HasExited) {
            Write-Host "`nStopping inference (PID $($inferProc.Id))..."
            $inferStopStartedAt = Get-Date
            Stop-ProcessTreeById -ProcessId $inferProc.Id
            Write-NaviPhaseMetric -RunContext $runContext -Operation "infer_process_stop" -StartedAt $inferStopStartedAt -Metadata ([ordered]@{
                process_id = $inferProc.Id
            }) | Out-Null
        }
        Stop-ListenersOnPorts -Ports @($ActorTelemetryPort, 5559, 5560)
    }
    exit 0
}

# ═══════════════════════════════════════════════════════════════════
# Legacy inference mode: Environment + Actor + Dashboard as 3 processes
# ═══════════════════════════════════════════════════════════════════
    $envArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "serve",
        "--mode", "step",
        "--pub", $EnvironmentPub,
        "--rep", $EnvironmentRep,
        "--actors", "$NumActors",
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )
$envArgs += @("--gmdag-file", $resolvedGmDagFile)

$actorArgs = @(
    "run",
    "--python", $PythonVersion,
    "--project", (Join-Path $repoRoot "projects\actor"),
    "navi-actor",
    "serve",
    "--sub", $ActorSub,
    "--pub", $ActorPub,
    "--mode", "step",
    "--temporal-core", "$TemporalCore",
    "--step-endpoint", $ActorStepEndpoint,
    "--azimuth-bins", "$AzimuthBins",
    "--elevation-bins", "$ElevationBins"
)

if (-not [string]::IsNullOrWhiteSpace($ActorPolicyCheckpoint)) {
    $actorArgs += @("--policy-checkpoint", $ActorPolicyCheckpoint)
}

$envLogOut = Join-Path $logsDir "environment.out.log"
$envLogErr = Join-Path $logsDir "environment.err.log"
$actorLogOut = Join-Path $logsDir "actor.out.log"
$actorLogErr = Join-Path $logsDir "actor.err.log"

$envProc = $null
$actorProc = $null

try {
    Write-Host "========================================================"
    Write-Host "  Navi Ghost-Matrix Inference"
    Write-Host "  Run ID     : $($runContext.RunId)"
    Write-Host "  Run Root   : $($runContext.RunRoot)"
    Write-Host "  Environment: sdfdag ($resolvedGmDagFile)"
    Write-Host "========================================================"
    Write-Host "Starting Environment..."
    $envProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $envArgs -StdOutFile $envLogOut -StdErrFile $envLogErr
    Write-Host "  PID: $($envProc.Id)"
    if (-not (Wait-ForPorts -Ports @(5559, 5560) -TimeoutSeconds 60)) {
        throw "Environment readiness failed: ports 5559/5560 did not bind in time."
    }

    Write-Host "Starting Actor..."
    $actorProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $actorArgs -StdOutFile $actorLogOut -StdErrFile $actorLogErr
    Write-Host "  PID: $($actorProc.Id)"
    if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds 60)) {
        throw "Actor readiness failed: port 5557 did not bind in time."
    }

    if (-not $NoDashboard) {
        Write-Host "Launching Auditor dashboard (foreground)..."
        Write-Host "  Tab = toggle manual/AI | WASD = move | ESC = quit"
        Write-Host "Logs:"
        Write-Host "  $envLogOut"
        Write-Host "  $actorLogOut"

        & uv run --python $PythonVersion --project (Join-Path $repoRoot "projects\auditor") navi-auditor dashboard --matrix-sub "tcp://localhost:5559" --actor-sub "tcp://localhost:5557" --step-endpoint "tcp://localhost:5560" $(if (-not [string]::IsNullOrWhiteSpace($Scene)) { "--scene"; $Scene })
    }
    else {
        Write-Host "Dashboard disabled. Processes running in background."
        Write-Host "  Env PID: $($envProc.Id)  Actor PID: $($actorProc.Id)"
        Write-Host "  Stop: Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -like '*navi-*' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }"
    }
}
finally {
    foreach ($proc in @($actorProc, $envProc)) {
        if ($null -ne $proc) {
            try {
                if (-not $proc.HasExited) {
                    Stop-ProcessTreeById -ProcessId $proc.Id
                }
            }
            catch {
            }
        }
    }
    Stop-ListenersOnPorts -Ports @(5557, 5559, 5560)
}
