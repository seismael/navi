<# .SYNOPSIS
  Launch the full Navi Ghost-Matrix stack.

  Two modes:
    1. Inference (default)  — Environment + Actor + Dashboard as 3 processes.
    2. Training  (-Train)   — train-sequential (in-process env+actor) + Dashboard.

.EXAMPLE
  # Inference with voxel backend
  .\run-ghost-stack.ps1 -Backend voxel

  # Multi-scene PPO training with live dashboard
  .\run-ghost-stack.ps1 -Train -TotalSteps 500000

  # Resume from checkpoint
  .\run-ghost-stack.ps1 -Train -TotalSteps 500000 -Checkpoint "checkpoints\policy_step_0010000.pt"
#>
param(
    # ── Mode ──
    [switch]$Train,

    # ── Common ──
    [string]$Backend = "mesh",
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [string]$HabitatScene = "",
    [string]$HabitatDatasetConfig = "",
    [switch]$NoPreKill,
    [switch]$NoDashboard,

    # ── Training params ──
    [string]$Manifest = "",
    [int]$TotalSteps = 500000,
    [int]$CheckpointEvery = 25000,
    [string]$CheckpointDir = "checkpoints",
    [string]$Checkpoint = "",
    [int]$LogEvery = 100,
    [int]$RolloutLength = 512,

    # ── Inference-mode ZMQ addresses ──
    [string]$EnvironmentPub = "tcp://*:5559",
    [string]$EnvironmentRep = "tcp://*:5560",
    [string]$ActorSub = "tcp://localhost:5559",
    [string]$ActorPub = "tcp://*:5557",
    [string]$ActorStepEndpoint = "tcp://localhost:5560",
    [string]$ActorPolicyCheckpoint = ""
)

# Standard Ghost-Matrix Fleet Size
$NumActors = 4

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
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
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
        catch {
        }
    }
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

$repoRoot = Get-RepoRoot
$logsDir = Join-Path $repoRoot "scripts\logs"

if (-not $NoPreKill) {
    Write-Host "Stopping stale Navi processes..."
    Stop-NaviProcesses
    Start-Sleep -Milliseconds 500
}

# ═══════════════════════════════════════════════════════════════════
# Training mode: train-sequential (in-process env+actor) + dashboard
# ═══════════════════════════════════════════════════════════════════
if ($Train) {
    # Resolve manifest
    if ([string]::IsNullOrWhiteSpace($Manifest)) {
        $Manifest = Join-Path $repoRoot "data\scenes\scene_manifest.json"
    }
    if (-not (Test-Path $Manifest)) {
        Write-Host "ERROR: Manifest not found: $Manifest"
        exit 1
    }

    # Resolve checkpoint dir to absolute
    if (-not [System.IO.Path]::IsPathRooted($CheckpointDir)) {
        $CheckpointDir = Join-Path $repoRoot "projects\actor\$CheckpointDir"
    }

    $trainArgs = @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor", "train-sequential",
        "--manifest", $Manifest,
        "--actors", "$NumActors",
        "--total-steps", $TotalSteps,
        "--shuffle",
        "--backend", $Backend,
        "--checkpoint-every", $CheckpointEvery,
        "--checkpoint-dir", $CheckpointDir,
        "--log-every", $LogEvery,
        "--rollout-length", $RolloutLength,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )

    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $trainArgs += @("--checkpoint", $Checkpoint)
    }

    $trainLogOut = Join-Path $logsDir "train-sequential.out.log"
    $trainLogErr = Join-Path $logsDir "train-sequential.err.log"

    $trainProc = $null
    try {
        Write-Host "========================================================"
        Write-Host "  Navi Ghost-Matrix Training"
        Write-Host "  Backend    : $Backend"
        Write-Host "  Actors     : $NumActors (Standard Fleet)"
        Write-Host "  Total Steps: $TotalSteps"
        Write-Host "  Checkpoints: every $CheckpointEvery → $CheckpointDir"
        Write-Host "  Dashboard  : $(if ($NoDashboard) { 'disabled' } else { 'enabled' })"
        Write-Host "========================================================"

        Write-Host "`nStarting train-sequential (background)..."
        $trainProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $trainArgs -StdOutFile $trainLogOut -StdErrFile $trainLogErr
        Write-Host "  PID: $($trainProc.Id)"
        Write-Host "  Logs: $trainLogOut"
        Write-Host "        $trainLogErr"

        if (-not $NoDashboard) {
            # Wait for ZMQ sockets to bind (env + actor start inside train-sequential)
            Write-Host "`nWaiting for ZMQ sockets to bind..."
            Start-Sleep -Seconds 4

            Write-Host "Launching Dashboard (foreground)..."
            Write-Host "  Tab = toggle manual/AI | WASD = move | ESC = quit"
            & uv run --project (Join-Path $repoRoot "projects\auditor") `
                navi-auditor dashboard `
                --matrix-sub "tcp://localhost:5559" `
                --actor-sub "tcp://localhost:5557" `
                --step-endpoint "tcp://localhost:5560"
        }
        else {
            Write-Host "`nDashboard disabled. Training runs in background."
            Write-Host "  Tail logs: Get-Content '$trainLogErr' -Wait"
            Write-Host "  Stop:      Stop-Process -Id $($trainProc.Id) -Force"
        }
    }
    finally {
        if ($null -ne $trainProc -and -not $trainProc.HasExited) {
            Write-Host "`nStopping train-sequential (PID $($trainProc.Id))..."
            Stop-Process -Id $trainProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    exit 0
}

# ═══════════════════════════════════════════════════════════════════
# Inference mode: Environment + Actor + Dashboard as 3 processes
# ═══════════════════════════════════════════════════════════════════
    $envArgs = @(
        "run",
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "serve",
        "--mode", "step",
        "--pub", $EnvironmentPub,
        "--rep", $EnvironmentRep,
        "--backend", $Backend,
        "--actors", "$NumActors",
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )

if ($Backend -eq "voxel") {
    $envArgs += @("--generator", "arena")
}
elseif ($Backend -eq "habitat") {
    if ([string]::IsNullOrWhiteSpace($HabitatScene)) {
        throw "HabitatScene is required when Backend=habitat"
    }
    $envArgs += @("--habitat-scene", $HabitatScene)
    if (-not [string]::IsNullOrWhiteSpace($HabitatDatasetConfig)) {
        $envArgs += @("--habitat-dataset-config", $HabitatDatasetConfig)
    }
}
elseif ($Backend -eq "mesh") {
    if ([string]::IsNullOrWhiteSpace($HabitatScene)) {
        throw "HabitatScene is required when Backend=mesh (path to .glb/.obj scene)"
    }
    $envArgs += @("--habitat-scene", $HabitatScene)
    $envArgs += @("--max-distance", "15")
    if (-not [string]::IsNullOrWhiteSpace($HabitatDatasetConfig)) {
        $envArgs += @("--habitat-dataset-config", $HabitatDatasetConfig)
    }
}

$actorArgs = @(
    "run",
    "--project", (Join-Path $repoRoot "projects\actor"),
    "navi-actor",
    "run",
    "--sub", $ActorSub,
    "--pub", $ActorPub,
    "--mode", "step",
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
    Write-Host "Starting Environment..."
    $envProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $envArgs -StdOutFile $envLogOut -StdErrFile $envLogErr
    Start-Sleep -Milliseconds 1200

    Write-Host "Starting Actor..."
    $actorProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $actorArgs -StdOutFile $actorLogOut -StdErrFile $actorLogErr
    Start-Sleep -Milliseconds 1200

    if (-not $NoDashboard) {
        Write-Host "Launching Auditor dashboard (foreground)..."
        Write-Host "  Tab = toggle manual/AI | WASD = move | ESC = quit"
        Write-Host "Logs:"
        Write-Host "  $envLogOut"
        Write-Host "  $actorLogOut"

        & uv run --project (Join-Path $repoRoot "projects\auditor") navi-auditor dashboard --matrix-sub "tcp://localhost:5559" --actor-sub "tcp://localhost:5557" --step-endpoint "tcp://localhost:5560" $(if (-not [string]::IsNullOrWhiteSpace($HabitatScene)) { "--scene"; $HabitatScene })
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
                    Stop-Process -Id $proc.Id -Force -ErrorAction Stop
                }
            }
            catch {
            }
        }
    }
}
