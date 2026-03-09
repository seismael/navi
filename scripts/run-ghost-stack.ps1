<# .SYNOPSIS
  Launch the full Navi Ghost-Matrix stack.

    Two modes:
        1. Inference (default)  — canonical sdfdag Environment + Actor + Dashboard as 3 processes.
        2. Training  (-Train)   — canonical sdfdag train (in-process env+actor) + Dashboard.

.EXAMPLE
    # Inference on the canonical compiled runtime
    .\run-ghost-stack.ps1 -GmDagFile .\artifacts\gmdag\corpus\replicacad\frl_apartment_stage.gmdag

    # Canonical PPO training on the full discovered corpus with live dashboard
    .\run-ghost-stack.ps1 -Train

    # Canonical PPO training with an explicit scene override and refresh
    .\run-ghost-stack.ps1 -Train -Scene .\data\scenes\replicacad\frl_apartment_stage.glb -AutoCompileGmDag

  # Resume from checkpoint
  .\run-ghost-stack.ps1 -Train -TotalSteps 500000 -Checkpoint "checkpoints\policy_step_0010000.pt"
#>
param(
    # ── Mode ──
    [switch]$Train,

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
    [switch]$NoPreKill,
    [switch]$NoDashboard,

    # ── Training params ──
    [string]$Manifest = "",
    [int]$TotalSteps = 0,
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
    Stop-ListenersOnPorts -Ports @(5557, 5559, 5560)
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

$repoRoot = Get-RepoRoot
$logsDir = Join-Path $repoRoot "scripts\logs"

$resolvedGmDagFile = ""
if ($Train) {
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
# Canonical training launch: train (in-process env+actor) + dashboard
# ═══════════════════════════════════════════════════════════════════
if ($Train) {
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

    $trainArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor", "train",
        "--actors", "$NumActors",
        "--total-steps", $TotalSteps,
        "--shuffle",
        "--checkpoint-every", $CheckpointEvery,
        "--checkpoint-dir", $CheckpointDir,
        "--log-every", $LogEvery,
        "--rollout-length", $RolloutLength,
        "--compile-resolution", $GmDagResolution,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins"
    )

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

    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $trainArgs += @("--checkpoint", $Checkpoint)
    }

    $trainLogOut = Join-Path $logsDir "train.out.log"
    $trainLogErr = Join-Path $logsDir "train.err.log"

    $trainProc = $null
    try {
        Write-Host "========================================================"
        Write-Host "  Navi Ghost-Matrix Training"
        Write-Host "  Backend    : sdfdag"
        Write-Host "  Corpus     : $(if ($resolvedGmDagFile) { $resolvedGmDagFile } elseif ($resolvedScene) { $resolvedScene } elseif ($resolvedManifest) { $resolvedManifest } elseif ($resolvedCorpusRoot) { $resolvedCorpusRoot } else { 'auto-discovered canonical corpus' })"
        Write-Host "  Actors     : $NumActors (Standard Fleet)"
        Write-Host "  Total Steps: $(if ($TotalSteps -le 0) { 'continuous until stopped' } else { $TotalSteps })"
        Write-Host "  Checkpoints: every $CheckpointEvery -> $CheckpointDir"
        Write-Host "  Dashboard  : $(if ($NoDashboard) { 'disabled' } else { 'enabled' })"
        Write-Host "========================================================"

        Write-Host "`nStarting canonical train (background)..."
        $trainProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $trainArgs -StdOutFile $trainLogOut -StdErrFile $trainLogErr
        Write-Host "  PID: $($trainProc.Id)"
        Write-Host "  Logs: $trainLogOut"
        Write-Host "        $trainLogErr"

        Write-Host "Verifying training telemetry socket (5557)..."
        if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds 60)) {
            Write-Host "ERROR: training telemetry failed readiness check (5557)."
            Write-Host "Check logs: $trainLogErr"
            if ($null -ne $trainProc -and -not $trainProc.HasExited) {
                Stop-ProcessTreeById -ProcessId $trainProc.Id
            }
            Stop-ListenersOnPorts -Ports @(5557, 5559, 5560)
            exit 1
        }
        else {
            Write-Host "  OK: actor telemetry is bound."
        }

        if (-not $NoDashboard) {
            # The canonical trainer only guarantees actor telemetry. The auditor
            # remains resilient when matrix/step streams are absent.
            Write-Host "`nWaiting for actor telemetry before launching dashboard..."
            if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds 60)) {
                Write-Host "ERROR: Dashboard launch aborted, actor telemetry not ready (5557)."
                if ($null -ne $trainProc -and -not $trainProc.HasExited) {
                    Stop-ProcessTreeById -ProcessId $trainProc.Id
                }
                Stop-ListenersOnPorts -Ports @(5557, 5559, 5560)
                exit 1
            }

            Write-Host "Launching Dashboard (foreground)..."
            Write-Host "  Tab = toggle manual/AI | WASD = move | ESC = quit"
            & uv run --project (Join-Path $repoRoot "projects\auditor") `
                --python $PythonVersion `
                navi-auditor dashboard `
                --actor-sub "tcp://localhost:5557" `
                --passive
        }
        else {
            Write-Host "`nDashboard disabled. Training runs in background."
            Write-Host "  Tail logs: Get-Content '$trainLogErr' -Wait"
            Write-Host "  Stop:      Stop-Process -Id $($trainProc.Id) -Force"

            # In no-dashboard mode, wait for canonical train to finish naturally.
            Wait-Process -Id $trainProc.Id
            $trainProc.Refresh()
            $exitCode = if ($null -eq $trainProc.ExitCode) { 0 } else { [int]$trainProc.ExitCode }
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
            Stop-ProcessTreeById -ProcessId $trainProc.Id
        }
        Stop-ListenersOnPorts -Ports @(5557, 5559, 5560)
    }
    exit 0
}

# ═══════════════════════════════════════════════════════════════════
# Inference mode: Environment + Actor + Dashboard as 3 processes
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
    Write-Host "  PID: $($envProc.Id)"
    if (-not (Wait-ForPorts -Ports @(5559, 5560) -TimeoutSeconds 20)) {
        throw "Environment readiness failed: ports 5559/5560 did not bind in time."
    }

    Write-Host "Starting Actor..."
    $actorProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $actorArgs -StdOutFile $actorLogOut -StdErrFile $actorLogErr
    Write-Host "  PID: $($actorProc.Id)"
    if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds 20)) {
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
