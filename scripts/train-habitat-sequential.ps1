<# .SYNOPSIS
  Train the CognitiveMambaPolicy on Habitat scenes sequentially.

  Reads a scene manifest JSON (produced by download-habitat-data.ps1),
  then for each scene:
    1. Starts a Environment with --backend habitat --habitat-scene <scene>
    2. Runs PPO training for a configured number of steps
    3. Saves a checkpoint and resumes on the next scene

  The policy carries over between scenes — the agent learns to generalise
  across different 3D environments.

.PARAMETER ManifestPath
  Path to scene_manifest.json produced by download-habitat-data.ps1.
  If not set, looks in data/habitat/scene_manifest.json.

.PARAMETER StepsPerScene
  Number of environment steps to train on each scene. Default: 10000.

.PARAMETER TotalSteps
  Optional cap on total steps across all scenes. 0 = no cap.

.PARAMETER CheckpointDir
  Directory for checkpoint files. Default: checkpoints/habitat

.PARAMETER ResumeCheckpoint
  Path to a .pt checkpoint to resume from.

.PARAMETER Backend
  Simulator backend: "habitat" (requires habitat-sim) or "mesh" (uses trimesh+embreex).
  Default: mesh (no habitat-sim dependency).

.PARAMETER NumActors
  Number of parallel actors to use for data collection. Default: 1.

.PARAMETER LogDir
  Directory for per-scene log files. Default: scripts/logs/habitat

.PARAMETER SkipScenes
  Number of scenes to skip from the beginning (for resuming).

.EXAMPLE
  .\train-habitat-sequential.ps1
  .\train-habitat-sequential.ps1 -StepsPerScene 20000 -Backend habitat
  .\train-habitat-sequential.ps1 -ResumeCheckpoint "checkpoints\habitat\scene_003.pt" -SkipScenes 4
#>
param(
    [string]$ManifestPath = "",
    [int]$StepsPerScene = 10000,
    [int]$TotalSteps = 0,
    [string]$CheckpointDir = "",
    [string]$ResumeCheckpoint = "",
    [string]$Backend = "mesh",
    [int]$NumActors = 1,
    [int]$AzimuthBins = 128,
    [int]$ElevationBins = 24,
    [int]$MinibatchSize = 32,
    [int]$PpoEpochs = 2,
    [double]$ExistentialTax = -0.002,
    [double]$EntropyCoeff = 0.02,
    [double]$LearningRate = 5e-4,
    [string]$LogDir = "",
    [int]$SkipScenes = 0
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

$repoRoot = Get-RepoRoot

# ── Resolve defaults ──────────────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $repoRoot "data\scenes\scene_manifest.json"
}
if ([string]::IsNullOrWhiteSpace($CheckpointDir)) {
    $CheckpointDir = Join-Path $repoRoot "checkpoints\habitat"
}
if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $repoRoot "scripts\logs\habitat"
}

if (-not (Test-Path $ManifestPath)) {
    Write-Host "ERROR: Scene manifest not found at $ManifestPath"
    Write-Host "Run download-habitat-data.ps1 first to generate it."
    exit 1
}

# Create output dirs
foreach ($dir in @($CheckpointDir, $LogDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# ── Load manifest ────────────────────────────────────────────────
$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$scenes = $manifest.scenes

if ($scenes.Count -eq 0) {
    Write-Host "ERROR: No scenes found in manifest. Download datasets first."
    exit 1
}

Write-Host "========================================================"
Write-Host "  Navi Sequential Habitat Training"
Write-Host "  Scenes     : $($scenes.Count)"
Write-Host "  Steps/scene: $StepsPerScene"
Write-Host "  Actors     : $NumActors"
Write-Host "  Resolution : ${AzimuthBins}x${ElevationBins}"
Write-Host "  Backend    : $Backend"
Write-Host "  Checkpoint : $CheckpointDir"
Write-Host "  Skip       : $SkipScenes scenes"
if ($TotalSteps -gt 0) {
    Write-Host "  Total cap  : $TotalSteps"
}
if ($ResumeCheckpoint) {
    Write-Host "  Resume from: $ResumeCheckpoint"
}
Write-Host "========================================================"
Write-Host ""

# ── Helper: kill environment and actor ────────────────────────
function Stop-NaviProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*navi-environment*" -or
            $_.CommandLine -like "*navi-actor*train-ppo*"
        )
    } | ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        catch { }
    }
    # Wait for ZMQ ports to be fully released
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        $held = netstat -ano 2>$null | Select-String "5559|5560|5557"
        if (-not $held) { break }
        # Kill any process still holding the ports
        foreach ($line in $held) {
            if ($line -match '\s(\d+)\s*$') {
                $procId = [int]$Matches[1]
                if ($procId -gt 0) {
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                }
            }
        }
        Start-Sleep -Seconds 1
    }
}

# ── Helper: wait for Environment to be ready ──────────────────
function Wait-Environment {
    param([string]$StdOutLog, [string]$StdErrLog = "", [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        # Check both stdout and stderr for the ready message
        foreach ($logFile in @($StdOutLog, $StdErrLog)) {
            if ([string]::IsNullOrWhiteSpace($logFile)) { continue }
            if (-not (Test-Path $logFile)) { continue }
            try {
                $fs = [IO.FileStream]::new($logFile, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
                $sr = [IO.StreamReader]::new($fs)
                $content = $sr.ReadToEnd()
                $sr.Close(); $fs.Close()
                if ($content -match "Environment starting") {
                    return $true
                }
            }
            catch { }
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# ── Helper: wait for actor training to finish ─────────────────────
function Wait-ActorProcess {
    param([System.Diagnostics.Process]$Proc, [string]$StdErrLog)
    while (-not $Proc.HasExited) {
        Start-Sleep -Seconds 5
        # Print latest log line
        if (Test-Path $StdErrLog) {
            try {
                $fs = [IO.FileStream]::new($StdErrLog, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
                $sr = [IO.StreamReader]::new($fs)
                $lines = $sr.ReadToEnd() -split "`n" | Where-Object { $_.Trim() -ne "" }
                $sr.Close(); $fs.Close()
                if ($lines.Count -gt 0) {
                    Write-Host "    $($lines[-1].Trim())" -ForegroundColor DarkGray
                }
            }
            catch { }
        }
    }
}

# ── Sequential training loop ─────────────────────────────────────
$currentCheckpoint = $ResumeCheckpoint
$globalSteps = 0
$env:PYTHONUNBUFFERED = "1"

for ($i = $SkipScenes; $i -lt $scenes.Count; $i++) {
    $scene = $scenes[$i]
    $scenePath = $scene.path
    $datasetName = $scene.dataset
    $sceneFile = [IO.Path]::GetFileNameWithoutExtension($scenePath)
    $sceneLabel = "{0:D3}_{1}_{2}" -f $i, $datasetName, $sceneFile

    # Check total step cap
    if ($TotalSteps -gt 0 -and $globalSteps -ge $TotalSteps) {
        Write-Host "Total step cap ($TotalSteps) reached. Stopping."
        break
    }

    $stepsThisScene = $StepsPerScene
    if ($TotalSteps -gt 0) {
        $stepsThisScene = [Math]::Min($stepsThisScene, $TotalSteps - $globalSteps)
    }

    Write-Host ""
    Write-Host "--------------------------------------------------------"
    Write-Host "  Scene $($i+1)/$($scenes.Count): $sceneLabel"
    Write-Host "  Path: $scenePath"
    Write-Host "  Steps: $stepsThisScene"
    Write-Host "--------------------------------------------------------"

    # Verify scene file exists
    if (-not (Test-Path $scenePath)) {
        Write-Host "  [SKIP] Scene file not found: $scenePath" -ForegroundColor Yellow
        continue
    }

    # Kill any lingering processes
    Stop-NaviProcesses

    # ── Start Environment ──
    $envOutLog = Join-Path $LogDir "env_${sceneLabel}.out.log"
    $envErrLog = Join-Path $LogDir "env_${sceneLabel}.err.log"
    $envArgs = @(
        "run",
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment", "serve",
        "--mode", "step",
        "--pub", "tcp://*:5559",
        "--rep", "tcp://*:5560",
        "--backend", $Backend,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins",
        "--habitat-scene", $scenePath,
        "--actors", "$NumActors"
    )

    Write-Host "  Starting Environment ($Backend)..."
    $envProc = Start-Process -FilePath "uv" -ArgumentList $envArgs `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $envOutLog `
        -RedirectStandardError $envErrLog `
        -PassThru

    $ready = Wait-Environment -StdOutLog $envOutLog -StdErrLog $envErrLog -TimeoutSeconds 60
    if (-not $ready) {
        Write-Host "  [ERROR] Environment failed to start. Check $envErrLog" -ForegroundColor Red
        if ($null -ne $envProc -and -not $envProc.HasExited) {
            Stop-Process -Id $envProc.Id -Force -ErrorAction SilentlyContinue
        }
        continue
    }
    Write-Host "  Environment ready."

    # ── Start PPO Training ──
    $actorOutLog = Join-Path $LogDir "actor_${sceneLabel}.out.log"
    $actorErrLog = Join-Path $LogDir "actor_${sceneLabel}.err.log"
    $ckptPath = Join-Path $CheckpointDir "scene_${sceneLabel}.pt"

    $actorArgs = @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor", "train-ppo",
        "--sub", "tcp://localhost:5559",
        "--pub", "tcp://*:5557",
        "--step-endpoint", "tcp://localhost:5560",
        "--actors", "$NumActors",
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins",
        "--steps", "$stepsThisScene",
        "--log-every", "100",
        "--checkpoint-every", "0",
        "--checkpoint-dir", $CheckpointDir,
        "--rollout-length", "512",
        "--ppo-epochs", "$PpoEpochs",
        "--minibatch-size", "$MinibatchSize",
        "--existential-tax", "$ExistentialTax",
        "--entropy-coeff", "$EntropyCoeff",
        "--learning-rate", "$LearningRate",
        "--bptt-len", "16"
    )

    if (-not [string]::IsNullOrWhiteSpace($currentCheckpoint)) {
        if (Test-Path $currentCheckpoint) {
            $actorArgs += @("--checkpoint", $currentCheckpoint)
        } else {
            Write-Host "  [WARN] Checkpoint not found, training from scratch: $currentCheckpoint" -ForegroundColor Yellow
            $currentCheckpoint = ""
        }
    }

    Write-Host "  Starting PPO training ($stepsThisScene steps)..."
    $actorProc = Start-Process -FilePath "uv" -ArgumentList $actorArgs `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $actorOutLog `
        -RedirectStandardError $actorErrLog `
        -PassThru

    # Wait for training to complete
    Wait-ActorProcess -Proc $actorProc -StdErrLog $actorErrLog

    Write-Host "  Training on scene $sceneLabel complete (exit code: $($actorProc.ExitCode))."

    # ── Save scene checkpoint ──
    # The trainer always saves a final checkpoint as policy-step-NNNNNNN.pt
    $latestCkpt = Get-ChildItem -Path $CheckpointDir -Filter "policy-step-*.pt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($latestCkpt) {
        # Rename latest checkpoint to scene-specific name
        if ($latestCkpt.FullName -ne $ckptPath) {
            Copy-Item $latestCkpt.FullName $ckptPath -Force
        }
        $currentCheckpoint = $ckptPath
        Write-Host "  Checkpoint: $ckptPath"
    }
    else {
        Write-Host "  [WARN] No checkpoint found after training." -ForegroundColor Yellow
    }

    # Shut down Environment for this scene
    if ($null -ne $envProc -and -not $envProc.HasExited) {
        Stop-Process -Id $envProc.Id -Force -ErrorAction SilentlyContinue
    }

    $globalSteps += $stepsThisScene
}

# Final cleanup
Stop-NaviProcesses

Write-Host ""
Write-Host "========================================================"
Write-Host "  Sequential training complete"
Write-Host "  Scenes trained : $([Math]::Min($scenes.Count, $scenes.Count - $SkipScenes))"
Write-Host "  Total steps    : $globalSteps"
Write-Host "  Final checkpoint: $currentCheckpoint"
Write-Host "  Logs           : $LogDir"
Write-Host "========================================================"
