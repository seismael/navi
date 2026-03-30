<#
.SYNOPSIS
    Incremental manual training across scenes - fly to teach the model.
.DESCRIPTION
    Launches the explorer dashboard on each scene in the compiled corpus.
    Recording starts automatically when you navigate. When you close the
    dashboard (ESC/Q), the model is trained on your accumulated demonstrations
    and the checkpoint is updated.  The next scene loads your improved model
    so learning accumulates across scenes.

    Workflow:
      1. Script discovers all .gmdag scenes in the corpus.
      2. For each scene: opens explorer with auto-recording.
      3. When you close the dashboard, BC training runs on all recorded data.
      4. Checkpoint is saved and used as the starting point for the next scene.
      5. Repeat until all scenes are done, or press Ctrl+C to stop.

.EXAMPLE
    .\scripts\run-manual-training.ps1
    # Iterates through the full corpus, training incrementally.

.EXAMPLE
    .\scripts\run-manual-training.ps1 -Scenes "scene1.gmdag","scene2.gmdag"
    # Train on specific scenes only.

.EXAMPLE
    .\scripts\run-manual-training.ps1 -Checkpoint artifacts\checkpoints\bc_base_model.pt
    # Resume from an existing checkpoint.
#>
param(
    [string]$CorpusRoot = "artifacts\gmdag\corpus",
    [string[]]$Scenes = @(),
    [string]$Checkpoint = "",
    [string]$DemoDir = "artifacts\demonstrations",
    [string]$CheckpointOutput = "artifacts\checkpoints\bc_base_model.pt",
    [int]$Epochs = 30,
    [float]$LearningRate = 1e-3,
    [int]$BpttLen = 8,
    [int]$MinibatchSize = 32,
    [float]$LinearSpeed = 1.5,
    [float]$YawRate = 1.5,
    [int]$MaxSteps = 1000,
    [string]$TemporalCore = ""
)

Set-Location $PSScriptRoot\..

# -- Discover scenes -------------------------------------------------------
if ($Scenes.Count -gt 0) {
    $sceneFiles = @()
    foreach ($s in $Scenes) {
        $resolved = Resolve-Path $s -ErrorAction SilentlyContinue
        if ($resolved) {
            $sceneFiles += $resolved.Path
        } else {
            Write-Host "  WARNING: Scene not found: $s" -ForegroundColor Yellow
        }
    }
} else {
    $sceneFiles = @(Get-ChildItem -Recurse -Filter "*.gmdag" $CorpusRoot | Select-Object -ExpandProperty FullName)
}

if ($sceneFiles.Count -eq 0) {
    Write-Host "ERROR: No .gmdag scene files found." -ForegroundColor Red
    Write-Host "  Corpus root: $CorpusRoot"
    Write-Host "  Run refresh-scene-corpus.ps1 first to compile scenes."
    exit 1
}

Write-Host "=== Ghost-Matrix Manual Training ===" -ForegroundColor Cyan
Write-Host "  Scenes:     $($sceneFiles.Count) .gmdag files"
Write-Host "  Demos dir:  $DemoDir"
Write-Host "  Checkpoint: $(if ($Checkpoint) { $Checkpoint } else { '(fresh start)' })"
Write-Host "  Output:     $CheckpointOutput"
Write-Host "  Epochs/scene: $Epochs"
Write-Host "  Steps/scene:  $MaxSteps (auto-close)"
Write-Host ""
Write-Host "  Fly with WASD -- dashboard auto-closes after $MaxSteps steps, then trains."
Write-Host "  Press Ctrl+C at any time to stop."
Write-Host ""

# Ensure demo directory exists
New-Item -ItemType Directory -Path $DemoDir -Force | Out-Null

# Track the current checkpoint for incremental training
$currentCheckpoint = $Checkpoint
$sceneIndex = 0

foreach ($scenePath in $sceneFiles) {
    $sceneIndex++
    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($scenePath)

    Write-Host "--- Scene $sceneIndex/$($sceneFiles.Count): $sceneName ---" -ForegroundColor Cyan
    Write-Host "  File: $scenePath"
    if ($currentCheckpoint) {
        Write-Host "  Model: $currentCheckpoint"
    }
    Write-Host ""

    # -- Launch explore with auto-recording --------------------------------
    $exploreArgs = @(
        "run", "--project", "projects/auditor",
        "explore",
        "--gmdag-file", $scenePath,
        "--record",
        "--max-steps", $MaxSteps,
        "--linear-speed", $LinearSpeed,
        "--yaw-rate", $YawRate
    )

    & uv @exploreArgs 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.Exception.Message } else { $_ } }
    $exploreExit = $LASTEXITCODE

    # Qt close() on Windows often exits with -1073741510 (STATUS_CONTROL_C_EXIT).
    # This is normal for auto-close after max-steps.
    if ($exploreExit -ne 0 -and $exploreExit -ne -1073741510) {
        Write-Host "  Explorer exited with code $exploreExit -- skipping training." -ForegroundColor Yellow
        Write-Host ""
        continue
    }

    # Check if any new demo files were created
    $demoFiles = @(Get-ChildItem -Filter "*.npz" $DemoDir -ErrorAction SilentlyContinue)
    if ($demoFiles.Count -eq 0) {
        Write-Host "  No demonstrations recorded -- skipping training." -ForegroundColor Yellow
        Write-Host ""
        continue
    }

    Write-Host ""
    Write-Host "  Training on $($demoFiles.Count) demonstration file(s)..." -ForegroundColor Green

    # -- Run BC training ---------------------------------------------------
    $trainArgs = @(
        "run", "--project", "projects/actor",
        "brain", "bc-pretrain",
        "--demonstrations", $DemoDir,
        "--output", $CheckpointOutput,
        "--epochs", $Epochs,
        "--learning-rate", $LearningRate,
        "--bptt-len", $BpttLen,
        "--minibatch-size", $MinibatchSize
    )

    if ($currentCheckpoint) {
        $trainArgs += @("--checkpoint", $currentCheckpoint)
    }

    if ($TemporalCore) {
        $trainArgs += @("--temporal-core", $TemporalCore)
    }

    & uv @trainArgs 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.Exception.Message } else { $_ } }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    # Update checkpoint for next scene
    $currentCheckpoint = $CheckpointOutput
    Write-Host "  Checkpoint updated: $currentCheckpoint" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=== Manual Training Complete ===" -ForegroundColor Green
Write-Host "  Final checkpoint: $CheckpointOutput"
Write-Host "  Total scenes: $sceneIndex"
Write-Host ""
Write-Host "  Next step -- fine-tune with RL:"
Write-Host "    uv run brain train --checkpoint $CheckpointOutput"
