<#
.SYNOPSIS
    Navigate scenes continuously -- record demonstrations without waiting for training.
.DESCRIPTION
    Launches the explorer dashboard on each scene in the compiled corpus.
    Recording starts automatically.  When the dashboard auto-closes (after
    MaxSteps) or you press ESC/Q, the next scene opens immediately -- no
    training runs between scenes.

    Demonstrations accumulate in the DemoDir and can be trained later with:
        .\scripts\run-bc-pretrain.ps1

    This separates the "fly" phase from the "train" phase so you can focus
    on navigating as many scenes as you want without interruption.

.EXAMPLE
    .\scripts\run-explore-scenes.ps1
    # Navigate the full corpus, 1000 steps per scene.

.EXAMPLE
    .\scripts\run-explore-scenes.ps1 -MaxSteps 2000
    # Longer sessions per scene.

.EXAMPLE
    .\scripts\run-explore-scenes.ps1 -CorpusRoot artifacts\gmdag\corpus\quake3-arenas
    # Only Quake 3 arenas.

.EXAMPLE
    .\scripts\run-explore-scenes.ps1 -Scenes "scene1.gmdag","scene2.gmdag"
    # Specific scenes only.
#>
param(
    [string]$CorpusRoot = "artifacts\gmdag\corpus",
    [string[]]$Scenes = @(),
    [string]$DemoDir = "artifacts\demonstrations",
    [int]$MaxSteps = 1000,
    [float]$LinearSpeed = 1.5,
    [float]$YawRate = 1.5
)

$ErrorActionPreference = "Continue"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

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

# Ensure demo directory exists
New-Item -ItemType Directory -Path $DemoDir -Force | Out-Null

# Count existing demos before we start
$existingDemos = @(Get-ChildItem -Filter "*.npz" $DemoDir -ErrorAction SilentlyContinue).Count

Write-Host "=== Ghost-Matrix Scene Explorer ===" -ForegroundColor Cyan
Write-Host "  Scenes:       $($sceneFiles.Count) .gmdag files"
Write-Host "  Demos dir:    $DemoDir"
Write-Host "  Steps/scene:  $MaxSteps (auto-close)"
Write-Host "  Existing demos: $existingDemos"
Write-Host ""
Write-Host "  Fly with WASD -- each scene auto-closes after $MaxSteps steps."
Write-Host "  ESC/Q to skip a scene early.  Ctrl+C to stop entirely."
Write-Host ""

$sceneIndex = 0
$newDemos = 0

foreach ($scenePath in $sceneFiles) {
    $sceneIndex++
    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($scenePath)

    Write-Host "--- Scene $sceneIndex/$($sceneFiles.Count): $sceneName ---" -ForegroundColor Cyan
    Write-Host "  File: $scenePath"
    Write-Host ""

    $demoBefore = @(Get-ChildItem -Filter "*.npz" $DemoDir -ErrorAction SilentlyContinue).Count

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

    $demoAfter = @(Get-ChildItem -Filter "*.npz" $DemoDir -ErrorAction SilentlyContinue).Count
    $sceneDemos = $demoAfter - $demoBefore

    if ($sceneDemos -gt 0) {
        $newDemos += $sceneDemos
        Write-Host "  Recorded $sceneDemos demo(s) -- total new: $newDemos" -ForegroundColor Green
    } else {
        Write-Host "  No demo recorded for this scene." -ForegroundColor Yellow
    }
    Write-Host ""
}

$totalDemos = @(Get-ChildItem -Filter "*.npz" $DemoDir -ErrorAction SilentlyContinue).Count

Write-Host "=== Exploration Complete ===" -ForegroundColor Green
Write-Host "  Scenes visited:  $sceneIndex"
Write-Host "  New demos:       $newDemos"
Write-Host "  Total demos:     $totalDemos"
Write-Host ""
Write-Host "  Next step -- train on your demonstrations:" -ForegroundColor Yellow
Write-Host "    .\scripts\run-bc-pretrain.ps1"
Write-Host ""
Write-Host "  Or with a custom checkpoint path:"
Write-Host "    .\scripts\run-bc-pretrain.ps1 -Output artifacts\checkpoints\my_model.pt"
Write-Host ""
Write-Host "  Then fine-tune with RL:"
Write-Host "    .\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt"
