param(
    [string]$ManifestPath = "",
    [int]$TotalSteps = 500000,
    [string]$CheckpointDir = "checkpoints/habitat",
    [string]$ResumeCheckpoint = "",
    [string]$Backend = "mesh",
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [int]$MinibatchSize = 32,
    [int]$PpoEpochs = 2,
    [double]$ExistentialTax = -0.02,
    [double]$EntropyCoeff = 0.02,
    [double]$LearningRate = 5e-4,
    [int]$BpttLen = 16,
    [int]$CheckpointEvery = 25000,
    [string]$LogDir = "scripts/logs/habitat"
)

# Standard Ghost-Matrix Fleet Size
$NumActors = 4

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

if (-not (Test-Path $ManifestPath)) {
    Write-Host "ERROR: Scene manifest not found at $ManifestPath"
    exit 1
}

# Create output dirs
foreach ($dir in @($CheckpointDir, $LogDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "========================================================"
Write-Host "  Navi Habitat Sequential Training"
Write-Host "  Manifest   : $ManifestPath"
Write-Host "  Steps      : $TotalSteps (Total)"
Write-Host "  Actors     : $NumActors (Standard Fleet)"
Write-Host "  Resolution : ${AzimuthBins}x${ElevationBins}"
Write-Host "  Backend    : $Backend"
Write-Host "  Checkpoint : $CheckpointDir (every $CheckpointEvery)"
Write-Host "  Optimizer  : LR=$LearningRate, Batch=$MinibatchSize, Epochs=$PpoEpochs"
Write-Host "  Reward     : Tax=$ExistentialTax, Entropy=$EntropyCoeff"
Write-Host "========================================================"
Write-Host ""

# ── Helper: kill environment and actor ────────────────────────
function Stop-NaviProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*navi-environment*" -or
            $_.CommandLine -like "*navi-actor*"
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

Stop-NaviProcesses

# ── Start Sequential Training ──
$actorArgs = @(
    "run",
    "--project", (Join-Path $repoRoot "projects\actor"),
    "navi-actor", "train-sequential",
    "--manifest", $ManifestPath,
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
    "--backend", $Backend,
    "--shuffle"
)


if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $actorArgs += @("--checkpoint", $ResumeCheckpoint)
}

$logOut = Join-Path $LogDir "train_sequential.out.log"
$logErr = Join-Path $LogDir "train_sequential.err.log"

Write-Host "  Starting sequential training engine..."
Write-Host "  Logs: $logErr"

Start-Process -FilePath "uv" -ArgumentList $actorArgs `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -NoNewWindow -Wait
