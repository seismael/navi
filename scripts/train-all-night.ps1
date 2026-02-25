param(
    [string]$ManifestPath = "",
    [int]$TotalSteps = 10000000,
    [string]$CheckpointDir = "checkpoints/all_night",
    [string]$ResumeCheckpoint = "",
    [string]$Backend = "mesh",
    [int]$NumActors = 4,
    [int]$AzimuthBins = 128,
    [int]$ElevationBins = 24,
    [int]$MinibatchSize = 32,
    [int]$PpoEpochs = 2,
    [double]$ExistentialTax = -0.02,
    [double]$EntropyCoeff = 0.02,
    [double]$LearningRate = 5e-4,
    [int]$BpttLen = 16,
    [string]$Encoder = "vit",
    [int]$CheckpointEvery = 25000,
    [string]$LogDir = "scripts/logs/all_night"
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

$repoRoot = Get-RepoRoot

# ── Resolve defaults ──────────────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $repoRoot "data\scenes\scene_manifest_all.json"
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
Write-Host "  Navi All-Night Continuous Training"
Write-Host "  Manifest   : $ManifestPath"
Write-Host "  Steps      : $TotalSteps (Total)"
Write-Host "  Actors     : $NumActors"
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

# ── Start Continuous Training ──
# ── Start Continuous Training ──
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
    "--encoder", "$Encoder",
    "--backend", $Backend
)

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
