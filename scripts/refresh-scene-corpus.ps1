<#
.SYNOPSIS
  Refresh the canonical Navi training corpus end-to-end.

.DESCRIPTION
  Canonical corpus refresh runs as a staged transaction:
    1. Download source datasets into a transient staging directory.
    2. Compile a fresh `.gmdag` corpus into a staged compiled directory.
    3. Promote the staged compiled corpus into `artifacts/gmdag/corpus` only after success.
    4. Remove transient downloads and scratch data after successful integration.

  This keeps the live compiled corpus intact until a full replacement is ready,
  automatically replaces unsuitable compile resolutions, and avoids retaining raw
  downloaded source datasets after integration.
#>
param(
    [string]$DataDir = "",
    [string]$Datasets = "all",
    [string]$Manifest = "",
    [string]$Scene = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$ScratchRoot = "",
    [int]$Resolution = 512,
    [int]$MinSceneBytes = 100000,
    [string]$PythonVersion = "3.12",
    [switch]$SkipDownload,
    [switch]$PreserveExisting,
    [switch]$KeepScratch
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Ensure-Directory {
    param([string]$PathToCreate)

    if (-not (Test-Path $PathToCreate)) {
        New-Item -ItemType Directory -Path $PathToCreate -Force | Out-Null
    }
}

function Remove-PathIfExists {
    param([string]$PathToRemove)

    if (Test-Path $PathToRemove) {
        Remove-Item -Path $PathToRemove -Recurse -Force
    }
}

function Copy-RefreshMetadata {
    param(
        [string]$StageSourceRoot,
        [string]$StageCompiledRoot
    )

    foreach ($fileName in @("scene_manifest.json", "scene_manifest_all.json")) {
        $candidate = Join-Path $StageSourceRoot $fileName
        if (Test-Path $candidate) {
            Copy-Item -Path $candidate -Destination (Join-Path $StageCompiledRoot $fileName) -Force
        }
    }
}

function Update-CompiledManifestForLiveCorpus {
    param(
        [string]$CompiledRoot,
        [string]$LiveCompiledRoot
    )

    $manifestPath = Join-Path $CompiledRoot "gmdag_manifest.json"
    if (-not (Test-Path $manifestPath)) {
        throw "Compiled manifest not found for live-corpus rewrite: $manifestPath"
    }

    $compiledRootPath = [System.IO.Path]::GetFullPath($CompiledRoot)
    $liveRootPath = [System.IO.Path]::GetFullPath($LiveCompiledRoot)
    $manifest = Get-Content -Path $manifestPath -Raw | ConvertFrom-Json
    $manifest.source_root = $liveRootPath
    $manifest.gmdag_root = $liveRootPath

    foreach ($scene in $manifest.scenes) {
        $scenePath = [string]$scene.gmdag_path
        if ([string]::IsNullOrWhiteSpace($scenePath)) {
            continue
        }

        $fullScenePath = [System.IO.Path]::GetFullPath($scenePath)
        if ($fullScenePath.StartsWith($compiledRootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
            $relativePath = $fullScenePath.Substring($compiledRootPath.Length).TrimStart('\', '/')
            $liveScenePath = (Join-Path $liveRootPath $relativePath)
            $scene.gmdag_path = $liveScenePath
            # Live manifests must not retain scratch download paths after promotion.
            $scene.source_path = $liveScenePath
        }
    }

    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8
}

function Cleanup-LegacyCanonicalSourceRoot {
    param([string]$RepoRoot)

    $legacyRoot = Join-Path $RepoRoot "data\scenes"
    if (-not (Test-Path $legacyRoot)) {
        return
    }

    $targets = @(
        (Join-Path $legacyRoot "replicacad"),
        (Join-Path $legacyRoot "apartment_1.glb"),
        (Join-Path $legacyRoot "skokloster-castle.glb"),
        (Join-Path $legacyRoot "van-gogh-room.glb"),
        (Join-Path $legacyRoot "scene_manifest.json"),
        (Join-Path $legacyRoot "scene_manifest_all.json")
    )

    foreach ($target in $targets) {
        if (-not (Test-Path $target)) {
            continue
        }
        Remove-Item -Path $target -Recurse -Force
        Write-Host "  Removed legacy source artifact: $target"
    }
}

$repoRoot = Get-RepoRoot
$logsRoot = Join-Path $repoRoot "scripts\logs\corpus-refresh"
Ensure-Directory -PathToCreate $logsRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcriptPath = Join-Path $logsRoot "refresh_$timestamp.log"
Start-Transcript -Path $transcriptPath -Force | Out-Null

$refreshSucceeded = $false
$backupRoot = ""

try {
    if ([string]::IsNullOrWhiteSpace($ScratchRoot)) {
        $ScratchRoot = Join-Path $repoRoot "artifacts\tmp\corpus-refresh"
    }
    if ([string]::IsNullOrWhiteSpace($DataDir)) {
        $DataDir = Join-Path $ScratchRoot "downloads"
    }
    if ([string]::IsNullOrWhiteSpace($CorpusRoot)) {
        $CorpusRoot = $DataDir
    }
    if ([string]::IsNullOrWhiteSpace($GmDagRoot)) {
        $GmDagRoot = Join-Path $repoRoot "artifacts\gmdag\corpus"
    }

    $stageSourceRoot = $DataDir
    $stageCompiledRoot = Join-Path $ScratchRoot "compiled"
    $finalCompiledRoot = $GmDagRoot

    Ensure-Directory -PathToCreate $ScratchRoot
    Ensure-Directory -PathToCreate (Split-Path -Parent $finalCompiledRoot)

    if (-not $PreserveExisting) {
        Remove-PathIfExists -PathToRemove $stageSourceRoot
        Remove-PathIfExists -PathToRemove $stageCompiledRoot
    }

    if (-not $SkipDownload) {
        $downloadArgs = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $PSScriptRoot "download-habitat-data.ps1"),
            "-DataDir", $stageSourceRoot,
            "-Datasets", $Datasets
        )
        if ($PreserveExisting) {
            $downloadArgs += "-PreserveExisting"
        }

        Write-Host "Refreshing source datasets into staging..."
        & powershell.exe @downloadArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset download refresh failed with exit code $LASTEXITCODE"
        }
    }

    $resolvedSourceRoot = if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
        if (Test-Path $CorpusRoot) {
            (Resolve-Path $CorpusRoot).Path
        }
        else {
            $CorpusRoot
        }
    }
    else {
        $stageSourceRoot
    }

    if (-not (Test-Path $resolvedSourceRoot)) {
        throw "Corpus source root not found: $resolvedSourceRoot"
    }

    Ensure-Directory -PathToCreate $stageCompiledRoot

    $resolvedScene = if (-not [string]::IsNullOrWhiteSpace($Scene)) {
        (Resolve-Path $Scene).Path
    } else {
        ""
    }
    $resolvedManifest = if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        (Resolve-Path $Manifest).Path
    } else {
        Join-Path $resolvedSourceRoot "scene_manifest_all.json"
    }

    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  Navi Corpus Refresh"
    Write-Host "  Source Stage : $resolvedSourceRoot"
    Write-Host "  Manifest     : $resolvedManifest"
    Write-Host "  Compiled Out : $stageCompiledRoot"
    Write-Host "  Live Corpus  : $finalCompiledRoot"
    Write-Host "  Resolution   : $Resolution"
    Write-Host "  Mode         : staged overwrite-first refresh"
    Write-Host "  Transcript   : $transcriptPath"
    Write-Host "========================================================"
    Write-Host ""

    $prepareArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment", "prepare-corpus",
        "--corpus-root", $resolvedSourceRoot,
        "--gmdag-root", $stageCompiledRoot,
        "--resolution", "$Resolution",
        "--min-scene-bytes", "$MinSceneBytes",
        "--force-recompile"
    )

    if (-not [string]::IsNullOrWhiteSpace($resolvedManifest) -and (Test-Path $resolvedManifest)) {
        $prepareArgs += @("--manifest", $resolvedManifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($resolvedScene)) {
        $prepareArgs += @("--scene", $resolvedScene)
    }

    & uv @prepareArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Corpus preparation failed with exit code $LASTEXITCODE"
    }

    $stageManifest = Join-Path $stageCompiledRoot "gmdag_manifest.json"
    $stageAssets = Get-ChildItem -Path $stageCompiledRoot -Recurse -File -Filter "*.gmdag"
    if (-not (Test-Path $stageManifest)) {
        throw "Prepared corpus is missing gmdag_manifest.json: $stageManifest"
    }
    if ($stageAssets.Count -eq 0) {
        throw "Prepared corpus contains no compiled .gmdag assets: $stageCompiledRoot"
    }

    Copy-RefreshMetadata -StageSourceRoot $resolvedSourceRoot -StageCompiledRoot $stageCompiledRoot
    Update-CompiledManifestForLiveCorpus -CompiledRoot $stageCompiledRoot -LiveCompiledRoot $finalCompiledRoot

    $backupRoot = Join-Path (Split-Path -Parent $finalCompiledRoot) ("corpus_backup_" + $timestamp)
    if (Test-Path $backupRoot) {
        Remove-PathIfExists -PathToRemove $backupRoot
    }

    if (Test-Path $finalCompiledRoot) {
        Move-Item -Path $finalCompiledRoot -Destination $backupRoot
    }

    try {
        Move-Item -Path $stageCompiledRoot -Destination $finalCompiledRoot
    }
    catch {
        if ((-not (Test-Path $finalCompiledRoot)) -and (Test-Path $backupRoot)) {
            Move-Item -Path $backupRoot -Destination $finalCompiledRoot
        }
        throw
    }

    if (Test-Path $backupRoot) {
        Remove-PathIfExists -PathToRemove $backupRoot
        $backupRoot = ""
    }

    Cleanup-LegacyCanonicalSourceRoot -RepoRoot $repoRoot

    if (-not $KeepScratch) {
        Remove-PathIfExists -PathToRemove $ScratchRoot
    }

    $refreshSucceeded = $true

    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  Corpus refresh complete"
    Write-Host "  Live corpus : $finalCompiledRoot"
    Write-Host "  Assets      : $($stageAssets.Count)"
    Write-Host "  Cleanup     : transient downloads removed"
    Write-Host "========================================================"
}
finally {
    if ((-not $refreshSucceeded) -and (-not [string]::IsNullOrWhiteSpace($backupRoot)) -and (Test-Path $backupRoot) -and (-not (Test-Path $GmDagRoot))) {
        Move-Item -Path $backupRoot -Destination $GmDagRoot
    }
    Stop-Transcript | Out-Null
}
