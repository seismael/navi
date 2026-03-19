<# .SYNOPSIS
  Download Habitat scene datasets for Navi Ghost-Matrix training.

    Downloads real dataset .glb scene files from HuggingFace (no authentication required):
        - Habitat Test Scenes  (~97 MB)  — 3 indoor scenes (apartment, castle, room)
        - ReplicaCAD Stages     (~38 MB)  — 5 indoor stage assets

    Generates a scene_manifest.json listing all scenes for canonical training.

.PARAMETER DataDir
  Root directory for downloaded data. Defaults to data/scenes
  relative to the repository root.

.PARAMETER Datasets
    Comma-separated list of datasets to download.
    Valid: test_scenes, replicacad
    Default: test_scenes,replicacad

.PARAMETER PreserveExisting
    Keep existing downloaded source assets instead of forcing a refresh.
    Canonical corpus refresh defaults to overwrite-first when this switch is omitted.

.EXAMPLE
  .\download-habitat-data.ps1
  .\download-habitat-data.ps1 -DataDir "D:\scene-data"
#>
param(
    [string]$DataDir = "",
        [string]$Datasets = "test_scenes,replicacad",
    [switch]$PreserveExisting
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

$repoRoot = Get-RepoRoot
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $repoRoot "data\scenes"
}

$preserveDownloads = $PreserveExisting.IsPresent

if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}

$downloadList = $Datasets -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
foreach ($dataset in $downloadList) {
    if ($dataset -notin @("test_scenes", "replicacad")) {
        throw "Unsupported canonical dataset selection '$dataset'. Real-scene canonical downloads support only 'test_scenes' and 'replicacad'."
    }
}

$sceneManifest = @()
$HF_BASE = "https://huggingface.co/datasets/ai-habitat"

# ── Helper: download a single file ──────────────────────────────
function Get-HFFile {
    param(
        [string]$Url,
        [string]$OutFile
    )
    if (Test-Path $OutFile) {
        if ($preserveDownloads) {
            $sizeMB = [math]::Round((Get-Item $OutFile).Length / 1MB, 2)
            Write-Host "    [keep] Existing file ($sizeMB MB): $(Split-Path $OutFile -Leaf)"
            return
        }

        Remove-Item $OutFile -Force
        Write-Host "    [refresh] Replacing existing file: $(Split-Path $OutFile -Leaf)"
    }
    $name = Split-Path $OutFile -Leaf
    Write-Host "    Downloading $name..."
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    $sizeMB = [math]::Round((Get-Item $OutFile).Length / 1MB, 2)
    Write-Host "    Done: $sizeMB MB"
}

# ═══════════════════════════════════════════════════════════════
# 1) Habitat Test Scenes — 3 indoor .glb files from HuggingFace
# ═══════════════════════════════════════════════════════════════
if ($downloadList -contains "test_scenes") {
    Write-Host ""
    Write-Host "=== Habitat Test Scenes (HuggingFace) ==="

    $testScenes = @(
        @{ name = "apartment_1.glb";       size = "~48 MB" },
        @{ name = "skokloster-castle.glb"; size = "~37 MB" },
        @{ name = "van-gogh-room.glb";     size = "~21 MB" }
    )

    foreach ($scene in $testScenes) {
        $url = "${HF_BASE}/habitat_test_scenes/resolve/main/$($scene.name)?download=true"
        $outFile = Join-Path $DataDir $scene.name
        Get-HFFile -Url $url -OutFile $outFile
        $sceneManifest += @{ dataset = "habitat_test_scenes"; path = $outFile }
    }
}

# ═══════════════════════════════════════════════════════════════
# 2) ReplicaCAD Stages — public indoor stages from HuggingFace
# ═══════════════════════════════════════════════════════════════
if ($downloadList -contains "replicacad") {
    Write-Host ""
    Write-Host "=== ReplicaCAD Stages (HuggingFace) ==="

    $replicaCadRoot = Join-Path $DataDir "replicacad"
    if (-not (Test-Path $replicaCadRoot)) {
        New-Item -ItemType Directory -Path $replicaCadRoot -Force | Out-Null
    }

    $replicaCadStages = @(
        @{ name = "frl_apartment_stage.glb"; size = "~7 MB" },
        @{ name = "Stage_v3_sc0_staging.glb"; size = "~8 MB" },
        @{ name = "Stage_v3_sc1_staging.glb"; size = "~8 MB" },
        @{ name = "Stage_v3_sc2_staging.glb"; size = "~8 MB" },
        @{ name = "Stage_v3_sc3_staging.glb"; size = "~8 MB" }
    )

    foreach ($scene in $replicaCadStages) {
        $url = "${HF_BASE}/ReplicaCAD_dataset/resolve/main/stages/$($scene.name)?download=true"
        $outFile = Join-Path $replicaCadRoot $scene.name
        Get-HFFile -Url $url -OutFile $outFile
        $sceneManifest += @{ dataset = "replicacad"; path = $outFile }
    }
}

function Get-SceneDatasetLabel {
    param([string]$FullPath)

    $normalized = $FullPath.Replace('/', '\').ToLowerInvariant()
    if ($normalized -match 'replicacad|stage_v3_sc|frl_apartment_stage') {
        return 'replicacad'
    }
    if ($normalized -match 'hssd') {
        return 'hssd'
    }
    if ($normalized -match 'test_scenes|apartment_1|skokloster-castle|van-gogh-room') {
        return 'habitat_test_scenes'
    }
    return 'misc'
}

function New-SceneManifestEntry {
    param([System.IO.FileInfo]$File)

    return @{
        dataset = Get-SceneDatasetLabel -FullPath $File.FullName
        path = $File.FullName
        size_mb = [math]::Round($File.Length / 1MB, 2)
    }
}

# ═══════════════════════════════════════════════════════════════
# Generate scene manifests JSON
# ═══════════════════════════════════════════════════════════════
$manifestTimestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

$manifestPath = Join-Path $DataDir "scene_manifest.json"
$manifestData = @{
    generated   = $manifestTimestamp
    data_dir    = $DataDir
    scene_count = $sceneManifest.Count
    scenes      = $sceneManifest
}
$manifestData | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

$allSceneFiles = Get-ChildItem -Path $DataDir -Recurse -File -Include *.glb,*.obj | Where-Object { $_.Length -ge 100000 }
$recursiveManifestPath = Join-Path $DataDir "scene_manifest_all.json"
$recursiveManifestData = @{
    generated   = $manifestTimestamp
    data_dir    = $DataDir
    scene_count = $allSceneFiles.Count
    scenes      = @($allSceneFiles | Sort-Object FullName | ForEach-Object { New-SceneManifestEntry -File $_ })
}
$recursiveManifestData | ConvertTo-Json -Depth 4 | Set-Content -Path $recursiveManifestPath -Encoding UTF8

# ── Summary ───────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================================"
Write-Host "  Download complete"
Write-Host "  Data directory : $DataDir"
Write-Host "  Scene manifest : $manifestPath"
Write-Host "  All-scene manifest : $recursiveManifestPath"
Write-Host "  Downloaded scenes  : $($sceneManifest.Count)"
Write-Host "  Discovered scenes  : $($allSceneFiles.Count)"
Write-Host "========================================================"
Write-Host ""

foreach ($ds in ($sceneManifest | Group-Object { $_.dataset })) {
    Write-Host "  $($ds.Name): $($ds.Count) scenes"
}

Write-Host ""
Write-Host "Next canonical step: prepare the full discovered corpus into .gmdag assets."
Write-Host "  uv run --project ..\projects\environment navi-environment prepare-corpus --manifest $recursiveManifestPath --force-recompile"
Write-Host "  .\train.ps1"
