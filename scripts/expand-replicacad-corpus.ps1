<#
.SYNOPSIS
  Download and compile ReplicaCAD dataset scenes into the live corpus.

.DESCRIPTION
  Incrementally adds new ReplicaCAD scenes (from any variant) to the existing
  corpus without disturbing other dataset folders. After compilation,
  each scene is quality-validated via qualify-gmdag. Regenerates the
  corpus manifest to include all scenes.

  Supports mesh repair (--Repair) for datasets with incomplete shells
  (e.g. ReplicaCAD_baked_lighting) which need hole-filling before
  voxelization to avoid high starvation from escaping rays.

.PARAMETER DatasetId
    HuggingFace dataset ID. Defaults to ai-habitat/ReplicaCAD_dataset.

.PARAMETER ScenesLimit
    Maximum number of scenes to download. 0 = all available.

.PARAMETER Repair
    Enable mesh repair (fill holes, fix normals) before compilation.
    Required for incomplete-shell datasets like ReplicaCAD_baked_lighting.
#>
param(
    [string]$DatasetId = "ai-habitat/ReplicaCAD_dataset",
    [int]$ScenesLimit = 0,
    [int]$Resolution = 512,
    [switch]$Repair,
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$corpusRoot = Join-Path $repoRoot "artifacts\gmdag\corpus"
$dsSlug = $DatasetId.Replace("/", "_")
$datasetDir = Join-Path $corpusRoot $dsSlug
$scratchDir = Join-Path $repoRoot "artifacts\tmp\replicacad-expand"
$dsId = $DatasetId

if (-not (Test-Path $datasetDir)) {
    New-Item -ItemType Directory -Path $datasetDir -Force | Out-Null
}
if (-not (Test-Path $scratchDir)) {
    New-Item -ItemType Directory -Path $scratchDir -Force | Out-Null
}

# ── Discover available scenes on HuggingFace ──────────────────────

Write-Host "Querying HuggingFace for available $dsSlug stages..."
$hfUrl = "https://huggingface.co/api/datasets/$dsId/tree/main/stages"
$items = Invoke-RestMethod -Uri $hfUrl -UseBasicParsing
$glbPaths = $items | Where-Object { $_.path -like "*.glb" } | ForEach-Object { $_.path } | Sort-Object

Write-Host "  Found $($glbPaths.Count) stages on HuggingFace"

if ($ScenesLimit -gt 0 -and $glbPaths.Count -gt $ScenesLimit) {
    $glbPaths = $glbPaths | Select-Object -First $ScenesLimit
    Write-Host "  Limited to $ScenesLimit scenes"
}

# ── Filter to scenes not already compiled ─────────────────────────

$newPaths = @()
$existingCount = 0
foreach ($hfPath in $glbPaths) {
    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($hfPath)
    $gmdagPath = Join-Path $datasetDir "$sceneName.gmdag"
    if (Test-Path $gmdagPath) {
        $existingCount++
    } else {
        $newPaths += $hfPath
    }
}

Write-Host "  Already compiled: $existingCount"
Write-Host "  New to download:  $($newPaths.Count)"

if ($newPaths.Count -eq 0) {
    Write-Host "All scenes already present. Nothing to do."
    exit 0
}

Write-Host ""
Write-Host "========================================================"
Write-Host "  $dsSlug Corpus Expansion"
Write-Host "  New scenes  : $($newPaths.Count)"
Write-Host "  Resolution  : $Resolution"
Write-Host "  Repair      : $Repair"
Write-Host "  Output      : $datasetDir"
Write-Host "========================================================"
Write-Host ""

# ── Download and compile each new scene ───────────────────────────

$compiled = 0
$failed = 0

foreach ($hfPath in $newPaths) {
    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($hfPath)
    $tempGlb = Join-Path $scratchDir "$sceneName.glb"
    $outputGmdag = Join-Path $datasetDir "$sceneName.gmdag"

    Write-Host "[$($compiled + $failed + 1)/$($newPaths.Count)] $sceneName"

    # Download
    $downloadUrl = "https://huggingface.co/datasets/$dsId/resolve/main/$($hfPath)?download=true"
    Write-Host "  Downloading..."
    & curl.exe -L -s -o $tempGlb $downloadUrl
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tempGlb)) {
        Write-Warning "  Download failed for $sceneName. Skipping."
        $failed++
        continue
    }
    $sizeMB = [math]::Round((Get-Item $tempGlb).Length / 1MB, 1)
    Write-Host "  Downloaded: ${sizeMB} MB"

    # Compile
    Write-Host "  Compiling to .gmdag (resolution=$Resolution, repair=$Repair)..."
    $compileArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment", "compile-gmdag",
        "--source", $tempGlb,
        "--output", $outputGmdag,
        "--resolution", "$Resolution"
    )
    if ($Repair) {
        $compileArgs += "--repair"
    }
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & uv @compileArgs
    $compileExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($compileExit -ne 0) {
        Write-Warning "  Compilation failed for $sceneName. Skipping."
        if (Test-Path $tempGlb) { Remove-Item $tempGlb -Force }
        $failed++
        continue
    }

    # Cleanup source
    Remove-Item $tempGlb -Force

    # Quality gate via real CUDA ray casting
    Write-Host "  Qualifying observation quality..."
    $qualifyArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment", "qualify-gmdag",
        "--gmdag-file", $outputGmdag,
        "--json"
    )
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $qualifyJson = & uv @qualifyArgs 2>$null
    $qualifyExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($qualifyExit -ne 0) {
        $qualifyObj = $qualifyJson | ConvertFrom-Json -ErrorAction SilentlyContinue
        $verdict = if ($qualifyObj) { $qualifyObj.verdict } else { "ERROR" }
        Write-Warning "  Quality gate FAILED ($verdict) for $sceneName. Removing."
        Remove-Item $outputGmdag -Force -ErrorAction SilentlyContinue
        $failed++
        continue
    }
    $qualifyObj = $qualifyJson | ConvertFrom-Json
    $verdict = $qualifyObj.verdict
    Write-Host "  Quality: $verdict (viable=$($qualifyObj.viable_candidates)/$($qualifyObj.total_candidates), starvation=$([math]::Round($qualifyObj.best_starvation * 100, 1))%)"

    $compiled++
    $gmdagSize = [math]::Round((Get-Item $outputGmdag).Length / 1MB, 1)
    Write-Host "  Done: ${gmdagSize} MB .gmdag"
}

# ── Regenerate manifest ───────────────────────────────────────────

Write-Host ""
Write-Host "Regenerating corpus manifest..."

$allGmdags = Get-ChildItem -Path $corpusRoot -Filter "*.gmdag" -Recurse
$manifestObj = @{
    generated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    source_root = $corpusRoot
    gmdag_root = $corpusRoot
    scene_count = $allGmdags.Count
    requested_resolution = $Resolution
    scenes = @()
}
foreach ($file in ($allGmdags | Sort-Object FullName)) {
    $relPath = $file.FullName.Replace($corpusRoot, "").TrimStart('\')
    $manifestObj.scenes += @{
        source_path = $relPath
        gmdag_path = $relPath
        dataset = (Split-Path (Split-Path $file.FullName -Parent) -Leaf)
        scene_name = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    }
}
$manifestPath = Join-Path $corpusRoot "gmdag_manifest.json"
$manifestObj | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath

# Cleanup scratch
if (Test-Path $scratchDir) {
    Remove-Item -Path $scratchDir -Recurse -Force
}

Write-Host ""
Write-Host "========================================================"
Write-Host "  Expansion Complete"
Write-Host "  New scenes compiled : $compiled"
Write-Host "  Failed              : $failed"
Write-Host "  Total corpus scenes : $($allGmdags.Count)"
Write-Host "  Manifest            : $manifestPath"
Write-Host "========================================================"
