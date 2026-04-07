<#
.SYNOPSIS
  Refresh the canonical Navi training corpus end-to-end with HuggingFace datasets.

.DESCRIPTION
  Canonical corpus refresh runs as a staged transaction:
    1. Download source datasets from HuggingFace (10 scenes each from 6 datasets = 60 scenes).
    2. Iterate through datasets one by one:
       - Download scene GLB.
       - Compile into .gmdag.
       - Cleanup GLB immediately after compilation to save space.
    3. Promote the staged compiled corpus into `artifacts/gmdag/corpus` only after success.
    4. Remove transient downloads and scratch data after successful integration.

  Target datasets (10 scenes each):
    - ai-habitat/ReplicaCAD_dataset
    - ai-habitat/ReplicaCAD_baked_lighting
    - ai-habitat/habitat_test_scenes

.PARAMETER Datasets
    Comma-separated list of HuggingFace dataset IDs to process.
    Defaults to all requested Navi datasets.

.PARAMETER ScenesPerDataset
    Number of scenes to fetch from each dataset. Defaults to 10.
#>
param(
    [string]$DataDir = "",
    [string]$Datasets = "ai-habitat/ReplicaCAD_dataset,ai-habitat/ReplicaCAD_baked_lighting,ai-habitat/habitat_test_scenes",
    [int]$ScenesPerDataset = 10,
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$ScratchRoot = "",
    [int]$Resolution = 512,
    [int]$MinSceneBytes = 100000,
    [string]$PythonVersion = "3.12",
    [switch]$KeepScratch,
    [switch]$ForceRecompile,
    [switch]$IncludeQuake3
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

# ── HuggingFace Helpers ────────────────────────────────────────

function Get-HFFileList {
    param(
        [string]$DatasetId,
        [string]$Path = ""
    )
    $url = "https://huggingface.co/api/datasets/$DatasetId/tree/main/$Path"
    try {
        $resp = Invoke-RestMethod -Uri $url -UseBasicParsing
        return $resp
    } catch {
        Write-Warning "Failed to list HF path: $url. Error: $_"
        return @()
    }
}

function Find-HFGlbs {
    param(
        [string]$DatasetId,
        [string]$RootPath,
        [int]$Limit = 10
    )
    $found = @()
    $stack = New-Object System.Collections.Generic.Stack[string]
    $stack.Push($RootPath)

    while ($stack.Count -gt 0 -and $found.Count -lt $Limit) {
        $currentPath = $stack.Pop()
        $items = Get-HFFileList -DatasetId $DatasetId -Path $currentPath
        foreach ($item in $items) {
            if ($item.type -eq "file" -and $item.path.EndsWith(".glb")) {
                $found += $item.path
                if ($found.Count -ge $Limit) { break }
            }
            elseif ($item.type -eq "directory") {
                $stack.Push($item.path)
            }
        }
    }
    return $found
}

function Download-HFFile {
    param(
        [string]$DatasetId,
        [string]$HFPath,
        [string]$OutFile
    )
    $baseUrl = "https://huggingface.co/datasets/$DatasetId/resolve/main"
    $downloadUrl = $baseUrl + "/" + $HFPath + "?download=true"
    
    Ensure-Directory (Split-Path $OutFile)
    Write-Host "    Download URL: $downloadUrl"
    & curl.exe -L -o $OutFile $downloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code $LASTEXITCODE for $downloadUrl"
    }
    if (-not (Test-Path $OutFile)) {
        throw "Download succeeded but file missing: $OutFile"
    }
    $size = (Get-Item $OutFile).Length
    $sizeMB = [math]::Round($size / 1MB, 2)
    Write-Host "    Done: $size bytes ($sizeMB MB)"
}

# ── Main Refresh Logic ─────────────────────────────────────────

$repoRoot = Get-RepoRoot
$logsRoot = Join-Path $repoRoot "artifacts\logs"
Ensure-Directory -PathToCreate $logsRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcriptPath = Join-Path $logsRoot "corpus_refresh_$timestamp.log"
Start-Transcript -Path $transcriptPath -Force | Out-Null

$refreshSucceeded = $false
$backupRoot = ""

try {
    if ([string]::IsNullOrWhiteSpace($ScratchRoot)) {
        $ScratchRoot = Join-Path $repoRoot "artifacts\tmp\corpus-refresh"
    }
    $stageSourceRoot = Join-Path $ScratchRoot "downloads"
    $stageCompiledRoot = Join-Path $ScratchRoot "compiled"
    
    if ([string]::IsNullOrWhiteSpace($GmDagRoot)) {
        $GmDagRoot = Join-Path $repoRoot "artifacts\gmdag\corpus"
    }
    $finalCompiledRoot = $GmDagRoot

    Ensure-Directory -PathToCreate $ScratchRoot
    Ensure-Directory -PathToCreate $stageSourceRoot
    Ensure-Directory -PathToCreate $stageCompiledRoot
    Ensure-Directory -PathToCreate (Split-Path -Parent $finalCompiledRoot)

    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  Navi Professional Corpus Refresh (HuggingFace)"
    Write-Host "  Datasets     : $Datasets"
    Write-Host "  Scenes/DS    : $ScenesPerDataset"
    Write-Host "  Compiled Out : $stageCompiledRoot"
    Write-Host "  Live Corpus  : $finalCompiledRoot"
    Write-Host "  Resolution   : $Resolution"
    Write-Host "  Mode         : Iterate One-By-One (Download -> Compile -> Cleanup)"
    Write-Host "========================================================"
    Write-Host ""

    $datasetList = $Datasets -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    
    foreach ($dsId in $datasetList) {
        Write-Host "Processing dataset: $dsId"
        
        # 1. Determine the root path for scenes in this dataset
        $searchRoot = ""
        if ($dsId -eq "ai-habitat/ReplicaCAD_dataset") { $searchRoot = "stages" }
        elseif ($dsId -eq "ai-habitat/ReplicaCAD_baked_lighting") { $searchRoot = "stages" }
        elseif ($dsId -eq "ai-habitat/habitat_test_scenes") { $searchRoot = "" }
        
        # 2. Find GLB files
        $scenePaths = Find-HFGlbs -DatasetId $dsId -RootPath $searchRoot -Limit $ScenesPerDataset
        Write-Host "  Found $($scenePaths.Count) scenes in HF:$dsId"
        
        # 3. Iterate One-By-One
        foreach ($hfPath in $scenePaths) {
            Write-Host "  Processing HF Path: '$hfPath'"
            if ([string]::IsNullOrWhiteSpace($hfPath)) { continue }

            $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($hfPath)
            $safeDsName = $dsId.Replace("/", "_")
            $tempGlb = Join-Path $stageSourceRoot "$safeDsName/$sceneName.glb"
            $outputGmdag = Join-Path $stageCompiledRoot "$safeDsName/$sceneName.gmdag"

            if ((Test-Path $outputGmdag) -and (-not $ForceRecompile)) {
                Write-Host "  [skip] Scene already compiled: $sceneName"
                continue
            }

            # A. Download
            Download-HFFile -DatasetId $dsId -HFPath $hfPath -OutFile $tempGlb
            
            # B. Compile
            Write-Host "    Compiling $sceneName to .gmdag..."
            $compileArgs = @(
                "run",
                "--python", $PythonVersion,
                "--project", (Join-Path $repoRoot "projects\environment"),
                "navi-environment", "compile-gmdag",
                "--source", $tempGlb,
                "--output", $outputGmdag,
                "--resolution", "$Resolution"
            )
            & uv @compileArgs
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "    Failed to compile $sceneName. Skipping."
                Remove-Item $tempGlb -Force
                continue
            }

            # C. Cleanup Source GLB
            Write-Host "    Cleaning up source GLB..."
            Remove-Item $tempGlb -Force
        }
    }

    # 4. Generate Final Manifest
    Write-Host ""
    Write-Host "Generating compiled corpus manifest..."

    $stageManifest = Join-Path $stageCompiledRoot "gmdag_manifest.json"
    $compiledFiles = Get-ChildItem -Path $stageCompiledRoot -Filter "*.gmdag" -Recurse
    if ($compiledFiles.Count -eq 0) {
        throw "No compiled `.gmdag` files were produced. Refresh failed."
    }

    $manifestObj = @{
        generated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        source_root = $finalCompiledRoot
        gmdag_root = $finalCompiledRoot
        scene_count = $compiledFiles.Count
        requested_resolution = $Resolution
        scenes = @()
    }
    foreach ($file in $compiledFiles) {
        $relPath = $file.FullName.Replace($stageCompiledRoot, "").TrimStart('\')
        $manifestObj.scenes += @{
            source_path = $relPath
            gmdag_path = $relPath
            dataset = (Split-Path (Split-Path $file.FullName -Parent) -Leaf)
            scene_name = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        }
    }
    $manifestObj | ConvertTo-Json -Depth 10 | Set-Content -Path $stageManifest

    $backupRoot = Join-Path (Split-Path -Parent $finalCompiledRoot) ("corpus_backup_" + $timestamp)
    if (Test-Path $finalCompiledRoot) {
        Write-Host "Backing up live corpus to $backupRoot"
        Move-Item -Path $finalCompiledRoot -Destination $backupRoot
    }

    Write-Host "Promoting staged corpus to $finalCompiledRoot"
    Move-Item -Path $stageCompiledRoot -Destination $finalCompiledRoot

    if (Test-Path $backupRoot) {
        Remove-PathIfExists -PathToRemove $backupRoot
        $backupRoot = ""
    }

    if (-not $KeepScratch) {
        Remove-PathIfExists -PathToRemove $ScratchRoot
    }

    $refreshSucceeded = $true

    # ── Optional: Quake 3 Arena Maps ───────────────────────────
    if ($IncludeQuake3) {
        Write-Host ""
        Write-Host "=== Quake 3 Arena Maps (via download-quake3-maps.ps1) ==="
        $q3Output = Join-Path $finalCompiledRoot "quake3-arenas"
        $q3Args = @(
            "-OutputRoot", $q3Output,
            "-Resolution", "$Resolution",
            "-PythonVersion", $PythonVersion
        )
        if ($ForceRecompile) { $q3Args += "-ForceRecompile" }
        $q3Script = Join-Path $PSScriptRoot "download-quake3-maps.ps1"
        & $q3Script @q3Args
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Quake 3 map download completed with errors."
        }
    }

    Write-Host ""
    Write-Host "========================================================"
    Write-Host "  Corpus refresh complete"
    Write-Host "  Live corpus : $finalCompiledRoot"
    Write-Host "  Assets      : $($stageAssets.Count)"
    Write-Host "========================================================"

} finally {
    if ((-not $refreshSucceeded) -and (-not [string]::IsNullOrWhiteSpace($backupRoot)) -and (Test-Path $backupRoot) -and (-not (Test-Path $GmDagRoot))) {
        Move-Item -Path $backupRoot -Destination $GmDagRoot
    }
    Stop-Transcript | Out-Null
}
