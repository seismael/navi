<#
.SYNOPSIS
    Download Quake 3 arena maps and compile them to .gmdag for Navi drone training.

.DESCRIPTION
    Downloads PK3 map archives from lvlworld.com, extracts BSP geometry using the
    voxel-dag bsp-to-obj converter, then compiles each OBJ to a .gmdag world cache
    via the navi-environment compile-gmdag CLI.

    Maps are selected from the curated manifest at data/quake3/quake3_map_manifest.json.
    The default output goes to artifacts/gmdag/corpus/quake3-arenas/ so dataset-based
    training discovery picks them up automatically under the "quake3-arenas" label.

.PARAMETER OutputRoot
    Root directory for compiled .gmdag outputs.
    Default: artifacts/gmdag/corpus/quake3-arenas

.PARAMETER Resolution
    Voxel-DAG compile resolution. Default: 512 (canonical).

.PARAMETER MapFilter
    Comma-separated list of map names from the manifest to download.
    If empty, downloads ALL maps in the manifest.

.PARAMETER Sources
    Comma-separated list of sources to include: "lvlworld", "openarena", or both.
    Default: "lvlworld"

.PARAMETER ForceRecompile
    Overwrite existing .gmdag outputs even if they already exist.

.PARAMETER PythonVersion
    Python version for uv run. Default: "3.12"

.PARAMETER Tessellation
    Bezier patch tessellation level for BSP conversion. Default: 4

.PARAMETER KeepIntermediate
    Keep intermediate OBJ files after compilation (for debugging).

.EXAMPLE
    .\download-quake3-maps.ps1 -MapFilter "padshop,japanese_castles"
    # Downloads just two maps and compiles them

.EXAMPLE
    .\download-quake3-maps.ps1 -Sources "lvlworld" -ForceRecompile
    # Downloads all lvlworld maps, recompiling any existing outputs
#>

[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [int]$Resolution = 512,
    [string]$MapFilter = "",
    [string]$Sources = "lvlworld",
    [switch]$ForceRecompile,
    [string]$PythonVersion = "3.12",
    [int]$Tessellation = 4,
    [switch]$KeepIntermediate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -- Resolve workspace root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptDir

# -- Resolve output root
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Join-Path (Join-Path (Join-Path $repoRoot "artifacts") "gmdag") "corpus") "quake3-arenas"
}

# -- Paths
$manifestPath = Join-Path (Join-Path $repoRoot "data") "quake3\quake3_map_manifest.json"
$tempDir = Join-Path (Join-Path $repoRoot "artifacts") "tmp\quake3-download"
$objTempDir = Join-Path $tempDir "obj"
$pk3TempDir = Join-Path $tempDir "pk3"

# -- Validate manifest
if (-not (Test-Path $manifestPath)) {
    Write-Error "Manifest not found at: $manifestPath"
    exit 1
}

Write-Host "------------------------------------------------------------"
Write-Host "  Quake 3 Arena Map Download and Compile Pipeline"
Write-Host "------------------------------------------------------------"
Write-Host "  Manifest:    $manifestPath"
Write-Host "  Output root: $OutputRoot"
Write-Host "  Resolution:  $Resolution"
Write-Host "  Sources:     $Sources"
Write-Host "  Tessellation: $Tessellation"
if ($MapFilter) { Write-Host "  Map filter:  $MapFilter" }
Write-Host "------------------------------------------------------------"

# -- Load manifest
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# -- Parse source filter
$sourceList = $Sources.Split(",") | ForEach-Object { $_.Trim().ToLower() }

# -- Parse map filter
$mapFilterSet = @{}
if (-not [string]::IsNullOrWhiteSpace($MapFilter)) {
    $MapFilter.Split(",") | ForEach-Object {
        $mapFilterSet[$_.Trim().ToLower()] = $true
    }
}

# -- Ensure temp and output dirs
foreach ($d in @($OutputRoot, $objTempDir, $pk3TempDir)) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}

# -- Collect maps to download
$mapsToProcess = @()

if ($sourceList -contains "lvlworld") {
    foreach ($map in $manifest.maps) {
        if ($map.source -ne "lvlworld") { continue }
        if ($mapFilterSet.Count -gt 0 -and -not $mapFilterSet.ContainsKey($map.name.ToLower())) {
            continue
        }
        $mapsToProcess += @{
            Name     = $map.name
            Source   = "lvlworld"
            Id       = $map.lvlworld_id
            Title    = $map.title
        }
    }
}

Write-Host ""
Write-Host "Maps to process: $($mapsToProcess.Count)"
if ($mapsToProcess.Count -eq 0) {
    Write-Host "No maps matched the filter. Check -MapFilter and -Sources parameters."
    exit 0
}

foreach ($m in $mapsToProcess) {
    Write-Host "  - $($m.Name) ($($m.Title)) [source=$($m.Source)]"
}
Write-Host ""

# -- Stats
$downloadOk = 0
$downloadFail = 0
$compileOk = 0
$compileFail = 0
$skipped = 0
$validationFail = 0
$validationResults = @()

# -- Process each map
foreach ($mapInfo in $mapsToProcess) {
    $mapName = $mapInfo.Name
    $outputGmdag = Join-Path $OutputRoot "$mapName.gmdag"

    Write-Host "------------------------------------------------------------"
    Write-Host "Processing: $mapName"

    # Skip if already compiled and not forcing
    if ((Test-Path $outputGmdag) -and -not $ForceRecompile) {
        Write-Host "  [SKIP] Already compiled: $outputGmdag"
        $skipped++
        continue
    }

    # -- Stage 1: Download PK3
    $pk3File = Join-Path $pk3TempDir "$mapName.pk3"

    if ($mapInfo.Source -eq "lvlworld") {
        $pageUrl = "https://lvlworld.com/download/id:$($mapInfo.Id)"
        Write-Host "  Fetching download page from lvlworld (id=$($mapInfo.Id))..."

        try {
            # Step 1: Fetch the download page to extract the tokenized download URL
            $pageResp = Invoke-WebRequest -Uri $pageUrl -UseBasicParsing
            $tokenMatch = [regex]::Match(
                $pageResp.Content,
                'location="/dl/"\+s\+"/(\d+)/([a-f0-9]+)/([a-f0-9]+)"'
            )
            if (-not $tokenMatch.Success) {
                throw "Could not extract download tokens from lvlworld page"
            }

            $mapId = $tokenMatch.Groups[1].Value
            $hash1 = $tokenMatch.Groups[2].Value
            $hash2 = $tokenMatch.Groups[3].Value
            $downloadUrl = "https://lvlworld.com/dl/lvl/$mapId/$hash1/$hash2"
            Write-Host "  Downloading PK3..."

            # Step 2: Download the actual PK3 file
            $curlArgs = @("-L", "-s", "-f", "-o", $pk3File, $downloadUrl)
            & curl.exe @curlArgs
            if ($LASTEXITCODE -ne 0) {
                throw "curl returned exit code $LASTEXITCODE"
            }

            # Validate it looks like a zip/pk3 (magic bytes PK = 80,75)
            if (-not (Test-Path $pk3File)) {
                throw "Downloaded file not found"
            }
            $fileBytes = [System.IO.File]::ReadAllBytes($pk3File)
            if ($fileBytes.Length -lt 1000 -or $fileBytes[0] -ne 80 -or $fileBytes[1] -ne 75) {
                throw "Downloaded file is not a valid PK3/ZIP (size=$($fileBytes.Length), magic=$($fileBytes[0]),$($fileBytes[1]))"
            }

            Write-Host "  Downloaded: $('{0:N0}' -f $fileBytes.Length) bytes"
            $downloadOk++
        }
        catch {
            Write-Host "  [FAIL] Download failed: $_" -ForegroundColor Red
            $downloadFail++
            if (Test-Path $pk3File) { Remove-Item $pk3File -Force }
            continue
        }
    }
    else {
        Write-Host "  [SKIP] Source '$($mapInfo.Source)' not yet implemented"
        continue
    }

    # -- Stage 2: Extract inner PK3 if download is a wrapper ZIP
    # lvlworld often wraps the actual PK3 inside another ZIP with a readme
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $outerZip = [System.IO.Compression.ZipFile]::OpenRead($pk3File)
        $innerPk3 = $outerZip.Entries | Where-Object { $_.Name -match '\.pk3$' } | Select-Object -First 1
        if ($innerPk3) {
            Write-Host "  Extracting inner PK3: $($innerPk3.Name) from wrapper ZIP..."
            $innerPath = Join-Path $pk3TempDir "$mapName`_inner.pk3"
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($innerPk3, $innerPath, $true)
            $outerZip.Dispose()
            # Replace the outer wrapper with the actual PK3
            Remove-Item $pk3File -Force
            Move-Item $innerPath $pk3File
            Write-Host "  Inner PK3: $('{0:N0}' -f (Get-Item $pk3File).Length) bytes"
        }
        else {
            $outerZip.Dispose()
            Write-Host "  PK3 contains BSP directly (no wrapper)"
        }
    }
    catch {
        Write-Host "  [WARN] Could not check for inner PK3: $_"
    }

    # -- Stage 3: Convert PK3/BSP to OBJ using bsp-to-obj
    $mapObjDir = Join-Path $objTempDir $mapName
    if (-not (Test-Path $mapObjDir)) {
        New-Item -ItemType Directory -Path $mapObjDir -Force | Out-Null
    }

    Write-Host "  Converting BSP to OBJ (tessellation=$Tessellation)..."
    try {
        $convertArgs = @(
            "run", "--python", $PythonVersion,
            "--project", "projects/voxel-dag",
            "bsp-to-obj",
            "--input", $pk3File,
            "--output", $mapObjDir,
            "--tessellation", "$Tessellation",
            "--export-spawns"
        )
        # Temporarily relax error preference since uv writes build info to stderr
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & uv @convertArgs
        $convertExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP
        if ($convertExit -ne 0) {
            throw "bsp-to-obj failed (exit=$convertExit)"
        }

        # Find the generated OBJ file(s)
        $objFiles = @(Get-ChildItem -Path $mapObjDir -Filter "*.obj" -Recurse)
        if ($objFiles.Count -eq 0) {
            throw "No OBJ files produced from BSP conversion"
        }
        Write-Host "  Produced $($objFiles.Count) OBJ file(s)"
    }
    catch {
        Write-Host "  [FAIL] BSP conversion failed: $_" -ForegroundColor Red
        $compileFail++
        continue
    }

    # -- Stage 3.5: Pre-compile quality validation
    # Select primary OBJ (largest file if multiple extracted)
    $primaryObj = $objFiles | Sort-Object Length -Descending | Select-Object -First 1
    Write-Host "  Validating OBJ quality..."
    $preValidationPassed = $true
    $faceCount = 0
    $objSizeBytes = $primaryObj.Length
    $spawnCount = 0

    # Check OBJ file size (minimum 10KB for meaningful geometry)
    if ($objSizeBytes -lt 10240) {
        Write-Host "  [FAIL] OBJ too small ($objSizeBytes bytes < 10KB minimum)" -ForegroundColor Red
        $preValidationPassed = $false
    }

    # Count faces in OBJ
    try {
        $faceLines = @(Select-String -Path $primaryObj.FullName -Pattern "^f " )
        $faceCount = $faceLines.Count
        if ($faceCount -lt 500) {
            Write-Host "  [FAIL] Too few faces ($faceCount < 500 minimum)" -ForegroundColor Red
            $preValidationPassed = $false
        }
        else {
            Write-Host "    Faces: $faceCount"
        }
    }
    catch {
        Write-Host "  [WARN] Could not count OBJ faces: $_"
    }

    # Check spawn points
    $spawnsFile = Join-Path $mapObjDir "spawns.json"
    if (Test-Path $spawnsFile) {
        try {
            $spawnsData = Get-Content $spawnsFile -Raw | ConvertFrom-Json
            $spawnCount = $spawnsData.Count
            if ($spawnCount -lt 2) {
                Write-Host "  [WARN] Only $spawnCount spawn point(s) (recommend >= 2)" -ForegroundColor Yellow
            }
            else {
                Write-Host "    Spawn points: $spawnCount"
            }
        }
        catch {
            Write-Host "  [WARN] Could not parse spawns.json: $_"
        }
    }
    else {
        Write-Host "  [WARN] No spawns.json found (--export-spawns may have failed)"
    }

    Write-Host "    OBJ size: $('{0:N0}' -f $objSizeBytes) bytes"

    if (-not $preValidationPassed) {
        Write-Host "  [REJECT] $mapName failed pre-compile validation" -ForegroundColor Red
        $validationFail++
        $validationResults += [PSCustomObject]@{
            Map      = $mapName
            Stage    = "pre-compile"
            Status   = "REJECTED"
            ObjSize  = $objSizeBytes
            Faces    = $faceCount
            Spawns   = $spawnCount
            GmdagSize = 0
            CheckSdfdag = "skipped"
        }
        # Cleanup and skip
        if (-not $KeepIntermediate) {
            Remove-Item $mapObjDir -Recurse -Force -ErrorAction SilentlyContinue
            Remove-Item $pk3File -Force -ErrorAction SilentlyContinue
        }
        continue
    }

    # -- Stage 4: Compile OBJ to GMDAG

    Write-Host "  Compiling OBJ to GMDAG (resolution=$Resolution)..."
    Write-Host "    Source OBJ: $($primaryObj.Name) ($('{0:N0}' -f $primaryObj.Length) bytes)"
    try {
        $compileArgs = @(
            "run", "--python", $PythonVersion,
            "--project", "projects/environment",
            "navi-environment", "compile-gmdag",
            "--source", $primaryObj.FullName,
            "--output", $outputGmdag,
            "--resolution", "$Resolution"
        )
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & uv @compileArgs
        $compileExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP
        if ($compileExit -ne 0) {
            throw "compile-gmdag failed (exit=$compileExit)"
        }

        if (-not (Test-Path $outputGmdag)) {
            throw "GMDAG output not found after compilation"
        }

        $gmdagSize = (Get-Item $outputGmdag).Length
        Write-Host "  Compiled: $outputGmdag ($('{0:N0}' -f $gmdagSize) bytes)"

        # -- Stage 4.5: Post-compile validation
        $postValidationPassed = $true
        $checkResult = "not-run"

        # Check GMDAG minimum size (50KB)
        if ($gmdagSize -lt 51200) {
            Write-Host "  [FAIL] GMDAG too small ($gmdagSize bytes < 50KB minimum)" -ForegroundColor Red
            $postValidationPassed = $false
        }

        # Run check-sdfdag to validate the compiled asset loads and is usable
        Write-Host "  Running check-sdfdag validation..."
        try {
            $checkArgs = @(
                "run", "--python", $PythonVersion,
                "--project", "projects/environment",
                "navi-environment", "check-sdfdag",
                "--gmdag-file", $outputGmdag,
                "--json"
            )
            $prevEAP2 = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $checkOutput = & uv @checkArgs 2>&1 | Out-String
            $checkExit = $LASTEXITCODE
            $ErrorActionPreference = $prevEAP2

            if ($checkExit -ne 0) {
                Write-Host "  [FAIL] check-sdfdag failed (exit=$checkExit)" -ForegroundColor Red
                $postValidationPassed = $false
                $checkResult = "FAIL (exit=$checkExit)"
            }
            else {
                $checkResult = "PASS"
                Write-Host "  [OK] check-sdfdag validation passed" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "  [WARN] check-sdfdag could not run: $_" -ForegroundColor Yellow
            $checkResult = "ERROR: $_"
        }

        # Run qualify-gmdag to validate observation quality via CUDA ray casting
        $qualifyResult = "SKIPPED"
        if ($postValidationPassed) {
            Write-Host "  Running qualify-gmdag observation quality check..."
            try {
                $qualifyArgs = @(
                    "run", "--python", $PythonVersion,
                    "--project", "projects/environment",
                    "navi-environment", "qualify-gmdag",
                    "--gmdag-file", $outputGmdag,
                    "--json"
                )
                $prevEAP3 = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $qualifyOutput = & uv @qualifyArgs 2>&1 | Out-String
                $qualifyExit = $LASTEXITCODE
                $ErrorActionPreference = $prevEAP3

                if ($qualifyExit -ne 0) {
                    $qualifyObj = $qualifyOutput | ConvertFrom-Json -ErrorAction SilentlyContinue
                    $verdict = if ($qualifyObj) { $qualifyObj.verdict } else { "ERROR" }
                    Write-Host "  [FAIL] qualify-gmdag: $verdict" -ForegroundColor Red
                    $postValidationPassed = $false
                    $qualifyResult = "FAIL ($verdict)"
                }
                else {
                    $qualifyObj = $qualifyOutput | ConvertFrom-Json
                    $qualifyResult = "$($qualifyObj.verdict) (viable=$($qualifyObj.viable_candidates)/$($qualifyObj.total_candidates))"
                    Write-Host "  [OK] qualify-gmdag: $qualifyResult" -ForegroundColor Green
                }
            }
            catch {
                Write-Host "  [WARN] qualify-gmdag could not run: $_" -ForegroundColor Yellow
                $qualifyResult = "ERROR: $_"
            }
        }

        if ($postValidationPassed) {
            Write-Host "  [OK] $mapName fully validated" -ForegroundColor Green
            $compileOk++
            $validationResults += [PSCustomObject]@{
                Map      = $mapName
                Stage    = "complete"
                Status   = "OK"
                ObjSize  = $primaryObj.Length
                Faces    = $faceCount
                Spawns   = $spawnCount
                GmdagSize = $gmdagSize
                CheckSdfdag = $checkResult
                QualifyGmdag = $qualifyResult
            }
        }
        else {
            Write-Host "  [REJECT] $mapName failed post-compile validation - removing GMDAG" -ForegroundColor Red
            Remove-Item $outputGmdag -Force -ErrorAction SilentlyContinue
            $validationFail++
            $validationResults += [PSCustomObject]@{
                Map      = $mapName
                Stage    = "post-compile"
                Status   = "REJECTED"
                ObjSize  = $primaryObj.Length
                Faces    = $faceCount
                Spawns   = $spawnCount
                GmdagSize = $gmdagSize
                CheckSdfdag = $checkResult
                QualifyGmdag = $qualifyResult
            }
        }
    }
    catch {
        Write-Host "  [FAIL] GMDAG compilation failed: $_" -ForegroundColor Red
        $compileFail++
        continue
    }

    # -- Cleanup intermediate files
    if (-not $KeepIntermediate) {
        Remove-Item $mapObjDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $pk3File -Force -ErrorAction SilentlyContinue
    }
}

# -- Final cleanup of temp dirs (if empty)
if (-not $KeepIntermediate) {
    foreach ($d in @($pk3TempDir, $objTempDir, $tempDir)) {
        if ((Test-Path $d) -and @(Get-ChildItem $d -Recurse -File).Count -eq 0) {
            Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# -- Update corpus manifest if it exists
$corpusRoot = Split-Path $OutputRoot -Parent
$manifestFile = Join-Path $corpusRoot "gmdag_manifest.json"
if ((Test-Path $manifestFile) -and $compileOk -gt 0) {
    Write-Host ""
    Write-Host "Updating corpus manifest: $manifestFile"
    try {
        $manifestJson = Get-Content $manifestFile -Raw | ConvertFrom-Json
        $newGmdags = @(Get-ChildItem -Path $OutputRoot -Filter "*.gmdag")
        $addedCount = 0
        foreach ($gf in $newGmdags) {
            $relPath = "quake3-arenas\$($gf.Name)"
            $alreadyPresent = $false
            foreach ($s in $manifestJson.scenes) {
                if ($s.scene_name -eq $gf.BaseName -and $s.dataset -eq "quake3-arenas") {
                    $alreadyPresent = $true
                    break
                }
            }
            if (-not $alreadyPresent) {
                $newEntry = [PSCustomObject]@{
                    source_path = $relPath
                    scene_name  = $gf.BaseName
                    dataset     = "quake3-arenas"
                    gmdag_path  = $relPath
                }
                $manifestJson.scenes += $newEntry
                $addedCount++
            }
        }
        if ($addedCount -gt 0) {
            $manifestJson | ConvertTo-Json -Depth 10 | Set-Content $manifestFile -Encoding UTF8
            Write-Host "  Added $addedCount new entry(ies) to manifest"
        }
        else {
            Write-Host "  Manifest already up to date"
        }
    }
    catch {
        Write-Host "  [WARN] Could not update manifest: $_"
    }
}

# -- Summary
Write-Host ""
Write-Host "============================================================"
Write-Host "  Pipeline Summary"
Write-Host "============================================================"
Write-Host "  Downloaded:         $downloadOk"
Write-Host "  Download failed:    $downloadFail"
Write-Host "  Compiled+valid:     $compileOk"
Write-Host "  Compile failed:     $compileFail"
Write-Host "  Validation failed:  $validationFail"
Write-Host "  Skipped:            $skipped"
Write-Host "  Output root:        $OutputRoot"
Write-Host "============================================================"

# -- Write validation report
if ($validationResults.Count -gt 0) {
    $reportPath = Join-Path $OutputRoot "validation_report.json"
    $report = [PSCustomObject]@{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        resolution = $Resolution
        tessellation = $Tessellation
        total_processed = $validationResults.Count
        passed = @($validationResults | Where-Object { $_.Status -eq "OK" }).Count
        rejected = @($validationResults | Where-Object { $_.Status -eq "REJECTED" }).Count
        results = $validationResults
    }
    $report | ConvertTo-Json -Depth 10 | Set-Content $reportPath -Encoding UTF8
    Write-Host ""
    Write-Host "Validation report: $reportPath"

    # Print per-map summary table
    Write-Host ""
    Write-Host "Per-map validation results:"
    Write-Host "  {0,-30} {1,-12} {2,-12} {3,10} {4,12}" -f "Map", "Status", "Stage", "Faces", "GMDAG Size"
    Write-Host "  {0,-30} {1,-12} {2,-12} {3,10} {4,12}" -f ("---" * 10), ("---" * 4), ("---" * 4), ("---" * 3), ("---" * 4)
    foreach ($vr in $validationResults) {
        $color = if ($vr.Status -eq "OK") { "Green" } else { "Red" }
        Write-Host ("  {0,-30} {1,-12} {2,-12} {3,10} {4,12}" -f $vr.Map, $vr.Status, $vr.Stage, $vr.Faces, $('{0:N0}' -f $vr.GmdagSize)) -ForegroundColor $color
    }
}

# -- List compiled outputs
$gmdagFiles = @(Get-ChildItem -Path $OutputRoot -Filter "*.gmdag" -ErrorAction SilentlyContinue)
if ($gmdagFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Compiled GMDAG files:"
    foreach ($f in $gmdagFiles) {
        Write-Host "  $($f.Name) ($('{0:N0}' -f $f.Length) bytes)"
    }
    Write-Host ""
    Write-Host "Ready-to-run commands:"
    Write-Host ""
    Write-Host "  # Training with Q3 maps only:"
    Write-Host "  .\scripts\run-ghost-stack.ps1 -Train -Datasets 'quake3-arenas'"
    Write-Host ""
    Write-Host "  # Manual exploration on Q3 corpus:"
    Write-Host "  .\scripts\run-explore-scenes.ps1 -CorpusRoot '$OutputRoot'"
    Write-Host ""
    Write-Host "  # Observe/explore a specific Q3 map:"
    $sampleGmdag = $gmdagFiles | Select-Object -First 1
    Write-Host "  uv run --project projects/auditor explore --gmdag-file '$($sampleGmdag.FullName)'"
}

if ($compileFail -gt 0 -or $downloadFail -gt 0) {
    exit 1
}
exit 0
