$ErrorActionPreference = "Stop"

$root = "data/scenes"
$outFile = "data/scenes/scene_manifest_all.json"

# Find all GLB files recursively
$files = Get-ChildItem -Path $root -Recurse -Filter "*.glb"
$sceneList = @()

foreach ($file in $files) {
    # Skip small/invalid files (e.g. < 100KB)
    if ($file.Length -lt 100000) { continue }
    
    $dataset = "misc"
    if ($file.FullName -match "hssd") { $dataset = "hssd" }
    elseif ($file.FullName -match "replicacad") { $dataset = "replicacad" }
    elseif ($file.FullName -match "habitat_test_scenes") { $dataset = "habitat_test_scenes" }
    
    # Use forward slashes for JSON compatibility
    # [char]92 is backslash, [char]47 is forward slash
    $path = $file.FullName.Replace([string][char]92, [string][char]47)
    
    $sceneObj = @{
        path = $path
        size_mb = [math]::Round($file.Length / 1MB, 2)
        dataset = $dataset
    }
    $sceneList += $sceneObj
}

$manifest = @{
    generated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    scene_count = $sceneList.Count
    scenes = $sceneList
}

$manifest | ConvertTo-Json -Depth 3 | Set-Content $outFile -Encoding UTF8
Write-Host "Generated manifest with $($sceneList.Count) scenes at $outFile"
