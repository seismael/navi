# Simplified Brain Launcher
param(
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48
)
$repoRoot = Resolve-Path "$PSScriptRoot\.."
uv run --project "$repoRoot\projects\actor" navi-actor serve --azimuth-bins $AzimuthBins --elevation-bins $ElevationBins $args
