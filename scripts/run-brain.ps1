# Simplified Brain Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [ValidateSet("step", "async")]
    [string]$Mode = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$uvArgs = @(
    "run",
    "--project", "$repoRoot\projects\actor",
    "navi-actor",
    "serve",
    "--azimuth-bins", "$AzimuthBins",
    "--elevation-bins", "$ElevationBins"
)

if ($Mode) {
    $uvArgs += @("--mode", $Mode)
}

if ($ForwardArgs) {
    $uvArgs += $ForwardArgs
}

uv @uvArgs
