# Simplified Brain Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [ValidateSet("step", "async")]
    [string]$Mode = "",
    [ValidateSet("mambapy", "gru")]
    [string]$TemporalCore = "gru",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$uvArgs = @(
    "run",
    "--project", "$repoRoot\projects\actor",
    "python",
    "-m",
    "navi_actor.cli",
    "serve",
    "--temporal-core", "$TemporalCore",
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
