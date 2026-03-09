# Simplified Environment Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$PythonVersion = "3.12",
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
    "--python", $PythonVersion,
    "--project", "$repoRoot\projects\environment",
    "navi-environment",
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
