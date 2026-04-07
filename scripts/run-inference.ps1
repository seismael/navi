<# .SYNOPSIS
  Run in-process canonical inference with a trained checkpoint.
  Wraps `navi-actor infer` with standard defaults.

.EXAMPLE
    # Inference on the full discovered corpus with dashboard
    .\run-inference.ps1 -Checkpoint ".\artifacts\checkpoints\bc_base_model.pt"

    # Deterministic inference on a specific dataset, no dashboard
    .\run-inference.ps1 -Checkpoint ".\my_model.pt" -Deterministic -Datasets "ai-habitat_ReplicaCAD_baked_lighting" -NoDashboard

    # Bounded inference: 10000 steps
    .\run-inference.ps1 -Checkpoint ".\my_model.pt" -TotalSteps 10000
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [int]$Actors = 4,
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [ValidateSet("gru", "mambapy", "mamba2")]
    [string]$TemporalCore = "mamba2",
    [int]$TotalSteps = 0,
    [int]$TotalEpisodes = 0,
    [switch]$Deterministic,
    [int]$LogEvery = 100,
    [switch]$NoDashboard,

    [string]$Scene = "",
    [string]$Manifest = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [string]$Datasets = "",
    [string]$ExcludeDatasets = "",
    [int]$GmDagResolution = 512,
    [string]$ActorPub = "tcp://localhost:5557",

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
    "infer",
    "--checkpoint", $Checkpoint,
    "--actors", "$Actors",
    "--temporal-core", "$TemporalCore",
    "--actor-pub", $ActorPub,
    "--log-every", $LogEvery,
    "--compile-resolution", $GmDagResolution,
    "--azimuth-bins", "$AzimuthBins",
    "--elevation-bins", "$ElevationBins"
)

if ($TotalSteps -gt 0) {
    $uvArgs += @("--total-steps", $TotalSteps)
}
if ($TotalEpisodes -gt 0) {
    $uvArgs += @("--total-episodes", $TotalEpisodes)
}
if ($Deterministic) {
    $uvArgs += "--deterministic"
}
if ($NoDashboard) {
    $uvArgs += "--no-emit-observation-stream"
}

if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) {
    $uvArgs += @("--gmdag-file", $GmDagFile)
}
elseif (-not [string]::IsNullOrWhiteSpace($Scene)) {
    $uvArgs += @("--scene", $Scene)
}

if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
    $uvArgs += @("--manifest", $Manifest)
}
if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
    $uvArgs += @("--corpus-root", $CorpusRoot)
}
if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) {
    $uvArgs += @("--gmdag-root", $GmDagRoot)
}
if (-not [string]::IsNullOrWhiteSpace($Datasets)) {
    $uvArgs += @("--datasets", $Datasets)
}
if (-not [string]::IsNullOrWhiteSpace($ExcludeDatasets)) {
    $uvArgs += @("--exclude-datasets", $ExcludeDatasets)
}

if ($ForwardArgs) {
    $uvArgs += $ForwardArgs
}

uv @uvArgs
