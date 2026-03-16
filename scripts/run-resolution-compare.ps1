[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Profiles = @("256x48", "512x96", "768x144"),
    [string]$Scene = "",
    [string]$Manifest = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [switch]$AutoCompileGmDag,
    [int]$GmDagResolution = 512,
    [int]$Actors = 4,
    [int]$TotalSteps = 2048,
    [int]$MinibatchSize = 64,
    [int]$PpoEpochs = 1,
    [double]$ExistentialTax = -0.02,
    [double]$EntropyCoeff = 0.02,
    [double]$LearningRate = 5e-4,
    [int]$BpttLen = 8,
    [int]$RolloutLength = 512,
    [int]$CheckpointEvery = 0,
    [ValidateSet("gru", "mambapy")]
    [string]$TemporalCore = "gru",
    [int]$Repeats = 3,
    [switch]$ProfileCudaEvents,
    [int]$BaseActorTelemetryPort = 5680,
    [string]$OutputRoot = "artifacts/benchmarks/resolution-compare",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Resolve-OutputRoot {
    param([string]$RepoRoot, [string]$Root)

    if ([System.IO.Path]::IsPathRooted($Root)) {
        return $Root
    }
    return Join-Path $RepoRoot $Root
}

function Stop-ProcessTreeById {
    param([int]$ProcessId)

    if ($ProcessId -le 0) {
        return
    }

    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        return
    }

    try {
        & taskkill /PID $ProcessId /T /F *> $null
    }
    catch {
    }
}

function Stop-NaviProcesses {
    param([int]$ActorTelemetryPort)

    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*navi-environment*" -or
            $_.CommandLine -like "*navi-actor*"
        )
    } | ForEach-Object {
        Stop-ProcessTreeById -ProcessId $_.ProcessId
    }

    $portsPattern = "5559|5560|5557|$ActorTelemetryPort"
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        $held = netstat -ano 2>$null | Select-String $portsPattern
        if (-not $held) {
            break
        }
        foreach ($line in $held) {
            if ($line -match '\s(\d+)\s*$') {
                $procId = [int]$Matches[1]
                if ($procId -gt 0) {
                    Stop-ProcessTreeById -ProcessId $procId
                }
            }
        }
        Start-Sleep -Seconds 1
    }
}

function Merge-Logs {
    param(
        [string]$StdErrPath,
        [string]$StdOutPath,
        [string]$MergedPath
    )

    $mergedLines = @()
    if (Test-Path $StdErrPath) {
        $mergedLines += Get-Content $StdErrPath
    }
    if (Test-Path $StdOutPath) {
        $mergedLines += Get-Content $StdOutPath
    }
    Set-Content -Encoding UTF8 -Path $MergedPath -Value $mergedLines
}

function Measure-PropertyMean {
    param(
        [object[]]$Rows,
        [string]$Property
    )

    $values = @($Rows | Where-Object { $null -ne $_.$Property } | ForEach-Object { [double]$_.$Property })
    if ($values.Count -eq 0) {
        return $null
    }
    return [double](($values | Measure-Object -Average).Average)
}

function Measure-PropertyMedian {
    param(
        [object[]]$Rows,
        [string]$Property
    )

    $values = @($Rows | Where-Object { $null -ne $_.$Property } | ForEach-Object { [double]$_.$Property } | Sort-Object)
    if ($values.Count -eq 0) {
        return $null
    }

    $mid = [int]($values.Count / 2)
    if (($values.Count % 2) -eq 1) {
        return [double]$values[$mid]
    }
    return [double](($values[$mid - 1] + $values[$mid]) / 2.0)
}

function Parse-ResolutionProfile {
    param([string]$Profile)

    if ([string]::IsNullOrWhiteSpace($Profile)) {
        throw "Resolution profile must not be empty."
    }

    $match = [regex]::Match($Profile.Trim(), '^(?<az>\d+)x(?<el>\d+)$')
    if (-not $match.Success) {
        throw "Invalid resolution profile '$Profile'. Use AxE format such as 256x48 or 512x96."
    }

    $az = [int]$match.Groups['az'].Value
    $el = [int]$match.Groups['el'].Value
    if ($az -le 0 -or $el -le 0) {
        throw "Resolution profile '$Profile' must contain positive integers."
    }

    return [pscustomobject]@{
        profile = "${az}x${el}"
        azimuth_bins = $az
        elevation_bins = $el
        rays_per_actor = $az * $el
    }
}

function Invoke-ResolutionRun {
    param(
        [string]$RepoRoot,
        [pscustomobject]$Resolution,
        [int]$RepeatIndex,
        [string]$RunRoot,
        [string]$SummaryScript,
        [string]$Scene,
        [string]$Manifest,
        [string]$CorpusRoot,
        [string]$GmDagRoot,
        [string]$GmDagFile,
        [bool]$AutoCompileGmDag,
        [int]$GmDagResolution,
        [int]$Actors,
        [int]$TotalSteps,
        [int]$MinibatchSize,
        [int]$PpoEpochs,
        [double]$ExistentialTax,
        [double]$EntropyCoeff,
        [double]$LearningRate,
        [int]$BpttLen,
        [int]$RolloutLength,
        [int]$CheckpointEvery,
        [string]$TemporalCore,
        [bool]$ProfileCudaEvents,
        [int]$ActorTelemetryPort,
        [string]$PythonVersion
    )

    $checkpointDir = Join-Path $RunRoot "checkpoints"
    $logDir = Join-Path $RunRoot "logs"
    $stdoutLog = Join-Path $logDir "train.out.log"
    $stderrLog = Join-Path $logDir "train.err.log"
    $artifactLog = Join-Path $RunRoot "train.log"
    $summaryJson = Join-Path $RunRoot "summary.json"

    foreach ($dir in @($RunRoot, $checkpointDir, $logDir)) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    $actorArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $RepoRoot "projects\actor"),
        "python", "-m", "navi_actor.cli", "train",
        "--actors", "$Actors",
        "--temporal-core", $TemporalCore,
        "--azimuth-bins", "$($Resolution.azimuth_bins)",
        "--elevation-bins", "$($Resolution.elevation_bins)",
        "--total-steps", "$TotalSteps",
        "--actor-pub", "tcp://*:$ActorTelemetryPort",
        "--checkpoint-every", "$CheckpointEvery",
        "--checkpoint-dir", $checkpointDir,
        "--minibatch-size", "$MinibatchSize",
        "--ppo-epochs", "$PpoEpochs",
        "--existential-tax", "$ExistentialTax",
        "--entropy-coeff", "$EntropyCoeff",
        "--learning-rate", "$LearningRate",
        "--bptt-len", "$BpttLen",
        "--rollout-length", "$RolloutLength",
        "--compile-resolution", "$GmDagResolution",
        "--shuffle",
        "--no-emit-observation-stream"
    )

    if (-not [string]::IsNullOrWhiteSpace($Scene)) {
        $actorArgs += @("--scene", $Scene)
    }
    if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        $actorArgs += @("--manifest", $Manifest)
    }
    if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) {
        $actorArgs += @("--corpus-root", $CorpusRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) {
        $actorArgs += @("--gmdag-root", $GmDagRoot)
    }
    if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) {
        $actorArgs += @("--gmdag-file", $GmDagFile)
    }
    if ($AutoCompileGmDag) {
        $actorArgs += "--force-corpus-refresh"
    }
    if ($ProfileCudaEvents) {
        $actorArgs += "--profile-cuda-events"
    }

    Stop-NaviProcesses -ActorTelemetryPort $ActorTelemetryPort
    Start-Sleep -Seconds 2

    Write-Host ("[{0} r{1}] Starting bounded trainer run..." -f $Resolution.profile, $RepeatIndex)
    $wall = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process `
        -FilePath "uv" `
        -ArgumentList $actorArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -NoNewWindow `
        -PassThru `
        -Wait
    $wall.Stop()

    Merge-Logs -StdErrPath $stderrLog -StdOutPath $stdoutLog -MergedPath $artifactLog

    if ($proc.ExitCode -ne 0) {
        throw "Resolution comparison run for '$($Resolution.profile)' repeat $RepeatIndex failed with exit code $($proc.ExitCode). See $artifactLog"
    }

    $null = & powershell -ExecutionPolicy Bypass -File $SummaryScript -LogPath $artifactLog -OutputJson $summaryJson
    if ($LASTEXITCODE -ne 0) {
        throw "Summary generation failed for '$($Resolution.profile)' repeat $RepeatIndex."
    }

    $summary = Get-Content $summaryJson -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        repeat = $RepeatIndex
        profile = $Resolution.profile
        azimuth_bins = $Resolution.azimuth_bins
        elevation_bins = $Resolution.elevation_bins
        rays_per_actor = $Resolution.rays_per_actor
        actor_pub = $summary.actor_pub
        steady_sps_mean = $summary.steady_sps_mean
        env_ms_mean = $summary.env_ms_mean
        mem_ms_mean = $summary.mem_ms_mean
        trans_ms_mean = $summary.trans_ms_mean
        host_extract_ms_mean = $summary.host_extract_ms_mean
        telemetry_publish_ms_mean = $summary.telemetry_publish_ms_mean
        ppo_update_ms_mean = $summary.ppo_update_ms_mean
        backward_ms_mean = $summary.backward_ms_mean
        gpu_backward_ms_mean = $summary.gpu_backward_ms_mean
        ppo_update_source = $summary.ppo_update_source
        final_checkpoint_path = $summary.final_checkpoint_path
        wall_seconds = [math]::Round($wall.Elapsed.TotalSeconds, 3)
        summary_json = $summaryJson
        train_log = $artifactLog
    }
}

$repoRoot = Get-RepoRoot
$resolvedOutputRoot = Resolve-OutputRoot -RepoRoot $repoRoot -Root $OutputRoot
if (-not (Test-Path $resolvedOutputRoot)) {
    New-Item -ItemType Directory -Path $resolvedOutputRoot -Force | Out-Null
}

$resolutions = @($Profiles | ForEach-Object { Parse-ResolutionProfile -Profile $_ })
if ($resolutions.Count -eq 0) {
    throw "At least one resolution profile is required."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path $resolvedOutputRoot "resolution-compare-$timestamp"
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

$summaryScript = Join-Path $repoRoot "scripts\summarize-bounded-train-log.ps1"
$baselineRays = [double]$resolutions[0].rays_per_actor
$runResults = @()

Write-Host "========================================================"
Write-Host "  Navi Observation Resolution Comparison"
Write-Host "  Temporal Core : $TemporalCore"
Write-Host "  Actors        : $Actors"
Write-Host "  Steps         : $TotalSteps"
Write-Host "  Profiles      : $($resolutions.profile -join ', ')"
Write-Host "  Repeats       : $Repeats"
Write-Host "  CUDA Profile  : $ProfileCudaEvents"
Write-Host "  Artifact Root : $artifactRoot"
Write-Host "========================================================"

for ($index = 0; $index -lt $resolutions.Count; $index++) {
    $resolution = $resolutions[$index]
    $profileRoot = Join-Path $artifactRoot $resolution.profile
    if (-not (Test-Path $profileRoot)) {
        New-Item -ItemType Directory -Path $profileRoot -Force | Out-Null
    }

    $repeatRuns = @()
    for ($repeatIndex = 1; $repeatIndex -le $Repeats; $repeatIndex++) {
        $runRoot = Join-Path $profileRoot ("repeat-{0:D2}" -f $repeatIndex)
        $port = $BaseActorTelemetryPort + ($index * 100) + ($repeatIndex - 1)
        $repeatRuns += Invoke-ResolutionRun `
            -RepoRoot $repoRoot `
            -Resolution $resolution `
            -RepeatIndex $repeatIndex `
            -RunRoot $runRoot `
            -SummaryScript $summaryScript `
            -Scene $Scene `
            -Manifest $Manifest `
            -CorpusRoot $CorpusRoot `
            -GmDagRoot $GmDagRoot `
            -GmDagFile $GmDagFile `
            -AutoCompileGmDag ([bool]$AutoCompileGmDag) `
            -GmDagResolution $GmDagResolution `
            -Actors $Actors `
            -TotalSteps $TotalSteps `
            -MinibatchSize $MinibatchSize `
            -PpoEpochs $PpoEpochs `
            -ExistentialTax $ExistentialTax `
            -EntropyCoeff $EntropyCoeff `
            -LearningRate $LearningRate `
            -BpttLen $BpttLen `
            -RolloutLength $RolloutLength `
            -CheckpointEvery $CheckpointEvery `
            -TemporalCore $TemporalCore `
            -ProfileCudaEvents ([bool]$ProfileCudaEvents) `
            -ActorTelemetryPort $port `
            -PythonVersion $PythonVersion
    }

    $profileSummaryJson = Join-Path $profileRoot "summary.json"
    $profileSummary = [ordered]@{
        profile = $resolution.profile
        azimuth_bins = $resolution.azimuth_bins
        elevation_bins = $resolution.elevation_bins
        rays_per_actor = $resolution.rays_per_actor
        rays_scale_vs_first = [math]::Round(($resolution.rays_per_actor / $baselineRays), 3)
        repeats = $Repeats
        actor_pubs = @($repeatRuns | ForEach-Object { $_.actor_pub })
        steady_sps_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'steady_sps_mean'
        steady_sps_median = Measure-PropertyMedian -Rows $repeatRuns -Property 'steady_sps_mean'
        env_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'env_ms_mean'
        env_ms_median = Measure-PropertyMedian -Rows $repeatRuns -Property 'env_ms_mean'
        mem_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'mem_ms_mean'
        trans_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'trans_ms_mean'
        host_extract_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'host_extract_ms_mean'
        telemetry_publish_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'telemetry_publish_ms_mean'
        ppo_update_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'ppo_update_ms_mean'
        ppo_update_ms_median = Measure-PropertyMedian -Rows $repeatRuns -Property 'ppo_update_ms_mean'
        backward_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'backward_ms_mean'
        backward_ms_median = Measure-PropertyMedian -Rows $repeatRuns -Property 'backward_ms_mean'
        gpu_backward_ms_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'gpu_backward_ms_mean'
        gpu_backward_ms_median = Measure-PropertyMedian -Rows $repeatRuns -Property 'gpu_backward_ms_mean'
        wall_seconds_mean = Measure-PropertyMean -Rows $repeatRuns -Property 'wall_seconds'
        runs = @($repeatRuns)
    }
    $profileSummary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $profileSummaryJson
    $runResults += [pscustomobject]$profileSummary
}

$comparison = [ordered]@{
    artifact_root = $artifactRoot
    temporal_core = $TemporalCore
    actors = $Actors
    total_steps = $TotalSteps
    repeats = $Repeats
    rollout_length = $RolloutLength
    minibatch_size = $MinibatchSize
    ppo_epochs = $PpoEpochs
    compile_resolution = $GmDagResolution
    profile_cuda_events = [bool]$ProfileCudaEvents
    profiles = @($resolutions.profile)
    runs = @($runResults)
}

$comparisonJson = Join-Path $artifactRoot "comparison-summary.json"
$comparisonCsv = Join-Path $artifactRoot "comparison-summary.csv"
$comparison | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $comparisonJson

$runResults | ForEach-Object {
    [pscustomobject]@{
        profile = $_.profile
        azimuth_bins = $_.azimuth_bins
        elevation_bins = $_.elevation_bins
        rays_per_actor = $_.rays_per_actor
        rays_scale_vs_first = $_.rays_scale_vs_first
        repeats = $_.repeats
        steady_sps_mean = $_.steady_sps_mean
        steady_sps_median = $_.steady_sps_median
        env_ms_mean = $_.env_ms_mean
        env_ms_median = $_.env_ms_median
        trans_ms_mean = $_.trans_ms_mean
        mem_ms_mean = $_.mem_ms_mean
        ppo_update_ms_mean = $_.ppo_update_ms_mean
        ppo_update_ms_median = $_.ppo_update_ms_median
        backward_ms_mean = $_.backward_ms_mean
        gpu_backward_ms_median = $_.gpu_backward_ms_median
        wall_seconds_mean = $_.wall_seconds_mean
    }
} | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $comparisonCsv

Write-Host ""
Write-Host "Comparison artifact root: $artifactRoot"
Write-Host "Comparison summary JSON: $comparisonJson"
Write-Host "Comparison summary CSV : $comparisonCsv"
Write-Host "Measured runs:"
foreach ($run in $runResults) {
    $ppoMeanText = if ($null -eq $run.ppo_update_ms_mean) { "n/a" } else { "{0:N2}ms" -f $run.ppo_update_ms_mean }
    $ppoMedianText = if ($null -eq $run.ppo_update_ms_median) { "n/a" } else { "{0:N2}ms" -f $run.ppo_update_ms_median }
    Write-Host ("  {0,-9} rays={1,7} scale={2,5:N2}x sps_mean={3,8:N2} sps_median={4,8:N2} env_mean={5,7:N3}ms trans_mean={6,7:N3}ms ppo_mean={7} ppo_median={8}" -f $run.profile, $run.rays_per_actor, $run.rays_scale_vs_first, $run.steady_sps_mean, $run.steady_sps_median, $run.env_ms_mean, $run.trans_ms_mean, $ppoMeanText, $ppoMedianText)
}