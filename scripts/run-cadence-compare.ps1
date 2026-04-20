<#
.SYNOPSIS
    H2 (PPO update cadence) bake-off wrapper. Sweeps `--rollout-length` across
    multiple values and produces side-by-side throughput + learning-quality
    summaries for the canonical trainer.

.DESCRIPTION
    Modeled on `run-temporal-compare.ps1`. Runs N bounded canonical-trainer
    invocations per cadence value (default 3 repeats), invokes the existing
    `summarize-bounded-train-log.ps1` to extract steady SPS, env/PPO/backward
    means, then writes one `comparison-summary.json` covering all cadences.

    AGENTS.md compliance:
      §2.7 Update-Frequency Rule  : explicitly authorises cadence sweeps on
                                    the one canonical trainer surface.
      §3.1 Benchmark Gate         : results from this wrapper are the required
                                    A/B evidence before any default flip in
                                    `projects/actor/src/navi_actor/config.py`.
      §3.3.1 Hot-Path Discipline  : runs the canonical `train` CLI unchanged;
                                    no monitor-only launcher, no parallel
                                    trainer mode introduced.
      §2.6 Strict Contract Rule   : this wrapper does NOT change defaults.
                                    Promotion of a winning cadence requires a
                                    follow-up change updating config.py + docs
                                    + tests in one pass.

.EXAMPLE
    # Default sweep on the active machine (256 baseline + 512, 1024 candidates).
    powershell -ExecutionPolicy Bypass -File .\scripts\run-cadence-compare.ps1

.EXAMPLE
    # Explicit cadence list, single repeat for a quick smoke comparison.
    powershell -ExecutionPolicy Bypass -File .\scripts\run-cadence-compare.ps1 `
        -RolloutLengths 256,512 -Repeats 1 -TotalSteps 1024
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [int[]]$RolloutLengths = @(256, 512),
    [string]$Scene = "",
    [string]$Manifest = "",
    [string]$CorpusRoot = "",
    [string]$GmDagRoot = "",
    [string]$GmDagFile = "",
    [switch]$AutoCompileGmDag,
    [int]$GmDagResolution = 512,
    [ValidateSet("gru", "mambapy", "mamba2")]
    [string[]]$TemporalCores = @("mamba2", "gru"),
    [int]$TotalSteps = 4096,
    [int]$AzimuthBins = 256,
    [int]$ElevationBins = 48,
    [int]$MinibatchSize = 64,
    [int]$PpoEpochs = 2,
    [double]$ExistentialTax = -0.02,
    [double]$EntropyCoeff = 0.01,
    [double]$LearningRate = 3e-4,
    [int]$BpttLen = 8,
    [int]$CheckpointEvery = 0,
    [int]$Repeats = 3,
    [switch]$ProfileCudaEvents,
    [int]$BaseActorTelemetryPort = 5594,
    [string]$OutputRoot = "artifacts/benchmarks/cadence-compare",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

trap [System.Management.Automation.BreakException] {
    Write-Host "`nInterrupted. Cleaning up Navi processes..."
    Stop-NaviProcesses -ActorTelemetryPort $BaseActorTelemetryPort
    exit 1
}

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

function Measure-PropertyMean {
    param([object[]]$Rows, [string]$Property)
    $values = @($Rows | Where-Object { $null -ne $_.$Property } | ForEach-Object { [double]$_.$Property })
    if ($values.Count -eq 0) { return $null }
    return [double](($values | Measure-Object -Average).Average)
}

function Measure-PropertyMedian {
    param([object[]]$Rows, [string]$Property)
    $values = @($Rows | Where-Object { $null -ne $_.$Property } | ForEach-Object { [double]$_.$Property } | Sort-Object)
    if ($values.Count -eq 0) { return $null }
    $mid = [int]($values.Count / 2)
    if (($values.Count % 2) -eq 1) {
        return [double]$values[$mid]
    }
    return [double](($values[$mid - 1] + $values[$mid]) / 2.0)
}

function Merge-Logs {
    param([string]$StdErrPath, [string]$StdOutPath, [string]$MergedPath)
    $mergedLines = @()
    if (Test-Path $StdErrPath) { $mergedLines += Get-Content $StdErrPath }
    if (Test-Path $StdOutPath) { $mergedLines += Get-Content $StdOutPath }
    Set-Content -Encoding UTF8 -Path $MergedPath -Value $mergedLines
}

function Invoke-CadenceRun {
    param(
        [string]$RepoRoot,
        [int]$RolloutLength,
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
        [string]$TemporalCore,
        [int]$TotalSteps,
        [int]$AzimuthBins,
        [int]$ElevationBins,
        [int]$MinibatchSize,
        [int]$PpoEpochs,
        [double]$ExistentialTax,
        [double]$EntropyCoeff,
        [double]$LearningRate,
        [int]$BpttLen,
        [int]$CheckpointEvery,
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
        "--actors", "4",
        "--temporal-core", $TemporalCore,
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins",
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
        "--no-auto-resume",
        "--shuffle"
    )

    if (-not [string]::IsNullOrWhiteSpace($Scene)) { $actorArgs += @("--scene", $Scene) }
    if (-not [string]::IsNullOrWhiteSpace($Manifest)) { $actorArgs += @("--manifest", $Manifest) }
    if (-not [string]::IsNullOrWhiteSpace($CorpusRoot)) { $actorArgs += @("--corpus-root", $CorpusRoot) }
    if (-not [string]::IsNullOrWhiteSpace($GmDagRoot)) { $actorArgs += @("--gmdag-root", $GmDagRoot) }
    if (-not [string]::IsNullOrWhiteSpace($GmDagFile)) { $actorArgs += @("--gmdag-file", $GmDagFile) }
    if ($AutoCompileGmDag) { $actorArgs += "--force-corpus-refresh" }
    if ($ProfileCudaEvents) { $actorArgs += "--profile-cuda-events" }

    Stop-NaviProcesses -ActorTelemetryPort $ActorTelemetryPort
    Start-Sleep -Seconds 2

    Write-Host ("[rollout={0} r{1}] Starting bounded trainer run..." -f $RolloutLength, $RepeatIndex)
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
        throw "Cadence comparison run for rollout=$RolloutLength repeat $RepeatIndex failed with exit code $($proc.ExitCode). See $artifactLog"
    }

    $null = & powershell -ExecutionPolicy Bypass -File $SummaryScript -LogPath $artifactLog -OutputJson $summaryJson
    if ($LASTEXITCODE -ne 0) {
        throw "Summary generation failed for rollout=$RolloutLength repeat $RepeatIndex."
    }

    $summary = Get-Content $summaryJson -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        repeat = $RepeatIndex
        rollout_length = $RolloutLength
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

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path $resolvedOutputRoot "cadence-compare-$timestamp"
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

$summaryScript = Join-Path $repoRoot "scripts\summarize-bounded-train-log.ps1"
$runResults = @()

Write-Host "========================================================"
Write-Host "  Navi PPO Cadence Comparison (H2)"
Write-Host "  Rollout Lengths : $($RolloutLengths -join ', ')"
Write-Host "  Temporal Cores  : $($TemporalCores -join ', ')"
Write-Host "  Steps / run     : $TotalSteps"
Write-Host "  Resolution      : ${AzimuthBins}x${ElevationBins}"
Write-Host "  Repeats         : $Repeats"
Write-Host "  CUDA Profile    : $ProfileCudaEvents"
Write-Host "  Artifact Root   : $artifactRoot"
Write-Host ("  Total runs      : {0}" -f ($RolloutLengths.Count * $TemporalCores.Count * $Repeats))
Write-Host "========================================================"

for ($coreIndex = 0; $coreIndex -lt $TemporalCores.Count; $coreIndex++) {
    $temporalCore = $TemporalCores[$coreIndex]
    $coreRoot = Join-Path $artifactRoot $temporalCore
    if (-not (Test-Path $coreRoot)) {
        New-Item -ItemType Directory -Path $coreRoot -Force | Out-Null
    }

    for ($index = 0; $index -lt $RolloutLengths.Count; $index++) {
        $rolloutLength = $RolloutLengths[$index]
        $cadenceRoot = Join-Path $coreRoot ("rollout-{0}" -f $rolloutLength)
        if (-not (Test-Path $cadenceRoot)) {
            New-Item -ItemType Directory -Path $cadenceRoot -Force | Out-Null
        }

        $repeatRuns = @()
        for ($repeatIndex = 1; $repeatIndex -le $Repeats; $repeatIndex++) {
            $runRoot = Join-Path $cadenceRoot ("repeat-{0:D2}" -f $repeatIndex)
            $port = $BaseActorTelemetryPort + ($coreIndex * 1000) + ($index * 100) + ($repeatIndex - 1)
            $repeatRuns += Invoke-CadenceRun `
                -RepoRoot $repoRoot `
                -RolloutLength $rolloutLength `
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
                -TemporalCore $temporalCore `
                -TotalSteps $TotalSteps `
                -AzimuthBins $AzimuthBins `
                -ElevationBins $ElevationBins `
                -MinibatchSize $MinibatchSize `
                -PpoEpochs $PpoEpochs `
                -ExistentialTax $ExistentialTax `
                -EntropyCoeff $EntropyCoeff `
                -LearningRate $LearningRate `
                -BpttLen $BpttLen `
                -CheckpointEvery $CheckpointEvery `
                -ProfileCudaEvents ([bool]$ProfileCudaEvents) `
                -ActorTelemetryPort $port `
                -PythonVersion $PythonVersion
        }

        $cadenceSummaryJson = Join-Path $cadenceRoot "summary.json"
        $cadenceSummary = [ordered]@{
            temporal_core = $temporalCore
            rollout_length = $rolloutLength
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
            repeat_runs = @($repeatRuns)
        }
        $cadenceSummary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $cadenceSummaryJson
        $runResults += [pscustomobject]$cadenceSummary
    }
}

$comparison = [ordered]@{
    artifact_root = $artifactRoot
    total_steps = $TotalSteps
    azimuth_bins = $AzimuthBins
    elevation_bins = $ElevationBins
    rollout_lengths = @($RolloutLengths)
    minibatch_size = $MinibatchSize
    ppo_epochs = $PpoEpochs
    temporal_cores = @($TemporalCores)
    repeats = $Repeats
    profile_cuda_events = [bool]$ProfileCudaEvents
    runs = @($runResults)
}

$comparisonJson = Join-Path $artifactRoot "comparison-summary.json"
$comparison | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $comparisonJson

Write-Host ""
Write-Host "Cadence comparison artifact root: $artifactRoot"
Write-Host "Comparison summary: $comparisonJson"
Write-Host "Measured runs:"
foreach ($run in $runResults) {
    $ppoMeanText = if ($null -eq $run.ppo_update_ms_mean) { "n/a" } else { "{0:N2}ms" -f $run.ppo_update_ms_mean }
    $bwdMeanText = if ($null -eq $run.backward_ms_mean) { "n/a" } else { "{0:N2}ms" -f $run.backward_ms_mean }
    Write-Host ("  core={0,-7} rollout={1,-5} sps_mean={2,8:N2} sps_median={3,8:N2} env_mean={4,7:N3}ms ppo_mean={5} bwd_mean={6}" -f $run.temporal_core, $run.rollout_length, $run.steady_sps_mean, $run.steady_sps_median, $run.env_ms_mean, $ppoMeanText, $bwdMeanText)
}

Write-Host ""
Write-Host "AGENTS.md Promotion Note:"
Write-Host "  This comparison provides the §3.1 Benchmark Gate evidence required"
Write-Host "  before flipping the canonical 'rollout_length' default in"
Write-Host "  projects/actor/src/navi_actor/config.py. A winning cadence ships in"
Write-Host "  one coherent change covering config, wrappers, tests, and docs"
Write-Host "  per §2.6 (Strict Contract Evolution)."
