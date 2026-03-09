param(
    [string]$GmDagFile = "",
    [int]$Actors = 4,
    [int]$TotalSteps = 2048,
    [int]$LogEvery = 256,
    [int]$AzimuthBins = 128,
    [int]$ElevationBins = 24,
    [int]$BasePort = 5700,
    [string]$OutputRoot = "artifacts/benchmarks/attribution-matrix"
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Get-RunName {
    param(
        [bool]$EnableEpisodicMemory,
        [bool]$EnableRewardShaping,
        [bool]$EmitObservationStream,
        [bool]$EmitTrainingTelemetry,
        [bool]$EmitPerfTelemetry
    )

    $disabled = @()
    if (-not $EnableEpisodicMemory) { $disabled += "no_mem" }
    if (-not $EnableRewardShaping) { $disabled += "no_shape" }
    if (-not $EmitObservationStream) { $disabled += "no_obs" }
    if (-not $EmitTrainingTelemetry) { $disabled += "no_train_tele" }
    if (-not $EmitPerfTelemetry) { $disabled += "no_perf_tele" }

    if ($disabled.Count -eq 0) {
        return "baseline"
    }

    return ($disabled -join "__")
}

function Get-StepMetrics {
    param([string[]]$Lines)

    $pattern = [regex]'\[step\s+(?<step>\d+)\].*?sps=(?<sps>[0-9.]+).*?fwd=(?<fwd>[0-9.]+)ms\s+pack=(?<pack>[0-9.]+)ms\s+env=(?<env>[0-9.]+)ms\s+mem=(?<mem>[0-9.]+)ms\s+trans=(?<trans>[0-9.]+)ms\s+\(shape=(?<shape>[0-9.]+)ms\s+madd=(?<madd>[0-9.]+)ms\s+buf=(?<buf>[0-9.]+)ms\s+host=(?<host>[0-9.]+)ms\s+tele=(?<tele>[0-9.]+)ms\)'
    $rows = @()
    foreach ($line in $Lines) {
        $match = $pattern.Match($line)
        if (-not $match.Success) {
            continue
        }
        $rows += [pscustomobject]@{
            step = [int]$match.Groups['step'].Value
            sps = [double]$match.Groups['sps'].Value
            fwd_ms = [double]$match.Groups['fwd'].Value
            pack_ms = [double]$match.Groups['pack'].Value
            env_ms = [double]$match.Groups['env'].Value
            mem_ms = [double]$match.Groups['mem'].Value
            trans_ms = [double]$match.Groups['trans'].Value
            shape_ms = [double]$match.Groups['shape'].Value
            madd_ms = [double]$match.Groups['madd'].Value
            buf_ms = [double]$match.Groups['buf'].Value
            host_ms = [double]$match.Groups['host'].Value
            tele_ms = [double]$match.Groups['tele'].Value
        }
    }
    return $rows
}

function Measure-Mean {
    param(
        [object[]]$Rows,
        [string]$Property
    )

    if (-not $Rows -or $Rows.Count -eq 0) {
        return $null
    }
    return [double](($Rows | Measure-Object -Property $Property -Average).Average)
}

function Get-OptMetrics {
    param([string[]]$Lines)

    $pattern = [regex]'Inline PPO optimization completed in\s+(?<ms>[0-9.]+)ms'
    $values = @()
    foreach ($line in $Lines) {
        $match = $pattern.Match($line)
        if ($match.Success) {
            $values += [double]$match.Groups['ms'].Value
        }
    }
    return $values
}

$repoRoot = Get-RepoRoot
$resolvedGmDag = $GmDagFile
if ([string]::IsNullOrWhiteSpace($resolvedGmDag)) {
    $compiledCorpusRoot = Join-Path $repoRoot "artifacts/gmdag/corpus"
    $compiledCandidates = if (Test-Path $compiledCorpusRoot) {
        Get-ChildItem -Path $compiledCorpusRoot -Recurse -File -Filter "*.gmdag" | Sort-Object FullName
    } else {
        @()
    }
    if ($compiledCandidates.Count -eq 0) {
        throw "No compiled corpus asset found. Run scripts/refresh-scene-corpus.ps1 or pass -GmDagFile explicitly."
    }
    $resolvedGmDag = $compiledCandidates[0].FullName
}
$resolvedGmDag = (Resolve-Path $resolvedGmDag).Path

$resolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path $repoRoot $OutputRoot
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $resolvedOutputRoot $timestamp
$runLogs = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $runLogs | Out-Null

$results = @()

for ($mask = 0; $mask -lt 32; $mask++) {
    $cfg = [ordered]@{
        EnableEpisodicMemory = (($mask -band 1) -eq 0)
        EnableRewardShaping = (($mask -band 2) -eq 0)
        EmitObservationStream = (($mask -band 4) -eq 0)
        EmitTrainingTelemetry = (($mask -band 8) -eq 0)
        EmitPerfTelemetry = (($mask -band 16) -eq 0)
    }

    $runName = Get-RunName @cfg
    $runDir = Join-Path $runRoot $runName
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $logPath = Join-Path $runLogs "$runName.log"
    $port = $BasePort + $mask
    $actorPub = "tcp://*:$port"

    $args = @(
        "run",
        "--project", (Join-Path $repoRoot "projects/actor"),
        "navi-actor", "train",
        "--gmdag-file", $resolvedGmDag,
        "--actors", "$Actors",
        "--total-steps", "$TotalSteps",
        "--azimuth-bins", "$AzimuthBins",
        "--elevation-bins", "$ElevationBins",
        "--rollout-length", "512",
        "--log-every", "$LogEvery",
        "--checkpoint-every", "0",
        "--checkpoint-dir", $runDir,
        "--actor-pub", $actorPub
    )

    if (-not $cfg.EnableEpisodicMemory) { $args += "--no-enable-episodic-memory" }
    if (-not $cfg.EnableRewardShaping) { $args += "--no-enable-reward-shaping" }
    if (-not $cfg.EmitObservationStream) { $args += "--no-emit-observation-stream" }
    if (-not $cfg.EmitTrainingTelemetry) { $args += "--no-emit-training-telemetry" }
    if (-not $cfg.EmitPerfTelemetry) { $args += "--no-emit-perf-telemetry" }

    Write-Host "[$($mask + 1)/32] $runName"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & uv @args *> $logPath
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    $lines = Get-Content $logPath
    $stepRows = Get-StepMetrics -Lines $lines
    $optRows = Get-OptMetrics -Lines $lines

    $result = [pscustomobject]@{
        mask = $mask
        run_name = $runName
        enable_episodic_memory = $cfg.EnableEpisodicMemory
        enable_reward_shaping = $cfg.EnableRewardShaping
        emit_observation_stream = $cfg.EmitObservationStream
        emit_training_telemetry = $cfg.EmitTrainingTelemetry
        emit_perf_telemetry = $cfg.EmitPerfTelemetry
        actor_pub = $actorPub
        exit_code = $exitCode
        wall_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        log_path = $logPath
        samples = $stepRows.Count
        steady_sps_mean = Measure-Mean -Rows $stepRows -Property 'sps'
        steady_sps_min = if ($stepRows.Count -gt 0) { [double](($stepRows | Measure-Object -Property sps -Minimum).Minimum) } else { $null }
        steady_sps_max = if ($stepRows.Count -gt 0) { [double](($stepRows | Measure-Object -Property sps -Maximum).Maximum) } else { $null }
        fwd_ms_mean = Measure-Mean -Rows $stepRows -Property 'fwd_ms'
        pack_ms_mean = Measure-Mean -Rows $stepRows -Property 'pack_ms'
        env_ms_mean = Measure-Mean -Rows $stepRows -Property 'env_ms'
        mem_ms_mean = Measure-Mean -Rows $stepRows -Property 'mem_ms'
        trans_ms_mean = Measure-Mean -Rows $stepRows -Property 'trans_ms'
        shape_ms_mean = Measure-Mean -Rows $stepRows -Property 'shape_ms'
        madd_ms_mean = Measure-Mean -Rows $stepRows -Property 'madd_ms'
        buf_ms_mean = Measure-Mean -Rows $stepRows -Property 'buf_ms'
        host_ms_mean = Measure-Mean -Rows $stepRows -Property 'host_ms'
        tele_ms_mean = Measure-Mean -Rows $stepRows -Property 'tele_ms'
        ppo_update_ms_mean = if ($optRows.Count -gt 0) { [double](($optRows | Measure-Object -Average).Average) } else { $null }
    }

    $results += $result

    $summaryLine = "  sps_mean={0:N1} env={1:N2}ms mem={2:N2}ms trans={3:N2}ms host={4:N2}ms tele={5:N2}ms ppo={6:N1}ms wall={7:N2}s" -f `
        $result.steady_sps_mean, $result.env_ms_mean, $result.mem_ms_mean, $result.trans_ms_mean, $result.host_ms_mean, $result.tele_ms_mean, $result.ppo_update_ms_mean, $result.wall_seconds
    Write-Host $summaryLine

    if ($exitCode -ne 0) {
        Write-Warning "Run failed for $runName (exit=$exitCode). See $logPath"
    }
}

$csvPath = Join-Path $runRoot "summary.csv"
$jsonPath = Join-Path $runRoot "summary.json"
$results | Sort-Object steady_sps_mean -Descending | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csvPath
$results | Sort-Object steady_sps_mean -Descending | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $jsonPath

Write-Host "Saved summary CSV: $csvPath"
Write-Host "Saved summary JSON: $jsonPath"
