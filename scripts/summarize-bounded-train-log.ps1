param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [string]$RepoLogPath = "logs/navi_actor_train.log",
    [string]$OutputJson = ""
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-WorkspacePath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path $Path).Path
    }
    return (Resolve-Path (Join-Path (Resolve-RepoRoot) $Path)).Path
}

function Parse-LogTimestamp {
    param([string]$Value)

    return [datetime]::ParseExact($Value, "yyyy-MM-dd HH:mm:ss", [System.Globalization.CultureInfo]::InvariantCulture)
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

function Get-CheckpointPaths {
    param([string[]]$Lines)

    $pattern = [regex]'Saving training checkpoint to\s+(?<path>.+)$'
    $paths = @()
    foreach ($line in $Lines) {
        $match = $pattern.Match($line)
        if ($match.Success) {
            $paths += $match.Groups['path'].Value.Trim()
        }
    }
    return $paths
}

function Get-RunDescriptor {
    param([string[]]$Lines)

    $pattern = [regex]'^\[(?<ts>[^\]]+)\].*Canonical PPO trainer started:\s+actors=(?<actors>\d+)\s+gmdag=(?<gmdag>.+?)\s+pub=(?<pub>\S+)'
    foreach ($line in $Lines) {
        $match = $pattern.Match($line)
        if ($match.Success) {
            return [pscustomobject]@{
                start_timestamp = $match.Groups['ts'].Value
                start_time = Parse-LogTimestamp $match.Groups['ts'].Value
                actors = [int]$match.Groups['actors'].Value
                gmdag = $match.Groups['gmdag'].Value.Trim()
                pub = $match.Groups['pub'].Value.Trim()
            }
        }
    }
    return $null
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

function Get-RepoRunWindow {
    param(
        [string[]]$RepoLines,
        [pscustomobject]$RunDescriptor
    )

    if ($null -eq $RunDescriptor) {
        return $null
    }

    $startPattern = [regex]'^\[(?<ts>[^\]]+)\].*Canonical PPO trainer started:\s+actors=(?<actors>\d+)\s+gmdag=(?<gmdag>.+?)\s+pub=(?<pub>\S+)'
    $matchedStartIndex = -1

    for ($index = 0; $index -lt $RepoLines.Count; $index++) {
        $match = $startPattern.Match($RepoLines[$index])
        if (-not $match.Success) {
            continue
        }

        $candidateTime = Parse-LogTimestamp $match.Groups['ts'].Value
        if ($candidateTime -lt $RunDescriptor.start_time) {
            continue
        }
        if ([int]$match.Groups['actors'].Value -ne $RunDescriptor.actors) {
            continue
        }
        if ($match.Groups['gmdag'].Value.Trim() -ne $RunDescriptor.gmdag) {
            continue
        }
        if ($match.Groups['pub'].Value.Trim() -ne $RunDescriptor.pub) {
            continue
        }

        $matchedStartIndex = $index
        break
    }

    if ($matchedStartIndex -lt 0) {
        return $null
    }

    $endIndex = $RepoLines.Count - 1
    for ($index = $matchedStartIndex + 1; $index -lt $RepoLines.Count; $index++) {
        if ($startPattern.IsMatch($RepoLines[$index])) {
            $endIndex = $index - 1
            break
        }
    }

    return $RepoLines[$matchedStartIndex..$endIndex]
}

$resolvedLogPath = Resolve-WorkspacePath $LogPath
$resolvedRepoLogPath = Resolve-WorkspacePath $RepoLogPath
$resolvedOutputJson = if ([string]::IsNullOrWhiteSpace($OutputJson)) {
    "$resolvedLogPath.summary.json"
} elseif ([System.IO.Path]::IsPathRooted($OutputJson)) {
    $OutputJson
} else {
    Join-Path (Resolve-RepoRoot) $OutputJson
}

$lines = Get-Content $resolvedLogPath
$stepRows = @(Get-StepMetrics -Lines $lines)
$artifactOptRows = @(Get-OptMetrics -Lines $lines)
$artifactCheckpoints = @(Get-CheckpointPaths -Lines $lines)
$runDescriptor = Get-RunDescriptor -Lines $lines

$ppoSource = if ($artifactOptRows.Count -gt 0) { "artifact-log" } else { "unavailable" }
$ppoRows = $artifactOptRows
$checkpointPaths = $artifactCheckpoints

if ($artifactOptRows.Count -eq 0 -and (Test-Path $resolvedRepoLogPath)) {
    $repoLines = Get-Content $resolvedRepoLogPath
    $repoWindow = Get-RepoRunWindow -RepoLines $repoLines -RunDescriptor $runDescriptor
    if ($null -ne $repoWindow) {
        $repoOptRows = @(Get-OptMetrics -Lines $repoWindow)
        $repoCheckpoints = @(Get-CheckpointPaths -Lines $repoWindow)
        if ($repoOptRows.Count -gt 0) {
            $ppoRows = $repoOptRows
            $ppoSource = "repo-log-fallback"
        }
        if ($repoCheckpoints.Count -gt 0) {
            $checkpointPaths = $repoCheckpoints
        }
    }
}

$finalCheckpointPath = if ($checkpointPaths) {
    $checkpointItems = @($checkpointPaths)
    [string]($checkpointItems[$checkpointItems.Count - 1])
} else {
    $null
}

$summary = [ordered]@{
    log_path = $resolvedLogPath
    repo_log_path = $resolvedRepoLogPath
    run_start_timestamp = if ($null -ne $runDescriptor) { $runDescriptor.start_timestamp } else { $null }
    actors = if ($null -ne $runDescriptor) { $runDescriptor.actors } else { $null }
    gmdag = if ($null -ne $runDescriptor) { $runDescriptor.gmdag } else { $null }
    actor_pub = if ($null -ne $runDescriptor) { $runDescriptor.pub } else { $null }
    samples = $stepRows.Count
    steady_sps_mean = Measure-Mean -Rows $stepRows -Property 'sps'
    env_ms_mean = Measure-Mean -Rows $stepRows -Property 'env_ms'
    mem_ms_mean = Measure-Mean -Rows $stepRows -Property 'mem_ms'
    trans_ms_mean = Measure-Mean -Rows $stepRows -Property 'trans_ms'
    host_extract_ms_mean = Measure-Mean -Rows $stepRows -Property 'host_ms'
    telemetry_publish_ms_mean = Measure-Mean -Rows $stepRows -Property 'tele_ms'
    ppo_update_ms_mean = if ($ppoRows.Count -gt 0) { [double](($ppoRows | Measure-Object -Average).Average) } else { $null }
    ppo_update_source = $ppoSource
    final_checkpoint_path = $finalCheckpointPath
}

$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $resolvedOutputJson

$status = "Summary saved: {0}`n  samples={1} sps={2:N2} env={3:N4}ms mem={4:N4}ms trans={5:N4}ms host={6:N4}ms tele={7:N4}ms ppo={8} source={9}" -f `
    $resolvedOutputJson, `
    $summary.samples, `
    $summary.steady_sps_mean, `
    $summary.env_ms_mean, `
    $summary.mem_ms_mean, `
    $summary.trans_ms_mean, `
    $summary.host_extract_ms_mean, `
    $summary.telemetry_publish_ms_mean, `
    $(if ($null -eq $summary.ppo_update_ms_mean) { "n/a" } else { "{0:N2}ms" -f $summary.ppo_update_ms_mean }), `
    $summary.ppo_update_source
Write-Host $status
