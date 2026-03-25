<# .SYNOPSIS
  Actor count scaling test — find the optimal actor count for max SPS
  on the current hardware (MX150 2GB).

  Runs canonical training with increasing actor counts, measures SPS,
  and reports a summary table.

.EXAMPLE
  .\scripts\run-actor-scaling-test.ps1
  .\scripts\run-actor-scaling-test.ps1 -StepsPerRun 3000
  .\scripts\run-actor-scaling-test.ps1 -ActorCounts 4,8,12,16,20
#>
param(
    [string]$ActorCountsStr = "4,8,12,16,20,24,28,32",
    [int]$StepsPerRun = 2000,
    [int]$RolloutLength = 256,
    [string]$TemporalCore = "mamba2",
    [int]$LogEvery = 100
)

$ActorCounts = $ActorCountsStr -split ',' | ForEach-Object { [int]$_.Trim() }

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resultsDir = Join-Path $repoRoot "artifacts\benchmarks\actor-scaling"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryFile = Join-Path $resultsDir "scaling_$timestamp.txt"

# Kill stale processes
function Stop-StaleNavi {
    $procs = Get-Process -Name "python*" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -match "navi" -or $_.CommandLine -match "navi" }
    # Also kill by port
    foreach ($port in @(5557, 5559, 5560)) {
        $lines = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+(\d+)"
        foreach ($line in $lines) {
            $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
            if ($parts.Length -ge 5) {
                $pid = 0
                if ([int]::TryParse($parts[-1], [ref]$pid) -and $pid -gt 0) {
                    try { taskkill /PID $pid /T /F *> $null } catch {}
                }
            }
        }
    }
}

$results = @()

Write-Host "=========================================================="
Write-Host "  Navi Actor Scaling Test"
Write-Host "  Actor counts : $($ActorCounts -join ', ')"
Write-Host "  Steps/run    : $StepsPerRun"
Write-Host "  Rollout len  : $RolloutLength"
Write-Host "  Temporal     : $TemporalCore"
Write-Host "  Results      : $summaryFile"
Write-Host "=========================================================="

foreach ($nActors in $ActorCounts) {
    Write-Host ""
    Write-Host "────────────────────────────────────────────────────────"
    Write-Host "  Testing $nActors actors..."
    Write-Host "────────────────────────────────────────────────────────"

    Stop-StaleNavi
    Start-Sleep -Seconds 2

    $runLogDir = Join-Path $resultsDir "run_${nActors}_actors_$timestamp"
    New-Item -ItemType Directory -Path $runLogDir -Force | Out-Null
    $logOut = Join-Path $runLogDir "stdout.log"
    $logErr = Join-Path $runLogDir "stderr.log"

    $trainArgs = @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "python", "-m", "navi_actor.cli", "train",
        "--actors", "$nActors",
        "--temporal-core", $TemporalCore,
        "--total-steps", $StepsPerRun,
        "--no-emit-observation-stream",
        "--shuffle",
        "--rollout-length", $RolloutLength,
        "--log-every", $LogEvery,
        "--checkpoint-every", "0",
        "--compile-resolution", "512",
        "--azimuth-bins", "256",
        "--elevation-bins", "48"
    )

    $startTime = Get-Date
    $proc = $null
    $exitCode = -1
    $crashed = $false
    $oom = $false

    try {
        $proc = Start-Process -FilePath "uv" -ArgumentList $trainArgs `
            -RedirectStandardOutput $logOut -RedirectStandardError $logErr `
            -PassThru -NoNewWindow -WorkingDirectory $repoRoot

        # Wait with timeout (5 minutes per run max)
        $maxWaitSec = 300
        $waited = 0
        while (-not $proc.HasExited -and $waited -lt $maxWaitSec) {
            Start-Sleep -Seconds 2
            $waited += 2
        }

        if (-not $proc.HasExited) {
            Write-Host "  TIMEOUT: killing after ${maxWaitSec}s"
            try { taskkill /PID $proc.Id /T /F *> $null } catch {}
            Start-Sleep -Seconds 2
            $crashed = $true
        }
        else {
            $proc.WaitForExit()
        }

        $exitCode = if ($null -eq $proc.ExitCode) { -1 } else { [int]$proc.ExitCode }
    }
    catch {
        Write-Host "  ERROR launching: $_"
        $crashed = $true
    }

    $elapsed = (Get-Date) - $startTime

    # Parse results from logs (check both stdout and stderr — periodic SPS goes to logging/stderr)
    $spsMean = 0.0
    $spsValues = @()
    $gpuMem = ""
    $totalSteps = 0
    $episodes = 0

    # Combine stdout + stderr for parsing (SPS lines go through logging → stderr)
    $allContent = ""
    foreach ($logFile in @($logOut, $logErr)) {
        if (Test-Path $logFile) {
            $c = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
            if ($c) { $allContent += "`n$c" }
        }
    }

    if ($allContent) {
        # Extract sps from periodic log lines: sps=123.4
        $spsMatches = [regex]::Matches($allContent, 'sps=(\d+\.?\d*)')
        foreach ($m in $spsMatches) {
            $spsValues += [double]$m.Groups[1].Value
        }

        # Extract final sps_mean from stdout
        $meanMatch = [regex]::Match($allContent, 'sps_mean=(\d+\.?\d*)')
        if ($meanMatch.Success) {
            $spsMean = [double]$meanMatch.Groups[1].Value
        }
        elseif ($spsValues.Count -gt 0) {
            # Skip first few warmup values, average the rest
            $skip = [Math]::Min(3, $spsValues.Count - 1)
            $steadyState = $spsValues[$skip..($spsValues.Count - 1)]
            if ($steadyState.Count -gt 0) {
                $spsMean = ($steadyState | Measure-Object -Average).Average
            }
        }

        # Extract steps and episodes from final line
        $stepsMatch = [regex]::Match($allContent, 'steps=(\d+)')
        if ($stepsMatch.Success) { $totalSteps = [int]$stepsMatch.Groups[1].Value }
        $epMatch = [regex]::Match($allContent, 'episodes=(\d+)')
        if ($epMatch.Success) { $episodes = [int]$epMatch.Groups[1].Value }
    }

    # Check stderr for OOM
    if (Test-Path $logErr) {
        $errContent = Get-Content $logErr -Raw -ErrorAction SilentlyContinue
        if ($errContent -and ($errContent -match "CUDA out of memory|OutOfMemoryError|out of memory")) {
            $oom = $true
            $crashed = $true
        }
    }

    # GPU memory from the unified log
    $unifiedLog = Join-Path $repoRoot "logs\navi_actor_train.log"
    if (Test-Path $unifiedLog) {
        $tail = Get-Content $unifiedLog -Tail 200 -ErrorAction SilentlyContinue
        if ($tail) {
            $tailText = $tail -join "`n"
            $gpuMatch = [regex]::Match($tailText, 'gpu_alloc_mb=(\d+\.?\d*)')
            if ($gpuMatch.Success) { $gpuMem = "$($gpuMatch.Groups[1].Value) MB" }
        }
    }

    $status = if ($oom) { "OOM" } elseif ($crashed) { "CRASH" } elseif ($exitCode -ne 0) { "FAIL($exitCode)" } else { "OK" }

    # Compute peak/min SPS
    $spsPeak = if ($spsValues.Count -gt 0) { ($spsValues | Measure-Object -Maximum).Maximum } else { 0 }
    $spsMin = if ($spsValues.Count -gt 0) { ($spsValues | Measure-Object -Minimum).Minimum } else { 0 }

    $result = [ordered]@{
        Actors   = $nActors
        Status   = $status
        SpsMean  = [math]::Round($spsMean, 1)
        SpsPeak  = [math]::Round($spsPeak, 1)
        SpsMin   = [math]::Round($spsMin, 1)
        Steps    = $totalSteps
        Episodes = $episodes
        GpuMem   = $gpuMem
        Elapsed  = "{0:mm\:ss}" -f $elapsed
        Exit     = $exitCode
    }
    $results += $result

    Write-Host "  Status   : $status"
    Write-Host "  SPS Mean : $($result.SpsMean)"
    Write-Host "  SPS Peak : $($result.SpsPeak)"
    Write-Host "  SPS Min  : $($result.SpsMin)"
    Write-Host "  Steps    : $totalSteps"
    Write-Host "  Episodes : $episodes"
    Write-Host "  GPU Mem  : $gpuMem"
    Write-Host "  Elapsed  : $($result.Elapsed)"

    if ($crashed -or $oom) {
        Write-Host "  >>> STOPPING: $status at $nActors actors"
        break
    }
}

# Print summary table
Write-Host ""
Write-Host "=========================================================="
Write-Host "  SCALING TEST RESULTS"
Write-Host "=========================================================="
Write-Host ("{0,-8} {1,-8} {2,-10} {3,-10} {4,-10} {5,-8} {6,-8} {7,-12} {8,-8}" -f "Actors", "Status", "SPS Mean", "SPS Peak", "SPS Min", "Steps", "Eps", "GPU Mem", "Time")
Write-Host ("=" * 82)

foreach ($r in $results) {
    Write-Host ("{0,-8} {1,-8} {2,-10} {3,-10} {4,-10} {5,-8} {6,-8} {7,-12} {8,-8}" -f $r.Actors, $r.Status, $r.SpsMean, $r.SpsPeak, $r.SpsMin, $r.Steps, $r.Episodes, $r.GpuMem, $r.Elapsed)
}

# Find optimal
$okResults = $results | Where-Object { $_.Status -eq "OK" -and $_.SpsMean -gt 0 }
if ($okResults) {
    $best = $okResults | Sort-Object -Property SpsMean -Descending | Select-Object -First 1
    Write-Host ""
    Write-Host "  OPTIMAL: $($best.Actors) actors @ $($best.SpsMean) SPS mean"
}

# Save to file
$results | ForEach-Object { [PSCustomObject]$_ } | Format-Table -AutoSize | Out-String | Set-Content $summaryFile
$results | ConvertTo-Json | Add-Content $summaryFile

Write-Host ""
Write-Host "Results saved to: $summaryFile"
