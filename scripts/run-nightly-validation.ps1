<#
.SYNOPSIS
  Run the canonical overnight training validation pipeline.

.DESCRIPTION
  This wrapper stitches together the existing canonical validation surfaces into
  one timestamped overnight workflow:
    1. runtime preflight (`check-sdfdag`, `dataset-audit`)
    2. focused actor/environment/auditor regression suites
    3. bounded end-to-end qualification
    4. bounded shared-model checkpoint and resume proof
    5. repeated environment drift benchmarks
    6. overnight canonical soak with checkpoint and attach monitoring
    7. summary and baseline-diff artifact emission
#>
param(
    [int]$SoakHours = 8,
    [int]$MonitorIntervalSeconds = 300,
    [int]$CheckpointEvery = 25000,
    [int]$CheckpointStallMinutes = 60,
    [int]$QualificationSteps = 512,
    [int]$QualificationCheckpointEvery = 256,
    [int]$QualificationResumeAdditionalSteps = 256,
    [int]$BoundedSteps = 1024,
    [int]$BoundedResumeSteps = 512,
    [int]$BoundedLogEvery = 128,
    [int]$AttachTimeoutSeconds = 20,
    [int]$AttachProbeTimeoutSeconds = 5,
    [string]$PythonVersion = "3.12",
    [string]$RunRoot = "artifacts/nightly",
    [switch]$NoPreKill
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-PowerShellExecutable {
    $pwshCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($null -ne $pwshCommand) {
        return $pwshCommand.Source
    }

    $powershellCommand = Get-Command powershell -ErrorAction SilentlyContinue
    if ($null -ne $powershellCommand) {
        return $powershellCommand.Source
    }

    throw "Neither pwsh nor powershell is available on PATH"
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

function Stop-ListenersOnPorts {
    param([int[]]$Ports)

    $pids = @()
    foreach ($port in $Ports) {
        $lines = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+(\d+)\s*$"
        foreach ($line in $lines) {
            $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
            if ($parts.Length -ge 5) {
                $procId = 0
                if ([int]::TryParse($parts[-1], [ref]$procId)) {
                    $pids += $procId
                }
            }
        }
    }

    foreach ($procId in ($pids | Sort-Object -Unique)) {
        Stop-ProcessTreeById -ProcessId $procId
    }
}

function Stop-NaviProcesses {
    $currentPid = $PID
    $patterns = @(
        "*navi-environment*",
        "*navi-actor*",
        "*navi-auditor*",
        "*run-ghost-stack.ps1*"
    )
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $_.ProcessId -ne $currentPid -and $cmd -and ($patterns | Where-Object { $cmd -like $_ })
    }

    foreach ($proc in $targets) {
        Stop-ProcessTreeById -ProcessId $proc.ProcessId
    }

    Stop-ListenersOnPorts -Ports @(5557, 5558, 5559, 5560, 5757, 5758)
}

function Wait-ForPorts {
    param(
        [int[]]$Ports,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $allReady = $true
        foreach ($port in $Ports) {
            $hit = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:$port\s+\S+\s+LISTENING\s+\d+\s*$"
            if (-not $hit) {
                $allReady = $false
                break
            }
        }
        if ($allReady) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Read-TextFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return ""
    }

    $fileStream = $null
    $reader = $null
    try {
        $fileStream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $reader = New-Object System.IO.StreamReader($fileStream)
        return $reader.ReadToEnd()
    }
    catch [System.IO.IOException] {
        return ""
    }
    finally {
        if ($null -ne $reader) {
            $reader.Dispose()
        }
        elseif ($null -ne $fileStream) {
            $fileStream.Dispose()
        }
    }
}

function Convert-StructuredJson {
    param([string]$Text)

    $trimmed = $Text.Trim()
    $rootStart = $trimmed.IndexOf("{")
    if ($rootStart -lt 0) {
        throw "Expected JSON object in command output, got: $trimmed"
    }
    return $trimmed.Substring($rootStart) | ConvertFrom-Json
}

function Invoke-JsonCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$OutputPath,
        [string]$ErrorPath
    )

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $OutputPath -RedirectStandardError $ErrorPath -Wait -PassThru
    $output = Read-TextFile -Path $OutputPath
    $stderr = Read-TextFile -Path $ErrorPath

    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $FilePath $($ArgumentList -join ' ')`n$($output + [Environment]::NewLine + $stderr)"
    }

    return Convert-StructuredJson -Text $output
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$LogPath
    )

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    $proc = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -Wait -PassThru
    $merged = @()
    if (Test-Path $stderrPath) {
        $merged += Get-Content $stderrPath
    }
    if (Test-Path $stdoutPath) {
        $merged += Get-Content $stdoutPath
    }
    Set-Content -Encoding UTF8 -Path $LogPath -Value $merged
    return [pscustomobject]@{
        exit_code = $proc.ExitCode
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        log_path = $LogPath
    }
}

function Start-BackgroundProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$StdOutFile,
        [string]$StdErrFile
    )

    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $StdOutFile -RedirectStandardError $StdErrFile -PassThru
}

function Save-Summary {
    param([System.Collections.IDictionary]$Summary, [string]$SummaryPath)

    $Summary | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $SummaryPath
}

function Add-PhaseResult {
    param(
        [System.Collections.IDictionary]$Summary,
        [string]$SummaryPath,
        [string]$Name,
        [string]$Status,
        [object]$Details
    )

    $Summary.phases += [ordered]@{
        name = $Name
        status = $Status
        timestamp = (Get-Date).ToString("o")
        details = $Details
    }
    Save-Summary -Summary $Summary -SummaryPath $SummaryPath
}

function Add-SoftWarning {
    param(
        [System.Collections.IDictionary]$Summary,
        [string]$SummaryPath,
        [string]$Message
    )

    $Summary.soft_warnings += [ordered]@{
        timestamp = (Get-Date).ToString("o")
        message = $Message
    }
    Save-Summary -Summary $Summary -SummaryPath $SummaryPath
}

function Get-CanonicalBenchmarkScenes {
    param([string]$CorpusRoot)

    $preferred = @("apartment_1.gmdag", "skokloster-castle.gmdag", "van-gogh-room.gmdag")
    $allScenes = @(Get-ChildItem -Path $CorpusRoot -Recurse -File -Filter "*.gmdag" | Sort-Object FullName)
    $selected = @()
    foreach ($name in $preferred) {
        $match = $allScenes | Where-Object { $_.Name -ieq $name } | Select-Object -First 1
        if ($null -ne $match) {
            $selected += $match
        }
    }
    if ($selected.Count -eq 0) {
        $selected = $allScenes | Select-Object -First 3
    }
    return @($selected)
}

function Get-LatestCheckpointInfo {
    param([string]$CheckpointDir)

    $checkpoints = @(Get-ChildItem -Path $CheckpointDir -File -Filter "policy_step_*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime)
    if ($checkpoints.Count -eq 0) {
        return $null
    }

    $latest = $checkpoints[-1]
    $stepMatch = [regex]::Match($latest.BaseName, "policy_step_(\d+)")
    $step = if ($stepMatch.Success) { [int]$stepMatch.Groups[1].Value } else { $null }
    return [ordered]@{
        count = $checkpoints.Count
        latest_path = $latest.FullName
        latest_step = $step
        latest_write_time = $latest.LastWriteTime.ToString("o")
    }
}

function Get-LatestLoggedSps {
    param([string]$LogPath)

    $logText = Read-TextFile -Path $LogPath
    if ([string]::IsNullOrWhiteSpace($logText)) {
        return $null
    }
    $matches = [regex]::Matches($logText, "sps=(\d+\.\d+)")
    if ($null -eq $matches -or $matches.Count -eq 0) {
        return $null
    }
    $last = $matches[$matches.Count - 1].Groups[1].Value
    return [double]$last
}

function Test-NonFiniteLogText {
    param([string]$Text)

    return [regex]::IsMatch($Text, "(?i)\b-?nan\b|\b-?inf\b")
}

$repoRoot = Get-RepoRoot
$powerShellExe = Get-PowerShellExecutable
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRelativeRoot = Join-Path $RunRoot $timestamp
$runDir = Join-Path $repoRoot $runRelativeRoot
$preflightDir = Join-Path $runDir "preflight"
$preflightTestDir = Join-Path $preflightDir "tests"
$qualificationRoot = Join-Path $runRelativeRoot "qualification"
$qualificationDir = Join-Path $runDir "qualification"
$boundedDir = Join-Path $runDir "bounded"
$resumeDir = Join-Path $runDir "resume_checks"
$benchmarkDir = Join-Path $runDir "benchmarks\environment"
$trainingDir = Join-Path $runDir "training"
$trainingCheckpointDir = Join-Path $trainingDir "checkpoints"
$trainingAttachDir = Join-Path $trainingDir "attach_checks"
$logsDir = Join-Path $runDir "logs"
$reportsDir = Join-Path $runDir "reports"
$summaryPath = Join-Path $reportsDir "nightly_summary.json"
$summaryMdPath = Join-Path $reportsDir "nightly_summary.md"

foreach ($dir in @($runDir, $preflightDir, $preflightTestDir, $qualificationDir, $boundedDir, $resumeDir, $benchmarkDir, $trainingDir, $trainingCheckpointDir, $trainingAttachDir, $logsDir, $reportsDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$summary = [ordered]@{
    profile = "nightly-validation"
    ok = $false
    status = "running"
    started_at = (Get-Date).ToString("o")
    run_dir = $runDir
    config = [ordered]@{
        soak_hours = $SoakHours
        monitor_interval_seconds = $MonitorIntervalSeconds
        checkpoint_every = $CheckpointEvery
        checkpoint_stall_minutes = $CheckpointStallMinutes
        qualification_steps = $QualificationSteps
        qualification_checkpoint_every = $QualificationCheckpointEvery
        bounded_steps = $BoundedSteps
        bounded_resume_steps = $BoundedResumeSteps
        python_version = $PythonVersion
    }
    hard_failures = @()
    soft_warnings = @()
    phases = @()
    artifacts = [ordered]@{
        run_manifest = Join-Path $runDir "run_manifest.json"
        preflight = $preflightDir
        qualification = $qualificationDir
        bounded = $boundedDir
        resume_checks = $resumeDir
        benchmarks = $benchmarkDir
        training = $trainingDir
        reports = $reportsDir
    }
}

$runManifest = [ordered]@{
    started_at = $summary.started_at
    repo_root = $repoRoot
    python_version = $PythonVersion
    soak_hours = $SoakHours
    checkpoint_every = $CheckpointEvery
    monitor_interval_seconds = $MonitorIntervalSeconds
}
$runManifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summary.artifacts.run_manifest
Save-Summary -Summary $summary -SummaryPath $summaryPath

$trainingProc = $null

try {
    if (-not $NoPreKill) {
        Stop-NaviProcesses
        Start-Sleep -Milliseconds 500
    }
    if (-not (Wait-ForPorts -Ports @() -TimeoutSeconds 1)) {
    }
    $portHits = netstat -ano 2>$null | Select-String "5557|5558|5559|5560"
    if ($portHits) {
        throw "Required Navi ports remain occupied after cleanup"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "cleanup" -Status "passed" -Details ([ordered]@{ ports = @(5557, 5558, 5559, 5560) })

    $checkJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "check-sdfdag",
        "--gmdag-file", (Join-Path $repoRoot "artifacts\gmdag\corpus\apartment_1.gmdag"),
        "--json"
    ) -OutputPath (Join-Path $preflightDir "check-sdfdag.out.json") -ErrorPath (Join-Path $preflightDir "check-sdfdag.err.log")
    if (-not $checkJson.ok) {
        throw "check-sdfdag preflight failed"
    }

    $auditJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "dataset-audit",
        "--gmdag-file", (Join-Path $repoRoot "artifacts\gmdag\corpus\apartment_1.gmdag"),
        "--actors", "4",
        "--steps", "100",
        "--warmup-steps", "20",
        "--json"
    ) -OutputPath (Join-Path $preflightDir "dataset-audit.out.json") -ErrorPath (Join-Path $preflightDir "dataset-audit.err.log")
    if (-not $auditJson.ok) {
        throw "dataset-audit preflight failed"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "preflight" -Status "passed" -Details ([ordered]@{ check_sdfdag = $checkJson; dataset_audit = $auditJson })

    $actorTests = Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath (Join-Path $preflightTestDir "actor.log") -ArgumentList @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "pytest",
        ".\projects\actor\tests\unit\test_trajectory_buffer.py",
        ".\projects\actor\tests\unit\test_ppo_learner.py",
        ".\projects\actor\tests\unit\test_training_state.py"
    )
    if ($actorTests.exit_code -ne 0) {
        throw "Actor nightly regression suite failed"
    }

    $environmentTests = Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath (Join-Path $preflightTestDir "environment.log") -ArgumentList @(
        "run",
        "--project", (Join-Path $repoRoot "projects\environment"),
        "pytest",
        ".\projects\environment\tests\unit\test_sdfdag_conventions.py",
        ".\projects\environment\tests\unit\test_voxel_dag_integration.py",
        ".\projects\environment\tests\integration\test_live_corpus_validation.py"
    )
    if ($environmentTests.exit_code -ne 0) {
        throw "Environment nightly regression suite failed"
    }

    $auditorTests = Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath (Join-Path $preflightTestDir "auditor.log") -ArgumentList @(
        "run",
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "pytest",
        ".\projects\auditor\tests\unit",
        ".\projects\auditor\tests\integration"
    )
    if ($auditorTests.exit_code -ne 0) {
        throw "Auditor nightly regression suite failed"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "focused-tests" -Status "passed" -Details ([ordered]@{ actor_log = $actorTests.log_path; environment_log = $environmentTests.log_path; auditor_log = $auditorTests.log_path })

    $qualificationLog = Invoke-LoggedProcess -FilePath $powerShellExe -WorkingDirectory $repoRoot -LogPath (Join-Path $logsDir "qualification.log") -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\qualify-canonical-stack.ps1"),
        "-TotalSteps", "$QualificationSteps",
        "-CheckpointEvery", "$QualificationCheckpointEvery",
        "-AttachTimeoutSeconds", "$AttachTimeoutSeconds",
        "-ResumeAdditionalSteps", "$QualificationResumeAdditionalSteps",
        "-RunRoot", $qualificationRoot,
        "-NoPreKill"
    )
    if ($qualificationLog.exit_code -ne 0) {
        throw "Canonical qualification wrapper failed"
    }
    $qualificationRuns = @(Get-ChildItem -Path $qualificationDir -Directory | Sort-Object Name)
    if ($qualificationRuns.Count -eq 0) {
        throw "Qualification did not produce a timestamped run directory"
    }
    $qualificationSummaryPath = Join-Path $qualificationRuns[-1].FullName "qualification.json"
    if (-not (Test-Path $qualificationSummaryPath)) {
        throw "Qualification summary artifact missing: $qualificationSummaryPath"
    }
    $qualificationSummary = Get-Content -Path $qualificationSummaryPath -Raw | ConvertFrom-Json
    if (-not $qualificationSummary.ok) {
        throw "Canonical qualification reported ok=false"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "qualification" -Status "passed" -Details ([ordered]@{ qualification_json = $qualificationSummaryPath })

    $boundedLog = Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath (Join-Path $boundedDir "train.log") -ArgumentList @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor",
        "train",
        "--actors", "4",
        "--total-steps", "$BoundedSteps",
        "--log-every", "$BoundedLogEvery",
        "--checkpoint-every", "$QualificationCheckpointEvery",
        "--checkpoint-dir", $boundedDir,
        "--actor-pub", "tcp://*:5757"
    )
    if ($boundedLog.exit_code -ne 0) {
        throw "Bounded nightly trainer validation failed"
    }
    $boundedSummaryLog = Invoke-LoggedProcess -FilePath $powerShellExe -WorkingDirectory $repoRoot -LogPath (Join-Path $boundedDir "summary.log") -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\summarize-bounded-train-log.ps1"),
        "-LogPath", (Join-Path $boundedDir "train.log"),
        "-OutputJson", (Join-Path $boundedDir "summary.json")
    )
    if ($boundedSummaryLog.exit_code -ne 0) {
        throw "Bounded trainer summary generation failed"
    }
    $boundedSummary = Get-Content -Path (Join-Path $boundedDir "summary.json") -Raw | ConvertFrom-Json
    if ($boundedSummary.samples -le 0) {
        throw "Bounded trainer summary reported zero samples"
    }
    $boundedCheckpointInfo = Get-LatestCheckpointInfo -CheckpointDir $boundedDir
    if ($null -eq $boundedCheckpointInfo) {
        throw "Bounded trainer did not emit a periodic checkpoint"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "bounded-shared-model" -Status "passed" -Details ([ordered]@{ summary_json = (Join-Path $boundedDir "summary.json"); checkpoint = $boundedCheckpointInfo.latest_path })

    $resumeRunDir = Join-Path $resumeDir "bounded_resume"
    New-Item -ItemType Directory -Force -Path $resumeRunDir | Out-Null
    $resumeLog = Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath (Join-Path $resumeRunDir "train.log") -ArgumentList @(
        "run",
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor",
        "train",
        "--actors", "4",
        "--total-steps", "$BoundedResumeSteps",
        "--log-every", "$BoundedLogEvery",
        "--checkpoint", $boundedCheckpointInfo.latest_path,
        "--checkpoint-every", "$QualificationCheckpointEvery",
        "--checkpoint-dir", $resumeRunDir,
        "--actor-pub", "tcp://*:5758"
    )
    if ($resumeLog.exit_code -ne 0) {
        throw "Bounded resume proof failed"
    }
    $resumeCheckpointInfo = Get-LatestCheckpointInfo -CheckpointDir $resumeRunDir
    if ($null -eq $resumeCheckpointInfo) {
        throw "Bounded resume proof did not emit a checkpoint"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "bounded-resume" -Status "passed" -Details ([ordered]@{ checkpoint = $resumeCheckpointInfo.latest_path })

    $corpusRoot = Join-Path $repoRoot "artifacts\gmdag\corpus"
    foreach ($scene in (Get-CanonicalBenchmarkScenes -CorpusRoot $corpusRoot)) {
        $sceneOut = Join-Path $benchmarkDir ($scene.BaseName + ".json")
        $sceneErr = Join-Path $benchmarkDir ($scene.BaseName + ".err.log")
        $benchJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
            "run",
            "--python", $PythonVersion,
            "--project", (Join-Path $repoRoot "projects\environment"),
            "navi-environment",
            "bench-sdfdag",
            "--gmdag-file", $scene.FullName,
            "--actors", "4",
            "--steps", "200",
            "--warmup-steps", "20",
            "--repeats", "5",
            "--json"
        ) -OutputPath $sceneOut -ErrorPath $sceneErr
        Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name ("benchmark-" + $scene.BaseName) -Status "passed" -Details $benchJson
    }

    $trainingOut = Join-Path $logsDir "soak.out.log"
    $trainingErr = Join-Path $logsDir "soak.err.log"
    $trainingProc = Start-BackgroundProcess -FilePath "uv" -WorkingDirectory $repoRoot -StdOutFile $trainingOut -StdErrFile $trainingErr -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\actor"),
        "navi-actor",
        "train",
        "--actors", "4",
        "--azimuth-bins", "256",
        "--elevation-bins", "48",
        "--total-steps", "0",
        "--checkpoint-every", "$CheckpointEvery",
        "--checkpoint-dir", $trainingCheckpointDir,
        "--minibatch-size", "64",
        "--ppo-epochs", "1",
        "--existential-tax", "-0.02",
        "--entropy-coeff", "0.02",
        "--learning-rate", "5e-4",
        "--bptt-len", "8",
        "--rollout-length", "512",
        "--compile-resolution", "512",
        "--shuffle"
    )
    if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds 120)) {
        throw "Overnight soak did not bind actor telemetry on 5557"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "soak-start" -Status "passed" -Details ([ordered]@{ pid = $trainingProc.Id; stdout = $trainingOut; stderr = $trainingErr })

    $deadline = (Get-Date).AddHours($SoakHours)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $MonitorIntervalSeconds
        $trainingProc.Refresh()
        if ($trainingProc.HasExited) {
            throw "Overnight soak exited early with code $($trainingProc.ExitCode)"
        }

        $checkpointInfo = Get-LatestCheckpointInfo -CheckpointDir $trainingCheckpointDir
        if ($null -eq $checkpointInfo) {
            Add-SoftWarning -Summary $summary -SummaryPath $summaryPath -Message "No overnight checkpoint has been produced yet"
        }
        else {
            $latestWrite = [datetime]::Parse($checkpointInfo.latest_write_time)
            if ($latestWrite -lt (Get-Date).AddMinutes(-1 * $CheckpointStallMinutes)) {
                throw "Overnight checkpoints stalled past ${CheckpointStallMinutes} minutes"
            }
        }

        $combinedTrainingText = (Read-TextFile $trainingErr) + [Environment]::NewLine + (Read-TextFile $trainingOut)
        if (Test-NonFiniteLogText -Text $combinedTrainingText) {
            throw "Detected non-finite marker in overnight soak logs"
        }

        $attachStamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $attachOut = Join-Path $trainingAttachDir ("attach_" + $attachStamp + ".json")
        $attachErr = Join-Path $trainingAttachDir ("attach_" + $attachStamp + ".err.log")
        try {
            $attachJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
                "run",
                "--python", $PythonVersion,
                "--project", (Join-Path $repoRoot "projects\auditor"),
                "navi-auditor",
                "dashboard-attach-check",
                "--actor-sub", "tcp://localhost:5557",
                "--timeout-seconds", "$AttachProbeTimeoutSeconds",
                "--json"
            ) -OutputPath $attachOut -ErrorPath $attachErr
            if (-not $attachJson.ok) {
                Add-SoftWarning -Summary $summary -SummaryPath $summaryPath -Message ("Passive attach warning: " + (($attachJson.issues) -join "; "))
            }
        }
        catch {
            Add-SoftWarning -Summary $summary -SummaryPath $summaryPath -Message ("Passive attach probe failed: " + $_.Exception.Message)
        }

        $latestSps = Get-LatestLoggedSps -LogPath $trainingErr
        if ($null -ne $latestSps -and $latestSps -lt 60.0) {
            Add-SoftWarning -Summary $summary -SummaryPath $summaryPath -Message ("Latest logged SPS dropped below 60: " + $latestSps)
        }
    }

    if ($null -ne $trainingProc -and -not $trainingProc.HasExited) {
        Stop-ProcessTreeById -ProcessId $trainingProc.Id
        Start-Sleep -Seconds 2
        try {
            $trainingProc.Refresh()
        }
        catch {
        }
    }

    $summary.status = "passed"
    $summary.ok = $true
    $summary.completed_at = (Get-Date).ToString("o")
    Save-Summary -Summary $summary -SummaryPath $summaryPath

    $latestCheckpointInfo = Get-LatestCheckpointInfo -CheckpointDir $trainingCheckpointDir
    $markdown = @(
        "# Nightly Validation Summary",
        "",
        "- status: passed",
        "- run_dir: $runDir",
        "- latest_checkpoint: $(if ($null -ne $latestCheckpointInfo) { $latestCheckpointInfo.latest_path } else { 'none' })",
        "- soft_warnings: $($summary.soft_warnings.Count)",
        "- phases: $($summary.phases.Count)"
    )
    Set-Content -Encoding UTF8 -Path $summaryMdPath -Value $markdown
}
catch {
    if ($null -ne $trainingProc -and -not $trainingProc.HasExited) {
        Stop-ProcessTreeById -ProcessId $trainingProc.Id
    }
    $summary.status = "failed"
    $summary.ok = $false
    $summary.completed_at = (Get-Date).ToString("o")
    $summary.hard_failures += $_.Exception.Message
    Save-Summary -Summary $summary -SummaryPath $summaryPath
    $markdown = @(
        "# Nightly Validation Summary",
        "",
        "- status: failed",
        "- run_dir: $runDir",
        "- hard_failure: $($_.Exception.Message)",
        "- soft_warnings: $($summary.soft_warnings.Count)",
        "- phases: $($summary.phases.Count)"
    )
    Set-Content -Encoding UTF8 -Path $summaryMdPath -Value $markdown
    throw
}