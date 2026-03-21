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

function Get-ProjectPythonExecutable {
    param([string]$ProjectPath)

    $candidate = Join-Path $ProjectPath ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    throw "Project-local Python interpreter not found: $candidate"
}

function Resolve-JsonCommandInvocation {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    if ($FilePath -ieq "uv" -and $ArgumentList.Count -ge 4 -and $ArgumentList[0] -eq "run") {
        $index = 1
        if ($index + 1 -lt $ArgumentList.Count -and $ArgumentList[$index] -eq "--python") {
            $index += 2
        }
        if ($index + 1 -lt $ArgumentList.Count -and $ArgumentList[$index] -eq "--project") {
            $projectPath = $ArgumentList[$index + 1]
            $index += 2
            if ($index -lt $ArgumentList.Count) {
                $cliName = $ArgumentList[$index]
                $remaining = @()
                if ($index + 1 -lt $ArgumentList.Count) {
                    $remaining = $ArgumentList[($index + 1)..($ArgumentList.Count - 1)]
                }
                if ($cliName -eq "navi-environment") {
                    return [pscustomobject]@{
                        file_path = Get-ProjectPythonExecutable -ProjectPath $projectPath
                        argument_list = @("-m", "navi_environment.cli") + $remaining
                    }
                }
                if ($cliName -eq "navi-auditor") {
                    return [pscustomobject]@{
                        file_path = Get-ProjectPythonExecutable -ProjectPath $projectPath
                        argument_list = @("-m", "navi_auditor.cli") + $remaining
                    }
                }
            }
        }
    }

    return [pscustomobject]@{
        file_path = $FilePath
        argument_list = $ArgumentList
    }
}

function Get-StructuredSurfaceRunnerPython {
    param([string]$ModuleName)

    if ($ModuleName -eq "navi_auditor.cli") {
        return $script:StructuredSurfaceRunnerPythonByModule[$ModuleName]
    }

    return $script:StructuredSurfaceRunnerPythonByModule["default"]
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
        "*navi_environment.cli*",
        "*navi-actor*",
        "*navi-auditor*",
        "*navi_auditor.cli*",
        "*run-structured-surface.py*",
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

function Quote-ProcessArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }
    if ($Value.Length -eq 0) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Join-ProcessArguments {
    param([string[]]$ArgumentList)

    if ($null -eq $ArgumentList -or $ArgumentList.Count -eq 0) {
        return ""
    }
    return (($ArgumentList | ForEach-Object { Quote-ProcessArgument -Value ([string]$_) }) -join ' ')
}

function ConvertTo-PowerShellSingleQuotedLiteral {
    param([string]$Value)

    if ($null -eq $Value) {
        return "''"
    }
    return "'" + ($Value -replace "'", "''") + "'"
}

function Invoke-NativeProcessCapture {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [int]$TimeoutSeconds = 300
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = Join-ProcessArguments -ArgumentList $ArgumentList
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true

    $stdoutBuilder = New-Object System.Text.StringBuilder
    $stderrBuilder = New-Object System.Text.StringBuilder
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo

    $stdoutHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $eventArgs)
        if ($null -ne $eventArgs.Data) {
            [void]$stdoutBuilder.AppendLine($eventArgs.Data)
        }
    }
    $stderrHandler = [System.Diagnostics.DataReceivedEventHandler]{
        param($sender, $eventArgs)
        if ($null -ne $eventArgs.Data) {
            [void]$stderrBuilder.AppendLine($eventArgs.Data)
        }
    }

    $process.add_OutputDataReceived($stdoutHandler)
    $process.add_ErrorDataReceived($stderrHandler)
    try {
        if (-not $process.Start()) {
            throw "Failed to start command: $FilePath $($ArgumentList -join ' ')"
        }
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()

        if ($TimeoutSeconds -gt 0) {
            if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
                Stop-ProcessTreeById -ProcessId $process.Id
                throw "Command timed out after ${TimeoutSeconds}s: $FilePath $($ArgumentList -join ' ')"
            }
        }
        else {
            $process.WaitForExit()
        }
        $process.WaitForExit()

        return [pscustomobject]@{
            exit_code = [int]$process.ExitCode
            stdout = $stdoutBuilder.ToString().TrimEnd("`r", "`n")
            stderr = $stderrBuilder.ToString().TrimEnd("`r", "`n")
        }
    }
    finally {
        try {
            $process.remove_OutputDataReceived($stdoutHandler)
            $process.remove_ErrorDataReceived($stderrHandler)
        }
        catch {
        }
        $process.Dispose()
    }
}

function Invoke-JsonCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$OutputPath,
        [string]$ErrorPath,
        [int]$TimeoutSeconds = 300,
        [string]$ProgressLogPath = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($ProgressLogPath)) {
        Add-Content -Encoding UTF8 -Path $ProgressLogPath -Value ("[{0}] json:start {1}" -f (Get-Date).ToString("o"), ($ArgumentList -join ' '))
    }

    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    if (Test-Path -LiteralPath $ErrorPath) {
        Remove-Item -LiteralPath $ErrorPath -Force
    }

    $resolvedInvocation = Resolve-JsonCommandInvocation -FilePath $FilePath -ArgumentList $ArgumentList
    $resolvedFilePath = [string]$resolvedInvocation.file_path
    $resolvedArgumentList = [string[]]$resolvedInvocation.argument_list

    $structuredRunnerUsed = $false
    $captureFilePath = $resolvedFilePath
    $captureArgumentList = $resolvedArgumentList
    if ($resolvedArgumentList.Count -ge 3 -and $resolvedArgumentList[0] -eq "-m") {
        $structuredRunnerUsed = $true
        $moduleName = [string]$resolvedArgumentList[1]
        $captureFilePath = Get-StructuredSurfaceRunnerPython -ModuleName $moduleName
        $captureArgumentList = @(
            $script:StructuredSurfaceRunnerScript,
            "--module", $moduleName,
            "--output-path", $OutputPath,
            "--error-path", $ErrorPath,
            "--"
        ) + $resolvedArgumentList[2..($resolvedArgumentList.Count - 1)]
    }

    if ($structuredRunnerUsed) {
        $tempScriptPath = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".ps1")
        try {
            $quotedRunner = ConvertTo-PowerShellSingleQuotedLiteral -Value $captureFilePath
            $quotedArgs = if ($captureArgumentList.Count -gt 0) {
                ($captureArgumentList | ForEach-Object { ConvertTo-PowerShellSingleQuotedLiteral -Value ([string]$_) }) -join ", "
            }
            else {
                ""
            }
            $scriptText = @(
                '$ErrorActionPreference = "Continue"',
                'if (Test-Path variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }',
                ('$runner = ' + $quotedRunner),
                ('$runnerArgs = @(' + $quotedArgs + ')'),
                '& $runner @runnerArgs',
                'exit $LASTEXITCODE'
            ) -join [Environment]::NewLine
            Set-Content -Encoding UTF8 -Path $tempScriptPath -Value $scriptText

            $proc = Start-Process -FilePath $powerShellExe -ArgumentList @(
                '-NoProfile',
                '-ExecutionPolicy', 'Bypass',
                '-File', $tempScriptPath
            ) -WorkingDirectory $WorkingDirectory -PassThru -Wait
            $exitCode = [int]$proc.ExitCode
        }
        finally {
            Remove-Item -LiteralPath $tempScriptPath -Force -ErrorAction SilentlyContinue
        }

        $stdout = if (Test-Path -LiteralPath $OutputPath) { Get-Content -LiteralPath $OutputPath -Raw } else { "" }
        $stderr = if (Test-Path -LiteralPath $ErrorPath) { Get-Content -LiteralPath $ErrorPath -Raw } else { "" }
        $result = [pscustomobject]@{ exit_code = $exitCode; stdout = $stdout; stderr = $stderr }
    }
    else {
        $result = Invoke-NativeProcessCapture -FilePath $captureFilePath -ArgumentList $captureArgumentList -WorkingDirectory $WorkingDirectory -TimeoutSeconds $TimeoutSeconds
        $stdout = $result.stdout
        $stderr = $result.stderr
        Set-Content -Encoding UTF8 -Path $OutputPath -Value $stdout
        Set-Content -Encoding UTF8 -Path $ErrorPath -Value $stderr
    }

    if (-not [string]::IsNullOrWhiteSpace($ProgressLogPath)) {
        Add-Content -Encoding UTF8 -Path $ProgressLogPath -Value ("[{0}] json:completed exit_code={1}" -f (Get-Date).ToString("o"), $result.exit_code)
    }

    if ([int]$result.exit_code -ne 0) {
        throw "Command failed with exit code $($result.exit_code): $resolvedFilePath $($resolvedArgumentList -join ' ')`n$($stdout + [Environment]::NewLine + $stderr)"
    }

    return Convert-StructuredJson -Text $stdout
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

    # Use cmd /c with native file redirects — most reliable cross-version
    # approach on Windows that avoids .NET Process event-handler crashes
    # and PowerShell Start-Process handle accumulation issues.
    $quotedArgs = ($ArgumentList | ForEach-Object {
        if ($_ -match '[\s"]') { "`"$($_ -replace '"', '\"')`"" } else { $_ }
    }) -join ' '
    $quotedFilePath = if ($FilePath -match '\s') { "`"$FilePath`"" } else { $FilePath }
    $cmdLine = "$quotedFilePath $quotedArgs > `"$stdoutPath`" 2> `"$stderrPath`""
    Push-Location $WorkingDirectory
    try {
        cmd /c $cmdLine
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    $stdoutText = if (Test-Path -LiteralPath $stdoutPath) { (Get-Content -LiteralPath $stdoutPath -Raw) } else { "" }
    $stderrText = if (Test-Path -LiteralPath $stderrPath) { (Get-Content -LiteralPath $stderrPath -Raw) } else { "" }
    $merged = @()
    if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
        $merged += ($stderrText -split "`r?`n")
    }
    if (-not [string]::IsNullOrWhiteSpace($stdoutText)) {
        $merged += ($stdoutText -split "`r?`n")
    }
    Set-Content -Encoding UTF8 -Path $LogPath -Value $merged
    return [pscustomobject]@{
        exit_code = $exitCode
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

function Invoke-ProjectPytest {
    param(
        [string]$ProjectPath,
        [string]$LogPath,
        [string[]]$Tests,
        [string]$BaseTempRelative = ""
    )

    $arguments = @(
        "run",
        "--project", $ProjectPath,
        "pytest"
    )
    if (-not [string]::IsNullOrWhiteSpace($BaseTempRelative)) {
        $arguments += @("--basetemp", $BaseTempRelative)
    }
    $arguments += $Tests

    return Invoke-LoggedProcess -FilePath "uv" -WorkingDirectory $repoRoot -LogPath $LogPath -ArgumentList $arguments
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
    $spsMatches = [regex]::Matches($logText, "sps=(\d+\.\d+)")
    if ($null -eq $spsMatches -or $spsMatches.Count -eq 0) {
        return $null
    }
    $last = $spsMatches[$spsMatches.Count - 1].Groups[1].Value
    return [double]$last
}

function Test-NonFiniteLogText {
    param([string]$Text)

    return [regex]::IsMatch($Text, "(?i)\b-?nan\b|\b-?inf\b")
}

$repoRoot = Get-RepoRoot
$observabilityModule = Join-Path $repoRoot "tools\Navi.Observability.psm1"
Import-Module $observabilityModule -Force
$cleanupSummary = Invoke-NaviGeneratedCleanup -RepoRoot $repoRoot
$powerShellExe = Get-PowerShellExecutable
$script:StructuredSurfaceRunnerPythonByModule = @{
    default = Join-Path $repoRoot "projects\environment\.venv\Scripts\python.exe"
    "navi_auditor.cli" = Join-Path $repoRoot "projects\auditor\.venv\Scripts\python.exe"
}
$script:StructuredSurfaceRunnerScript = Join-Path $repoRoot "scripts\run-structured-surface.py"
$runContext = New-NaviRunContext -RepoRoot $repoRoot -Profile "nightly-validation" -BaseRelativeRoot $RunRoot
$runRelativeRoot = $runContext.RunRelativeRoot
$runDir = $runContext.RunRoot
$runManifestPath = Join-Path $runContext.ManifestRoot "run-nightly-validation.json"
$preflightDir = Join-Path $runDir "preflight"
$preflightTestDir = Join-Path $preflightDir "tests"
$qualificationRoot = Join-Path $runRelativeRoot "qualification"
$qualificationDir = Join-Path $runDir "qualification"
$validationEvidenceDir = Join-Path $runDir "validation"
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
$progressLogPath = Join-Path $reportsDir "progress.log"

foreach ($dir in @($runDir, $preflightDir, $preflightTestDir, $qualificationDir, $validationEvidenceDir, $boundedDir, $resumeDir, $benchmarkDir, $trainingDir, $trainingCheckpointDir, $trainingAttachDir, $logsDir, $reportsDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$summary = [ordered]@{
    profile = "nightly-validation"
    run_id = $runContext.RunId
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
    current_activity = "initializing"
    artifacts = [ordered]@{
        run_manifest = $runManifestPath
        preflight = $preflightDir
        qualification = $qualificationDir
        validation = $validationEvidenceDir
        bounded = $boundedDir
        resume_checks = $resumeDir
        benchmarks = $benchmarkDir
        training = $trainingDir
        reports = $reportsDir
        progress_log = $progressLogPath
    }
}

$runManifest = [ordered]@{
    run_id = $runContext.RunId
    started_at = $summary.started_at
    repo_root = $repoRoot
    python_version = $PythonVersion
    soak_hours = $SoakHours
    checkpoint_every = $CheckpointEvery
    monitor_interval_seconds = $MonitorIntervalSeconds
    cleanup_removed = @($cleanupSummary.removed)
}
$runManifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summary.artifacts.run_manifest
Save-Summary -Summary $summary -SummaryPath $summaryPath

function Set-CurrentActivity {
    param(
        [System.Collections.IDictionary]$Summary,
        [string]$SummaryPath,
        [string]$ProgressLogPath,
        [string]$Activity
    )

    $Summary.current_activity = $Activity
    Save-Summary -Summary $Summary -SummaryPath $SummaryPath
    Add-Content -Encoding UTF8 -Path $ProgressLogPath -Value ("[{0}] {1}" -f (Get-Date).ToString("o"), $Activity)
}

$trainingProc = $null

try {
    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "cleanup:start"
    if (-not $NoPreKill) {
        Stop-NaviProcesses
        Start-Sleep -Milliseconds 500
    }
    if (-not (Wait-ForPorts -Ports @() -TimeoutSeconds 1)) {
    }
    $portHits = netstat -ano 2>$null | Select-String "^\s*TCP\s+\S+:(5557|5558|5559|5560)\s+\S+\s+LISTENING\s+\d+\s*$"
    if ($portHits) {
        throw "Required Navi ports remain occupied after cleanup"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "cleanup" -Status "passed" -Details ([ordered]@{ ports = @(5557, 5558, 5559, 5560) })

    $corpusRoot = Join-Path $repoRoot "artifacts\gmdag\corpus"
    $preflightScene = Get-ChildItem -Path $corpusRoot -Recurse -File -Filter "apartment_1.gmdag" | Select-Object -First 1
    if ($null -eq $preflightScene) {
        throw "Preflight scene apartment_1.gmdag not found under $corpusRoot"
    }
    $preflightGmdag = $preflightScene.FullName

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "preflight:check-sdfdag:start"
    $checkJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "check-sdfdag",
        "--gmdag-file", $preflightGmdag,
        "--json"
    ) -OutputPath (Join-Path $preflightDir "check-sdfdag.out.json") -ErrorPath (Join-Path $preflightDir "check-sdfdag.err.log") -TimeoutSeconds 240 -ProgressLogPath $progressLogPath
    if (-not $checkJson.ok) {
        throw "check-sdfdag preflight failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "preflight:dataset-audit:start"
    $auditJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "dataset-audit",
        "--gmdag-file", $preflightGmdag,
        "--actors", "4",
        "--steps", "100",
        "--warmup-steps", "20",
        "--json"
    ) -OutputPath (Join-Path $preflightDir "dataset-audit.out.json") -ErrorPath (Join-Path $preflightDir "dataset-audit.err.log") -TimeoutSeconds 240 -ProgressLogPath $progressLogPath
    if (-not $auditJson.ok) {
        throw "dataset-audit preflight failed"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "preflight" -Status "passed" -Details ([ordered]@{ check_sdfdag = $checkJson; dataset_audit = $auditJson })

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:contracts:start"
    $contractsTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\contracts") -LogPath (Join-Path $preflightTestDir "contracts.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\contracts\tests"
    )
    if ($contractsTests.exit_code -ne 0) {
        throw "Contracts nightly regression suite failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:actor:start"
    $actorTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\actor") -LogPath (Join-Path $preflightTestDir "actor.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\actor\tests"
    )
    if ($actorTests.exit_code -ne 0) {
        throw "Actor nightly regression suite failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:environment:start"
    $environmentTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\environment") -LogPath (Join-Path $preflightTestDir "environment.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\environment\tests"
    )
    if ($environmentTests.exit_code -ne 0) {
        throw "Environment nightly regression suite failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:voxel-dag:start"
    $voxelDagTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\voxel-dag") -LogPath (Join-Path $preflightTestDir "voxel-dag.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\voxel-dag\tests"
    )
    if ($voxelDagTests.exit_code -ne 0) {
        throw "Voxel-dag nightly regression suite failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:torch-sdf:start"
    $torchSdfTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\torch-sdf") -LogPath (Join-Path $preflightTestDir "torch-sdf.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\torch-sdf\tests"
    )
    if ($torchSdfTests.exit_code -ne 0) {
        throw "Torch-SDF nightly regression suite failed"
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "focused-tests:auditor:start"
    $auditorTests = Invoke-ProjectPytest -ProjectPath (Join-Path $repoRoot "projects\auditor") -LogPath (Join-Path $preflightTestDir "auditor.log") -BaseTempRelative ".pytest_tmp" -Tests @(
        ".\projects\auditor\tests\unit",
        ".\projects\auditor\tests\integration"
    )
    if ($auditorTests.exit_code -ne 0) {
        throw "Auditor nightly regression suite failed"
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "focused-tests" -Status "passed" -Details ([ordered]@{ contracts_log = $contractsTests.log_path; actor_log = $actorTests.log_path; environment_log = $environmentTests.log_path; voxel_dag_log = $voxelDagTests.log_path; torch_sdf_log = $torchSdfTests.log_path; auditor_log = $auditorTests.log_path })

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "qualification:start"
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
    $qualificationEvidence = [ordered]@{
        qualification_json = $qualificationSummaryPath
        corpus_check_json = $qualificationSummary.artifacts.corpus_check_json
        asset_check_json = $qualificationSummary.artifacts.asset_check_json
        benchmark_json = $qualificationSummary.artifacts.benchmark_json
    }
    Add-PhaseResult -Summary $summary -SummaryPath $summaryPath -Name "qualification" -Status "passed" -Details $qualificationEvidence

    foreach ($sourcePath in @($qualificationSummary.artifacts.corpus_check_json, $qualificationSummary.artifacts.asset_check_json, $qualificationSummary.artifacts.benchmark_json)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$sourcePath) -and (Test-Path $sourcePath)) {
            Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $validationEvidenceDir ([System.IO.Path]::GetFileName($sourcePath))) -Force
        }
    }

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "bounded-shared-model:start"
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

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "bounded-resume:start"
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

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "benchmarks:start"
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

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "soak:start"
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

    Set-CurrentActivity -Summary $summary -SummaryPath $summaryPath -ProgressLogPath $progressLogPath -Activity "finalize:success"
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
    Add-Content -Encoding UTF8 -Path $progressLogPath -Value ("[{0}] catch:{1}" -f (Get-Date).ToString("o"), $_.Exception.Message)
    if ($null -ne $trainingProc -and -not $trainingProc.HasExited) {
        Stop-ProcessTreeById -ProcessId $trainingProc.Id
    }
    $summary.status = "failed"
    $summary.ok = $false
    $summary.completed_at = (Get-Date).ToString("o")
    $summary.current_activity = "failed"
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