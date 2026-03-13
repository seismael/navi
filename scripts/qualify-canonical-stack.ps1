<#
.SYNOPSIS
  Run the first end-to-end canonical stack qualification.

.DESCRIPTION
  Executes a passive observer-safe qualification pass over the canonical path:
    1. Auditor dataset preflight (`dataset-audit --json`)
    2. Optional staged corpus refresh into a sandboxed compiled corpus root
    3. Bounded canonical training through `run-ghost-stack.ps1 -Train -NoDashboard`
    3. Headless passive dashboard attach proof against the live actor stream
    4. Passive training-stream recording
    5. Checkpoint resume proof from a produced periodic checkpoint
    6. Replay plus headless passive dashboard attach proof against the replay PUB
    7. JSON artifact emission for later comparison
#>
param(
    [int]$TotalSteps = 512,
    [int]$CheckpointEvery = 256,
    [int]$StartupTimeoutSeconds = 90,
    [int]$AttachTimeoutSeconds = 20,
    [int]$ResumeAdditionalSteps = 0,
    [float]$ReplaySpeed = 1.0,
    [string]$PythonVersion = "3.12",
    [string]$RunRoot = "artifacts\qualification\canonical_stack",
    [switch]$EnableCorpusRefreshQualification,
    [string]$RefreshSourceRoot = "",
    [string]$RefreshManifest = "",
    [string]$RefreshScene = "",
    [int]$RefreshResolution = 512,
    [int]$RefreshMinSceneBytes = 100000,
    [switch]$SkipDatasetAudit,
    [switch]$SkipResumeQualification,
    [switch]$NoPreKill
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
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
        $cmd -and ($patterns | Where-Object { $cmd -like $_ })
    }

    foreach ($proc in $targets) {
        Stop-ProcessTreeById -ProcessId $proc.ProcessId
    }

    Stop-ListenersOnPorts -Ports @(5557, 5558, 5559, 5560)
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
        [int]$TimeoutSeconds = 300
    )

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

    if ([int]$result.exit_code -ne 0) {
        $message = [string]::Format(
            "Command failed with exit code {0}; {1} {2}{3}{4}",
            $result.exit_code,
            $resolvedFilePath,
            ($resolvedArgumentList -join ' '),
            [Environment]::NewLine,
            ($stdout + [Environment]::NewLine + $stderr).Trim()
        )
        throw $message
    }

    return Convert-StructuredJson -Text $stdout
}

function Start-BackgroundProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$StdOutFile,
        [string]$StdErrFile
    )

    $logDir = Split-Path $StdOutFile -Parent
    if (-not [string]::IsNullOrWhiteSpace($logDir) -and -not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $StdOutFile -RedirectStandardError $StdErrFile -PassThru
}

function Wait-ForProcessExit {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds,
        [string]$Label
    )

    if ($null -eq $Process) {
        return
    }

    try {
        $Process.Refresh()
    }
    catch {
    }

    if ($Process.HasExited) {
        return
    }

    if ($TimeoutSeconds -le 0) {
        try {
            Wait-Process -Id $Process.Id -ErrorAction SilentlyContinue
        }
        catch {
            if (-not $_.Exception.Message.Contains("Cannot find a process with the process identifier")) {
                throw
            }
        }
        try {
            $Process.Refresh()
        }
        catch {
        }
        return
    }

    try {
        Wait-Process -Id $Process.Id -Timeout $TimeoutSeconds -ErrorAction SilentlyContinue
    }
    catch {
        if (-not $_.Exception.Message.Contains("Cannot find a process with the process identifier")) {
            throw
        }
    }
    try {
        $Process.Refresh()
    }
    catch {
    }
    if (-not $Process.HasExited) {
        throw "$Label did not exit within ${TimeoutSeconds}s"
    }
}

function Get-ExitCodeOrZero {
    param([System.Diagnostics.Process]$Process)

    $Process.Refresh()
    if ($null -eq $Process.ExitCode) {
        return 0
    }
    return [int]$Process.ExitCode
}

function Read-TextFile {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return ""
    }
    return [System.IO.File]::ReadAllText($Path)
}

function Get-TrainingCompletionStep {
    param([string]$Text)

    $match = [regex]::Match($Text, "Training complete \| .*? steps=(\d+) ")
    if (-not $match.Success) {
        return $null
    }

    $parsedStep = 0
    if (-not [int]::TryParse($match.Groups[1].Value, [ref]$parsedStep)) {
        return $null
    }
    return $parsedStep
}

function Resolve-DefaultRefreshSourceRoot {
    param([string]$RepoRoot)

    return Join-Path $RepoRoot "data\scenes"
}

function Resolve-DefaultRefreshManifest {
    param([string]$RepoRoot)

    return Join-Path (Resolve-DefaultRefreshSourceRoot -RepoRoot $RepoRoot) "scene_manifest_all.json"
}

function Resolve-OptionalPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    return (Resolve-Path $Path).Path
}

function Resolve-RefreshTrainingScene {
    param(
        [string]$CompiledManifestPath,
        [string]$RequestedScene
    )

    if ([string]::IsNullOrWhiteSpace($RequestedScene)) {
        return ""
    }
    if (-not (Test-Path $CompiledManifestPath)) {
        throw "Compiled manifest not found: $CompiledManifestPath"
    }

    $manifest = Get-Content -Path $CompiledManifestPath -Raw | ConvertFrom-Json
    $entries = @($manifest.scenes)
    if ($entries.Count -eq 0) {
        throw "Compiled manifest contains no scenes: $CompiledManifestPath"
    }

    $requestedResolved = ""
    if (Test-Path $RequestedScene) {
        $requestedResolved = (Resolve-Path $RequestedScene).Path
    }
    $normalizedRequest = $RequestedScene.Replace("\", "/").ToLowerInvariant()
    $requestedStem = [System.IO.Path]::GetFileNameWithoutExtension($RequestedScene).ToLowerInvariant()

    foreach ($entry in $entries) {
        $candidatePaths = @()
        if ($null -ne $entry.gmdag_path) {
            $candidatePaths += [string]$entry.gmdag_path
        }
        if ($null -ne $entry.source_path) {
            $candidatePaths += [string]$entry.source_path
        }

        foreach ($candidatePath in $candidatePaths) {
            if ([string]::IsNullOrWhiteSpace($candidatePath)) {
                continue
            }
            if (-not [string]::IsNullOrWhiteSpace($requestedResolved) -and $candidatePath -ieq $requestedResolved) {
                return $candidatePath
            }
            $normalizedCandidate = $candidatePath.Replace("\", "/").ToLowerInvariant()
            if ($normalizedCandidate -eq $normalizedRequest -or $normalizedCandidate.EndsWith("/" + $normalizedRequest)) {
                return $candidatePath
            }
            if ([System.IO.Path]::GetFileNameWithoutExtension($candidatePath).ToLowerInvariant() -eq $requestedStem) {
                return $candidatePath
            }
        }

        $entrySceneName = [string]$entry.scene_name
        if (-not [string]::IsNullOrWhiteSpace($entrySceneName) -and $entrySceneName.ToLowerInvariant() -eq $requestedStem) {
            return [string]$entry.gmdag_path
        }
    }

    if ($entries.Count -eq 1 -and -not [string]::IsNullOrWhiteSpace([string]$entries[0].gmdag_path)) {
        return [string]$entries[0].gmdag_path
    }

    throw "Corpus refresh qualification scene '$RequestedScene' could not be mapped to a refreshed compiled asset"
}

function Resolve-CompiledManifestPath {
    param([string]$CompiledRoot)

    if ([string]::IsNullOrWhiteSpace($CompiledRoot)) {
        return ""
    }

    $manifestPath = Join-Path $CompiledRoot "gmdag_manifest.json"
    if (Test-Path $manifestPath) {
        return $manifestPath
    }
    return ""
}

function Resolve-RepresentativeCompiledAsset {
    param(
        [string]$CompiledRoot,
        [string]$PreferredAsset = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($PreferredAsset) -and (Test-Path $PreferredAsset)) {
        return (Resolve-Path $PreferredAsset).Path
    }

    if ([string]::IsNullOrWhiteSpace($CompiledRoot) -or -not (Test-Path $CompiledRoot)) {
        return ""
    }

    $assets = @(Get-ChildItem -Path $CompiledRoot -Recurse -File -Filter "*.gmdag" | Sort-Object FullName)
    if ($assets.Count -eq 0) {
        return ""
    }
    return $assets[0].FullName
}

$repoRoot = Get-RepoRoot
$powerShellExe = Get-PowerShellExecutable
$script:StructuredSurfaceRunnerPythonByModule = @{
    default = Join-Path $repoRoot "projects\environment\.venv\Scripts\python.exe"
    "navi_auditor.cli" = Join-Path $repoRoot "projects\auditor\.venv\Scripts\python.exe"
}
$script:StructuredSurfaceRunnerScript = Join-Path $repoRoot "scripts\run-structured-surface.py"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $repoRoot (Join-Path $RunRoot $timestamp)
$logsDir = Join-Path $runDir "logs"
$checkpointDir = Join-Path $runDir "checkpoints"
$refreshSandboxRoot = Join-Path $runDir "refresh"
$refreshScratchRoot = Join-Path $refreshSandboxRoot "scratch"
$refreshCompiledRoot = Join-Path $refreshSandboxRoot "gmdag_corpus"
$validationDir = Join-Path $runDir "validation"
$sharedTrainOutLog = Join-Path $repoRoot "scripts\logs\train.out.log"
$sharedTrainErrLog = Join-Path $repoRoot "scripts\logs\train.err.log"
$recordingPath = Join-Path $runDir "training_session.zarr"
$summaryPath = Join-Path $runDir "qualification.json"
$refreshTrainingScene = ""

New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null
New-Item -ItemType Directory -Path $validationDir -Force | Out-Null

$summary = [ordered]@{
    profile = "canonical-stack-qualification"
    ok = $false
    run_dir = $runDir
    total_steps = $TotalSteps
    checkpoint_every = $CheckpointEvery
    resume_additional_steps = $ResumeAdditionalSteps
    replay_speed = $ReplaySpeed
    dataset_audit = $null
    corpus_refresh = $null
    validation = $null
    live_attach = $null
    resume = $null
    replay_attach = $null
    artifacts = [ordered]@{
        qualification_json = $summaryPath
        recording = $recordingPath
        checkpoint_dir = $checkpointDir
        final_checkpoint = $null
        periodic_checkpoints = @()
        refresh_compiled_root = $null
        refresh_manifest = $null
        refresh_transcript = $null
        validation_dir = $validationDir
        corpus_check_json = $null
        asset_check_json = $null
        benchmark_json = $null
        resume_source_checkpoint = $null
        resume_checkpoint_dir = $null
        resume_final_checkpoint = $null
        logs_dir = $logsDir
    }
    issues = @()
}

$trainProc = $null
$resumeTrainProc = $null
$recordProc = $null
$replayProc = $null
$liveAttachProc = $null
$replayAttachProc = $null

try {
    if (-not $NoPreKill) {
        Stop-NaviProcesses
        Start-Sleep -Milliseconds 500
    }

    if (-not $SkipDatasetAudit) {
        $summary.dataset_audit = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
            "run",
            "--python", $PythonVersion,
            "--project", (Join-Path $repoRoot "projects\auditor"),
            "navi-auditor",
            "dataset-audit",
            "--json"
        )
        if (-not $summary.dataset_audit.ok) {
            throw "dataset-audit preflight failed"
        }
    }

    $resolvedRefreshSourceRoot = Resolve-OptionalPath -Path $RefreshSourceRoot
    if ([string]::IsNullOrWhiteSpace($resolvedRefreshSourceRoot)) {
        $resolvedRefreshSourceRoot = Resolve-DefaultRefreshSourceRoot -RepoRoot $repoRoot
    }
    $resolvedRefreshManifest = Resolve-OptionalPath -Path $RefreshManifest
    if ([string]::IsNullOrWhiteSpace($resolvedRefreshManifest)) {
        $resolvedRefreshManifest = Resolve-DefaultRefreshManifest -RepoRoot $repoRoot
    }
    $resolvedRefreshScene = Resolve-OptionalPath -Path $RefreshScene

    if ($EnableCorpusRefreshQualification) {
        if (-not (Test-Path $resolvedRefreshSourceRoot)) {
            throw "Corpus refresh qualification source root not found: $resolvedRefreshSourceRoot"
        }
        if ((-not [string]::IsNullOrWhiteSpace($resolvedRefreshManifest)) -and (-not (Test-Path $resolvedRefreshManifest))) {
            $resolvedRefreshManifest = ""
        }

        $beforeRefreshTranscripts = @()
        $refreshLogsRoot = Join-Path $repoRoot "scripts\logs\corpus-refresh"
        if (Test-Path $refreshLogsRoot) {
            $beforeRefreshTranscripts = @(Get-ChildItem -Path $refreshLogsRoot -File -Filter "refresh_*.log" | Select-Object -ExpandProperty FullName)
        }

        $refreshArgs = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $repoRoot "scripts\refresh-scene-corpus.ps1"),
            "-SkipDownload",
            "-CorpusRoot", $resolvedRefreshSourceRoot,
            "-GmDagRoot", $refreshCompiledRoot,
            "-ScratchRoot", $refreshScratchRoot,
            "-Resolution", $RefreshResolution,
            "-MinSceneBytes", $RefreshMinSceneBytes,
            "-PythonVersion", $PythonVersion
        )
        if (-not [string]::IsNullOrWhiteSpace($resolvedRefreshManifest)) {
            $refreshArgs += @("-Manifest", $resolvedRefreshManifest)
        }
        if (-not [string]::IsNullOrWhiteSpace($resolvedRefreshScene)) {
            $refreshArgs += @("-Scene", $resolvedRefreshScene)
        }

        & $powerShellExe @refreshArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Corpus refresh qualification failed with exit code $LASTEXITCODE"
        }

        $refreshManifestPath = Join-Path $refreshCompiledRoot "gmdag_manifest.json"
        if (-not (Test-Path $refreshManifestPath)) {
            throw "Corpus refresh qualification did not produce $refreshManifestPath"
        }

        $refreshAssets = @(Get-ChildItem -Path $refreshCompiledRoot -Recurse -File -Filter "*.gmdag" | Sort-Object FullName)
        if ($refreshAssets.Count -eq 0) {
            throw "Corpus refresh qualification produced no compiled assets under $refreshCompiledRoot"
        }

        if (-not [string]::IsNullOrWhiteSpace($resolvedRefreshScene)) {
            $refreshTrainingScene = Resolve-RefreshTrainingScene -CompiledManifestPath $refreshManifestPath -RequestedScene $resolvedRefreshScene
        }

        $afterRefreshTranscripts = @()
        if (Test-Path $refreshLogsRoot) {
            $afterRefreshTranscripts = @(Get-ChildItem -Path $refreshLogsRoot -File -Filter "refresh_*.log" | Sort-Object LastWriteTime | Select-Object -ExpandProperty FullName)
        }
        $refreshTranscript = ($afterRefreshTranscripts | Where-Object { $_ -notin $beforeRefreshTranscripts } | Select-Object -Last 1)

        $summary.corpus_refresh = [ordered]@{
            ok = $true
            source_root = $resolvedRefreshSourceRoot
            manifest = $resolvedRefreshManifest
            scene = $resolvedRefreshScene
            training_scene = $refreshTrainingScene
            compiled_root = $refreshCompiledRoot
            compiled_manifest = $refreshManifestPath
            compiled_scene_count = $refreshAssets.Count
            transcript = $refreshTranscript
        }
        $summary.artifacts.refresh_compiled_root = $refreshCompiledRoot
        $summary.artifacts.refresh_manifest = $refreshManifestPath
        $summary.artifacts.refresh_transcript = $refreshTranscript
    }

    $activeCompiledRoot = if ($EnableCorpusRefreshQualification) { $refreshCompiledRoot } else { Join-Path $repoRoot "artifacts\gmdag\corpus" }
    $activeManifestPath = if ($EnableCorpusRefreshQualification) { $summary.artifacts.refresh_manifest } else { Resolve-CompiledManifestPath -CompiledRoot $activeCompiledRoot }
    $preferredValidationAsset = if (-not [string]::IsNullOrWhiteSpace($refreshTrainingScene) -and (Test-Path $refreshTrainingScene)) { $refreshTrainingScene } else { "" }
    $validationAsset = Resolve-RepresentativeCompiledAsset -CompiledRoot $activeCompiledRoot -PreferredAsset $preferredValidationAsset

    if (-not (Test-Path $activeCompiledRoot)) {
        throw "Qualification validation root not found: $activeCompiledRoot"
    }
    if ([string]::IsNullOrWhiteSpace($validationAsset)) {
        throw "Qualification could not resolve a representative compiled asset under $activeCompiledRoot"
    }

    $expectedValidationResolution = if ($EnableCorpusRefreshQualification) { $RefreshResolution } else { 512 }
    $corpusCheckArgs = @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "check-sdfdag",
        "--gmdag-root", $activeCompiledRoot,
        "--expected-resolution", "$expectedValidationResolution"
    )
    if (-not [string]::IsNullOrWhiteSpace($activeManifestPath)) {
        $corpusCheckArgs += @("--manifest", $activeManifestPath)
    }
    $corpusCheckArgs += "--json"
    $corpusCheckJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList $corpusCheckArgs -TimeoutSeconds 300
    if (-not $corpusCheckJson.ok) {
        throw "Qualification active-corpus check-sdfdag validation failed"
    }
    $corpusCheckPath = Join-Path $validationDir "check-sdfdag-corpus.json"
    $corpusCheckJson | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $corpusCheckPath

    $assetCheckJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "check-sdfdag",
        "--gmdag-file", $validationAsset,
        "--json"
    ) -TimeoutSeconds 300
    if (-not $assetCheckJson.ok) {
        throw "Qualification representative-asset check-sdfdag validation failed"
    }
    $assetCheckPath = Join-Path $validationDir "check-sdfdag-asset.json"
    $assetCheckJson | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $assetCheckPath

    $benchJson = Invoke-JsonCommand -FilePath "uv" -WorkingDirectory $repoRoot -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\environment"),
        "navi-environment",
        "bench-sdfdag",
        "--gmdag-file", $validationAsset,
        "--actors", "4",
        "--steps", "100",
        "--warmup-steps", "20",
        "--azimuth-bins", "256",
        "--elevation-bins", "48",
        "--json"
    ) -TimeoutSeconds 300
    $benchPath = Join-Path $validationDir "bench-sdfdag-sample.json"
    $benchJson | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $benchPath

    $summary.validation = [ordered]@{
        ok = $true
        compiled_root = $activeCompiledRoot
        manifest = $activeManifestPath
        representative_asset = $validationAsset
        corpus_check = $corpusCheckJson
        asset_check = $assetCheckJson
        benchmark = $benchJson
    }
    $summary.artifacts.corpus_check_json = $corpusCheckPath
    $summary.artifacts.asset_check_json = $assetCheckPath
    $summary.artifacts.benchmark_json = $benchPath

    $trainOut = Join-Path $logsDir "train_wrapper.out.log"
    $trainErr = Join-Path $logsDir "train_wrapper.err.log"
    $recordOut = Join-Path $logsDir "record.out.log"
    $recordErr = Join-Path $logsDir "record.err.log"
    $liveAttachOut = Join-Path $logsDir "live_attach.out.json"
    $liveAttachErr = Join-Path $logsDir "live_attach.err.log"
    $resumeTrainOut = Join-Path $logsDir "resume_train_wrapper.out.log"
    $resumeTrainErr = Join-Path $logsDir "resume_train_wrapper.err.log"
    $replayOut = Join-Path $logsDir "replay.out.log"
    $replayErr = Join-Path $logsDir "replay.err.log"
    $replayAttachOut = Join-Path $logsDir "replay_attach.out.json"
    $replayAttachErr = Join-Path $logsDir "replay_attach.err.log"

    $trainArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot "scripts\run-ghost-stack.ps1"),
        "-Train",
        "-NoDashboard",
        "-NoPreKill",
        "-PythonVersion", $PythonVersion,
        "-TotalSteps", $TotalSteps,
        "-CheckpointEvery", $CheckpointEvery,
        "-CheckpointDir", $checkpointDir
    )
    if ($EnableCorpusRefreshQualification) {
        $trainArgs += @(
            "-GmDagRoot", $refreshCompiledRoot
        )
        if (-not [string]::IsNullOrWhiteSpace($refreshTrainingScene)) {
            $trainArgs += @("-Scene", $refreshTrainingScene)
        }
    }
    $trainProc = Start-BackgroundProcess -FilePath $powerShellExe -ArgumentList $trainArgs -WorkingDirectory $repoRoot -StdOutFile $trainOut -StdErrFile $trainErr

    if (-not (Wait-ForPorts -Ports @(5557) -TimeoutSeconds $StartupTimeoutSeconds)) {
        throw "Actor telemetry port 5557 did not bind in time"
    }

    $recordProc = Start-BackgroundProcess -FilePath "uv" -WorkingDirectory $repoRoot -StdOutFile $recordOut -StdErrFile $recordErr -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "record",
        "--sub", "tcp://localhost:5557",
        "--out", $recordingPath
    )

    Start-Sleep -Seconds 2

    $liveAttachProc = Start-BackgroundProcess -FilePath "uv" -WorkingDirectory $repoRoot -StdOutFile $liveAttachOut -StdErrFile $liveAttachErr -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "dashboard-attach-check",
        "--actor-sub", "tcp://localhost:5557",
        "--timeout-seconds", $AttachTimeoutSeconds,
        "--json"
    )
    Wait-ForProcessExit -Process $liveAttachProc -TimeoutSeconds ($AttachTimeoutSeconds + 5) -Label "live dashboard attach check"
    $summary.live_attach = Convert-StructuredJson -Text ([System.IO.File]::ReadAllText($liveAttachOut))
    if (-not $summary.live_attach.ok) {
        throw "Live passive attach proof failed"
    }

    Wait-ForProcessExit -Process $trainProc -TimeoutSeconds 0 -Label "canonical training wrapper"
    $trainExitCode = Get-ExitCodeOrZero -Process $trainProc
    if ($trainExitCode -ne 0) {
        throw "Canonical training wrapper exited with code $trainExitCode"
    }

    Start-Sleep -Seconds 1
    if ($null -ne $recordProc -and -not $recordProc.HasExited) {
        Stop-ProcessTreeById -ProcessId $recordProc.Id
        Start-Sleep -Seconds 1
        $recordProc.Refresh()
    }

    if (-not (Test-Path $recordingPath)) {
        throw "Recorder did not produce $recordingPath"
    }

    $finalCheckpoint = Join-Path $checkpointDir "policy_final.pt"
    if (-not (Test-Path $finalCheckpoint)) {
        throw "Final checkpoint missing at $finalCheckpoint"
    }
    $summary.artifacts.final_checkpoint = $finalCheckpoint
    $periodicCheckpoints = @(Get-ChildItem -Path $checkpointDir -Filter "policy_step_*.pt" -File | Sort-Object Name | Select-Object -ExpandProperty FullName)
    $summary.artifacts.periodic_checkpoints = $periodicCheckpoints
    if ($CheckpointEvery -gt 0 -and $TotalSteps -ge $CheckpointEvery -and $periodicCheckpoints.Count -eq 0) {
        throw "Expected at least one periodic checkpoint in $checkpointDir"
    }

    if (-not $SkipResumeQualification) {
        if ($CheckpointEvery -le 0) {
            throw "Checkpoint resume qualification requires CheckpointEvery > 0"
        }
        if ($periodicCheckpoints.Count -eq 0) {
            throw "Checkpoint resume qualification requires at least one periodic checkpoint"
        }

        $resumeSourceCheckpoint = $periodicCheckpoints[-1]
        $resumeStepDelta = if ($ResumeAdditionalSteps -gt 0) { $ResumeAdditionalSteps } else { $CheckpointEvery }
        $resumeTotalSteps = $TotalSteps + $resumeStepDelta
        $resumeCheckpointDir = Join-Path $runDir "resume_checkpoints"
        New-Item -ItemType Directory -Path $resumeCheckpointDir -Force | Out-Null

        $summary.artifacts.resume_source_checkpoint = $resumeSourceCheckpoint
        $summary.artifacts.resume_checkpoint_dir = $resumeCheckpointDir

        $resumeTrainArgs = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $repoRoot "scripts\run-ghost-stack.ps1"),
            "-Train",
            "-NoDashboard",
            "-NoPreKill",
            "-PythonVersion", $PythonVersion,
            "-TotalSteps", $resumeTotalSteps,
            "-CheckpointEvery", $CheckpointEvery,
            "-CheckpointDir", $resumeCheckpointDir,
            "-Checkpoint", $resumeSourceCheckpoint
        )
        if ($EnableCorpusRefreshQualification) {
            $resumeTrainArgs += @(
                "-GmDagRoot", $refreshCompiledRoot
            )
            if (-not [string]::IsNullOrWhiteSpace($refreshTrainingScene)) {
                $resumeTrainArgs += @("-Scene", $refreshTrainingScene)
            }
        }
        $resumeTrainProc = Start-BackgroundProcess -FilePath $powerShellExe -ArgumentList $resumeTrainArgs -WorkingDirectory $repoRoot -StdOutFile $resumeTrainOut -StdErrFile $resumeTrainErr

        Wait-ForProcessExit -Process $resumeTrainProc -TimeoutSeconds 0 -Label "resume training wrapper"
        $resumeExitCode = Get-ExitCodeOrZero -Process $resumeTrainProc
        if ($resumeExitCode -ne 0) {
            throw "Resume training wrapper exited with code $resumeExitCode"
        }

        $resumeStdOutText = (
            (Read-TextFile -Path $resumeTrainOut) +
            [Environment]::NewLine +
            (Read-TextFile -Path $sharedTrainOutLog) +
            [Environment]::NewLine +
            (Read-TextFile -Path $sharedTrainErrLog)
        )
        $expectedResumeLine = "Loaded checkpoint: $resumeSourceCheckpoint"
        if (-not $resumeStdOutText.Contains($expectedResumeLine)) {
            throw "Resume qualification did not confirm checkpoint load from $resumeSourceCheckpoint"
        }

        $resumeCompletionStep = Get-TrainingCompletionStep -Text $resumeStdOutText
        if ($null -eq $resumeCompletionStep) {
            throw "Resume qualification could not parse the resumed training completion step"
        }
        if ($resumeCompletionStep -lt $resumeTotalSteps) {
            throw "Resume qualification completed at step $resumeCompletionStep before reaching target $resumeTotalSteps"
        }

        $resumeFinalCheckpoint = Join-Path $resumeCheckpointDir "policy_final.pt"
        if (-not (Test-Path $resumeFinalCheckpoint)) {
            throw "Resume final checkpoint missing at $resumeFinalCheckpoint"
        }
        $summary.artifacts.resume_final_checkpoint = $resumeFinalCheckpoint

        $summary.resume = [ordered]@{
            ok = $true
            source_checkpoint = $resumeSourceCheckpoint
            target_total_steps = $resumeTotalSteps
            completion_step = $resumeCompletionStep
            final_checkpoint = $resumeFinalCheckpoint
        }
    }

    $replayAttachProc = Start-BackgroundProcess -FilePath "uv" -WorkingDirectory $repoRoot -StdOutFile $replayAttachOut -StdErrFile $replayAttachErr -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "dashboard-attach-check",
        "--actor-sub", "tcp://localhost:5558",
        "--timeout-seconds", $AttachTimeoutSeconds,
        "--json"
    )

    Start-Sleep -Seconds 1
    $replayProc = Start-BackgroundProcess -FilePath "uv" -WorkingDirectory $repoRoot -StdOutFile $replayOut -StdErrFile $replayErr -ArgumentList @(
        "run",
        "--python", $PythonVersion,
        "--project", (Join-Path $repoRoot "projects\auditor"),
        "navi-auditor",
        "replay",
        "--input", $recordingPath,
        "--pub", "tcp://*:5558",
        "--speed", $ReplaySpeed
    )

    Wait-ForProcessExit -Process $replayAttachProc -TimeoutSeconds ($AttachTimeoutSeconds + 5) -Label "replay dashboard attach check"
    $summary.replay_attach = Convert-StructuredJson -Text ([System.IO.File]::ReadAllText($replayAttachOut))
    if (-not $summary.replay_attach.ok) {
        throw "Replay passive attach proof failed"
    }

    Wait-ForProcessExit -Process $replayProc -TimeoutSeconds 0 -Label "replay command"
    $replayExitCode = Get-ExitCodeOrZero -Process $replayProc
    if ($replayExitCode -ne 0) {
        throw "Replay command exited with code $replayExitCode"
    }

    $summary.ok = $true
}
catch {
    $summary.issues += $_.Exception.Message
}
finally {
    foreach ($proc in @($replayProc, $replayAttachProc, $resumeTrainProc, $recordProc, $liveAttachProc, $trainProc)) {
        if ($null -ne $proc) {
            try {
                $proc.Refresh()
            }
            catch {
            }
            if ($null -ne $proc -and -not $proc.HasExited) {
                Stop-ProcessTreeById -ProcessId $proc.Id
            }
        }
    }
    Stop-ListenersOnPorts -Ports @(5557, 5558, 5559, 5560)
    $summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding utf8
}

Get-Content $summaryPath
if (-not $summary.ok) {
    exit 1
}
