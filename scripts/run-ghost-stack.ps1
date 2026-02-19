param(
    [string]$SectionManagerPub = "tcp://*:5559",
    [string]$SectionManagerRep = "tcp://*:5560",
    [string]$ActorSub = "tcp://localhost:5559",
    [string]$ActorPub = "tcp://*:5557",
    [string]$ActorStepEndpoint = "tcp://localhost:5560",
    [string]$ActorPolicy = "shallow",
    [string]$ActorPolicyCheckpoint = "",
    [string]$AuditorSub = "tcp://localhost:5559,tcp://localhost:5557",
    [string]$AuditorOutput = "session.zarr",
    [switch]$NoPreKill
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Resolve-Path (Join-Path $PSScriptRoot "..")
    return $root.Path
}

function Stop-NaviProcesses {
    $patterns = @(
        "*navi-section-manager*",
        "*navi-actor*",
        "*navi-auditor*"
    )
    $targets = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $cmd -and ($patterns | Where-Object { $cmd -like $_ })
    }

    foreach ($proc in $targets) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

function Start-BackgroundUv {
    param(
        [string]$RepoRoot,
        [string[]]$UvArgs,
        [string]$StdOutFile,
        [string]$StdErrFile
    )

    $logDir = Split-Path $StdOutFile -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }

    return Start-Process -FilePath "uv" -ArgumentList $UvArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $StdOutFile -RedirectStandardError $StdErrFile -PassThru
}

$repoRoot = Get-RepoRoot
$logsDir = Join-Path $repoRoot "scripts\logs"

if (-not $NoPreKill) {
    Stop-NaviProcesses
    Start-Sleep -Milliseconds 300
}

$sectionArgs = @(
    "run",
    "--project", (Join-Path $repoRoot "projects\section-manager"),
    "navi-section-manager",
    "serve",
    "--mode", "step",
    "--pub", $SectionManagerPub,
    "--rep", $SectionManagerRep,
    "--generator", "rooms"
)

$actorArgs = @(
    "run",
    "--project", (Join-Path $repoRoot "projects\actor"),
    "navi-actor",
    "run",
    "--sub", $ActorSub,
    "--pub", $ActorPub,
    "--mode", "step",
    "--step-endpoint", $ActorStepEndpoint,
    "--policy", $ActorPolicy
)

if ($ActorPolicy -eq "learned") {
    if ([string]::IsNullOrWhiteSpace($ActorPolicyCheckpoint)) {
        throw "ActorPolicyCheckpoint is required when ActorPolicy=learned"
    }
    $actorArgs += @("--policy-checkpoint", $ActorPolicyCheckpoint)
}

$sectionLogOut = Join-Path $logsDir "section-manager.out.log"
$sectionLogErr = Join-Path $logsDir "section-manager.err.log"
$actorLogOut = Join-Path $logsDir "actor.out.log"
$actorLogErr = Join-Path $logsDir "actor.err.log"

$sectionProc = $null
$actorProc = $null

try {
    Write-Host "Starting Section Manager..."
    $sectionProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $sectionArgs -StdOutFile $sectionLogOut -StdErrFile $sectionLogErr
    Start-Sleep -Milliseconds 1200

    Write-Host "Starting Actor..."
    $actorProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $actorArgs -StdOutFile $actorLogOut -StdErrFile $actorLogErr
    Start-Sleep -Milliseconds 1200

    Write-Host "Launching Auditor dashboard (foreground)..."
    Write-Host "Dashboard mode: Tab to toggle between AI / manual control"
    Write-Host "Controls: WASD or arrow keys to move, ESC or Q to quit"
    Write-Host "Logs:"
    Write-Host "  $sectionLogOut"
    Write-Host "  $sectionLogErr"
    Write-Host "  $actorLogOut"
    Write-Host "  $actorLogErr"

    & uv run --project (Join-Path $repoRoot "projects\auditor") navi-auditor dashboard --matrix-sub "tcp://localhost:5559" --step-endpoint "tcp://localhost:5560"
}
finally {
    foreach ($proc in @($actorProc, $sectionProc)) {
        if ($null -ne $proc) {
            try {
                if (-not $proc.HasExited) {
                    Stop-Process -Id $proc.Id -Force -ErrorAction Stop
                }
            }
            catch {
            }
        }
    }
}
